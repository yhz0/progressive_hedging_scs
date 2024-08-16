include("utils.jl")
include("smps_cor.jl")
include("smps_tim.jl")
include("smps_sto.jl")
include("ph_helper.jl")

using JuMP, Gurobi, MathOptInterface
const optimizer = Gurobi.Optimizer
const MOI = MathOptInterface
const MAX_SAMPLE = 1000000
const N_WORKERS = Threads.nthreads()

# Load smps cor, tim, sto files
const cor = read_cor("spInput/$(instance_name)/$(instance_name).cor")
const tim = read_implicit_tim("spInput/$(instance_name)/$(instance_name).tim")
const sto = read_sto("spInput/$(instance_name)/$(instance_name).sto")

# Define resources
rng = MersenneTwister(42)  # rng

const model = JuMP.read_from_file("spInput/$(instance_name)/$(instance_name).cor"; format = MathOptInterface.FileFormats.FORMAT_MPS) # main model
const first_stage_var_names = name.(get_first_stage_vars(model, tim)) # var names
const first_stage_var_length = length(first_stage_var_names) # var length
const scenario_lambda = zeros(first_stage_var_length, MAX_SAMPLE) # dual values
const scenario_x = zeros(first_stage_var_length, MAX_SAMPLE) # primal values
const model_copies = Vector{Model}(undef, N_WORKERS) # model resources
const first_stage_variables = Vector{Vector{VariableRef}}(undef, N_WORKERS) # first stage variables for each worker
const original_objective = Vector{AffExpr}(undef, N_WORKERS) # original objective functions for each worker
const scenario_objective_value = Vector{Float64}(undef, MAX_SAMPLE) # objective values
const conjugate_direction = zeros(first_stage_var_length, MAX_SAMPLE) # conjugate direction

const x_bar = zeros(first_stage_var_length) # consensus solution

const random_vectors = Vector{Float64}[] # random vectors

function expand_random_vector_pool(rng::MersenneTwister, sto::spStoType, n::Int)
    """
    Expand the random vector pool to n vectors.
    """
    if length(random_vectors) >= n
        return
    end
    while length(random_vectors) < n
        push!(random_vectors, generate_random_vector(rng, sto))
    end
end


# Initialize model copies
for i in 1:N_WORKERS
    model_copies[i] = direct_model(optimizer())
    MOI.copy_to(backend(model_copies[i]), model)

    # Get first stage variables
    first_stage_variables[i] = get_first_stage_vars(model_copies[i], tim)

    # Save the objective
    original_objective[i] = objective_function(model_copies[i])

    # Save variable references
    first_stage_variables[i] = get_first_stage_vars(model_copies[i], tim)

    # Convert objective to QuadExpr
    obj_expr = convert(QuadExpr, objective_function(model_copies[i]))
    set_objective(model_copies[i], MIN_SENSE, obj_expr)

    # Set the model to single threaded mode
    set_optimizer_attribute(model_copies[i], "THREADS", 1)
    set_silent(model_copies[i])

end

x_bar .= get_crash_solution(model, generate_random_vector(rng, sto), tim, sto, optimizer)
println("Crash solution = $x_bar")
const rho::Float64 = 1.0


function solve_subproblems(
    sample_index_set::Union{Vector{Int64}, UnitRange{Int64}},
    step_size::Union{Nothing, Float64} = nothing
    )
    """
    Solve all subproblems in sample_index_set, given 
    scenario_lambda, x_bar
    Store scenario solutions in scenario_x,
    scenario objective values in scenario_objective_value.
    If step_size is not nothing, then use the lambda + step_size * conjugate_direction
    as the dual values.
    """
    # Task pool
    tasks = []

    # Spawn each sample as a task
    for i in sample_index_set
        task = Threads.@spawn begin
            # Get tid. The following code will use model_copies[tid]
            tid = Threads.threadid()

            # Replace the RHS
            x = first_stage_variables[tid]
            new_lambda = Vector{Float64}(undef, length(x))
            if isnothing(step_size)
                new_lambda .= scenario_lambda[:, i]
            else
                new_lambda .= scenario_lambda[:, i] + step_size * conjugate_direction[:, i]
            end

            # Change coefficients assuming random_vectors[i] exists
            change_coefficients!(model_copies[tid], random_vectors[i], sto)

            # Change objective and add regularization terms
            old_objective = original_objective[tid]
            new_objective = add_regularization_terms(old_objective + new_lambda' * (x - x_bar), x - x_bar, rho/2.0)
            set_objective(model_copies[tid], MIN_SENSE, new_objective)

            # Set warm start for each variable
            for j in 1:length(x)
                set_start_value(x[j], scenario_x[j, i])
            end

            # Optimize model and set solution
            optimize!(model_copies[tid])

            # Check solution status
            if termination_status(model_copies[tid]) == MOI.OPTIMAL
                # Update the solution matrix
                optimal_values = value.(first_stage_variables[tid])
                for j in 1:length(optimal_values)
                    scenario_x[j, i] = optimal_values[j]
                end
                # println("[$tid] Solved $i")
            else
                println("[$tid] Model $i failed to solve optimally. $(termination_status(model_copies[tid]))")
                # write_to_file(model_copies[tid], "qp_subproblem_error_$i.lp")
            end

            scenario_objective_value[i] = value(new_objective)

        end
        push!(tasks, task)
    end

    # Wait for tasks to finish
    for i in eachindex(tasks)
        fetch(tasks[i])
    end

end

function get_average_objective_value(sample_index_set::Union{Vector{Int64}, UnitRange{Int64}})::Float64
    """
    Get the average dual objective value of the samples in sample_index_set.
    """

    s::Float64 = 0.0
    for i in sample_index_set
        s += scenario_objective_value[i]
    end
    return s / length(sample_index_set)
end

function calculate_x_bar(sample_index_set::Union{Vector{Int64}, UnitRange{Int64}})
    """
    Calculate the consensus solution x_bar using the samples in sample_index_set.
    """
    _x_bar = zeros(first_stage_var_length)
    for i in sample_index_set
        _x_bar .+= scenario_x[:, i]
    end
    _x_bar ./= length(sample_index_set)
    return _x_bar
end