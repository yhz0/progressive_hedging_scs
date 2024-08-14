include("utils.jl")
include("smps_cor.jl")
include("smps_tim.jl")
include("smps_sto.jl")
include("ph_helper.jl")

using JuMP, CPLEX, MathOptInterface
const MOI = MathOptInterface
const N_SAMPLE = 1000
const N_WORKERS = Threads.nthreads()

function solve(instance_name::String, max_iter::Int, rho::Float64)

    # Load smps cor, tim, sto files
    cor = read_cor("spInput/$(instance_name)/$(instance_name).cor")
    tim = read_implicit_tim("spInput/$(instance_name)/$(instance_name).tim")
    sto = read_sto("spInput/$(instance_name)/$(instance_name).sto")

    # Load template model
    model = JuMP.read_from_file("spInput/$(instance_name)/$(instance_name).cor"; format = MathOptInterface.FileFormats.FORMAT_MPS)

    # Initialize:
    # Generate N_SAMPLE random vectors
    rng = MersenneTwister()
    random_vectors = Vector{Float64}[]  # Create an array of arrays
    for _ in 1:N_SAMPLE
        push!(random_vectors, generate_random_vector(rng, sto))
    end

    # Initialize primal and dual variables and rho
    first_stage_var_names = name.(get_first_stage_vars(model, tim))
    # x_bar = zeros(length(first_stage_var_names))
    x_bar = get_crash_solution(model, generate_random_vector(rng, sto), tim, sto, CPLEX.Optimizer)
    println("Crash solution = $x_bar")
    
    scenario_lambda = zeros(length(first_stage_var_names), N_SAMPLE)

    # Current solution for each worker
    scenario_x = zeros(length(first_stage_var_names), N_SAMPLE)

    # Generate N_WORKERS copies of model, put into an array
    model_copies = Vector{Model}(undef, N_WORKERS)

    # For storing the first stage variables
    first_stage_variables = Vector{Vector{VariableRef}}(undef, N_WORKERS)

    # For storing original objective functions for each worker
    original_objective = Vector{AffExpr}(undef, N_WORKERS)

    # For storing objective values
    scenario_objective_value = Vector{Float64}(undef, N_SAMPLE)

    # For storing results
    dual_objective_value_history = Vector{Float64}()

    # Conjugate direction
    conjugate_direction = zeros(length(first_stage_var_names), N_SAMPLE)

    println("Initialized...")

    # Initialize these objects
    for i in 1:N_WORKERS
        # Copy the model
        model_copies[i] = direct_model(CPLEX.Optimizer())
        MOI.copy_to(backend(model_copies[i]), model)

        # Save the objective
        original_objective[i] = objective_function(model_copies[i])

        # Save variable references
        first_stage_variables[i] = get_first_stage_vars(model_copies[i], tim)

        # Convert objective to QuadExpr
        obj_expr = convert(QuadExpr, objective_function(model_copies[i]))
        set_objective(model_copies[i], MIN_SENSE, obj_expr)

        # Set the model to single threaded mode
        set_optimizer_attribute(model_copies[i], "CPX_PARAM_THREADS", 1)
        set_silent(model_copies[i])

    end

    # Outer Loop: iteration
    for iter = 1:max_iter
        # Step 1: Solve all problems

        # Task pool
        tasks = []

        # Spawn each sample as a task
        for i = 1:N_SAMPLE
            task = Threads.@spawn begin
                # Get tid. The following code will use model_copies[tid]
                tid = Threads.threadid()

                # Replace the RHS
                x = first_stage_variables[tid]
                lambda = scenario_lambda[:, i]
                # Change coefficients
                change_coefficients!(model_copies[tid], random_vectors[i], sto)

                # Change objective and add regularization terms
                old_objective = original_objective[tid]
                new_objective = add_regularization_terms(old_objective + lambda' * (x - x_bar), x - x_bar, rho/2.0)
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
        for i = 1:N_SAMPLE
            fetch(tasks[i])
        end

        dual_obj = sum(scenario_objective_value) / N_SAMPLE
        push!(dual_objective_value_history, dual_obj)

        # Step 2 Averaging
        x_bar .= sum(scenario_x, dims=2) ./ N_SAMPLE

        # println("iter = $iter, Dobj = $dual_obj, xbar=$x_bar")
        println("iter = $iter, Dobj = $dual_obj")

        # Step 3 Dual update
        g = scenario_x .- x_bar
        if iter == 1
            conjugate_direction .= g
        else
            # conjugate_direction .= 1/2* g + 1/2*conjugate_direction
            dd = sum(conjugate_direction .* conjugate_direction; dims=1)
            dg = sum(conjugate_direction .* g; dims=1)
            gg = sum(g .* g; dims=1)

            # ratio on conjugate direction
            gamma = clamp.((gg - dg) ./ (dd - 2 * dg + gg), 0, 1)
            
            conjugate_direction .= gamma .* conjugate_direction + (1 .- gamma) .* g
        end
        scenario_lambda += rho/iter * conjugate_direction
    end

    return dual_objective_value_history
end

solve("20", 100, 0.5)