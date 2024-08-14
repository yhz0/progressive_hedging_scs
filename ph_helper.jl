using JuMP
include("smps_sto.jl")
function get_first_stage_vars(model::Model, tim::spImplicitTimType)::Vector{VariableRef}
    """
    Accepts a JuMP model and an spImplicitTimType structure (two-stage only).
    Returns a vector of variable references for the first stage.
    """
    # Initialize an empty vector to store first stage variables
    first_stage_vars = VariableRef[]

    # Get the breaking point variable name from the second stage
    breakpoint_name = tim.periods[2].position.col_name

    # Iterate over all variables in the model
    for var in all_variables(model)
        # If the variable name matches the breaking point, exit the loop
        if name(var) == breakpoint_name
            break
        end
        # Add the variable to the list of first stage variables
        push!(first_stage_vars, var)
    end

    return first_stage_vars
end

function build_zero_solution(vars::Vector{VariableRef})::Dict{String, Float64}
    """
    Given a vector of vars, return a dictionary mapping each variable's name to 0.0.
    """
    # Initialize an empty dictionary
    zero_solution = Dict{String, Float64}()

    # Iterate over the variables and set each to 0.0
    for var in vars
        zero_solution[name(var)] = 0.0
    end

    return zero_solution
end

function add_regularization_terms(expr::AffExpr, vars::Vector{AffExpr}, rho::Float64)::QuadExpr
    """
    Given an affine expression, this function returns a quadratic expression by adding squared terms of the variables,
    scaled by rho, to the original affine expression.
    """

    quad_expr = convert(QuadExpr, expr)
    # Add squared terms scaled by rho for each variable
    for var in vars
        quad_expr += rho * var^2
    end
    
    return quad_expr
end

function change_coefficients!(model::Model, omega::Vector{Float64}, sto::spStoType)
    """
    Change the coefficients of the model represented by sto and omega.
    This function modifies the model in place based on stochastic values in omega.
    """

    # Assert that sto.indep has the same length as omega
    @assert length(sto.indep) == length(omega) "The length of sto.indep must match the length of omega."
    
    # Loop through i and keys(sto.indep), which are of type spSmpsPosition
    for (i, pos) in enumerate(keys(sto.indep))
        if pos.col_name == "RHS" || pos.col_name == "rhs"
            # Set the normalized right-hand side of row_name to omega[i]
            set_normalized_rhs(constraint_by_name(model, pos.row_name), omega[i])
        else
            # Set normalized coefficient of variable in column pos.col_name in row pos.row_name to omega[i]
            set_normalized_coefficient(constraint_by_name(model, pos.row_name), variable_by_name(model, pos.col_name), omega[i])
        end
    end
end

function get_crash_solution(model::Model, omega::Vector{Float64}, tim::spImplicitTimType, sto::spStoType, optimizer)
    # Make a copy of the model
    model = copy(model)
    set_optimizer(model, optimizer)

    # Change the coefficients to omega
    change_coefficients!(model, omega, sto)

    set_silent(model)
    optimize!(model)

    if termination_status(model) != OPTIMAL
        error("Unable to get crash solution, termination_status = $(termination_status(model))")
    end

    return value.(get_first_stage_vars(model, tim))
end