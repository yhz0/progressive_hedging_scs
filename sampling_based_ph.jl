const instance_name = "baa99-20"
include("ph_routines.jl")

const MAX_ITER = 500

n_samples(iter::Int) = 1000 + 10 * iter

# true if the sample is solved for the first timis_first_solve = fill(true, MAX_SAMPLE)
is_first_solve = fill(true, MAX_SAMPLE)

function sample_based_ph()

    # Initialze dual obj to -Inf
    dual_obj::Float64 = -Inf

    for iter = 1:MAX_ITER
        current_sample_number = n_samples(iter)

        # Expand the random vector pool
        expand_random_vector_pool(rng, sto, current_sample_number)

        sample_set = 1:current_sample_number

        # Solve the subproblems
        solve_subproblems(sample_set)

        # Update the dual objective
        dual_obj= get_average_objective_value(sample_set)

        # Log the dual objective
        println("Iteration $iter: dual_obj = $dual_obj")

        # Update x_bar
        x_bar .= calculate_x_bar(sample_set)

        for i in sample_set
            # Decide direction
            g = scenario_x[:, i] - x_bar
            if is_first_solve[i]
                conjugate_direction[:, i] .= g
                is_first_solve[i] = false
            else
                dd = conjugate_direction[:, i]' * conjugate_direction[:, i]
                dg = conjugate_direction[:, i]' * g
                gg = g' * g

                # ratio on conjugate direction
                gamma = clamp((gg - dg) / (dd - 2 * dg + gg), 0, 1)
                
                conjugate_direction[:, i] .= gamma * conjugate_direction[:, i] + (1 - gamma) * g
            end
            
            step_size = rho

            # Update lambda with conjugate direction
            scenario_lambda[:, i] += step_size * conjugate_direction[:, i]
        end
    end
end

sample_based_ph()


