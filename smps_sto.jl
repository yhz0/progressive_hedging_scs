using OrderedCollections, Random

"""
Describe a type of random variable in the SMPS sto format, INDEP section.
"""
abstract type spSmpsIndepDistribution end

"""
Scalar discrete distribution.
"""
struct spSmpsDiscreteDistribution <: spSmpsIndepDistribution
    value::Vector{Float64}
    probability::Vector{Float64}
end

"""
Scalar normal distribution with mean and variance.
"""
struct spSmpsNormalDistribution <: spSmpsIndepDistribution
    mean::Float64
    stddev::Float64
end

"""
Scalar continuous uniform distribution on [a, b].
"""
struct spSmpsUniformDistribution <: spSmpsIndepDistribution
    left::Float64
    right::Float64
end

"""
Sto file representation describes distribution of coefficients.
"""
struct spStoType
    problem_name::String
    indep::OrderedDict{spSmpsPosition, spSmpsIndepDistribution}
end

"""
Read stochastic types. There is no error checking.
"""
function read_sto(sto_path::String)::spStoType
    local lines
    open(sto_path, "r") do io
        lines = readlines(io)
    end

    local section::String
    local section_keywords::Vector{String}
    local problem_name::String
    local indep = OrderedDict{spSmpsPosition, spSmpsIndepDistribution}()
    supported_sections::Vector{String} = ["STOCH", "INDEP", "ENDATA"]
    
    filter!(s -> (!isempty(s) && s[1] != '*'), lines)

    for line in lines
        token = split(line)

        if line[1] == ' '
            # data lines

            if section == "INDEP"
                col_name = token[1]
                row_name = token[2]
                pos = spSmpsPosition(col_name, row_name)

                # Only support univariate indep distribution and REPLACE mode for now
                if length(section_keywords) > 1
                    error("Trailing/unsupported section_keywords $section_keywords")
                end

                # use section_keywords to determine the distribution type
                if section_keywords[1] == "UNIFORM"
                    a = parse(Float64, token[3])
                    b = parse(Float64, token[4])
                    indep[pos] = spSmpsUniformDistribution(a, b)
                elseif section_keywords[1] == "NORMAL"
                    m = parse(Float64, token[3])
                    v = parse(Float64, token[4])
                    indep[pos] = spSmpsNormalDistribution(m, v)
                elseif section_keywords[1] == "DISCRETE"
                    # For discrete distribution, we first check if we have position
                    # recorded. If not, we populate it with empty distribution first.
                    if !(pos in keys(indep))
                        indep[pos] = spSmpsDiscreteDistribution(Float64[], Float64[])
                    end
                    r = Ref{spSmpsDiscreteDistribution}(indep[pos])
                    
                    v = parse(Float64, token[3])
                    p = parse(Float64, token[4])
                    push!(r[].value, v)
                    push!(r[].probability, p)

                else
                    error("Unknown or unsupported section_keywords $section_keywords")
                end
            end

        else
            #section header lines
            section = token[1]
            @assert(section in supported_sections)
            section_keywords = token[2:end]

            if section == "STOCH"
                problem_name = section_keywords[1]
            end
        end
    end

    return spStoType(problem_name, indep)
end


function generate_random_element(rng, dist::spSmpsDiscreteDistribution)::Float64
    """
    Generates a random element from a spSmpsDiscreteDistribution using the given random number generator,
    relying solely on the Random package for randomness.
    """
    # Compute cumulative probabilities
    cumulative_probs = cumsum(dist.probability)
    
    # Generate a random number between 0 and 1
    random_value = rand(rng)
    
    # Find the first index where the cumulative probability exceeds the random value
    for (index, cp) in enumerate(cumulative_probs)
        if random_value <= cp
            return dist.value[index]
        end
    end
    
    # In case of rounding issues, return the last element
    return dist.value[end]
end

function generate_random_element(rng, dist::spSmpsNormalDistribution)::Float64
    """
    Generates a random element from a spSmpsNormalDistribution using the given random number generator.
    This uses randn to generate a standard normally distributed number, then scales and shifts it to match the desired mean and standard deviation.
    """
    # randn(rng) generates a standard normal variable (mean 0, stddev 1)
    # We scale it by the standard deviation and shift it by the mean
    return dist.mean + dist.stddev * randn(rng)
end

function generate_random_element(rng, dist::spSmpsUniformDistribution)::Float64
    """
    Generates a random element from a spSmpsUniformDistribution using the given random number generator.
    This uses rand to generate a uniformly distributed number within the specified range [left, right].
    """
    # rand(rng, range) generates a uniformly distributed number in the specified range
    return rand(rng, dist.left, dist.right)
end

function generate_random_vector(rng, sto::spStoType)::Vector{Float64}
    """
    Generates a Float64 vector distributed as specified in the sto.
    """
    # Initialize omega as a zero vector with the same length as the number of keys in sto.indep
    omega = zeros(Float64, length(keys(sto.indep)))

    # Loop through the values of sto.indep (they are in order due to OrderedDict)
    for (i, dist) in enumerate(values(sto.indep))
        # Call generate_random_element to get a random value for each distribution
        omega[i] = generate_random_element(rng, dist)
    end

    return omega
end