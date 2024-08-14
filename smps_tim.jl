"""
Structure for pointing out a position in the LP template,
containing a col_name and a row_name.
"""
struct spSmpsPosition
    col_name::String
    row_name::String
end

"""
Structure for storing one single period.
"""
struct spSmpsImplicitPeriod
    period_name::String
    position::spSmpsPosition
end

"""
Structure for storing information to splice the template into periods.
Only implicit type is supported.
"""
struct spImplicitTimType
    problem_name::String
    periods::Vector{spSmpsImplicitPeriod}
end

"""
Read implicit time file into TimType structure.
"""
function read_implicit_tim(tim_path::String)::spImplicitTimType
    local lines
    open(tim_path, "r") do io
        lines = readlines(io)
    end
    
    local section::String = ""
    local problem_name::String = ""
    local periods::Vector{spSmpsImplicitPeriod} = []
    supported_sections::Vector{String} = ["TIME", "PERIODS", "ENDATA"]

    for line in lines
        token = split(line)
        if line[1] == ' '
            @assert section == "PERIODS"
            @assert length(token) == 3
            col_name = token[1]
            row_name = token[2]
            period_name = token[3]
            period = spSmpsImplicitPeriod(period_name, spSmpsPosition(col_name, row_name))
            push!(periods, period)
        else
            section = token[1]
            @assert section in supported_sections
            if section == "TIME"
                problem_name = token[2]
            end
        end
    end

    return spImplicitTimType(problem_name, periods)
end
