using SparseArrays

"""
A problem template defined by SMPS format.
"""
struct spCorType
    problem_name::String
    directions::Vector{Char}
    row_names::Vector{String}
    col_names::Vector{String}
    template_matrix::SparseMatrixCSC{Float64, Int}
    rhs::SparseVector{Float64, Int}
    lower_bound::Vector{Float64}
    upper_bound::Vector{Float64}
    col_mapping::Dict{String, Int}
    row_mapping::Dict{String, Int}
end

import Base.show
"Display for cor format."
Base.show(io::IO, cor::spCorType) = print(io, "spCorType $(cor.problem_name)")

"""
Parse cor file into tokens for intermediate representation.
"""
function _tokenize_cor(io::IO)
    supported_sections::Vector{String} = ["NAME", "ROWS", "COLUMNS",
        "RHS", "BOUNDS", "ENDATA"]
    tokens = Dict(section_name => [] for section_name in supported_sections)

    # Read cor and remove empty or comment rows
    lines = readlines(io)
    filter!(s -> (!isempty(s) && s[1] != '*'), lines)

    section = ""

    for line in lines
        token = split(line)

        # Look at section line first
        if line[1] != ' '
            # Split each row into tokens first
            section = token[1]
            @assert(section in supported_sections)

            # special case for NAME section because it is on
            # the same line
            if token[1] == "NAME"
                push!(tokens["NAME"], token[2])
            end
        else
            # Data lines
            push!(tokens[section], token)
        end
    end

    return tokens
end

"""
Parse row tokens into constraint direction and list of row names.
"""
function _parse_row_tokens(tokens)
    direction::Vector{Char} = [t[1][1] for t in tokens]
    row_names::Vector{String} = [t[2] for t in tokens]
    return direction, row_names
end

"""
Extract variable names from column tokens.
"""
function _parse_unique_columns(tokens)
    col_names::Vector{String} = [t[1] for t in tokens]
    return unique(col_names)
end

"""
Extract the coefficient matrix. The first row is assumed to be
the objective row.
"""
function _parse_column_to_matrix(tokens, row_names, col_names)
    # Build mapping from the names to the assigned indices
    col_mapping = Dict(col => i for (i, col) in enumerate(col_names))
    row_mapping = Dict(row => i for (i, row) in enumerate(row_names))

    M = spzeros(length(row_names), length(col_names))

    for token in tokens
        col_name = token[1]
        col_index = col_mapping[col_name]
        
        # Iterate the rest of the line in pairs, and populate the entry
        for (row_name, vstring) in Iterators.partition(token[2:end], 2)
            row_index = row_mapping[row_name]
            v = parse(Float64, vstring)
            M[row_index, col_index] = v
        end
    end

    return M
end

"""
Extract the rhs from RHS tokens, assuming all missing entries are zeros.
"""
function _parse_rhs(tokens, row_names)
    row_mapping = Dict(row => i for (i, row) in enumerate(row_names))
    rhs = zeros(length(row_names))
    for token in tokens
        for (row_name, vstring) in Iterators.partition(token[2:end], 2)
            row_index = row_mapping[row_name]
            rhs[row_index] = parse(Float64, vstring)
        end
    end
    return rhs
end


"""
Parse the bounds section of cor tokens. Return the lower_bound and upper_bound.
If lower bound is missing, assuming a zero lower bound.
If upper bound is missing, assuming infinity upper bound.
"""
function _parse_bounds(tokens, col_names)
    supported_bound_types = ["LO", "UP", "FX", "FR", "MI", "PL"]
    col_mapping = Dict(col => i for (i, col) in enumerate(col_names))
    lower_bound::Vector{Float64} = fill(0.0, length(col_names))
    upper_bound::Vector{Float64} = fill(Inf, length(col_names))
    for token in tokens
        bound_type = token[1]
        @assert(bound_type in supported_bound_types,
            "Unsupported bound type $bound_type for variable $(token[3])")
        col_index = col_mapping[token[3]]

        if bound_type == "LO"
            lower_bound[col_index] = parse(Float64, token[4])
        elseif bound_type == "UP"
            upper_bound[col_index] = parse(Float64, token[4])
        elseif bound_type == "FX"
            lower_bound[col_index] = parse(Float64, token[4])
            upper_bound[col_index] = parse(Float64, token[4])
        elseif bound_type == "FR"
            lower_bound[col_index] = -Inf
            upper_bound[col_index] = +Inf
        elseif bound_type == "MI"
            lower_bound[col_index] = -Inf
        elseif bound_type == "PL"
            upper_bound[col_index] = +Inf
        else
            error("Unknown bound type $bound_type")
        end
    end

    return lower_bound, upper_bound
end

"""
Read cor file into core type.
"""
function read_cor(cor_path::String)::spCorType
    local token
    open(cor_path, "r") do io
        token = _tokenize_cor(io)
    end
    problem_name::String = token["NAME"][1]
    directions, row_names = _parse_row_tokens(token["ROWS"])
    col_names = _parse_unique_columns(token["COLUMNS"])
    template_matrix = _parse_column_to_matrix(token["COLUMNS"],
        row_names, col_names)
    rhs = _parse_rhs(token["RHS"], row_names)
    lb, ub = _parse_bounds(token["BOUNDS"], col_names)

    # Generate mapping from row to indices
    col_mapping = get_name_mapping(col_names)
    row_mapping = get_name_mapping(row_names)

    # Check that the first tow is the objective row
    @assert(directions[1] == 'N',
        "First row or cor file is not objective. $directions")

    cor = spCorType(
        problem_name,
        directions,
        row_names,
        col_names,
        template_matrix,
        rhs,
        lb,
        ub,
        col_mapping,
        row_mapping
    )
    return cor
end

