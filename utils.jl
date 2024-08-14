"""
Turn a vector into dict for reverse finding.
"""
function get_name_mapping(v::Vector{T})::Dict{T, Int} where T
    d = Dict{T, Int}()
    for i in eachindex(v)
        d[v[i]] = i
    end
    return d
end