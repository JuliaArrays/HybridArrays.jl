
"""
    size_to_tuple(::Type{S}) where S<:Tuple

Converts a size given by `Tuple{N, M, ...}` into a tuple `(N, M, ...)`.
"""
Base.@pure function size_to_tuple(::Type{S}) where S<:Tuple
    return tuple(S.parameters...)
end
