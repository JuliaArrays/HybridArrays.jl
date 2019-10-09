
"""
    size_to_tuple(::Type{S}) where S<:Tuple

Converts a size given by `Tuple{N, M, ...}` into a tuple `(N, M, ...)`.
"""
Base.@pure function size_to_tuple(::Type{S}) where S<:Tuple
    return tuple(S.parameters...)
end


# FOR COMPATIBILITY WITH STATIC ARRAYS 0.11.0 -- REMOVE DURING UPDATE TO THE NEXT VERSION

# Base.strides is intentionally not defined for SArray, see PR #658 for discussion
Base.strides(a::MArray) = Base.size_to_strides(1, size(a)...)
Base.strides(a::SizedArray) = strides(a.data)

Base.unsafe_view(A::AbstractArray, i1::StaticArrays.StaticIndexing, indices::StaticArrays.StaticIndexing...) = Base.unsafe_view(A, StaticArrays.unwrap(i1), map(StaticArrays.unwrap, indices)...)

# Views of views need a new method for Base.SubArray because storing indices
# wrapped in StaticIndexing in field indices of SubArray causes all sorts of problems.
# Additionally, in some cases the SubArray constructor may be called directly
# instead of unsafe_view so we need this method too (Base._maybe_reindex
# is a good example)
Base.SubArray(A::AbstractArray, indices::NTuple{<:Any,StaticArrays.StaticIndexing}) = Base.SubArray(A, map(StaticArrays.unwrap, indices))
