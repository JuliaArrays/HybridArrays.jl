
Base.one(A::HA) where {HA<:HybridMatrix} = HA(one(parent(A)))

# This should make one(sized subarray) return SArray
@inline function StaticArrays._construct_sametype(a::Type{<:SSubArray{Tuple{S,S},T}}, elements) where {S,T}
    return SMatrix{S,S,T}(elements)
end
@inline function StaticArrays._construct_sametype(a::SSubArray{Tuple{S,S},T}, elements) where {S,T}
    return SMatrix{S,S,T}(elements)
end

Base.fill!(A::HybridArray, x) = fill!(parent(A), x)

@inline function Base.zero(a::HybridArray{S}) where {S}
    return HybridArray{S}(zero(parent(a)))
end

@inline function Base.zero(a::SSubArray{S,T}) where {S,T}
    return StaticArrays.zeros(SArray{S,T})
end
