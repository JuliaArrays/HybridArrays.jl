
function StaticArrays._one(s::Size, ::Type{SM}) where {Sel, N, SM <: HybridArrays.SSubArray{Tuple{N,N}, Sel}}
    return StaticArrays._one(s, SMatrix{N,N,Sel})
end

Base.one(A::HA) where {HA<:HybridMatrix} = HA(one(A.data))

@inline function Base.zero(a::HA) where {S, Sel, HA <: SSubArray{S, Sel}}
    return StaticArrays.zeros(SArray{S, Sel})
end

Base.fill!(A::HybridArray, x) = fill!(A.data, x)
