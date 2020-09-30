
Base.one(A::HA) where {HA<:HybridMatrix} = HA(one(A.data))

Base.fill!(A::HybridArray, x) = fill!(A.data, x)

function Base.zero(a::HybridArray{S}) where {S}
    return HybridArray{S}(zero(a.data))
end
