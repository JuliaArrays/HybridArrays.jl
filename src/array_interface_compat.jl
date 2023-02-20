
function ArrayInterface.ismutable(::Type{HybridArray{S,T,N,M,TData}}) where {S,T,N,M,TData}
    return ArrayInterface.ismutable(TData)
end

function ArrayInterface.can_setindex(::Type{HybridArray{S,T,N,M,TData}}) where {S,T,N,M,TData}
    return ArrayInterface.can_setindex(TData)
end

function ArrayInterface.parent_type(::Type{HybridArray{S,T,N,M,TData}}) where {S,T,N,M,TData}
    return TData
end

function ArrayInterface.restructure(x::HybridArray{S}, y) where {S}
    return HybridArray{S}(reshape(convert(Array, y), size(x)...))
end
