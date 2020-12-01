
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

function ArrayInterface.strides(x::HybridArray)
    return ArrayInterface.strides(x.data)
end

@generated function ArrayInterface.strides(x::HybridArray{S,T,N,N,Array{T,N}}) where {S,T,N}
    collected_strides = []
    i = 1
    for (argnum, Sarg) in enumerate(S.parameters)
        if i > 0
            push!(collected_strides, ArrayInterface.StaticInt(i))
        else
            push!(collected_strides, :(datastrides[$argnum]))
        end
        if Sarg isa Integer
            i *= Sarg
        else
            i = -1
        end
    end
    return quote
        datastrides = strides(x.data)
        return tuple($(collected_strides...))
    end
end

@generated function ArrayInterface.size(x::HybridArray{S}) where {S}
    collected_sizes = []
    for (argnum, Sarg) in enumerate(S.parameters)
        if Sarg isa Integer
            push!(collected_sizes, ArrayInterface.StaticInt(Sarg))
        else
            push!(collected_sizes, :(datasize[$argnum]))
        end
    end
    return quote
        datasize = ArrayInterface.size(x.data)
        return tuple($(collected_sizes...))
    end
end

ArrayInterface.contiguous_axis(::Type{HybridArray{S,T,N,N,TData}}) where {S,T,N,TData} = ArrayInterface.contiguous_axis(TData)
ArrayInterface.contiguous_batch_size(::Type{HybridArray{S,T,N,N,TData}}) where {S,T,N,TData} = ArrayInterface.contiguous_batch_size(TData)
ArrayInterface.stride_rank(::Type{HybridArray{S,T,N,N,TData}}) where {S,T,N,TData} = ArrayInterface.stride_rank(TData)
