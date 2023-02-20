function StaticArrayInterface.strides(x::HybridArray)
    return StaticArrayInterface.strides(parent(x))
end

@generated function StaticArrayInterface.strides(x::HybridArray{S,T,N,N,Array{T,N}}) where {S<:Tuple,T,N}
    collected_strides = []
    i = 1
    for (argnum, Sarg) in enumerate(S.parameters)
        if i > 0
            push!(collected_strides, StaticArrayInterface.StaticInt(i))
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
        datastrides = strides(parent(x))
        return tuple($(collected_strides...))
    end
end

@generated function StaticArrayInterface.size(x::HybridArray{S}) where {S}
    collected_sizes = []
    for (argnum, Sarg) in enumerate(S.parameters)
        if Sarg isa Integer
            push!(collected_sizes, StaticArrayInterface.StaticInt(Sarg))
        else
            push!(collected_sizes, :(datasize[$argnum]))
        end
    end
    return quote
        datasize = StaticArrayInterface.size(parent(x))
        return tuple($(collected_sizes...))
    end
end

StaticArrayInterface.contiguous_axis(::Type{HybridArray{S,T,N,N,TData}}) where {S,T,N,TData} = StaticArrayInterface.contiguous_axis(TData)
StaticArrayInterface.contiguous_batch_size(::Type{HybridArray{S,T,N,N,TData}}) where {S,T,N,TData} = StaticArrayInterface.contiguous_batch_size(TData)
StaticArrayInterface.stride_rank(::Type{HybridArray{S,T,N,N,TData}}) where {S,T,N,TData} = StaticArrayInterface.stride_rank(TData)

StaticArrayInterface.dense_dims(x::HybridArray) = StaticArrayInterface.dense_dims(x.data)
