
Base.@propagate_inbounds (::Type{HybridArray{S,T,N,M,TData}})(a::AbstractArray) where {S,T,N,M,TData<:AbstractArray{<:Any,M}} = convert(HybridArray{S,T,N,M,TData}, a)
Base.@propagate_inbounds (::Type{HybridArray{S,T,N,M}})(a::AbstractArray) where {S,T,N,M} = convert(HybridArray{S,T,N,M}, a)
Base.@propagate_inbounds (::Type{HybridArray{S,T,N}})(a::AbstractArray) where {S,T,N} = convert(HybridArray{S,T,N}, a)
Base.@propagate_inbounds (::Type{HybridArray{S,T}})(a::AbstractArray) where {S,T} = convert(HybridArray{S,T,StaticArrays.tuple_length(S)}, a)

# Overide some problematic default behaviour
@inline convert(::Type{SA}, sa::HybridArray) where {SA<:HybridArray} = SA(sa.data)
@inline convert(::Type{SA}, sa::SA) where {SA<:HybridArray} = sa

# Back to Array (unfortunately need both convert and construct to overide other methods)
@inline Array(sa::HybridArray{S}) where {S} = Array(reshape(sa.data, size_to_tuple(S)))
@inline Array{T}(sa::HybridArray{S,T}) where {T,S} = Array(reshape(sa.data, size_to_tuple(S)))
@inline Array{T,N}(sa::HybridArray{S,T,N}) where {T,S,N} = Array(reshape(sa.data, size_to_tuple(S)))

@inline convert(::Type{Array}, sa::HybridArray{S}) where {S} = Array(reshape(sa.data, size_to_tuple(S)))
@inline convert(::Type{Array{T}}, sa::HybridArray{S,T}) where {T,S} = Array(reshape(sa.data, size_to_tuple(S)))
@inline convert(::Type{Array{T,N}}, sa::HybridArray{S,T,N}) where {T,S,N} = Array(reshape(sa.data, size_to_tuple(S)))

@inline function convert(::Type{HybridArray{S,T,N,M,TData}}, a::AbstractArray) where {S,T,N,M,U,TData<:AbstractArray{U,M}}
    # TODO: dimension checking
    as = convert(TData, a)
    return HybridArray{S,T,N,M,TData}(as)
end

@inline function convert(::Type{HybridArray{S,T,N,M}}, a::TData) where {S,T,N,M,U,TData<:AbstractArray{U,M}}
    # TODO: dimension checking
    as = similar(a, T)
    copyto!(as, a)
    return HybridArray{S,T,N,M,typeof(as)}(as)
end

@inline function convert(::Type{HybridArray{S,T,N,M}}, a::TData) where {S,T,N,M,TData<:AbstractArray{T,M}}
    # TODO: dimension checking?
    return HybridArray{S,T,N,M,typeof(a)}(a)
end

@inline function convert(::Type{HybridArray{S,T,N}}, a::TData) where {S,T,N,M,U,TData<:AbstractArray{U,M}}
    # TODO: dimension checking
    as = similar(a, T)
    copyto!(as, a)
    return HybridArray{S,T,N,M,typeof(as)}(as)
end

@inline function convert(::Type{HybridArray{S,T,N}}, a::TData) where {S,T,N,M,TData<:AbstractArray{T,M}}
    # TODO: dimension checking?
    return HybridArray{S,T,N,M,typeof(a)}(a)
end
