
Base.@propagate_inbounds (::Type{HybridArray{S,T,N,M,TData}})(a::AbstractArray) where {S,T,N,M,TData<:AbstractArray{<:Any,M}} = convert(HybridArray{S,T,N,M,TData}, a)
Base.@propagate_inbounds (::Type{HybridArray{S,T,N,M}})(a::AbstractArray) where {S,T,N,M} = convert(HybridArray{S,T,N,M}, a)
Base.@propagate_inbounds (::Type{HybridArray{S,T,N}})(a::AbstractArray) where {S,T,N} = convert(HybridArray{S,T,N}, a)
Base.@propagate_inbounds (::Type{HybridArray{S,T}})(a::AbstractArray) where {S,T} = convert(HybridArray{S,T,StaticArrays.tuple_length(S)}, a)

# Overide some problematic default behaviour
@inline convert(::Type{SA}, sa::HybridArray) where {SA<:HybridArray} = SA(sa.data)
@inline convert(::Type{SA}, sa::SA) where {SA<:HybridArray} = sa

# Back to Array (unfortunately need both convert and construct to overide other methods)
@inline Array(sa::HybridArray{S}) where {S} = Array(sa.data)
@inline Array{T}(sa::HybridArray{S,T}) where {T,S} = Array{T}(sa.data)
@inline Array{T,N}(sa::HybridArray{S,T,N}) where {T,S,N} = Array{T,N}(sa.data)

@inline convert(::Type{Array}, sa::HybridArray) = convert(Array, sa.data)
@inline convert(::Type{Array{T}}, sa::HybridArray{S,T}) where {T,S} = convert(Array, sa.data)
@inline convert(::Type{Array{T,N}}, sa::HybridArray{S,T,N}) where {T,S,N} = convert(Array, sa.data)

function check_compatible_sizes(::Type{S}, a::NTuple{N,Int}) where {S,N}
    st = size_to_tuple(S)
    for (s1, s2) ∈ zip(st, a)
        if !isa(s1, Dynamic) && s1 != s2
            error("Array $a has size incompatible with $S.")
        end
    end
    return true
end

@inline function convert(::Type{HybridArray{S,T,N,M,TData}}, a::AbstractArray{<:Any,M}) where {S,T,N,M,U,TData<:AbstractArray{U,M}}
    check_compatible_sizes(S, size(a))
    as = convert(TData, a)
    return HybridArray{S,T,N,M,TData}(as)
end

@inline function convert(::Type{HybridArray{S,T,N,M}}, a::TData) where {S,T,N,M,U,TData<:AbstractArray{U,M}}
    check_compatible_sizes(S, size(a))
    as = similar(a, T)
    copyto!(as, a)
    return HybridArray{S,T,N,M,typeof(as)}(as)
end

@inline function convert(::Type{HybridArray{S,T,N,M}}, a::TData) where {S,T,N,M,TData<:AbstractArray{T,M}}
    check_compatible_sizes(S, size(a))
    return HybridArray{S,T,N,M,typeof(a)}(a)
end

@inline function convert(::Type{HybridArray{S,T,N}}, a::TData) where {S,T,N,M,U,TData<:AbstractArray{U,M}}
    check_compatible_sizes(S, size(a))
    as = similar(a, T)
    copyto!(as, a)
    return HybridArray{S,T,N,M,typeof(as)}(as)
end

@inline function convert(::Type{HybridArray{S,T,N}}, a::TData) where {S,T,N,M,TData<:AbstractArray{T,M}}
    check_compatible_sizes(S, size(a))
    return HybridArray{S,T,N,M,typeof(a)}(a)
end
