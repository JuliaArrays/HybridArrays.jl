module HybridArrays

import Base: convert,
    copy,
    copyto!,
    hcat,
    vcat,
    similar,
    axes,
    getindex,
    dataids,
    promote_rule,
    pointer,
    strides,
    setindex!,
    size,
    length,
    +,
    -,
    *

import Base.Array

using StaticArrays
using StaticArrays: Dynamic, StaticIndexing
import StaticArrays: _setindex!_scalar, Size

using LinearAlgebra
using Requires


@generated function hasdynamic(::Type{Size}) where Size<:Tuple
    for s ∈ Size.parameters
        if isa(s, Dynamic)
            return true
        end
    end
    return false
end

function all_dynamic_fixed_val(::Type{Tuple{}})
    return Val(:dynamic_fixed_true)
end
function all_dynamic_fixed_val(::Type{Size}) where Size<:Tuple
    return error("No indices given for size $Size")
end

function all_dynamic_fixed_val(::Type{Size}, inds::StaticArrays.StaticIndexing{<:Union{Int, AbstractArray, Colon}}...) where Size<:Tuple
    return all_dynamic_fixed_val(Size, map(StaticArrays.unwrap, inds)...)
end

@generated function all_dynamic_fixed_val(::Type{Size}, inds::Union{Int, AbstractArray, Colon}...) where Size<:Tuple
    all_fixed = true
    for (i, param) in enumerate(Size.parameters)

        destatizing = (inds[i] <: AbstractArray && !(
            inds[i] <: StaticArray ||
            inds[i] <: Base.Slice ||
            inds[i] <: SOneTo))

        nonstatizing = inds[i] == Colon || inds[i] <: Base.Slice || destatizing

        if destatizing || (isa(param, Dynamic) && nonstatizing)
            all_fixed = false
            break
        end

    end

    if all_fixed
        return Val(:dynamic_fixed_true)
    else
        return Val(:dynamic_fixed_false)
    end
end

function has_dynamic(::Type{Size}) where Size<:Tuple
    for param in Size.parameters
        if isa(param, Dynamic)
            return true
        end
    end
    return false
end

@generated function all_dynamic_fixed_val(::Type{Size}, inds::Union{Colon, Base.Slice}) where Size<:Tuple
    if has_dynamic(Size)
        return Val(:dynamic_fixed_false)
    else
        return Val(:dynamic_fixed_true)
    end
end

@generated function tuple_nodynamic_prod(::Type{Size}) where Size<:Tuple
    i = 1
    for s ∈ Size.parameters
        if !isa(s, Dynamic)
            i *= s
        end
    end
    return i
end

# conversion of SizedArray from StaticArrays.jl
"""
    HybridArray{Tuple{dims...}}(array)

Wraps an `AbstractArray` with a combination of static and dynamic sizes,
so to take advantage of the (faster) methods defined by the StaticArrays
package. The size is checked once upon construction to determine if the
number of elements (`length`) match, but the array may be reshaped.
"""
struct HybridArray{S<:Tuple, T, N, M, TData<:AbstractArray{T,M}} <: AbstractArray{T, N}
    data::TData

    function HybridArray{S, T, N, M, TData}(a::TData) where {S, T, N, M, TData<:AbstractArray{T,M}}
        tnp = tuple_nodynamic_prod(S)
        lena = length(a)
        dynamic_nodivisible = hasdynamic(S) && tnp != 0 && mod(lena, tnp) != 0
        nodynamic_notequal = !hasdynamic(S) && lena != StaticArrays.tuple_prod(S)
        if nodynamic_notequal || dynamic_nodivisible || (tnp == 0 && lena != 0)
            error("Dimensions $(size(a)) don't match static size $S")
        end
        new{S,T,N,M,TData}(a)
    end

    function HybridArray{S, T, N, 1}(::UndefInitializer) where {S, T, N}
        new{S, T, N, 1, Array{T, 1}}(Array{T, 1}(undef, StaticArrays.tuple_prod(S)))
    end
    function HybridArray{S, T, N, N}(::UndefInitializer) where {S, T, N}
        new{S, T, N, N, Array{T, N}}(Array{T, N}(undef, size_to_tuple(S)...))
    end
end

@inline HybridArray{S,T,N}(a::TData) where {S,T,N,M,TData<:AbstractArray{T,M}} = HybridArray{S,T,N,M,TData}(a)
@inline HybridArray{S,T}(a::TData) where {S,T,M,TData<:AbstractArray{T,M}} = HybridArray{S,T,StaticArrays.tuple_length(S),M,TData}(a)
@inline HybridArray{S}(a::TData) where {S,T,M,TData<:AbstractArray{T,M}} = HybridArray{S,T,StaticArrays.tuple_length(S),M,TData}(a)

@inline HybridArray{S,T,N}(::UndefInitializer) where {S,T,N} = HybridArray{S,T,N,N}(undef)
@inline HybridArray{S,T}(::UndefInitializer) where {S,T} = HybridArray{S,T,StaticArrays.tuple_length(S),StaticArrays.tuple_length(S)}(undef)

@inline HybridArray{S,T}(::UndefInitializer, d::Integer...) where {S,T} = HybridArray{S,T}(Array{T}(undef, d))

@generated function (::Type{HybridArray{S,T,N,M,TData}})(x::NTuple{L,Any}) where {S,T,N,M,TData,L}
    if L != StaticArrays.tuple_prod(S)
        error("Dimension mismatch")
    end
    exprs = [:(a[$i] = x[$i]) for i = 1:L]
    return quote
        $(Expr(:meta, :inline))
        a = HybridArray{S,T,N,M}(undef)
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end

@inline HybridArray{S,T,N}(x::Tuple) where {S,T,N} = HybridArray{S,T,N,N,Array{T,N}}(x)
@inline HybridArray{S,T}(x::Tuple) where {S,T} = HybridArray{S,T,StaticArrays.tuple_length(S)}(x)
@inline HybridArray{S}(x::NTuple{L,T}) where {S,T,L} = HybridArray{S,T}(x)

HybridVector{S,T,M} = HybridArray{Tuple{S},T,1,M}
@inline HybridVector{S}(a::TData) where {S,T,M,TData<:AbstractArray{T,M}} = HybridArray{Tuple{S},T,1,M,TData}(a)
@inline HybridVector{S}(x::NTuple{L,T}) where {S,T,L} = HybridArray{Tuple{S},T,1,1,Vector{T}}(x)
@inline HybridVector{S,T}(::UndefInitializer, d::Integer) where {S,T} = HybridVector{S,T}(Vector{T}(undef, d))

HybridMatrix{S1,S2,T,M} = HybridArray{Tuple{S1,S2},T,2,M}
@inline HybridMatrix{S1,S2}(a::TData) where {S1,S2,T,M,TData<:AbstractArray{T,M}} = HybridArray{Tuple{S1,S2},T,2,M,TData}(a)
@inline HybridMatrix{S1,S2}(x::NTuple{L,T}) where {S1,S2,T,L} = HybridArray{Tuple{S1,S2},T,2,2,Matrix{T}}(x)
@inline HybridMatrix{S1,S2,T}(::UndefInitializer, d1::Integer, d2::Integer) where {S1,S2,T} = HybridMatrix{S1,S2,T}(Matrix{T}(undef, d1, d2))

export HybridArray, HybridMatrix, HybridVector

include("SSubArray.jl")

include("abstractarray.jl")
include("arraymath.jl")
include("broadcast.jl")
include("convert.jl")
include("indexing.jl")
include("linalg.jl")
include("utils.jl")

function __init__()
    @require ArrayInterface="4fba245c-0d91-5ea0-9b3e-6abc04ee57a9" begin
        include("array_interface_compat.jl")
    end
    @require StaticArrayInterface="0d7ed370-da01-4f52-bd93-41d350b8b718" begin
        include("static_array_interface_compat.jl")
    end
end

end # module
