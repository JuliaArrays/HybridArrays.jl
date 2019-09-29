
const HybridMatrixLike{T} = Union{
    HybridMatrix{<:Any, <:Any, T},
    Transpose{T, <:HybridMatrix{T}},
    Adjoint{T, <:HybridMatrix{T}},
    Symmetric{T, <:HybridMatrix{T}},
    Hermitian{T, <:HybridMatrix{T}},
    Diagonal{T, <:HybridMatrix{<:Any, T}}
}

const HybridVecOrMatLike{T} = Union{HybridVector{<:Any, T}, HybridMatrixLike{T}}

# Binary ops
# Between arrays
@inline +(a::HybridArray, b::HybridArray) = a .+ b
@inline +(a::AbstractArray, b::HybridArray) = a .+ b
@inline +(a::StaticArray, b::HybridArray) = a .+ b
@inline +(a::HybridArray, b::AbstractArray) = a .+ b
@inline +(a::HybridArray, b::StaticArray) = a .+ b

@inline -(a::HybridArray, b::HybridArray) = a .- b
@inline -(a::AbstractArray, b::HybridArray) = a .- b
@inline -(a::StaticArray, b::HybridArray) = a .- b
@inline -(a::HybridArray, b::AbstractArray) = a .- b
@inline -(a::HybridArray, b::StaticArray) = a .- b

# Scalar-array
@inline *(a::Number, b::HybridArray) = a .* b
@inline *(a::HybridArray, b::Number) = a .* b

@inline /(a::HybridArray, b::Number) = a ./ b
@inline \(a::Number, b::HybridArray) = a .\ b


@inline vcat(a::HybridVecOrMatLike) = a
@inline vcat(a::HybridVecOrMatLike, b::HybridVecOrMatLike) = _vcat(Size(a), Size(b), a, b)
@inline vcat(a::HybridVecOrMatLike, b::HybridVecOrMatLike, c::HybridVecOrMatLike...) = vcat(vcat(a,b), vcat(c...))

@generated function _vcat(::Size{Sa}, ::Size{Sb}, a::HybridVecOrMatLike, b::HybridVecOrMatLike) where {Sa, Sb}
    if Size(Sa)[2] != Size(Sb)[2]
        throw(DimensionMismatch("Tried to vcat arrays of size $Sa and $Sb"))
    end

    if a <: HybridVector && b <: HybridVector
        Snew = (Sa[1] + Sb[1],)
    else
        Snew = (Sa[1] + Sb[1], Size(Sa)[2])
    end

    return quote
        Base.@_inline_meta
        @inbounds return _h_similar_type($a, promote_type(eltype(a), eltype(b)), Size($Snew))(vcat(a.data, b.data))
    end
end

@inline hcat(a::HybridVecOrMatLike, b::HybridVecOrMatLike) = _hcat(Size(a), Size(b), a, b)
@inline hcat(a::HybridVecOrMatLike, b::HybridVecOrMatLike, c::HybridVecOrMatLike...) = hcat(hcat(a,b), hcat(c...))

# used in _vcat and _hcat
@inline +(::Dynamic, ::Dynamic) = Dynamic()

@generated function _hcat(::Size{Sa}, ::Size{Sb}, a::HybridVecOrMatLike, b::HybridVecOrMatLike) where {Sa, Sb}
    if Sa[1] != Sb[1]
        throw(DimensionMismatch("Tried to hcat arrays of size $Sa and $Sb"))
    end

    Snew = (Sa[1], Size(Sa)[2] + Size(Sb)[2])

    return quote
        Base.@_inline_meta
        return _h_similar_type($a, promote_type(eltype(a), eltype(b)), Size($Snew))(hcat(a.data, b.data))
    end
end
