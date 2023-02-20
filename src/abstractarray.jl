
@inline Base.parent(sa::HybridArray) = sa.data

Base.dataids(sa::HybridArray) = Base.dataids(parent(sa))

@inline Base.elsize(sa::HybridArray) = Base.elsize(parent(sa))

@inline size(sa::HybridArray{S,T,N,N}) where {S<:Tuple,T,N} = size(parent(sa))

@inline length(sa::HybridArray) = length(parent(sa))

@inline strides(sa::HybridArray{S,T,N,N}) where {S<:Tuple,T,N} = strides(parent(sa))

@inline pointer(sa::HybridArray) = pointer(parent(sa))

@generated function _sized_abstract_array_axes(::Type{S}, ax::Tuple) where {S<:Tuple}
    exprs = Any[]
    map(enumerate(S.parameters)) do (i, si)
        if isa(si, Dynamic)
            push!(exprs, :(ax[$i]))
        else
            push!(exprs, SOneTo(si))
        end
    end
    return Expr(:tuple, exprs...)
end

function axes(sa::HybridArray{S}) where {S<:Tuple}
    ax = axes(parent(sa))
    return _sized_abstract_array_axes(S, ax)
end


function promote_rule(::Type{<:HybridArray{S,T,N,M,TDataA}}, ::Type{<:HybridArray{S,U,N,M,TDataB}}) where {S<:Tuple,T,U,N,M,TDataA,TDataB}
    TU = promote_type(T,U)
    HybridArray{S,TU,N,M,promote_type(TDataA, TDataB)::Type{<:AbstractArray{TU}}}
end

@inline copy(a::HybridArray{S, T, N, M}) where {S<:Tuple, T, N, M} = begin
    parentcopy = copy(parent(a))
    HybridArray{S, T, N, M, typeof(parentcopy)}(parentcopy)
end


homogenized_last(::StaticArrays.HeterogeneousBaseShape) = StaticArrays.Dynamic()
homogenized_last(a::SOneTo) = last(a)

hybrid_homogenize_shape(::Tuple{}) = ()
hybrid_homogenize_shape(shape::Tuple{Vararg{StaticArrays.HeterogeneousShape}}) = Size(map(homogenized_last, shape))
hybrid_homogenize_shape(shape::Tuple{Vararg{StaticArrays.HeterogeneousBaseShape}}) = map(last, shape)

similar(a::HA, ::Type{T2}) where {S, HA<:HybridArray{S}, T2} = HybridArray{S, T2}(similar(parent(a), T2))
function similar(a::HybridArray, ::Type{T2}, shape::StaticArrays.HeterogeneousShapeTuple) where {T2}
    s = hybrid_homogenize_shape(shape)
    HT = hybridarray_similar_type(T2, s, StaticArrays.length_val(s))
    return HT(similar(a.data, T2, shape))
end
function similar(a::HybridArray, shape::StaticArrays.HeterogeneousShapeTuple)
    return similar(a, eltype(a), shape)
end
function similar(a::HybridArray, ::Type{T2}, shape::Tuple{SOneTo, Vararg{SOneTo}}) where {T2}
    return similar(a.data, T2, StaticArrays.homogenize_shape(shape))
end
function similar(a::HybridArray, shape::Tuple{SOneTo, Vararg{SOneTo}})
    return similar(a.data, StaticArrays.homogenize_shape(shape))
end

similar(::Type{<:HybridArray{S,T,N,M}},::Type{T2}) where {S,T,N,M,T2} = HybridArray{S,T2,N,M}(undef)
similar(::Type{SA},::Type{T},s::Size{S}) where {SA<:HybridArray,T,S} = hybridarray_similar_type(T,s,StaticArrays.length_val(s))(undef)
hybridarray_similar_type(::Type{T},s::Size{S},::Type{Val{D}}) where {T,S,D} = HybridArray{Tuple{S...},T,D,length(s)}

# for internal use only (used in vcat and hcat)
# TODO: try to make this less hacky
# adding method to similar_type ends up being used in broadcasting for some reason?
_h_similar_type(::Type{A},::Type{T},s::Size{S}) where {A<:HybridArray,T,S} = hybridarray_similar_type(T,s,StaticArrays.length_val(s))


Size(::Type{<:HybridArray{S}}) where {S} = Size(S)

Base.IndexStyle(a::HybridArray) = Base.IndexStyle(parent(a))
Base.IndexStyle(::Type{HA}) where {S,T,N,M,TData,HA<:HybridArray{S,T,N,M,TData}} = Base.IndexStyle(TData)

Base.vec(a::HybridArray) = vec(parent(a))

StaticArrays.similar_type(::Type{<:SSubArray},::Type{T},s::Size{S}) where {S,T} = StaticArrays.default_similar_type(T,s,StaticArrays.length_val(s))
