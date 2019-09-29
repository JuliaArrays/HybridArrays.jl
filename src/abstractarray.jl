
Base.dataids(sa::HybridArray) = Base.dataids(sa.data)

@inline size(sa::HybridArray{S}) where S = size(sa.data)

@inline length(sa::HybridArray) = length(sa.data)

@generated function _sized_abstract_array_axes(::Type{S}, ax::Tuple) where S<:Tuple
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

function axes(sa::HybridArray{S}) where S
    ax = axes(sa.data)
    return _sized_abstract_array_axes(S, ax)
end


function promote_rule(::Type{<:HybridArray{S,T,N,M,TDataA}}, ::Type{<:HybridArray{S,U,N,M,TDataB}}) where {S,T,U,N,M,TDataA,TDataB}
    TU = promote_type(T,U)
    HybridArray{S,TU,N,M,promote_type(TDataA, TDataB)::Type{<:AbstractArray{TU}}}
end

@inline copy(a::HybridArray) = typeof(a)(copy(a.data))

similar(::Type{<:HybridArray{S,T,N,M}},::Type{T2}) where {S,T,N,M,T2} = HybridArray{S,T2,N,M}(undef)
similar(::Type{SA},::Type{T},s::Size{S}) where {SA<:HybridArray,T,S} = hybridarray_similar_type(T,s,StaticArrays.length_val(s))(undef)
hybridarray_similar_type(::Type{T},s::Size{S},::Type{Val{D}}) where {T,S,D} = HybridArray{Tuple{S...},T,D,length(s)}

# for internal use only (used in vcat and hcat)
# TODO: try to make this less hacky
# adding method to similar_type ends up being used in broadcasting for some reason?
_h_similar_type(::Type{A},::Type{T},s::Size{S}) where {A<:HybridArray,T,S} = hybridarray_similar_type(T,s,StaticArrays.length_val(s))


Size(::Type{<:HybridArray{S}}) where {S} = Size(S)
