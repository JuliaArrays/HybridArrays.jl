@inline function getindex(sa::HybridArray{S}, ::Colon) where S
    return HybridArray{S}(getindex(sa.data, :))
end

Base.@propagate_inbounds function getindex(sa::HybridArray{S}, inds::Int...) where S
    return getindex(sa.data, inds...)
end

Base.@propagate_inbounds function getindex(sa::HybridArray{S}, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon}...) where S
    _getindex(all_dynamic_fixed_val(S, inds...), sa, inds...)
end

Base.@propagate_inbounds function _getindex(::Val{:dynamic_fixed_true}, sa::HybridArray, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon}...)
    return _getindex_all_static(sa, inds...)
end

function _get_indices(i::Tuple{}, j::Int)
    return ()
end

function _get_indices(i::Tuple, j::Int, i1::Type{Int}, inds...)
    return (:(inds[$j]), _get_indices(i, j+1, inds...)...)
end

function _get_indices(i::Tuple, j::Int, i1::Type{T}, inds...) where T<:StaticArray{<:Tuple, Int}
    return (:(inds[$j][$(i[1])]), _get_indices(i[2:end], j+1, inds...)...)
end

function _get_indices(i::Tuple, j::Int, i1::Type{Colon}, inds...)
    return (i[1], _get_indices(i[2:end], j+1, inds...)...)
end

@generated function _getindex_all_static(sa::HybridArray{S,T}, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon}...) where {S,T}
    newsize = new_out_size_nongen(S, inds...)
    exprs = Vector{Expr}(undef, length(newsize))

    indices = CartesianIndices(newsize)
    exprs = similar(indices, Expr)
    for current_ind ∈ indices
        cinds = _get_indices(current_ind.I, 1, inds...)
        exprs[current_ind.I...] = :(getindex(sa.data, $(cinds...)))
    end
    Tnewsize = Tuple{newsize...}
    return quote
        Base.@_propagate_inbounds_meta
        SArray{$Tnewsize,$T}(tuple($(exprs...)))
    end
end

function new_out_size_nongen(::Type{Size}, inds...) where Size
    os = []
    map(Size.parameters, inds) do s, i
        if i == Int
        elseif i <: StaticVector
            push!(os, i.parameters[1].parameters[1])
        elseif i == Colon
            push!(os, s)
        else
            error("Unknown index type: $i")
        end
    end
    return tuple(os...)
end

@generated function new_out_size(::Type{Size}, inds...) where Size
    os = []
    map(Size.parameters, inds) do s, i
        if i == Int
        elseif i <: StaticVector
            push!(os, i.parameters[1].parameters[1])
        elseif i == Colon
            push!(os, s)
        else
            error("Unknown index type: $i")
        end
    end
    return Tuple{os...}
end

@inline function _getindex(::Val{:dynamic_fixed_false}, sa::HybridArray{S}, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon}...) where S
    newsize = new_out_size(S, inds...)
    return HybridArray{newsize}(getindex(sa.data, inds...))
end

# setindex stuff

Base.@propagate_inbounds function setindex!(a::HybridArray, value, inds::Int...)
    Base.@boundscheck checkbounds(a, inds...)
    _setindex!_scalar(Size(a), a, value, inds...)
end

@generated function _setindex!_scalar(::Size{S}, a::HybridArray, value, inds::Int...) where S
    if length(inds) == 0
        return quote
            @_propagate_inbounds_meta
            a[1] = value
        end
    end

    stride = :(1)
    ind_expr = :()
    for i ∈ 1:length(inds)
        if i == 1
            ind_expr = :(inds[1])
        else
            ind_expr = :($ind_expr + $stride * (inds[$i] - 1))
        end
        if isa(S[i], StaticArrays.Dynamic)
            stride = :($stride * size(a.data, $i))
        else
            stride = :($stride * $(S[i]))
        end
    end
    return quote
        Base.@_inline_meta
        Base.@_propagate_inbounds_meta
        a.data[$ind_expr] = value
    end
end
