@inline function getindex(sa::HybridArray{S}, ::Colon) where S
    return HybridArray{S}(getindex(sa.data, :))
end

Base.@propagate_inbounds function getindex(sa::HybridArray{S}, inds::Int...) where S
    return getindex(sa.data, inds...)
end

Base.@propagate_inbounds function getindex(sa::HybridArray{S}, inds::Union{Int, AbstractArray, Colon}...) where S
    _getindex(all_dynamic_fixed_val(sa, inds...), sa, inds...)
end

# general fallback allowing, e.g. `..` from EllipsisNotation.jl
Base.@propagate_inbounds function getindex(sa::HybridArray{S}, inds...) where S
    getindex(sa, to_indices(sa, inds)...)
end

Base.@propagate_inbounds function _getindex(::Val{:dynamic_fixed_true}, sa::HybridArray, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon, Base.Slice}...)
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

function _get_indices(i::Tuple, j::Int, i1::Union{Type{Colon},Type{<:Base.Slice}}, inds...)
    return (i[1], _get_indices(i[2:end], j+1, inds...)...)
end

_totally_linear() = true
_totally_linear(inds...) = false
_totally_linear(inds::Type{Int}...) = true
_totally_linear(inds::Union{Type{Colon}, Type{<:Base.Slice}}...) = true
_totally_linear(i1::Union{Type{Colon}, Type{<:Base.Slice}}, inds...) = _totally_linear(inds...)

# this gets types as arguments since it's called from `@generated function`s
function new_out_size_nongen(::Type{<:HybridArray{Size}}, inds::Union{Type{Int}, Type{<:AbstractArray}, Type{Colon}}...) where Size
    os = []
    map(Size.parameters, inds) do s, i
        if i == Int
            # nothing to do since this size will be dropped
        elseif i <: StaticVector
            push!(os, length(i))
        elseif i == Colon || i <: Base.Slice
            push!(os, s)
        else
            error("Unknown index type: $i")
        end
    end
    return tuple(os...)
end

# general fallback allowing, e.g. `..` from EllipsisNotation.jl
function new_out_size_nongen(a::Type{<:HybridArray{Size}}, inds...) where Size
    return new_out_size_nongen(a, to_indices(a, inds)...)
end

# this gets types as arguments since it's called from `@generated function`s
"""
    _get_linear_inds(sa::Type{<:HybridArray{Size}}, inds...)

Returns linear indices for given `Size` and access indices.
Order is selected to make setindex! with array input be linearly indexed by
position of index in returned vector.
"""
function _get_linear_inds(sa::Type{<:HybridArray{Size}}, inds...) where Size
    newsize = new_out_size_nongen(sa, inds...)
    indices = CartesianIndices(newsize)
    out_inds = Any[]
    sizeprods = Union{Int, Expr}[1]
    needs_dynsize = false
    for (i, s) in enumerate(Size.parameters[1:(end-1)])
        if isa(s, Int) && isa(sizeprods[end], Int)
            push!(sizeprods, s*sizeprods[end])
        elseif isa(s, Int)
            push!(sizeprods, :($s*$(sizeprods[end])))
        elseif isa(s, Dynamic)
            needs_dynsize = true
            push!(sizeprods, :(dynsize[$i]*$(sizeprods[end])))
        end
    end

    needs_sizeprod = false
    if _totally_linear(inds...)
        i1 = 0
        for (iik, ik) ∈ enumerate(inds)
            if isa(ik, Type{Int})
                needs_sizeprod = true
                i1 = :($i1 + (inds[$iik]-1)*sizeprods[$iik])
            end
        end
        out_inds = tuple((:(i1 + $k) for k ∈ 1:StaticArrays.tuple_prod(newsize))...)

        preamble = if needs_sizeprod && needs_dynsize
            quote
                dynsize = size(sa)
                sizeprods = tuple($(sizeprods...))
                i1 = $i1
            end
        elseif needs_sizeprod
            quote
                sizeprods = tuple($(sizeprods...))
                i1 = $i1
            end
        else
            quote
                i1 = $i1
            end
        end
        return preamble, out_inds
    else
        return nothing
    end
end

@generated function _getindex_all_static(sa::HybridArray{S,T}, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon, Base.Slice}...) where {S,T}
    newsize = new_out_size_nongen(sa, inds...)
    exprs = Vector{Expr}(undef, length(newsize))

    indices = CartesianIndices(newsize)
    exprs = similar(indices, Expr)
    Tnewsize = Tuple{newsize...}

    lininds = _get_linear_inds(sa, inds...)
    if lininds === nothing
        for current_ind ∈ indices
            cinds = _get_indices(current_ind.I, 1, inds...)
            exprs[current_ind.I...] = :(getindex(sadata, $(cinds...)))
        end
        return quote
            Base.@_propagate_inbounds_meta
            sadata = sa.data
            SArray{$Tnewsize,$T}(tuple($(exprs...)))
        end
    else
        exprs = [:(getindex(sadata, $id)) for id ∈ lininds[2]]
        return quote
            Base.@_propagate_inbounds_meta
            sadata = sa.data
            $(lininds[1])
            SArray{$Tnewsize,$T}(tuple($(exprs...)))
        end
    end
end

function new_out_size(sa::HybridArray{Size}, inds::StaticArrays.StaticIndexing...) where Size
    return new_out_size(sa, map(StaticArrays.unwrap, inds)...)
end


# _get_static_vector_length is used in a generated function so using a generic function
# may not be a good idea
_get_static_vector_length(::Type{<:StaticVector{N}}) where {N} = N

@generated function new_out_size(::HybridArray{Size}, inds::Union{Int, AbstractArray, Colon}...) where Size
    os = []
    map(Size.parameters, inds) do s, i
        if i == Int
            # nothing to do since this size will be dropped
        elseif i <: StaticVector
            push!(os, _get_static_vector_length(i))
        elseif i == Colon || i <: Base.Slice
            push!(os, s)
        elseif i <: SOneTo
            push!(os, i.parameters[1])
        elseif i <: Base.OneTo || i <: AbstractArray
            push!(os, Dynamic())
        else
            error("Unknown index type: $i")
        end
    end
    return Tuple{os...}
end

# general fallback allowing, e.g. `..` from EllipsisNotation.jl
function new_out_size(a::HybridArray{Size}, inds...) where Size
    return new_out_size(a, to_indices(a, inds)...)
end

@inline function _getindex(::Val{:dynamic_fixed_false}, sa::HybridArray{S}, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon, Base.Slice}...) where S
    newsize = new_out_size(sa, inds...)
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
            Base.@_propagate_inbounds_meta
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
        # it's possible that one of the trailing indices is an additional 1.
        if length(S) < i || isa(S[i], StaticArrays.Dynamic)
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

Base.@propagate_inbounds function setindex!(sa::HybridArray{S}, value, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon, Base.Slice}...) where S
    _setindex!(all_dynamic_fixed_val(sa, inds...), sa, value, inds...)
end

@inline function _setindex!(::Val{:dynamic_fixed_false}, sa::HybridArray{S}, value, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon, Base.Slice}...) where S
    newsize = new_out_size(sa, inds...)
    return HybridArray{newsize}(setindex!(sa.data, value, inds...))
end

Base.@propagate_inbounds function _setindex!(::Val{:dynamic_fixed_true}, sa::HybridArray, value, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon, Base.Slice}...)
    return _setindex!_all_static(sa, value, inds...)
end

@generated function _setindex!_all_static(sa::HybridArray{S,T}, v::AbstractArray, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon, Base.Slice}...) where {S,T}
    newsize = new_out_size_nongen(sa, inds...)
    exprs = Vector{Expr}(undef, length(newsize))

    indices = CartesianIndices(newsize)
    Tnewsize = Tuple{newsize...}

    if v <: StaticArray
        newlen = StaticArrays.tuple_prod(newsize)
        if Length(v) != newlen
            return quote
                throw(DimensionMismatch("tried to assign $(length(v))-element array to $($newlen) destination"))
            end
        end
    end

    lininds = _get_linear_inds(sa, inds...)
    if lininds === nothing
        exprs = similar(indices, Expr)
        for current_ind ∈ indices
            cinds = _get_indices(current_ind.I, 1, inds...)
            exprs[current_ind.I...] = :(setindex!(sadata, v[$(current_ind.I...)], $(cinds...)))
        end

        if v <: StaticArray
            return quote
                Base.@_propagate_inbounds_meta
                sadata = sa.data
                @inbounds $(Expr(:block, exprs...))
            end
        else
            return quote
                Base.@_propagate_inbounds_meta
                if size(v) != $newsize
                    throw(DimensionMismatch("tried to assign array of size $(size(v)) to destination of size $($newsize)"))
                end
                sadata = sa.data
                @inbounds $(Expr(:block, exprs...))
            end
        end

    else
        exprs = [:(setindex!(sadata, v[$iid], $id)) for (iid, id) ∈ enumerate(lininds[2])]

        if v <: StaticArray
            return quote
                Base.@_propagate_inbounds_meta
                $(lininds[1])
                sadata = sa.data
                @inbounds $(Expr(:block, exprs...))
            end
        else
            return quote
                Base.@_propagate_inbounds_meta
                if size(v) != $newsize
                    throw(DimensionMismatch("tried to assign array of size $(size(v)) to destination of size $($newsize)"))
                end
                $(lininds[1])
                sadata = sa.data
                @inbounds $(Expr(:block, exprs...))
            end
        end
    end
end

@inline function _view_hybrid(a::HybridArray{S}, ::Val{:dynamic_fixed_true}, inner_view, indices...) where {S}
    new_size = new_out_size(a, indices...)
    return SizedArray{new_size}(inner_view)
end

@inline function _view_hybrid(a::HybridArray{S}, ::Val{:dynamic_fixed_false}, inner_view, indices...) where {S}
    new_size = new_out_size(a, indices...)
    return HybridArray{new_size}(inner_view)
end

@inline function Base.view(
    a::HybridArray{S},
    indices...,
) where {S}
    inner_view = invoke(view, Tuple{AbstractArray, typeof(indices).parameters...}, a, indices...)
    return _view_hybrid(a, all_dynamic_fixed_val(a, indices...), inner_view, indices...)
end
