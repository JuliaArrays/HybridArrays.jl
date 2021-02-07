
module TestEllipsisNotationCompat # use a module to avoid cluttering the global namespace

using EllipsisNotation
using HybridArrays
using Test

#=
There is a method ambiguity
```julia
  MethodError: getindex(::HybridArray{Tuple{2,StaticArrays.Dynamic()},Float64,2,2,Array{Float64,2}}, ::EllipsisNotation.Ellipsis) is ambiguous. Candidates:
    getindex(A::AbstractArray, ::EllipsisNotation.Ellipsis) in EllipsisNotation at EllipsisNotation/XpBpL/src/EllipsisNotation.jl:58
    getindex(sa::HybridArray{S,T,N,M,TData} where TData<:AbstractArray{T,M} where M where N where T, inds...) where S in HybridArrays at HybridArrays/src/indexing.jl:14
  Possible fix, define
    getindex(::HybridArray{S,T,N,M,TData} where TData<:AbstractArray{T,M} where M where N where T, ::EllipsisNotation.Ellipsis) where S
```
since EllipsisNotation.jl defines
```julia
# avoid copying if indexing with .. alone, see
# https://github.com/JuliaDiffEq/OrdinaryDiffEq.jl/issues/214
Base.getindex(A::AbstractArray, ::Ellipsis) = A
```
This can be fixed by removing the current definition
```julia
# general fallback allowing, e.g. `..` from EllipsisNotation.jl
Base.@propagate_inbounds function getindex(sa::HybridArray{S}, inds...) where S
    getindex(sa, to_indices(sa, inds)...)
end
```
in HybridArrays/src/indexing.jl:14. If we do that, we don't catch something like
`u_hybrid[1, ..]` anymore, meaning that this will result in an `Array` instead of
a `HybridArray` (as it is for `u_hybrid[1, :]`).

I think the basic reason for this is that HybridArrays.jl interrupts directly at the
level of `getindex(::HybridArray, inds...)` in general, circumventing the mechanisms
of `to_indices` etc. from `Base`. I somehow get the impression that this can't be
the best approach since we run into this issue. Something like writing a method
```julia
_unsafe_getindex(::IndexLinear{…},::HybridArray{…},::Int64{…},::Base.Slice{…})
```
does not seem to be the right approach either. Since I'm not an expert on HybbridArrays.jl,
I don't really know how to proceed further from here. I could try something but I thought
it might be better to share my experience so far and let also others have a look.
I'm definitely willing to invest some more work into this, but it will probably be best
if others have a look, too.
=#

@testset "Compatibility with EllipsisNotation" begin
  @test_skip length(Test.detect_ambiguities(HybridArrays)) == 0

  u_array  = rand(2, 10)
  u_hybrid = HybridArray{Tuple{2, HybridArrays.Dynamic()}}(copy(u_array))
  @test u_hybrid        ≈ u_array
  @test u_hybrid[1, ..] ≈ u_array[1, ..]
  @test u_hybrid[.., 1] ≈ u_array[.., 1]
  @test_skip u_hybrid[..]    ≈ u_array[..]

  @test typeof(u_hybrid[1, ..]) == typeof(u_hybrid[1, :])
  @test typeof(u_hybrid[.., 1]) == typeof(u_hybrid[:, 1])
  @test_skip typeof(u_hybrid[..])    == typeof(u_hybrid[:, :])

  @inferred u_hybrid[1, ..]
  @inferred u_hybrid[.., 1]
  @test_skip @inferred u_hybrid[..]

  new_values = rand(10)
  u_array[1, ..]  .= new_values
  u_hybrid[1, ..] .= new_values
  @test u_hybrid        ≈ u_array
  @test u_hybrid[1, ..] ≈ u_array[1, ..]
  @test u_hybrid[.., 1] ≈ u_array[.., 1]
  @test_skip u_hybrid[..]    ≈ u_array[..]

  new_values = rand(2)
  u_array[.., 1]  .= new_values
  u_hybrid[.., 1] .= new_values
  @test u_hybrid        ≈ u_array
  @test u_hybrid[1, ..] ≈ u_array[1, ..]
  @test u_hybrid[.., 1] ≈ u_array[.., 1]
  @test_skip u_hybrid[..]    ≈ u_array[..]

  # these are also skipped...
  # new_values = rand(2, 10)
  # u_array[..]  .= new_values
  # u_hybrid[..] .= new_values
  # @test u_hybrid        ≈ u_array
  # @test u_hybrid[1, ..] ≈ u_array[1, ..]
  # @test u_hybrid[.., 1] ≈ u_array[.., 1]
  # @test_skip u_hybrid[..]    ≈ u_array[..]

  new_values = rand(2, 10)
  u_array  .= new_values
  u_hybrid .= new_values
  @test u_hybrid        ≈ u_array
  @test u_hybrid[1, ..] ≈ u_array[1, ..]
  @test u_hybrid[.., 1] ≈ u_array[.., 1]
  @test_skip u_hybrid[..]    ≈ u_array[..]
end

end # module
