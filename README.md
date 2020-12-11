| Status | Coverage |
| :----: | :----: |
| [![CI](https://github.com/mateuszbaran/HybridArrays.jl/workflows/CI/badge.svg)](https://github.com/mateuszbaran/HybridArrays.jl/actions?query=workflow%3ACI+branch%3Amaster) | [ ![codecov.io](http://codecov.io/github/mateuszbaran/HybridArrays.jl/coverage.svg?branch=master)](http://codecov.io/github/mateuszbaran/HybridArrays.jl?branch=master) |

# HybridArrays.jl

Arrays with both statically and dynamically sized axes in Julia. This is a convenient replacement for the commonly used `Arrays`s of `SArray`s which are fast but not easy to mutate. `HybridArray` makes this easier: any `AbstractArray` can be wrapped in a structure that specifies which axes are statically sized. Based on this information code for `getindex`, `setindex!` and broadcasting is (or should soon be, not all cases have been optimized yet) as efficient as for `Arrays`s of `SArray`s while mutation of single elements is possible, as well as other operations on the wrapped array.

Views are statically sized where possible for fast and convenient mutation of `HybridArray`s.

Example:

```julia
julia> using HybridArrays, StaticArrays

julia> A = HybridArray{Tuple{2,2,StaticArrays.Dynamic()}}(randn(2,2,100));

julia> A[1,1,10] = 12
12

julia> A[:,:,10]
2×2 SArray{Tuple{2,2},Float64,2,4} with indices SOneTo(2)×SOneTo(2):
 12.0       -1.39943
 -0.450564  -0.140096

julia> A[2,:,:]
2×100 HybridArray{Tuple{2,StaticArrays.Dynamic()},Float64,2,2,Array{Float64,2}} with indices SOneTo(2)×Base.OneTo(100):
 -0.262977  1.40715  -0.110194    …  -1.67315    2.30679   0.931161
 -0.432229  3.04082  -0.00971933     -0.905037  -0.446818  0.777833

julia> A[:,:,10] .*= 2
2×2 SizedArray{Tuple{2,2},Float64,2,2,SubArray{Float64,2,HybridArray{Tuple{2,2,StaticArrays.Dynamic()},Float64,3,3,Array{Float64,3}},Tuple{Base.Slice{SOneTo{2}},Base.Slice{SOneTo{2}},Int64},true}} with indices SOneTo(2)×SOneTo(2):
 24.0       -2.79886
 -0.901128  -0.280193

julia> A[:,:,10] = SMatrix{2,2}(1:4)
2×2 SArray{Tuple{2,2},Int64,2,4} with indices SOneTo(2)×SOneTo(2):
 1  3
 2  4
```

`HybridArrays.jl` is implements (optionally loaded) [`ArrayInterface`](https://github.com/SciML/ArrayInterface.jl/) methods for compatibility with [`LoopVectorization`](https://github.com/chriselrod/LoopVectorization.jl).

Tips:

- If possible, statically known dimensions should come first. This way the most common access pattern where indices of dynamic dimensions are specified will be faster.
- Since version 0.4 of `HybridArrays`, Julia 1.5 or newer is required for best performance (most importantly the memory layout changes). It still works correctly on earlier versions of Julia but versions from the 0.3.x line may be faster in some cases on Julia <=1.4.

Code of this package is based on the code of the [`StaticArrays`](https://github.com/JuliaArrays/StaticArrays.jl).
