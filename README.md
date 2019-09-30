| Status | Coverage |
| :----: | :----: |
| [![Build Status](https://travis-ci.com/mateuszbaran/HybridArrays.jl.svg?branch=master)](https://travis-ci.com/mateuszbaran/HybridArrays.jl) [![Build status](https://ci.appveyor.com/api/projects/status/72nb8pp4pp6e2q7x?svg=true)](https://ci.appveyor.com/project/mateuszbaran/hybridarrays-jl) | [ ![codecov.io](http://codecov.io/github/mateuszbaran/HybridArrays.jl/coverage.svg?branch=master)](http://codecov.io/github/mateuszbaran/HybridArrays.jl?branch=master) |

# HybridArrays.jl

Arrays with both statically and dynamically sized axes in Julia. This is a convenient replacement for the commonly used `Arrays`s of `SArray`s which are fast but not easy to mutate. `HybridArray` makes this easier: any `AbstractArray` can be wrapped in a structure that specifies which axes are statically sized. Based on this information code for `getindex`, `setindex!` and broadcasting is (or should soon be, not all cases have been optimized yet) as efficient as for `Arrays`s of `SArray`s while mutation of single elements is possible, as well as other operations on the wrapped array.

There are also statically sized views for fast and convenient mutation of `HybridArray`s.

Example:
```julia
julia> using HybridArrays, StaticArrays

julia> A = HybridArray{Tuple{2,2,StaticArrays.Dynamic()}}(randn(2,2,100));

julia> A[1,1,10] = 12
12

julia> A[:,:,10]
2×2 SArray{Tuple{2,2},Float64,2,4} with indices SOneTo(2)×SOneTo(2):
 12.0       -0.264816
  0.615372  -1.00042

julia> A[2,:,:]
2×100 HybridArray{Tuple{2,StaticArrays.Dynamic()},Float64,2,2,Array{Float64,2}} with indices SOneTo(2)×Base.OneTo(100):
  1.26017  -0.401046  1.46593    1.01009   …  0.862791  -0.0928537  -1.60457
 -1.00588   0.581524  0.639293  -0.445845     2.0826    -1.40952     0.166665

julia> A[:,:,10] .*= 2
2×2 HybridArrays.SSubArray{Tuple{2,2},Float64,2,HybridArray{Tuple{2,2,StaticArrays.Dynamic()},Float64,3,3,Array{Float64,3}},Tuple{Base.Slice{SOneTo{2}},Base.Slice{SOneTo{2}},Int64},false} with indices SOneTo(2)×SOneTo(2):
 24.0      -0.529633
  1.23074  -2.00083

julia> A[:,:,10] = SMatrix{2,2}(1:4)
2×2 SArray{Tuple{2,2},Int64,2,4} with indices SOneTo(2)×SOneTo(2):
 1  3
 2  4
```

Tips:
  - If possible, statically known dimensions should come first. This way the most common access pattern where indices of dynamic dimensions are specified will be faster.

Code of this package is based on the code of the [`StaticArrays`](https://github.com/JuliaArrays/StaticArrays.jl) package and the `SubArray` type from Julia base.
