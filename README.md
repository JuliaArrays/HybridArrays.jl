# HybridArrays.jl

Arrays with both statically and dynamically sized axes in Julia. This is a convenient replacement for the commonly used `Arrays`s of `SArray`s which are fast but not easy to mutate. `HybridArray` makes this easier: any `AbstractArray` can be wrapped in a structure that specifies which axes are statically sized. Based on this information code for `getindex`, `setindex!` and broadcasting is (or should soon be, not all cases have been optimized yet) as efficient as for `Arrays`s of `SArray`s while mutation of single elements is possible, as well as other operations on the wrapped array.

In preparation are statically sized views for fast and convenient mutation of `HybridArray`s.

Example:
```julia
julia> using HybridArrays

julia> A = HybridArray{Tuple{2,2,StaticArrays.Dynamic()}}(randn(2,2,100));

julia> A[1,1,10] = 12
12

julia> A[:,:,10]
2×2 SArray{Tuple{2,2},Float64,2,4} with indices SOneTo(2)×SOneTo(2):
 12.0       -1.59817
  0.805929  -1.36063

julia> A[2,:,:]
2×100 HybridArray{Tuple{2,StaticArrays.Dynamic()},Float64,2,2,Array{Float64,2}} with indices SOneTo(2)×Base.OneTo(100):
 -1.25316   -0.848573  -0.875348   0.31877   0.315297  0.211012  …   0.99563   -0.137054  -0.0793494  -0.345719   1.05423   -0.185278
  0.262436   1.25696   -0.533936  -0.49609  -0.544297  0.725697     -0.772447   0.407075   0.604891   -2.30221   -0.830824  -0.204973
```

Tips:
  - If possible, statically known dimensions should come first. This way the most common access pattern where indices of dynamic dimensions are specified will be faster.

Code of this package is based on the code of the [`StaticArrays`](https://github.com/JuliaArrays/StaticArrays.jl) package.
