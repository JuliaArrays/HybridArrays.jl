
module TestEllipsisNotationCompat # use a module to avoid cluttering the global namespace

using EllipsisNotation
using HybridArrays
using Test


@testset "Compatibility with EllipsisNotation" begin
  u_array  = rand(2, 10)
  u_hybrid = HybridArray{Tuple{2, HybridArrays.Dynamic()}}(copy(u_array))
  @test u_hybrid        ≈ u_array
  @test u_hybrid[1, ..] ≈ u_array[1, ..]
  @test u_hybrid[.., 1] ≈ u_array[.., 1]
  @test u_hybrid[..]    ≈ u_array[..]

  @test_broken typeof(u_hybrid[1, ..]) == typeof(u_hybrid[1, :])
  @test_broken typeof(u_hybrid[.., 1]) == typeof(u_hybrid[:, 1])
  @test_broken typeof(u_hybrid[..])    == typeof(u_hybrid[:])

  @inferred u_hybrid[1, ..]
  @inferred u_hybrid[.., 1]
  @inferred u_hybrid[..]

  let new_values = rand(10)
    u_array[1, ..]  .= new_values
    u_hybrid[1, ..] .= new_values
    @test u_hybrid        ≈ u_array
    @test u_hybrid[1, ..] ≈ u_array[1, ..]
    @test u_hybrid[.., 1] ≈ u_array[.., 1]
    @test u_hybrid[..]    ≈ u_array[..]
  end

  let new_values = rand(2)
    u_array[.., 1]  .= new_values
    u_hybrid[.., 1] .= new_values
    @test u_hybrid        ≈ u_array
    @test u_hybrid[1, ..] ≈ u_array[1, ..]
    @test u_hybrid[.., 1] ≈ u_array[.., 1]
    @test u_hybrid[..]    ≈ u_array[..]
  end

  let new_values = rand(2, 10)
    u_array  .= new_values
    u_hybrid .= new_values
    @test u_hybrid        ≈ u_array
    @test u_hybrid[1, ..] ≈ u_array[1, ..]
    @test u_hybrid[.., 1] ≈ u_array[.., 1]
    @test u_hybrid[..]    ≈ u_array[..]
  end
end


@testset "Compatibility with Cartesian indices" begin
  u_array  = rand(2, 3, 4)
  u_hybrid = HybridArray{Tuple{2, 3, 4}}(copy(u_array))
  @test u_hybrid        ≈ u_array
  @test u_hybrid[CartesianIndex(1, 2), :] ≈ u_array[CartesianIndex(1, 2), :]
  @test u_hybrid[:, CartesianIndex(1, 2)] ≈ u_array[:, CartesianIndex(1, 2)]

  @test_broken typeof(u_hybrid[CartesianIndex(1, 2), :]) == typeof(u_hybrid[1, 2, :])
  @test_broken typeof(u_hybrid[:, CartesianIndex(1, 2)]) == typeof(u_hybrid[:, 1, 2])

  @inferred u_hybrid[CartesianIndex(1, 2), :]
  @inferred u_hybrid[:, CartesianIndex(1, 2)]
  @inferred u_hybrid[CartesianIndex(1, 2, 3)]

  let new_values = rand(4)
    u_array[CartesianIndex(1, 2), :]  .= new_values
    u_hybrid[CartesianIndex(1, 2), :] .= new_values
    @test u_hybrid        ≈ u_array
    @test u_hybrid[CartesianIndex(1, 2), :] ≈ u_array[CartesianIndex(1, 2), :]
    @test u_hybrid[:, CartesianIndex(1, 2)] ≈ u_array[:, CartesianIndex(1, 2)]
  end

  let new_values = rand(2)
    u_array[:, CartesianIndex(1, 2)]  .= new_values
    u_hybrid[:, CartesianIndex(1, 2)] .= new_values
    @test u_hybrid        ≈ u_array
    @test u_hybrid[CartesianIndex(1, 2), :] ≈ u_array[CartesianIndex(1, 2), :]
    @test u_hybrid[:, CartesianIndex(1, 2)] ≈ u_array[:, CartesianIndex(1, 2)]
  end
end


end # module
