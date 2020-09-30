using Test, Random, HybridArrays, StaticArrays


######### Tests #########
@testset "Statically sized SubArray" begin
    A = HybridArray{Tuple{2,2,StaticArrays.Dynamic()}}(fill(0.0, 2, 2, 10))
    Av = view(A, :, :, SOneTo(3))
    @test isa(Av, SizedArray{Tuple{2,2,3},Float64,3,3,SubArray{Float64,3,HybridArray{Tuple{2,2,StaticArrays.Dynamic()},Float64,3,3,Array{Float64,3}},Tuple{Base.Slice{SOneTo{2}},Base.Slice{SOneTo{2}},SOneTo{3}},true}})

    @test (@inferred Av[:,:,1]) === SA[0.0 0.0; 0.0 0.0]
    @test (@inferred Av[:,SOneTo(2),1]) === SA[0.0 0.0; 0.0 0.0]
end
