using Test, Random, HybridArrays, StaticArrays


######### Tests #########
@testset "Statically sized SubArray" begin
    A = HybridArray{Tuple{2,2,StaticArrays.Dynamic()}}(fill(0.0, 2, 2, 10))
    Av = view(A, :, :, SOneTo(3))
    @test isa(Av, SizedArray{Tuple{2,2,3},Float64,3,3,SubArray{Float64,3,HybridArray{Tuple{2,2,StaticArrays.Dynamic()},Float64,3,3,Array{Float64,3}},Tuple{Base.Slice{SOneTo{2}},Base.Slice{SOneTo{2}},SOneTo{3}},true}})

    @test (@inferred Av[:,:,1]) === SA[0.0 0.0; 0.0 0.0]
    @test (@inferred Av[:,SOneTo(2),1]) === SA[0.0 0.0; 0.0 0.0]
    @test StaticArrays.similar_type(Av) === SArray{Tuple{2,2,3},Float64,3,12}

    # views that leave dynamically sized axes should return HybridArray (see issue #31)
    B = HybridArray{Tuple{2,5,StaticArrays.Dynamic()}}(randn(2, 5, 3))
    @test isa((@inferred view(B, :, StaticArrays.SUnitRange(2, 4), :)), HybridArray{Tuple{2,3,StaticArrays.Dynamic()},Float64,3,3,<:SubArray})
    Bv = view(B, :, StaticArrays.SUnitRange(2, 4), :)
    @test Bv == B[:, StaticArrays.SUnitRange(2, 4), :]
end
