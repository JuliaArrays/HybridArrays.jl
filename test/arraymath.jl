using StaticArrays, HybridArrays, Test, LinearAlgebra

@testset "AbstractArray interface" begin
    @testset "one()" begin
        M = HybridMatrix{2, StaticArrays.Dynamic()}([1 2; 4 5])
        @test (@inferred one(M)) == @SMatrix [1 0; 0 1]
        @test isa(one(M), HybridMatrix{2,StaticArrays.Dynamic(),Int})

        Mv = view(M, :, SOneTo(2))
        @test (@inferred one(Mv)) === @SMatrix [1 0; 0 1]
    end

    @testset "zero()" begin
        M = HybridMatrix{2, StaticArrays.Dynamic()}([1 2; 4 5])
        @test (@inferred zero(M)) == @SMatrix [0 0; 0 0]
        @test isa(one(M), HybridMatrix{2,StaticArrays.Dynamic(),Int})

        Mv = view(M, :, SOneTo(2))
        @test (@inferred zero(Mv)) === @SMatrix [0 0; 0 0]
    end

    @testset "fill!()" begin
        M = HybridMatrix{2, StaticArrays.Dynamic(), Float64}([1 2; 4 5])
        fill!(M, 3)
        @test all(M .== 3.)
    end

end
