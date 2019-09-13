using StaticArrays, HybridArrays, Test, LinearAlgebra

@testset "AbstractArray interface" begin
    @testset "size and length" begin
        M = HybridMatrix{2, StaticArrays.Dynamic()}([1 2 3; 4 5 6])

        @test length(M) == 6
        @test size(M) == (2, 3)
        @test Base.isassigned(M, 2, 2) == true
    end

    @testset "reshape" begin
        # TODO?
    end

    @testset "copy" begin
        M = HybridMatrix{2, StaticArrays.Dynamic()}([1 2; 3 4])
        @test @inferred(copy(M))::HybridMatrix == M
        @test copy(M).data !== M.data
    end

end
