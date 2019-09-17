@testset "Linear algebra" begin

    @testset "HybridMatrix as a (mathematical) vector space" begin
        c = 2
        m1 = HybridMatrix{2,StaticArrays.Dynamic()}([2 4; 6 8])
        m2 = HybridMatrix{2,StaticArrays.Dynamic()}([4 3; 2 1])

        @test @inferred(m1 * c) == @SMatrix [4 8; 12 16]
        @test @inferred(m1 / c) == @SMatrix [1.0 2.0; 3.0 4.0]
        @test @inferred(c \ m1)::HybridMatrix ≈ @SMatrix [1.0 2.0; 3.0 4.0]

        @test @inferred(m1 + m2) == @SMatrix [6  7;  8 9]
        @test @inferred(m1 - m2) == @SMatrix [-2  1; 4 7]
    end
end
