using HybridArrays
using StaticArrays
using Test

@testset "Linear algebra" begin

    @testset "HybridMatrix as a (mathematical) vector space" begin
        c = 2
        a1 = [2 4; 6 8]
        s1 = SMatrix{2,2}(a1)
        m1 = HybridMatrix{2,StaticArrays.Dynamic()}(a1)
        a2 = [4 3; 2 1]
        s2 = SMatrix{2,2}(a2)
        m2 = HybridMatrix{2,StaticArrays.Dynamic()}(a2)

        @test @inferred(c * m1)::HybridMatrix == @SMatrix [4 8; 12 16]
        @test @inferred(m1 * c)::HybridMatrix == @SMatrix [4 8; 12 16]
        @test @inferred(m1 / c)::HybridMatrix == @SMatrix [1.0 2.0; 3.0 4.0]
        @test @inferred(c \ m1)::HybridMatrix ≈ @SMatrix [1.0 2.0; 3.0 4.0]

        @test @inferred(m1 + m2) == @SMatrix [6  7;  8 9]
        @test @inferred(m1 - m2) == @SMatrix [-2  1; 4 7]

        @test @inferred(a1 + m2) == @SMatrix [6  7;  8 9]
        @test @inferred(a1 - m2) == @SMatrix [-2  1; 4 7]
        @test @inferred(m1 + a2) == @SMatrix [6  7;  8 9]
        @test @inferred(m1 - a2) == @SMatrix [-2  1; 4 7]

        @test @inferred(s1 + m2) == @SMatrix [6  7;  8 9]
        @test @inferred(s1 - m2) == @SMatrix [-2  1; 4 7]
        @test @inferred(m1 + s2) == @SMatrix [6  7;  8 9]
        @test @inferred(m1 - s2) == @SMatrix [-2  1; 4 7]

        @test vcat(m1) == m1
        @test hcat(m1) == m1
        @test @inferred(vcat(m1, m2)) == vcat(a1, a2)
        @test @inferred(hcat(m1, m2)) == hcat(a1, a2)
        @test isa(vcat(m1, m2), HybridMatrix{4,StaticArrays.Dynamic()})
        @test isa(hcat(m1, m2), HybridMatrix{2,StaticArrays.Dynamic()})
    end
end
