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

    @testset "convertsions" begin
        M = HybridMatrix{2, StaticArrays.Dynamic()}([1 2 3; 4 5 6])
        @test convert(Array, M) === M.data
        @test_broken convert(Matrix, M) === M.data
        @test convert(Matrix{Int}, M) === M.data

        @test Array(M) == M
        @test Array(M) !== M.data
        @test Matrix(M) == M
        @test Matrix(M) !== M.data
        @test Matrix{Int}(M) == M
        @test Matrix{Int}(M) !== M.data
        @test_throws MethodError Vector(M)
    end

    @testset "copy" begin
        M = HybridMatrix{2, StaticArrays.Dynamic()}([1 2; 3 4])
        @test @inferred(copy(M))::HybridMatrix == M
        @test copy(M).data !== M.data
    end

    @testset "similar" begin
        M = HybridMatrix{2, StaticArrays.Dynamic(), Int}([1 2; 3 4])

        @test isa(@inferred(similar(M)), HybridMatrix{2, StaticArrays.Dynamic(), Int})
        @test isa(@inferred(similar(M, Float64)), HybridMatrix{2, StaticArrays.Dynamic(), Float64})
    end

end
