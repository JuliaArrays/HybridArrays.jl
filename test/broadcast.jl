using HybridArrays
using StaticArrays
using Test

struct ScalarTest end
Base.:(+)(x::Number, y::ScalarTest) = x
Broadcast.broadcastable(x::ScalarTest) = Ref(x)

@testset "broadcasting" begin
    Ai = rand(Int, 2, 3, 4)
    Bi = HybridArray{Tuple{2,3,StaticArrays.Dynamic()}, Int, 3}(Ai)

    Af = rand(Float64, 2, 3, 4)
    Bf = HybridArray{Tuple{2,3,StaticArrays.Dynamic()}, Float64, 3}(Af)

    Bi[1,2,:] .= [110, 111, 112, 113]
    @test Bi[1,2,:] == @SVector [110, 111, 112, 113]

    @testset "Scalar Broadcast" begin
        @test Bf == @inferred(Bf .+ ScalarTest())
        @test Bf .+ 1 == @inferred(Bf .+ Ref(1))
    end

    @testset "AbstractArray-of-HybridArray with scalar math" begin
        v = [Bf]
        @test @inferred(v .* 1.0)::typeof(v) == v
    end

    @testset "2x2 HybridMatrix with HybridVector" begin
        m = HybridMatrix{2,StaticArrays.Dynamic()}([1 2; 3 4])
        v = HybridVector{2}([1, 4])
        @test @inferred(broadcast(+, m, v)) == @SMatrix [2 3; 7 8]
        @test @inferred(m .+ v) == @SMatrix [2 3; 7 8]
        @test @inferred(v .+ m) == @SMatrix [2 3; 7 8]
        @test @inferred(m .* v) == @SMatrix [1 2; 12 16]
        @test @inferred(v .* m) == @SMatrix [1 2; 12 16]
        @test @inferred(m ./ v) == @SMatrix [1 2; 3/4 1]
        @test @inferred(v ./ m) == @SMatrix [1 1/2; 4/3 1]
        @test @inferred(m .- v) == @SMatrix [0 1; -1 0]
        @test @inferred(v .- m) == @SMatrix [0 -1; 1 0]
        @test @inferred(m .^ v) == @SMatrix [1 2; 81 256]
        @test @inferred(v .^ m) == @SMatrix [1 1; 64 256]
        # StaticArrays Issue #546
        @test @inferred(m ./ (v .* v')) == @SMatrix [1.0 0.5; 0.75 0.25]
        testinf(m, v) = m ./ (v .* v')
        @test @inferred(testinf(m, v)) == @SMatrix [1.0 0.5; 0.75 0.25]

        # mutating
        m .+= v .+ [1, 2]
        @test m == [3 4; 9 10]
        m = HybridMatrix{2,StaticArrays.Dynamic()}([1 2; 3 4])
        m .+= v .+ SA[1, 2]
        @test m == [3 4; 9 10]
    end

    @testset "2x2 HybridMatrix with 1x2 HybridMatrix" begin
        # StaticArrays Issues #197, #242: broadcast between SArray and row-like SMatrix
        m1 = HybridMatrix{2,StaticArrays.Dynamic()}([1 2; 3 4])
        m2 = HybridMatrix{1,StaticArrays.Dynamic()}([1 4])
        @test @inferred(broadcast(+, m1, m2)) == @SMatrix [2 6; 4 8]
        @test @inferred(m1 .+ m2) == @SMatrix [2 6; 4 8]
        @test @inferred(m2 .+ m1) == @SMatrix [2 6; 4 8]
        @test @inferred(m1 .* m2) == @SMatrix [1 8; 3 16]
        @test @inferred(m2 .* m1) == @SMatrix [1 8; 3 16]
        @test @inferred(m1 ./ m2) == @SMatrix [1 1/2; 3 1]
        @test @inferred(m2 ./ m1) == @SMatrix [1 2; 1/3 1]
        @test @inferred(m1 .- m2) == @SMatrix [0 -2; 2 0]
        @test @inferred(m2 .- m1) == @SMatrix [0 2; -2 0]
        @test @inferred(m1 .^ m2) == @SMatrix [1 16; 3 256]

        # mutating
        m1 .+= m2
        @test m1 == [2 6; 4 8]
    end

    @testset "1x2 HybridMatrix with SVector" begin
        # StaticArrays Issues #197, #242: broadcast between SVector and row-like SVector
        m = HybridMatrix{1,StaticArrays.Dynamic()}([1 2])
        v = SVector(1, 4)
        @test @inferred(broadcast(+, m, v)) == @SMatrix [2 3; 5 6]
        @test @inferred(m .+ v) == @SMatrix [2 3; 5 6]
        @test @inferred(v .+ m) == @SMatrix [2 3; 5 6]
        @test @inferred(m .* v) == @SMatrix [1 2; 4 8]
        @test @inferred(v .* m) == @SMatrix [1 2; 4 8]
        @test @inferred(m ./ v) == @SMatrix [1 2; 1/4 1/2]
        @test @inferred(v ./ m) == @SMatrix [1 1/2; 4 2]
        @test @inferred(m .- v) == @SMatrix [0 1; -3 -2]
        @test @inferred(v .- m) == @SMatrix [0 -1; 3 2]
        @test @inferred(m .^ v) == @SMatrix [1 2; 1 16]
        @test @inferred(v .^ m) == @SMatrix [1 1; 4 16]
    end

    @testset "HybridMatrix with HybridMatrix" begin
        m1 = HybridMatrix{2,StaticArrays.Dynamic()}([1 2; 3 4])
        m2 = HybridMatrix{2,StaticArrays.Dynamic()}([1 3; 4 5])
        @test @inferred(broadcast(+, m1, m2)) == @SMatrix [2 5; 7 9]
        @test @inferred(m1 .+ m2) == @SMatrix [2 5; 7 9]
        @test @inferred(m2 .+ m1) == @SMatrix [2 5; 7 9]
        @test @inferred(m1 .* m2) == @SMatrix [1 6; 12 20]
        @test @inferred(m2 .* m1) == @SMatrix [1 6; 12 20]
        # StaticArrays Issue #199: broadcast with empty SArray
        @test @inferred(HybridVector{1}([1]) .+ HybridVector{0,Int}([])) === SVector{0,Union{}}()
        @test_broken @inferred(HybridVector{0,Int}([]) .+ SVector(1)) === SVector{0,Union{}}()
        # StaticArrays Issue #200: broadcast with Adjoint
        @test @inferred(m1 .+ m2') == @SMatrix [2 6; 6 9]
        @test @inferred(m1 .+ transpose(m2)) == @SMatrix [2 6; 6 9]
        # StaticArrays Issue 382: infinite recursion in Base.Broadcast.broadcast_indices with Adjoint
        @test @inferred(HybridVector{2}([1,1])' .+ [1, 1]) == [2 2; 2 2]
        @test @inferred(transpose(HybridVector{2}([1,1])) .+ [1, 1]) == [2 2; 2 2]
        @test @inferred(HybridVector{StaticArrays.Dynamic()}([1,1])' .+ [1, 1]) == [2 2; 2 2]
        @test @inferred(transpose(HybridVector{StaticArrays.Dynamic()}([1,1])) .+ [1, 1]) == [2 2; 2 2]
    end

    @testset "HybridMatrix with Scalar" begin
        m = HybridMatrix{2,StaticArrays.Dynamic()}([1 2; 3 4])
        @test @inferred(broadcast(+, m, 2)) == @SMatrix [3 4; 5 6]
        @test @inferred(m .+ 2) == @SMatrix [3 4; 5 6]
        @test @inferred(2 .+ m) == @SMatrix [3 4; 5 6]
        @test @inferred(m .* 2) == @SMatrix [2 4; 6 8]
        @test @inferred(2 .* m) == @SMatrix [2 4; 6 8]
        @test @inferred(m ./ 2) == @SMatrix [1/2 1; 3/2 2]
        @test @inferred(2 ./ m) == @SMatrix [2 1; 2/3 1/2]
        @test @inferred(m .- 2) == @SMatrix [-1 0; 1 2]
        @test @inferred(2 .- m) == @SMatrix [1 0; -1 -2]
        @test @inferred(m .^ 2) == @SMatrix [1 4; 9 16]
        @test @inferred(2 .^ m) == @SMatrix [2 4; 8 16]

        # mutating
        m .+= 2
        @test m == [3 4; 5 6]
    end
    @testset "Empty arrays" begin
        @test @inferred(1.0 .+ HybridMatrix{2,0,Float64}(zeros(2,0))) == HybridMatrix{2,0,Float64}(zeros(2,0))
        @test @inferred(1.0 .+ HybridMatrix{0,2,Float64}(zeros(0,2))) == HybridMatrix{0,2,Float64}(zeros(0,2))
        @test @inferred(1.0 .+ HybridArray{Tuple{2,StaticArrays.Dynamic(),0},Float64}(zeros(2,3,0))) == HybridArray{Tuple{2,StaticArrays.Dynamic(),0},Float64}(zeros(2,3,0))
        @test @inferred(HybridVector{0,Float64}(zeros(0)) .+ HybridMatrix{0,2,Float64}(zeros(0,2))) == HybridMatrix{0,2,Float64}(zeros(0,2))
        m = HybridMatrix{0,2,Float64}(zeros(0,2))
        @test @inferred(broadcast!(+, m, m, HybridVector{0,Float64}(zeros(0)))) == HybridMatrix{0,2,Float64}(zeros(0,2))
    end

    @testset "Mutating broadcast!" begin
        # No setindex! error
        A = HybridMatrix{2,StaticArrays.Dynamic()}([1 0; 0 1])
        @test @inferred(broadcast!(+, A, A, SVector(1, 4))) == @MMatrix [2 1; 4 5]
        A = HybridMatrix{2,StaticArrays.Dynamic()}([1 0; 0 1])
        @test @inferred(broadcast!(+, A, A, @SMatrix([1  4]))) == @MMatrix [2 4; 1 5]
        A = HybridMatrix{1,StaticArrays.Dynamic()}([1 0])
        @test_throws DimensionMismatch broadcast!(+, A, A, SVector(1, 4))
        A = HybridMatrix{1,StaticArrays.Dynamic()}([1 0])
        @test @inferred(broadcast!(+, A, A, @SMatrix([1 4]))) == @MMatrix [2 4]
        A = HybridMatrix{1,StaticArrays.Dynamic()}([1 0])
        @test @inferred(broadcast!(+, A, A, 2)) == @MMatrix [3 2]
    end

    @testset "broadcast! with mixtures of SArray and Array" begin
        A = HybridVector{StaticArrays.Dynamic()}([0, 0])
        @test @inferred(broadcast!(+, A, [1,2])) == [1,2]
    end

    @testset "eltype after broadcast" begin
        # test cases StaticArrays issue #198
        let a = HybridVector{4, Number}(Number[2, 2.0, 4//2, 2+0im])
            @test eltype(a .+ 2) == Number
            @test eltype(a .- 2) == Number
            @test eltype(a * 2) == Number
            @test eltype(a / 2) == Number
        end
        let a = HybridVector{3, Real}(Real[2, 2.0, 4//2])
            @test eltype(a .+ 2) == Real
            @test eltype(a .- 2) == Real
            @test eltype(a * 2) == Real
            @test eltype(a / 2) == Real
        end
        let a = HybridVector{3, Real}(Real[2, 2.0, 4//2])
            @test eltype(a .+ 2.0) == Float64
            @test eltype(a .- 2.0) == Float64
            @test eltype(a * 2.0) == Float64
            @test eltype(a / 2.0) == Float64
        end
        let a = broadcast(Float32, HybridVector{3}([3, 4, 5]))
            @test eltype(a) == Float32
        end
    end

    @testset "broadcast general scalars" begin
        # StaticArrays Issue #239 - broadcast with non-numeric element types
        @eval @enum Axis aX aY aZ
        @test (HybridVector{3}([aX,aY,aZ]) .== Ref(aX)) == HybridVector{3}([true,false,false])
        mv = HybridVector{3}([aX,aY,aZ])
        @test broadcast!(identity, mv, Ref(aX)) == HybridVector{3}([aX,aX,aX])
        @test mv == HybridVector{3}([aX,aX,aX])
    end

    @testset "broadcast! with Array destination" begin
        # Issue #385
        a = zeros(3, 3)
        b = HybridMatrix{3,StaticArrays.Dynamic()}([1 2 3; 4 5 6; 7 8 9])
        a .= b
        @test a == b

        c = HybridVector{3}([1, 2, 3])
        a .= c
        @test a == [1 1 1; 2 2 2; 3 3 3]

        d = HybridVector{4}([1, 2, 3, 4])
        @test_throws DimensionMismatch a .= d
    end
end
