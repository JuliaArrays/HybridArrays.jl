using HybridArrays
using StaticArrays
using Test

@testset "Inner Constructors" begin
    @test HybridArray{Tuple{2}, Int, 1, 1, Vector{Int}}((3, 4)).data == [3, 4]
    @test HybridArray{Tuple{2}, Int, 1}([3, 4]).data == [3, 4]
    @test HybridArray{Tuple{2, 2}, Int, 2}(collect(3:6)).data == collect(3:6)
    @test size(HybridArray{Tuple{4, 5}, Int, 2}(undef).data) == (4, 5)
    @test size(HybridArray{Tuple{4, 5}, Int}(undef).data) == (4, 5)

    # Bad input
    @test_throws Exception SArray{Tuple{1},Int,1}([2 3])

    # Bad parameters
    @test_throws Exception HybridArray{Tuple{1},Int,2}(undef)
    @test_throws Exception SArray{Tuple{3, 4},Int,1}(undef)

    # Parameter/input size mismatch
    @test_throws Exception HybridArray{Tuple{1},Int,2}([2; 3])
    @test_throws Exception HybridArray{Tuple{1},Int,2}((2, 3))
end

@testset "Outer Constructors" begin
    # From Array
    @test @inferred(HybridArray{Tuple{2},Float64,1}([1,2]))::HybridArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
    @test @inferred(HybridArray{Tuple{2},Float64}([1,2]))::HybridArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
    @test @inferred(HybridArray{Tuple{2}}([1,2]))::HybridArray{Tuple{2},Int,1,1} == [1,2]
    @test @inferred(HybridArray{Tuple{2,2}}([1 2;3 4]))::HybridArray{Tuple{2,2},Int,2,2} == [1 2; 3 4]

    # Uninitialized
    @test @inferred(HybridArray{Tuple{2,2},Int,2}(undef)) isa HybridArray{Tuple{2,2},Int,2,2}
    @test @inferred(HybridArray{Tuple{2,2},Int}(undef)) isa HybridArray{Tuple{2,2},Int,2,2}

    # From Tuple
    @test @inferred(HybridArray{Tuple{2},Float64,1,1,Vector{Float64}}((1,2)))::HybridArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
    @test @inferred(HybridArray{Tuple{2},Float64}((1,2)))::HybridArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
    @test @inferred(HybridArray{Tuple{2}}((1,2)))::HybridArray{Tuple{2},Int,1,1} == [1,2]
    @test @inferred(HybridArray{Tuple{2,2}}((1,2,3,4)))::HybridArray{Tuple{2,2},Int,2,2} == [1 3; 2 4]
end

@testset "HybridVector and HybridMatrix" begin
    @test @inferred(HybridVector{2}([1,2]))::HybridArray{Tuple{2},Int,1,1} == [1,2]
    @test @inferred(HybridVector{2}((1,2)))::HybridArray{Tuple{2},Int,1,1} == [1,2]
    # Reshaping
    @test @inferred(HybridVector{2}([1 2]))::HybridArray{Tuple{2},Int,1,2} == [1,2]
    # Back to Vector
    @test Vector(HybridVector{2}((1,2))) == [1,2]
    @test convert(Vector, HybridVector{2}((1,2))) == [1,2]

    @test @inferred(HybridMatrix{2,2}([1 2; 3 4]))::HybridArray{Tuple{2,2},Int,2,2} == [1 2; 3 4]
    # Reshaping
    @test @inferred(HybridMatrix{2,2}((1,2,3,4)))::HybridArray{Tuple{2,2},Int,2,2} == [1 3; 2 4]
    # Back to Matrix
    @test Matrix(HybridMatrix{2,2}([1 2;3 4])) == [1 2; 3 4]
    @test convert(Matrix, HybridMatrix{2,2}([1 2;3 4])) == [1 2; 3 4]
end

@testset "setindex" begin
    sa = HybridArray{Tuple{2}, Int, 1}([3, 4])
    sa[1] = 2
    @test sa.data == [2, 4]
end

@testset "aliasing" begin
    a1 = rand(4)
    a2 = copy(a1)
    sa1 = HybridVector{4}(a1)
    sa2 = HybridVector{4}(a2)
    @test Base.mightalias(a1, sa1)
    @test Base.mightalias(sa1, HybridVector{4}(a1))
    @test !Base.mightalias(a2, sa1)
    @test !Base.mightalias(sa1, HybridVector{4}(a2))
    @test Base.mightalias(sa1, view(sa1, 1:2))
    @test Base.mightalias(a1, view(sa1, 1:2))
    @test Base.mightalias(sa1, view(a1, 1:2))
end

@testset "back to Array" begin
    @test Array(HybridArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
    @test Array{Int}(HybridArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
    @test Array{Int, 1}(HybridArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
    @test Vector(HybridArray{Tuple{4}, Int, 1}(collect(3:6))) == collect(3:6)
    @test convert(Vector, HybridArray{Tuple{4}, Int, 1}(collect(3:6))) == collect(3:6)
    @test Matrix(SMatrix{2,2}((1,2,3,4))) == [1 3; 2 4]
    @test convert(Matrix, SMatrix{2,2}((1,2,3,4))) == [1 3; 2 4]
    @test convert(Array, HybridArray{Tuple{2,2,2,2}, Int}(ones(2,2,2,2))) == ones(2,2,2,2)
    # Conversion after reshaping
    @test_broken Array(HybridMatrix{2,2,Int,1,Vector{Int}}([1,2,3,4])) == [1 3; 2 4]
end

@testset "promotion" begin
    @test @inferred(promote_type(HybridVector{1,Float64,1,Vector{Float64}}, HybridVector{1,BigFloat,1,Vector{BigFloat}})) == HybridVector{1,BigFloat,1,Vector{BigFloat}}
    @test @inferred(promote_type(HybridVector{2,Int,1,Vector{Int}}, HybridVector{2,Float64,1,Vector{Float64}})) === HybridVector{2,Float64,1,Vector{Float64}}
    @test @inferred(promote_type(HybridMatrix{2,3,Float32,2,Matrix{Float32}}, HybridMatrix{2,3,Complex{Float64},2,Matrix{Complex{Float64}}})) === HybridMatrix{2,3,Complex{Float64},2,Matrix{Complex{Float64}}}
end

@testset "dynamically sized axes" begin
    A = rand(Int, 2, 3, 4)
    B = HybridArray{Tuple{2,3,StaticArrays.Dynamic()}, Int, 3}(A)
    C = rand(Int, 2, 3, 4, 5)
    D = HybridArray{Tuple{2,3,StaticArrays.Dynamic(),StaticArrays.Dynamic()}, Int, 4}(C)
    @test size(B) == size(A)
    @test size(D) == size(C)
    @test axes(B) == (SOneTo(2), SOneTo(3), axes(A, 3))
    @test axes(B, 1) == SOneTo(2)
    @test axes(B, 2) == SOneTo(3)

    @test B[1,2,3] == A[1,2,3]
    @test B[1,:,:] == A[1,:,:]
    inds = @SVector [2, 1]
    @test B[1,inds,:] == A[1,inds,:]
    @test B[:,:,2] == A[:,:,2]
    @test B[:,:,@SVector [2, 3]] == A[:,:,[2, 3]]

    B[1,2,3] = 42
    @test B[1,2,3] == 42
    B[:,2,3] = @SVector [10, 11]
    @test B[:,2,3] == @SVector [10, 11]
    B[:,:,1] = @SMatrix [1 2 3; 4 5 6]
    @test B[:,:,1] == @SMatrix [1 2 3; 4 5 6]
    B[1,2,:] = [10, 11, 12, 13]
    @test B[1,2,:] == @SVector [10, 11, 12, 13]
    B[:,2,:] = @SMatrix [1 2 3 4; 5 6 7 8]
    @test B[:,2,:] == @SMatrix [1 2 3 4; 5 6 7 8]
    B[:,2,:] = [11 12 13 14; 15 16 17 18]
    @test B[:,2,:] == [11 12 13 14; 15 16 17 18]

    D[:,2,3,4] = @SVector [10, 11]
    @test D[:,2,3,4] == @SVector [10, 11]
    D[:,:,1,2] = @SMatrix [1 2 3; 4 5 6]
    @test D[:,:,1,2] == @SMatrix [1 2 3; 4 5 6]
    D[1,2,:,1] = [10, 11, 12, 13]
    @test D[1,2,:,1] == @SVector [10, 11, 12, 13]

    @test_throws DimensionMismatch (D[:,2,3,4] = @SVector [10, 11, 11])
    @test_throws DimensionMismatch (D[:,2,3,4] = [10, 11, 11])
    @test_throws DimensionMismatch (B[1,2,:] = [10, 11])
end

include("abstractarray.jl")
include("arraymath.jl")
include("broadcast.jl")
include("linalg.jl")
include("ssubarray.jl")
