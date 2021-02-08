using StaticArrays, HybridArrays, Test, LinearAlgebra

@testset "AbstractArray interface" begin
    @testset "size and length" begin
        M = HybridMatrix{2, StaticArrays.Dynamic()}([1 2 3; 4 5 6])

        @test length(M) == 6
        @test size(M) == (2, 3)
        @test Base.isassigned(M, 2, 2) == true
        @test (@inferred Size(M)) == Size(Tuple{2, StaticArrays.Dynamic()})
        @test (@inferred Size(typeof(M))) == Size(Tuple{2, StaticArrays.Dynamic()})
    end

    @testset "reshape" begin
        # TODO?
    end

    @testset "convert" begin
        MM = [1 2 3; 4 5 6]
        M = HybridMatrix{2, StaticArrays.Dynamic()}(MM)
        @test parent(@inferred convert(HybridArray{Tuple{2,StaticArrays.Dynamic()}}, MM)) == M
        @test parent(@inferred convert(HybridArray{Tuple{2,StaticArrays.Dynamic()},Int}, MM)) == M
        @test parent(@inferred convert(HybridArray{Tuple{2,StaticArrays.Dynamic()},Float64}, MM)) == M
        @test parent(@inferred convert(HybridArray{Tuple{2,StaticArrays.Dynamic()},Float64,2}, MM)) == M
        @test parent(@inferred convert(HybridArray{Tuple{2,StaticArrays.Dynamic()},Float64,2,2}, MM)) == M
        @test parent(@inferred convert(HybridArray{Tuple{2,StaticArrays.Dynamic()},Float64,2,2,Matrix{Float64}}, MM)) == M
        @test convert(typeof(M), M) === M
        if VERSION >= v"1.1"
            @test convert(HybridArray{Tuple{2,StaticArrays.Dynamic()},Float64}, M) == M
        end
        @test convert(Array, M) === parent(M)
        @test convert(Array{Int}, M) === parent(M)
        @test convert(Matrix, M) === parent(M)
        @test convert(Matrix{Int}, M) === parent(M)

        @test Array(M) == M
        @test Array(M) !== parent(M)
        @test Matrix(M) == M
        @test Matrix(M) !== parent(M)
        @test Matrix{Int}(M) == M
        @test Matrix{Int}(M) !== parent(M)
        @test_throws MethodError Vector(M)
    end

    @testset "copy" begin
        M = HybridMatrix{2, StaticArrays.Dynamic()}([1 2; 3 4])
        @test @inferred(copy(M))::HybridMatrix == M
        @test parent(copy(M)) !== parent(M)
    end

    @testset "similar" begin
        M = HybridMatrix{2, StaticArrays.Dynamic(), Int}([1 2; 3 4])

        @test isa(@inferred(similar(M)), HybridMatrix{2, StaticArrays.Dynamic(), Int})
        @test isa(@inferred(similar(M, Float64)), HybridMatrix{2, StaticArrays.Dynamic(), Float64})
        @test isa(@inferred(similar(M, Float64, Size(Tuple{3, 3}))), HybridMatrix{3, 3, Float64})
    end

    @testset "IndexStyle" begin
        M = HybridMatrix{2, StaticArrays.Dynamic(), Int}([1 2; 3 4])
        MT = HybridMatrix{2, StaticArrays.Dynamic(), Int}([1 2; 3 4]')
        @test (@inferred IndexStyle(M)) === IndexLinear()
        @test (@inferred IndexStyle(MT)) === IndexCartesian()
        @test (@inferred IndexStyle(typeof(M))) === IndexLinear()
        @test (@inferred IndexStyle(typeof(MT))) === IndexCartesian()
    end

    @testset "vec" begin
        M = HybridMatrix{2, StaticArrays.Dynamic(), Int}([1 2; 3 4])
        Mv = vec(M)
        @test Mv == [1, 3, 2, 4]
        @test Mv isa Vector{Int}
        Mv[2] = 100
        @test M[2, 1] == 100
    end

    @testset "errors" begin
        @test_throws TypeError HybridArrays.new_out_size_nongen(Size{Tuple{1,2}}, 'a')
    end

    @testset "elsize, eltype" begin
        for T in [Int8, Int16, Int32, Int64, Int128, Float16, Float32, Float64, BigFloat]
            M_array  = rand(T, 2, 2)
            M_hybrid = HybridMatrix{2, StaticArrays.Dynamic()}(M_array)
            @test Base.eltype(M_hybrid) === Base.eltype(M_array)
            @test Base.elsize(M_hybrid) === Base.elsize(M_array)
        end
    end

    @testset "strides" begin
        M = HybridMatrix{2, StaticArrays.Dynamic(), Int}([1 2; 3 4])

        @test strides(M) == strides(parent(M))
    end

    @testset "pointer" begin
        M = HybridMatrix{2, StaticArrays.Dynamic(), Int}([1 2; 3 4])
        MT = HybridMatrix{2, StaticArrays.Dynamic(), Int}([1 2; 3 4]')

        @test pointer(M) == pointer(parent(M))
        if VERSION >= v"1.5"
            # pointer on Adjoint is not available on earilier versions of Julia
            @test pointer(MT) == pointer(parent(MT))
        end
    end

    @testset "unsafe_convert" begin
        M = HybridMatrix{2, StaticArrays.Dynamic(), Int}([1 2; 3 4])
        @test Base.unsafe_convert(Ptr{Int}, M) === pointer(parent(M))
    end

    @test HybridArrays._totally_linear() === true
end
