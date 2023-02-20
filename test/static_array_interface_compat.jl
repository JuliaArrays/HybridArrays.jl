
using HybridArrays, StaticArrayInterface, Test, StaticArrays
using StaticArrayInterface: StaticInt

@testset "StaticArrayInterface compatibility" begin
    M = HybridMatrix{2, StaticArrays.Dynamic()}([1 2; 4 5])
    MV = HybridMatrix{3, StaticArrays.Dynamic()}(view(randn(10, 10), 1:2:5, 1:3:12))

    M2 = HybridArray{Tuple{2, 3, StaticArrays.Dynamic(), StaticArrays.Dynamic()}}(randn(2, 3, 5, 7))
    @test (@inferred StaticArrayInterface.strides(M2)) === (StaticInt(1), StaticInt(2), StaticInt(6), 30)
    @test (@inferred StaticArrayInterface.strides(MV)) === (2, 30)

    @test StaticArrayInterface.contiguous_axis(typeof(M2)) === StaticArrayInterface.contiguous_axis(typeof(parent(M2)))
    @test StaticArrayInterface.contiguous_batch_size(typeof(M2)) === StaticArrayInterface.contiguous_batch_size(typeof(parent(M2)))
    @test StaticArrayInterface.stride_rank(typeof(M2)) === StaticArrayInterface.stride_rank(typeof(parent(M2)))
    @test StaticArrayInterface.contiguous_axis(typeof(M')) === StaticArrayInterface.contiguous_axis(typeof(parent(M)'))
    @test StaticArrayInterface.contiguous_batch_size(typeof(M')) === StaticArrayInterface.contiguous_batch_size(typeof(parent(M)'))
    @test StaticArrayInterface.stride_rank(typeof(M')) === StaticArrayInterface.stride_rank(typeof(parent(M)'))
end
