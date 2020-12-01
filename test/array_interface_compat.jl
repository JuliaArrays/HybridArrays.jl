
using HybridArrays, ArrayInterface, Test, StaticArrays
using ArrayInterface: StaticInt

@testset "ArrayInterface compatibility" begin
    M = HybridMatrix{2, StaticArrays.Dynamic()}([1 2; 4 5])
    MV = HybridMatrix{3, StaticArrays.Dynamic()}(view(randn(10, 10), 1:2:5, 1:3:12))
    @test ArrayInterface.ismutable(M)
    @test ArrayInterface.can_setindex(M)
    @test ArrayInterface.parent_type(M) === Matrix{Int}
    @test ArrayInterface.restructure(M, [2, 4, 6, 8]) == HybridMatrix{2, StaticArrays.Dynamic()}([2 6; 4 8])
    @test isa(ArrayInterface.restructure(M, [2, 4, 6, 8]), HybridMatrix{2, StaticArrays.Dynamic()})

    M2 = HybridArray{Tuple{2, 3, StaticArrays.Dynamic(), StaticArrays.Dynamic()}}(randn(2, 3, 5, 7))
    @test (@inferred ArrayInterface.strides(M2)) === (StaticInt(1), StaticInt(2), StaticInt(6), 30)
    @test (@inferred ArrayInterface.strides(MV)) === (2, 30)
    @test (@inferred ArrayInterface.size(M2)) === (StaticInt(2), StaticInt(3), 5, 7)

    @test ArrayInterface.contiguous_axis(typeof(M2)) === ArrayInterface.contiguous_axis(typeof(M2.data))
    @test ArrayInterface.contiguous_batch_size(typeof(M2)) === ArrayInterface.contiguous_batch_size(typeof(M2.data))
    @test ArrayInterface.stride_rank(typeof(M2)) === ArrayInterface.stride_rank(typeof(M2.data))
    @test ArrayInterface.contiguous_axis(typeof(M')) === ArrayInterface.contiguous_axis(typeof(M.data'))
    @test ArrayInterface.contiguous_batch_size(typeof(M')) === ArrayInterface.contiguous_batch_size(typeof(M.data'))
    @test ArrayInterface.stride_rank(typeof(M')) === ArrayInterface.stride_rank(typeof(M.data'))
end
