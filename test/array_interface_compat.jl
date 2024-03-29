
using HybridArrays, ArrayInterface, Test, StaticArrays

@testset "ArrayInterface compatibility" begin
    M = HybridMatrix{2, StaticArrays.Dynamic()}([1 2; 4 5])
    MV = HybridMatrix{3, StaticArrays.Dynamic()}(view(randn(10, 10), 1:2:5, 1:3:12))
    @test ArrayInterface.ismutable(M)
    @test ArrayInterface.can_setindex(M)
    @test ArrayInterface.parent_type(M) === Matrix{Int}
    @test ArrayInterface.restructure(M, [2, 4, 6, 8]) == HybridMatrix{2, StaticArrays.Dynamic()}([2 6; 4 8])
    @test isa(ArrayInterface.restructure(M, [2, 4, 6, 8]), HybridMatrix{2, StaticArrays.Dynamic()})
end
