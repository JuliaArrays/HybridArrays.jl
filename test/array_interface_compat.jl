
using HybridArrays, ArrayInterface, Test, StaticArrays

@testset "ArrayInterface compatibility" begin
    M = HybridMatrix{2, StaticArrays.Dynamic()}([1 2; 4 5])
    @test ArrayInterface.ismutable(M)
    @test ArrayInterface.can_setindex(M)
    @test ArrayInterface.parent_type(M) === Matrix{Int}
    @test ArrayInterface.restructure(M, [2, 4, 6, 8]) == HybridMatrix{2, StaticArrays.Dynamic()}([2 6; 4 8])
end
