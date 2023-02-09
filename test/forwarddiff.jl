using Test, HybridArrays, ForwardDiff, StaticArrays

@testset "ForwardDiff" begin
    x = [rand(SVector{3}) for _ in 1:4]
    xh = HybridMatrix{3,StaticArrays.Dynamic()}(reduce(hcat,x))
    f(x) = 1.0
    @test iszero(ForwardDiff.gradient(f, xh))
end