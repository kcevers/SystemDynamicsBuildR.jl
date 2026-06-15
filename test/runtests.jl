
using Test
using SystemDynamicsBuildR 

@testset "SystemDynamicsBuildR tests" begin
    include("test_custom_func.jl")
    include("test_with_rng.jl")
    include("test_clean.jl")
    include("test_ensemble.jl")
    include("test_dependencies.jl")
end;