
using Test
using SystemDynamicsBuildR 

@testset "SystemDynamicsBuildR tests" begin
    include("test_sdbuildR_units.jl")
    include("test_custom_func.jl")
    include("test_clean.jl")
    include("test_ensemble.jl")
    include("test_unit_func.jl")
end;