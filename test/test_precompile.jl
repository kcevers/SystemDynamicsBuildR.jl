# Tests for the PrecompileTools workload in src/precompile.jl.
#
# @compile_workload only runs while the package is being precompiled, so its body is
# unreachable from here. The body therefore lives in SystemDynamicsBuildR._pc_workload(),
# which these tests call directly.
#
# This is not only about coverage. A workload that has drifted out of step with the
# current SciML API does not fail loudly - it just stops precompiling the paths it was
# meant to cover, and the per-session JIT cost quietly comes back. Calling it here turns
# that into a test failure.

using Test
using DataFrames
using SciMLBase
using SystemDynamicsBuildR

# Every solver sdbuildR's sim_methods() can emit. Precompilation is per-signature, so a
# solver missing from the workload pays its full JIT cost on first use. Keep in step
# with solver_dict in sdbuildR's R/sim_methods.R.
const EXPECTED_SOLVERS = (:Euler, :Midpoint, :Heun, :RK4, :BS3, :Tsit5,
                          :Vern6, :Vern7, :Vern8, :Vern9, :Rosenbrock23)

@testset "precompile workload" begin
    out = SystemDynamicsBuildR._pc_workload()

    @testset "covers every solver sim_methods() can emit" begin
        @test keys(out.solves) === EXPECTED_SOLVERS

        for name in EXPECTED_SOLVERS
            prob, sol = out.solves[name]
            @test SciMLBase.successful_retcode(sol)
            # Fixed saveat, so every solver returns the same three time points.
            @test sol.t == [0.0, 0.5, 1.0]
            @test length(sol.u[1]) == 2
            @test all(isfinite, reduce(vcat, sol.u))
        end
    end

    @testset "single-run cleanup and CSV" begin
        @test out.df isa DataFrame
        @test names(out.df) == ["time", "variable", "value"]
        @test Set(out.df.variable) == Set(["prey", "predator"])
        @test nrow(out.df) == 6                      # 3 time points x 2 stocks

        @test out.param_names == ["alpha", "beta", "delta", "gamma"]
        @test out.param_values == [1.1, 0.4, 0.1, 0.4]
        @test out.init_val_names == ["prey", "predator"]
        @test out.init_values == [10.0, 5.0]

        @test out.csv_written                        # CSV.write actually produced a file
    end

    @testset "SavingCallback path" begin
        @test length(out.saved.t) > 0
        @test out.saved.saveval[1] isa Tuple{Float64, Float64}
        # Intermediaries are appended to the time series, so this frame is longer.
        @test nrow(out.df_cb) > nrow(out.df)
        @test Set(["total", "product"]) ⊆ Set(out.df_cb.variable)
    end

    @testset "ensemble path" begin
        @test length(out.ens_sol.u) == 2       # .u holds the trajectories
        @test out.ts_df isa DataFrame
        @test names(out.ts_df) == ["condition", "sim", "time", "variable", "value"]
        @test out.param_df isa DataFrame
        @test out.init_df isa DataFrame

        # prob_func perturbs u0 per trajectory, so the two runs must differ.
        @test out.init_df.value[out.init_df.variable .== "prey"] == [10.1, 10.2]

        @test out.summ isa DataFrame
        @test Set(["mean", "median", "quant1", "quant2"]) ⊆ Set(names(out.summ))

        @test out.combos == [[1.0], [1.1]]
        @test out.total_sims == 4                    # 2 combinations x 2 replicates
    end

    @testset "helpers on the hot path" begin
        @test out.constants == (a = 1.0, b = [1.0, 2.0])
        @test out.init_dict == Dict("prey" => 10.0, "predator" => 5.0)
        @test out.interp == [1.5]
        @test 0.0 <= out.rng_val <= 1.0
        @test out.is_fn == false
    end
end
