using CSV
using DataFrames
using DiffEqCallbacks
using OrdinaryDiffEq
using OrdinaryDiffEqLowOrderRK
using SciMLBase
using StatsBase
using Random

@testset "Bundled dependency smoke tests" begin

    @testset "CSV" begin
        df = DataFrame(x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0])
        tmp = tempname() * ".csv"
        CSV.write(tmp, df)
        result = CSV.read(tmp, DataFrame)
        @test result.x ≈ [1.0, 2.0, 3.0]
        @test result.y ≈ [4.0, 5.0, 6.0]
    end

    @testset "DiffEqCallbacks" begin
        # Mirrors R-script usage: SavedValues + SavingCallback capture
        # intermediary (non-stock) variables during ODE integration
        f(u, p, t) = u
        u0 = [1.0]
        tspan = (0.0, 1.0)
        saveat = [0.0, 0.5, 1.0]

        saved = SavedValues(Float64, Any)
        cb = SavingCallback((u, _t, _integrator) -> copy(u), saved, saveat=saveat)

        prob = ODEProblem(f, u0, tspan)
        solve(prob, Tsit5(), callback=cb)

        @test saved.t == saveat
        @test length(saved.saveval) == 3
        @test saved.saveval[1] ≈ [1.0]
    end

    @testset "StatsBase" begin
        # Mirrors R-script usage: Julia equivalent of R's sample()

        # Unweighted sampling without replacement
        result = StatsBase.sample(1:10, 3, replace=false)
        @test length(result) == 3
        @test length(unique(result)) == 3
        @test all(x -> x in 1:10, result)

        # Unweighted sampling with replacement
        result_rep = StatsBase.sample(1:5, 20, replace=true)
        @test length(result_rep) == 20

        # Weighted sampling — all weight on one item
        items = ["a", "b", "c"]
        weights = StatsBase.pweights([0.0, 0.0, 1.0])
        result_w = StatsBase.sample(items, weights, 5, replace=true)
        @test all(x -> x == "c", result_w)
    end

    @testset "DiffEqCallbacks — ensemble with per-trajectory intermediaries" begin
        function decay!(du, u, p, _t)
            du[1] = -p.rate * u[1]
        end
        tspan = (0.0, 2.0)
        saveat = [0.0, 1.0, 2.0]
        n = 3

        intermediaries = Vector{SavedValues{Float64, Any}}(undef, n)
        for i in eachindex(intermediaries)
            intermediaries[i] = SavedValues(Float64, Any)
        end

        base_prob = ODEProblem(decay!, [10.0], tspan, (rate=0.5,))

        function prob_func(prob, ctx)
            remake(prob, callback=SavingCallback(
                (u, _t, _integrator) -> copy(u),
                intermediaries[ctx.sim_id],
                saveat=saveat
            ))
        end

        ensemble_prob = EnsembleProblem(base_prob, prob_func=prob_func)
        solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories=n, saveat=saveat)

        for i in 1:n
            @test intermediaries[i].t == saveat
            @test length(intermediaries[i].saveval) == 3
        end
    end

    @testset "SciMLBase — output_func" begin
        neg_exp(u, _p, _t) = -u
        base_prob = ODEProblem(neg_exp, [1.0], (0.0, 1.0))

        function output_func(sol, _ctx)
            return (t=sol.t, u=sol.u, p=sol.prob.p), false
        end

        ensemble_prob = EnsembleProblem(base_prob, output_func=output_func)
        sol = solve(ensemble_prob, Tsit5(), EnsembleThreads(),
                    trajectories=2, saveat=0.5)

        @test haskey(sol.u[1], :t)
        @test haskey(sol.u[1], :u)
        @test sol.u[1].t == sol.u[2].t
    end

    @testset "DataFrames — post-processing patterns" begin
        df = DataFrame(
            variable=["S", "I", "R", "S", "I"],
            time=[0.0, 0.0, 0.0, 1.0, 1.0],
            value=[100.0, 10.0, 0.0, 90.0, 15.0]
        )

        selected = Set(["S", "I"])
        filter!(row -> row.variable in selected, df)
        @test nrow(df) == 4
        @test all(row -> row.variable in selected, eachrow(df))

        df2 = DataFrame(time=[0.0, 1.0], mean=[5.0, 6.0], sd=[0.1, 0.2])
        select!(df2, Not(:time))
        @test !("time" in names(df2))
        @test "mean" in names(df2)
    end

    @testset "Solver smoke tests" begin
        function exp_decay!(du, u, _p, _t)
            du[1] = -u[1]
        end
        u0 = [1.0]
        tspan = (0.0, 1.0)

        @testset "OrdinaryDiffEqLowOrderRK — fixed-step" begin
            for (name, alg, dt) in [
                ("Euler",    Euler(),    0.05),
                ("Heun",     Heun(),     0.05),
                ("Midpoint", Midpoint(), 0.05),
                ("RK4",      RK4(),      0.1),
            ]
                @testset "$name" begin
                    sol = solve(ODEProblem(exp_decay!, u0, tspan), alg, dt=dt)
                    @test sol.u[end][1] < 1.0
                    @test sol.retcode == ReturnCode.Success
                end
            end
        end

        @testset "OrdinaryDiffEqLowOrderRK — adaptive" begin
            sol = solve(ODEProblem(exp_decay!, u0, tspan), BS3())
            @test sol.u[end][1] < 1.0
            @test sol.retcode == ReturnCode.Success
        end

        @testset "OrdinaryDiffEqTsit5 / OrdinaryDiffEqVerner" begin
            for (name, alg) in [
                ("Tsit5", Tsit5()),
                ("Vern6", Vern6()),
                ("Vern7", Vern7()),
                ("Vern8", Vern8()),
                ("Vern9", Vern9()),
            ]
                @testset "$name" begin
                    sol = solve(ODEProblem(exp_decay!, u0, tspan), alg)
                    @test sol.u[end][1] < 1.0
                    @test sol.retcode == ReturnCode.Success
                end
            end
        end

        @testset "OrdinaryDiffEqRosenbrock — mildly stiff" begin
            sol = solve(ODEProblem(exp_decay!, u0, tspan), Rosenbrock23())
            @test sol.u[end][1] < 1.0
            @test sol.retcode == ReturnCode.Success
        end

    end

end
