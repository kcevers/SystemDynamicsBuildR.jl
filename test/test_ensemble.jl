using SystemDynamicsBuildR.ensemble
using DataFrames
using Statistics
using OrdinaryDiffEq
using SciMLBase
using DiffEqCallbacks
using Random

@testset "ensemble tests" begin

    @testset "Parameter combination generation" begin
        @testset "Crossed design - basic" begin
            param_ranges = Dict(
                :alpha => [0.1, 0.5, 1.0],
                :beta => [2.0, 5.0]
            )
            
            combinations, total = generate_param_combinations(
                param_ranges; crossed=true, n_replicates=10
            )
            
            # Should have 3 × 2 = 6 combinations
            @test length(combinations) == 6
            @test total == 60  # 6 × 10
            
            # Check that all combinations are unique
            @test length(unique(combinations)) == 6
        end

        @testset "Crossed design - three parameters" begin
            param_ranges = Dict(
                :a => [1, 2],
                :b => [10, 20],
                :c => [100, 200]
            )
            
            combinations, total = generate_param_combinations(
                param_ranges; crossed=true, n_replicates=5
            )
            
            @test length(combinations) == 8  # 2 × 2 × 2
            @test total == 40  # 8 × 5
        end

        @testset "Non-crossed design - paired" begin
            param_ranges = Dict(
                :alpha => [0.1, 0.5, 1.0],
                :beta => [2.0, 5.0, 10.0]
            )
            
            combinations, total = generate_param_combinations(
                param_ranges; crossed=false, n_replicates=10
            )
            
            # Should have 3 combinations (paired)
            @test length(combinations) == 3
            @test total == 30  # 3 × 10
            
            # Check pairing is correct (sorted keys: alpha, beta)
            @test combinations[1] == [0.1, 2.0]
            @test combinations[2] == [0.5, 5.0]
            @test combinations[3] == [1.0, 10.0]
        end

        @testset "Non-crossed design - error on mismatched lengths" begin
            param_ranges = Dict(
                :alpha => [0.1, 0.5],
                :beta => [2.0, 5.0, 10.0]  # Different length
            )
            
            @test_throws ArgumentError generate_param_combinations(
                param_ranges; crossed=false, n_replicates=10
            )
        end

        @testset "Single parameter" begin
            param_ranges = Dict(:alpha => [0.1, 0.5, 1.0])
            
            combinations, total = generate_param_combinations(
                param_ranges; crossed=true, n_replicates=5
            )
            
            @test length(combinations) == 3
            @test total == 15
        end

        @testset "Different replicate counts" begin
            param_ranges = Dict(:a => [1, 2], :b => [10, 20])
            
            _, total_10 = generate_param_combinations(
                param_ranges; crossed=true, n_replicates=10
            )
            _, total_100 = generate_param_combinations(
                param_ranges; crossed=true, n_replicates=100
            )
            
            @test total_10 == 40   # 4 × 10
            @test total_100 == 400 # 4 × 100
        end
    end

    @testset "Transform intermediaries" begin
        @testset "Basic transformation with actual ODE solutions" begin
            # Define exponential decay: dS/dt = -r*S
            function decay!(du, u, p, t)
                du[1] = -p.rate * u[1]
            end
            
            # Solve two trajectories with stochastic initial conditions
            Random.seed!(123)
            intermediaries = []
            for _ in 1:2
                u0 = 10.0 + randn() * 0.5
                prob = ODEProblem(decay!, [u0], (0.0, 2.0), (rate = 0.5,))
                sol = solve(prob, Tsit5(), saveat = 0.5)
                
                # Compute outflow as intermediary (outflow = rate * stock)
                outflow = 0.5 .* vec(sol.u)
                push!(intermediaries, (t = sol.t, saveval = outflow))
            end
            
            transformed = transform_intermediaries(intermediaries, [:outflow])
            
            @test length(transformed) == 2
            @test all(!isempty(t.t) for t in transformed)
            @test all(!isempty(t.u) for t in transformed)
        end

        @testset "Empty intermediaries with actual structure" begin
            # Test with properly structured but empty result (edge case)
            intermediaries = [
                (t = Float64[], saveval = Float64[])
            ]
            
            transformed = transform_intermediaries(intermediaries)
            
            @test length(transformed) == 1
            @test isempty(transformed[1].t)
            @test isempty(transformed[1].u)
        end

        @testset "Nothing intermediaries from ensemble with proper handling" begin
            # Simulate ensemble with some trajectories having no saved intermediaries
            intermediaries = [nothing, nothing]
            
            transformed = transform_intermediaries(intermediaries)
            
            @test length(transformed) == 2
            @test isempty(transformed[1].t)
            @test isempty(transformed[2].t)
        end
    end

    @testset "ensemble_to_df - basic functionality" begin
        @testset "Single variable, single trajectory from ODE" begin
            # Simple exponential decay: dS/dt = -0.5*S
            function decay!(du, u, p, t)
                du[1] = -p.alpha * u[1]
            end
            
            Random.seed!(42)
            u0 = 10.0
            prob = ODEProblem(decay!, [u0], (0.0, 2.0), (alpha = 0.5, beta = 2.0))
            sol = solve(prob, Tsit5(), saveat = 1.0)
            
            solve_out = [
                (
                    t = sol.t,
                    u = vec(sol.u),
                    u0 = u0,
                    p = (alpha = 0.5, beta = 2.0)
                )
            ]
            
            ts_df, param_df, init_df = ensemble_to_df(
                solve_out, [:stock], nothing, nothing, 1
            )
            
            # Check time series
            @test nrow(ts_df) == length(sol.t)
            @test all(ts_df.condition .== 1)
            @test all(ts_df.sim .== 1)
            @test ts_df.time == sol.t
            @test all(ts_df.variable .== "stock")
            
            # Check parameters
            @test nrow(param_df) == 2
            @test "alpha" in param_df.variable
            @test "beta" in param_df.variable
            
            # Check initial values
            @test nrow(init_df) == 1
            @test init_df.variable == ["stock"]
            @test init_df.value[1] ≈ u0
        end

        @testset "Multiple variables from S-I-R compartmental model" begin
            # S-I-R model: dS/dt = -β*S*I, dI/dt = β*S*I - γ*I, dR/dt = γ*I
            function sir!(du, u, p, t)
                S, I, R = u
                β, γ = p.beta, p.gamma
                du[1] = -β * S * I
                du[2] = β * S * I - γ * I
                du[3] = γ * I
            end
            
            Random.seed!(456)
            u0 = [0.99, 0.01, 0.0]
            prob = ODEProblem(sir!, u0, (0.0, 2.0), (beta = 0.5, gamma = 0.1))
            sol = solve(prob, Tsit5(), saveat = 0.5)
            
            solve_out = [
                (
                    t = sol.t,
                    u = vec.(sol.u),  # Convert array of arrays
                    u0 = u0,
                    p = (beta = 0.5,)
                )
            ]
            
            ts_df, param_df, init_df = ensemble_to_df(
                solve_out, [:S, :I, :R], nothing, nothing, 1
            )
            
            @test nrow(ts_df) == length(sol.t) * 3  # 3 variables × time points
            @test "S" in ts_df.variable
            @test "I" in ts_df.variable
            @test "R" in ts_df.variable
            
            # Check values for each variable
            s_data = subset(ts_df, :variable => ByRow(==("S")))
            i_data = subset(ts_df, :variable => ByRow(==("I")))
            r_data = subset(ts_df, :variable => ByRow(==("R")))
            
            @test nrow(s_data) == length(sol.t)
            @test nrow(i_data) == length(sol.t)
            @test nrow(r_data) == length(sol.t)
        end

        @testset "Multiple trajectories with ensemble_n from real ODE ensemble" begin
            # Define base model
            function decay!(du, u, p, t)
                du[1] = -p.rate * u[1]
            end
            
            # Create ensemble with 4 trajectories (2 parameter combos × 2 replicates)
            Random.seed!(789)
            alpha_vals = [0.3, 0.6]
            solve_out = []
            
            for (j, alpha) in enumerate(alpha_vals)
                for i in 1:2
                    u0 = 10.0 + randn() * 0.1
                    prob = ODEProblem(decay!, [u0], (0.0, 1.0), (rate = alpha,))
                    sol = solve(prob, Tsit5(), saveat = 0.25)
                    
                    push!(solve_out, (
                        t = sol.t,
                        u = vec(sol.u),
                        u0 = u0,
                        p = (rate = alpha,)
                    ))
                end
            end
            
            ts_df, param_df, init_df = ensemble_to_df(
                solve_out, [:stock], nothing, nothing, 2  # 2 replicates per condition
            )
            
            # Check condition indices (parameter combination)
            @test 1 in ts_df.condition
            @test 2 in ts_df.condition

            # Check sim indices (replicate)
            @test all(s -> s in [1, 2], ts_df.sim)

            # Verify structure matches 2x2 ensemble
            for combo_j in [1, 2]
                for replicate_i in [1, 2]
                    subset_data = subset(ts_df, [:condition, :sim] => ByRow((condition, sim) -> condition == combo_j && sim == replicate_i))
                    @test nrow(subset_data) > 0
                end
            end
        end

        @testset "With intermediaries from S-I-R model" begin
            # Define S-I-R model
            function sir!(du, u, p, t)
                S, I = u
                β, γ = p.beta, p.gamma
                du[1] = -β * S * I
                du[2] = β * S * I - γ * I
            end
            
            Random.seed!(101)
            u0 = [0.99, 0.01]
            prob = ODEProblem(sir!, u0, (0.0, 1.0), (beta = 0.5, gamma = 0.1))
            sol = solve(prob, Tsit5(), saveat = 0.25)
            
            # Compute transmission rate as intermediary
            transmission = [0.5 * u[1] * u[2] for u in sol.u]
            
            solve_out = [
                (t = sol.t, u = vec.(sol.u), u0 = u0, p = (beta = 0.5,))
            ]
            
            intermediaries = [
                (t = sol.t, saveval = transmission)
            ]
            
            ts_df, _, _ = ensemble_to_df(
                solve_out, [:S, :I], intermediaries, [:transmission], 1
            )
            
            # Should have both stocks and intermediaries
            @test "S" in ts_df.variable
            @test "I" in ts_df.variable
            @test "transmission" in ts_df.variable
            
            # Verify counts
            s_data = subset(ts_df, :variable => ByRow(==("S")))
            trans_data = subset(ts_df, :variable => ByRow(==("transmission")))
            
            @test nrow(s_data) == length(sol.t)
            @test nrow(trans_data) == length(sol.t)
        end

        @testset "Vector parameters from ODE ensemble" begin
            # Model with multiple parameters as vector
            function model!(du, u, p, t)
                du[1] = -p[1] * u[1] + p[2]
            end

            Random.seed!(111)
            u0 = [5.0]
            prob = ODEProblem(model!, u0, (0.0, 1.0), [0.5, 2.0, 3.0])
            sol = solve(prob, Tsit5(), saveat = 0.5)

            solve_out = [
                (t = sol.t, u = vec(sol.u), u0 = u0[1], p = [0.5, 2.0, 3.0])
            ]

            _, param_df, _ = ensemble_to_df(
                solve_out, [:u], nothing, nothing, 1
            )

            @test nrow(param_df) == 3
            @test "p1" in param_df.variable
            @test "p2" in param_df.variable
            @test "p3" in param_df.variable
        end

        @testset "Scalar state variable from out-of-place ODE" begin
            # Out-of-place scalar ODE: sol.u is Vector{Float64}, not Vector{Vector{Float64}}
            # This exercises the else branch at ensemble.jl:155-157 and the scalar u_val
            # path inside process_solution_like.
            decay_scalar(u, p, t) = -p.rate * u

            Random.seed!(222)
            solve_out = []
            for _ in 1:3
                u0 = 10.0 + randn() * 0.2
                prob = ODEProblem(decay_scalar, u0, (0.0, 2.0), (rate = 0.5,))
                sol = solve(prob, Tsit5(), saveat = 1.0)
                push!(solve_out, (t = sol.t, u = sol.u, u0 = u0, p = (rate = 0.5,)))
            end

            # sol.u[1] is a plain Float64 → hits the scalar branch
            @test !isa(solve_out[1].u[1], AbstractVector)

            ts_df, param_df, init_df = ensemble_to_df(
                solve_out, [:stock], nothing, nothing, 1
            )

            n_times = length(solve_out[1].t)
            @test nrow(ts_df) == 3 * n_times
            @test all(ts_df.variable .== "stock")
            @test all(ts_df.value .> 0)  # exponential decay stays positive

            @test nrow(param_df) == 3   # 3 trajectories × 1 param
            @test "rate" in param_df.variable

            @test nrow(init_df) == 3
            @test all(init_df.variable .== "stock")
        end

        @testset "EnsembleSolution passed directly (with output_func)" begin
            # Mirrors the real usage pattern: solve(EnsembleProblem) returns an
            # EnsembleSolution; ensemble_to_df must unwrap it before indexing.
            function decay!(du, u, p, t)
                du[1] = -p.rate * u[1]
            end

            Random.seed!(444)
            base_prob = ODEProblem(decay!, [10.0], (0.0, 2.0), (rate = 0.5,))

            function prob_func_decay(prob, ctx)
                u0_new = 10.0 + randn(ctx.rng) * 0.2
                remake(prob, u0 = [u0_new])
            end

            function output_func_decay(sol, ctx)
                return (t = sol.t, u = sol.u, p = sol.prob.p, u0 = sol.prob.u0), false
            end

            ensemble_prob = EnsembleProblem(base_prob,
                prob_func = prob_func_decay,
                output_func = output_func_decay)
            solve_out = solve(ensemble_prob, Tsit5(), EnsembleSerial(),
                              trajectories = 3, saveat = 1.0)

            # solve_out is an EnsembleSolution, not a plain Vector
            @test !isa(solve_out, Vector)

            ts_df, param_df, init_df = ensemble_to_df(
                solve_out, [:stock], nothing, nothing, 1
            )

            n_times = length(solve_out.u[1].t)
            @test nrow(ts_df) == 3 * n_times
            @test all(ts_df.variable .== "stock")
            @test all(ts_df.value .> 0)
            @test nrow(param_df) == 3
            @test "rate" in param_df.variable
            @test nrow(init_df) == 3
            @test all(init_df.variable .== "stock")
        end

        @testset "EnsembleSolution with intermediaries via SavingCallback" begin
            # Full integration test matching the user's R-script pattern:
            # EnsembleProblem + output_func + SavingCallback intermediaries.
            function decay_cb!(du, u, p, t)
                du[1] = -p.rate * u[1]
            end

            function save_flow(current_state, t, integrator)
                return integrator.p.rate * current_state[1]  # outflow = rate * stock
            end

            n_traj = 3
            saveat_times = 0.0:1.0:2.0

            intermediaries_cb = Vector{SavedValues{Float64, Any}}(undef, n_traj)
            for k in eachindex(intermediaries_cb)
                intermediaries_cb[k] = SavedValues(Float64, Any)
            end

            Random.seed!(555)
            base_prob_cb = ODEProblem(decay_cb!, [10.0], (0.0, 2.0), (rate = 0.5,))

            function prob_func_cb(prob, ctx)
                k = ctx.sim_id
                u0_new = 10.0 + randn(ctx.rng) * 0.2
                cb = SavingCallback(save_flow, intermediaries_cb[k], saveat = saveat_times)
                remake(prob, u0 = [u0_new], callback = cb)
            end

            function output_func_cb(sol, ctx)
                return (t = sol.t, u = sol.u, p = sol.prob.p, u0 = sol.prob.u0), false
            end

            ensemble_prob_cb = EnsembleProblem(base_prob_cb,
                prob_func = prob_func_cb,
                output_func = output_func_cb)
            solve_out_cb = solve(ensemble_prob_cb, Tsit5(), EnsembleSerial(),
                                 trajectories = n_traj, saveat = saveat_times)

            ts_df, param_df_cb, init_df_cb = ensemble_to_df(
                solve_out_cb, [:stock],
                intermediaries_cb, [:flow],
                1
            )

            @test "stock" in ts_df.variable
            @test "flow" in ts_df.variable

            # Each variable should have n_traj × n_time_points rows
            n_times = length(saveat_times)
            stock_rows = subset(ts_df, :variable => ByRow(==("stock")))
            flow_rows  = subset(ts_df, :variable => ByRow(==("flow")))
            @test nrow(stock_rows) == n_traj * n_times
            @test nrow(flow_rows)  == n_traj * n_times

            # Flow values should be positive (rate × positive stock)
            @test all(flow_rows.value .> 0)

            # param_df: n_traj trajectories × 1 parameter (:rate)
            @test nrow(param_df_cb) == n_traj
            @test "rate" in param_df_cb.variable

            # init_df: n_traj trajectories × 1 initial condition (:stock)
            @test nrow(init_df_cb) == n_traj
            @test all(init_df_cb.variable .== "stock")
        end

        @testset "No parameters (empty NamedTuple constants)" begin
            # Models with no free parameters return param_df with zero rows.
            # param_df must still have the correct columns so ensemble_summ works.
            function grow!(du, u, p, t)
                du[1] = u[1]
            end

            u0 = [1.0]
            prob_np = ODEProblem(grow!, u0, (0.0, 2.0), ())  # empty params
            solve_out = []
            for i in 1:3
                sol = solve(prob_np, Tsit5(), saveat = 1.0)
                push!(solve_out, (t = sol.t, u = sol.u, u0 = u0[1], p = ()))
            end

            ts_df, param_df, init_df = ensemble_to_df(solve_out, [:S], nothing, nothing, 1)

            # timeseries: 3 trajectories × n_times time points, all for variable "S"
            n_times = length(solve_out[1].t)
            @test nrow(ts_df) == 3 * n_times
            @test all(ts_df.variable .== "S")
            @test all(ts_df.value .> 0)  # exponential growth stays positive

            # ensemble_summ on the timeseries must work and return one row per (condition, time) pair
            summ = ensemble_summ(ts_df)
            n_j = length(unique(ts_df.condition))
            @test nrow(summ) == n_j * n_times

            # param_df must have the correct column schema even with zero rows
            @test nrow(param_df) == 0
            @test hasproperty(param_df, :condition)
            @test hasproperty(param_df, :sim)
            @test hasproperty(param_df, :variable)
            @test hasproperty(param_df, :value)

            # ensemble_summ must not error on the empty param_df
            param_df[!, :time] .= 0.0
            summ_empty = ensemble_summ(param_df)
            @test nrow(summ_empty) == 0

            # init_df: 3 trajectories × 1 initial condition (:S), scalar u0
            @test nrow(init_df) == 3
            @test all(init_df.variable .== "S")
        end
    end

    @testset "Real ensemble with stocks and intermediaries across replicas" begin
        # Define a simple model: dS/dt = -α*S (exponential decay)
        # S is stock, flow = α*S is intermediary
        function decay!(du, u, p, t)
            du[1] = -p.alpha * u[1]
        end
        
        # Parameter combinations: 2 values of alpha
        alphas = [0.5, 1.0]
        ensemble_n = 3  # 3 replicates per parameter combo
        
        # Storage for ensemble results
        solve_out = []
        intermediaries_list = []
        
        Random.seed!(42)
        
        # Create base problem with dummy parameters (will be set by prob_func)
        base_prob = ODEProblem(decay!, [10.0], (0.0, 10.0), (alpha = 0.5,))
        
        # Total trajectories: 2 alphas × 3 replicates = 6 trajectories
        total_trajectories = length(alphas) * ensemble_n
        
        # Function to generate different parameters and initial conditions for each trajectory
        function prob_func(prob, ctx)
            # Map trajectory index to (combo_idx, replicate_idx)
            i = ctx.sim_id
            combo_idx = div(i - 1, ensemble_n) + 1

            # Get alpha value for this parameter combination
            alpha_val = alphas[combo_idx]

            # Each trajectory gets a different initial condition
            u0_new = 10.0 + randn(ctx.rng) * 0.1

            # Update both u0 and parameter α
            remake(prob, u0 = [u0_new], p = (alpha = alpha_val,))
        end
        
        # Create and solve the ensemble problem
        ensemble_prob = EnsembleProblem(base_prob, prob_func = prob_func)
        ensemble_sol = solve(ensemble_prob, Tsit5(), EnsembleThreads(), 
                             trajectories = total_trajectories, saveat = 0.5)
        
        # Extract results from all trajectories in order
        for i in 1:total_trajectories
            # Map trajectory back to parameter combination
            combo_idx = div(i - 1, ensemble_n) + 1
            alpha_val = alphas[combo_idx]
            
            # Get solution for this trajectory
            sol = ensemble_sol.u[i]
            
            # Get initial condition from this trajectory's actual u0
            u0_actual = sol.u[1][1]
            
            push!(solve_out, (
                t = sol.t,
                u = vec(sol.u),
                u0 = u0_actual,
                p = (alpha = alpha_val,)
            ))
            
            # Compute intermediary values from solution (flow = α*S)
            flow_vals = alpha_val .* vec(sol.u)
            
            push!(intermediaries_list, (
                t = sol.t,
                saveval = flow_vals
            ))
        end
        
        # Convert to dataframes
        ts_df, param_df, init_df = ensemble_to_df(
            solve_out,
            [:stock],
            intermediaries_list,
            [:flow],
            ensemble_n
        )
        
        # Test 1: Check that we have both stocks and non-stocks
        @test "stock" in ts_df.variable
        @test "flow" in ts_df.variable
        
        # Test 2: Check that both condition and sim are present with correct ranges
        @test all(c -> c in [1, 2], ts_df.condition)
        @test all(s -> s in [1, 2, 3], ts_df.sim)
        
        # Test 3: Non-stocks should have values for ALL replicas
        # If bug exists, only i=1 will have non-NA flow values
        flow_data = subset(ts_df, :variable => ByRow(==("flow")))
        stock_data = subset(ts_df, :variable => ByRow(==("stock")))
        
        # Stocks should work fine for all combos and replicates
        stock_counts = combine(groupby(stock_data, [:condition, :sim]), nrow => :count)
        @test all(row -> row.count > 0, eachrow(stock_counts))

        # Non-stocks must also work for all combos and replicates
        # This will FAIL if the bug exists (only sim=1 populated)
        flow_counts = combine(groupby(flow_data, [:condition, :sim]), nrow => :count)
        @test all(row -> row.count > 0, eachrow(flow_counts))
        
        # Test 4: No missing or NaN values in flows
        @test all(.!ismissing.(flow_data.value))
        @test all(.!isnan.(flow_data.value))
        
        # Test 5: Check that each (j, i) combination has expected coverage
        for combo_j in [1, 2]
            for replicate_i in [1, 2, 3]
                flow_subset = subset(flow_data,
                    [:condition, :sim] => ByRow((condition, sim) -> condition == combo_j && sim == replicate_i))
                
                # Should have flow data for this replicate
                @test nrow(flow_subset) > 0
                @test all(.!ismissing.(flow_subset.value))
            end
        end
    end

    @testset "ensemble_summ - statistical summaries" begin
        @testset "Basic statistics from ODE ensemble" begin
            # Create ensemble with 3 replicates at 2 time points
            function decay!(du, u, p, t)
                du[1] = -p.rate * u[1]
            end
            
            Random.seed!(404)
            timeseries_data = []
            
            for i in 1:3  # 3 replicates
                u0 = 10.0 + randn() * 0.1
                prob = ODEProblem(decay!, [u0], (0.0, 1.0), (rate = 0.5,))
                sol = solve(prob, Tsit5(), saveat = [0.0, 1.0])
                
                for (t_idx, t_val) in enumerate(sol.t)
                    push!(timeseries_data, (
                        condition = 1,
                        sim = i,
                        time = t_val,
                        variable = "stock",
                        value = sol.u[t_idx][1]
                    ))
                end
            end
            
            timeseries_df = DataFrame(timeseries_data)
            stats = ensemble_summ(timeseries_df, [0.025, 0.975])
            
            @test nrow(stats) == 2  # 2 time points
            
            # Check time 0.0
            t0_stats = subset(stats, :time => ByRow(==(0.0)))
            @test !isnan(t0_stats.mean[1])
            @test !isnan(t0_stats.median[1])
            
            # Check time 1.0
            t1_stats = subset(stats, :time => ByRow(==(1.0)))
            @test !isnan(t1_stats.mean[1])
            @test t1_stats.mean[1] < t0_stats.mean[1]  # Should decay
        end

        @testset "Handling NaN and missing from actual simulations" begin
            # Simulate data where some replicates fail to compute (represented as NaN)
            timeseries_df = DataFrame(
                condition = [1, 1, 1, 1],
                sim = [1, 2, 3, 4],
                time = [0.0, 0.0, 0.0, 0.0],
                variable = ["x", "x", "x", "x"],
                value = [10.0, NaN, 12.0, missing]
            )
            
            stats = ensemble_summ(timeseries_df, [0.025, 0.975], ["mean", "median", "missing_count"])

            @test nrow(stats) == 1
            @test stats.mean[1] ≈ 11.0  # mean of [10.0, 12.0]
            @test stats.missing_count[1] == 2
        end

        @testset "All NaN values from failed ensemble" begin
            # All trajectories failed (represented as NaN)
            timeseries_df = DataFrame(
                condition = [1, 1],
                sim = [1, 2],
                time = [0.0, 0.0],
                variable = ["x", "x"],
                value = [NaN, NaN]
            )
            
            stats = ensemble_summ(timeseries_df, [0.025, 0.975], ["mean", "median", "var"])

            @test isnan(stats.mean[1])
            @test isnan(stats.median[1])
            @test isnan(stats.var[1])
        end

        @testset "Multiple variables from S-I-R ensemble" begin
            # Create S-I-R ensemble results
            function sir!(du, u, p, t)
                S, I = u
                β, γ = p.beta, p.gamma
                du[1] = -β * S * I
                du[2] = β * S * I - γ * I
            end
            
            Random.seed!(505)
            timeseries_data = []
            
            for i in 1:2
                u0 = [0.99 - 0.001*i, 0.01 + 0.001*i]
                prob = ODEProblem(sir!, u0, (0.0, 1.0), (beta = 0.5, gamma = 0.1))
                sol = solve(prob, Tsit5(), saveat = [0.0])
                
                push!(timeseries_data, (
                    condition = 1, sim = i, time = 0.0, variable = "S",
                    value = sol.u[1][1]
                ))
                push!(timeseries_data, (
                    condition = 1, sim = i, time = 0.0, variable = "I",
                    value = sol.u[1][2]
                ))
            end

            timeseries_df = DataFrame(timeseries_data)
            stats = ensemble_summ(timeseries_df)

            @test nrow(stats) >= 2  # At least S and I statistics
            
            s_stats = subset(stats, :variable => ByRow(==("S")))
            i_stats = subset(stats, :variable => ByRow(==("I")))
            
            @test !isnan(s_stats.mean[1])
            @test !isnan(i_stats.mean[1])
            @test s_stats.mean[1] > i_stats.mean[1]  # S should dominate initially
        end

        @testset "Custom quantiles from ODE ensemble" begin
            # Create ensemble data
            Random.seed!(606)
            values = sort(randn(100) .* 2 .+ 50)  # Some realistic values
            
            timeseries_df = DataFrame(
                condition = [1 for _ in 1:100],
                sim = 1:100,
                time = [0.0 for _ in 1:100],
                variable = ["x" for _ in 1:100],
                value = values
            )
            
            stats = ensemble_summ(timeseries_df, [0.1, 0.5, 0.9])

            @test hasproperty(stats, :quant1)   # 0.1 → quant1
            @test hasproperty(stats, :quant2)   # 0.5 → quant2
            @test hasproperty(stats, :quant3)   # 0.9 → quant3

            @test stats.quant1[1] < stats.quant2[1]  # Lower quantile
            @test stats.quant2[1] < stats.quant3[1]  # Higher quantile
        end

        @testset "Selectable summary statistics" begin
            # One group with a known set of values plus one NaN, so every
            # statistic has a checkable value and missing_count is non-zero.
            timeseries_df = DataFrame(
                condition = fill(1, 6),
                sim = 1:6,
                time = fill(0.0, 6),
                variable = fill("x", 6),
                value = [1.0, 2.0, 3.0, 4.0, 5.0, NaN]
            )

            all_stats = ["mean", "median", "sd", "var", "min", "max", "missing_count"]
            stats = ensemble_summ(timeseries_df, [0.025, 0.975], all_stats)

            @test nrow(stats) == 1
            @test stats.mean[1] ≈ 3.0
            @test stats.median[1] ≈ 3.0
            @test stats.sd[1] ≈ std([1.0, 2.0, 3.0, 4.0, 5.0])
            @test stats.var[1] ≈ var([1.0, 2.0, 3.0, 4.0, 5.0])
            @test stats.min[1] ≈ 1.0
            @test stats.max[1] ≈ 5.0
            @test stats.missing_count[1] == 1

            # Columns: grouping keys, the requested stats in catalog order, then
            # quant columns in the order of `quantiles`.
            @test names(stats) ==
                  ["condition", "time", "variable", all_stats..., "quant1", "quant2"]

            # Requesting a subset in scrambled order still yields catalog order.
            scrambled = ensemble_summ(timeseries_df, [0.5], ["max", "mean", "sd"])
            @test names(scrambled) ==
                  ["condition", "time", "variable", "mean", "sd", "max", "quant1"]

            # Threaded variant produces identical columns and values.
            stats_thr = ensemble_summ_threaded(timeseries_df, [0.025, 0.975], all_stats)
            @test names(stats_thr) == names(stats)
            for col in ["mean", "median", "sd", "var", "min", "max"]
                @test isapprox(stats[!, col], stats_thr[!, col], rtol = 1e-10)
            end
            @test stats_thr.missing_count[1] == 1

            # All-missing group returns NaN for every statistic (empty path).
            empty_df = DataFrame(
                condition = [1, 1], sim = [1, 2], time = [0.0, 0.0],
                variable = ["x", "x"], value = [NaN, NaN]
            )
            empty_stats = ensemble_summ(empty_df, [0.5], all_stats)
            for col in ["mean", "median", "sd", "var", "min", "max"]
                @test isnan(empty_stats[1, col])
            end
            @test empty_stats.missing_count[1] == 2
        end

        @testset "Summary statistic helpers" begin
            v = [1.0, 2.0, 3.0, 4.0, 5.0]
            @test ensemble._ensemble_stat_value("mean", v) ≈ 3.0
            @test ensemble._ensemble_stat_value("median", v) ≈ 3.0
            @test ensemble._ensemble_stat_value("sd", v) ≈ std(v)
            @test ensemble._ensemble_stat_value("var", v) ≈ var(v)
            @test ensemble._ensemble_stat_value("min", v) ≈ 1.0
            @test ensemble._ensemble_stat_value("max", v) ≈ 5.0
            @test_throws ErrorException ensemble._ensemble_stat_value("bogus", v)

            # Ordering helper sorts by catalog order and drops unknown names.
            @test ensemble._order_ensemble_stats(["max", "mean", "bogus"]) ==
                  ["mean", "max"]
            @test ensemble._order_ensemble_stats(String[]) == String[]
        end

        @testset "Multiple parameter combinations from ODE ensemble" begin
            # Create ensemble with 2 parameter combos, 2 time points, varying replicates
            function decay!(du, u, p, t)
                du[1] = -p.rate * u[1]
            end
            
            Random.seed!(707)
            timeseries_data = []
            
            for alpha_idx in [1, 2]
                alpha = 0.3 + alpha_idx * 0.2
                for replicate in 1:2
                    for t_val in [0.0, 1.0]
                        u0 = 10.0
                        prob = ODEProblem(decay!, [u0], (0.0, 1.0), (rate = alpha,))
                        sol = solve(prob, Tsit5(), saveat = [0.0, 1.0])
                        t_idx = findfirst(x -> x ≈ t_val, sol.t)
                        
                        push!(timeseries_data, (
                            condition = alpha_idx,
                            sim = replicate,
                            time = t_val,
                            variable = "stock",
                            value = sol.u[t_idx][1]
                        ))
                    end
                end
            end
            
            timeseries_df = DataFrame(timeseries_data)
            stats = ensemble_summ(timeseries_df)
            
            @test nrow(stats) >= 2  # One row per j (at least)
            
            j1_stats = subset(stats, :condition => ByRow(==(1)))
            j2_stats = subset(stats, :condition => ByRow(==(2)))
            
            @test nrow(j1_stats) > 0
            @test nrow(j2_stats) > 0
        end
    end

    @testset "Threading variants" begin
        @testset "ensemble_to_df_threaded produces same results from ODE ensemble" begin
            # Create ensemble with S-I-R model
            function sir!(du, u, p, t)
                S, I = u
                β, γ = p.beta, p.gamma
                du[1] = -β * S * I
                du[2] = β * S * I - γ * I
            end
            
            Random.seed!(202)
            solve_out = []
            for i in 1:2
                u0 = [0.99 - 0.001*i, 0.01 + 0.001*i]
                prob = ODEProblem(sir!, u0, (0.0, 1.0), (beta = 0.5, gamma = 0.1))
                sol = solve(prob, Tsit5(), saveat = 0.25)
                push!(solve_out, (
                    t = sol.t,
                    u = vec.(sol.u),
                    u0 = u0,
                    p = (beta = 0.5,)
                ))
            end
            
            ts1, p1, i1 = ensemble_to_df(solve_out, [:S, :I], nothing, nothing, 1)
            ts2, p2, i2 = ensemble_to_df_threaded(solve_out, [:S, :I], nothing, nothing, 1)
            
            # Results should be identical
            @test ts1.condition == ts2.condition
            @test ts1.sim == ts2.sim
            @test ts1.time == ts2.time
            @test ts1.variable == ts2.variable
            @test ts1.value == ts2.value

            @test p1 == p2
            @test i1 == i2
        end

        @testset "ensemble_summ_threaded produces same results from multi-variable ODE" begin
            # Create multi-variable ensemble data from S-I-R model
            function sir!(du, u, p, t)
                S, I = u
                β, γ = p.beta, p.gamma  
                du[1] = -β * S * I
                du[2] = β * S * I - γ * I
            end
            
            Random.seed!(808)
            timeseries_data = []
            
            for i in 1:100
                u0 = [0.99, 0.01]
                prob = ODEProblem(sir!, u0, (0.0, 1.0), (beta = 0.5, gamma = 0.1))
                sol = solve(prob, Tsit5(), saveat = [0.0])
                
                push!(timeseries_data, (
                    condition = 1, sim = i, time = 0.0, variable = "S",
                    value = sol.u[1][1]
                ))
                push!(timeseries_data, (
                    condition = 1, sim = i, time = 0.0, variable = "I",
                    value = sol.u[1][2]
                ))
            end

            timeseries_df = DataFrame(timeseries_data)

            req_stats = ["mean", "median", "var"]
            stats1 = ensemble_summ(timeseries_df, [0.025, 0.975], req_stats)
            stats2 = ensemble_summ_threaded(timeseries_df, [0.025, 0.975], req_stats)

            # Results should be identical (very small numerical differences possible)
            @test isapprox(stats1.mean, stats2.mean, rtol=1e-10)
            @test isapprox(stats1.median, stats2.median, rtol=1e-10)
            @test isapprox(stats1.var, stats2.var, rtol=1e-10)
        end

        @testset "ensemble_to_df_threaded - scalar state variable" begin
            # Same scalar ODE as the non-threaded test; verifies the threaded version
            # also handles the scalar u path (ensemble.jl:408-410, 484-492).
            decay_scalar_t(u, p, t) = -p.rate * u

            Random.seed!(333)
            solve_out = []
            for _ in 1:3
                u0 = 10.0 + randn() * 0.2
                prob = ODEProblem(decay_scalar_t, u0, (0.0, 2.0), (rate = 0.5,))
                sol = solve(prob, Tsit5(), saveat = 1.0)
                push!(solve_out, (t = sol.t, u = sol.u, u0 = u0, p = (rate = 0.5,)))
            end

            ts1, p1, i1 = ensemble_to_df(solve_out, [:stock], nothing, nothing, 1)
            ts2, p2, i2 = ensemble_to_df_threaded(solve_out, [:stock], nothing, nothing, 1)

            @test ts1.condition == ts2.condition
            @test ts1.sim == ts2.sim
            @test ts1.time == ts2.time
            @test ts1.variable == ts2.variable
            @test ts1.value == ts2.value
            @test p1 == p2
            @test i1 == i2
        end
    end

    @testset "Integration tests" begin
        @testset "Full workflow with ODE ensemble and statistical summaries" begin
            # Generate parameters
            param_ranges = Dict(:alpha => [0.1, 0.5], :beta => [1.0, 2.0])
            combinations, total = generate_param_combinations(
                param_ranges; crossed=true, n_replicates=2
            )
            
            @test length(combinations) == 4
            @test total == 8
            
            # Define exponential model: dS/dt = -α*S + β (with forcing term)
            function forced_decay!(du, u, p, t)
                du[1] = -p.alpha * u[1] + p.beta
            end
            
            Random.seed!(303)
            
            # Create ensemble by solving ODE for each parameter combination
            solve_out = []
            for (j, combo) in enumerate(combinations)
                alpha_val, beta_val = combo[1], combo[2]
                for i in 1:2
                    # Stochastic initial condition for each replicate
                    u0 = combo[1] * 10 + randn() * 0.5
                    prob = ODEProblem(
                        forced_decay!, 
                        [u0], 
                        (0.0, 1.0), 
                        (alpha = alpha_val, beta = beta_val)
                    )
                    sol = solve(prob, Tsit5(), saveat = 0.5)
                    
                    push!(solve_out, (
                        t = sol.t,
                        u = vec(sol.u),
                        u0 = u0,
                        p = (alpha = alpha_val, beta = beta_val)
                    ))
                end
            end
            
            # Convert to DataFrame
            ts_df, param_df, init_df = ensemble_to_df(
                solve_out, [:stock], nothing, nothing, 2
            )
            
            @test nrow(ts_df) > 0
            @test maximum(ts_df.condition) == 4  # 4 parameter combinations
            @test maximum(ts_df.sim) == 2  # 2 replicates
            
            # Compute summaries
            stats = ensemble_summ(ts_df, [0.025, 0.975], ["mean", "var", "missing_count"])

            # Should have stats for each time point (excluding time=0.0 in some cases)
            @test nrow(stats) >= 1
            @test all(stats.missing_count .== 0)

            # Verify statistical measures make sense
            @test all(stats.mean .> 0)  # All means should be positive
            @test all(stats.var .>= 0)  # Variance should be non-negative
        end
    end

end