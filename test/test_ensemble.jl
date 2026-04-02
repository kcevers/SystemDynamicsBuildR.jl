using SystemDynamicsBuildR.ensemble
using DataFrames
using Unitful
using Statistics
using OrdinaryDiffEq
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
            @test all(ts_df.j .== 1)
            @test all(ts_df.i .== 1)
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
            
            # Check j indices (parameter combination)
            @test 1 in ts_df.j
            @test 2 in ts_df.j
            
            # Check i indices (replicate)
            @test all(i -> i in [1, 2], ts_df.i)
            
            # Verify structure matches 2x2 ensemble
            for combo_j in [1, 2]
                for replicate_i in [1, 2]
                    subset_data = subset(ts_df, [:j, :i] => ByRow((j, i) -> j == combo_j && i == replicate_i))
                    @test nrow(subset_data) > 0
                end
            end
        end

        @testset "With Unitful quantities in ODE output" begin
            # Define model with units
            function decay_units!(du, u, p, t)
                du[1] = -p.rate * u[1]
            end
            
            u0 = 10.0u"mol"
            prob = ODEProblem(decay_units!, [u0], (0.0u"hr", 2.0u"hr"), (rate = 0.5u"hr^-1",))
            sol = solve(prob, Tsit5(), saveat = 1.0u"hr")
            
            solve_out = [
                (
                    t = sol.t,
                    u = vec(sol.u),
                    u0 = u0,
                    p = (rate = 0.5u"hr^-1",)
                )
            ]
            
            ts_df, param_df, init_df = ensemble_to_df(
                solve_out, [:concentration], nothing, nothing, 1
            )
            
            # Units should be stripped
            @test all(typeof.(ts_df.time) .== Float64)
            @test all(typeof.(ts_df.value) .== Float64)
            @test all(typeof.(param_df.value) .== Float64)
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
        function prob_func(prob, i, repeat)
            # Map trajectory index to (combo_idx, replicate_idx)
            combo_idx = div(i - 1, ensemble_n) + 1
            replicate_idx = rem(i - 1, ensemble_n) + 1
            
            # Get alpha value for this parameter combination
            alpha_val = alphas[combo_idx]
            
            # Each trajectory gets a different initial condition
            u0_new = 10.0 + randn() * 0.1
            
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
            sol = ensemble_sol[i]
            
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
        
        # Test 2: Check that both j and i are present with correct ranges
        @test all(j -> j in [1, 2], ts_df.j)
        @test all(i -> i in [1, 2, 3], ts_df.i)
        
        # Test 3: Non-stocks should have values for ALL replicas
        # If bug exists, only i=1 will have non-NA flow values
        flow_data = subset(ts_df, :variable => ByRow(==("flow")))
        stock_data = subset(ts_df, :variable => ByRow(==("stock")))
        
        # Stocks should work fine for all combos and replicates
        stock_counts = combine(groupby(stock_data, [:j, :i]), nrow => :count)
        @test all(row -> row.count > 0, eachrow(stock_counts))
        
        # Non-stocks must also work for all combos and replicates
        # This will FAIL if the bug exists (only i=1 populated)
        flow_counts = combine(groupby(flow_data, [:j, :i]), nrow => :count)
        @test all(row -> row.count > 0, eachrow(flow_counts))
        
        # Test 4: No missing or NaN values in flows
        @test all(.!ismissing.(flow_data.value))
        @test all(.!isnan.(flow_data.value))
        
        # Test 5: Check that each (j, i) combination has expected coverage
        for combo_j in [1, 2]
            for replicate_i in [1, 2, 3]
                flow_subset = subset(flow_data, 
                    [:j, :i] => ByRow((j, i) -> j == combo_j && i == replicate_i))
                
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
                        j = 1,
                        i = i,
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
                j = [1, 1, 1, 1],
                i = [1, 2, 3, 4],
                time = [0.0, 0.0, 0.0, 0.0],
                variable = ["x", "x", "x", "x"],
                value = [10.0, NaN, 12.0, missing]
            )
            
            stats = ensemble_summ(timeseries_df)
            
            @test nrow(stats) == 1
            @test stats.mean[1] ≈ 11.0  # mean of [10.0, 12.0]
            @test stats.missing_count[1] == 2
        end

        @testset "All NaN values from failed ensemble" begin
            # All trajectories failed (represented as NaN)
            timeseries_df = DataFrame(
                j = [1, 1],
                i = [1, 2],
                time = [0.0, 0.0],
                variable = ["x", "x"],
                value = [NaN, NaN]
            )
            
            stats = ensemble_summ(timeseries_df)
            
            @test isnan(stats.mean[1])
            @test isnan(stats.median[1])
            @test isnan(stats.variance[1])
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
                    j = 1, i = i, time = 0.0, variable = "S",
                    value = sol.u[1][1]
                ))
                push!(timeseries_data, (
                    j = 1, i = i, time = 0.0, variable = "I",
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
                j = [1 for _ in 1:100],
                i = 1:100,
                time = [0.0 for _ in 1:100],
                variable = ["x" for _ in 1:100],
                value = values
            )
            
            stats = ensemble_summ(timeseries_df, [0.1, 0.5, 0.9])
            
            @test hasproperty(stats, :q1)   # 0.1 → q1
            @test hasproperty(stats, :q5)   # 0.5 → q5
            @test hasproperty(stats, :q9)   # 0.9 → q9
            
            @test stats.q1[1] < stats.q5[1]  # Lower quantile
            @test stats.q5[1] < stats.q9[1]  # Higher quantile
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
                            j = alpha_idx,
                            i = replicate,
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
            
            j1_stats = subset(stats, :j => ByRow(==(1)))
            j2_stats = subset(stats, :j => ByRow(==(2)))
            
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
            @test ts1.j == ts2.j
            @test ts1.i == ts2.i
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
                    j = 1, i = i, time = 0.0, variable = "S",
                    value = sol.u[1][1]
                ))
                push!(timeseries_data, (
                    j = 1, i = i, time = 0.0, variable = "I",
                    value = sol.u[1][2]
                ))
            end
            
            timeseries_df = DataFrame(timeseries_data)
            
            stats1 = ensemble_summ(timeseries_df)
            stats2 = ensemble_summ_threaded(timeseries_df)
            
            # Results should be identical (very small numerical differences possible)
            @test isapprox(stats1.mean, stats2.mean, rtol=1e-10)
            @test isapprox(stats1.median, stats2.median, rtol=1e-10)
            @test isapprox(stats1.variance, stats2.variance, rtol=1e-10)
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
            @test maximum(ts_df.j) == 4  # 4 parameter combinations
            @test maximum(ts_df.i) == 2  # 2 replicates
            
            # Compute summaries
            stats = ensemble_summ(ts_df)
            
            # Should have stats for each time point (excluding time=0.0 in some cases)
            @test nrow(stats) >= 1
            @test all(stats.missing_count .== 0)
            
            # Verify statistical measures make sense
            @test all(stats.mean .> 0)  # All means should be positive
            @test all(stats.variance .>= 0)  # Variance should be non-negative
        end
    end

end