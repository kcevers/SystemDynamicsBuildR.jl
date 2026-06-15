using Test
using SystemDynamicsBuildR.clean
using DataFrames

@testset "clean tests" begin

    @testset "saveat_func - interpolation" begin
        @testset "Linear interpolation" begin
            t = [0.0, 1.0, 2.0, 3.0]
            y = [0.0, 10.0, 20.0, 30.0]
            new_times = [0.5, 1.5, 2.5]

            result = saveat_func(t, y, new_times)

            @test result ≈ [5.0, 15.0, 25.0]
        end

        @testset "Extrapolation (nearest)" begin
            t = [1.0, 2.0, 3.0]
            y = [10.0, 20.0, 30.0]

            # Before range
            @test saveat_func(t, y, [0.0])[1] ≈ 10.0

            # After range
            @test saveat_func(t, y, [5.0])[1] ≈ 30.0
        end

        @testset "Single point" begin
            t = [1.0, 2.0, 3.0]
            y = [10.0, 20.0, 30.0]

            result = saveat_func(t, y, [2.0])
            @test result[1] ≈ 20.0
        end

        @testset "Multiple interpolations" begin
            t = [0.0, 1.0, 2.0]
            y = [0.0, 5.0, 10.0]
            new_times = [0.25, 0.5, 0.75, 1.25]

            result = saveat_func(t, y, new_times)
            @test result ≈ [1.25, 2.5, 3.75, 6.25]
        end
    end

    @testset "clean_df - basic functionality" begin
        @testset "Single variable, scalar parameter" begin
            prob = (
                p = 0.5,
                u0 = 10.0
            )

            solve_out = (
                t = [0.0, 1.0, 2.0],
                u = [10.0, 11.0, 12.0]
            )

            df, p_vals, p_names, u0_vals, u0_names = clean_df(
                prob, solve_out, [:x]
            )

            # Check DataFrame
            @test nrow(df) == 3
            @test df.time == [0.0, 1.0, 2.0]
            @test df.variable == ["x", "x", "x"]
            @test df.value == [10.0, 11.0, 12.0]

            # Check parameters
            @test p_names == ["p1"]
            @test p_vals == [0.5]

            # Check initial values
            @test u0_names == ["x"]
            @test u0_vals == [10.0]
        end

        @testset "Single variable, NamedTuple parameters" begin
            prob = (
                p = (alpha = 0.1, beta = 2.0),
                u0 = 10.0
            )

            solve_out = (
                t = [0.0, 1.0],
                u = [10.0, 11.0]
            )

            df, p_vals, p_names, u0_vals, u0_names = clean_df(
                prob, solve_out, [:x]
            )

            # Check parameters
            @test "alpha" in p_names
            @test "beta" in p_names
            @test 0.1 in p_vals
            @test 2.0 in p_vals
        end

        @testset "Multiple variables" begin
            prob = (
                p = (k = 0.5,),
                u0 = [10.0, 20.0]
            )

            solve_out = (
                t = [0.0, 1.0],
                u = [[10.0, 20.0], [11.0, 21.0]]
            )

            df, p_vals, p_names, u0_vals, u0_names = clean_df(
                prob, solve_out, [:x, :y]
            )

            # Check DataFrame
            @test nrow(df) == 4  # 2 times × 2 variables
            @test "x" in df.variable
            @test "y" in df.variable

            # Check x values
            x_data = subset(df, :variable => ByRow(==("x")))
            @test x_data.value == [10.0, 11.0]

            # Check y values
            y_data = subset(df, :variable => ByRow(==("y")))
            @test y_data.value == [20.0, 21.0]

            # Check initial values
            @test length(u0_vals) == 2
            @test u0_vals == [10.0, 20.0]
        end

        @testset "Vector parameters" begin
            prob = (
                p = [0.1, 0.5, 1.0],
                u0 = 10.0
            )

            solve_out = (
                t = [0.0, 1.0],
                u = [10.0, 11.0]
            )

            df, p_vals, p_names, u0_vals, u0_names = clean_df(
                prob, solve_out, [:x]
            )

            # Check parameter names
            @test p_names == ["p1", "p2", "p3"]
            @test p_vals == [0.1, 0.5, 1.0]
        end

        @testset "NamedTuple initial values" begin
            prob = (
                p = nothing,
                u0 = (S = 990.0, I = 10.0, R = 0.0)
            )

            solve_out = (
                t = [0.0, 1.0],
                u = [(S = 990.0, I = 10.0, R = 0.0),
                     (S = 985.0, I = 14.0, R = 1.0)]
            )

            df, p_vals, p_names, u0_vals, u0_names = clean_df(
                prob, solve_out, [:S, :I, :R]
            )

            # Check initial values
            @test u0_names == ["S", "I", "R"]
            @test u0_vals == [990.0, 10.0, 0.0]
        end

        @testset "With intermediaries" begin
            prob = (
                p = (k = 0.5,),
                u0 = 10.0
            )

            solve_out = (
                t = [0.0, 1.0, 2.0],
                u = [10.0, 11.0, 12.0]
            )

            intermediaries = (
                t = [0.0, 1.0, 2.0],
                saveval = [100.0, 110.0, 120.0]
            )

            df, p_vals, p_names, u0_vals, u0_names = clean_df(
                prob, solve_out, [:x], intermediaries, [:x_squared]
            )

            # Should have both main and intermediate variables
            @test "x" in df.variable
            @test "x_squared" in df.variable

            x_data = subset(df, :variable => ByRow(==("x")))
            x_sq_data = subset(df, :variable => ByRow(==("x_squared")))

            @test x_data.value == [10.0, 11.0, 12.0]
            @test x_sq_data.value == [100.0, 110.0, 120.0]
        end

        @testset "Multiple intermediate variables" begin
            prob = (
                p = nothing,
                u0 = [1.0, 2.0]
            )

            solve_out = (
                t = [0.0, 1.0],
                u = [[1.0, 2.0], [1.5, 2.5]]
            )

            intermediaries = (
                t = [0.0, 1.0],
                saveval = [[10.0, 20.0], [15.0, 25.0]]
            )

            df, _, _, _, _ = clean_df(
                prob, solve_out, [:x, :y],
                intermediaries, [:x_times_10, :y_times_10]
            )

            @test nrow(df) == 8  # 2 times × (2 state + 2 intermediate)
            @test "x_times_10" in df.variable
            @test "y_times_10" in df.variable
        end

        @testset "Empty intermediaries" begin
            prob = (p = nothing, u0 = 10.0)
            solve_out = (t = [0.0, 1.0], u = [10.0, 11.0])
            intermediaries = (t = Float64[], saveval = Float64[])

            df, _, _, _, _ = clean_df(
                prob, solve_out, [:x], intermediaries, [:int]
            )

            # Should only have main variable
            @test all(df.variable .== "x")
            @test nrow(df) == 2
        end
    end

    @testset "clean_constants" begin
        @testset "Basic filtering" begin
            constants = (
                a = 1.0,
                b = 2.0,
                c = [1.0, 2.0, 3.0]
            )

            result = clean_constants(constants)

            @test haskey(result, :a)
            @test haskey(result, :b)
            @test haskey(result, :c)
            @test result.a == 1.0
            @test result.b == 2.0
            @test result.c == [1.0, 2.0, 3.0]
        end

        @testset "Filter non-numeric types" begin
            constants = (
                a = 1.0,
                b = "string",      # Should be filtered
                c = sin,           # Should be filtered
                d = [1.0, 2.0],
                e = :symbol,       # Should be filtered
                f = 2.0
            )

            result = clean_constants(constants)

            @test haskey(result, :a)
            @test !haskey(result, :b)  # String filtered
            @test !haskey(result, :c)  # Function filtered
            @test haskey(result, :d)
            @test !haskey(result, :e)  # Symbol filtered
            @test haskey(result, :f)
        end

        @testset "Empty NamedTuple" begin
            constants = NamedTuple()
            result = clean_constants(constants)
            @test isempty(result)
        end

        @testset "All filtered out" begin
            constants = (
                a = "string",
                b = sin,
                c = :symbol
            )

            result = clean_constants(constants)
            @test isempty(result)
        end
    end

    @testset "clean_init" begin
        @testset "Basic usage" begin
            init = [10.0, 20.0, 30.0]
            init_names = [:S, :I, :R]

            result = clean_init(init, init_names)

            @test result isa Dict
            @test result[:S] == 10.0
            @test result[:I] == 20.0
            @test result[:R] == 30.0
        end

        @testset "String names" begin
            init = [1.0, 2.0]
            init_names = ["x", "y"]

            result = clean_init(init, init_names)

            @test result["x"] == 1.0
            @test result["y"] == 2.0
        end

        @testset "Single value" begin
            init = [42.0]
            init_names = [:answer]

            result = clean_init(init, init_names)

            @test length(result) == 1
            @test result[:answer] == 42.0
        end

        @testset "Integer values" begin
            init = [1, 2, 3]
            init_names = [:a, :b, :c]

            result = clean_init(init, init_names)

            # Should convert to Float64
            @test result[:a] == 1.0
            @test result[:b] == 2.0
            @test result[:c] == 3.0
        end
    end

    @testset "Integration tests" begin
        @testset "Full workflow" begin
            prob = (
                p = (birth_rate = 0.1, death_rate = 0.05),
                u0 = [1000.0, 100.0]
            )

            solve_out = (
                t = [0.0, 1.0, 2.0],
                u = [
                    [1000.0, 100.0],
                    [1050.0, 105.0],
                    [1100.0, 110.0]
                ]
            )

            df, p_vals, p_names, u0_vals, u0_names = clean_df(
                prob, solve_out, [:population_a, :population_b]
            )

            # Check structure
            @test nrow(df) == 6  # 3 times × 2 variables
            @test length(p_vals) == 2
            @test length(u0_vals) == 2

            # All values should be Float64
            @test all(x -> isa(x, Float64), df.time)
            @test all(x -> isa(x, Float64), df.value)
            @test all(x -> isa(x, Float64), p_vals)
            @test all(x -> isa(x, Float64), u0_vals)

            # Clean constants
            constants = (
                birth_rate = 0.1,
                death_rate = 0.05,
                name = "test"  # Should be filtered
            )
            clean_const = clean_constants(constants)

            @test haskey(clean_const, :birth_rate)
            @test haskey(clean_const, :death_rate)
            @test !haskey(clean_const, :name)

            # Clean initial conditions
            init_dict = clean_init(prob.u0, [:population_a, :population_b])

            @test init_dict[:population_a] == 1000.0
            @test init_dict[:population_b] == 100.0
        end

        @testset "Workflow with intermediaries" begin
            prob = (
                p = (α = 0.5, β = 0.3),
                u0 = [10.0, 20.0]
            )

            solve_out = (
                t = [0.0, 1.0],
                u = [[10.0, 20.0], [12.0, 22.0]]
            )

            intermediaries = (
                t = [0.0, 1.0],
                saveval = [[30.0, 40.0], [34.0, 44.0]]
            )

            df, p_vals, p_names, u0_vals, u0_names = clean_df(
                prob, solve_out, [:x, :y],
                intermediaries, [:sum, :product]
            )

            # Check all variable types present
            @test "x" in df.variable
            @test "y" in df.variable
            @test "sum" in df.variable
            @test "product" in df.variable

            # Verify values
            sum_data = subset(df, :variable => ByRow(==("sum")))
            @test sum_data.value == [30.0, 34.0]
        end

        @testset "Interpolation workflow" begin
            # Original solution
            t = [0.0, 1.0, 2.0, 3.0, 4.0]
            y = [0.0, 10.0, 20.0, 30.0, 40.0]

            # Interpolate at finer resolution
            new_times = 0.0:0.5:4.0
            interpolated = saveat_func(t, y, collect(new_times))

            @test length(interpolated) == length(new_times)
            @test interpolated[1] ≈ 0.0   # t=0.0
            @test interpolated[3] ≈ 10.0  # t=1.0
            @test interpolated[2] ≈ 5.0   # t=0.5 (interpolated)
        end
    end

    @testset "Edge cases and error handling" begin
        @testset "Empty solution" begin
            prob = (p = nothing, u0 = 10.0)
            solve_out = (t = Float64[], u = Float64[])

            df, p_vals, p_names, u0_vals, u0_names = clean_df(
                prob, solve_out, [:x]
            )

            @test nrow(df) == 0
            @test isempty(df.time)
        end

        @testset "No parameters" begin
            prob = (p = nothing, u0 = 10.0)
            solve_out = (t = [0.0, 1.0], u = [10.0, 11.0])

            df, p_vals, p_names, u0_vals, u0_names = clean_df(
                prob, solve_out, [:x]
            )

            @test isempty(p_vals)
            @test isempty(p_names)
        end

        @testset "Very large values" begin
            prob = (p = nothing, u0 = 1e10)
            solve_out = (t = [0.0, 1.0], u = [1e10, 1e11])

            df, _, _, _, _ = clean_df(prob, solve_out, [:x])

            @test df.value[1] ≈ 1e10
            @test df.value[2] ≈ 1e11
        end

        @testset "Very small values" begin
            prob = (p = nothing, u0 = 1e-10)
            solve_out = (t = [0.0, 1.0], u = [1e-10, 1e-11])

            df, _, _, _, _ = clean_df(prob, solve_out, [:x])

            @test df.value[1] ≈ 1e-10
            @test df.value[2] ≈ 1e-11
        end

        @testset "Negative values" begin
            prob = (p = -0.5, u0 = -10.0)
            solve_out = (t = [0.0, 1.0], u = [-10.0, -11.0])

            df, p_vals, _, u0_vals, _ = clean_df(prob, solve_out, [:x])

            @test df.value == [-10.0, -11.0]
            @test p_vals == [-0.5]
            @test u0_vals == [-10.0]
        end

        @testset "Zero values" begin
            prob = (p = 0.0, u0 = 0.0)
            solve_out = (t = [0.0, 1.0], u = [0.0, 0.0])

            df, p_vals, _, u0_vals, _ = clean_df(prob, solve_out, [:x])

            @test all(df.value .== 0.0)
            @test p_vals == [0.0]
            @test u0_vals == [0.0]
        end

        @testset "Single time point" begin
            prob = (p = nothing, u0 = 10.0)
            solve_out = (t = [0.0], u = [10.0])

            df, _, _, _, _ = clean_df(prob, solve_out, [:x])

            @test nrow(df) == 1
            @test df.time == [0.0]
            @test df.value == [10.0]
        end

        @testset "Many variables" begin
            n_vars = 10
            prob = (p = nothing, u0 = collect(1.0:10.0))
            solve_out = (
                t = [0.0, 1.0],
                u = [collect(1.0:10.0), collect(2.0:11.0)]
            )

            var_names = [Symbol("x$i") for i in 1:n_vars]
            df, _, _, _, _ = clean_df(prob, solve_out, var_names)

            @test nrow(df) == 20  # 2 times × 10 variables
            @test length(unique(df.variable)) == n_vars
        end
    end

end
