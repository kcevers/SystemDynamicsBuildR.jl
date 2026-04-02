using SystemDynamicsBuildR.custom_func
using Unitful
using DataInterpolations

@testset "custom_func tests" begin

    @testset "Type checking utilities" begin
        @testset "is_function_or_interp" begin
            @test is_function_or_interp(sin) == true
            @test is_function_or_interp(cos) == true
            @test is_function_or_interp(5) == false
            @test is_function_or_interp("string") == false
            
            # Test with interpolation object
            f = itp([1.0, 2.0], [3.0, 4.0])
            @test is_function_or_interp(f) == true
        end
    end

    @testset "Interpolation functions" begin
        @testset "itp - linear interpolation" begin
            f = itp([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
            @test f(1.0) ≈ 10.0
            @test f(1.5) ≈ 15.0
            @test f(2.0) ≈ 20.0
            @test f(2.5) ≈ 25.0
            @test f(3.0) ≈ 30.0
            
            # Test extrapolation (nearest by default)
            @test f(0.0) ≈ 10.0  # Before range
            @test f(4.0) ≈ 30.0  # After range
        end

        @testset "itp - automatic sorting" begin
            # Provide unsorted x values
            f = itp([3.0, 1.0, 2.0], [30.0, 10.0, 20.0])
            @test f(1.5) ≈ 15.0
            @test f(2.5) ≈ 25.0
        end

        @testset "itp - constant interpolation" begin
            f = itp([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], method="constant")
            @test f(1.0) ≈ 10.0
            @test f(1.9) ≈ 10.0  # Should hold previous value
            @test f(2.0) ≈ 20.0
        end

        @testset "itp - invalid method" begin
            @test_throws ArgumentError itp([1.0, 2.0], [3.0, 4.0], method="invalid")
        end
    end

    @testset "Signal generation" begin
        @testset "ramp - basic functionality" begin
            times = [0.0, 10.0]
            r = make_ramp(u"s", times, 2.0, 6.0, 10.0)
            
            @test r(0.0) ≈ 0.0   # Before ramp
            @test r(2.0) ≈ 0.0   # Start of ramp
            @test r(4.0) ≈ 5.0   # Middle of ramp
            @test r(6.0) ≈ 10.0  # End of ramp
            @test r(8.0) ≈ 10.0  # After ramp

            # Also works with start time after the end of times
            r2 = make_ramp(u"s", times, 15.0, 20.0, 10.0)
            @test r2(10.0) ≈ 0.0
            @test r2(20.0) ≈ 0.0

            # Also works with start time before the start of times
            r3 = make_ramp(u"s", times, -5.0, 5.0, 10.0)
            @test r3(0.0) ≈ 5.0
            @test r3(2.5) ≈ 7.5
            @test r3(5.0) ≈ 10.0
            @test r3(7.5) ≈ 10.0

            # Also works with start time before the start of times
            r4 = make_ramp(u"s", times, -5.0, -1.0, 10.0)
            @test r4(0.0) ≈ 10.0
            @test r4(10.0) ≈ 10.0
            
        end

        @testset "ramp - negative height" begin
            times = [0.0, 10.0]
            r = make_ramp(u"s", times, 2.0, 6.0, -10.0)
            
            @test r(4.0) ≈ -5.0  # Decreasing ramp
        end

        @testset "ramp - assertions" begin
            times = [0.0, 10.0]
            @test_throws AssertionError make_ramp(u"s", times, 6.0, 2.0, 10.0)  # finish < start
        end

        @testset "make_step - basic functionality" begin
            times = [0.0, 10.0]
            s = make_step(u"s", times, 5.0, 2.0)
            
            @test s(4.9) ≈ 0.0  # Before step
            @test s(5.0) ≈ 2.0  # At step
            @test s(5.1) ≈ 2.0  # After step
            @test s(10.0) ≈ 2.0

            # Also works with start time after the end of times
            s2 = make_step(u"s", times, 15.0, 3.0)
            @test s2(10.0) ≈ 0.0    

            # Also works with start time before the start of times
            s3 = make_step(u"s", times, -5.0, 4.0)
            @test s3(0.0) ≈ 4.0
            @test s3(5.0) ≈ 4.0

            # Works with negative height
            s4 = make_step(u"s", times, 5.0, -2.0)
            @test s4(5.0) ≈ -2.0

        end

        @testset "pulse - single pulse" begin
            times = [0.0, 20.0]
            p = make_pulse(u"s", times, 5.0, 1.0, 2.0)
            
            @test p(4.0) ≈ 0.0  # Before pulse
            @test p(5.5) ≈ 1.0  # During pulse
            @test p(6.5) ≈ 1.0  # Still during pulse
            @test p(7.1) ≈ 0.0  # After pulse

            # Repeat pulse
            p_r = make_pulse(u"s", times, 1.0, 1.0, 0.5, 2.0)
            @test p_r(0.5) ≈ 0.0
            @test p_r(1.0) ≈ 1.0
            @test p_r(1.4) ≈ 1.0
            @test p_r(1.6) ≈ 0.0
            @test p_r(3.0) ≈ 1.0
            @test p_r(3.4) ≈ 1.0
            @test p_r(3.6) ≈ 0.0

            # Edge cases
            # Also works with pulse starting before the time range
            p2 = make_pulse(u"s", times, -5.0, 1.0, 2.0)
            @test p2(-5.0) ≈ 1.0
            @test p2(-3.0) ≈ 0.0
            @test p2(0.0) ≈ 0.0
            @test p2(10.0) ≈ 0.0

            # Also works with pulse starting after the time range
            p3 = make_pulse(u"s", times, 25.0, 1.0, 2.0)
            @test p3(20.0) ≈ 0.0

            # Also works if width is equal to repeat_interval
            p4 = make_pulse(u"s", times, 5.0, 2.0, 1.0, 1.0)
            @test p4(0.0) ≈ 0.0
            @test p4(5.0) ≈ 2.0
            @test p4(20.0) ≈ 2.0

        end

        @testset "pulse - repeated pulses" begin
            times = [0.0, 30.0]
            p = make_pulse(u"s", times, 5.0, 1.0, 2.0, 10.0)  # Every 10 seconds
            
            @test p(5.5) ≈ 1.0   # First pulse
            @test p(8.0) ≈ 0.0   # Between pulses
            @test p(15.5) ≈ 1.0  # Second pulse
            @test p(25.5) ≈ 1.0  # Third pulse
        end

        @testset "pulse - width validation" begin
            times = [0.0, 20.0]
            @test_throws ArgumentError make_pulse(u"s", times, 5.0, 1.0, 0.0)
            @test_throws ArgumentError make_pulse(u"s", times, 5.0, 1.0, -1.0)
        end

        @testset "signal normalization with unitless timeline" begin
            times = [0.0, 10.0]

            ramp = make_ramp(u"s", times, 2.0u"s", 6.0u"s", 10.0)
            @test !(ramp(4.0) isa Unitful.Quantity)
            @test ramp(0.0) ≈ 0.0
            @test ramp(2.0) ≈ 0.0
            @test ramp(4.0) ≈ 5.0
            @test ramp(6.0) ≈ 10.0
            @test ramp(8.0) ≈ 10.0

            step = make_step(u"s", times, 5000.0u"ms", 2.0)
            @test !(step(4.0) isa Unitful.Quantity)
            @test step(4.9) ≈ 0.0
            @test step(5.0) ≈ 2.0
            @test step(9.0) ≈ 2.0

            pulse = make_pulse(u"s", [0.0, 20.0], 5.0u"s", 1.0, 2.0u"s", 10.0u"s")
            @test !(pulse(5.5) isa Unitful.Quantity)
            @test pulse(4.0) ≈ 0.0
            @test pulse(5.5) ≈ 1.0
            @test pulse(7.1) ≈ 0.0
            @test pulse(15.5) ≈ 1.0
        end

        @testset "signal normalization helpers" begin
            times = [0.0, 10.0]

            start_unitless, finish_unitless = SystemDynamicsBuildR.custom_func._normalize_time_units(
                times, u"s", 2.0, 6.0
            )
            @test start_unitless == 2.0
            @test finish_unitless == 6.0

            start_unitful, finish_unitful = SystemDynamicsBuildR.custom_func._normalize_time_units(
                times, u"s", 2.0u"hr", 6000.0u"ms"
            )
            @test start_unitful == 7200.0
            @test finish_unitful == 6.0

            pulse_start, pulse_width, pulse_repeat = SystemDynamicsBuildR.custom_func._normalize_pulse_units(
                times, u"s", 1.5u"hr", 500.0u"ms", 2.0u"hr"
            )
            @test pulse_start == 5400.0
            @test pulse_width == 0.5
            @test pulse_repeat == 7200.0
        end

        @testset "seasonal - basic wave" begin
            times = [0.0u"yr", 2.0u"yr"]
            wave = make_seasonal(0.1u"yr", times, 1.0u"yr", 0.0u"yr")
            
            @test wave(0.0u"yr") ≈ 1.0   # Peak
            @test wave(0.25u"yr") ≈ 0.0 atol=0.01  # Zero crossing
            @test wave(0.5u"yr") ≈ -1.0  # Trough
            @test wave(0.75u"yr") ≈ 0.0 atol=0.01   # Zero crossing
            @test wave(1.0u"yr") ≈ 1.0   # Back to peak
        end

        @testset "seasonal - with phase shift" begin
            times = [0.0u"yr", 2.0u"yr"]
            wave = make_seasonal(0.1u"yr", times, 1.0u"yr", 0.25u"yr")
            
            @test wave(0.0u"yr") ≈ 0.0 atol=0.01  # Shifted by quarter period
            @test wave(0.25u"yr") ≈ 1.0 atol=0.1  # Peak at shifted position (with tolerance for sampling)
        end

        @testset "seasonal - period validation" begin
            times = [0.0u"yr", 2.0u"yr"]
            @test_throws AssertionError make_seasonal(0.1u"yr", times, 0.0u"yr")
            @test_throws AssertionError make_seasonal(0.1u"yr", times, -1.0u"yr")
        end
    end

    @testset "Mathematical functions" begin
        @testset "round_IM - Insight Maker rounding" begin
            # Test 0.5 rounds up (unlike Julia's default)
            @test round_IM(0.5) == 1.0
            @test round_IM(1.5) == 2.0
            @test round_IM(2.5) == 3.0
            
            # Negative numbers
            @test round_IM(-0.5) == 0.0
            @test round_IM(-1.5) == -1.0
            
            # With digits
            @test round_IM(3.14159, 2) == 3.14
            @test round_IM(3.145, 2) == 3.15  # 0.5 rounds up
            @test round_IM(2.675, 2) ≈ 2.68 atol=0.01
        end

        @testset "logit function" begin
            @test logit(0.5) ≈ 0.0
            @test logit(0.75) ≈ log(3)
            @test logit(0.25) ≈ log(1/3)
            @test logit(0.9) ≈ log(9)
        end

        @testset "expit function" begin
            @test expit(0.0) ≈ 0.5
            @test expit(10.0) ≈ 1.0 atol=0.001
            @test expit(-10.0) ≈ 0.0 atol=0.001
            
            # Test that expit is inverse of logit
            @test expit(logit(0.7)) ≈ 0.7
            @test expit(logit(0.3)) ≈ 0.3
        end

        @testset "logistic function" begin
            # Standard logistic
            @test logistic(0.0) ≈ 0.5
            @test logistic(0.0, 1.0, 0.0, 1.0) ≈ 0.5
            
            # With different parameters
            @test logistic(5.0, 1.0, 5.0, 1.0) ≈ 0.5  # At midpoint
            @test logistic(10.0, 1.0, 5.0, 10.0) ≈ 10.0 atol=0.1  # Near upper bound
            
            # Steeper slope
            @test logistic(5.0, 2.0, 5.0, 1.0) ≈ 0.5
            @test logistic(5.5, 2.0, 5.0, 1.0) > logistic(5.5, 1.0, 5.0, 1.0)  # Steeper
            
            # Assertions for infinite values
            @test_throws AssertionError logistic(0.0, Inf, 0.0, 1.0)
            @test_throws AssertionError logistic(0.0, 1.0, Inf, 1.0)
            @test_throws AssertionError logistic(0.0, 1.0, 0.0, Inf)
        end

        @testset "nonnegative - scalars" begin
            @test nonnegative(5.0) == 5.0
            @test nonnegative(-3.0) == 0.0
            @test nonnegative(0.0) == 0.0
            @test nonnegative(1.5) == 1.5
        end

        @testset "nonnegative - arrays" begin
            @test nonnegative([-1.0, 2.0, -3.0, 4.0]) == [0.0, 2.0, 0.0, 4.0]
            @test nonnegative([5.0, 10.0]) == [5.0, 10.0]
        end

        @testset "nonnegative - with units" begin
            @test nonnegative(-5.0u"m") == 0.0u"m"
            @test nonnegative(3.0u"m") == 3.0u"m"
            @test nonnegative([-2.0u"kg", 5.0u"kg"]) == [0.0u"kg", 5.0u"kg"]
        end
    end

    @testset "Rounding utilities" begin
        @testset "round_ - basic" begin
            @test round_(3.14159, 2) ≈ 3.14
            @test round_(3.14159, 0) ≈ 3.0
            @test round_(3.6) ≈ 4.0
            
            # Using keyword argument
            @test round_(3.14159, digits=2) ≈ 3.14
            @test round_(3.14159; digits=3) ≈ 3.142
        end

        @testset "round_ - with units" begin
            @test round_(3.14159u"m", 2) ≈ 3.14u"m"
            @test round_(5.6u"kg", 0) ≈ 6.0u"kg"
            @test round_(5.6u"kg"; digits=0) ≈ 6.0u"kg"
        end

        @testset "round_ - float digits" begin
            # Should handle float digits by converting to Int
            @test round_(3.14159, 2.0) ≈ 3.14
            @test round_(3.14159; digits=2.0) ≈ 3.14
        end
    end

    @testset "Random sampling functions" begin
        @testset "rbool" begin
            # Deterministic tests
            @test rbool(0.0) == false
            @test rbool(1.0) == true
            
            # Probabilistic test (may rarely fail due to randomness)
            results = [rbool(0.5) for _ in 1:1000]
            true_count = sum(results)
            @test 400 < true_count < 600  # Should be around 500
        end

        @testset "rdist" begin
            # Equal probabilities
            values = ["a", "b", "c"]
            probs = [1.0, 1.0, 1.0]
            results = [rdist(values, probs) for _ in 1:300]
            @test all(r in values for r in results)
            
            # Weighted probabilities (deterministic when one is 1, others 0)
            @test rdist([1, 2, 3], [0.0, 1.0, 0.0]) == 2
            
            # Normalized automatically
            @test rdist([1, 2], [2.0, 2.0]) in [1, 2]
        end

         @testset "rdist - error handling" begin
            @test_throws ArgumentError rdist([1, 2], [0.5, 0.5, 0.5])  # Length mismatch
            @test_throws ArgumentError rdist([1, 2], [0.0, 0.0])  # All zero probs
            @test_throws ArgumentError rdist([1, 2], [-1.0, 2.0])  # Negative prob
            @test_throws ArgumentError rdist([1, 2], [1.0, -0.5])  # Negative prob
        end
    end

    @testset "String and array utilities" begin
        @testset "indexof - strings" begin
            @test indexof("hello world", "world") == 7
            @test indexof("hello", "ll") == 3
            @test indexof("hello", "h") == 1
            @test indexof("hello", "x") == 0  # Not found
            @test indexof("hello", "hello") == 1
        end

        @testset "indexof - arrays" begin
            @test indexof([1, 2, 3, 4], 3) == 3
            @test indexof([1, 2, 3, 4], 1) == 1
            @test indexof([1, 2, 3, 4], 5) == 0  # Not found
            @test indexof(["a", "b", "c"], "b") == 2
        end

        @testset "contains_IM - strings" begin
            @test contains_IM("hello world", "world") == true
            @test contains_IM("hello world", "hello") == true
            @test contains_IM("hello world", "llo wor") == true
            @test contains_IM("hello", "x") == false
        end

        @testset "contains_IM - arrays" begin
            @test contains_IM([1, 2, 3, 4], 3) == true
            @test contains_IM([1, 2, 3, 4], 1) == true
            @test contains_IM([1, 2, 3, 4], 5) == false
            @test contains_IM(["a", "b", "c"], "b") == true
        end
    end

    @testset "Operators" begin
        @testset "modulus operator ⊕" begin
            @test (7 ⊕ 3) == 1
            @test (10 ⊕ 5) == 0
            @test (15 ⊕ 4) == 3
            @test (5.5 ⊕ 2.0) ≈ 1.5
            
            # Should be same as mod()
            @test (17 ⊕ 5) == mod(17, 5)
            @test (23.7 ⊕ 4.2) ≈ mod(23.7, 4.2)
        end
    end

    @testset "Integration tests" begin
        @testset "Signal generation with units" begin
            # Test ramp with Unitful times
            times = [0.0u"yr", 10.0u"yr"]
            r = make_ramp(u"yr", times, 2.0u"yr", 6.0u"yr", 100.0)
            @test r(4.0u"yr") ≈ 50.0
            
            # Test step with units
            s = make_step(u"yr", times, 5.0u"yr", 10.0)
            @test s(4.0u"yr") ≈ 0.0
            @test s(6.0u"yr") ≈ 10.0
        end

        @testset "Chaining mathematical functions" begin
            # Test that logit and expit are inverses
            for p in [0.1, 0.3, 0.5, 0.7, 0.9]
                @test expit(logit(p)) ≈ p
            end
            
            # Test logistic and nonnegative
            @test nonnegative(logistic(-10.0)) ≈ logistic(-10.0)
            @test nonnegative(logistic(0.0) - 1.0) ≈ 0.0
        end

        @testset "Interpolation of signal" begin
            # Create a ramp and sample it with interpolation
            times = [0.0, 10.0]
            r = make_ramp(u"s", times, 0.0, 10.0, 100.0)
            
            # Sample at specific points
            sample_times = [0.0, 2.5, 5.0, 7.5, 10.0]
            sample_values = [r(t) for t in sample_times]
            
            # Create interpolation of sampled values
            f = itp(sample_times, sample_values)
            
            # Should match original ramp at sample points
            @test f(5.0) ≈ r(5.0)
            @test f(7.5) ≈ r(7.5)
        end
    end

end