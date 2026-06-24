using SystemDynamicsBuildR.custom_func

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

        @testset "itp - extrapolation aliases and extra modes" begin
            # "nearest" is an alias for constant extrapolation
            f = itp([1.0, 3.0], [10.0, 30.0], extrapolation="nearest")
            @test f(0.0) ≈ 10.0   # below range → clamp to first y
            @test f(4.0) ≈ 30.0   # above range → clamp to last y

            # "NA" is an alias for missing extrapolation
            f_na = itp([1.0, 3.0], [10.0, 30.0], extrapolation="NA")
            @test ismissing(f_na(0.0))
            @test ismissing(f_na(4.0))
            @test f_na(2.0) ≈ 20.0  # within range still works

            # invalid extrapolation string
            @test_throws ArgumentError itp([1.0, 2.0], [3.0, 4.0], extrapolation="invalid")
        end

        @testset "itp - linear extrapolation" begin
            # slope = (30-10)/(3-1) = 10 per unit
            f = itp([1.0, 3.0], [10.0, 30.0], extrapolation="linear")
            @test f(0.0) ≈ 0.0    # left: 10 + 10*(0-1) = 0
            @test f(4.0) ≈ 40.0   # right: 30 + 10*(4-3) = 40
            @test f(2.0) ≈ 20.0   # within range unaffected

            # missing extrapolation
            f_miss = itp([1.0, 3.0], [10.0, 30.0], extrapolation="missing")
            @test ismissing(f_miss(0.0))
            @test ismissing(f_miss(4.0))
            @test f_miss(2.0) ≈ 20.0

            # error extrapolation
            f_err = itp([1.0, 3.0], [10.0, 30.0], extrapolation="error")
            @test_throws ErrorException f_err(0.0)
            @test_throws ErrorException f_err(4.0)
            @test f_err(2.0) ≈ 20.0
        end

        @testset "itp - constant interpolation extrapolation" begin
            # missing extrapolation with constant method
            f_miss = itp([1.0, 3.0], [10.0, 30.0], method="constant", extrapolation="missing")
            @test ismissing(f_miss(0.0))   # below range
            @test f_miss(1.0) ≈ 10.0      # at left boundary
            @test f_miss(2.0) ≈ 10.0      # within range (step holds)

            # error extrapolation with constant method
            f_err = itp([1.0, 3.0], [10.0, 30.0], method="constant", extrapolation="error")
            @test_throws ErrorException f_err(0.0)  # below range
            @test f_err(1.5) ≈ 10.0                # within range
        end

        @testset "Interpolator - constructor validation" begin
            Interpolator = SystemDynamicsBuildR.custom_func.Interpolator

            # length mismatch
            @test_throws ArgumentError Interpolator([1.0, 2.0], [3.0])

            # unsorted x (itp sorts before calling Interpolator, so must test directly)
            @test_throws ArgumentError Interpolator([2.0, 1.0], [10.0, 20.0])

            # constant method + linear extrapolation is incompatible
            @test_throws ErrorException itp([1.0, 2.0], [3.0, 4.0],
                method="constant", extrapolation="linear")
        end

        @testset "Interpolator - array input" begin
            f = itp([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
            result = f([1.0, 1.5, 2.0, 2.5, 3.0])
            @test result ≈ [10.0, 15.0, 20.0, 25.0, 30.0]
        end
    end

    @testset "Signal generation" begin
        @testset "ramp - basic functionality" begin
            times = [0.0, 10.0]
            r = make_ramp(times, 2.0, 6.0, 10.0)

            @test r(0.0) ≈ 0.0   # Before ramp
            @test r(2.0) ≈ 0.0   # Start of ramp
            @test r(4.0) ≈ 5.0   # Middle of ramp
            @test r(6.0) ≈ 10.0  # End of ramp
            @test r(8.0) ≈ 10.0  # After ramp

            # Also works with start time after the end of times
            r2 = make_ramp(times, 15.0, 20.0, 10.0)
            @test r2(10.0) ≈ 0.0
            @test r2(20.0) ≈ 0.0

            # Also works with start time before the start of times
            r3 = make_ramp(times, -5.0, 5.0, 10.0)
            @test r3(0.0) ≈ 5.0
            @test r3(2.5) ≈ 7.5
            @test r3(5.0) ≈ 10.0
            @test r3(7.5) ≈ 10.0

            # Also works with finish before start of times
            r4 = make_ramp(times, -5.0, -1.0, 10.0)
            @test r4(0.0) ≈ 10.0
            @test r4(10.0) ≈ 10.0
        end

        @testset "ramp - negative height" begin
            times = [0.0, 10.0]
            r = make_ramp(times, 2.0, 6.0, -10.0)

            @test r(4.0) ≈ -5.0  # Decreasing ramp
        end

        @testset "ramp - assertions" begin
            times = [0.0, 10.0]
            @test_throws AssertionError make_ramp(times, 6.0, 2.0, 10.0)  # finish < start
        end

        @testset "make_step - basic functionality" begin
            times = [0.0, 10.0]
            s = make_step(times, 5.0, 2.0)

            @test s(4.9) ≈ 0.0  # Before step
            @test s(5.0) ≈ 2.0  # At step
            @test s(5.1) ≈ 2.0  # After step
            @test s(10.0) ≈ 2.0

            # Also works with start time after the end of times
            s2 = make_step(times, 15.0, 3.0)
            @test s2(10.0) ≈ 0.0

            # Also works with start time before the start of times
            s3 = make_step(times, -5.0, 4.0)
            @test s3(0.0) ≈ 4.0
            @test s3(5.0) ≈ 4.0

            # Works with negative height
            s4 = make_step(times, 5.0, -2.0)
            @test s4(5.0) ≈ -2.0
        end

        @testset "pulse - single pulse" begin
            times = [0.0, 20.0]
            p = make_pulse(times, 5.0, 1.0, 2.0)

            @test p(4.0) ≈ 0.0  # Before pulse
            @test p(5.5) ≈ 1.0  # During pulse
            @test p(6.5) ≈ 1.0  # Still during pulse
            @test p(7.1) ≈ 0.0  # After pulse

            # Repeat pulse
            p_r = make_pulse(times, 1.0, 1.0, 0.5, 2.0)
            @test p_r(0.5) ≈ 0.0
            @test p_r(1.0) ≈ 1.0
            @test p_r(1.4) ≈ 1.0
            @test p_r(1.6) ≈ 0.0
            @test p_r(3.0) ≈ 1.0
            @test p_r(3.4) ≈ 1.0
            @test p_r(3.6) ≈ 0.0

            # Also works with pulse starting before the time range
            p2 = make_pulse(times, -5.0, 1.0, 2.0)
            @test p2(-5.0) ≈ 1.0
            @test p2(-3.0) ≈ 0.0
            @test p2(0.0) ≈ 0.0
            @test p2(10.0) ≈ 0.0

            # Also works with pulse starting after the time range
            p3 = make_pulse(times, 25.0, 1.0, 2.0)
            @test p3(20.0) ≈ 0.0

            # Also works if width is equal to repeat_interval
            p4 = make_pulse(times, 5.0, 2.0, 1.0, 1.0)
            @test p4(0.0) ≈ 0.0
            @test p4(5.0) ≈ 2.0
            @test p4(20.0) ≈ 2.0
        end

        @testset "pulse - repeated pulses" begin
            times = [0.0, 30.0]
            p = make_pulse(times, 5.0, 1.0, 2.0, 10.0)  # Every 10 seconds

            @test p(5.5) ≈ 1.0   # First pulse
            @test p(8.0) ≈ 0.0   # Between pulses
            @test p(15.5) ≈ 1.0  # Second pulse
            @test p(25.5) ≈ 1.0  # Third pulse
        end

        @testset "pulse - width validation" begin
            times = [0.0, 20.0]
            @test_throws ArgumentError make_pulse(times, 5.0, 1.0, 0.0)
            @test_throws ArgumentError make_pulse(times, 5.0, 1.0, -1.0)
        end

        @testset "seasonal - basic wave" begin
            # Signature mirrors R seasonal(times, period, shift); the converter
            # emits make_seasonal(times, period, shift) where times is the
            # (start, stop) tuple. The closure is exact (no dt / sampling).
            times = (0.0, 2.0)
            wave = make_seasonal(times, 1.0, 0.0)

            @test wave(0.0) ≈ 1.0   # Peak
            @test wave(0.25) ≈ 0.0 atol=0.01  # Zero crossing
            @test wave(0.5) ≈ -1.0  # Trough
            @test wave(0.75) ≈ 0.0 atol=0.01  # Zero crossing
            @test wave(1.0) ≈ 1.0   # Back to peak
        end

        @testset "seasonal - with phase shift" begin
            times = (0.0, 2.0)
            wave = make_seasonal(times, 1.0, 0.25)

            @test wave(0.0) ≈ 0.0 atol=0.01   # Shifted by quarter period
            @test wave(0.25) ≈ 1.0 atol=0.1   # Peak at shifted position
        end

        @testset "seasonal - period validation" begin
            times = (0.0, 2.0)
            @test_throws AssertionError make_seasonal(times, 0.0)
            @test_throws AssertionError make_seasonal(times, -1.0)
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

        @testset "hill function" begin
            # At midpoint, output is always upper/2
            @test hill(1.0, 1.0, 1.0, 1.0) ≈ 0.5
            @test hill(2.0, 1.0, 2.0, 10.0) ≈ 5.0
            @test hill(2.0, 2.0, 2.0, 1.0) ≈ 0.5

            # Increasing x beyond midpoint approaches upper
            @test hill(100.0, 1.0, 1.0, 1.0) ≈ 1.0 atol=0.01

            # Higher slope → steeper transition
            @test hill(3.0, 4.0, 2.0, 1.0) > hill(3.0, 1.0, 2.0, 1.0)

            # Upper parameter scales the output
            @test hill(1.0, 1.0, 1.0, 5.0) ≈ 2.5

            # Assertions for infinite values
            @test_throws AssertionError hill(1.0, Inf, 1.0, 1.0)
            @test_throws AssertionError hill(1.0, 1.0, Inf, 1.0)
            @test_throws AssertionError hill(1.0, 1.0, 1.0, Inf)
        end

        @testset "ricker function" begin
            # NOTE: ricker uses POSITIONAL args (location, upper, shape, a, b) to
            # match the R signature order, because sdbuildR's R->Julia translation
            # drops argument names and emits positional calls (e.g. ricker(x, 2, 10, 1)).

            # Peaks at x = location with value upper
            @test ricker(1, 1, 1) ≈ 1 atol=1e-9
            @test ricker(2, 2, 5) ≈ 5 atol=1e-9
            @test ricker(3, 3, 10, 2) ≈ 10 atol=1e-9

            # Equals 0 at x = 0
            @test ricker(0, 2, 5) ≈ 0 atol=1e-9
            @test ricker(0, 1, 1, 0.5) ≈ 0 atol=1e-9

            # location is the global maximum (hump shape)
            x = collect(0:0.01:20)
            vals = ricker(x, 4, 3, 1.5)
            @test x[argmax(vals)] ≈ 4 atol=0.01
            @test all(diff(vals[x .<= 4]) .>= 0)   # increasing before the peak
            @test all(diff(vals[x .>= 4]) .<= 0)   # decreasing after the peak

            # upper scales the curve
            xs = collect(0.1:0.1:10)
            @test ricker(xs, 2, 6) ≈ 6 .* ricker(xs, 2, 1) atol=1e-9

            # shape = 1 reduces to the standard Ricker a*x*exp(-b*x)
            location = 2; upper = 1
            a = upper * exp(1) / location
            b = 1 / location
            xv = [0.5, 1, 3, 5]
            @test ricker(xv, location, upper, 1) ≈ a .* xv .* exp.(-b .* xv) atol=1e-9

            # shape = alpha matches generalized form C * x^alpha * exp(-(alpha/location)*x)
            location = 2; upper = 3; alpha = 2.5
            C = upper * exp(alpha) / location^alpha
            xv = [0.5, 1, 2, 4, 6]
            @test ricker(xv, location, upper, alpha) ≈
                  C .* xv .^ alpha .* exp.(-(alpha / location) .* xv) atol=1e-9

            # larger shape narrows the peak (smaller values off-peak)
            x_off = 8.0  # away from the peak at location = 4
            broad = ricker(x_off, 4, 1, 0.5)
            base = ricker(x_off, 4, 1, 1)
            narrow = ricker(x_off, 4, 1, 3)
            @test narrow < base
            @test base < broad

            # Vectorized over x, and broadcastable (as sdbuildR emits ricker.(...))
            v = ricker(collect(0:0.5:5), 2)
            @test length(v) == 11
            @test eltype(v) <: AbstractFloat
            @test ricker.(collect(0:0.5:5), 2) ≈ v atol=1e-9

            # a, b parameterization (positions 5 and 6) equals the standard curve at shape = 1
            a = 2.5; b = 0.4
            xv = [0.0, 0.5, 1, 2.5, 5, 9]
            @test ricker(xv, 1, 1, 1, a, b) ≈ a .* xv .* exp.(-b .* xv) atol=1e-9

            # a, b equals the expanded form a*x^shape*exp(-b*x) for any shape
            a = 1.8; b = 0.6; shp = 2.5
            xv = [0.5, 1, 2, 4, 6]
            @test ricker(xv, 1, 1, shp, a, b) ≈ a .* xv .^ shp .* exp.(-b .* xv) atol=1e-9

            # a, b maps to location = shape/b, upper = a*(location/e)^shape
            location = shp / b
            upper = a * (location / exp(1))^shp
            xv = [0.5, 1, 2, 4]
            @test ricker(xv, 1, 1, shp, a, b) ≈
                  ricker(xv, location, upper, shp) atol=1e-9

            # a, b override the (default-filled) location/upper that translation injects,
            # so this must NOT error (mirrors ricker.(N, 1.0, 1.0, 1.0, 2.5, 0.4))
            @test ricker(3, 1, 1, 1, 2.5, 0.4) ≈ 2.5 * 3 * exp(-0.4 * 3) atol=1e-9

            # Errors on incomplete or non-numeric parameterization
            @test_throws ArgumentError ricker(1, 1, 1, 1, 2)          # b missing
            @test_throws ArgumentError ricker(1, 1, 1, 1, nothing, 0.5)  # a missing
            @test_throws ArgumentError ricker(1, 1, 1, 1, "x", 0.5)   # non-numeric a
            @test_throws ArgumentError ricker(1, 1, 1, "c")           # non-numeric shape
            @test_throws ArgumentError ricker(1, "a")                 # non-numeric location
            @test_throws ArgumentError ricker(1, 1, "b")             # non-numeric upper
        end

        @testset "r_min" begin
            # Single vector — R-style scalar minimum
            @test r_min([3.0, 1.0, 2.0])       == 1.0
            @test r_min([5.0])                  == 5.0
            @test r_min([-1.0, -3.0, -2.0])     == -3.0

            # Multiple scalar args — delegates to Base.min
            @test r_min(3.0, 1.0, 2.0) == 1.0
            @test r_min(5.0)            == 5.0

            # Single scalar (e.g. times[1]) — returned unchanged
            @test r_min(0.0)            == 0.0
            @test r_min(7)              == 7

            # Other single collections — R-style scalar minimum
            @test r_min((0.0, 182.5))   == 0.0      # tuple
            @test r_min((3.0, 1.0, 2.0)) == 1.0     # tuple
            @test r_min(2:5)            == 2        # range
            @test r_min((a = 3.0, b = 1.0)) == 1.0  # named tuple
        end

        @testset "r_max" begin
            # Single vector — R-style scalar maximum
            @test r_max([3.0, 1.0, 2.0])       == 3.0
            @test r_max([5.0])                  == 5.0
            @test r_max([-1.0, -3.0, -2.0])     == -1.0

            # Multiple scalar args — delegates to Base.max
            @test r_max(3.0, 1.0, 2.0) == 3.0
            @test r_max(5.0)            == 5.0

            # Single scalar (e.g. times[1]) — returned unchanged
            @test r_max(0.0)            == 0.0
            @test r_max(7)              == 7

            # Other single collections — R-style scalar maximum
            @test r_max((0.0, 182.5))    == 182.5   # tuple
            @test r_max((3.0, 1.0, 2.0)) == 3.0     # tuple
            @test r_max(2:5)             == 5       # range
            @test r_max((a = 3.0, b = 1.0)) == 3.0  # named tuple
        end

        @testset "r_diff" begin
            # Defaults — equivalent to Base.diff / R's diff(); args are positional
            # to mirror R's diff(x, lag, differences)
            @test r_diff([1, 3, 6, 10]) == [2, 3, 4]
            @test r_diff([1, 3, 6, 10]) == diff([1, 3, 6, 10])
            @test r_diff([1.0, 2.5, 2.0]) == [1.5, -0.5]

            # lag argument (2nd positional)
            @test r_diff([1, 3, 6, 10], 2) == [5, 7]
            @test r_diff([1, 3, 6, 10], 3) == [9]

            # differences argument (3rd positional; iterated differencing)
            @test r_diff([1, 3, 6, 10], 1, 2) == [1, 1]
            @test r_diff([1, 4, 9, 16, 25], 1, 2) == [2, 2, 2]

            # lag and differences combined
            @test r_diff(collect(1:10), 2, 2) == zeros(Int, 6)

            # Float literals are accepted (the converter may emit them)
            @test r_diff([1, 3, 6, 10], 2.0) == [5, 7]

            # Result shorter by lag * differences; empty when too short
            @test r_diff([1, 2, 3], 3) == Int[]
            @test isempty(r_diff([1, 2], 1, 5))

            # Single element / empty input
            @test r_diff([5]) == Int[]
            @test r_diff(Int[]) == Int[]

            # Works on ranges
            @test r_diff(1:5) == [1, 1, 1, 1]

            # Invalid arguments
            @test_throws ArgumentError r_diff([1, 2, 3], 0)
            @test_throws ArgumentError r_diff([1, 2, 3], 1, 0)
        end

        @testset "r_as_logical" begin
            # Numbers: any nonzero is true (unlike Base.Bool)
            @test r_as_logical(2) === true
            @test r_as_logical(0) === false
            @test r_as_logical(-1.5) === true
            @test r_as_logical(0.0) === false
            @test_throws InexactError Bool(2)   # contrast with Base behaviour

            # Bool passes through
            @test r_as_logical(true) === true
            @test r_as_logical(false) === false

            # Strings R accepts
            @test r_as_logical("TRUE") === true
            @test r_as_logical("T") === true
            @test r_as_logical("false") === false
            @test r_as_logical("F") === false
            @test ismissing(r_as_logical("yes"))   # R returns NA
            @test ismissing(r_as_logical("1"))     # R: as.logical("1") is NA

            # missing passes through
            @test ismissing(r_as_logical(missing))

            # Broadcasts like R's as.logical(vector)
            @test r_as_logical.([0, 1, 2, -3]) == [false, true, true, true]
        end

        @testset "r_grep" begin
            x = ["apple", "berry", "avocado", "Cherry"]

            # Default: indices of matching elements
            @test r_grep("a", x) == [1, 3]
            @test r_grep("z", x) == Int[]

            # value = true returns the matching elements
            @test r_grep("a", x, false, false, true) == ["apple", "avocado"]

            # ignore_case
            @test r_grep("cherry", x, true) == [4]
            @test r_grep("cherry", x, false) == Int[]

            # invert selects non-matching (8th positional arg)
            @test r_grep("a", x, false, false, false, false, false, true) == [2, 4]

            # fixed = true treats pattern literally
            @test r_grep(".", ["a.b", "ab", "c.d"], false, false, false, true) == [1, 3]
            @test r_grep(".", ["a.b", "ab", "c.d"]) == [1, 2, 3]  # regex: . matches any
        end

        @testset "r_rbind" begin
            # Vectors become rows
            @test r_rbind([1, 2, 3], [4, 5, 6]) == [1 2 3; 4 5 6]
            @test size(r_rbind([1, 2, 3], [4, 5, 6])) == (2, 3)

            # Single vector becomes a 1-row matrix (unlike vcat)
            @test r_rbind([1, 2, 3]) == reshape([1, 2, 3], 1, 3)

            # Matrices are stacked vertically
            @test r_rbind([1 2; 3 4], [5 6]) == [1 2; 3 4; 5 6]

            # Scalars become 1×1 rows
            @test r_rbind(1, 2) == reshape([1, 2], 2, 1)
        end

        @testset "r_upper_tri / r_lower_tri" begin
            m = zeros(3, 3)

            @test r_upper_tri(m) == Bool[0 1 1; 0 0 1; 0 0 0]
            @test r_upper_tri(m, true) == Bool[1 1 1; 0 1 1; 0 0 1]
            @test r_lower_tri(m) == Bool[0 0 0; 1 0 0; 1 1 0]
            @test r_lower_tri(m, true) == Bool[1 0 0; 1 1 0; 1 1 1]

            # Non-square works too
            @test r_upper_tri(zeros(2, 3)) == Bool[0 1 1; 0 0 1]
            @test eltype(r_upper_tri(m)) == Bool
        end

        @testset "r_na_omit" begin
            @test r_na_omit([1, missing, 3]) == [1, 3]
            @test r_na_omit([1, 2, 3]) == [1, 2, 3]
            @test r_na_omit(Union{Missing,Int}[missing, missing]) == Int[]
            # Result is an eager vector (supports indexing/length), unlike skipmissing
            r = r_na_omit([1.0, missing, 2.0])
            @test r isa AbstractVector
            @test length(r) == 2
            @test r[2] == 2.0
        end

        @testset "r_range" begin
            # Single collection
            @test r_range([3.0, 1.0, 2.0]) == [1.0, 3.0]
            @test r_range(2:5) == [2, 5]
            # Multiple scalar args
            @test r_range(4, 2, 7) == [2, 7]
            @test r_range(5) == [5, 5]
            # Returns a vector, not a tuple (contrast with extrema)
            @test r_range([3, 1, 2]) isa Vector
            @test extrema([3, 1, 2]) == (1, 3)
        end

        @testset "r_match" begin
            # Vector x: positions of first matches, missing when absent
            res = r_match(["b", "d", "a"], ["a", "b", "c"])
            @test res[1] == 2
            @test ismissing(res[2])
            @test res[3] == 1

            # Scalar x
            @test r_match(3, [1, 2, 3, 3]) == 3
            @test ismissing(r_match(9, [1, 2, 3]))

            # First occurrence is returned
            @test r_match([2], [2, 2, 2]) == [1]
        end

        @testset "r_sort" begin
            @test r_sort([3, 1, 2]) == [1, 2, 3]
            @test r_sort([3, 1, 2], false) == [1, 2, 3]
            @test r_sort([3, 1, 2], true) == [3, 2, 1]
            @test r_sort([2.5, -1.0, 0.0]) == [-1.0, 0.0, 2.5]
            # Does not mutate input
            v = [3, 1, 2]
            r_sort(v, true)
            @test v == [3, 1, 2]
        end

        @testset "r_rowsums / r_colsums / r_rowmeans / r_colmeans" begin
            m = [1 2 3; 4 5 6]
            @test r_rowsums(m) == [6, 15]
            @test r_colsums(m) == [5, 7, 9]
            @test r_rowmeans(m) == [2.0, 5.0]
            @test r_colmeans(m) == [2.5, 3.5, 4.5]
            # Results are vectors, not 1xn / nx1 matrices
            @test r_rowsums(m) isa AbstractVector
            @test r_colsums(m) isa AbstractVector
        end

        @testset "r_cummax / r_cummin" begin
            @test r_cummax([1, 3, 2, 5, 4]) == [1, 3, 3, 5, 5]
            @test r_cummin([5, 3, 4, 1, 2]) == [5, 3, 3, 1, 1]
            @test r_cummax([2.0]) == [2.0]
            @test r_cummax(Int[]) == Int[]
        end

        @testset "r_rep" begin
            # times (2nd positional)
            @test r_rep([1, 2], 3) == [1, 2, 1, 2, 1, 2]
            @test r_rep([1, 2]) == [1, 2]
            # each (4th positional)
            @test r_rep([1, 2], 1, -1, 2) == [1, 1, 2, 2]
            # each and times combined: each first, then tile
            @test r_rep([1, 2], 2, -1, 2) == [1, 1, 2, 2, 1, 1, 2, 2]
            # length_out (3rd positional) recycles/truncates, overrides times
            @test r_rep([1, 2], 1, 5) == [1, 2, 1, 2, 1]
            @test r_rep([1, 2, 3], 1, 2) == [1, 2]
            # scalar input
            @test r_rep(7, 3) == [7, 7, 7]
            # float literals accepted (converter may emit them)
            @test r_rep([1, 2], 3.0) == [1, 2, 1, 2, 1, 2]
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

        @testset "floor division operator ⊘" begin
            @test (7 ⊘ 2) == 3
            @test (10 ⊘ 5) == 2
            @test (15 ⊘ 4) == 3
            @test (7.5 ⊘ 2.0) == 3.0

            # Floored (follows sign of divisor), matching R's %/%
            @test (-7 ⊘ 2) == -4
            @test (7 ⊘ -2) == -4

            # Should be same as fld()
            @test (17 ⊘ 5) == fld(17, 5)
            @test (-23.7 ⊘ 4.2) == fld(-23.7, 4.2)
        end
    end

    @testset "Integration tests" begin
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
            r = make_ramp(times, 0.0, 10.0, 100.0)

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
