"""
    custom_func

Custom utility functions for system dynamics modeling, including:
- Signal generation (ramps, steps, pulses, seasonal waves)
- Interpolation and extrapolation
- Mathematical utilities (logistic, logit, expit)
- Random sampling functions
- String/array utilities

This module is designed to work seamlessly with Unitful quantities.
"""
module custom_func

using Unitful
using Distributions
using ..unit_func: convert_u

export itp, make_ramp, make_step, make_pulse, make_seasonal
export round_IM, logit, expit, logistic
export nonnegative, rbool, rdist
export indexof, contains_IM, round_
export is_function_or_interp, ⊕

# ============================================================================
# Type Checking Utilities
# ============================================================================

"""
    is_function_or_interp(x)

Check if `x` is a Function or an AbstractInterpolation object.

# Examples
```julia
julia> is_function_or_interp(sin)
true

julia> is_function_or_interp(itp([1, 2], [3, 4]))
true

julia> is_function_or_interp(5)
false
```
"""
is_function_or_interp(x) = isa(x, Function) || isa(x, Interpolator)

# ============================================================================
# Interpolation Functions
# ============================================================================

"""
    itp(x, y; method="linear", extrapolation="constant")

Create an interpolation function from vectors `x` and `y`.

# Arguments
- `x::AbstractVector`: Independent variable values (will be sorted)
- `y::AbstractVector`: Dependent variable values
- `method::String="linear"`: Interpolation method ("linear" or "constant")
- `extrapolation::String="constant"`: Extrapolation behavior ("constant", "linear", "missing", or "error")

# Returns
- `Interpolator`: Interpolation object that can be called as a function

# Examples
```julia
julia> f = itp([1, 2, 3], [10, 20, 30])
julia> f(1.5)
15.0

julia> f = itp([1, 3, 2], [10, 30, 20])  # Automatically sorted
julia> f(2.5)
25.0

julia> f = itp([1, 2, 3], [10, 20, 30], extrapolation="missing")
julia> f(5.0)  # Outside range
missing
```
"""
function itp(x, y; method="linear", extrapolation="constant")
    # Ensure y is sorted along x
    idx = sortperm(x)
    x_sorted = x[idx]
    y_sorted = y[idx]

    # Convert string arguments to symbols
    method_sym = Symbol(method)
    extrap_sym = Symbol(extrapolation)
    
    # Validate method
    if !(method_sym in (:linear, :constant))
        throw(ArgumentError("Method must be 'linear' or 'constant', got: $method"))
    end
    
    # Validate and handle extrapolation aliases for backwards compatibility
    if extrapolation == "nearest"
        # Map old "nearest" to "constant" for backwards compatibility
        extrap_sym = :constant
    elseif extrapolation == "NA"
        # Map old "NA" to "missing" for backwards compatibility
        extrap_sym = :missing
    elseif !(extrap_sym in (:constant, :linear, :missing, :error))
        throw(ArgumentError("Extrapolation must be 'constant', 'linear', 'missing', or 'error', got: $extrapolation"))
    end

    return Interpolator(x_sorted, y_sorted, method=method_sym, extrap=extrap_sym)
end


"""
Unified interpolator with configurable interpolation and extrapolation methods
"""
struct Interpolator{TX,TY}
    x::TX
    y::TY
    method::Symbol
    extrap::Symbol
    
    function Interpolator(x, y; method=:linear, extrap=:constant)
        @assert length(x) == length(y) "x and y must have same length"
        @assert issorted(x) "x must be sorted"
        @assert method in (:linear, :constant) "method must be :linear or :constant"
        @assert extrap in (:constant, :linear, :missing, :error) "extrap must be :constant, :linear, :missing, or :error"
        
        # Validate extrap is compatible with method
        if method == :constant && extrap == :linear
            error("Linear extrapolation is not supported for constant interpolation")
        end
        
        new{typeof(x), typeof(y)}(x, y, method, extrap)
    end
end

# Make it callable for scalar inputs
function (interp::Interpolator)(x::Number)
    if interp.method == :linear
        return _linear_interp(interp, x)
    elseif interp.method == :constant
        return _constant_interp(interp, x)
    end
end

# Evaluate interpolation element-wise for vector/matrix inputs.
function (interp::Interpolator)(x::AbstractArray)
    return interp.(x)
end

# Internal linear interpolation
function _linear_interp(interp::Interpolator, x)
    idx = searchsortedlast(interp.x, x)
    
    # Handle extrapolation
    if idx == 0
        if interp.extrap == :constant
            return first(interp.y)
        elseif interp.extrap == :linear
            x1, x2 = interp.x[1], interp.x[2]
            y1, y2 = interp.y[1], interp.y[2]
            slope = (y2 - y1) / (x2 - x1)
            return y1 + slope * (x - x1)
        elseif interp.extrap == :missing
            return missing
        elseif interp.extrap == :error
            error("x=$x is below range [$(first(interp.x)), $(last(interp.x))]")
        end
    elseif idx >= length(interp.x)
        if interp.extrap == :constant
            return last(interp.y)
        elseif interp.extrap == :linear
            x1, x2 = interp.x[end-1], interp.x[end]
            y1, y2 = interp.y[end-1], interp.y[end]
            slope = (y2 - y1) / (x2 - x1)
            return y2 + slope * (x - x2)
        elseif interp.extrap == :missing
            return missing
        elseif interp.extrap == :error
            error("x=$x is above range [$(first(interp.x)), $(last(interp.x))]")
        end
    end
    
    # Interpolate
    x1, x2 = interp.x[idx], interp.x[idx+1]
    y1, y2 = interp.y[idx], interp.y[idx+1]
    t = (x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)
end

# Internal constant interpolation
function _constant_interp(interp::Interpolator, x)
    idx = searchsortedlast(interp.x, x)
    
    # Handle extrapolation
    if idx == 0
        if interp.extrap == :constant
            return first(interp.y)
        elseif interp.extrap == :missing
            return missing
        elseif interp.extrap == :error
            error("x=$x is below range [$(first(interp.x)), $(last(interp.x))]")
        end
    end
    
    idx = clamp(idx, 1, length(interp.y))
    return interp.y[idx]
end

# ============================================================================
# Signal Generation Functions
# ============================================================================

_zero_with_unit(x) = isa(x, Unitful.Quantity) ? zero(x) : 0.0

_normalize_time_arg(times, time_units, value) = begin
    normalized = convert_u(value, time_units)
    if eltype(times) <: Unitful.Quantity
        return normalized
    end

    return Unitful.ustrip(normalized)
end

"""
    make_ramp(time_units, times, start, finish, height=1.0)

Create a ramp signal that linearly increases from 0 to `height` between `start` and `finish` times.

The ramp starts at height 0 at time `start`, increases linearly, and reaches `height` at time `finish`.
Outside this range, the value is constant (0 before start, height after finish).

# Arguments
- `time_units`: Units for time (e.g., u"yr", u"d")
- `times`: Time vector or range (start, end)
- `start`: Start time of ramp
- `finish`: End time of ramp
- `height=1.0`: Maximum height of ramp (can be negative for decreasing ramp)

# Returns
- Interpolation function that can be evaluated at any time

# Examples
```julia
julia> r = make_ramp(u"yr", [0.0, 10.0], 2.0, 5.0, 10.0)
julia> r(3.5)  # Halfway through ramp
5.0
```
"""
function make_ramp(time_units, times, start, finish, height=1.0)
    @assert start < finish "The finish time of the ramp cannot be before the start time. To specify a decreasing ramp, set the height to a negative value."

    # Normalize units between times and ramp parameters
    start, finish = _normalize_time_units(times, time_units, start, finish)
    
    # Initialize with the same unit type as height when applicable
    start_h_ramp = _zero_with_unit(height)
    add_y = _zero_with_unit(height)

    if start > last(times)
        # If the pulse starts after the end of the time range, return a zero function    
        func = itp(times, [add_y; add_y], method="constant", extrapolation="nearest")
        return func
    elseif finish < first(times)
        # If the ramp finishes before the start of the time range, return a constant height function
        func = itp(times, [height; height], method="constant", extrapolation="nearest")
        return func
    end   
       

    x = [start, finish]
    y = [start_h_ramp, height]

    # If the ramp is after the start time, add a zero at the start
    if start > first(times) || finish < first(times)
        x = [first(times); x]
        y = [add_y; y]
    end

    func = itp(x, y, method="linear", extrapolation="nearest")
    return func
end

"""
    make_step(time_units, times, start, height=1.0)

Create a step signal that jumps from 0 to `height` at time `start`.

# Arguments
- `time_units`: Units for time
- `times`: Time vector or range
- `start`: Time when step occurs
- `height=1.0`: Height of step

# Returns
- Interpolation function representing the step signal

# Examples
```julia
julia> s = make_step(u"s", [0.0, 10.0], 5.0, 2.0)
julia> s(4.9)  # Before step 
0.0
julia> s(5.1)  # After step
2.0
```
"""
function make_step(time_units, times, start, height=1.0)
    # Normalize units
    start = _normalize_single_time(times, time_units, start)
    
    add_y = _zero_with_unit(height)

    # If the step starts after the end of the time range, return a zero function    
    if start > last(times)
        func = itp(times, [add_y; add_y], method="constant", extrapolation="nearest")
        return func
    end   

    x = [start, times[2]]
    y = [height, height]

    # If the step is after the start time, add a zero at the start
    if start > first(times)
        x = [first(times); x]
        y = [add_y; y]
    end

    func = itp(x, y, method="constant", extrapolation="nearest")
    return func
end

"""
    make_pulse(time_units, times, start, height=1.0, width=1.0*time_units, repeat_interval=nothing)

Create a pulse signal with specified width and optional repetition.

# Arguments
- `time_units`: Units for time
- `times`: Time vector or range
- `start`: Start time of first pulse
- `height=1.0`: Height of pulse
- `width=1.0*time_units`: Duration of each pulse
- `repeat_interval=nothing`: Time between pulse starts (nothing = single pulse)

# Returns
- Interpolation function representing the pulse train

# Examples
```julia
julia> p = make_pulse(u"s", [0.0, 20.0], 5.0, 1.0, 2.0, 10.0)  # Pulse every 10s
julia> p(6.0)  # During first pulse
1.0
julia> p(8.0)  # Between pulses
0.0
```
"""
function make_pulse(time_units, times, start, height=1.0, width=1.0 * time_units, repeat_interval=nothing)
    # Validate width
    width_value = Unitful.ustrip(convert_u(width, time_units))
    if width_value <= 0.0
        throw(ArgumentError("The width of the pulse cannot be equal to or less than 0; to indicate an 'instantaneous' pulse, specify the simulation step size (dt)."))
    end

    # Normalize units
    start, width, repeat_interval = _normalize_pulse_units(times, time_units, start, width, repeat_interval)

    add_y = _zero_with_unit(height)

    if start > last(times)
        # If the pulse starts after the end of the time range, return a zero function    
        func = itp(times, [add_y; add_y], method="constant", extrapolation="nearest")
        return func
    end   

    # Define start and end times of pulses
    last_time = last(times)

    if isnothing(repeat_interval)
        # Single pulse
        signal_times = [start; start + width]
        signal_y = [height; add_y]
    else 

        start_ts = collect(start:repeat_interval:last_time)

        # When width is equal or greater than repeat interval, it's basically continuously 1
        if width >= repeat_interval
            signal_times = [start_ts; ]
            signal_y = [fill(height, length(start_ts)); ]
        else
            # Build signal as vectors of times and y-values
            end_ts = start_ts .+ width
            signal_times = [start_ts; end_ts]
            signal_y = [fill(height, length(start_ts)); fill(add_y, length(end_ts))]
        end
    end

    # Add zeros at boundaries if needed
    if minimum(signal_times) > first(times)
        signal_times = [first(times); signal_times]
        signal_y = [add_y; signal_y]
    end

    if maximum(signal_times) < last_time
        signal_times = [signal_times; last_time]
        signal_y = [signal_y; add_y]
    end

    # Sort by time
    perm = sortperm(signal_times)
    x = signal_times[perm]
    y = signal_y[perm]
    func = itp(x, y, method="constant", extrapolation="nearest")

    return func
end

"""
    make_seasonal(times, dt, period=u"1yr", shift=u"0yr")

Create a seasonal cosine wave with specified period and phase shift.

The wave oscillates between -1 and 1 with the formula: cos(2π(t - shift)/period)

# Arguments
- `dt`: Time step for sampling
- `times`: Time range [start, end]
- `period=u"1yr"`: Period of oscillation
- `shift=u"0yr"`: Phase shift (positive = delay)

# Returns
- Interpolation function representing the seasonal pattern

# Examples
```julia
julia> wave = make_seasonal(0.1u"yr", [0.0u"yr", 2.0u"yr"], 1.0u"yr")
julia> wave(0.0u"yr")  # Peak of cosine
1.0
julia> wave(0.5u"yr")  # Trough
-1.0
```
"""
function make_seasonal(dt, times, period=u"1yr", shift=u"0yr")
    @assert Unitful.ustrip(period) > 0 "The period of the seasonal wave must be greater than 0."

    time_vec = times[1]:dt:times[2]
    if isa(first(time_vec), Unitful.Quantity) || isa(period, Unitful.Quantity) || isa(shift, Unitful.Quantity)
        time_num = Unitful.ustrip.(time_vec)
        shift_num = isa(shift, Unitful.Quantity) ? Unitful.ustrip(shift) : shift
        period_num = isa(period, Unitful.Quantity) ? Unitful.ustrip(period) : period
        phase = 2 * pi .* (time_num .- shift_num) ./ period_num
    else
        phase = 2 * pi .* (time_vec .- shift) ./ period
    end
    y = cos.(phase)
    func = itp(time_vec, y, method="linear", extrapolation="nearest")

    return func
end

# ============================================================================
# Helper Functions for Unit Conversion
# ============================================================================

"""
    _normalize_time_units(times, time_units, start, finish)

Internal helper to normalize time units between simulation times and signal parameters.
"""
function _normalize_time_units(times, time_units, start, finish)
    if eltype(times) <: Unitful.Quantity
        # Times have units, ensure start/finish match
        start = convert_u(start, time_units)
        finish = convert_u(finish, time_units)
    else
        # Times are unitless, coerce through time_units and strip to numeric values
        start = _normalize_time_arg(times, time_units, start)
        finish = _normalize_time_arg(times, time_units, finish)
    end
    return start, finish
end

"""
    _normalize_single_time(times, time_units, start)

Internal helper to normalize a single time value.
"""
function _normalize_single_time(times, time_units, start)
    if eltype(times) <: Unitful.Quantity
        start = convert_u(start, time_units)
    else
        start = _normalize_time_arg(times, time_units, start)
    end
    return start
end

"""
    _normalize_pulse_units(times, time_units, start, width, repeat_interval)

Internal helper to normalize time units for pulse signals.
"""
function _normalize_pulse_units(times, time_units, start, width, repeat_interval)
    if eltype(times) <: Unitful.Quantity
        start = convert_u(start, time_units)
        width = convert_u(width, time_units)
        if !isnothing(repeat_interval)
            repeat_interval = convert_u(repeat_interval, time_units)
        end
    else
        start = _normalize_time_arg(times, time_units, start)
        width = _normalize_time_arg(times, time_units, width)
        if !isnothing(repeat_interval)
            repeat_interval = _normalize_time_arg(times, time_units, repeat_interval)
        end
    end
    return start, width, repeat_interval
end

# ============================================================================
# Mathematical Functions
# ============================================================================

"""
    round_IM(x::Real, digits::Int=0)

Round a number using Insight Maker's convention where 0.5 rounds up.

Note: Julia's default `round()` uses banker's rounding where 0.5 rounds to nearest even.
This function always rounds 0.5 up to match Insight Maker behavior.

# Examples
```julia
julia> round_IM(0.5)
1.0
julia> round_IM(1.5)
2.0
julia> round_IM(2.5)
3.0
```
"""
function round_IM(x::Real, digits::Int=0)
    scaled_x = x * 10.0^digits
    frac = scaled_x % 1
    
    # Check if fractional part is exactly ±0.5
    if abs(frac) == 0.5
        return ceil(scaled_x) / 10.0^digits
    else
        return round(scaled_x) / 10.0^digits
    end
end

"""
    logit(p)

Compute the logit (log-odds) function: log(p / (1 - p))

# Examples
```julia
julia> logit(0.5)
0.0
julia> logit(0.75)
1.0986122886681098
```
"""
logit(p) = log(p / (1 - p))

"""
    expit(x)

Compute the expit (inverse logit) function: 1 / (1 + exp(-x))

Also known as the logistic sigmoid function.

# Examples
```julia
julia> expit(0.0)
0.5
julia> expit(10.0)
0.9999546021312976
```
"""
expit(x) = 1 / (1 + exp(-x))

"""
    logistic(x, slope=1.0, midpoint=0.0, upper=1.0)

Compute a generalized logistic function with adjustable slope, midpoint, and upper bound.

Formula: upper / (1 + exp(-slope * (x - midpoint)))

# Arguments
- `x`: Input value
- `slope=1.0`: Steepness of the curve
- `midpoint=0.0`: x-value at the inflection point
- `upper=1.0`: Maximum asymptotic value

# Examples
```julia
julia> logistic(0.0, 1.0, 0.0, 1.0)  # Standard logistic at midpoint
0.5
julia> logistic(5.0, 2.0, 5.0, 10.0)  # Steeper curve, shifted
5.0
```
"""
function logistic(x, slope=1.0, midpoint=0.0, upper=1.0)
    @assert isfinite(Unitful.ustrip(slope)) && isfinite(Unitful.ustrip(midpoint)) && isfinite(Unitful.ustrip(upper)) "slope, midpoint, and upper must be finite numeric values"
    return upper / (1 + exp(-slope * (x - midpoint)))
end

"""
    nonnegative(x)

Ensure value(s) are non-negative by returning max(0, x).

Works with scalars, arrays, and Unitful quantities.

# Examples
```julia
julia> nonnegative(-5)
0.0
julia> nonnegative(3)
3.0
julia> nonnegative([-1, 2, -3])
3-element Vector{Float64}: [0.0, 2.0, 0.0]
```
"""
nonnegative(x::Real) = max(0.0, x)
nonnegative(x::Unitful.Quantity) = max(0.0, Unitful.ustrip(x)) * Unitful.unit(x)
nonnegative(x::AbstractArray{<:Real}) = max.(0.0, x)
nonnegative(x::AbstractArray{<:Unitful.Quantity}) = max.(0.0, Unitful.ustrip.(x)) .* Unitful.unit.(x)

# ============================================================================
# Rounding Utilities
# ============================================================================

"""
    round_(x, digits=0)

Flexible rounding function that handles both regular numbers and Unitful quantities.

# Examples
```julia
julia> round_(3.14159, digits=2)
3.14
julia> round_(5.6u"m", digits=0)
6.0 m
```
"""
round_(x, digits::Real) = round(x, digits=round(Int, digits))
round_(x; digits::Real=0) = round(x, digits=round(Int, digits))
round_(x::Unitful.Quantity, digits::Real) = round(Unitful.ustrip(x), digits=round(Int, digits)) * Unitful.unit(x)
round_(x::Unitful.Quantity; digits::Real=0) = round(Unitful.ustrip(x), digits=round(Int, digits)) * Unitful.unit(x)

# ============================================================================
# Random Sampling Functions
# ============================================================================

"""
    rbool(p)

Generate a random boolean value with probability `p` of being true.

Equivalent to Insight Maker's RandBoolean() function.

# Examples
```julia
julia> rbool(0.7)  # 70% chance of true
true
julia> rbool(0.0)  # Always false
false
```
"""
rbool(p) = rand() < p

"""
    rdist(a::Vector, b::Vector{<:Real})

Sample randomly from vector `a` with probabilities given by vector `b`.

Probabilities are automatically normalized to sum to 1.

# Arguments
- `a`: Vector of values to sample from
- `b`: Vector of probabilities (will be normalized, must all be non-negative)

# Examples
```julia
julia> rdist(["red", "green", "blue"], [0.5, 0.3, 0.2])
"red"  # (with 50% probability)
```
"""
function rdist(a::Vector{T}, b::Vector{<:Real}) where T
    if length(a) != length(b)
        throw(ArgumentError("Length of a and b must match"))
    end
    
    if any(x -> x < 0, b)
        throw(ArgumentError("All probabilities must be non-negative"))
    end
    
    b_sum = sum(b)
    if b_sum <= 0
        throw(ArgumentError("Sum of probabilities must be positive"))
    end
    
    b_normalized = b / b_sum
    return a[rand(Distributions.Categorical(b_normalized))]
end

# ============================================================================
# String and Array Utilities
# ============================================================================

"""
    indexof(haystack, needle)

Find the index of `needle` in `haystack`.

Works with both strings and arrays. Returns 0 if not found (Insight Maker convention).

# Examples
```julia
julia> indexof("hello", "ll")
3
julia> indexof([1, 2, 3, 4], 3)
3
julia> indexof("hello", "x")
0
```
"""
function indexof(haystack, needle)
    if isa(haystack, AbstractString) && isa(needle, AbstractString)
        pos = findfirst(needle, haystack)
        return isnothing(pos) ? 0 : first(pos)
    else
        pos = findfirst(==(needle), haystack)
        return isnothing(pos) ? 0 : pos
    end
end

"""
    contains_IM(haystack, needle)

Check if `haystack` contains `needle`.

Works with both strings and arrays.

# Examples
```julia
julia> contains_IM("hello world", "world")
true
julia> contains_IM([1, 2, 3], 2)
true
julia> contains_IM("hello", "x")
false
```
"""
function contains_IM(haystack, needle)
    if isa(haystack, AbstractString) && isa(needle, AbstractString)
        return occursin(needle, haystack)
    else
        return needle in haystack
    end
end

# ============================================================================
# Operators
# ============================================================================

"""
    ⊕(x, y)

Modulus operator (x mod y).

Unicode alternative to `mod(x, y)`.

# Examples
```julia
julia> 7 ⊕ 3
1
julia> 10 ⊕ 5
0
```
"""
⊕(x, y) = mod(x, y)

end # module
