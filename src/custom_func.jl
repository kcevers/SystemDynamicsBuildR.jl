"""
    custom_func

Custom utility functions for system dynamics modeling, including:
- Signal generation (ramps, steps, pulses, seasonal waves)
- Interpolation and extrapolation
- Mathematical utilities (logistic, logit, expit)
- Random sampling functions
- String/array utilities
"""
module custom_func

using Distributions
using Random

export itp, make_ramp, make_step, make_pulse, make_seasonal
export round_IM, logit, expit, logistic, hill, r_min, r_max, r_diff
export r_as_logical, r_grep, r_rbind, r_upper_tri, r_lower_tri
export r_na_omit, r_range, r_match, r_sort
export r_rowsums, r_colsums, r_rowmeans, r_colmeans, r_cummax, r_cummin, r_rep
export nonnegative, rbool, rdist, with_rng
export indexof, contains_IM, round_
export is_function_or_interp, ⊕, ⊘

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
        extrap_sym = :constant
    elseif extrapolation == "NA"
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
        if length(x) != length(y)
            throw(ArgumentError("x and y must have same length"))
        end
        if !issorted(x)
            throw(ArgumentError("x must be sorted"))
        end
        if !(method in (:linear, :constant))
            throw(ArgumentError("method must be :linear or :constant"))
        end
        if !(extrap in (:constant, :linear, :missing, :error))
            throw(ArgumentError("extrap must be :constant, :linear, :missing, or :error"))
        end

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

"""
    make_ramp(times, start, finish, height=1.0)

Create a ramp signal that linearly increases from 0 to `height` between `start` and `finish` times.

The ramp starts at height 0 at time `start`, increases linearly, and reaches `height` at time `finish`.
Outside this range, the value is constant (0 before start, height after finish).

# Arguments
- `times`: Time vector or range [start, end]
- `start`: Start time of ramp
- `finish`: End time of ramp
- `height=1.0`: Maximum height of ramp (can be negative for decreasing ramp)

# Returns
- Interpolation function that can be evaluated at any time

# Examples
```julia
julia> r = make_ramp([0.0, 10.0], 2.0, 5.0, 10.0)
julia> r(3.5)  # Halfway through ramp
5.0
```
"""
function make_ramp(times, start, finish, height=1.0)
    @assert start < finish "The finish time of the ramp cannot be before the start time. To specify a decreasing ramp, set the height to a negative value."

    if start > last(times)
        func = itp(times, [0.0; 0.0], method="constant", extrapolation="nearest")
        return func
    elseif finish < first(times)
        func = itp(times, [height; height], method="constant", extrapolation="nearest")
        return func
    end

    x = [start, finish]
    y = [0.0, height]

    if start > first(times)
        x = [first(times); x]
        y = [0.0; y]
    end

    func = itp(x, y, method="linear", extrapolation="nearest")
    return func
end

"""
    make_step(times, start, height=1.0)

Create a step signal that jumps from 0 to `height` at time `start`.

# Arguments
- `times`: Time vector or range
- `start`: Time when step occurs
- `height=1.0`: Height of step

# Returns
- Interpolation function representing the step signal

# Examples
```julia
julia> s = make_step([0.0, 10.0], 5.0, 2.0)
julia> s(4.9)  # Before step
0.0
julia> s(5.1)  # After step
2.0
```
"""
function make_step(times, start, height=1.0)
    if start > last(times)
        func = itp(times, [0.0; 0.0], method="constant", extrapolation="nearest")
        return func
    end

    x = [start, last(times)]
    y = [height, height]

    if start > first(times)
        x = [first(times); x]
        y = [0.0; y]
    end

    func = itp(x, y, method="constant", extrapolation="nearest")
    return func
end

"""
    make_pulse(times, start, height=1.0, width=1.0, repeat_interval=nothing)

Create a pulse signal with specified width and optional repetition.

# Arguments
- `times`: Time vector or range
- `start`: Start time of first pulse
- `height=1.0`: Height of pulse
- `width=1.0`: Duration of each pulse
- `repeat_interval=nothing`: Time between pulse starts (nothing = single pulse)

# Returns
- Interpolation function representing the pulse train

# Examples
```julia
julia> p = make_pulse([0.0, 20.0], 5.0, 1.0, 2.0, 10.0)  # Pulse every 10s
julia> p(6.0)  # During first pulse
1.0
julia> p(8.0)  # Between pulses
0.0
```
"""
function make_pulse(times, start, height=1.0, width=1.0, repeat_interval=nothing)
    if width <= 0.0
        throw(ArgumentError("The width of the pulse cannot be equal to or less than 0; to indicate an 'instantaneous' pulse, specify the simulation step size (dt)."))
    end

    if start > last(times)
        func = itp(times, [0.0; 0.0], method="constant", extrapolation="nearest")
        return func
    end

    last_time = last(times)

    if isnothing(repeat_interval)
        signal_times = [start; start + width]
        signal_y = [height; 0.0]
    else
        start_ts = collect(start:repeat_interval:last_time)

        if width >= repeat_interval
            signal_times = [start_ts; ]
            signal_y = [fill(height, length(start_ts)); ]
        else
            end_ts = start_ts .+ width
            signal_times = [start_ts; end_ts]
            signal_y = [fill(height, length(start_ts)); fill(0.0, length(end_ts))]
        end
    end

    if minimum(signal_times) > first(times)
        signal_times = [first(times); signal_times]
        signal_y = [0.0; signal_y]
    end

    if maximum(signal_times) < last_time
        signal_times = [signal_times; last_time]
        signal_y = [signal_y; 0.0]
    end

    perm = sortperm(signal_times)
    x = signal_times[perm]
    y = signal_y[perm]
    func = itp(x, y, method="constant", extrapolation="nearest")

    return func
end

"""
    make_seasonal(dt, times, period=1.0, shift=0.0)

Create a seasonal cosine wave with specified period and phase shift.

The wave oscillates between -1 and 1 with the formula: cos(2π(t - shift)/period)

# Arguments
- `dt`: Time step for sampling
- `times`: Time range [start, end]
- `period=1.0`: Period of oscillation
- `shift=0.0`: Phase shift (positive = delay)

# Returns
- Interpolation function representing the seasonal pattern

# Examples
```julia
julia> wave = make_seasonal(0.1, [0.0, 2.0], 1.0)
julia> wave(0.0)  # Peak of cosine
1.0
julia> wave(0.5)  # Trough
-1.0
```
"""
function make_seasonal(dt, times, period=1.0, shift=0.0)
    @assert period > 0 "The period of the seasonal wave must be greater than 0."

    time_vec = times[1]:dt:times[2]
    phase = 2 * pi .* (time_vec .- shift) ./ period
    y = cos.(phase)
    func = itp(time_vec, y, method="linear", extrapolation="nearest")

    return func
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
    @assert isfinite(slope) && isfinite(midpoint) && isfinite(upper) "slope, midpoint, and upper must be finite numeric values"
    return upper / (1 + exp(-slope * (x - midpoint)))
end

"""
    hill(x, slope=1.0, midpoint=0.5, upper=1.0)

Compute the Hill function with configurable slope, midpoint, and upper asymptote.

Formula: upper * x^slope / (midpoint^slope + x^slope)

At x = midpoint the function returns exactly upper / 2.

# Arguments
- `x`: Input value (typically ≥ 0)
- `slope=1.0`: Hill coefficient controlling steepness; curve is S-shaped when slope > 1
- `midpoint=0.5`: x-value at which the output equals upper / 2 (EC50)
- `upper=1.0`: Maximum asymptotic value

# Examples
```julia
julia> hill(1.0, 1.0, 1.0, 1.0)   # At midpoint → upper/2
0.5
julia> hill(2.0, 2.0, 2.0, 10.0)  # Steeper, different scale
5.0
```
"""
function hill(x, slope=1.0, midpoint=0.5, upper=1.0)
    @assert isfinite(slope) && isfinite(midpoint) && isfinite(upper) "slope, midpoint, and upper must be finite numeric values"
    return upper * x^slope / (midpoint^slope + x^slope)
end

"""
    r_min(x::Number)
    r_min(c)
    r_min(args...)

Return the minimum value, R-style.

When called with a single scalar, returns it unchanged. When called with a
single collection (vector, tuple, range, array, or any iterable), returns the
scalar minimum of all elements (equivalent to Julia's `minimum`). When called
with multiple scalar arguments, delegates to `Base.min`.

Avoids type piracy on `Base.min` while matching R's `min()` behaviour.

# Examples
```julia
julia> r_min([3.0, 1.0, 2.0])
1.0
julia> r_min((0.0, 182.5))
0.0
julia> r_min(3.0, 1.0, 2.0)
1.0
julia> r_min(5.0)
5.0
```
"""
r_min(x::Number)         = x
r_min(c)                 = minimum(c)
r_min(args...)           = Base.min(args...)

"""
    r_max(x::Number)
    r_max(c)
    r_max(args...)

Return the maximum value, R-style.

When called with a single scalar, returns it unchanged. When called with a
single collection (vector, tuple, range, array, or any iterable), returns the
scalar maximum of all elements (equivalent to Julia's `maximum`). When called
with multiple scalar arguments, delegates to `Base.max`.

Avoids type piracy on `Base.max` while matching R's `max()` behaviour.

# Examples
```julia
julia> r_max([3.0, 1.0, 2.0])
3.0
julia> r_max((0.0, 182.5))
182.5
julia> r_max(3.0, 1.0, 2.0)
3.0
julia> r_max(5.0)
5.0
```
"""
r_max(x::Number)         = x
r_max(c)                 = maximum(c)
r_max(args...)           = Base.max(args...)

"""
    r_diff(x, lag = 1, differences = 1)

Return lagged and iterated differences, R-style.

Mirrors R's `diff()`: starting from `x`, compute `differences` successive
rounds of differencing, where each round subtracts elements `lag` positions
apart (`x[i+lag] - x[i]`). With the defaults (`lag = 1`, `differences = 1`)
this is equivalent to Julia's `Base.diff`.

The result is shorter than the input by `lag * differences` elements. If the
input is not long enough, an empty vector is returned (matching R).

# Examples
```julia
julia> r_diff([1, 3, 6, 10])
3-element Vector{Int64}:
 2
 3
 4
julia> r_diff([1, 3, 6, 10], 2)
2-element Vector{Int64}:
 5
 7
julia> r_diff([1, 3, 6, 10], 1, 2)
2-element Vector{Int64}:
 1
 1
```
"""
function r_diff(x::AbstractVector, lag::Real = 1, differences::Real = 1)
    # `lag`/`differences` are accepted as Real (the converter may pass float
    # literals) and converted to Int; R's diff(x, lag, differences) is positional.
    li = Int(lag)
    di = Int(differences)
    li >= 1 || throw(ArgumentError("`lag` must be >= 1, got $li"))
    di >= 1 || throw(ArgumentError("`differences` must be >= 1, got $di"))
    r = x
    for _ in 1:di
        length(r) > li || return r[1:0]
        r = r[(li + 1):end] .- r[1:(end - li)]
    end
    return r
end

"""
    r_as_logical(x)

Convert to a logical value, R-style.

Mirrors R's `as.logical()`: any nonzero number is `true` and zero is `false`
(unlike Julia's `Bool`, which errors on values other than 0/1). Strings such as
`"TRUE"`, `"T"`, `"FALSE"`, `"F"` (any capitalization R accepts) convert to the
corresponding `Bool`; unrecognized strings yield `missing` (as R returns `NA`).
`missing` is returned unchanged. Broadcast with `r_as_logical.(x)` for vectors.

# Examples
```julia
julia> r_as_logical(2)
true
julia> r_as_logical(0)
false
julia> r_as_logical("TRUE")
true
```
"""
r_as_logical(x::Bool)            = x
r_as_logical(x::Real)            = !iszero(x)
r_as_logical(::Missing)          = missing
function r_as_logical(x::AbstractString)
    s = strip(x)
    s in ("TRUE", "true", "True", "T")  ? true  :
    s in ("FALSE", "false", "False", "F") ? false : missing
end

"""
    r_grep(pattern, x, ignore_case=false, perl=false, value=false,
           fixed=false, useBytes=false, invert=false)

Search for `pattern` in the elements of `x`, R-style.

Mirrors R's `grep()`: by default returns the integer indices of the elements of
`x` that match `pattern` (a regular expression unless `fixed = true`). With
`value = true` the matching elements themselves are returned; with
`invert = true` the non-matching elements are selected; `ignore_case = true`
makes matching case-insensitive. The `perl` and `useBytes` arguments are
accepted for signature compatibility but ignored.

# Examples
```julia
julia> r_grep("a", ["apple", "berry", "avocado"])
2-element Vector{Int64}:
 1
 3
julia> r_grep("a", ["apple", "berry", "avocado"]; value = true)
2-element Vector{String}:
 "apple"
 "avocado"
```
"""
function r_grep(pattern, x, ignore_case = false, perl = false, value = false,
                fixed = false, useBytes = false, invert = false)
    if fixed
        needle = ignore_case ? lowercase(pattern) : pattern
        test = s -> occursin(needle, ignore_case ? lowercase(s) : s)
    else
        rx = ignore_case ? Regex(pattern, "i") : Regex(pattern)
        test = s -> occursin(rx, s)
    end
    idx = findall(s -> invert ? !test(s) : test(s), x)
    return value ? x[idx] : idx
end

"""
    r_rbind(args...)

Bind arguments together by rows, R-style.

Mirrors R's `rbind()`: vectors become the rows of the resulting matrix
(`r_rbind([1, 2], [3, 4])` is a 2×2 matrix), matrices are stacked vertically,
and scalars become 1×1 rows. This differs from `vcat`, which would concatenate
vectors into a single long vector instead of stacking them as rows.

# Examples
```julia
julia> r_rbind([1, 2, 3], [4, 5, 6])
2×3 Matrix{Int64}:
 1  2  3
 4  5  6
```
"""
r_rbind(args...) = reduce(vcat, map(_as_row, args))
_as_row(a::AbstractMatrix) = a
_as_row(a::AbstractVector) = permutedims(a)
_as_row(a)                 = permutedims([a])

"""
    r_upper_tri(x, diag=false)
    r_lower_tri(x, diag=false)

Return a logical matrix marking the upper (or lower) triangle of `x`, R-style.

Mirrors R's `upper.tri()`/`lower.tri()`: the result is a `Bool` matrix that is
`true` in the upper (lower) triangle and `false` elsewhere. When `diag = true`
the diagonal is included. This differs from `LinearAlgebra.UpperTriangular`,
which returns a view of the *values* rather than a logical mask.

# Examples
```julia
julia> r_upper_tri(zeros(3, 3))
3×3 Matrix{Bool}:
 0  1  1
 0  0  1
 0  0  0
```
"""
function r_upper_tri(x::AbstractMatrix, diag::Bool = false)
    m, n = size(x)
    return Bool[diag ? (j >= i) : (j > i) for i in 1:m, j in 1:n]
end
function r_lower_tri(x::AbstractMatrix, diag::Bool = false)
    m, n = size(x)
    return Bool[diag ? (j <= i) : (j < i) for i in 1:m, j in 1:n]
end

"""
    r_na_omit(x)

Remove missing values, R-style.

Mirrors R's `na.omit()`: returns a new vector with `missing` elements removed.
Unlike `Base.skipmissing`, which returns a lazy iterator, this eagerly collects
into a vector so the result supports indexing, `length`, concatenation, etc.

# Examples
```julia
julia> r_na_omit([1, missing, 3])
2-element Vector{Int64}:
 1
 3
```
"""
r_na_omit(x) = collect(skipmissing(x))

"""
    r_range(x)
    r_range(args...)

Return the minimum and maximum, R-style.

Mirrors R's `range()`: returns a 2-element vector `[min, max]`. With a single
collection it spans that collection; with multiple scalar arguments it spans
those values. This differs from `Base.extrema`, which returns a `(min, max)`
tuple.

# Examples
```julia
julia> r_range([3.0, 1.0, 2.0])
2-element Vector{Float64}:
 1.0
 3.0
julia> r_range(4, 2, 7)
2-element Vector{Int64}:
 2
 7
```
"""
r_range(x) = [minimum(x), maximum(x)]
function r_range(x::Number, rest::Number...)
    v = (x, rest...)
    return [minimum(v), maximum(v)]
end

"""
    r_match(x, table)

Return the positions of (first) matches of `x` in `table`, R-style.

Mirrors R's `match()`: for each element of `x`, returns the index of its first
occurrence in `table`, or `missing` if absent (R returns `NA`). This is distinct
from Julia's `Base.match`, which performs regular-expression matching.

# Examples
```julia
julia> r_match(["b", "d", "a"], ["a", "b", "c"])
3-element Vector{Union{Missing, Int64}}:
 2
  missing
 1
```
"""
r_match(x::AbstractVector, table) =
    [(i = findfirst(==(xi), table); i === nothing ? missing : i) for xi in x]
r_match(x, table) = (i = findfirst(==(x), table); i === nothing ? missing : i)

"""
    r_sort(x, decreasing=false)

Sort a collection, R-style.

Mirrors R's `sort()`: ascending by default, descending when `decreasing=true`.
This wrapper exists because R's `decreasing` argument maps to Julia's `rev`
keyword, which the converter cannot express directly.

# Examples
```julia
julia> r_sort([3, 1, 2])
3-element Vector{Int64}:
 1
 2
 3
julia> r_sort([3, 1, 2], true)
3-element Vector{Int64}:
 3
 2
 1
```
"""
r_sort(x, decreasing::Bool = false) = sort(x; rev = decreasing)

"""
    r_rowsums(m)
    r_colsums(m)
    r_rowmeans(m)
    r_colmeans(m)

Row/column sums and means of a matrix, R-style.

Mirror R's `rowSums()`/`colSums()`/`rowMeans()`/`colMeans()`: each returns a
plain **vector**. This differs from `sum(m; dims=...)` / `mean(m; dims=...)`,
which keep the reduced dimension (returning a 1×n or n×1 matrix).

# Examples
```julia
julia> r_rowsums([1 2; 3 4])
2-element Vector{Int64}:
 3
 7
```
"""
r_rowsums(m::AbstractMatrix)  = vec(sum(m, dims = 2))
r_colsums(m::AbstractMatrix)  = vec(sum(m, dims = 1))
r_rowmeans(m::AbstractMatrix) = vec(sum(m, dims = 2)) ./ size(m, 2)
r_colmeans(m::AbstractMatrix) = vec(sum(m, dims = 1)) ./ size(m, 1)

"""
    r_cummax(x)
    r_cummin(x)

Cumulative maximum / minimum, R-style.

Mirror R's `cummax()`/`cummin()`: element `i` of the result is the max/min of
`x[1:i]`. (Base Julia has `cumsum`/`cumprod` but no `cummax`/`cummin`.)

# Examples
```julia
julia> r_cummax([1, 3, 2, 5, 4])
5-element Vector{Int64}:
 1
 3
 3
 5
 5
```
"""
r_cummax(x) = accumulate(max, x)
r_cummin(x) = accumulate(min, x)

"""
    r_rep(x, times=1, length_out=-1, each=1)

Replicate elements of `x`, R-style.

Mirrors R's `rep()` for the common scalar forms. `each` repeats every element in
place; `times` tiles the whole (post-`each`) sequence; `length_out` (when `>= 0`)
recycles/truncates to that length and takes precedence over `times`. Arguments
are positional to match how the converter emits them.

# Examples
```julia
julia> r_rep([1, 2], 3)
6-element Vector{Int64}:
 1
 2
 1
 2
 1
 2
julia> r_rep([1, 2], 1, -1, 2)   # each = 2
4-element Vector{Int64}:
 1
 1
 2
 2
```
"""
function r_rep(x, times::Real = 1, length_out::Real = -1, each::Real = 1)
    t = Int(times)
    lo = Int(length_out)
    e = Int(each)
    base = x isa AbstractArray ? vec(collect(x)) : [x]
    y = e > 1 ? repeat(base, inner = e) : base
    if lo >= 0
        isempty(y) && return y
        return [y[((i - 1) % length(y)) + 1] for i in 1:lo]
    end
    return t == 1 ? y : repeat(y, t)
end

"""
    nonnegative(x)

Ensure value(s) are non-negative by returning max(0, x).

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
nonnegative(x::AbstractArray{<:Real}) = max.(0.0, x)

# ============================================================================
# Rounding Utilities
# ============================================================================

"""
    round_(x, digits=0)

Flexible rounding function.

# Examples
```julia
julia> round_(3.14159, digits=2)
3.14
julia> round_(3.14159, 2)
3.14
```
"""
round_(x, digits::Real) = round(x, digits=round(Int, digits))
round_(x; digits::Real=0) = round(x, digits=round(Int, digits))

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

"""
    with_rng(f, src)

Run `f()` with the task-local default RNG temporarily set from `src` — an
integer seed, a `Xoshiro`, or the `TaskLocalRNG` itself — then restore the
caller's RNG state. Inside `f`, bare `rand()` / `randn()` (and functions built
on them, e.g. [`rbool`](@ref), [`rdist`](@ref)) draw from the installed stream;
no rng argument is threaded through.

For a seed or a `Xoshiro`, the caller's global RNG state is saved before `f`
runs and restored afterwards, even if `f` throws. A `Xoshiro` passed as `src`
is not mutated.

Passing a `TaskLocalRNG` (i.e. `Random.default_rng()`) is a no-op passthrough:
it *is* the task-local stream, so bare `rand()`/`randn()` already draw from it
and there is nothing to install or restore. This is the case for the per-trajectory
`ctx.rng` of a SciML `EnsembleProblem` (new `prob_func(prob, ctx)` interface),
which SciML has already seeded for the trajectory — so `with_rng(ctx.rng) do … end`
runs `f` against that already-installed stream and advances it normally.

Any other `AbstractRNG` (e.g. `MersenneTwister`) is rejected: routing bare
`rand()`/`randn()` through it is not possible because the task-local default
RNG is a `Xoshiro`. For those, call `rand(rng, …)` / `randn(rng, …)` explicitly.

# Examples
```julia
julia> with_rng(() -> rand(3), 1234) == with_rng(() -> rand(3), 1234)
true

julia> Random.seed!(99); before = rand(2);

julia> with_rng(() -> rand(5), 1234);  # isolated; does not advance global stream

julia> Random.seed!(99); rand(2) == before
true
```
"""
function with_rng(f, seed::Integer)
    rng   = Random.default_rng()
    saved = copy(rng)
    Random.seed!(rng, seed)
    try
        return f()
    finally
        copy!(rng, saved)
    end
end

function with_rng(f, src::Xoshiro)
    rng   = Random.default_rng()
    saved = copy(rng)
    copy!(rng, src)            # install src's state; src itself is not mutated
    try
        return f()
    finally
        copy!(rng, saved)
    end
end

# The task-local default RNG is a singleton: when it is handed to us (e.g. an
# ensemble's per-trajectory ctx.rng, already seeded by SciML) bare rand()/randn()
# already draw from it, so there is nothing to install or restore — just run f().
with_rng(f, ::Random.TaskLocalRNG) = f()

function with_rng(f, src::AbstractRNG)
    error("with_rng can route bare rand()/randn() only through a seed, a Xoshiro, or " *
          "the TaskLocalRNG (the task-local default RNG). For a $(typeof(src)), pass a " *
          "seed/Xoshiro, or call rand(rng, …) / randn(rng, …) explicitly with your $(typeof(src)).")
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

"""
    ⊘(x, y)

Floor division operator (floor(x / y)).

Unicode alternative to `fld(x, y)`, matching R's `%/%` (floored, so the result
follows the sign of the divisor: `-7 ⊘ 2 == -4`).

# Examples
```julia
julia> 7 ⊘ 2
3
julia> -7 ⊘ 2
-4
```
"""
⊘(x, y) = fld(x, y)

end # module
