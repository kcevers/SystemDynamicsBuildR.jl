"""
    clean

Utilities for cleaning and formatting simulation results from DifferentialEquations.jl.

This module provides:
- Conversion of single (non-ensemble) solutions to DataFrame format
- Extraction of parameters and initial values
- Processing of intermediate calculation results
- Unit stripping for Unitful quantities
- Data interpolation at specific time points

Designed to work with DifferentialEquations.jl problem and solution objects.
"""
module clean

using Unitful
using DataFrames
using ..custom_func: itp, is_function_or_interp

export clean_df, clean_constants, clean_init, saveat_func

# ============================================================================
# Interpolation Utilities
# ============================================================================

"""
    saveat_func(t, y, new_times)

Interpolate solution values at specific time points.

Creates a linear interpolation of the solution and evaluates it at the
requested times. Uses nearest-neighbor extrapolation outside the time range.

# Arguments
- `t`: Original time points (vector)
- `y`: Original values (vector)
- `new_times`: New time points where interpolation is desired (vector)

# Returns
- Vector of interpolated values at `new_times`

# Examples
```julia
julia> t = [0.0, 1.0, 2.0]
julia> y = [10.0, 20.0, 30.0]
julia> new_t = [0.5, 1.5]
julia> saveat_func(t, y, new_t)
2-element Vector{Float64}:
 15.0
 25.0
```

# Notes
- Interpolation method: linear
- Extrapolation method: nearest neighbor (constant outside range)
- Automatically handles sorting of time points
"""
function saveat_func(t, y, new_times)
    f = itp(t, y, method="linear", extrapolation="nearest")
    return f.(new_times)
end

# ============================================================================
# DataFrame Conversion Functions
# ============================================================================

"""
    clean_df(prob, solve_out, init_names, intermediaries=nothing, intermediary_names=nothing)

Convert a single (non-ensemble) simulation result to tidy DataFrame format.

Extracts time series data, parameters, and initial values from a DifferentialEquations.jl
solution object. Supports both main solution and intermediate calculations.

# Arguments
- `prob`: Problem object from DifferentialEquations.jl
- `solve_out`: Solution object from DifferentialEquations.jl
- `init_names`: Names of state variables (Symbol or String vector)
- `intermediaries=nothing`: Optional intermediate values from SavingCallback
- `intermediary_names=nothing`: Optional names for intermediate variables

# Returns
Named tuple containing:
- `timeseries_df`: DataFrame with columns `[time, variable, value]`
- `param_values`: Vector of parameter values (Float64)
- `param_names`: Vector of parameter names (String)
- `init_values`: Vector of initial condition values (Float64)
- `init_names`: Vector of initial condition names (String)

# Examples
```julia
using DifferentialEquations

# Define and solve a simple ODE
function f!(du, u, p, t)
    du[1] = p.α * u[1]
end

u0 = [1.0]
tspan = (0.0, 10.0)
p = (α = -0.1,)
prob = ODEProblem(f!, u0, tspan, p)
sol = solve(prob)

# Convert to DataFrame
df, p_vals, p_names, u0_vals, u0_names = clean_df(
    prob, sol, [:x]
)

# Access data
using DataFrames
first(df, 5)  # First 5 rows
```

# Notes
- Units are automatically stripped from Unitful quantities
- Functions and interpolations in parameters are excluded
- Handles NamedTuple, Vector, and scalar parameter formats
- Supports multiple state variables
- Intermediate calculations are appended to time series if provided
"""
function clean_df(prob, solve_out, init_names, 
                  intermediaries=nothing, intermediary_names=nothing)
    
    # Extract parameter names and values
    param_names = String[]
    param_values = Float64[]
    params = prob.p

    if isa(params, NamedTuple)
        for (key, val) in pairs(params)
            if !is_function_or_interp(val)
                push!(param_names, string(key))
                val_stripped = isa(val, Quantity) ? ustrip(val) : Float64(val)
                push!(param_values, val_stripped)
            end
        end
    elseif isa(params, AbstractVector)
        for i in eachindex(params)
            if !is_function_or_interp(params[i])
                push!(param_names, "p$i")
                val_stripped = isa(params[i], Quantity) ? 
                    ustrip(params[i]) : Float64(params[i])
                push!(param_values, val_stripped)
            end
        end
    elseif isa(params, Number)
        push!(param_names, "p1")
        val_stripped = isa(params, Quantity) ? ustrip(params) : Float64(params)
        push!(param_values, val_stripped)
    end

    # Extract initial values
    init_values = Float64[]
    init_vals = prob.u0
    init_val_names = [string(name) for name in init_names]
    init_val_symbols = Symbol.(init_val_names)

    if isa(init_vals, NamedTuple)
        for init_sym in init_val_symbols
            init_val = getproperty(init_vals, init_sym)
            init_val_stripped = isa(init_val, Quantity) ? 
                ustrip(init_val) : Float64(init_val)
            push!(init_values, init_val_stripped)
        end
    elseif isa(init_vals, AbstractVector)
        for init_val in init_vals
            init_val_stripped = isa(init_val, Quantity) ? 
                ustrip(init_val) : Float64(init_val)
            push!(init_values, init_val_stripped)
        end
    else
        # Single initial value
        init_val_stripped = isa(init_vals, Quantity) ? 
            ustrip(init_vals) : Float64(init_vals)
        push!(init_values, init_val_stripped)
    end

    # Pre-allocate vectors for time series
    time_vec = Float64[]
    variable_vec = String[]
    value_vec = Float64[]

    # Handle empty solution
    if isempty(solve_out.t)
        timeseries_df = DataFrame(
            time = time_vec,
            variable = variable_vec,
            value = value_vec;
            copycols = false
        )
        return timeseries_df, param_values, param_names, init_values, init_val_names
    end

    has_time_units = isa(solve_out.t[1], Quantity)

    # Determine number of variables and their names
    if isa(solve_out.u[1], AbstractVector)
        n_vars = length(solve_out.u[1])
        var_names = [string(name) for name in init_names]
    else
        n_vars = 1
        var_names = [string(init_names[1])]
    end
    var_name_symbols = Symbol.(var_names)

    # Provide capacity hints to reduce push! reallocations.
    n_t = length(solve_out.t)
    base_rows = n_t * n_vars
    int_rows = if !isnothing(intermediaries) && !isnothing(intermediary_names) && !isempty(intermediaries.t)
        n_int_vars = isa(intermediaries.saveval[1], Union{AbstractVector, Tuple}) ? length(intermediaries.saveval[1]) : 1
        length(intermediaries.t) * n_int_vars
    else
        0
    end
    sizehint!(time_vec, base_rows + int_rows)
    sizehint!(variable_vec, base_rows + int_rows)
    sizehint!(value_vec, base_rows + int_rows)

    # Process main solution
    for (t_idx, t_raw) in enumerate(solve_out.t)
        t_val = has_time_units ? ustrip(t_raw) : Float64(t_raw)
        u_val = solve_out.u[t_idx]

        if isa(u_val, NamedTuple)
            # Handle NamedTuple format (e.g., (S=990.0, I=10.0, R=0.0))
            for (var_name, var_sym) in zip(var_names, var_name_symbols)
                var_val = getproperty(u_val, var_sym)
                if !isa(var_val, Function)
                    val_stripped = isa(var_val, Quantity) ? 
                        ustrip(var_val) : Float64(var_val)

                    push!(time_vec, t_val)
                    push!(variable_vec, var_name)
                    push!(value_vec, val_stripped)
                end
            end
        elseif isa(u_val, Union{AbstractVector, Tuple})
            # Multiple state variables in array/tuple format
            for (var_idx, var_val) in enumerate(u_val)
                if !isa(var_val, Function)
                    val_stripped = isa(var_val, Quantity) ? 
                        ustrip(var_val) : Float64(var_val)
                    var_name = var_idx <= length(var_names) ? 
                        var_names[var_idx] : "var_$var_idx"

                    push!(time_vec, t_val)
                    push!(variable_vec, var_name)
                    push!(value_vec, val_stripped)
                end
            end
        else
            # Single state variable
            if !isa(u_val, Function)
                val_stripped = isa(u_val, Quantity) ? 
                    ustrip(u_val) : Float64(u_val)

                push!(time_vec, t_val)
                push!(variable_vec, var_names[1])
                push!(value_vec, val_stripped)
            end
        end
    end

    # Process intermediaries if provided
    if !isnothing(intermediaries) && !isnothing(intermediary_names) && 
       !isempty(intermediaries.t)
        
        int_has_time_units = isa(intermediaries.t[1], Quantity)
        int_var_names = [string(name) for name in intermediary_names]

        for (t_idx, t_raw) in enumerate(intermediaries.t)
            t_val = int_has_time_units ? ustrip(t_raw) : Float64(t_raw)
            saved_val = intermediaries.saveval[t_idx]

            if isa(saved_val, Union{AbstractVector, Tuple})
                # Multiple intermediate variables
                for (var_idx, var_val) in enumerate(saved_val)
                    if !isa(var_val, Function)
                        val_stripped = isa(var_val, Quantity) ? 
                            ustrip(var_val) : Float64(var_val)
                        var_name = var_idx <= length(int_var_names) ? 
                            int_var_names[var_idx] : "int_var_$var_idx"

                        push!(time_vec, t_val)
                        push!(variable_vec, var_name)
                        push!(value_vec, val_stripped)
                    end
                end
            else
                # Single intermediate variable
                if !isa(saved_val, Function)
                    val_stripped = isa(saved_val, Quantity) ? 
                        ustrip(saved_val) : Float64(saved_val)

                    push!(time_vec, t_val)
                    push!(variable_vec, int_var_names[1])
                    push!(value_vec, val_stripped)
                end
            end
        end
    end

    # Create DataFrame
    timeseries_df = DataFrame(
        time = time_vec,
        variable = variable_vec,
        value = value_vec;
        copycols = false
    )

    return timeseries_df, param_values, param_names, init_values, init_val_names
end

# ============================================================================
# Data Cleaning Functions
# ============================================================================

"""
    clean_constants(constants)

Clean and filter a NamedTuple of constants, removing units and non-numeric values.

Strips units from Unitful quantities and filters out values that are not
Float64 or Vector types. Useful for preparing constants for use in solvers
or exports.

# Arguments
- `constants`: NamedTuple containing constant values

# Returns
- Filtered NamedTuple containing only Float64 and Vector values with units stripped

# Examples
```julia
julia> constants = (
    a = 1.0u"m",
    b = 2.0,
    c = [1.0, 2.0],
    d = "string",  # Will be filtered out
    e = sin        # Will be filtered out
)

julia> clean_constants(constants)
(a = 1.0, b = 2.0, c = [1.0, 2.0])
```

# Notes
- Only Float64 and Vector values are retained
- Functions, strings, and other types are removed
- Units are stripped from Unitful quantities
- Order of keys may not be preserved
"""
function clean_constants(constants)
    # Strip units from all values
    constants = (; (
        name => isa(val, Unitful.Quantity) ? Unitful.ustrip(val) : val 
        for (name, val) in pairs(constants)
    )...)

    # Find keys where values are Float64 or Vector
    valid_keys = [
        k for k in keys(constants) 
        if isa(constants[k], Float64) || isa(constants[k], Vector)
    ]

    # Convert valid_keys to a tuple for NamedTuple construction
    valid_keys_tuple = Tuple(valid_keys)

    # Reconstruct filtered named tuple
    filtered_constants = NamedTuple{valid_keys_tuple}(
        constants[k] for k in valid_keys
    )

    return filtered_constants
end

"""
    clean_init(init, init_names)

Create a dictionary mapping initial condition names to their unitless values.

Strips units from initial values and creates a name-value mapping suitable
for analysis or export.

# Arguments
- `init`: Vector of initial condition values (may have Unitful units)
- `init_names`: Vector of names for initial conditions (Symbols or Strings)

# Returns
- Dict mapping names (as Symbols) to unitless Float64 values

# Examples
```julia
julia> init = [10.0u"m", 5.0u"m/s"]
julia> init_names = [:position, :velocity]
julia> clean_init(init, init_names)
Dict{Symbol, Float64} with 2 entries:
  :position => 10.0
  :velocity => 5.0

julia> init = [1.0, 2.0, 3.0]  # No units
julia> init_names = [:S, :I, :R]
julia> clean_init(init, init_names)
Dict{Symbol, Float64} with 3 entries:
  :S => 1.0
  :I => 2.0
  :R => 3.0
```

# Notes
- Automatically strips units from Unitful quantities
- Returns Float64 values even if input is Int
- Names are converted to Symbols if provided as Strings
"""
function clean_init(init, init_names)
    return Dict(init_names .=> Unitful.ustrip.(init))
end

end # module