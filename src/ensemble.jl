"""
    ensemble

Utilities for ensemble simulations and analysis in system dynamics models.

This module provides:
- Parameter combination generation for sensitivity analysis
- Conversion of simulation results to DataFrame format
- Statistical summaries across ensemble trajectories
- Both single-threaded and multi-threaded implementations for performance

Ensemble simulations allow exploring model behavior across parameter spaces
and quantifying uncertainty through multiple replicate runs.
"""
module ensemble

using Unitful
using Statistics
using DataFrames
using ..custom_func: is_function_or_interp

export generate_param_combinations
export ensemble_to_df, ensemble_to_df_threaded
export ensemble_summ, ensemble_summ_threaded
export transform_intermediaries

@inline function _group_replicate_indices(traj_ids::AbstractVector{Int}, ensemble_n::Int)
    n = length(traj_ids)
    j_vec = Vector{Int}(undef, n)
    i_vec = Vector{Int}(undef, n)
    @inbounds for k in 1:n
        traj = traj_ids[k]
        j_vec[k] = div(traj - 1, ensemble_n) + 1
        i_vec[k] = rem(traj - 1, ensemble_n) + 1
    end
    return j_vec, i_vec
end

# ============================================================================
# Data Transformation Functions
# ============================================================================

"""
    transform_intermediaries(intermediaries, intermediary_names=nothing)

Transform intermediate calculation results to match the format of solution outputs.

Creates pseudo-solution objects that can be processed with the same logic as
main simulation results, enabling unified data processing.

# Arguments
- `intermediaries`: Vector of intermediate calculation results from simulations
- `intermediary_names=nothing`: Optional names for the intermediate variables

# Returns
- Vector of pseudo-solution objects with structure `(t=..., u=..., p=nothing)`

# Details
Each pseudo-solution contains:
- `t`: Time points where intermediates were saved
- `u`: Saved intermediate values at those times
- `p`: Parameters (always `nothing` for intermediaries)

Empty trajectories are replaced with empty pseudo-solutions for consistency.
"""
function transform_intermediaries(intermediaries, intermediary_names=nothing)
    n = length(intermediaries)
    transformed = Vector{Any}(undef, n)

    for (traj_idx, intermediate_vals) in enumerate(intermediaries)
        if !isnothing(intermediate_vals) && !isempty(intermediate_vals.t)
            # Create a pseudo-solution object with the same structure as solve_out
            pseudo_solution = (
                t = intermediate_vals.t,
                u = intermediate_vals.saveval,
                p = nothing  # intermediaries don't have parameters
            )
            transformed[traj_idx] = pseudo_solution
        else
            # Create empty pseudo-solution for consistency
            transformed[traj_idx] = (t=Float64[], u=Float64[], p=nothing)
        end
    end

    return transformed
end

# ============================================================================
# Parameter Combination Generation
# ============================================================================

"""
    generate_param_combinations(param_ranges; crossed=true, n_replicates=100)

Generate parameter combinations for ensemble simulations.

Creates parameter sets for exploring model behavior across a parameter space,
supporting both full factorial (crossed) and paired designs.

# Arguments
- `param_ranges`: Dict or NamedTuple mapping parameter names to ranges/vectors
- `crossed::Bool=true`: If true, generate all combinations (Cartesian product);
  if false, pair parameters element-wise (requires equal-length ranges)
- `n_replicates::Int=100`: Number of replicate simulations per parameter set

# Returns
- `param_combinations`: Vector of parameter value vectors (one per combination)
- `total_sims`: Total number of simulations (combinations × replicates)

# Examples
```julia
# Crossed design (full factorial)
param_ranges = Dict(
    :alpha => [0.1, 0.5, 1.0],
    :beta => [2.0, 5.0]
)
combinations, total = generate_param_combinations(
    param_ranges; crossed=true, n_replicates=50
)
# Produces: 3 × 2 = 6 combinations, 6 × 50 = 300 total simulations

# Non-crossed design (paired parameters)
param_ranges = Dict(
    :alpha => [0.1, 0.5, 1.0],
    :beta => [2.0, 5.0, 10.0]  # Must match length
)
combinations, total = generate_param_combinations(
    param_ranges; crossed=false, n_replicates=100
)
# Produces: 3 combinations, 3 × 100 = 300 total simulations
```

# Throws
- `ArgumentError`: If `crossed=false` and parameter ranges have different lengths
"""
function generate_param_combinations(param_ranges;
                                      crossed=true, 
                                      n_replicates=100)
    # Sort keys for consistent ordering
    names_list = sort(collect(keys(param_ranges)))
    values_list = [param_ranges[name] for name in names_list]

    # Generate parameter combinations
    if crossed
        # All combinations (Cartesian product)
        param_combinations = collect(Iterators.product(values_list...))
        param_combinations = [collect(combo) for combo in vec(param_combinations)]
    else
        # Paired combinations (requires all ranges to have same length)
        lengths = [length(range) for range in values_list]
        if !all(l == lengths[1] for l in lengths)
            throw(ArgumentError(
                "For non-crossed design, all parameter ranges must have the same length. " *
                "Got lengths: $(lengths)"
            ))
        end
        param_combinations = [
            [values_list[i][j] for i in 1:length(values_list)] 
            for j in 1:lengths[1]
        ]
    end

    # Calculate total simulations
    total_sims = length(param_combinations) * n_replicates

    return param_combinations, total_sims
end

# ============================================================================
# DataFrame Conversion Functions
# ============================================================================

"""
    ensemble_to_df(solve_out, init_names, intermediaries, intermediary_names, ensemble_n)

Convert ensemble simulation results to long-format DataFrames.

Processes simulation outputs into tidy DataFrames suitable for analysis and
visualization, with separate tables for time series, parameters, and initial values.

# Arguments
- `solve_out`: Vector of solution objects from ensemble simulations
- `init_names`: Names of state variables
- `intermediaries`: Vector of intermediate calculation results (or `nothing`)
- `intermediary_names`: Names of intermediate variables (or `nothing`)
- `ensemble_n`: Number of replicates per parameter combination

# Returns
Named tuple containing:
- `timeseries_df`: Time series data with columns `j, i, time, variable, value`
  - `j`: Parameter combination index
  - `i`: Replicate index within combination
  - `time`: Simulation time
  - `variable`: Variable name
  - `value`: Variable value
- `param_df`: Parameter values with columns `j, i, variable, value`
- `init_df`: Initial values with columns `j, i, variable, value`

# Examples
```julia
# After running ensemble simulations
timeseries, params, inits = ensemble_to_df(
    results, 
    [:S, :I, :R],  # State variable names
    nothing,       # No intermediaries
    nothing,
    100            # 100 replicates per condition
)

# Access data
using DataFrames
subset(timeseries, :variable => ByRow(==("I")))  # Get infected counts
```

# Notes
- Units are automatically stripped from Unitful quantities
- Functions and interpolations in parameters are excluded
- Empty trajectories are handled gracefully
"""
function ensemble_to_df(solve_out, init_names,
                        intermediaries, intermediary_names, ensemble_n)
    # Get dimensions from first trajectory
    first_result = solve_out[1]

    # Determine number of variables and their names
    if isa(first_result.u[1], AbstractVector)
        var_names = [string(name) for name in init_names]
    else
        var_names = [string(init_names[1])]
    end

    # Transform intermediaries to solve_out format
    transformed_intermediaries = nothing
    if !isnothing(intermediaries)
        transformed_intermediaries = transform_intermediaries(intermediaries, intermediary_names)
    end

    # Helper function to process solution-like data
    function process_solution_like(solutions, var_names_to_use)
        if isnothing(solutions)
            return Int[], Float64[], String[], Float64[]
        end

        # Calculate total rows needed
        total_rows = 0
        for sol in solutions
            if !isempty(sol.t)
                if isa(sol.u[1], Union{AbstractVector, Tuple})
                    total_rows += length(sol.t) * length(sol.u[1])
                else
                    total_rows += length(sol.t)
                end
            end
        end

        if total_rows == 0
            return Int[], Float64[], String[], Float64[]
        end

        # Pre-allocate output vectors
        trajectory_vec = Vector{Int}(undef, total_rows)
        time_vec = Vector{Float64}(undef, total_rows)
        variable_vec = Vector{String}(undef, total_rows)
        value_vec = Vector{Float64}(undef, total_rows)

        row_idx = 1

        # Process each trajectory
        for (traj_idx, result) in enumerate(solutions)
            if !isempty(result.t)
                has_time_units = isa(result.t[1], Quantity)

                for (t_idx, t_raw) in enumerate(result.t)
                    t_val = has_time_units ? ustrip(t_raw) : Float64(t_raw)
                    u_val = result.u[t_idx]

                    if isa(u_val, Union{AbstractVector, Tuple})
                        # Multiple variables
                        for (var_idx, var_val) in enumerate(u_val)
                            if !isa(var_val, Function)
                                val_stripped = isa(var_val, Quantity) ? ustrip(var_val) : Float64(var_val)
                                var_name = var_idx <= length(var_names_to_use) ? 
                                    var_names_to_use[var_idx] : "var_$var_idx"

                                trajectory_vec[row_idx] = traj_idx
                                time_vec[row_idx] = t_val
                                variable_vec[row_idx] = var_name
                                value_vec[row_idx] = val_stripped
                                row_idx += 1
                            end
                        end
                    else
                        # Single variable
                        if !isa(u_val, Function)
                            val_stripped = isa(u_val, Quantity) ? ustrip(u_val) : Float64(u_val)
                            var_name = var_names_to_use[1]

                            trajectory_vec[row_idx] = traj_idx
                            time_vec[row_idx] = t_val
                            variable_vec[row_idx] = var_name
                            value_vec[row_idx] = val_stripped
                            row_idx += 1
                        end
                    end
                end
            end
        end

        # Trim to actual size
        resize!(trajectory_vec, row_idx - 1)
        resize!(time_vec, row_idx - 1)
        resize!(variable_vec, row_idx - 1)
        resize!(value_vec, row_idx - 1)

        return trajectory_vec, time_vec, variable_vec, value_vec
    end

    # Process main solution
    main_traj, main_time, main_var, main_val = process_solution_like(solve_out, var_names)

    # Process intermediaries if present
    if !isnothing(transformed_intermediaries)
        int_var_names = [string(name) for name in intermediary_names]
        int_traj, int_time, int_var, int_val = process_solution_like(
            transformed_intermediaries, int_var_names
        )

        # Combine all data
        append!(main_traj, int_traj)
        append!(main_time, int_time)
        append!(main_var, int_var)
        append!(main_val, int_val)
    end

    # Create time series DataFrame
    j_vec, i_vec = _group_replicate_indices(main_traj, ensemble_n)
    timeseries_df = DataFrame(
        j = j_vec,  # Parameter combination index
        i = i_vec,  # Replicate index
        time = main_time,
        variable = main_var,
        value = main_val;
        copycols = false
    )

    # Extract parameter names (excluding functions/interpolations)
    param_names = String[]
    param_symbols = Symbol[]
    param_indices = Int[]
    params_are_namedtuple = false
    first_params = solve_out[1].p
    if isa(first_params, NamedTuple)
        params_are_namedtuple = true
        for (key, val) in pairs(first_params)
            if !is_function_or_interp(val)
                push!(param_names, string(key))
                push!(param_symbols, key)
            end
        end
    elseif isa(first_params, AbstractVector)
        for i in eachindex(first_params)
            if !is_function_or_interp(first_params[i])
                push!(param_names, "p$i")
                push!(param_indices, i)
            end
        end
    end

    # Create parameters DataFrame
    param_df = DataFrame()
    if !isempty(param_names)
        n_trajectories = length(solve_out)
        n_params = length(param_names)
        total_param_rows = n_trajectories * n_params
        param_j_vec = Vector{Int}(undef, total_param_rows)
        param_i_vec = Vector{Int}(undef, total_param_rows)
        param_name_vec = Vector{String}(undef, total_param_rows)
        param_value_vec = Vector{Float64}(undef, total_param_rows)

        row_idx = 1

        for (traj_idx, result) in enumerate(solve_out)
            params = result.p
            for param_idx in eachindex(param_names)
                if params_are_namedtuple
                    param_val = getproperty(params, param_symbols[param_idx])
                else
                    param_val = params[param_indices[param_idx]]
                end

                param_val_stripped = isa(param_val, Quantity) ? ustrip(param_val) : Float64(param_val)

                param_j_vec[row_idx] = div(traj_idx - 1, ensemble_n) + 1
                param_i_vec[row_idx] = rem(traj_idx - 1, ensemble_n) + 1
                param_name_vec[row_idx] = param_names[param_idx]
                param_value_vec[row_idx] = param_val_stripped
                row_idx += 1
            end
        end

        param_df = DataFrame(
            j = param_j_vec,
            i = param_i_vec,
            variable = param_name_vec,
            value = param_value_vec;
            copycols = false
        )
    end

    # Extract initial values
    init_val_names = [string(name) for name in init_names]

    init_df = DataFrame()
    if !isempty(init_val_names)
        n_trajectories = length(solve_out)
        n_inits = length(init_val_names)
        total_init_rows = n_trajectories * n_inits
        init_j_vec = Vector{Int}(undef, total_init_rows)
        init_i_vec = Vector{Int}(undef, total_init_rows)
        init_name_vec = Vector{String}(undef, total_init_rows)
        init_value_vec = Vector{Float64}(undef, total_init_rows)
        init_name_symbols = Symbol.(init_val_names)

        row_idx = 1

        for (traj_idx, result) in enumerate(solve_out)
            init_vals = result.u0

            if isa(init_vals, NamedTuple)
                for init_idx in eachindex(init_val_names)
                    init_val = getproperty(init_vals, init_name_symbols[init_idx])
                    init_val_stripped = isa(init_val, Quantity) ? ustrip(init_val) : Float64(init_val)

                    init_j_vec[row_idx] = div(traj_idx - 1, ensemble_n) + 1
                    init_i_vec[row_idx] = rem(traj_idx - 1, ensemble_n) + 1
                    init_name_vec[row_idx] = init_val_names[init_idx]
                    init_value_vec[row_idx] = init_val_stripped
                    row_idx += 1
                end
            elseif isa(init_vals, AbstractVector)
                for (init_idx, init_name) in enumerate(init_val_names)
                    init_val = init_vals[init_idx]
                    init_val_stripped = isa(init_val, Quantity) ? ustrip(init_val) : Float64(init_val)

                    init_j_vec[row_idx] = div(traj_idx - 1, ensemble_n) + 1
                    init_i_vec[row_idx] = rem(traj_idx - 1, ensemble_n) + 1
                    init_name_vec[row_idx] = init_name
                    init_value_vec[row_idx] = init_val_stripped
                    row_idx += 1
                end
            else
                # Single initial value
                init_val_stripped = isa(init_vals, Quantity) ? ustrip(init_vals) : Float64(init_vals)

                init_j_vec[row_idx] = div(traj_idx - 1, ensemble_n) + 1
                init_i_vec[row_idx] = rem(traj_idx - 1, ensemble_n) + 1
                init_name_vec[row_idx] = init_val_names[1]
                init_value_vec[row_idx] = init_val_stripped
                row_idx += 1
            end
        end

        if row_idx <= total_init_rows
            resize!(init_j_vec, row_idx - 1)
            resize!(init_i_vec, row_idx - 1)
            resize!(init_name_vec, row_idx - 1)
            resize!(init_value_vec, row_idx - 1)
        end

        init_df = DataFrame(
            j = init_j_vec,
            i = init_i_vec,
            variable = init_name_vec,
            value = init_value_vec;
            copycols = false
        )
    end

    return timeseries_df, param_df, init_df
end

"""
    ensemble_to_df_threaded(solve_out, init_names, intermediaries, intermediary_names, ensemble_n)

Multi-threaded version of `ensemble_to_df` for improved performance on large ensembles.

Uses Julia's multi-threading to parallelize data processing across CPU cores.
Significantly faster for large ensembles (>1000 trajectories).

# Arguments
Same as `ensemble_to_df`.

# Returns
Same as `ensemble_to_df`.

# Performance Notes
- Requires Julia to be started with multiple threads: `julia --threads=auto`
- Check available threads: `Threads.nthreads()`
- Most beneficial for large ensembles (>1000 trajectories)
- Memory usage slightly higher due to pre-allocation

# Examples
```julia
# Start Julia with: julia --threads=auto
# or set JULIA_NUM_THREADS environment variable

println("Using ", Base.Threads.nthreads(), " threads")

# Process large ensemble efficiently
timeseries, params, inits = ensemble_to_df_threaded(
    large_results,
    [:S, :I, :R],
    nothing, nothing,
    1000  # 1000 replicates
)
```
"""
function ensemble_to_df_threaded(solve_out, init_names, 
                                  intermediaries, intermediary_names, ensemble_n)
    # Similar structure to ensemble_to_df but with threading
    n_trajectories = length(solve_out)

    first_result = solve_out[1]

    if isa(first_result.u[1], AbstractVector)
        var_names = [string(name) for name in init_names]
    else
        var_names = [string(init_names[1])]
    end

    transformed_intermediaries = nothing
    if !isnothing(intermediaries)
        transformed_intermediaries = transform_intermediaries(intermediaries, intermediary_names)
    end

    function process_solution_like(solutions, var_names_to_use)
        if isnothing(solutions)
            return Int[], Float64[], String[], Float64[]
        end

        # First pass: calculate row counts for each trajectory (threaded)
        row_counts = Vector{Int}(undef, length(solutions))

        Base.Threads.@threads for i in 1:length(solutions)
            sol = solutions[i]
            count = 0
            if !isempty(sol.t)
                for u_val in sol.u
                    if isa(u_val, Union{AbstractVector, Tuple})
                        for var_val in u_val
                            if !isa(var_val, Function)
                                count += 1
                            end
                        end
                    elseif !isa(u_val, Function)
                        count += 1
                    end
                end
            end
            row_counts[i] = count
        end

        total_rows = sum(row_counts)

        if total_rows == 0
            return Int[], Float64[], String[], Float64[]
        end

        # Pre-allocate output arrays
        trajectory_vec = Vector{Int}(undef, total_rows)
        time_vec = Vector{Float64}(undef, total_rows)
        variable_vec = Vector{String}(undef, total_rows)
        value_vec = Vector{Float64}(undef, total_rows)

        # Calculate start indices for each trajectory
        start_indices = Vector{Int}(undef, length(solutions))
        start_indices[1] = 1
        for i in 2:length(solutions)
            start_indices[i] = start_indices[i-1] + row_counts[i-1]
        end

        # Second pass: fill arrays in parallel
        Base.Threads.@threads for traj_idx in 1:length(solutions)
            result = solutions[traj_idx]
            if !isempty(result.t)
                has_time_units = isa(result.t[1], Quantity)

                row_idx = start_indices[traj_idx]

                for (t_idx, t_raw) in enumerate(result.t)
                    t_val = has_time_units ? ustrip(t_raw) : Float64(t_raw)
                    u_val = result.u[t_idx]

                    if isa(u_val, Union{AbstractVector, Tuple})
                        for (var_idx, var_val) in enumerate(u_val)
                            if !isa(var_val, Function)
                                val_stripped = isa(var_val, Quantity) ? ustrip(var_val) : Float64(var_val)
                                var_name = var_idx <= length(var_names_to_use) ? 
                                    var_names_to_use[var_idx] : "var_$var_idx"

                                trajectory_vec[row_idx] = traj_idx
                                time_vec[row_idx] = t_val
                                variable_vec[row_idx] = var_name
                                value_vec[row_idx] = val_stripped
                                row_idx += 1
                            end
                        end
                    else
                        if !isa(u_val, Function)
                            val_stripped = isa(u_val, Quantity) ? ustrip(u_val) : Float64(u_val)
                            var_name = var_names_to_use[1]

                            trajectory_vec[row_idx] = traj_idx
                            time_vec[row_idx] = t_val
                            variable_vec[row_idx] = var_name
                            value_vec[row_idx] = val_stripped
                            row_idx += 1
                        end
                    end
                end
            end
        end

        return trajectory_vec, time_vec, variable_vec, value_vec
    end

    # Process main solution
    main_traj, main_time, main_var, main_val = process_solution_like(solve_out, var_names)

    # Process intermediaries
    if !isnothing(transformed_intermediaries)
        int_var_names = [string(name) for name in intermediary_names]
        int_traj, int_time, int_var, int_val = process_solution_like(
            transformed_intermediaries, int_var_names
        )

        append!(main_traj, int_traj)
        append!(main_time, int_time)
        append!(main_var, int_var)
        append!(main_val, int_val)
    end

    # Create DataFrame
    j_vec, i_vec = _group_replicate_indices(main_traj, ensemble_n)
    timeseries_df = DataFrame(
        j = j_vec,
        i = i_vec,
        time = main_time,
        variable = main_var,
        value = main_val;
        copycols = false
    )

    # Extract parameter names
    param_names = String[]
    param_symbols = Symbol[]
    param_indices = Int[]
    params_are_namedtuple = false
    first_params = solve_out[1].p
    if isa(first_params, NamedTuple)
        params_are_namedtuple = true
        for (key, val) in pairs(first_params)
            if !is_function_or_interp(val)
                push!(param_names, string(key))
                push!(param_symbols, key)
            end
        end
    elseif isa(first_params, AbstractVector)
        for i in eachindex(first_params)
            if !is_function_or_interp(first_params[i])
                push!(param_names, "p$i")
                push!(param_indices, i)
            end
        end
    end

    # Create parameters DataFrame (threaded)
    param_df = DataFrame()
    if !isempty(param_names)
        total_param_rows = n_trajectories * length(param_names)
        param_j_vec = Vector{Int}(undef, total_param_rows)
        param_i_vec = Vector{Int}(undef, total_param_rows)
        param_name_vec = Vector{String}(undef, total_param_rows)
        param_value_vec = Vector{Float64}(undef, total_param_rows)

        Base.Threads.@threads for traj_idx in 1:n_trajectories
            result = solve_out[traj_idx]
            params = result.p

            for (param_idx, param_name) in enumerate(param_names)
                row_idx = (traj_idx - 1) * length(param_names) + param_idx

                if params_are_namedtuple
                    param_val = getproperty(params, param_symbols[param_idx])
                else
                    param_val = params[param_indices[param_idx]]
                end

                param_val_stripped = isa(param_val, Quantity) ? ustrip(param_val) : Float64(param_val)

                param_j_vec[row_idx] = div(traj_idx - 1, ensemble_n) + 1
                param_i_vec[row_idx] = rem(traj_idx - 1, ensemble_n) + 1
                param_name_vec[row_idx] = param_name
                param_value_vec[row_idx] = param_val_stripped
            end
        end

        param_df = DataFrame(
            j = param_j_vec,
            i = param_i_vec,
            variable = param_name_vec,
            value = param_value_vec;
            copycols = false
        )
    end

    # Extract initial values (threaded)
    init_val_names = [string(name) for name in init_names]
    init_val_symbols = Symbol.(init_val_names)

    init_df = DataFrame()
    if !isempty(init_val_names)
        total_init_rows = n_trajectories * length(init_val_names)
        init_j_vec = Vector{Int}(undef, total_init_rows)
        init_i_vec = Vector{Int}(undef, total_init_rows)
        init_name_vec = Vector{String}(undef, total_init_rows)
        init_value_vec = Vector{Float64}(undef, total_init_rows)

        Base.Threads.@threads for traj_idx in 1:n_trajectories
            result = solve_out[traj_idx]
            init_vals = result.u0

            if isa(init_vals, NamedTuple)
                for (init_idx, init_name) in enumerate(init_val_names)
                    row_idx = (traj_idx - 1) * length(init_val_names) + init_idx
                    init_val = getproperty(init_vals, init_val_symbols[init_idx])
                    init_val_stripped = isa(init_val, Quantity) ? ustrip(init_val) : Float64(init_val)

                    init_j_vec[row_idx] = div(traj_idx - 1, ensemble_n) + 1
                    init_i_vec[row_idx] = rem(traj_idx - 1, ensemble_n) + 1
                    init_name_vec[row_idx] = init_name
                    init_value_vec[row_idx] = init_val_stripped
                end
            elseif isa(init_vals, AbstractVector)
                for (init_idx, init_name) in enumerate(init_val_names)
                    row_idx = (traj_idx - 1) * length(init_val_names) + init_idx
                    init_val = init_vals[init_idx]
                    init_val_stripped = isa(init_val, Quantity) ? ustrip(init_val) : Float64(init_val)

                    init_j_vec[row_idx] = div(traj_idx - 1, ensemble_n) + 1
                    init_i_vec[row_idx] = rem(traj_idx - 1, ensemble_n) + 1
                    init_name_vec[row_idx] = init_name
                    init_value_vec[row_idx] = init_val_stripped
                end
            else
                # Single initial value
                row_idx = (traj_idx - 1) * length(init_val_names) + 1
                init_val_stripped = isa(init_vals, Quantity) ? ustrip(init_vals) : Float64(init_vals)

                init_j_vec[row_idx] = div(traj_idx - 1, ensemble_n) + 1
                init_i_vec[row_idx] = rem(traj_idx - 1, ensemble_n) + 1
                init_name_vec[row_idx] = init_val_names[1]
                init_value_vec[row_idx] = init_val_stripped
            end
        end

        init_df = DataFrame(
            j = init_j_vec,
            i = init_i_vec,
            variable = init_name_vec,
            value = init_value_vec;
            copycols = false
        )
    end

    return timeseries_df, param_df, init_df
end

# ============================================================================
# Statistical Summary Functions
# ============================================================================

"""
    ensemble_summ(timeseries_df, quantiles=[0.025, 0.975])

Compute summary statistics across ensemble trajectories.

Calculates mean, median, variance, and quantiles for each variable at each
time point and parameter combination.

# Arguments
- `timeseries_df`: DataFrame from `ensemble_to_df` or `ensemble_to_df_threaded`
- `quantiles::Vector=[0.025, 0.975]`: Quantiles to compute (default: 95% interval)

# Returns
DataFrame with columns:
- `j`: Parameter combination index
- `time`: Simulation time
- `variable`: Variable name
- `mean`: Mean across replicates
- `median`: Median across replicates
- `variance`: Variance across replicates
- `missing_count`: Number of missing/NaN values
- `q<quantile>`: Quantile columns (e.g., `q025`, `q975`)

# Examples
```julia
# Get summary statistics with default 95% interval
stats = ensemble_summ(timeseries_df)

# Custom quantiles (90% interval)
stats = ensemble_summ(timeseries_df, [0.05, 0.95])

# Access results
using DataFrames
subset(stats, :variable => ByRow(==("I")))  # Infected summary
```

# Notes
- Missing and NaN values are excluded from statistics
- If all values are missing/NaN, statistics are NaN
- Quantile columns named by removing leading "0." (e.g., 0.025 → q025)
"""
function ensemble_summ(timeseries_df, quantiles=[0.025, 0.975])
    # Group by parameter combination, time, and variable
    stats_df = combine(groupby(timeseries_df, [:j, :time, :variable])) do group
        values = group.value

        # Filter out missing and NaN
        is_valid = .!(ismissing.(values) .| isnan.(values))
        clean_values = values[is_valid]
        num_missing = count(!, is_valid)

        if isempty(clean_values)
            # Return NaNs if no valid values
            result = (
                mean = NaN,
                median = NaN,
                variance = NaN,
                missing_count = num_missing
            )

            for q in quantiles
                q_str = replace(string(q), r"^0\." => "")
                result = merge(result, (Symbol("q$q_str") => NaN,))
            end
        else
            # Compute statistics
            result = (
                mean = mean(clean_values),
                median = Statistics.median(clean_values),
                variance = var(clean_values),
                missing_count = num_missing
            )

            for q in quantiles
                q_str = replace(string(q), r"^0\." => "")
                result = merge(result, (Symbol("q$q_str") => Statistics.quantile(clean_values, q),))
            end
        end

        return result
    end

    return stats_df
end

"""
    ensemble_summ_threaded(timeseries_df, quantiles=[0.025, 0.975])

Multi-threaded version of `ensemble_summ` for improved performance.

Computes summary statistics in parallel across CPU cores. Significantly
faster for large datasets with many groups.

# Arguments
Same as `ensemble_summ`.

# Returns
Same as `ensemble_summ`.

# Performance Notes
- Requires Julia with multiple threads: `julia --threads=auto`
- Most beneficial for large datasets (>10,000 groups)
- Memory usage similar to single-threaded version

# Examples
```julia
# Start Julia with multiple threads
using Base.Threads
println("Using ", Base.Threads.nthreads(), " threads")

# Compute statistics efficiently
stats = ensemble_summ_threaded(large_timeseries_df)
```
"""
function ensemble_summ_threaded(timeseries_df, quantiles=[0.025, 0.975])
    # Group the data
    grouped_df = groupby(timeseries_df, [:j, :time, :variable])

    # Get group keys and create result arrays
    group_keys = keys(grouped_df)
    n_groups = length(group_keys)

    # Pre-allocate result arrays
    j_vals = Vector{Int}(undef, n_groups)
    time_vals = Vector{Float64}(undef, n_groups)
    variable_vals = Vector{String}(undef, n_groups)
    mean_vals = Vector{Float64}(undef, n_groups)
    variance_vals = Vector{Float64}(undef, n_groups)
    median_vals = Vector{Float64}(undef, n_groups)
    missing_counts = Vector{Int}(undef, n_groups)

    # Pre-allocate quantile arrays
    quantile_arrays = Dict{String, Vector{Float64}}()
    for q in quantiles
        q_str = replace(string(q), r"^0\." => "")
        quantile_arrays["q$q_str"] = Vector{Float64}(undef, n_groups)
    end

    # Process groups in parallel
    Base.Threads.@threads for i in 1:n_groups
        group = grouped_df[i]
        key = group_keys[i]
        values = group.value

        # Filter NaN/missing
        is_valid = .!(ismissing.(values) .| isnan.(values))
        clean_values = values[is_valid]
        num_missing = count(!, is_valid)

        # Extract group keys
        j_vals[i] = key.j
        time_vals[i] = key.time
        variable_vals[i] = key.variable

        # Compute statistics
        if isempty(clean_values)
            mean_vals[i] = NaN
            variance_vals[i] = NaN
            median_vals[i] = NaN
            for q in quantiles
                q_str = replace(string(q), r"^0\." => "")
                quantile_arrays["q$q_str"][i] = NaN
            end
        else
            mean_vals[i] = mean(clean_values)
            variance_vals[i] = var(clean_values)
            median_vals[i] = Statistics.median(clean_values)
            for q in quantiles
                q_str = replace(string(q), r"^0\." => "")
                quantile_arrays["q$q_str"][i] = Statistics.quantile(clean_values, q)
            end
        end

        missing_counts[i] = num_missing
    end

    # Create result DataFrame with ordered columns
    stats_df = DataFrame(
        j = j_vals,
        time = time_vals,
        variable = variable_vals,
        mean = mean_vals,
        median = median_vals,
        variance = variance_vals,
        missing_count = missing_counts;
        copycols = false
    )

    # Add quantile columns in order
    for q in quantiles
        q_str = replace(string(q), r"^0\." => "")
        stats_df[!, Symbol("q$q_str")] = quantile_arrays["q$q_str"]
    end

    return stats_df
end

end # module