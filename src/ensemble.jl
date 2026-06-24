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
"""
function transform_intermediaries(intermediaries, _intermediary_names=nothing)
    n = length(intermediaries)
    transformed = Vector{Any}(undef, n)

    for (traj_idx, intermediate_vals) in enumerate(intermediaries)
        if !isnothing(intermediate_vals) && !isempty(intermediate_vals.t)
            pseudo_solution = (
                t = intermediate_vals.t,
                u = intermediate_vals.saveval,
                p = nothing
            )
            transformed[traj_idx] = pseudo_solution
        else
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

# Arguments
- `param_ranges`: Dict or NamedTuple mapping parameter names to ranges/vectors
- `crossed::Bool=true`: If true, generate all combinations (Cartesian product);
  if false, pair parameters element-wise (requires equal-length ranges)
- `n_replicates::Int=100`: Number of replicate simulations per parameter set

# Returns
- `param_combinations`: Vector of parameter value vectors (one per combination)
- `total_sims`: Total number of simulations (combinations × replicates)

# Throws
- `ArgumentError`: If `crossed=false` and parameter ranges have different lengths
"""
function generate_param_combinations(param_ranges;
                                      crossed=true,
                                      n_replicates=100)
    names_list = sort(collect(keys(param_ranges)))
    values_list = [param_ranges[name] for name in names_list]

    if crossed
        param_combinations = collect(Iterators.product(values_list...))
        param_combinations = [collect(combo) for combo in vec(param_combinations)]
    else
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

    total_sims = length(param_combinations) * n_replicates

    return param_combinations, total_sims
end

# ============================================================================
# DataFrame Conversion Functions
# ============================================================================

"""
    ensemble_to_df(solve_out, init_names, intermediaries, intermediary_names, ensemble_n)

Convert ensemble simulation results to long-format DataFrames.

# Arguments
- `solve_out`: Vector of solution objects from ensemble simulations
- `init_names`: Names of state variables
- `intermediaries`: Vector of intermediate calculation results (or `nothing`)
- `intermediary_names`: Names of intermediate variables (or `nothing`)
- `ensemble_n`: Number of replicates per parameter combination

# Returns
Named tuple containing:
- `timeseries_df`: Time series data with columns `condition, sim, time, variable, value`
- `param_df`: Parameter values with columns `condition, sim, variable, value`
- `init_df`: Initial values with columns `condition, sim, variable, value`
"""
function ensemble_to_df(solve_out, init_names,
                        intermediaries, intermediary_names, ensemble_n)
    # Unwrap SciMLBase.EnsembleSolution so element access uses plain Vector indexing
    if !isa(solve_out, Vector) && hasproperty(solve_out, :u)
        solve_out = solve_out.u
    end
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

        trajectory_vec = Vector{Int}(undef, total_rows)
        time_vec = Vector{Float64}(undef, total_rows)
        variable_vec = Vector{String}(undef, total_rows)
        value_vec = Vector{Float64}(undef, total_rows)

        row_idx = 1

        for (traj_idx, result) in enumerate(solutions)
            if !isempty(result.t)
                for (t_idx, t_raw) in enumerate(result.t)
                    t_val = Float64(t_raw)
                    u_val = result.u[t_idx]

                    if isa(u_val, Union{AbstractVector, Tuple})
                        for (var_idx, var_val) in enumerate(u_val)
                            if !isa(var_val, Function)
                                var_name = var_idx <= length(var_names_to_use) ?
                                    var_names_to_use[var_idx] : "var_$var_idx"

                                trajectory_vec[row_idx] = traj_idx
                                time_vec[row_idx] = t_val
                                variable_vec[row_idx] = var_name
                                value_vec[row_idx] = Float64(var_val)
                                row_idx += 1
                            end
                        end
                    else
                        if !isa(u_val, Function)
                            var_name = var_names_to_use[1]

                            trajectory_vec[row_idx] = traj_idx
                            time_vec[row_idx] = t_val
                            variable_vec[row_idx] = var_name
                            value_vec[row_idx] = Float64(u_val)
                            row_idx += 1
                        end
                    end
                end
            end
        end

        resize!(trajectory_vec, row_idx - 1)
        resize!(time_vec, row_idx - 1)
        resize!(variable_vec, row_idx - 1)
        resize!(value_vec, row_idx - 1)

        return trajectory_vec, time_vec, variable_vec, value_vec
    end

    main_traj, main_time, main_var, main_val = process_solution_like(solve_out, var_names)

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

    j_vec, i_vec = _group_replicate_indices(main_traj, ensemble_n)
    timeseries_df = DataFrame(
        condition = j_vec,
        sim = i_vec,
        time = main_time,
        variable = main_var,
        value = main_val;
        copycols = false
    )

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

    param_df = DataFrame(condition=Int[], sim=Int[], variable=String[], value=Float64[])
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

                param_j_vec[row_idx] = div(traj_idx - 1, ensemble_n) + 1
                param_i_vec[row_idx] = rem(traj_idx - 1, ensemble_n) + 1
                param_name_vec[row_idx] = param_names[param_idx]
                param_value_vec[row_idx] = Float64(param_val)
                row_idx += 1
            end
        end

        param_df = DataFrame(
            condition = param_j_vec,
            sim = param_i_vec,
            variable = param_name_vec,
            value = param_value_vec;
            copycols = false
        )
    end

    init_val_names = [string(name) for name in init_names]

    init_df = DataFrame(condition=Int[], sim=Int[], variable=String[], value=Float64[])
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

                    init_j_vec[row_idx] = div(traj_idx - 1, ensemble_n) + 1
                    init_i_vec[row_idx] = rem(traj_idx - 1, ensemble_n) + 1
                    init_name_vec[row_idx] = init_val_names[init_idx]
                    init_value_vec[row_idx] = Float64(init_val)
                    row_idx += 1
                end
            elseif isa(init_vals, AbstractVector)
                for (init_idx, init_name) in enumerate(init_val_names)
                    init_val = init_vals[init_idx]

                    init_j_vec[row_idx] = div(traj_idx - 1, ensemble_n) + 1
                    init_i_vec[row_idx] = rem(traj_idx - 1, ensemble_n) + 1
                    init_name_vec[row_idx] = init_name
                    init_value_vec[row_idx] = Float64(init_val)
                    row_idx += 1
                end
            else
                init_j_vec[row_idx] = div(traj_idx - 1, ensemble_n) + 1
                init_i_vec[row_idx] = rem(traj_idx - 1, ensemble_n) + 1
                init_name_vec[row_idx] = init_val_names[1]
                init_value_vec[row_idx] = Float64(init_vals)
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
            condition = init_j_vec,
            sim = init_i_vec,
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
"""
function ensemble_to_df_threaded(solve_out, init_names,
                                  intermediaries, intermediary_names, ensemble_n)
    # Unwrap SciMLBase.EnsembleSolution so element access uses plain Vector indexing
    if !isa(solve_out, Vector) && hasproperty(solve_out, :u)
        solve_out = solve_out.u
    end
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

        trajectory_vec = Vector{Int}(undef, total_rows)
        time_vec = Vector{Float64}(undef, total_rows)
        variable_vec = Vector{String}(undef, total_rows)
        value_vec = Vector{Float64}(undef, total_rows)

        start_indices = Vector{Int}(undef, length(solutions))
        start_indices[1] = 1
        for i in 2:length(solutions)
            start_indices[i] = start_indices[i-1] + row_counts[i-1]
        end

        Base.Threads.@threads for traj_idx in 1:length(solutions)
            result = solutions[traj_idx]
            if !isempty(result.t)
                row_idx = start_indices[traj_idx]

                for (t_idx, t_raw) in enumerate(result.t)
                    t_val = Float64(t_raw)
                    u_val = result.u[t_idx]

                    if isa(u_val, Union{AbstractVector, Tuple})
                        for (var_idx, var_val) in enumerate(u_val)
                            if !isa(var_val, Function)
                                var_name = var_idx <= length(var_names_to_use) ?
                                    var_names_to_use[var_idx] : "var_$var_idx"

                                trajectory_vec[row_idx] = traj_idx
                                time_vec[row_idx] = t_val
                                variable_vec[row_idx] = var_name
                                value_vec[row_idx] = Float64(var_val)
                                row_idx += 1
                            end
                        end
                    else
                        if !isa(u_val, Function)
                            var_name = var_names_to_use[1]

                            trajectory_vec[row_idx] = traj_idx
                            time_vec[row_idx] = t_val
                            variable_vec[row_idx] = var_name
                            value_vec[row_idx] = Float64(u_val)
                            row_idx += 1
                        end
                    end
                end
            end
        end

        return trajectory_vec, time_vec, variable_vec, value_vec
    end

    main_traj, main_time, main_var, main_val = process_solution_like(solve_out, var_names)

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

    j_vec, i_vec = _group_replicate_indices(main_traj, ensemble_n)
    timeseries_df = DataFrame(
        condition = j_vec,
        sim = i_vec,
        time = main_time,
        variable = main_var,
        value = main_val;
        copycols = false
    )

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

    param_df = DataFrame(condition=Int[], sim=Int[], variable=String[], value=Float64[])
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

                param_j_vec[row_idx] = div(traj_idx - 1, ensemble_n) + 1
                param_i_vec[row_idx] = rem(traj_idx - 1, ensemble_n) + 1
                param_name_vec[row_idx] = param_name
                param_value_vec[row_idx] = Float64(param_val)
            end
        end

        param_df = DataFrame(
            condition = param_j_vec,
            sim = param_i_vec,
            variable = param_name_vec,
            value = param_value_vec;
            copycols = false
        )
    end

    init_val_names = [string(name) for name in init_names]
    init_val_symbols = Symbol.(init_val_names)

    init_df = DataFrame(condition=Int[], sim=Int[], variable=String[], value=Float64[])
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

                    init_j_vec[row_idx] = div(traj_idx - 1, ensemble_n) + 1
                    init_i_vec[row_idx] = rem(traj_idx - 1, ensemble_n) + 1
                    init_name_vec[row_idx] = init_name
                    init_value_vec[row_idx] = Float64(init_val)
                end
            elseif isa(init_vals, AbstractVector)
                for (init_idx, init_name) in enumerate(init_val_names)
                    row_idx = (traj_idx - 1) * length(init_val_names) + init_idx
                    init_val = init_vals[init_idx]

                    init_j_vec[row_idx] = div(traj_idx - 1, ensemble_n) + 1
                    init_i_vec[row_idx] = rem(traj_idx - 1, ensemble_n) + 1
                    init_name_vec[row_idx] = init_name
                    init_value_vec[row_idx] = Float64(init_val)
                end
            else
                row_idx = (traj_idx - 1) * length(init_val_names) + 1
                init_j_vec[row_idx] = div(traj_idx - 1, ensemble_n) + 1
                init_i_vec[row_idx] = rem(traj_idx - 1, ensemble_n) + 1
                init_name_vec[row_idx] = init_val_names[1]
                init_value_vec[row_idx] = Float64(init_vals)
            end
        end

        init_df = DataFrame(
            condition = init_j_vec,
            sim = init_i_vec,
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

# Canonical catalog of supported summary statistics, in column order. Mirrors
# the R-side registry `ensemble_stat_funs` in sdbuildR (R/ensemble_r.R).
const ENSEMBLE_STAT_NAMES = ["mean", "median", "sd", "var", "min", "max", "missing_count"]

# Order a requested set of statistic names by the canonical catalog order,
# dropping any names not in the catalog (the R side validates the set).
_order_ensemble_stats(stats) = filter(s -> string(s) in string.(stats), ENSEMBLE_STAT_NAMES)

# Compute a single named statistic over a vector of valid (non-missing) values.
# `missing_count` is handled by the callers, since it depends on the pre-filter
# count rather than the clean values.
function _ensemble_stat_value(name::AbstractString, v)
    if name == "mean"
        return mean(v)
    elseif name == "median"
        return Statistics.median(v)
    elseif name == "sd"
        return Statistics.std(v)
    elseif name == "var"
        return var(v)
    elseif name == "min"
        return minimum(v)
    elseif name == "max"
        return maximum(v)
    else
        error("Unknown ensemble summary statistic: $name")
    end
end

"""
    ensemble_summ(timeseries_df, quantiles=[0.025, 0.975], stats=["mean", "median"])

Compute summary statistics across ensemble trajectories.

# Arguments
- `timeseries_df`: DataFrame from `ensemble_to_df` or `ensemble_to_df_threaded`
- `quantiles::Vector=[0.025, 0.975]`: Quantiles to compute (default: 95% interval)
- `stats::Vector=["mean", "median"]`: Which summary statistics to compute. Any of
  `"mean"`, `"median"`, `"sd"`, `"var"`, `"min"`, `"max"`, `"missing_count"`.

# Returns
DataFrame with columns: `condition, time, variable`, one column per requested
statistic (in catalog order), then `quant1, quant2, ...` for each quantile (in
the order given).
"""
function ensemble_summ(timeseries_df, quantiles=[0.025, 0.975], stats=["mean", "median"])
    stats = _order_ensemble_stats(stats)
    stats_df = combine(groupby(timeseries_df, [:condition, :time, :variable])) do group
        values = group.value

        is_valid = .!(ismissing.(values) .| isnan.(values))
        clean_values = values[is_valid]
        num_missing = count(!, is_valid)
        is_empty = isempty(clean_values)

        result = NamedTuple()
        for s in stats
            val = if s == "missing_count"
                num_missing
            elseif is_empty
                NaN
            else
                _ensemble_stat_value(s, clean_values)
            end
            result = merge(result, (Symbol(s) => val,))
        end
        for (qi, q) in enumerate(quantiles)
            val = is_empty ? NaN : Statistics.quantile(clean_values, q)
            result = merge(result, (Symbol("quant$qi") => val,))
        end

        return result
    end

    return stats_df
end

"""
    ensemble_summ_threaded(timeseries_df, quantiles=[0.025, 0.975], stats=["mean", "median"])

Multi-threaded version of `ensemble_summ` for improved performance.

# Arguments
Same as `ensemble_summ`.

# Returns
Same as `ensemble_summ`.
"""
function ensemble_summ_threaded(timeseries_df, quantiles=[0.025, 0.975], stats=["mean", "median"])
    stats = _order_ensemble_stats(stats)
    grouped_df = groupby(timeseries_df, [:condition, :time, :variable])

    group_keys = keys(grouped_df)
    n_groups = length(group_keys)

    condition_vals = Vector{Int}(undef, n_groups)
    time_vals = Vector{Float64}(undef, n_groups)
    variable_vals = Vector{String}(undef, n_groups)

    stat_arrays = Dict{String, Vector{Float64}}()
    for s in stats
        stat_arrays[s] = Vector{Float64}(undef, n_groups)
    end
    quant_arrays = [Vector{Float64}(undef, n_groups) for _ in quantiles]

    Base.Threads.@threads for i in 1:n_groups
        group = grouped_df[i]
        key = group_keys[i]
        values = group.value

        is_valid = .!(ismissing.(values) .| isnan.(values))
        clean_values = values[is_valid]
        num_missing = count(!, is_valid)
        is_empty = isempty(clean_values)

        condition_vals[i] = key.condition
        time_vals[i] = key.time
        variable_vals[i] = key.variable

        for s in stats
            stat_arrays[s][i] = if s == "missing_count"
                num_missing
            elseif is_empty
                NaN
            else
                _ensemble_stat_value(s, clean_values)
            end
        end
        for (qi, q) in enumerate(quantiles)
            quant_arrays[qi][i] = is_empty ? NaN : Statistics.quantile(clean_values, q)
        end
    end

    stats_df = DataFrame(
        condition = condition_vals,
        time = time_vals,
        variable = variable_vals;
        copycols = false
    )

    for s in stats
        stats_df[!, Symbol(s)] = stat_arrays[s]
    end
    for (qi, _) in enumerate(quantiles)
        stats_df[!, Symbol("quant$qi")] = quant_arrays[qi]
    end

    return stats_df
end

end # module
