# precompile.jl
#
# PrecompileTools workload. Included last from SystemDynamicsBuildR.jl, so it runs in
# this package's namespace and the inferred/native code it produces is stored in this
# package's pkgimage.
#
# Why this exists: sdbuildR generates a fresh Julia script per simulate() call and
# include()s it into Main. Without a workload, every new Julia session pays the SciML
# JIT cost from scratch - measured at ~4 s for the first solve of each solver and ~3.4 s
# for the first CSV.write. Exercising those call paths here moves that cost into the
# one-time precompilation of this package.
#
# The workload mirrors the shapes sdbuildR's generated scripts actually produce (see
# R/scripts.R in the sdbuildR repo): an in-place f!(du, u, p, t) with a NamedTuple
# parameter, solved with adaptive = false and explicit dt/saveat/tstops.
#
# Precompilation is per-signature: a solver left out of the workload still pays its full
# JIT cost on first use, so every algorithm reachable from sdbuildR's sim_methods() is
# listed below.

using PrecompileTools

using CSV
using DataFrames
using DiffEqCallbacks
using SciMLBase

using OrdinaryDiffEqLowOrderRK   # Euler, Midpoint, Heun, RK4, BS3
using OrdinaryDiffEqTsit5        # Tsit5
using OrdinaryDiffEqVerner       # Vern6, Vern7, Vern8, Vern9
using OrdinaryDiffEqRosenbrock   # Rosenbrock23

# ---------------------------------------------------------------------------
# Helpers, defined at module level so the workload only has to call them.
# ---------------------------------------------------------------------------

# Shaped like the ODE function sdbuildR generates: in-place, NamedTuple parameters.
function _pc_ode!(du, u, p, t)
    du[1] = p.alpha * u[1] - p.beta * u[1] * u[2]
    du[2] = p.delta * u[1] * u[2] - p.gamma * u[2]
    nothing
end

# Shaped like the save_intermediaries callback.
_pc_saving(u, t, integrator) = (u[1] + u[2], u[1] * u[2])

# Shaped like the ensemble prob_func sdbuildR generates: two arguments, with the
# trajectory index read off the EnsembleContext as ctx.sim_id.
_pc_prob_func(prob, ctx) = remake(prob; u0 = [10.0 + 0.1 * ctx.sim_id, 5.0])

# One solve, exactly as run_ode_julia calls it. Taking `alg` as an argument means each
# call site below is a distinct concrete signature, which is what forces specialization.
function _pc_solve(alg, u0, tspan, pars, dt, saveat, tstops)
    prob = ODEProblem(_pc_ode!, u0, tspan, pars)
    sol = solve(prob, alg; dt = dt, saveat = saveat, tstops = tstops, adaptive = false)
    return prob, sol
end

"""
    _pc_workload()

Run the precompilation workload once and return its results.

The body lives in a plain function rather than directly inside `@compile_workload` for
two reasons. `@compile_workload` only executes while the package is being precompiled
(it is gated on `jl_generating_output()`), so code written inline there can never be
reached by the test suite and shows up as uncovered. And a workload that silently stops
matching the current SciML API keeps "working" - it just quietly stops precompiling the
paths it was meant to cover. Calling this from the test suite catches both.

Effectiveness is unaffected: `@compile_workload` caches direct *and* indirect callees,
so everything reached from here is still precompiled.

Returns a `NamedTuple` of the artefacts produced, for the tests to assert on.
"""
function _pc_workload()
    u0 = [10.0, 5.0]
    tspan = (0.0, 1.0)
    pars = (alpha = 1.1, beta = 0.4, delta = 0.1, gamma = 0.4)
    dt = 0.1
    saveat = [0.0, 0.5, 1.0]
    tstops = saveat
    init_names = ["prey", "predator"]
    intermediary_names = ["total", "product"]

    # Every solver sim_methods() can emit. Written out one call per solver rather than
    # looped, so each is a concrete signature that is guaranteed to specialize.
    solves = (
        Euler        = _pc_solve(Euler(), u0, tspan, pars, dt, saveat, tstops),
        Midpoint     = _pc_solve(Midpoint(), u0, tspan, pars, dt, saveat, tstops),
        Heun         = _pc_solve(Heun(), u0, tspan, pars, dt, saveat, tstops),
        RK4          = _pc_solve(RK4(), u0, tspan, pars, dt, saveat, tstops),
        BS3          = _pc_solve(BS3(), u0, tspan, pars, dt, saveat, tstops),
        Tsit5        = _pc_solve(Tsit5(), u0, tspan, pars, dt, saveat, tstops),
        Vern6        = _pc_solve(Vern6(), u0, tspan, pars, dt, saveat, tstops),
        Vern7        = _pc_solve(Vern7(), u0, tspan, pars, dt, saveat, tstops),
        Vern8        = _pc_solve(Vern8(), u0, tspan, pars, dt, saveat, tstops),
        Vern9        = _pc_solve(Vern9(), u0, tspan, pars, dt, saveat, tstops),
        Rosenbrock23 = _pc_solve(Rosenbrock23(), u0, tspan, pars, dt, saveat, tstops),
    )

    prob, sol = solves.RK4

    # Single-run cleanup + CSV, as post_ode_julia does.
    df, param_values, param_names, init_values, init_val_names =
        clean_df(prob, sol, init_names)

    csv_path = tempname() * ".csv"
    CSV.write(csv_path, df)
    csv_written = isfile(csv_path)
    rm(csv_path; force = true)

    # The SavingCallback path.
    saved = SavedValues(Float64, Tuple{Float64, Float64})
    cb = SavingCallback(_pc_saving, saved; saveat = saveat)
    cb_prob = ODEProblem(_pc_ode!, u0, tspan, pars)
    cb_sol = solve(cb_prob, RK4(); dt = dt, saveat = saveat, tstops = tstops,
                   adaptive = false, callback = cb)
    df_cb, = clean_df(cb_prob, cb_sol, init_names, saved, intermediary_names)

    # Ensemble path. sdbuildR passes EnsembleThreads() only when threading is on, and no
    # ensemble algorithm at all otherwise; both are distinct signatures.
    ens_prob = EnsembleProblem(ODEProblem(_pc_ode!, u0, tspan, pars);
                               prob_func = _pc_prob_func)
    ens_sol = solve(ens_prob, RK4();
                    dt = dt, saveat = saveat, tstops = tstops,
                    adaptive = false, trajectories = 2)
    solve(ens_prob, RK4(), EnsembleThreads();
          dt = dt, saveat = saveat, tstops = tstops,
          adaptive = false, trajectories = 2)

    ts_df, param_df, init_df = ensemble_to_df(ens_sol, init_names, nothing, nothing, 2)
    summ = ensemble_summ(ts_df, [0.025, 0.975], ["mean", "median"])
    combos, total_sims = generate_param_combinations(Dict("alpha" => [1.0, 1.1]);
                                                     crossed = true, n_replicates = 2)

    # Small helpers on the hot path.
    constants = clean_constants((a = 1.0, b = [1.0, 2.0]))
    init_dict = clean_init([10.0, 5.0], init_names)
    interp = saveat_func([0.0, 1.0], [1.0, 2.0], [0.5])
    rng_val = with_rng(() -> rand(), 1)
    is_fn = is_function_or_interp(1.0)

    return (; solves, df, param_values, param_names, init_values, init_val_names,
            csv_written, saved, df_cb, ens_sol, ts_df, param_df, init_df, summ,
            combos, total_sims, constants, init_dict, interp, rng_val, is_fn)
end

# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

@compile_workload begin
    _pc_workload()
end
