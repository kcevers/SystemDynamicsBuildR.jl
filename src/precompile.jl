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

# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

@setup_workload begin
    u0 = [10.0, 5.0]
    tspan = (0.0, 1.0)
    pars = (alpha = 1.1, beta = 0.4, delta = 0.1, gamma = 0.4)
    dt = 0.1
    saveat = [0.0, 0.5, 1.0]
    tstops = saveat
    init_names = ["prey", "predator"]
    intermediary_names = ["total", "product"]

    @compile_workload begin
        # --- every solver sim_methods() can emit -------------------------------
        _pc_solve(Euler(), u0, tspan, pars, dt, saveat, tstops)
        _pc_solve(Midpoint(), u0, tspan, pars, dt, saveat, tstops)
        _pc_solve(Heun(), u0, tspan, pars, dt, saveat, tstops)
        _pc_solve(BS3(), u0, tspan, pars, dt, saveat, tstops)
        _pc_solve(Tsit5(), u0, tspan, pars, dt, saveat, tstops)
        _pc_solve(Vern6(), u0, tspan, pars, dt, saveat, tstops)
        _pc_solve(Vern7(), u0, tspan, pars, dt, saveat, tstops)
        _pc_solve(Vern8(), u0, tspan, pars, dt, saveat, tstops)
        _pc_solve(Vern9(), u0, tspan, pars, dt, saveat, tstops)
        _pc_solve(Rosenbrock23(), u0, tspan, pars, dt, saveat, tstops)

        prob, sol = _pc_solve(RK4(), u0, tspan, pars, dt, saveat, tstops)

        # --- single-run cleanup + CSV, as post_ode_julia does ------------------
        df, param_values, param_names, init_values, init_val_names =
            clean_df(prob, sol, init_names)

        csv_path = tempname() * ".csv"
        CSV.write(csv_path, df)
        rm(csv_path; force = true)

        # --- the SavingCallback path ------------------------------------------
        saved = SavedValues(Float64, Tuple{Float64, Float64})
        cb = SavingCallback(_pc_saving, saved; saveat = saveat)
        cb_prob = ODEProblem(_pc_ode!, u0, tspan, pars)
        cb_sol = solve(cb_prob, RK4(); dt = dt, saveat = saveat, tstops = tstops,
                       adaptive = false, callback = cb)
        clean_df(cb_prob, cb_sol, init_names, saved, intermediary_names)

        # --- ensemble path ----------------------------------------------------
        # sdbuildR passes EnsembleThreads() only when threading is on, and no
        # ensemble algorithm at all otherwise; both are distinct signatures.
        ens_prob = EnsembleProblem(ODEProblem(_pc_ode!, u0, tspan, pars);
                                   prob_func = _pc_prob_func)
        ens_sol = solve(ens_prob, RK4();
                        dt = dt, saveat = saveat, tstops = tstops,
                        adaptive = false, trajectories = 2)
        solve(ens_prob, RK4(), EnsembleThreads();
              dt = dt, saveat = saveat, tstops = tstops,
              adaptive = false, trajectories = 2)

        ts_df, param_df, init_df =
            ensemble_to_df(ens_sol, init_names, nothing, nothing, 2)
        ensemble_summ(ts_df, [0.025, 0.975], ["mean", "median"])
        generate_param_combinations(Dict("alpha" => [1.0, 1.1]);
                                    crossed = true, n_replicates = 2)

        # --- small helpers on the hot path ------------------------------------
        clean_constants((a = 1.0, b = [1.0, 2.0]))
        clean_init([10.0, 5.0], init_names)
        saveat_func([0.0, 1.0], [1.0, 2.0], [0.5])
        with_rng(() -> rand(), 1)
        is_function_or_interp(1.0)
    end
end
