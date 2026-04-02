using SystemDynamicsBuildR.clean
using SystemDynamicsBuildR.ensemble
using Unitful
using Statistics

function build_clean_inputs(n_t::Int=2000, n_vars::Int=4)
    t = collect(range(0.0, 100.0, length=n_t))
    u = [collect(1.0:n_vars) .+ 0.01 * i for i in 1:n_t]

    prob = (
        p = (alpha = 0.1, beta = 2.0, gamma = 0.05),
        u0 = collect(1.0:n_vars)
    )

    solve_out = (t = t, u = u)
    init_names = Symbol.("x" .* string.(1:n_vars))

    intermediaries = (
        t = t,
        saveval = [collect(10.0 .+ (1.0:n_vars) .+ 0.02 * i) for i in 1:n_t]
    )
    intermediary_names = Symbol.("int" .* string.(1:n_vars))

    return prob, solve_out, init_names, intermediaries, intermediary_names
end

function build_ensemble_inputs(n_traj::Int=200, n_t::Int=300, n_vars::Int=3, ensemble_n::Int=10)
    init_names = Symbol.("x" .* string.(1:n_vars))

    solve_out = [
        (
            t = collect(range(0.0, 50.0, length=n_t)),
            u = [collect(1.0:n_vars) .+ 0.01 * (traj + ti) for ti in 1:n_t],
            u0 = collect(1.0:n_vars) .+ 0.1 * traj,
            p = (alpha = 0.1 + 0.001 * traj, beta = 1.5)
        )
        for traj in 1:n_traj
    ]

    intermediaries = [
        (
            t = collect(range(0.0, 50.0, length=n_t)),
            saveval = [collect(10.0 .+ (1.0:n_vars) .+ 0.02 * (traj + ti)) for ti in 1:n_t]
        )
        for traj in 1:n_traj
    ]
    intermediary_names = Symbol.("int" .* string.(1:n_vars))

    return solve_out, init_names, intermediaries, intermediary_names, ensemble_n
end

function run_bench(label::String, f::Function; n::Int=8)
    f() # warm-up/compile

    times_ns = Vector{Float64}(undef, n)
    allocs = Vector{Float64}(undef, n)

    for i in 1:n
        allocs[i] = @allocated f()
        t0 = time_ns()
        f()
        times_ns[i] = (time_ns() - t0)
    end

    println("\n", label)
    println("  median time: ", round(median(times_ns) / 1e6, digits=3), " ms")
    println("  median alloc: ", round(median(allocs) / 1024^2, digits=3), " MiB")
end

println("Building benchmark inputs...")
prob, sol_clean, init_names_clean, inter_clean, inter_names_clean = build_clean_inputs()
sol_ens, init_names_ens, inter_ens, inter_names_ens, ensemble_n = build_ensemble_inputs()

run_bench("clean_df without intermediaries", () -> clean_df(prob, sol_clean, init_names_clean))

run_bench(
    "clean_df with intermediaries",
    () -> clean_df(prob, sol_clean, init_names_clean, inter_clean, inter_names_clean)
)

run_bench(
    "ensemble_to_df",
    () -> ensemble_to_df(sol_ens, init_names_ens, inter_ens, inter_names_ens, ensemble_n)
)

run_bench(
    "ensemble_to_df_threaded",
    () -> ensemble_to_df_threaded(sol_ens, init_names_ens, inter_ens, inter_names_ens, ensemble_n)
)

println("\nDone.")
