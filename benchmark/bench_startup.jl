using Statistics

function run_cmd_seconds(cmd::Cmd)
    t0 = time_ns()
    run(cmd)
    return (time_ns() - t0) / 1e9
end

function benchmark_cmd(label::String, cmd::Cmd; runs::Int=5)
    # Warm-up once to avoid one-off shell/julia launcher effects in the median.
    run(cmd)

    times = Vector{Float64}(undef, runs)
    for i in 1:runs
        times[i] = run_cmd_seconds(cmd)
    end

    println("\n", label)
    println("  median: ", round(median(times), digits=3), " s")
    println("  min:    ", round(minimum(times), digits=3), " s")
    println("  max:    ", round(maximum(times), digits=3), " s")
end

println("Benchmarking startup/load in fresh Julia processes...")

using_cmd = `$(Base.julia_cmd()) --project -e "using SystemDynamicsBuildR"`
using_and_call_cmd = `$(Base.julia_cmd()) --project -e "using SystemDynamicsBuildR; SystemDynamicsBuildR.saveat_func([0.0,1.0],[0.0,1.0],[0.5])"`

benchmark_cmd("using SystemDynamicsBuildR", using_cmd)
benchmark_cmd("using + first function call", using_and_call_cmd)

println("\nDone.")
