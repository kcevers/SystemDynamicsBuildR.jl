using SystemDynamicsBuildR.custom_func
using Random
using OrdinaryDiffEq
using SciMLBase

@testset "with_rng" begin

    @testset "reproducibility" begin
        # Same seed -> identical draws
        a = with_rng(() -> rand(3), 1234)
        b = with_rng(() -> rand(3), 1234)
        @test a == b

        # randn() is routed through the installed stream too
        @test with_rng(() -> randn(3), 55) == with_rng(() -> randn(3), 55)

        # Functions built on bare rand() (rbool, rdist) are routed as well
        @test with_rng(() -> rbool(0.5), 7) == with_rng(() -> rbool(0.5), 7)
    end

    @testset "global state is not disrupted" begin
        # Full internal Xoshiro state must be byte-identical before/after the block
        Random.seed!(999)
        rand(5)                                  # advance the global stream
        snap = copy(Random.default_rng())
        with_rng(() -> rand(5), 1234)
        @test copy(Random.default_rng()) == snap

        # Continuity: draws after the block match the no-block baseline
        Random.seed!(999); rand(5); no_block   = rand(4)
        Random.seed!(999); rand(5)
        with_rng(() -> rand(5), 1234);    with_block = rand(4)
        @test no_block == with_block
    end

    @testset "state restored even when f throws" begin
        Random.seed!(42)
        snap = copy(Random.default_rng())
        @test_throws ErrorException with_rng(() -> error("boom"), 7)
        @test copy(Random.default_rng()) == snap
    end

    @testset "Xoshiro source" begin
        src    = Xoshiro(2024)
        before = copy(src)
        out    = with_rng(() -> rand(3), src)
        @test src == before                      # src is not mutated
        @test out == rand(copy(src), 3)          # reproduces src's own stream
    end

    @testset "TaskLocalRNG is a no-op passthrough" begin
        # The task-local RNG passed in IS the installed stream: f() runs against it
        # and advances it, with no save/restore. Matches an already-seeded ctx.rng.
        tlrng = Random.default_rng()

        Random.seed!(123)
        expected = rand(3)                       # baseline draws from the seeded stream
        Random.seed!(123)
        got = with_rng(() -> rand(3), tlrng)     # same stream via passthrough
        @test got == expected

        # Passthrough advances the stream (no restore): consecutive blocks continue
        # the sequence rather than replaying it, unlike the seed/Xoshiro path.
        Random.seed!(123); a = rand(3); b = rand(3)   # six draws in sequence
        Random.seed!(123)
        pa = with_rng(() -> rand(3), tlrng)
        pb = with_rng(() -> rand(3), tlrng)
        @test (pa, pb) == (a, b)
    end

    @testset "non-Xoshiro AbstractRNG errors" begin
        @test_throws ErrorException with_rng(() -> rand(), MersenneTwister(1))
    end

    @testset "ensemble prob_func with ctx.rng" begin
        # New SciML interface: prob_func(prob, ctx); ctx.rng is the per-trajectory
        # TaskLocalRNG, already seeded by SciML. with_rng(ctx.rng) must run f against
        # that stream (no error, no restore) so bare rand() is reproducible per seed.
        rhs(u, p, t) = -u
        base = ODEProblem(rhs, [1.0], (0.0, 1.0))

        prob_func = function (prob, ctx)
            u0 = with_rng(ctx.rng) do
                [1.0 + rand()]            # bare rand(), routed through ctx.rng
            end
            remake(prob, u0 = u0)
        end

        ens = EnsembleProblem(base; prob_func = prob_func)

        sol1 = solve(ens, Tsit5(), EnsembleSerial(); trajectories = 5, seed = 42)
        sol2 = solve(ens, Tsit5(), EnsembleSerial(); trajectories = 5, seed = 42)

        # sol.u holds the per-trajectory ODESolutions; t.u[1][1] is the drawn u0
        u0s1 = [t.u[1][1] for t in sol1.u]
        u0s2 = [t.u[1][1] for t in sol2.u]

        @test u0s1 == u0s2                  # reproducible across runs with same seed
        @test length(unique(u0s1)) == 5     # each trajectory drew a distinct u0

        sol3 = solve(ens, Tsit5(), EnsembleSerial(); trajectories = 5, seed = 7)
        @test [t.u[1][1] for t in sol3.u] != u0s1   # a different seed -> different draws
    end

end
