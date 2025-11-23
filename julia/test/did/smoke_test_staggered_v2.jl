"""
Smoke test for staggered DiD implementations (Fixed for NamedTuple results).

Tests basic functionality of StaggeredTWFE, CallawaySantAnna, and SunAbraham.
"""

using Test
using CausalEstimators
using Random

@testset "Staggered DiD Smoke Tests" begin

    # Create synthetic data large enough for bootstrap
    # 3 cohorts (treated at t=3, t=5, never), 10 units per cohort, 8 periods
    Random.seed!(123)

    n_periods = 8
    n_units_per_cohort = 10

    # Unit IDs: 0-9 (cohort 3), 10-19 (cohort 5), 20-29 (never-treated)
    unit_id = repeat(0:29, inner=n_periods)

    # Time periods: 0, 1, 2, ..., 7
    time = repeat(0:(n_periods-1), outer=30)

    # Treatment time per unit
    treatment_time = vcat(
        fill(3.0, 10),   # Units 0-9: treated at t=3
        fill(5.0, 10),   # Units 10-19: treated at t=5
        fill(Inf, 10)    # Units 20-29: never-treated
    )
    treatment_time_full = repeat(treatment_time, inner=n_periods)

    # Treatment indicator
    treatment = zeros(Bool, length(unit_id))
    for i in 1:length(unit_id)
        uid = unit_id[i]
        t = time[i]
        tt = treatment_time[uid + 1]  # +1 for 1-indexing
        treatment[i] = isfinite(tt) && (t >= tt)
    end

    # Outcomes: Y = 10 + 5*treatment + noise
    outcomes = 10.0 .+ 5.0 * treatment .+ randn(length(unit_id))

    @testset "StaggeredDiDProblem Construction" begin
        @test_nowarn StaggeredDiDProblem(
            outcomes,
            treatment,
            time,
            unit_id,
            treatment_time,
            (alpha = 0.05,)
        )

        problem = StaggeredDiDProblem(
            outcomes,
            treatment,
            time,
            unit_id,
            treatment_time,
            (alpha = 0.05,)
        )

        @test length(problem.outcomes) == length(outcomes)
        @test length(problem.treatment) == length(treatment)
        @test length(problem.time) == length(time)
        @test length(problem.unit_id) == length(unit_id)
        @test length(problem.treatment_time) == 30  # One per unit
    end

    @testset "StaggeredTWFE Estimator" begin
        problem = StaggeredDiDProblem(
            outcomes,
            treatment,
            time,
            unit_id,
            treatment_time,
            (alpha = 0.05,)
        )

        estimator = StaggeredTWFE(cluster_se = true)

        # Test solve
        result = solve(problem, estimator)

        # Check result fields (NamedTuple)
        @test result.estimate isa Float64
        @test result.se isa Float64
        @test result.t_stat isa Float64
        @test result.p_value isa Float64
        @test result.ci_lower isa Float64
        @test result.ci_upper isa Float64
        @test result.retcode == :Success

        # Check sensible values (treatment effect should be roughly 5)
        @test 2.0 < result.estimate < 8.0  # Wide range due to bias
        @test result.se > 0
        @test 0 <= result.p_value <= 1
    end

    @testset "CallawaySantAnna Estimator" begin
        problem = StaggeredDiDProblem(
            outcomes,
            treatment,
            time,
            unit_id,
            treatment_time,
            (alpha = 0.05,)
        )

        estimator = CallawaySantAnna(
            aggregation = :simple,
            control_group = :nevertreated,
            alpha = 0.05,
            n_bootstrap = 50,  # Small for speed
            random_seed = 123
        )

        # Test solve
        result = solve(problem, estimator)

        # Check result fields
        @test result.att isa Float64
        @test result.se isa Float64
        @test result.att_gt isa Vector
        @test result.retcode == :Success

        # Check sensible values (treatment effect should be roughly 5)
        @test 2.0 < result.att < 8.0
        @test result.se > 0
        @test 0 <= result.p_value <= 1
    end

    @testset "SunAbraham Estimator" begin
        problem = StaggeredDiDProblem(
            outcomes,
            treatment,
            time,
            unit_id,
            treatment_time,
            (alpha = 0.05,)
        )

        estimator = SunAbraham(
            alpha = 0.05,
            cluster_se = true
        )

        # Test solve
        result = solve(problem, estimator)

        # Check result fields
        @test result.att isa Float64
        @test result.se isa Float64
        @test result.cohort_effects isa Vector
        @test result.weights isa Vector
        @test result.retcode == :Success

        # Check weights sum to 1
        @test sum(r.weight for r in result.weights) ≈ 1.0

        # Check sensible values (treatment effect should be roughly 5)
        # Wider range to accommodate variance with small sample
        @test 1.0 < result.att < 10.0
        @test result.se > 0
        @test 0 <= result.p_value <= 1
    end
end

println("✓ All staggered DiD smoke tests passed!")
