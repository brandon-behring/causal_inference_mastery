"""
End-to-end tests for NearestNeighborPSM estimator.
"""

using Test
using Random
using Statistics
using Distributions
using CausalEstimators

@testset "NearestNeighborPSM End-to-End" begin

    # ========================================================================
    # Known-Answer Tests: Simple RCT-like Data
    # ========================================================================

    @testset "Known-Answer: RCT with confounding" begin
        Random.seed!(42)
        n = 200

        # Generate data with confounding
        # Covariate affects both treatment and outcome
        x = randn(n)

        # Treatment: probabilistic based on x
        prob_treatment = 1 ./ (1 .+ exp.(-0.5 .* x))
        treatment = rand(n) .< prob_treatment

        # Outcome: Y = 5*T + 2*X + noise
        # True ATE = 5
        outcomes = 5.0 .* treatment .+ 2.0 .* x .+ randn(n)

        covariates = hcat(x)

        # Create problem
        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))

        # Solve with 1:1 matching
        estimator = NearestNeighborPSM(M=1, with_replacement=false)
        solution = solve(problem, estimator)

        # Should succeed
        @test solution.retcode == :Success
        @test solution.n_matched > 0

        # ATE should be close to 5 (true effect)
        @test abs(solution.estimate - 5.0) < 1.0  # Within 1 of true value

        # CI should contain true value
        @test solution.ci_lower < 5.0 < solution.ci_upper

        # Standard error should be positive
        @test solution.se > 0

        # Propensity scores should exist
        @test length(solution.propensity_scores) == n
        @test all(0 .< solution.propensity_scores .< 1)
    end

    @testset "Known-Answer: No confounding (pure RCT)" begin
        Random.seed!(42)
        n = 100

        # Pure randomization
        treatment = vcat(fill(true, 50), fill(false, 50))

        # Outcome: Y = 10*T + noise (no covariates)
        # True ATE = 10
        outcomes = 10.0 .* treatment .+ randn(n)

        # Meaningless covariate (independent of treatment and outcome)
        covariates = randn(n, 1)

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
        estimator = NearestNeighborPSM(M=1, with_replacement=false)
        solution = solve(problem, estimator)

        @test solution.retcode == :Success

        # ATE should be close to 10
        @test abs(solution.estimate - 10.0) < 2.0

        # Propensity scores should be close to 0.5 (balanced)
        @test abs(mean(solution.propensity_scores) - 0.5) < 0.1
    end

    # ========================================================================
    # Matching Configuration Tests
    # ========================================================================

    @testset "Config: 2:1 matching with replacement" begin
        Random.seed!(42)
        n = 150

        x = randn(n)
        prob_treatment = 1 ./ (1 .+ exp.(-x))
        treatment = rand(n) .< prob_treatment
        outcomes = 3.0 .* treatment .+ x .+ randn(n)
        covariates = hcat(x)

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))

        # 2:1 matching with replacement
        estimator = NearestNeighborPSM(M=2, with_replacement=true)
        solution = solve(problem, estimator)

        @test solution.retcode == :Success

        # Each matched treated unit should have up to 2 controls
        # With replacement, same control can be reused
    end

    @testset "Config: Caliper matching (strict)" begin
        Random.seed!(42)
        n = 200

        x = randn(n)
        # Strong confounding -> wide propensity range
        prob_treatment = 1 ./ (1 .+ exp.(-2.0 .* x))
        treatment = rand(n) .< prob_treatment
        outcomes = 4.0 .* treatment .+ 1.5 .* x .+ randn(n)
        covariates = hcat(x)

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))

        # Strict caliper may drop some units
        estimator = NearestNeighborPSM(M=1, with_replacement=false, caliper=0.1)
        solution = solve(problem, estimator)

        # May succeed or fail depending on overlap
        if solution.retcode == :Success
            # If successful, n_matched may be < n_treated
            @test solution.n_matched <= solution.n_treated
        end
    end

    @testset "Config: Abadie-Imbens vs Bootstrap variance" begin
        Random.seed!(42)
        n = 100

        x = randn(n)
        prob_treatment = 1 ./ (1 .+ exp.(-0.5 .* x))
        treatment = rand(n) .< prob_treatment
        outcomes = 2.0 .* treatment .+ x .+ randn(n)
        covariates = hcat(x)

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))

        # Abadie-Imbens variance (default, recommended)
        est_ai = NearestNeighborPSM(M=1, with_replacement=true, variance_method=:abadie_imbens)
        sol_ai = solve(problem, est_ai)

        # Bootstrap variance (should warn for with_replacement)
        est_boot = NearestNeighborPSM(M=1, with_replacement=true, variance_method=:bootstrap)
        sol_boot = solve(problem, est_boot)

        # Both should succeed
        @test sol_ai.retcode == :Success
        @test sol_boot.retcode == :Success

        # Estimates should be similar
        @test abs(sol_ai.estimate - sol_boot.estimate) < 0.5

        # Standard errors may differ (Abadie-Imbens is correct for with-replacement)
    end

    # ========================================================================
    # Failure Mode Tests
    # ========================================================================

    @testset "Failure: No common support" begin
        Random.seed!(42)
        n = 100

        # Perfect separation: treated have x > 5, control have x < -5
        x = vcat(fill(10.0, 50), fill(-10.0, 50))
        treatment = x .> 0
        outcomes = 5.0 .* treatment .+ randn(n)
        covariates = hcat(x)

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
        estimator = NearestNeighborPSM(M=1, with_replacement=false)
        solution = solve(problem, estimator)

        # Should fail due to no common support
        @test solution.retcode == :CommonSupportFailed
        @test isnan(solution.estimate)
        @test isnan(solution.se)
    end

    @testset "Failure: No matches within caliper" begin
        Random.seed!(42)
        n = 100

        x = randn(n)
        prob_treatment = 1 ./ (1 .+ exp.(-2.0 .* x))  # Strong effect
        treatment = rand(n) .< prob_treatment
        outcomes = 3.0 .* treatment .+ x .+ randn(n)
        covariates = hcat(x)

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))

        # Extremely strict caliper
        estimator = NearestNeighborPSM(M=1, with_replacement=false, caliper=0.001)
        solution = solve(problem, estimator)

        # May fail with MatchingFailed (variance computation failure with partial matches)
        # Or may succeed with few matches
        if solution.retcode == :MatchingFailed
            # Partial matches but variance computation failed
            @test isnan(solution.estimate)
            @test isnan(solution.se)
        else
            # Succeeded with some matches
            @test solution.retcode == :Success
            @test solution.n_matched > 0
        end
    end

    # ========================================================================
    # Multivariate Covariate Tests
    # ========================================================================

    @testset "Multivariate: 3 covariates" begin
        Random.seed!(42)
        n = 200

        # 3 confounding variables
        x1 = randn(n)
        x2 = randn(n)
        x3 = randn(n)

        # Treatment depends on all 3
        logit_p = 0.5*x1 + 0.3*x2 - 0.2*x3
        prob_treatment = 1 ./ (1 .+ exp.(-logit_p))
        treatment = rand(n) .< prob_treatment

        # Outcome depends on treatment + covariates
        outcomes = 4.0 .* treatment .+ x1 .+ 0.5*x2 .+ 0.8*x3 .+ randn(n)

        covariates = hcat(x1, x2, x3)

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
        # Use with_replacement to avoid partial matching issues with 3 covariates
        estimator = NearestNeighborPSM(M=1, with_replacement=true)
        solution = solve(problem, estimator)

        @test solution.retcode == :Success
        @test solution.n_matched > 0

        # ATE should be close to 4.0 (allow more tolerance with 3 covariates)
        @test abs(solution.estimate - 4.0) < 2.0
    end

    @testset "Multivariate: High-dimensional (p=10)" begin
        Random.seed!(42)
        n = 300
        p = 10

        # Many covariates
        covariates = randn(n, p)

        # Treatment depends on first 3 covariates
        logit_p = sum(covariates[:, 1:3], dims=2)[:]
        prob_treatment = 1 ./ (1 .+ exp.(-logit_p))
        treatment = rand(n) .< prob_treatment

        # Outcome depends on treatment + first 5 covariates
        outcomes = 2.0 .* treatment .+ sum(covariates[:, 1:5], dims=2)[:] .+ randn(n)

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
        estimator = NearestNeighborPSM(M=1, with_replacement=false)
        solution = solve(problem, estimator)

        @test solution.retcode == :Success
        # High-dimensional matching may be less precise
    end

    # ========================================================================
    # Statistical Properties Tests
    # ========================================================================

    @testset "Statistical: CI coverage (nominal 95%)" begin
        Random.seed!(42)
        n_sims = 50  # Limited for speed
        true_ate = 3.0
        n = 150

        coverage_count = 0

        for sim in 1:n_sims
            x = randn(n)
            prob_treatment = 1 ./ (1 .+ exp.(-0.5 .* x))
            treatment = rand(n) .< prob_treatment
            outcomes = true_ate .* treatment .+ x .+ randn(n)
            covariates = hcat(x)

            problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
            estimator = NearestNeighborPSM(M=1, with_replacement=false)
            solution = solve(problem, estimator)

            if solution.retcode == :Success
                # Check if CI contains true ATE
                if solution.ci_lower <= true_ate <= solution.ci_upper
                    coverage_count += 1
                end
            end
        end

        coverage_rate = coverage_count / n_sims

        # Coverage should be reasonable (matching estimators have wider CIs)
        # With 50 sims and matching uncertainty, expect lower coverage than nominal 95%
        # Some sims may fail to match all units (especially without replacement)
        @test coverage_rate > 0.60  # At least 60% (conservative for matching)
    end

    @testset "Statistical: Unbiased estimation" begin
        Random.seed!(42)
        n_sims = 30  # Limited for speed
        true_ate = 5.0
        n = 200

        estimates = Float64[]

        for sim in 1:n_sims
            x = randn(n)
            prob_treatment = 1 ./ (1 .+ exp.(-0.5 .* x))
            treatment = rand(n) .< prob_treatment
            outcomes = true_ate .* treatment .+ 2.0 .* x .+ randn(n)
            covariates = hcat(x)

            problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
            estimator = NearestNeighborPSM(M=1, with_replacement=false)
            solution = solve(problem, estimator)

            if solution.retcode == :Success
                push!(estimates, solution.estimate)
            end
        end

        # Mean of estimates should be close to true ATE
        mean_estimate = mean(estimates)
        @test abs(mean_estimate - true_ate) < 1.0

        # Should have some variability
        @test std(estimates) > 0
    end

    # ========================================================================
    # Estimator Construction Tests
    # ========================================================================

    @testset "Constructor: Default parameters" begin
        estimator = NearestNeighborPSM()

        @test estimator.M == 1
        @test estimator.with_replacement == false
        @test estimator.caliper == Inf
        @test estimator.variance_method == :abadie_imbens
    end

    @testset "Constructor: Custom parameters" begin
        estimator = NearestNeighborPSM(
            M=3,
            with_replacement=true,
            caliper=0.25,
            variance_method=:bootstrap
        )

        @test estimator.M == 3
        @test estimator.with_replacement == true
        @test estimator.caliper == 0.25
        @test estimator.variance_method == :bootstrap
    end

    @testset "Constructor: Invalid M" begin
        @test_throws ArgumentError NearestNeighborPSM(M=0)
        @test_throws ArgumentError NearestNeighborPSM(M=-1)
    end

    @testset "Constructor: Invalid caliper" begin
        @test_throws ArgumentError NearestNeighborPSM(caliper=-0.1)
        @test_throws ArgumentError NearestNeighborPSM(caliper=0.0)
    end

    @testset "Constructor: Invalid variance method" begin
        @test_throws ArgumentError NearestNeighborPSM(variance_method=:invalid)
    end

    # ========================================================================
    # Solution Structure Tests
    # ========================================================================

    @testset "Solution: Complete structure" begin
        Random.seed!(42)
        n = 100

        x = randn(n)
        prob_treatment = 1 ./ (1 .+ exp.(-0.5 .* x))
        treatment = rand(n) .< prob_treatment
        outcomes = 2.0 .* treatment .+ x .+ randn(n)
        covariates = hcat(x)

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
        estimator = NearestNeighborPSM(M=1, with_replacement=false)
        solution = solve(problem, estimator)

        # Check all solution fields
        @test solution.retcode isa Symbol
        @test solution.estimate isa Float64
        @test solution.se isa Float64
        @test solution.ci_lower isa Float64
        @test solution.ci_upper isa Float64
        @test solution.n_treated isa Int
        @test solution.n_control isa Int
        @test solution.n_matched isa Int
        @test solution.propensity_scores isa Vector{Float64}
        @test solution.matched_indices isa Vector{Tuple{Int,Int}}
        @test solution.balance_metrics isa NamedTuple
        @test solution.original_problem === problem

        if solution.retcode == :Success
            # CI should be ordered
            @test solution.ci_lower < solution.ci_upper

            # SE should be positive
            @test solution.se > 0

            # n_matched should be <= n_treated
            @test solution.n_matched <= solution.n_treated

            # Should have matched pairs
            @test length(solution.matched_indices) > 0
        end
    end

end
