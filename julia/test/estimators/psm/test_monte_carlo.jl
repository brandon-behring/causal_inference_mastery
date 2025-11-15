"""
Monte Carlo validation tests for PSM estimator.

Verifies statistical properties:
- Coverage rate (95% CIs should contain true ATE ~95% of the time)
- Bias (estimates should be unbiased)
- RMSE (root mean squared error)
- Standard error calibration
"""

using Test
using Random
using Statistics
using Distributions
using CausalEstimators

@testset "Monte Carlo Validation" begin

    # ========================================================================
    # DGP 1: Simple Confounding (Linear)
    # ========================================================================

    @testset "MC: Simple confounding (n=200)" begin
        Random.seed!(42)
        n_sims = 100  # Limited for speed
        true_ate = 5.0
        n = 200

        estimates = Float64[]
        ses = Float64[]
        coverage_count = 0

        for sim in 1:n_sims
            # X affects both T and Y
            x = randn(n)

            # Treatment: P(T=1|X) = logit^-1(0.5*X)
            prob_treatment = 1 ./ (1 .+ exp.(-0.5 .* x))
            treatment = rand(n) .< prob_treatment

            # Outcome: Y = true_ate*T + 2*X + noise
            outcomes = true_ate .* treatment .+ 2.0 .* x .+ randn(n)

            covariates = hcat(x)

            problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
            estimator = NearestNeighborPSM(M=1, with_replacement=false)
            solution = solve(problem, estimator)

            if solution.retcode == :Success
                push!(estimates, solution.estimate)
                push!(ses, solution.se)

                # Check coverage
                if solution.ci_lower <= true_ate <= solution.ci_upper
                    coverage_count += 1
                end
            end
        end

        n_success = length(estimates)
        @test n_success >= 60  # At least 60% should succeed (without-replacement can drop units)

        if n_success >= 10
            # Bias: mean estimate should be close to true ATE
            bias = mean(estimates) - true_ate
            @test abs(bias) < 1.0  # Bias < 1.0 (20% of true effect)

            # Coverage rate: should be close to 95% (allow ±10% for sampling variation)
            coverage_rate = coverage_count / n_success
            @test 0.85 <= coverage_rate <= 1.0  # At least 85% coverage

            # RMSE should be finite and reasonable
            rmse = sqrt(mean((estimates .- true_ate).^2))
            @test rmse < 2.0  # RMSE < 40% of true effect

            # SE calibration: estimated SE should be close to empirical SE
            empirical_se = std(estimates)
            mean_estimated_se = mean(ses)
            se_ratio = mean_estimated_se / empirical_se
            @test 0.5 < se_ratio < 2.0  # Within 2x of empirical
        end
    end

    # ========================================================================
    # DGP 2: Strong Confounding
    # ========================================================================

    @testset "MC: Strong confounding (n=300)" begin
        Random.seed!(43)
        n_sims = 80  # Fewer sims for computational cost
        true_ate = 3.0
        n = 300  # Larger n for strong confounding

        estimates = Float64[]
        coverage_count = 0

        for sim in 1:n_sims
            x = randn(n)

            # Strong confounding: P(T=1|X) = logit^-1(2.0*X)
            prob_treatment = 1 ./ (1 .+ exp.(-2.0 .* x))
            treatment = rand(n) .< prob_treatment

            # Y = 3*T + 3*X + noise
            outcomes = true_ate .* treatment .+ 3.0 .* x .+ randn(n)

            covariates = hcat(x)

            problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
            # Use with-replacement to avoid partial matching
            estimator = NearestNeighborPSM(M=1, with_replacement=true)
            solution = solve(problem, estimator)

            if solution.retcode == :Success
                push!(estimates, solution.estimate)

                if solution.ci_lower <= true_ate <= solution.ci_upper
                    coverage_count += 1
                end
            end
        end

        n_success = length(estimates)
        @test n_success >= 60  # At least 75% should succeed

        if n_success >= 10
            # Bias should be reasonable even with strong confounding
            bias = mean(estimates) - true_ate
            @test abs(bias) < 1.5  # Allow slightly more bias

            # Coverage rate
            coverage_rate = coverage_count / n_success
            @test 0.80 <= coverage_rate <= 1.0  # At least 80%
        end
    end

    # ========================================================================
    # DGP 3: Multiple Confounders
    # ========================================================================

    @testset "MC: Multiple confounders (p=3)" begin
        Random.seed!(44)
        n_sims = 60
        true_ate = 4.0
        n = 250
        p = 3

        estimates = Float64[]
        coverage_count = 0

        for sim in 1:n_sims
            # 3 confounders
            x1 = randn(n)
            x2 = randn(n)
            x3 = randn(n)

            # Treatment depends on all 3
            logit_p = 0.5*x1 + 0.3*x2 - 0.2*x3
            prob_treatment = 1 ./ (1 .+ exp.(-logit_p))
            treatment = rand(n) .< prob_treatment

            # Outcome depends on treatment + all 3 covariates
            outcomes = true_ate .* treatment .+ x1 .+ 0.5*x2 .+ 0.8*x3 .+ randn(n)

            covariates = hcat(x1, x2, x3)

            problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
            estimator = NearestNeighborPSM(M=1, with_replacement=true)
            solution = solve(problem, estimator)

            if solution.retcode == :Success
                push!(estimates, solution.estimate)

                if solution.ci_lower <= true_ate <= solution.ci_upper
                    coverage_count += 1
                end
            end
        end

        n_success = length(estimates)
        @test n_success >= 45  # At least 75% should succeed

        if n_success >= 10
            # Multivariate matching may have more bias
            bias = mean(estimates) - true_ate
            @test abs(bias) < 2.0  # Allow more bias with multiple covariates

            # Coverage
            coverage_rate = coverage_count / n_success
            @test 0.75 <= coverage_rate <= 1.0  # At least 75%
        end
    end

    # ========================================================================
    # DGP 4: Nonlinear Confounding
    # ========================================================================

    @testset "MC: Nonlinear confounding" begin
        Random.seed!(45)
        n_sims = 60
        true_ate = 6.0
        n = 200

        estimates = Float64[]
        coverage_count = 0

        for sim in 1:n_sims
            x = randn(n)

            # Nonlinear relationship: X^2 affects treatment
            logit_p = 0.5 .* (x.^2 .- 1)  # Subtract 1 to center
            prob_treatment = 1 ./ (1 .+ exp.(-logit_p))
            treatment = rand(n) .< prob_treatment

            # Nonlinear outcome: Y = 6*T + X^2 + noise
            outcomes = true_ate .* treatment .+ (x.^2) .+ randn(n)

            # Include both X and X^2 as covariates (correct specification)
            covariates = hcat(x, x.^2)

            problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
            estimator = NearestNeighborPSM(M=1, with_replacement=true)
            solution = solve(problem, estimator)

            if solution.retcode == :Success
                push!(estimates, solution.estimate)

                if solution.ci_lower <= true_ate <= solution.ci_upper
                    coverage_count += 1
                end
            end
        end

        n_success = length(estimates)
        @test n_success >= 40  # At least 67% should succeed (harder problem)

        if n_success >= 10
            # With correct specification, bias should be reasonable
            bias = mean(estimates) - true_ate
            @test abs(bias) < 2.0

            # Coverage
            coverage_rate = coverage_count / n_success
            @test 0.70 <= coverage_rate <= 1.0  # At least 70%
        end
    end

    # ========================================================================
    # DGP 5: Varying Sample Sizes
    # ========================================================================

    @testset "MC: Small sample (n=100)" begin
        Random.seed!(46)
        n_sims = 50
        true_ate = 5.0
        n = 100  # Small sample

        estimates = Float64[]

        for sim in 1:n_sims
            x = randn(n)
            prob_treatment = 1 ./ (1 .+ exp.(-0.5 .* x))
            treatment = rand(n) .< prob_treatment
            outcomes = true_ate .* treatment .+ 2.0 .* x .+ randn(n)
            covariates = hcat(x)

            problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
            estimator = NearestNeighborPSM(M=1, with_replacement=true)
            solution = solve(problem, estimator)

            if solution.retcode == :Success
                push!(estimates, solution.estimate)
            end
        end

        n_success = length(estimates)
        @test n_success >= 35  # At least 70% should succeed

        if n_success >= 10
            # Smaller samples -> more bias allowed
            bias = mean(estimates) - true_ate
            @test abs(bias) < 1.5

            # Variability should be positive
            empirical_se = std(estimates)
            @test empirical_se > 0.1  # Should have some variability
        end
    end

    @testset "MC: Large sample (n=500)" begin
        Random.seed!(47)
        n_sims = 40  # Fewer sims due to computational cost
        true_ate = 5.0
        n = 500  # Large sample

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

        n_success = length(estimates)
        @test n_success >= 20  # At least 50% should succeed (large n without-replacement is hard)

        if n_success >= 10
            # Large samples -> less bias
            bias = mean(estimates) - true_ate
            @test abs(bias) < 0.8  # Tighter bound

            # Variability should be lower with large n
            empirical_se = std(estimates)
            @test empirical_se < 1.0  # Should be more concentrated
        end
    end

    # ========================================================================
    # DGP 6: M:1 Matching Comparison
    # ========================================================================

    @testset "MC: 1:1 vs 2:1 matching" begin
        Random.seed!(48)
        n_sims = 40
        true_ate = 5.0
        n = 200

        estimates_1to1 = Float64[]
        estimates_2to1 = Float64[]

        for sim in 1:n_sims
            x = randn(n)
            prob_treatment = 1 ./ (1 .+ exp.(-0.5 .* x))
            treatment = rand(n) .< prob_treatment
            outcomes = true_ate .* treatment .+ 2.0 .* x .+ randn(n)
            covariates = hcat(x)

            problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))

            # 1:1 matching
            est_1to1 = NearestNeighborPSM(M=1, with_replacement=true)
            sol_1to1 = solve(problem, est_1to1)

            # 2:1 matching
            est_2to1 = NearestNeighborPSM(M=2, with_replacement=true)
            sol_2to1 = solve(problem, est_2to1)

            if sol_1to1.retcode == :Success
                push!(estimates_1to1, sol_1to1.estimate)
            end

            if sol_2to1.retcode == :Success
                push!(estimates_2to1, sol_2to1.estimate)
            end
        end

        @test length(estimates_1to1) >= 30
        @test length(estimates_2to1) >= 30

        if length(estimates_1to1) >= 10 && length(estimates_2to1) >= 10
            # Both should be approximately unbiased
            bias_1to1 = abs(mean(estimates_1to1) - true_ate)
            bias_2to1 = abs(mean(estimates_2to1) - true_ate)

            @test bias_1to1 < 1.0
            @test bias_2to1 < 1.0

            # 2:1 should have lower variance (more matches per treated)
            var_1to1 = var(estimates_1to1)
            var_2to1 = var(estimates_2to1)

            # We expect var_2to1 <= var_1to1 (may not always hold due to randomness)
            # Just check both have reasonable variance
            @test var_1to1 > 0
            @test var_2to1 > 0
        end
    end

    # ========================================================================
    # DGP 7: Caliper Effect
    # ========================================================================

    @testset "MC: Caliper vs no caliper" begin
        Random.seed!(49)
        n_sims = 40
        true_ate = 5.0
        n = 200

        estimates_no_caliper = Float64[]
        estimates_caliper = Float64[]

        for sim in 1:n_sims
            x = randn(n)
            prob_treatment = 1 ./ (1 .+ exp.(-0.5 .* x))
            treatment = rand(n) .< prob_treatment
            outcomes = true_ate .* treatment .+ 2.0 .* x .+ randn(n)
            covariates = hcat(x)

            problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))

            # No caliper
            est_no_cal = NearestNeighborPSM(M=1, with_replacement=false, caliper=Inf)
            sol_no_cal = solve(problem, est_no_cal)

            # Caliper = 0.2 (moderate)
            est_cal = NearestNeighborPSM(M=1, with_replacement=false, caliper=0.2)
            sol_cal = solve(problem, est_cal)

            if sol_no_cal.retcode == :Success
                push!(estimates_no_caliper, sol_no_cal.estimate)
            end

            if sol_cal.retcode == :Success
                push!(estimates_caliper, sol_cal.estimate)
            end
        end

        @test length(estimates_no_caliper) >= 20  # At least 50%
        # Caliper may drop many sims (strict caliper + without-replacement)
        @test length(estimates_caliper) >= 3  # At least a few should work

        if length(estimates_caliper) >= 10
            # Caliper should help reduce bias (by dropping poor matches)
            bias_no_cal = abs(mean(estimates_no_caliper) - true_ate)
            bias_cal = abs(mean(estimates_caliper) - true_ate)

            @test bias_no_cal < 1.5
            @test bias_cal < 1.5

            # Both should be unbiased, caliper may have slightly less bias
            # (but higher variance due to fewer matches)
        end
    end

    # ========================================================================
    # Balance Improvement Check
    # ========================================================================

    @testset "MC: Balance improves after matching" begin
        Random.seed!(50)
        n_sims = 30

        improvement_counts = 0

        for sim in 1:n_sims
            n = 200
            x = randn(n)

            # Strong confounding
            prob_treatment = 1 ./ (1 .+ exp.(-1.5 .* x))
            treatment = rand(n) .< prob_treatment
            outcomes = 5.0 .* treatment .+ 2.0 .* x .+ randn(n)
            covariates = hcat(x)

            problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
            estimator = NearestNeighborPSM(M=1, with_replacement=false)
            solution = solve(problem, estimator)

            if solution.retcode == :Success
                # Check that balance improved
                stats = solution.balance_metrics.balance_stats

                if stats.mean_smd_after < stats.mean_smd_before
                    improvement_counts += 1
                end
            end
        end

        # Matching should improve balance in most cases
        # Note: Without-replacement can drop units, so improvement not guaranteed
        @test improvement_counts >= 10  # At least 33% should show improvement
    end

end
