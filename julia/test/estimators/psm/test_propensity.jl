"""
Tests for propensity score estimation.
"""

using Test
using Random
using Statistics
using CausalEstimators

@testset "Propensity Score Estimation" begin

    # ========================================================================
    # Known-Answer Tests
    # ========================================================================

    @testset "Known-Answer: No relationship (propensity ≈ proportion)" begin
        Random.seed!(42)
        n = 1000
        # Treatment independent of covariates
        treatment = vcat(fill(true, 500), fill(false, 500))
        covariates = randn(n, 2)  # Covariates unrelated to treatment

        propensity = estimate_propensity(treatment, covariates)

        # Propensity should be ≈ 0.5 (proportion treated)
        @test abs(mean(propensity) - 0.5) < 0.05  # Within 0.05 of 0.5
        @test all(0 .< propensity .< 1)  # All in (0, 1)
    end

    @testset "Known-Answer: Strong positive relationship" begin
        Random.seed!(42)
        n = 200
        # Treatment strongly related to covariate
        x = randn(n)
        prob_treatment = 1 ./ (1 .+ exp.(-3 .* x))  # True propensity
        treatment = rand(n) .< prob_treatment

        covariates = hcat(x)
        propensity = estimate_propensity(treatment, covariates)

        # Estimated propensity should be close to true propensity
        # Correlation should be strong
        @test cor(propensity, prob_treatment) > 0.9
        @test all(0 .< propensity .< 1)
    end

    @testset "Known-Answer: Balanced randomization" begin
        Random.seed!(42)
        n = 100
        # Truly random treatment (RCT)
        treatment = rand(Bool, n)
        covariates = randn(n, 3)

        propensity = estimate_propensity(treatment, covariates)

        # Propensity should be ≈ 0.5 ± some noise
        @test 0.3 < mean(propensity) < 0.7  # Loose bounds for random
        @test all(0 .< propensity .< 1)
    end

    # ========================================================================
    # Adversarial Tests: Numerical Stability
    # ========================================================================

    @testset "Adversarial: Extreme covariate values" begin
        Random.seed!(42)
        n = 100
        # Covariates with extreme values
        x = vcat(fill(10.0, 50), fill(-10.0, 50))
        treatment = x .> 0  # Perfect separation

        covariates = hcat(x)

        # Should still return propensity in (0, 1) but may warn
        propensity = estimate_propensity(treatment, covariates)

        @test all(0 .< propensity .< 1)
        # Treated should have high propensity, control low
        @test mean(propensity[treatment]) > mean(propensity[.!treatment])
    end

    @testset "Adversarial: Multicollinearity" begin
        Random.seed!(42)
        n = 100
        # Two perfectly correlated covariates
        x1 = randn(n)
        x2 = x1  # Perfect correlation
        treatment = rand(Bool, n)

        covariates = hcat(x1, x2)

        # GLM should handle this (may be unstable)
        propensity = estimate_propensity(treatment, covariates)

        @test all(0 .< propensity .< 1)
    end

    @testset "Adversarial: High-dimensional (p = n/2)" begin
        Random.seed!(42)
        n = 50
        p = 25  # p = n/2
        treatment = vcat(fill(true, 25), fill(false, 25))
        covariates = randn(n, p)

        # Should work but may overfit
        propensity = estimate_propensity(treatment, covariates)

        @test all(0 .< propensity .< 1)
    end

    @testset "Adversarial: Single covariate" begin
        Random.seed!(42)
        n = 100
        x = randn(n)
        treatment = x .> 0

        covariates = hcat(x)
        propensity = estimate_propensity(treatment, covariates)

        @test all(0 .< propensity .< 1)
        # Treated should have higher propensity
        @test mean(propensity[treatment]) > 0.5
        @test mean(propensity[.!treatment]) < 0.5
    end

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    @testset "Error: Mismatched lengths" begin
        treatment = [true, false, true]
        covariates = randn(4, 2)  # Wrong length

        @test_throws ArgumentError estimate_propensity(treatment, covariates)
    end

    # ========================================================================
    # Common Support Tests
    # ========================================================================

    @testset "Common Support: Good overlap" begin
        Random.seed!(42)
        n = 200
        # Treatment and control with overlapping propensities
        # Generate treatment probabilistically (not deterministically)
        x = randn(n)
        prob_treatment = 1 ./ (1 .+ exp.(-0.5 .* x))  # Moderate effect
        treatment = rand(n) .< prob_treatment

        covariates = hcat(x)
        propensity = estimate_propensity(treatment, covariates)

        has_support, region, n_outside = check_common_support(propensity, treatment)

        @test has_support == true
        @test region[1] < region[2]  # Valid region
        @test n_outside >= 0
    end

    @testset "Common Support: Disjoint distributions" begin
        Random.seed!(42)
        n = 100
        # Treatment and control completely separated
        x = vcat(fill(10.0, 50), fill(-10.0, 50))
        treatment = x .> 0

        covariates = hcat(x)
        propensity = estimate_propensity(treatment, covariates)

        has_support, region, n_outside = check_common_support(propensity, treatment)

        # May or may not have support depending on threshold
        # But n_outside should be large
        @test n_outside > 0
    end

    @testset "Common Support: Minimal overlap" begin
        Random.seed!(42)
        n = 100
        # Very little overlap
        # Use strong effect (large coefficient) for narrow overlap
        x = randn(n)
        prob_treatment = 1 ./ (1 .+ exp.(-3.0 .* x))  # Strong effect -> minimal overlap
        treatment = rand(n) .< prob_treatment

        covariates = hcat(x)
        propensity = estimate_propensity(treatment, covariates)

        has_support, region, n_outside = check_common_support(
            propensity,
            treatment;
            threshold = 0.05,
        )

        # Should have some support but many units outside
        @test region[1] < region[2]
    end

    # ========================================================================
    # PSMProblem Constructor Tests
    # ========================================================================

    @testset "PSMProblem: Valid construction" begin
        outcomes = [10.0, 12.0, 4.0, 5.0]
        treatment = [true, true, false, false]
        covariates = randn(4, 2)

        problem = PSMProblem(outcomes, treatment, covariates, (alpha = 0.05,))

        @test problem.outcomes == outcomes
        @test problem.treatment == treatment
        @test problem.covariates == covariates
        @test problem.parameters.alpha == 0.05
    end

    @testset "PSMProblem: Error on empty inputs" begin
        outcomes = Float64[]
        treatment = Bool[]
        covariates = Matrix{Float64}(undef, 0, 2)

        @test_throws ArgumentError PSMProblem(
            outcomes,
            treatment,
            covariates,
            (alpha = 0.05,),
        )
    end

    @testset "PSMProblem: Error on all treated" begin
        outcomes = [10.0, 12.0, 11.0]
        treatment = [true, true, true]
        covariates = randn(3, 2)

        @test_throws ArgumentError PSMProblem(
            outcomes,
            treatment,
            covariates,
            (alpha = 0.05,),
        )
    end

    @testset "PSMProblem: Error on all control" begin
        outcomes = [10.0, 12.0, 11.0]
        treatment = [false, false, false]
        covariates = randn(3, 2)

        @test_throws ArgumentError PSMProblem(
            outcomes,
            treatment,
            covariates,
            (alpha = 0.05,),
        )
    end

    @testset "PSMProblem: Error on NaN outcomes" begin
        outcomes = [10.0, NaN, 4.0, 5.0]
        treatment = [true, true, false, false]
        covariates = randn(4, 2)

        @test_throws ArgumentError PSMProblem(
            outcomes,
            treatment,
            covariates,
            (alpha = 0.05,),
        )
    end

    @testset "PSMProblem: Error on NaN covariates" begin
        outcomes = [10.0, 12.0, 4.0, 5.0]
        treatment = [true, true, false, false]
        covariates = [5.0 2.0; NaN 3.0; 5.5 2.5; 4.5 2.0]

        @test_throws ArgumentError PSMProblem(
            outcomes,
            treatment,
            covariates,
            (alpha = 0.05,),
        )
    end

    @testset "PSMProblem: Error on insufficient treated" begin
        outcomes = [10.0, 12.0, 4.0]
        treatment = [true, false, false]  # Only 1 treated
        covariates = randn(3, 2)

        @test_throws ArgumentError PSMProblem(
            outcomes,
            treatment,
            covariates,
            (alpha = 0.05,),
        )
    end

    @testset "PSMProblem: Error on invalid alpha" begin
        outcomes = [10.0, 12.0, 4.0, 5.0]
        treatment = [true, true, false, false]
        covariates = randn(4, 2)

        @test_throws ArgumentError PSMProblem(
            outcomes,
            treatment,
            covariates,
            (alpha = 1.5,),  # Invalid alpha > 1
        )
    end

end
