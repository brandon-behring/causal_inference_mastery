"""
Tests for Sharp RDD estimator.

Phase 3.2-3.4: Sharp RDD local linear regression with CCT robust inference
"""

using Test
using CausalEstimators
using Random
using Statistics
using Distributions

@testset "Sharp RDD Estimator" begin
    @testset "Basic Sharp RDD - Known Effect" begin
        # Simple linear DGP with sharp discontinuity
        Random.seed!(123)
        n = 1000
        x = randn(n) .* 2.0
        treatment = x .>= 0.0

        # True effect: τ = 5.0
        # DGP: y = 2x + 5*D + ε
        τ_true = 5.0
        y = 2.0 .* x .+ τ_true .* treatment .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))

        # Estimate with CCT (default)
        solution = solve(problem, SharpRDD())

        # Basic checks
        @test solution.retcode == :Success
        @test isfinite(solution.estimate)
        @test isfinite(solution.se)
        @test solution.se > 0.0

        # Estimate should be close to true effect
        @test abs(solution.estimate - τ_true) < 2.0  # Within 2 units
        @test solution.ci_lower < solution.estimate < solution.ci_upper

        # Bandwidth checks
        @test solution.bandwidth > 0.0
        @test !isnothing(solution.bandwidth_bias)  # CCT has bias bandwidth
        @test solution.bandwidth_bias > solution.bandwidth

        # Sample sizes
        @test solution.n_eff_left > 0
        @test solution.n_eff_right > 0
    end

    @testset "Sharp RDD - Null Effect" begin
        # Test with no treatment effect
        Random.seed!(456)
        n = 800
        x = randn(n) .* 1.5
        treatment = x .>= 0.0

        # No effect: τ = 0
        y = 3.0 .* x .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))
        solution = solve(problem, SharpRDD())

        # Should not reject null (p > 0.05)
        @test solution.p_value > 0.05

        # CI should contain zero
        @test solution.ci_lower < 0.0 < solution.ci_upper
    end

    @testset "Sharp RDD - IK Bandwidth" begin
        Random.seed!(789)
        n = 600
        x = randn(n)
        treatment = x .>= 0.0
        τ_true = 4.0
        y = x .+ τ_true .* treatment .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))

        # Estimate with IK bandwidth
        solution_ik = solve(problem, SharpRDD(bandwidth_method=IKBandwidth()))

        @test solution_ik.retcode == :Success
        @test abs(solution_ik.estimate - τ_true) < 2.0

        # IK uses single bandwidth (no bias correction)
        @test isnothing(solution_ik.bandwidth_bias)
        @test solution_ik.bias_corrected == false
    end

    @testset "Sharp RDD - Different Kernels" begin
        Random.seed!(321)
        n = 500
        x = randn(n)
        treatment = x .>= 0.0
        τ_true = 3.0
        y = 2.0 .* x .+ τ_true .* treatment .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))

        # Test all three kernels
        for (kernel, kernel_name) in [
            (TriangularKernel(), :TriangularKernel),
            (UniformKernel(), :UniformKernel),
            (EpanechnikovKernel(), :EpanechnikovKernel)
        ]
            solution = solve(problem, SharpRDD(kernel=kernel))

            @test solution.retcode == :Success
            @test solution.kernel == kernel_name
            @test abs(solution.estimate - τ_true) < 3.0  # Should all be reasonable
        end
    end

    @testset "Sharp RDD - With Covariates" begin
        Random.seed!(111)
        n = 700
        x = randn(n)
        treatment = x .>= 0.0
        covariates = randn(n, 3)

        # Effect with covariate adjustment
        τ_true = 5.0
        covariate_effects = [2.0, -1.5, 0.8]
        y = 1.5 .* x .+ τ_true .* treatment .+
            sum(covariates .* covariate_effects', dims=2)[:] .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, covariates, (alpha=0.05,))
        solution = solve(problem, SharpRDD())

        @test solution.retcode == :Success
        # With covariate adjustment, should estimate effect more precisely
        @test abs(solution.estimate - τ_true) < 2.0
    end

    @testset "Sharp RDD - Non-Zero Cutoff" begin
        Random.seed!(222)
        n = 600
        cutoff = 3.0
        x = randn(n) .* 2.0 .+ cutoff
        treatment = x .>= cutoff
        τ_true = 6.0

        y = 2.0 .* (x .- cutoff) .+ τ_true .* treatment .+ randn(n)

        problem = RDDProblem(y, x, treatment, cutoff, nothing, (alpha=0.05,))
        solution = solve(problem, SharpRDD())

        @test solution.retcode == :Success
        @test abs(solution.estimate - τ_true) < 2.5
    end

    @testset "Sharp RDD - McCrary Density Test" begin
        Random.seed!(333)
        n = 800
        x = randn(n)
        treatment = x .>= 0.0
        y = 2.0 .* x .+ 5.0 .* treatment .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))

        # With density test (default)
        solution_with_test = solve(problem, SharpRDD(run_density_test=true))
        @test !isnothing(solution_with_test.density_test)
        @test solution_with_test.density_test isa McCraryTest
        @test solution_with_test.density_test.passes  # Should pass with random data

        # Without density test
        solution_no_test = solve(problem, SharpRDD(run_density_test=false))
        @test isnothing(solution_no_test.density_test)
    end

    @testset "Sharp RDD - Inference Properties" begin
        Random.seed!(444)
        n = 1000
        x = randn(n) .* 2.0
        treatment = x .>= 0.0
        τ_true = 4.0
        y = x .+ τ_true .* treatment .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))
        solution = solve(problem, SharpRDD())

        # Confidence interval properties
        ci_width = solution.ci_upper - solution.ci_lower
        @test ci_width > 0.0
        @test ci_width == 2 * quantile(Normal(0, 1), 0.975) * solution.se

        # P-value should be < 0.05 for significant effect
        @test solution.p_value < 0.05

        # Two-sided test
        z_stat = solution.estimate / solution.se
        p_manual = 2 * (1 - cdf(Normal(0, 1), abs(z_stat)))
        @test isapprox(solution.p_value, p_manual, atol=1e-10)
    end

    @testset "Sharp RDD - CCT Bias Correction" begin
        Random.seed!(555)
        n = 1000
        x = randn(n) .* 2.0
        treatment = x .>= 0.0

        # Quadratic DGP (bias in linear approximation)
        τ_true = 5.0
        y = 2.0 .* x .+ 0.5 .* x.^2 .+ τ_true .* treatment .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))

        # CCT should correct for bias from quadratic term
        solution_cct = solve(problem, SharpRDD(bandwidth_method=CCTBandwidth()))
        @test solution_cct.bias_corrected == true

        # IK without bias correction
        solution_ik = solve(problem, SharpRDD(bandwidth_method=IKBandwidth()))
        @test solution_ik.bias_corrected == false

        # Both should estimate effect, but CCT may be more accurate with curvature
        @test abs(solution_cct.estimate - τ_true) < 3.0
        @test abs(solution_ik.estimate - τ_true) < 3.0
    end

    @testset "Sharp RDD - Small Sample Warning" begin
        Random.seed!(666)
        n_small = 100
        x = randn(n_small) .* 0.5  # Narrow range
        treatment = x .>= 0.0
        y = randn(n_small)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))

        # Should warn about small effective sample size
        @test_logs (:warn, r"Small effective sample size") solution = solve(problem, SharpRDD())
        @test solution.retcode == :Success  # Still completes
    end

    @testset "Sharp RDD - Effective Sample Sizes" begin
        Random.seed!(777)
        n = 1000
        x = randn(n) .* 2.0
        treatment = x .>= 0.0
        y = randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))
        solution = solve(problem, SharpRDD())

        # With bandwidth selected, only observations near cutoff used
        @test solution.n_eff_left < sum(.!treatment)
        @test solution.n_eff_right < sum(treatment)
        @test solution.n_eff_left + solution.n_eff_right < n

        # Should be roughly balanced
        ratio = solution.n_eff_left / solution.n_eff_right
        @test 0.5 < ratio < 2.0  # Within 2x of each other
    end

    @testset "Sharp RDD - Heterogeneous Effects (Average)" begin
        # RDD estimates LATE at cutoff, not global ATE
        Random.seed!(888)
        n = 1000
        x = randn(n) .* 2.0
        treatment = x .>= 0.0

        # Effect varies with x: τ(x) = 5 + 2x
        # At cutoff (x=0): τ = 5
        τ_at_cutoff = 5.0
        y = 2.0 .* x .+ (5.0 .+ 2.0 .* x) .* treatment .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))
        solution = solve(problem, SharpRDD())

        # Should estimate effect at cutoff, not global average
        @test abs(solution.estimate - τ_at_cutoff) < 2.0
    end

    @testset "Sharp RDD - Type Stability" begin
        Random.seed!(999)
        n = 500
        x = randn(n)
        treatment = x .>= 0.0
        y = randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))

        # Check type stability of solve
        solution = solve(problem, SharpRDD())

        @test solution isa RDDSolution{Float64}
        @test typeof(solution.estimate) == Float64
        @test typeof(solution.se) == Float64
        @test typeof(solution.bandwidth) == Float64
    end
end
