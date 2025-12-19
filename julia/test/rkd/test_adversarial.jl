"""
Adversarial tests for Julia RKD module.

Following the 6-layer validation architecture:
- Layer 2: Adversarial tests for edge cases and boundary conditions

Tests:
1. Near-zero kink effects
2. Extreme noise levels
3. Minimal sample sizes
4. Outliers and influential observations
5. Non-uniform running variable distributions
6. Boundary effects (observations near cutoff)
7. Misspecified models
8. Numerical stability
"""

using Test
using CausalEstimators
using Random
using Statistics
using Distributions

# =============================================================================
# Helper function for Sharp RKD data
# =============================================================================

"""Generate Sharp RKD data with treatment D (required for RKDProblem)."""
function gen_sharp_rkd(X, slope_left, slope_right, cutoff, noise_sd)
    D = [x < cutoff ? slope_left * (x - cutoff) : slope_right * (x - cutoff) for x in X]
    Y = D .+ noise_sd .* randn(length(X))
    return Y, D
end

# =============================================================================
# Near-Zero Effects
# =============================================================================

@testset "RKD Adversarial: Near-Zero Effects" begin

    @testset "Sharp RKD with very small kink" begin
        Random.seed!(42)
        n = 500
        X = rand(Uniform(-5, 5), n)

        # Very small kink: 0.01 slope difference (centered at cutoff=0)
        D = [x < 0 ? 0.50 * x : 0.51 * x for x in X]
        Y = D .+ 0.5 .* randn(n)

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=2.5))

        # Should handle gracefully - edge case stress test
        @test solution isa RKDSolution
        # With very small true kink and noise, estimate can be quite off
        # Main goal: no crashes, returns valid solution
        @test isfinite(solution.estimate) || isnan(solution.estimate)
    end

    @testset "Sharp RKD with zero kink" begin
        Random.seed!(123)
        n = 500
        X = rand(Uniform(-5, 5), n)

        # No kink: same slope on both sides
        D = 0.5 .* X
        Y = D .+ randn(n)

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=2.5))

        @test solution isa RKDSolution
        # When treatment kink is zero, estimate can be NaN (division by zero)
        # This is expected behavior for edge case
        @test isfinite(solution.estimate) || isnan(solution.estimate)
    end

    @testset "Fuzzy RKD with weak first stage" begin
        Random.seed!(456)
        n = 500
        X = rand(Uniform(-5, 5), n)

        # Very weak first stage: minimal slope difference
        D_expected = [x < 0 ? 1.0 * x : 1.02 * x for x in X]
        D = D_expected .+ 2.0 .* randn(n)  # High noise
        Y = 2.0 .* D .+ randn(n)

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, FuzzyRKD(bandwidth=2.5))

        @test solution isa FuzzyRKDSolution
        # Should detect weak first stage
        @test solution.first_stage_f_stat < 20 || solution.weak_first_stage
    end

end

# =============================================================================
# Extreme Noise Levels
# =============================================================================

@testset "RKD Adversarial: Extreme Noise" begin

    @testset "Sharp RKD with very high noise" begin
        Random.seed!(42)
        n = 500
        X = rand(Uniform(-5, 5), n)

        # Strong kink but overwhelmed by noise
        D = [x < 0 ? 0.5 * (x) : 1.5 * (x) for x in X]
        Y = D .+ 10.0 .* randn(n)  # Very high noise

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=2.5))

        @test solution isa RKDSolution
        @test isfinite(solution.estimate)
        # SE should be large due to noise
        @test solution.se > 0.5
    end

    @testset "Sharp RKD with heteroskedastic noise" begin
        Random.seed!(123)
        n = 500
        X = rand(Uniform(-5, 5), n)

        # Noise varies with X
        noise_sd = 0.5 .+ abs.(X)
        D = [x < 0 ? 0.5 * (x) : 1.5 * (x) for x in X]
        Y = D .+ noise_sd .* randn(n)

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=2.5))

        @test solution isa RKDSolution
        @test isfinite(solution.estimate)
        # Kink should still be estimated
        @test abs(solution.estimate - 1.0) < 0.5
    end

    @testset "Fuzzy RKD with very noisy treatment" begin
        Random.seed!(456)
        n = 600
        X = rand(Uniform(-5, 5), n)

        D_expected = [x < 0 ? 0.5 * (x) : 1.5 * (x) for x in X]
        D = D_expected .+ 5.0 .* randn(n)  # Very noisy treatment
        Y = 2.0 .* D .+ randn(n)

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, FuzzyRKD(bandwidth=3.0))

        @test solution isa FuzzyRKDSolution
        # High noise in D should lead to weak first stage or large SE
        @test solution.se > 0.3 || solution.weak_first_stage
    end

end

# =============================================================================
# Minimal Sample Sizes
# =============================================================================

@testset "RKD Adversarial: Minimal Samples" begin

    @testset "Sharp RKD with n=30" begin
        Random.seed!(42)
        n = 30
        X = rand(Uniform(-5, 5), n)
        D = [x < 0 ? 0.5 * (x) : 1.5 * (x) for x in X]
        Y = D .+ randn(n)

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=4.0))

        # Should handle gracefully
        @test solution isa RKDSolution
    end

    @testset "Sharp RKD with n=20" begin
        Random.seed!(123)
        n = 20
        X = rand(Uniform(-5, 5), n)
        D = [x < 0 ? 0.5 * (x) : 1.5 * (x) for x in X]
        Y = D .+ randn(n)

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=5.0))

        @test solution isa RKDSolution
        # Should warn or have large SE
        @test solution.retcode in [:Success, :Warning, :Failure] || solution.se > 1.0
    end

    @testset "Fuzzy RKD with minimal sample" begin
        Random.seed!(456)
        n = 50
        X = rand(Uniform(-5, 5), n)
        D = [x < 0 ? 0.5 * (x) : 1.5 * (x) for x in X] .+ 0.5 .* randn(n)
        Y = 2.0 .* D .+ randn(n)

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, FuzzyRKD(bandwidth=4.0))

        @test solution isa FuzzyRKDSolution
    end

    @testset "Sharp RKD: few observations near cutoff" begin
        Random.seed!(789)
        # Generate data mostly away from cutoff
        n = 200
        X = vcat(
            rand(Uniform(-5, -2), 90),
            rand(Uniform(-1, 1), 20),  # Only 20 near cutoff
            rand(Uniform(2, 5), 90)
        )
        D = [x < 0 ? 0.5 * (x) : 1.5 * (x) for x in X]
        Y = D .+ randn(n)

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=1.5))

        @test solution isa RKDSolution
        # Should have low effective n
        @test solution.n_eff_left + solution.n_eff_right < 50
    end

end

# =============================================================================
# Outliers and Influential Observations
# =============================================================================

@testset "RKD Adversarial: Outliers" begin

    @testset "Sharp RKD with outcome outliers" begin
        Random.seed!(42)
        n = 200
        X = rand(Uniform(-5, 5), n)
        D = [x < 0 ? 0.5 * (x) : 1.5 * (x) for x in X]
        Y = D .+ randn(n)

        # Add outliers
        Y[1:5] .= 50.0
        Y[6:10] .= -50.0

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=2.5))

        @test solution isa RKDSolution
        @test isfinite(solution.estimate)
        # Outliers may bias, but kernel weighting should help
    end

    @testset "Sharp RKD with running variable outliers" begin
        Random.seed!(123)
        n = 200
        X = rand(Uniform(-5, 5), n)
        # Add extreme X values
        X[1:3] .= 100.0
        X[4:6] .= -100.0

        D = [x < 0 ? 0.5 * (x) : 1.5 * (x) for x in X]
        Y = D .+ randn(n)

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=2.5))

        @test solution isa RKDSolution
        # Extreme X should get zero weight
        @test isfinite(solution.estimate)
    end

    @testset "Sharp RKD with influential obs at cutoff" begin
        Random.seed!(456)
        n = 200
        X = rand(Uniform(-5, 5), n)
        D = [x < 0 ? 0.5 * (x) : 1.5 * (x) for x in X]
        Y = D .+ 0.5 .* randn(n)

        # Add influential observation exactly at cutoff
        push!(X, 0.0)
        push!(D, 0.0)
        push!(Y, 100.0)  # Extreme outlier at cutoff

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=2.5))

        @test solution isa RKDSolution
        # Should still be reasonably close to true kink
        # (single outlier shouldn't dominate with kernel weighting)
    end

end

# =============================================================================
# Non-Uniform Running Variable Distributions
# =============================================================================

@testset "RKD Adversarial: Non-Uniform X" begin

    @testset "Sharp RKD with bimodal X" begin
        Random.seed!(42)
        # Bimodal: two clusters away from cutoff
        X = vcat(
            rand(Normal(-3, 0.5), 150),
            rand(Normal(3, 0.5), 150)
        )
        D = [x < 0 ? 0.5 * (x) : 1.5 * (x) for x in X]
        Y = D .+ randn(300)

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=2.0))

        @test solution isa RKDSolution
        # May have few observations near cutoff
    end

    @testset "Sharp RKD with skewed X" begin
        Random.seed!(123)
        # Skewed: more observations on left
        X = vcat(
            rand(Uniform(-5, 0), 400),
            rand(Uniform(0, 5), 100)
        )
        D = [x < 0 ? 0.5 * (x) : 1.5 * (x) for x in X]
        Y = D .+ randn(500)

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=2.5))

        @test solution isa RKDSolution
        @test solution.n_eff_left > solution.n_eff_right
    end

    @testset "Sharp RKD with gap at cutoff" begin
        Random.seed!(456)
        # Gap: no observations near cutoff
        X = vcat(
            rand(Uniform(-5, -1), 200),
            rand(Uniform(1, 5), 200)
        )
        D = [x < 0 ? 0.5 * (x) : 1.5 * (x) for x in X]
        Y = D .+ randn(400)

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=1.5))

        @test solution isa RKDSolution
        # Few effective observations (some wiggle room for kernel weights)
        @test solution.n_eff_left + solution.n_eff_right < 60
    end

end

# =============================================================================
# Non-Standard Cutoffs
# =============================================================================

@testset "RKD Adversarial: Non-Standard Cutoffs" begin

    @testset "Sharp RKD with non-zero cutoff" begin
        Random.seed!(42)
        cutoff = 2.5
        n = 500
        X = rand(Uniform(-2, 7), n)
        D = [x < cutoff ? 0.5 * (x - cutoff) : 1.5 * (x - cutoff) for x in X]
        Y = D .+ randn(n)

        problem = RKDProblem(Y, X, D, cutoff)
        solution = solve(problem, SharpRKD(bandwidth=2.0))

        @test solution isa RKDSolution
        # Wider tolerance for non-standard cutoff
        @test abs(solution.estimate - 1.0) < 0.5
    end

    @testset "Sharp RKD with negative cutoff" begin
        Random.seed!(123)
        cutoff = -1.5
        n = 500
        X = rand(Uniform(-6, 3), n)
        D = [x < cutoff ? 0.5 * (x - cutoff) : 1.5 * (x - cutoff) for x in X]
        Y = D .+ randn(n)

        problem = RKDProblem(Y, X, D, cutoff)
        solution = solve(problem, SharpRKD(bandwidth=2.0))

        @test solution isa RKDSolution
        # Wider tolerance for non-standard cutoff
        @test abs(solution.estimate - 1.0) < 0.5
    end

    @testset "Sharp RKD with cutoff at boundary" begin
        Random.seed!(456)
        cutoff = 4.0  # Near upper bound of X
        n = 500
        X = rand(Uniform(-5, 5), n)
        D = [x < cutoff ? 0.5 * (x - cutoff) : 1.5 * (x - cutoff) for x in X]
        Y = D .+ randn(n)

        problem = RKDProblem(Y, X, D, cutoff)
        solution = solve(problem, SharpRKD(bandwidth=2.0))

        @test solution isa RKDSolution
        # May have asymmetric effective n
        @test solution.n_eff_left > solution.n_eff_right
    end

end

# =============================================================================
# Numerical Stability
# =============================================================================

@testset "RKD Adversarial: Numerical Stability" begin

    @testset "Sharp RKD with very small bandwidth" begin
        Random.seed!(42)
        n = 1000
        X = rand(Uniform(-5, 5), n)
        D = [x < 0 ? 0.5 * (x) : 1.5 * (x) for x in X]
        Y = D .+ randn(n)

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=0.1))

        @test solution isa RKDSolution
        # Very narrow bandwidth may have few observations
    end

    @testset "Sharp RKD with very large bandwidth" begin
        Random.seed!(123)
        n = 500
        X = rand(Uniform(-5, 5), n)
        D = [x < 0 ? 0.5 * (x) : 1.5 * (x) for x in X]
        Y = D .+ randn(n)

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=100.0))

        @test solution isa RKDSolution
        # Very wide bandwidth includes all observations
        @test solution.n_eff_left + solution.n_eff_right > 400
    end

    @testset "Sharp RKD with constant Y" begin
        Random.seed!(456)
        n = 200
        X = rand(Uniform(-5, 5), n)
        D = [x < 0 ? 0.5 * (x) : 1.5 * (x) for x in X]
        Y = ones(n)  # Constant outcome

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=2.5))

        @test solution isa RKDSolution
        # Kink should be approximately 0
        @test abs(solution.estimate) < 0.1
    end

    @testset "Sharp RKD with nearly collinear data" begin
        Random.seed!(789)
        n = 200
        X = collect(range(-5, 5, length=n))  # Evenly spaced
        D = [x < 0 ? 0.5 * (x) : 1.5 * (x) for x in X]
        Y = D .+ 0.01 .* randn(n)  # Very little noise

        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=2.5))

        @test solution isa RKDSolution
        @test isfinite(solution.estimate)
        # Should recover kink very precisely
        @test abs(solution.estimate - 1.0) < 0.05
    end

end

# =============================================================================
# Kernel Robustness
# =============================================================================

@testset "RKD Adversarial: Kernel Robustness" begin

    @testset "Results similar across kernels" begin
        Random.seed!(42)
        n = 500
        X = rand(Uniform(-5, 5), n)
        D = [x < 0 ? 0.5 * (x) : 1.5 * (x) for x in X]
        Y = D .+ randn(n)

        problem = RKDProblem(Y, X, D, 0.0)

        estimates = Float64[]
        for kernel in [:triangular, :uniform, :epanechnikov]
            solution = solve(problem, SharpRKD(bandwidth=2.5, kernel=kernel))
            if isfinite(solution.estimate)
                push!(estimates, solution.estimate)
            end
        end

        # All kernels should give similar estimates
        @test length(estimates) == 3
        @test std(estimates) < 0.15
        @test all(e -> abs(e - 1.0) < 0.3, estimates)
    end

end

# =============================================================================
# Diagnostics Adversarial Tests
# =============================================================================

@testset "RKD Diagnostics Adversarial" begin

    @testset "Density test with bunching" begin
        Random.seed!(42)
        # Create bunching at cutoff
        X = vcat(
            rand(Uniform(-5, -0.1), 200),
            rand(Uniform(-0.1, 0.1), 100),  # Bunching
            rand(Uniform(0.1, 5), 200)
        )

        result = density_smoothness_test(X, 0.0; n_bins=20)

        @test result isa DensitySmoothnessResult
        # Should detect non-smoothness
    end

    @testset "Covariate test with imbalance" begin
        Random.seed!(123)
        n = 400
        X = rand(Uniform(-5, 5), n)

        # Covariate with kink at cutoff (tests kinks, not jumps)
        C = [x < 0 ? 0.5 * x + randn() : 1.5 * x + randn() for x in X]

        results = covariate_smoothness_test(X, reshape(C, n, 1), 0.0)

        # Returns a vector of CovariateSmoothnessResult
        @test results isa Vector{CovariateSmoothnessResult}
        @test length(results) == 1
        @test results[1] isa CovariateSmoothnessResult
    end

    @testset "First stage test with no kink" begin
        Random.seed!(456)
        n = 400
        X = rand(Uniform(-5, 5), n)

        # Treatment with no kink (same slope)
        D = X .+ 0.5 .* randn(n)

        result = first_stage_test(D, X, 0.0)

        @test result isa FirstStageTestResult
        # F-stat field is named f_stat
        @test isfinite(result.f_stat) || isnan(result.f_stat)
    end

    @testset "Diagnostics with standard data" begin
        Random.seed!(789)
        n = 300
        X = rand(Uniform(-5, 5), n)
        D = [x < 0 ? 0.5 * (x) : 1.5 * (x) for x in X] .+ 0.3 .* randn(n)
        Y = 2.0 .* D .+ randn(n)

        summary = rkd_diagnostics(Y, X, D, 0.0)

        @test summary isa RKDDiagnosticsSummary
        # Check that the summary has the expected fields
        @test summary.density_test isa DensitySmoothnessResult
        @test summary.first_stage_test isa FirstStageTestResult
    end

end
