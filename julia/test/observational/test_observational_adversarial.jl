"""
Layer 2: Adversarial tests for Julia observational estimator edge cases.

Tests boundary conditions, extreme inputs, and degenerate cases.
All tests should either:
1. Return valid results (graceful handling)
2. Throw explicit ArgumentError with diagnostic message
3. Never fail silently

References:
    - Rosenbaum & Rubin (1983). The central role of the propensity score.
    - Bang & Robins (2005). Doubly robust estimation.
"""

using Test
using Statistics
using Random
using LinearAlgebra

# Include main module
include("../../src/CausalEstimators.jl")
using .CausalEstimators

# =============================================================================
# Input Validation Tests
# =============================================================================

@testset "Observational Input Validation" begin

    @testset "NaN in outcomes" begin
        X = randn(100, 3)
        T = rand(100) .> 0.5
        Y = randn(100)
        Y[50] = NaN

        @test_throws ArgumentError ObservationalProblem(Y, T, X)
    end

    @testset "Inf in outcomes" begin
        X = randn(100, 3)
        T = rand(100) .> 0.5
        Y = randn(100)
        Y[50] = Inf

        @test_throws ArgumentError ObservationalProblem(Y, T, X)
    end

    @testset "-Inf in outcomes" begin
        X = randn(100, 3)
        T = rand(100) .> 0.5
        Y = randn(100)
        Y[50] = -Inf

        @test_throws ArgumentError ObservationalProblem(Y, T, X)
    end

    @testset "NaN in covariates" begin
        X = randn(100, 3)
        X[50, 2] = NaN
        T = rand(100) .> 0.5
        Y = randn(100)

        @test_throws ArgumentError ObservationalProblem(Y, T, X)
    end

    @testset "Inf in covariates" begin
        X = randn(100, 3)
        X[50, 2] = Inf
        T = rand(100) .> 0.5
        Y = randn(100)

        @test_throws ArgumentError ObservationalProblem(Y, T, X)
    end

    @testset "Mismatched dimensions - Y and T" begin
        X = randn(100, 3)
        T = rand(50) .> 0.5  # Wrong length
        Y = randn(100)

        @test_throws ArgumentError ObservationalProblem(Y, T, X)
    end

    @testset "Mismatched dimensions - Y and X" begin
        X = randn(50, 3)  # Wrong rows
        T = rand(100) .> 0.5
        Y = randn(100)

        @test_throws ArgumentError ObservationalProblem(Y, T, X)
    end

    @testset "Zero covariates" begin
        X = randn(100, 0)  # No covariates
        T = rand(100) .> 0.5
        Y = randn(100)

        @test_throws ArgumentError ObservationalProblem(Y, T, X)
    end

end

# =============================================================================
# Treatment Variation Tests
# =============================================================================

@testset "Observational Treatment Variation" begin

    @testset "All treated (no control)" begin
        X = randn(100, 3)
        T = trues(100)  # All treated
        Y = randn(100)

        @test_throws ArgumentError ObservationalProblem(Y, T, X)
    end

    @testset "All control (no treated)" begin
        X = randn(100, 3)
        T = falses(100)  # All control
        Y = randn(100)

        @test_throws ArgumentError ObservationalProblem(Y, T, X)
    end

    @testset "Extreme imbalance - 95% treated (IPW)" begin
        Random.seed!(123)
        X = randn(100, 3)
        T = vcat(trues(95), falses(5))  # 95 treated, 5 control
        Y = randn(100)

        # Should create problem
        problem = ObservationalProblem(Y, T, X)

        # IPW may succeed but with warning
        solution = solve(problem, ObservationalIPW())
        @test solution.retcode in [:Success, :Warning]
    end

    @testset "Extreme imbalance - 95% control (IPW)" begin
        Random.seed!(124)
        X = randn(100, 3)
        T = vcat(trues(5), falses(95))  # 5 treated, 95 control
        Y = randn(100)

        problem = ObservationalProblem(Y, T, X)

        # IPW may succeed but with warning
        solution = solve(problem, ObservationalIPW())
        @test solution.retcode in [:Success, :Warning]
    end

    @testset "Extreme imbalance - 95% treated (DR)" begin
        Random.seed!(125)
        X = randn(100, 3)
        T = vcat(trues(95), falses(5))  # 95 treated, 5 control
        Y = randn(100)

        problem = ObservationalProblem(Y, T, X)

        # DR may succeed
        solution = solve(problem, DoublyRobust())
        @test solution.retcode in [:Success, :Warning]
    end

end

# =============================================================================
# Propensity Edge Cases
# =============================================================================

@testset "Propensity Score Edge Cases" begin

    @testset "Pre-provided propensity - valid" begin
        Random.seed!(200)
        n = 100
        X = randn(n, 3)
        T = rand(n) .> 0.5
        Y = randn(n)
        propensity = 0.3 .+ 0.4 .* rand(n)  # All in (0.3, 0.7)

        problem = ObservationalProblem(
            Y, T, X;
            propensity=propensity
        )

        solution = solve(problem, ObservationalIPW())
        @test solution.retcode == :Success
        @test isfinite(solution.estimate)
    end

    @testset "Pre-provided propensity - boundary values rejected" begin
        Random.seed!(201)
        n = 100
        X = randn(n, 3)
        T = rand(n) .> 0.5
        Y = randn(n)
        propensity = rand(n)
        propensity[1] = 0.0  # Exact zero

        @test_throws ArgumentError ObservationalProblem(
            Y, T, X;
            propensity=propensity
        )
    end

    @testset "Pre-provided propensity - 1.0 rejected" begin
        Random.seed!(202)
        n = 100
        X = randn(n, 3)
        T = rand(n) .> 0.5
        Y = randn(n)
        propensity = rand(n)
        propensity[1] = 1.0  # Exact one

        @test_throws ArgumentError ObservationalProblem(
            Y, T, X;
            propensity=propensity
        )
    end

    @testset "Pre-provided propensity - wrong length" begin
        Random.seed!(203)
        n = 100
        X = randn(n, 3)
        T = rand(n) .> 0.5
        Y = randn(n)
        propensity = rand(50)  # Wrong length

        @test_throws ArgumentError ObservationalProblem(
            Y, T, X;
            propensity=propensity
        )
    end

    @testset "Near-extreme propensities handled by trimming" begin
        Random.seed!(204)
        n = 200
        # Create data where propensity will be near extremes
        X = randn(n, 3)
        X[:, 1] = X[:, 1] .* 3  # Strong signal -> extreme propensities

        logit = 2.0 .* X[:, 1]
        true_prop = 1.0 ./ (1.0 .+ exp.(-logit))
        T = [rand() < p for p in true_prop]

        # Ensure both groups
        if sum(T) == 0 || sum(T) == n
            T[1] = true
            T[end] = false
        end

        Y = randn(n)

        problem = ObservationalProblem(
            Y, T, X;
            trim_threshold=0.05  # Trim extreme propensities
        )

        # Should handle gracefully
        solution = solve(problem, ObservationalIPW())
        @test solution.retcode in [:Success, :Warning]
        @test isfinite(solution.estimate)
    end

end

# =============================================================================
# Outcome Edge Cases
# =============================================================================

@testset "Outcome Edge Cases" begin

    @testset "Constant outcomes" begin
        Random.seed!(300)
        n = 100
        X = randn(n, 3)
        T = rand(n) .> 0.5
        Y = fill(5.0, n)  # Constant outcome

        problem = ObservationalProblem(Y, T, X)

        # Should produce zero effect estimate
        solution = solve(problem, ObservationalIPW())
        @test solution.retcode in [:Success, :Warning]
        @test abs(solution.estimate) < 1e-6
    end

    @testset "Very large outcomes" begin
        Random.seed!(301)
        n = 100
        X = randn(n, 3)
        T = rand(n) .> 0.5
        Y = randn(n) .* 1e6  # Large scale

        problem = ObservationalProblem(Y, T, X)

        solution = solve(problem, ObservationalIPW())
        @test solution.retcode in [:Success, :Warning]
        @test isfinite(solution.estimate)
    end

    @testset "Very small outcomes" begin
        Random.seed!(302)
        n = 100
        X = randn(n, 3)
        T = rand(n) .> 0.5
        Y = randn(n) .* 1e-6  # Tiny scale

        problem = ObservationalProblem(Y, T, X)

        solution = solve(problem, ObservationalIPW())
        @test solution.retcode in [:Success, :Warning]
        @test isfinite(solution.estimate)
    end

    @testset "Perfect treatment effect" begin
        Random.seed!(303)
        n = 100
        X = randn(n, 3)
        T = rand(n) .> 0.5
        Y = float.(T) .* 10.0  # Treated = 10, Control = 0

        problem = ObservationalProblem(Y, T, X)

        solution = solve(problem, ObservationalIPW())
        @test solution.retcode in [:Success, :Warning]
        # Effect should be near 10
        @test 8.0 < solution.estimate < 12.0
    end

end

# =============================================================================
# Covariate Edge Cases
# =============================================================================

@testset "Covariate Edge Cases" begin

    @testset "Single covariate" begin
        Random.seed!(400)
        n = 100
        X = randn(n, 1)  # Single covariate
        T = rand(n) .> 0.5
        Y = randn(n) .+ float.(T) .* 2.0

        problem = ObservationalProblem(Y, T, X)

        solution = solve(problem, ObservationalIPW())
        @test solution.retcode in [:Success, :Warning]
        @test isfinite(solution.estimate)
    end

    @testset "Many covariates (p > n/5)" begin
        Random.seed!(401)
        n = 100
        p = 30  # 30 > 100/5 = 20
        X = randn(n, p)
        T = rand(n) .> 0.5
        Y = randn(n) .+ float.(T) .* 2.0

        problem = ObservationalProblem(Y, T, X)

        # May have issues but should not crash
        try
            solution = solve(problem, ObservationalIPW())
            @test solution.retcode in [:Success, :Warning, :Error]
        catch e
            # High-dimensional may fail - acceptable
            @test e isa ArgumentError || e isa LinearAlgebra.SingularException
        end
    end

    @testset "Collinear covariates" begin
        Random.seed!(402)
        n = 100
        X = randn(n, 2)
        X[:, 2] = X[:, 1] .* 2.0 .+ 0.01 .* randn(n)  # Nearly collinear
        T = rand(n) .> 0.5
        Y = randn(n) .+ float.(T) .* 2.0

        problem = ObservationalProblem(Y, T, X)

        # May handle gracefully or error - both acceptable
        try
            solution = solve(problem, ObservationalIPW())
            @test solution.retcode in [:Success, :Warning]
        catch e
            @test e isa Union{ArgumentError, LinearAlgebra.SingularException}
        end
    end

    @testset "Constant covariate" begin
        Random.seed!(403)
        n = 100
        X = randn(n, 3)
        X[:, 2] .= 5.0  # Constant column
        T = rand(n) .> 0.5
        Y = randn(n) .+ float.(T) .* 2.0

        problem = ObservationalProblem(Y, T, X)

        # Should handle gracefully
        try
            solution = solve(problem, ObservationalIPW())
            @test solution.retcode in [:Success, :Warning]
        catch e
            # Constant column may cause regression issues
            @test e isa Union{ArgumentError, LinearAlgebra.SingularException}
        end
    end

end

# =============================================================================
# Sample Size Edge Cases
# =============================================================================

@testset "Sample Size Edge Cases" begin

    @testset "Minimum viable sample (10 per group)" begin
        Random.seed!(500)
        n = 20
        X = randn(n, 2)
        T = vcat(trues(10), falses(10))
        Y = randn(n) .+ float.(T) .* 2.0

        problem = ObservationalProblem(Y, T, X)

        solution = solve(problem, ObservationalIPW())
        @test solution.retcode in [:Success, :Warning]
        @test isfinite(solution.estimate)
    end

    @testset "Large sample (n=5000)" begin
        Random.seed!(501)
        n = 5000
        X = randn(n, 5)
        T = rand(n) .> 0.5
        Y = randn(n) .+ float.(T) .* 2.0

        problem = ObservationalProblem(Y, T, X)

        solution = solve(problem, ObservationalIPW())
        @test solution.retcode == :Success
        @test isfinite(solution.estimate)
        # Large sample should have small SE
        @test solution.se < 0.5
    end

end

# =============================================================================
# Data Type Tests
# =============================================================================

@testset "Data Type Handling" begin

    @testset "Float32 outcomes" begin
        Random.seed!(600)
        n = 100
        X = randn(Float32, n, 3)
        T = rand(n) .> 0.5
        Y = randn(Float32, n) .+ Float32.(T) .* 2.0f0

        problem = ObservationalProblem(Y, T, X)

        solution = solve(problem, ObservationalIPW())
        @test solution.retcode in [:Success, :Warning]
        @test isfinite(solution.estimate)
    end

    @testset "Integer treatment (0/1)" begin
        Random.seed!(601)
        n = 100
        X = randn(n, 3)
        T = rand([0, 1], n)  # Integers
        Y = randn(n) .+ float.(T) .* 2.0

        problem = ObservationalProblem(Y, T, X)

        solution = solve(problem, ObservationalIPW())
        @test solution.retcode in [:Success, :Warning]
        @test isfinite(solution.estimate)
    end

end

# =============================================================================
# DR-Specific Edge Cases
# =============================================================================

@testset "Doubly Robust Specific Edge Cases" begin

    @testset "DR with extreme imbalance" begin
        Random.seed!(700)
        n = 100
        X = randn(n, 3)
        T = vcat(trues(90), falses(10))
        Y = randn(n) .+ float.(T) .* 2.0

        problem = ObservationalProblem(Y, T, X)

        solution = solve(problem, DoublyRobust())
        @test solution.retcode in [:Success, :Warning]
        @test isfinite(solution.estimate)
    end

    @testset "DR outcome model fit" begin
        Random.seed!(701)
        n = 200
        X = randn(n, 3)
        T = rand(n) .> 0.5
        # Outcome strongly depends on X
        Y = 2.0 .* X[:, 1] .+ X[:, 2] .+ randn(n) .+ float.(T) .* 2.0

        problem = ObservationalProblem(Y, T, X)

        solution = solve(problem, DoublyRobust())
        @test solution.retcode == :Success

        # Outcome model R² should be reasonable
        @test solution.mu0_r2 > 0.3 || solution.mu1_r2 > 0.3
    end

end
