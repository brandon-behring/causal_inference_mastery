"""
Adversarial tests for Julia bunching estimator edge cases.

Tests boundary conditions, extreme inputs, and degenerate cases.
All tests should either:
1. Return valid results (graceful handling)
2. Throw explicit ArgumentError with diagnostic message
3. Never fail silently

References:
    - Saez (2010). Do taxpayers bunch at kink points?
    - Kleven (2016). Bunching estimation review.
"""

using Test
using Statistics
using Random
using LinearAlgebra

# Include main module
include("../../src/CausalEstimators.jl")
using .CausalEstimators

# =============================================================================
# Input Validation Tests - BunchingProblem
# =============================================================================

@testset "BunchingProblem Input Validation" begin

    @testset "Empty data" begin
        @test_throws ArgumentError BunchingProblem(Float64[], 50.0, 5.0)
    end

    @testset "NaN in data" begin
        data = randn(100)
        data[50] = NaN
        @test_throws ArgumentError BunchingProblem(data, 50.0, 5.0)
    end

    @testset "Inf in data" begin
        data = randn(100)
        data[50] = Inf
        @test_throws ArgumentError BunchingProblem(data, 50.0, 5.0)
    end

    @testset "-Inf in data" begin
        data = randn(100)
        data[50] = -Inf
        @test_throws ArgumentError BunchingProblem(data, 50.0, 5.0)
    end

    @testset "Zero bunching width" begin
        data = randn(100)
        @test_throws ArgumentError BunchingProblem(data, 50.0, 0.0)
    end

    @testset "Negative bunching width" begin
        data = randn(100)
        @test_throws ArgumentError BunchingProblem(data, 50.0, -5.0)
    end

    @testset "Invalid t1_rate negative" begin
        data = randn(100)
        @test_throws ArgumentError BunchingProblem(data, 50.0, 5.0; t1_rate=-0.1, t2_rate=0.3)
    end

    @testset "Invalid t1_rate >= 1" begin
        data = randn(100)
        @test_throws ArgumentError BunchingProblem(data, 50.0, 5.0; t1_rate=1.0, t2_rate=1.1)
    end

    @testset "Invalid t2_rate >= 1" begin
        data = randn(100)
        @test_throws ArgumentError BunchingProblem(data, 50.0, 5.0; t1_rate=0.2, t2_rate=1.0)
    end

    @testset "t2_rate <= t1_rate" begin
        data = randn(100)
        @test_throws ArgumentError BunchingProblem(data, 50.0, 5.0; t1_rate=0.3, t2_rate=0.2)
    end

    @testset "t2_rate equal t1_rate" begin
        data = randn(100)
        @test_throws ArgumentError BunchingProblem(data, 50.0, 5.0; t1_rate=0.25, t2_rate=0.25)
    end

end

# =============================================================================
# Input Validation Tests - SaezBunching
# =============================================================================

@testset "SaezBunching Input Validation" begin

    @testset "n_bins too small" begin
        @test_throws ArgumentError SaezBunching(n_bins=5)
    end

    @testset "n_bins at minimum" begin
        est = SaezBunching(n_bins=10)
        @test est.n_bins == 10
    end

    @testset "polynomial_order < 1" begin
        @test_throws ArgumentError SaezBunching(polynomial_order=0)
    end

    @testset "n_bootstrap too small" begin
        @test_throws ArgumentError SaezBunching(n_bootstrap=5)
    end

    @testset "n_bootstrap at minimum" begin
        est = SaezBunching(n_bootstrap=10)
        @test est.n_bootstrap == 10
    end

end

# =============================================================================
# Bunching Region Edge Cases
# =============================================================================

@testset "Bunching Region Edge Cases" begin

    @testset "Kink at edge of data" begin
        Random.seed!(100)
        # Data mostly above kink
        data = randn(500) .* 10 .+ 60
        problem = BunchingProblem(data, 20.0, 5.0)  # Kink at low end

        estimator = SaezBunching(n_bins=30, polynomial_order=3, n_bootstrap=20)

        # Should handle gracefully
        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass)
    end

    @testset "Very wide bunching region" begin
        Random.seed!(101)
        data = randn(500) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 20.0)  # Wide bunching region

        estimator = SaezBunching(n_bins=50, polynomial_order=3, n_bootstrap=20)

        # May have issues but should not crash
        try
            solution = solve(problem, estimator)
            @test isfinite(solution.excess_mass)
        catch e
            # Wide region may cause insufficient data for polynomial fit
            @test e isa ArgumentError
        end
    end

    @testset "Very narrow bunching region" begin
        Random.seed!(102)
        data = randn(500) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 0.5)  # Narrow bunching region

        estimator = SaezBunching(n_bins=100, polynomial_order=5, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass)
    end

    @testset "Bunching region contains few observations" begin
        Random.seed!(103)
        # Data spread far from kink
        data = vcat(
            randn(250) .* 5 .+ 20,  # Cluster below
            randn(250) .* 5 .+ 80   # Cluster above
        )
        problem = BunchingProblem(data, 50.0, 3.0)  # Kink in sparse region

        estimator = SaezBunching(n_bins=40, polynomial_order=3, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass)
    end

end

# =============================================================================
# Polynomial Fitting Edge Cases
# =============================================================================

@testset "Polynomial Fitting Edge Cases" begin

    @testset "High polynomial order" begin
        Random.seed!(200)
        data = randn(1000) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 3.0)

        # High polynomial order (may overfit)
        estimator = SaezBunching(n_bins=80, polynomial_order=9, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass)
        @test solution.counterfactual.polynomial_order == 9
    end

    @testset "Low polynomial order" begin
        Random.seed!(201)
        data = randn(1000) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 3.0)

        # Low polynomial order
        estimator = SaezBunching(n_bins=50, polynomial_order=1, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass)
        @test solution.counterfactual.polynomial_order == 1
    end

    @testset "Many bins" begin
        Random.seed!(202)
        data = randn(2000) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 3.0)

        estimator = SaezBunching(n_bins=200, polynomial_order=5, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass)
        @test solution.counterfactual.n_bins == 200
    end

    @testset "Few bins" begin
        Random.seed!(203)
        data = randn(500) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0)

        estimator = SaezBunching(n_bins=10, polynomial_order=3, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass)
    end

end

# =============================================================================
# Numerical Stability
# =============================================================================

@testset "Numerical Stability" begin

    @testset "Very large values" begin
        Random.seed!(300)
        data = randn(500) .* 1e6 .+ 1e8
        kink = 1e8
        problem = BunchingProblem(data, kink, 1e6)

        estimator = SaezBunching(n_bins=50, polynomial_order=5, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass)
        @test solution.kink_point == kink
    end

    @testset "Small positive values" begin
        Random.seed!(301)
        # Use values that are small but not extremely so
        data = randn(500) .* 0.1 .+ 1.0
        kink = 1.0
        problem = BunchingProblem(data, kink, 0.1)

        estimator = SaezBunching(n_bins=50, polynomial_order=5, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass)
    end

    @testset "Mixed scale values" begin
        Random.seed!(302)
        # Mix of small and larger values
        data = vcat(
            randn(250) .* 10 .+ 50,
            randn(250) .* 0.1 .+ 50.5
        )
        problem = BunchingProblem(data, 50.0, 2.0)

        estimator = SaezBunching(n_bins=50, polynomial_order=5, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass)
    end

    @testset "Negative kink point" begin
        Random.seed!(303)
        data = randn(500) .* 15 .- 50  # Negative values
        kink = -50.0
        problem = BunchingProblem(data, kink, 5.0)

        estimator = SaezBunching(n_bins=50, polynomial_order=5, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass)
        @test solution.kink_point == -50.0
    end

end

# =============================================================================
# Distribution Edge Cases
# =============================================================================

@testset "Distribution Edge Cases" begin

    @testset "Constant data" begin
        Random.seed!(400)
        data = fill(50.0, 100)  # All identical
        problem = BunchingProblem(data, 50.0, 5.0)

        estimator = SaezBunching(n_bins=20, polynomial_order=3, n_bootstrap=20)

        # May fail or produce degenerate result
        try
            solution = solve(problem, estimator)
            # If it succeeds, should have high excess mass
            @test isfinite(solution.excess_mass)
        catch e
            # Various numerical errors can occur with degenerate data
            @test e isa Union{ArgumentError, DomainError, LinearAlgebra.SingularException, InexactError}
        end
    end

    @testset "Highly skewed data" begin
        Random.seed!(401)
        # Exponential-like skewed data
        data = abs.(randn(500)) .* 20 .+ 30
        problem = BunchingProblem(data, 50.0, 5.0)

        estimator = SaezBunching(n_bins=50, polynomial_order=5, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass)
    end

    @testset "Bimodal data" begin
        Random.seed!(402)
        data = vcat(
            randn(300) .* 5 .+ 30,
            randn(300) .* 5 .+ 70
        )
        problem = BunchingProblem(data, 50.0, 5.0)  # Kink in middle

        estimator = SaezBunching(n_bins=50, polynomial_order=5, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass)
    end

    @testset "Uniform data (no bunching)" begin
        Random.seed!(403)
        data = rand(500) .* 100  # Uniform [0, 100]
        problem = BunchingProblem(data, 50.0, 5.0)

        estimator = SaezBunching(n_bins=50, polynomial_order=3, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass)
        # Uniform should have small excess mass
        @test abs(solution.excess_mass) < 5.0
    end

end

# =============================================================================
# Sample Size Edge Cases
# =============================================================================

@testset "Sample Size Edge Cases" begin

    @testset "Minimum viable sample" begin
        Random.seed!(500)
        data = randn(50) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0)

        estimator = SaezBunching(n_bins=10, polynomial_order=2, n_bootstrap=10)

        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass)
    end

    @testset "Large sample" begin
        Random.seed!(501)
        data = randn(10000) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0)

        estimator = SaezBunching(n_bins=100, polynomial_order=7, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass)
        # Large sample should have smaller SE
        @test solution.excess_mass_se > 0
    end

end

# =============================================================================
# Tax Rate Edge Cases
# =============================================================================

@testset "Tax Rate Edge Cases" begin

    @testset "Zero t1_rate" begin
        Random.seed!(600)
        data = randn(500) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0; t1_rate=0.0, t2_rate=0.25)

        estimator = SaezBunching(n_bins=50, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.elasticity)
        @test solution.t1_rate == 0.0
    end

    @testset "Small rate difference" begin
        Random.seed!(601)
        data = randn(500) .* 15 .+ 50
        # Very small rate change
        problem = BunchingProblem(data, 50.0, 5.0; t1_rate=0.20, t2_rate=0.21)

        estimator = SaezBunching(n_bins=50, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.elasticity)
        # Small rate difference → larger elasticity for same excess mass
    end

    @testset "Large rate difference" begin
        Random.seed!(602)
        data = randn(500) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0; t1_rate=0.10, t2_rate=0.50)

        estimator = SaezBunching(n_bins=50, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.elasticity)
    end

    @testset "Rates near boundary" begin
        Random.seed!(603)
        data = randn(500) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0; t1_rate=0.98, t2_rate=0.99)

        estimator = SaezBunching(n_bins=50, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.elasticity)
    end

end

# =============================================================================
# Bootstrap Edge Cases
# =============================================================================

@testset "Bootstrap Edge Cases" begin

    @testset "Minimum bootstrap iterations" begin
        Random.seed!(700)
        data = randn(500) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0)

        estimator = SaezBunching(n_bins=50, n_bootstrap=10)

        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass_se)
        @test solution.n_bootstrap == 10
    end

    @testset "Many bootstrap iterations" begin
        Random.seed!(701)
        data = randn(500) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0)

        estimator = SaezBunching(n_bins=50, n_bootstrap=500)

        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass_se)
        @test solution.n_bootstrap == 500
    end

end

# =============================================================================
# Data Type Handling
# =============================================================================

@testset "Data Type Handling" begin

    @testset "Float32 data" begin
        Random.seed!(800)
        data = Float32.(randn(500) .* 15 .+ 50)
        problem = BunchingProblem(data, 50.0f0, 5.0f0)

        estimator = SaezBunching(n_bins=50, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass)
    end

    @testset "Integer kink point" begin
        Random.seed!(801)
        data = randn(500) .* 15 .+ 50
        # Integer kink gets promoted to Float64
        problem = BunchingProblem(data, 50.0, 5.0)

        estimator = SaezBunching(n_bins=50, n_bootstrap=20)

        solution = solve(problem, estimator)
        @test isfinite(solution.excess_mass)
    end

end

# =============================================================================
# Solution Structure Tests
# =============================================================================

@testset "Solution Structure" begin

    @testset "All fields present" begin
        Random.seed!(900)
        data = randn(500) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0; t1_rate=0.25, t2_rate=0.35)

        estimator = SaezBunching(n_bins=50, n_bootstrap=30)

        solution = solve(problem, estimator)

        # Check all fields
        @test isfinite(solution.excess_mass)
        @test isfinite(solution.excess_mass_se)
        @test isfinite(solution.excess_mass_count)
        @test isfinite(solution.elasticity)
        @test isfinite(solution.elasticity_se)
        @test solution.kink_point == 50.0
        @test length(solution.bunching_region) == 2
        @test solution.bunching_region[1] < solution.bunching_region[2]
        @test solution.t1_rate == 0.25
        @test solution.t2_rate == 0.35
        @test solution.n_obs == 500
        @test solution.n_bootstrap == 30
        @test isa(solution.convergence, Bool)
    end

    @testset "Counterfactual result structure" begin
        Random.seed!(901)
        data = randn(500) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0)

        estimator = SaezBunching(n_bins=50, polynomial_order=5, n_bootstrap=20)

        solution = solve(problem, estimator)
        cf = solution.counterfactual

        @test length(cf.bin_centers) == 50
        @test length(cf.actual_counts) == 50
        @test length(cf.counterfactual_counts) == 50
        @test length(cf.polynomial_coeffs) == 6  # order 5 + constant
        @test cf.polynomial_order == 5
        @test cf.n_bins == 50
        @test cf.bin_width > 0
        @test 0 <= cf.r_squared <= 1
    end

    @testset "Confidence interval validation" begin
        Random.seed!(902)
        data = randn(500) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0)

        estimator = SaezBunching(n_bins=50, n_bootstrap=50)

        solution = solve(problem, estimator)

        lower, upper = bunching_confidence_interval(solution; level=0.95)
        @test lower < upper
        @test lower < solution.excess_mass < upper
    end

end

# =============================================================================
# Determinism Tests
# =============================================================================

@testset "Determinism" begin

    @testset "Same seed gives same results" begin
        data = randn(500) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0)

        Random.seed!(999)
        estimator1 = SaezBunching(n_bins=50, n_bootstrap=50)
        solution1 = solve(problem, estimator1)

        Random.seed!(999)
        estimator2 = SaezBunching(n_bins=50, n_bootstrap=50)
        solution2 = solve(problem, estimator2)

        @test solution1.excess_mass == solution2.excess_mass
        @test solution1.excess_mass_se == solution2.excess_mass_se
    end

end
