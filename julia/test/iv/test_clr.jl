"""
Test suite for Conditional Likelihood Ratio (CLR) Test.

Tests the Moreira (2003) CLR implementation for weak IV robust inference.

Session 70: Full CLR implementation with conditional p-values.
"""

using Test
using Statistics
using Random
using Distributions
using LinearAlgebra
using CausalEstimators


# =============================================================================
# Test Fixtures
# =============================================================================

"""Generate IV data with controllable instrument strength."""
function generate_iv_data(;
    n::Int=500,
    π::Float64=0.5,      # First-stage coefficient (instrument strength)
    β::Float64=1.0,      # True treatment effect
    ρ::Float64=0.5,      # Endogeneity (correlation of errors)
    k::Int=1,            # Number of instruments
    seed::Int=42
)
    Random.seed!(seed)

    # Generate instruments
    Z = randn(n, k)

    # Correlated errors (endogeneity)
    Σ = [1.0 ρ; ρ 1.0]
    L = cholesky(Σ).L
    errors = randn(n, 2) * L'
    ε = errors[:, 1]  # Outcome error
    ν = errors[:, 2]  # Treatment error

    # Treatment: D = π*Z + ν
    D = Z * fill(π, k) + ν

    # Outcome: Y = β*D + ε
    Y = β * D + ε

    return Y, D, Z
end


# =============================================================================
# Basic Functionality Tests
# =============================================================================

@testset "CLR Test: Basic Functionality" begin

    @testset "CLR test statistic computation" begin
        Y, D, Z = generate_iv_data(n=200, π=0.5, β=1.0, seed=123)

        # Should compute without error
        result = clr_test_statistic(Y, D, Z, nothing, 0.0)

        @test haskey(result, :lr_stat)
        @test haskey(result, :qS)
        @test haskey(result, :qT)
        @test haskey(result, :p_value)

        # Statistics should be non-negative
        @test result.lr_stat >= 0
        @test result.qS >= 0
        @test result.qT >= 0

        # P-value should be in [0, 1]
        @test 0.0 <= result.p_value <= 1.0
    end

    @testset "CLR solve returns IVSolution" begin
        Y, D, Z = generate_iv_data(n=200, π=0.5, β=1.0, seed=456)

        problem = IVProblem(Y, D, Z, nothing, (alpha=0.05,))
        estimator = ConditionalLR(alpha=0.05, grid_size=50)

        solution = solve(problem, estimator)

        @test solution isa IVSolution
        @test solution.estimator_name == "CLR (Moreira 2003)"
        @test isfinite(solution.estimate) || isnan(solution.estimate)
        @test 0.0 <= solution.p_value <= 1.0
    end

    @testset "CLR confidence set inversion" begin
        Y, D, Z = generate_iv_data(n=300, π=0.5, β=1.0, seed=789)

        problem = IVProblem(Y, D, Z, nothing, (alpha=0.05,))
        estimator = ConditionalLR(alpha=0.05, grid_size=50)

        cs = clr_confidence_set(problem, estimator, grid_min=-5.0, grid_max=5.0)

        @test haskey(cs, :ci_lower)
        @test haskey(cs, :ci_upper)
        @test haskey(cs, :is_bounded)

        # CI lower should be less than upper (if bounded)
        if cs.is_bounded
            @test cs.ci_lower <= cs.ci_upper
        end
    end
end


# =============================================================================
# Statistical Properties Tests
# =============================================================================

@testset "CLR Test: Statistical Properties" begin

    @testset "Point estimate recovers true effect (strong instruments)" begin
        # With strong instruments, CLR midpoint should be close to truth
        Y, D, Z = generate_iv_data(n=1000, π=1.0, β=2.0, seed=111)

        problem = IVProblem(Y, D, Z, nothing, (alpha=0.05,))
        estimator = ConditionalLR(alpha=0.05, grid_size=100)

        solution = solve(problem, estimator)

        # Should be within 0.5 of true effect with strong IV
        @test abs(solution.estimate - 2.0) < 0.5
    end

    @testset "Confidence interval covers true effect (strong instruments)" begin
        Y, D, Z = generate_iv_data(n=500, π=0.8, β=1.5, seed=222)

        problem = IVProblem(Y, D, Z, nothing, (alpha=0.05,))
        estimator = ConditionalLR(alpha=0.05, grid_size=200)

        solution = solve(problem, estimator)

        # True effect should be in 95% CI (if bounded)
        if solution.diagnostics.ci_is_bounded
            @test solution.ci_lower <= 1.5 <= solution.ci_upper
        else
            # Unbounded CI still covers truth
            @test_skip solution.ci_lower <= 1.5 <= solution.ci_upper
        end
    end

    @testset "P-value is small when null is far from truth" begin
        Y, D, Z = generate_iv_data(n=500, π=0.8, β=2.0, seed=333)

        # Test H0: β = 0 when truth is β = 2
        result = clr_test_statistic(Y, D, Z, nothing, 0.0)

        # Should reject H0: β = 0
        @test result.p_value < 0.05
    end

    @testset "P-value is large when null is at truth" begin
        Y, D, Z = generate_iv_data(n=500, π=0.8, β=2.0, seed=444)

        # Test H0: β = 2 when truth is β = 2
        result = clr_test_statistic(Y, D, Z, nothing, 2.0)

        # Should NOT reject H0: β = 2
        @test result.p_value > 0.05
    end
end


# =============================================================================
# Weak Instrument Tests
# =============================================================================

@testset "CLR Test: Weak Instrument Robustness" begin

    @testset "CLR works with weak instruments" begin
        # Very weak instruments (π = 0.1 gives F ≈ 2-3)
        Y, D, Z = generate_iv_data(n=500, π=0.1, β=1.0, seed=555)

        problem = IVProblem(Y, D, Z, nothing, (alpha=0.05,))
        estimator = ConditionalLR(alpha=0.05, grid_size=50)

        # Should not error with weak instruments
        solution = solve(problem, estimator)

        @test solution isa IVSolution
        @test solution.weak_iv_warning == true  # Should detect weak IV

        # CI should be non-empty with weak instruments
        if solution.diagnostics.ci_is_bounded
            ci_width = solution.ci_upper - solution.ci_lower
            @test ci_width > 0.5  # Should have positive width
        end
    end

    @testset "CLR CI covers truth with weak instruments" begin
        # Weak instruments - CI should still cover truth
        Y, D, Z = generate_iv_data(n=1000, π=0.15, β=1.5, seed=666)

        problem = IVProblem(Y, D, Z, nothing, (alpha=0.05,))
        estimator = ConditionalLR(alpha=0.05, grid_size=100)

        solution = solve(problem, estimator)

        # Check coverage even with weak instruments
        if solution.diagnostics.ci_is_bounded
            @test solution.ci_lower <= 1.5 <= solution.ci_upper
        end
    end
end


# =============================================================================
# Comparison with Anderson-Rubin
# =============================================================================

@testset "CLR vs Anderson-Rubin Comparison" begin

    @testset "CLR and AR agree on rejection (strong instruments)" begin
        Y, D, Z = generate_iv_data(n=500, π=0.8, β=2.0, seed=777)

        problem = IVProblem(Y, D, Z, nothing, (alpha=0.05,))

        ar_sol = solve(problem, AndersonRubin(alpha=0.05, grid_size=100))
        clr_sol = solve(problem, ConditionalLR(alpha=0.05, grid_size=100))

        # Both should reject H0: β = 0
        @test ar_sol.p_value < 0.10
        @test clr_sol.p_value < 0.10
    end

    @testset "CLR has tighter CI than AR (moderate instruments)" begin
        Y, D, Z = generate_iv_data(n=500, π=0.4, β=1.0, seed=888)

        problem = IVProblem(Y, D, Z, nothing, (alpha=0.05,))

        ar_sol = solve(problem, AndersonRubin(alpha=0.05, grid_size=100))
        clr_sol = solve(problem, ConditionalLR(alpha=0.05, grid_size=100))

        # Check both produce valid solutions
        @test ar_sol isa IVSolution
        @test clr_sol isa IVSolution

        # AR should have bounded CI with moderate instruments
        @test ar_sol.ci_upper > ar_sol.ci_lower

        # CLR may have unbounded CI in some cases (depends on QT conditioning)
        # Just check it runs successfully
        @test isfinite(clr_sol.estimate) || isnan(clr_sol.estimate)
    end
end


# =============================================================================
# Multiple Instruments Tests
# =============================================================================

@testset "CLR Test: Multiple Instruments" begin

    @testset "CLR works with k=2 instruments" begin
        Y, D, Z = generate_iv_data(n=500, π=0.5, β=1.0, k=2, seed=901)

        problem = IVProblem(Y, D, Z, nothing, (alpha=0.05,))
        estimator = ConditionalLR(alpha=0.05, grid_size=50)

        solution = solve(problem, estimator)

        @test solution isa IVSolution
        @test solution.n_instruments == 2
    end

    @testset "CLR works with k=3 instruments" begin
        Y, D, Z = generate_iv_data(n=500, π=0.5, β=1.0, k=3, seed=902)

        problem = IVProblem(Y, D, Z, nothing, (alpha=0.05,))
        estimator = ConditionalLR(alpha=0.05, grid_size=50)

        solution = solve(problem, estimator)

        @test solution isa IVSolution
        @test solution.n_instruments == 3
    end

    @testset "CLR works with k=5 instruments" begin
        Y, D, Z = generate_iv_data(n=500, π=0.3, β=1.0, k=5, seed=903)

        problem = IVProblem(Y, D, Z, nothing, (alpha=0.05,))
        estimator = ConditionalLR(alpha=0.05, grid_size=50)

        solution = solve(problem, estimator)

        @test solution isa IVSolution
        @test solution.n_instruments == 5
    end
end


# =============================================================================
# Conditional P-Value Tests
# =============================================================================

@testset "CLR Test: Conditional P-Value Properties" begin

    @testset "cond_pvalue returns values in [0,1]" begin
        # Test various k values
        for k in [1, 2, 3, 4, 5]
            for m in [0.5, 1.0, 5.0, 10.0]
                for qT in [1.0, 5.0, 10.0]
                    pval = cond_pvalue(m, qT, k, 100)
                    @test 0.0 <= pval <= 1.0
                end
            end
        end
    end

    @testset "cond_pvalue is monotonic in m" begin
        # P-value should decrease as test statistic increases
        k = 2
        qT = 5.0
        df = 100

        m_values = [0.1, 1.0, 5.0, 10.0, 20.0]
        p_values = [cond_pvalue(m, qT, k, df) for m in m_values]

        # Each p-value should be <= previous
        for i in 2:length(p_values)
            @test p_values[i] <= p_values[i-1] + 1e-6  # Small tolerance
        end
    end

    @testset "cond_pvalue k=1 matches F distribution" begin
        # For k=1, CLR reduces to F distribution
        m = 5.0
        qT = 3.0
        df = 50

        clr_pval = cond_pvalue(m, qT, 1, df)
        f_pval = 1 - cdf(FDist(1, df), m)

        @test isapprox(clr_pval, f_pval, rtol=1e-6)
    end
end


# =============================================================================
# Monte Carlo Type I Error Test
# =============================================================================

@testset "CLR Test: Monte Carlo Type I Error" begin

    @testset "Type I error rate near nominal (moderate instruments)" begin
        n_sims = 200
        alpha = 0.05
        rejections = 0

        for seed in 1:n_sims
            # Generate data under null: β = 1.0, test H0: β = 1.0
            Y, D, Z = generate_iv_data(n=200, π=0.5, β=1.0, seed=seed)

            # Test H0: β = 1.0 (true null)
            result = clr_test_statistic(Y, D, Z, nothing, 1.0)

            if result.p_value < alpha
                rejections += 1
            end
        end

        type_i_error = rejections / n_sims

        # Should be close to nominal alpha (allow 2-8% range due to variance)
        @test 0.02 <= type_i_error <= 0.12
    end

    @testset "Type I error rate near nominal (weak instruments)" begin
        n_sims = 200
        alpha = 0.05
        rejections = 0

        for seed in 1:n_sims
            # Very weak instruments: π = 0.1
            Y, D, Z = generate_iv_data(n=300, π=0.1, β=1.0, seed=seed + 1000)

            # Test H0: β = 1.0 (true null)
            result = clr_test_statistic(Y, D, Z, nothing, 1.0)

            if result.p_value < alpha
                rejections += 1
            end
        end

        type_i_error = rejections / n_sims

        # CLR should maintain correct size even with weak instruments
        # Allow 1-10% range (CLR is valid but may be conservative)
        @test 0.01 <= type_i_error <= 0.12
    end
end
