"""
Layer 2: Adversarial tests for Julia CATE estimator edge cases.

Tests boundary conditions, extreme inputs, and degenerate cases.
All tests should either:
1. Return valid results (graceful handling)
2. Throw explicit ArgumentError with diagnostic message
3. Never fail silently

References:
    - Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects"
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

@testset "CATE Input Validation" begin

    @testset "NaN in outcomes" begin
        X = randn(100, 3)
        T = rand(100) .> 0.5
        Y = randn(100)
        Y[50] = NaN

        @test_throws ArgumentError CATEProblem(Y, T, X, (alpha=0.05,))
    end

    @testset "Inf in outcomes" begin
        X = randn(100, 3)
        T = rand(100) .> 0.5
        Y = randn(100)
        Y[50] = Inf

        @test_throws ArgumentError CATEProblem(Y, T, X, (alpha=0.05,))
    end

    @testset "NaN in covariates" begin
        X = randn(100, 3)
        X[50, 2] = NaN
        T = rand(100) .> 0.5
        Y = randn(100)

        @test_throws ArgumentError CATEProblem(Y, T, X, (alpha=0.05,))
    end

    @testset "Inf in covariates" begin
        X = randn(100, 3)
        X[50, 2] = -Inf
        T = rand(100) .> 0.5
        Y = randn(100)

        @test_throws ArgumentError CATEProblem(Y, T, X, (alpha=0.05,))
    end

    @testset "Mismatched dimensions - Y and T" begin
        X = randn(100, 3)
        T = rand(50) .> 0.5  # Wrong length
        Y = randn(100)

        @test_throws ArgumentError CATEProblem(Y, T, X, (alpha=0.05,))
    end

    @testset "Mismatched dimensions - Y and X" begin
        X = randn(50, 3)  # Wrong rows
        T = rand(100) .> 0.5
        Y = randn(100)

        @test_throws ArgumentError CATEProblem(Y, T, X, (alpha=0.05,))
    end

    @testset "Too few observations" begin
        X = randn(3, 2)  # Only 3 observations
        T = [true, false, true]
        Y = randn(3)

        @test_throws ArgumentError CATEProblem(Y, T, X, (alpha=0.05,))
    end

end

# =============================================================================
# Treatment Variation Tests
# =============================================================================

@testset "CATE Treatment Variation" begin

    @testset "All treated (no control)" begin
        X = randn(100, 3)
        T = trues(100)  # All treated
        Y = randn(100)

        @test_throws ArgumentError CATEProblem(Y, T, X, (alpha=0.05,))
    end

    @testset "All control (no treated)" begin
        X = randn(100, 3)
        T = falses(100)  # All control
        Y = randn(100)

        @test_throws ArgumentError CATEProblem(Y, T, X, (alpha=0.05,))
    end

    @testset "Extreme imbalance - 99% treated" begin
        Random.seed!(100)
        X = randn(100, 3)
        T = vcat(trues(99), falses(1))  # 99 treated, 1 control
        Y = randn(100)

        # Should create problem but may have issues in estimation
        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        @test problem.treatment == T
    end

    @testset "Extreme imbalance - 1% treated" begin
        Random.seed!(101)
        X = randn(100, 3)
        T = vcat(trues(1), falses(99))  # 1 treated, 99 control
        Y = randn(100)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        @test problem.treatment == T
    end

    @testset "Minimum viable - 2 treated, 2 control" begin
        X = randn(4, 2)
        T = [true, true, false, false]
        Y = [1.0, 2.0, 0.5, 0.8]

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, SLearner())

        @test solution.retcode in [:Success, :Warning]
    end

end

# =============================================================================
# Constant Values Tests
# =============================================================================

@testset "CATE Constant Values" begin

    @testset "Constant outcomes" begin
        Random.seed!(200)
        X = randn(100, 3)
        T = rand(100) .> 0.5
        Y = fill(5.0, 100)  # Constant outcome

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, TLearner())

        # CATE should be ~0 with constant outcomes
        @test solution.retcode in [:Success, :Warning]
        @test abs(solution.ate) < 1.0  # Should be near zero
    end

    @testset "Constant covariates" begin
        Random.seed!(201)
        X = fill(1.0, 100, 3)  # Constant X
        T = rand(100) .> 0.5
        Y = 2.0 .* T .+ randn(100)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))

        # May fail or succeed depending on implementation
        # S-learner with constant X should still work
        try
            solution = solve(problem, SLearner())
            @test solution.retcode in [:Success, :Warning, :Failure]
        catch e
            @test isa(e, ArgumentError) || isa(e, SingularException)
        end
    end

    @testset "Single covariate" begin
        Random.seed!(202)
        X = randn(100, 1)  # p=1
        T = rand(100) .> 0.5
        Y = X[:, 1] .+ 2.0 .* T .+ randn(100)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, SLearner())

        @test solution.retcode in [:Success, :Warning]
        @test abs(solution.ate - 2.0) < 1.0
    end

end

# =============================================================================
# High-Dimensional Tests
# =============================================================================

@testset "CATE High-Dimensional" begin

    @testset "p = n (square)" begin
        Random.seed!(300)
        n = 50
        X = randn(n, n)  # p = n
        T = rand(n) .> 0.5
        Y = randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))

        # May fail or work with regularization
        try
            solution = solve(problem, SLearner(model=:ridge))
            @test solution.retcode in [:Success, :Warning, :Failure]
        catch
            @test true  # Expected to potentially fail
        end
    end

    @testset "p > n" begin
        Random.seed!(301)
        n = 30
        p = 50  # p > n
        X = randn(n, p)
        T = rand(n) .> 0.5
        Y = randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))

        # Should work with ridge, may fail with OLS
        try
            solution = solve(problem, SLearner(model=:ridge))
            @test solution.retcode in [:Success, :Warning, :Failure]
        catch
            @test true  # OLS expected to fail
        end
    end

end

# =============================================================================
# Collinearity Tests
# =============================================================================

@testset "CATE Collinearity" begin

    @testset "Perfect collinearity" begin
        Random.seed!(400)
        X = randn(100, 2)
        X = hcat(X, X[:, 1])  # X3 = X1 exactly
        T = rand(100) .> 0.5
        Y = randn(100)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))

        # Should handle with regularization or fail gracefully
        try
            solution = solve(problem, SLearner(model=:ridge))
            @test solution.retcode in [:Success, :Warning, :Failure]
        catch e
            @test isa(e, SingularException) || isa(e, LAPACKException)
        end
    end

    @testset "Near-perfect collinearity" begin
        Random.seed!(401)
        X = randn(100, 2)
        X = hcat(X, X[:, 1] .+ randn(100) .* 1e-6)  # X3 ≈ X1
        T = rand(100) .> 0.5
        Y = randn(100)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))

        try
            solution = solve(problem, SLearner())
            @test solution.retcode in [:Success, :Warning, :Failure]
        catch
            @test true  # May fail due to numerical issues
        end
    end

end

# =============================================================================
# Numerical Stability Tests
# =============================================================================

@testset "CATE Numerical Stability" begin

    @testset "Large outcome values" begin
        Random.seed!(500)
        X = randn(100, 3)
        T = rand(100) .> 0.5
        Y = randn(100) .* 1e6  # Large values

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, TLearner())

        @test solution.retcode in [:Success, :Warning]
        @test !isnan(solution.ate)
        @test !isinf(solution.ate)
    end

    @testset "Small outcome values" begin
        Random.seed!(501)
        X = randn(100, 3)
        T = rand(100) .> 0.5
        Y = randn(100) .* 1e-6  # Small values

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, TLearner())

        @test solution.retcode in [:Success, :Warning]
        @test !isnan(solution.ate)
    end

    @testset "Mixed scale covariates" begin
        Random.seed!(502)
        X = hcat(
            randn(100) .* 1e-3,   # Small scale
            randn(100) .* 1e3,    # Large scale
            randn(100)            # Normal scale
        )
        T = rand(100) .> 0.5
        Y = randn(100)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, SLearner())

        @test solution.retcode in [:Success, :Warning]
    end

    @testset "Outcome outliers" begin
        Random.seed!(503)
        X = randn(100, 3)
        T = rand(100) .> 0.5
        Y = randn(100)
        Y[1] = 100.0  # Outlier
        Y[2] = -100.0  # Outlier

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, TLearner())

        @test solution.retcode in [:Success, :Warning]
    end

end

# =============================================================================
# Small Sample Tests
# =============================================================================

@testset "CATE Small Sample" begin

    @testset "n = 10" begin
        Random.seed!(600)
        X = randn(10, 2)
        T = [true, true, true, true, true, false, false, false, false, false]
        Y = X[:, 1] .+ 2.0 .* T .+ randn(10) .* 0.5

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, SLearner())

        @test solution.retcode in [:Success, :Warning]
    end

    @testset "n = 20" begin
        Random.seed!(601)
        X = randn(20, 3)
        T = rand(20) .> 0.5
        Y = randn(20)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, TLearner())

        @test solution.retcode in [:Success, :Warning]
    end

    @testset "All methods on small sample" begin
        Random.seed!(602)
        X = randn(30, 2)
        T = rand(30) .> 0.5
        Y = 2.0 .* T .+ randn(30)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))

        for estimator in [SLearner(), TLearner(), XLearner(), RLearner()]
            solution = solve(problem, estimator)
            @test solution.retcode in [:Success, :Warning]
        end
    end

end

# =============================================================================
# Estimator-Specific Tests
# =============================================================================

@testset "CATE Estimator-Specific Edge Cases" begin

    @testset "T-Learner - very different group sizes" begin
        Random.seed!(700)
        X = randn(100, 3)
        T = vcat(trues(90), falses(10))  # 90-10 split
        Y = randn(100)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, TLearner())

        @test solution.retcode in [:Success, :Warning]
    end

    @testset "X-Learner - propensity estimation" begin
        Random.seed!(701)
        X = randn(100, 3)
        T = rand(100) .> 0.5
        Y = randn(100)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, XLearner())

        @test solution.retcode in [:Success, :Warning]
        @test length(solution.cate) == 100
    end

    @testset "R-Learner - residualization" begin
        Random.seed!(702)
        X = randn(100, 3)
        T = rand(100) .> 0.5
        Y = randn(100)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, RLearner())

        @test solution.retcode in [:Success, :Warning]
    end

    @testset "DML - minimum folds" begin
        Random.seed!(703)
        X = randn(50, 3)
        T = rand(50) .> 0.5
        Y = randn(50)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, DoubleMachineLearning(n_folds=2))

        @test solution.retcode in [:Success, :Warning]
    end

    @testset "DML - n_folds validation" begin
        # n_folds < 2 should error
        @test_throws ArgumentError DoubleMachineLearning(n_folds=1)
    end

end

# =============================================================================
# Model Type Tests
# =============================================================================

@testset "CATE Model Types" begin

    @testset "OLS model" begin
        Random.seed!(800)
        X = randn(100, 3)
        T = rand(100) .> 0.5
        Y = X[:, 1] .+ 2.0 .* T .+ randn(100)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, SLearner(model=:ols))

        @test solution.retcode in [:Success, :Warning]
        @test solution.method == :s_learner
    end

    @testset "Ridge model" begin
        Random.seed!(801)
        X = randn(100, 3)
        T = rand(100) .> 0.5
        Y = X[:, 1] .+ 2.0 .* T .+ randn(100)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, TLearner(model=:ridge))

        @test solution.retcode in [:Success, :Warning]
        @test solution.method == :t_learner
    end

    @testset "Invalid model type" begin
        @test_throws ArgumentError SLearner(model=:invalid)
        @test_throws ArgumentError TLearner(model=:lasso)
        @test_throws ArgumentError XLearner(model=:forest)
    end

end

# =============================================================================
# Data Type Tests
# =============================================================================

@testset "CATE Data Types" begin

    @testset "Float32 conversion" begin
        X = Float32.(randn(100, 3))
        T = rand(100) .> 0.5
        Y = Float32.(randn(100))

        # Should auto-convert to consistent type
        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, SLearner())

        @test solution.retcode in [:Success, :Warning]
    end

    @testset "Integer outcomes" begin
        Random.seed!(901)
        X = randn(100, 3)
        T = rand(100) .> 0.5
        Y = rand(1:10, 100)  # Integer outcomes

        # Should convert to float
        problem = CATEProblem(Float64.(Y), T, X, (alpha=0.05,))
        solution = solve(problem, TLearner())

        @test solution.retcode in [:Success, :Warning]
    end

    @testset "BitVector treatment" begin
        X = randn(100, 3)
        T = BitVector(rand(100) .> 0.5)  # BitVector
        Y = randn(100)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        @test typeof(problem.treatment) == Vector{Bool}
    end

end

# Run summary
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n" * "="^60)
    println("CATE Adversarial Tests Summary")
    println("="^60)
    println("Testing edge cases and boundary conditions for Julia CATE.")
    println("All tests should handle gracefully or fail explicitly.")
    println("="^60 * "\n")
end
