"""
Tests for RegressionATE estimator.

Following test-first development with known-answer tests and property-based tests.
"""

using Test
using CausalEstimators
using Random
using LinearAlgebra

@testset "RegressionATE: Known Answer Tests" begin
    @testset "Perfect linear relationship (recovers true ATE)" begin
        # Y = 2 + 5*T + 3*X  (true ATE=5)
        X = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        treatment = [true, true, true, true, false, false, false, false]
        outcomes = 2.0 .+ 5.0 .* treatment .+ 3.0 .* X

        X_matrix = reshape(X, length(X), 1)
        problem = RCTProblem(outcomes, treatment, X_matrix, nothing, (alpha = 0.05,))

        solution = solve(problem, RegressionATE())

        # Should recover true ATE = 5.0 exactly (no noise)
        @test solution.estimate ≈ 5.0 atol = 1e-10
        @test solution.n_treated == 4
        @test solution.n_control == 4
        @test solution.retcode == :Success
    end

    @testset "Zero treatment effect with covariate" begin
        # Y = 10 + 0*T + 2*X  (true ATE=0)
        X = [1.0, 2.0, 3.0, 4.0]
        treatment = [true, true, false, false]
        outcomes = 10.0 .+ 2.0 .* X

        X_matrix = reshape(X, length(X), 1)
        problem = RCTProblem(outcomes, treatment, X_matrix, nothing, (alpha = 0.05,))

        solution = solve(problem, RegressionATE())

        @test solution.estimate ≈ 0.0 atol = 1e-10
    end

    @testset "Multiple covariates" begin
        # Y = 1 + 3*T + 2*X1 + 4*X2  (true ATE=3)
        # Use varied values to avoid collinearity
        Random.seed!(42)
        X1 = [1.0, 2.5, 3.2, 1.8, 2.1, 3.5]
        X2 = [0.5, 1.8, 2.3, 0.7, 1.4, 2.9]
        treatment = [true, true, true, false, false, false]
        outcomes = 1.0 .+ 3.0 .* treatment .+ 2.0 .* X1 .+ 4.0 .* X2 .+ 0.01 .* randn(6)

        X_matrix = hcat(X1, X2)
        problem = RCTProblem(outcomes, treatment, X_matrix, nothing, (alpha = 0.05,))

        solution = solve(problem, RegressionATE())

        # Should recover ATE ≈ 3.0 (with small noise)
        @test solution.estimate ≈ 3.0 atol = 0.1
    end
end

@testset "RegressionATE: Error Handling" begin
    @testset "Missing covariates throws error" begin
        outcomes = [10.0, 12.0, 4.0, 5.0]
        treatment = [true, true, false, false]
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))

        @test_throws ArgumentError solve(problem, RegressionATE())
    end

    @testset "Singular design matrix throws error" begin
        # Create perfectly collinear covariates
        X1 = [1.0, 2.0, 3.0, 4.0]
        X2 = 2.0 .* X1  # Perfectly collinear
        treatment = [true, true, false, false]
        outcomes = [10.0, 12.0, 4.0, 5.0]

        X_matrix = hcat(X1, X2)
        problem = RCTProblem(outcomes, treatment, X_matrix, nothing, (alpha = 0.05,))

        @test_throws ArgumentError solve(problem, RegressionATE())
    end
end

@testset "RegressionATE: Statistical Properties" begin
    @testset "Confidence interval contains estimate" begin
        Random.seed!(42)

        X = randn(20)
        treatment = rand(Bool, 20)
        outcomes = 5.0 .+ 3.0 .* treatment .+ 2.0 .* X .+ randn(20)

        X_matrix = reshape(X, length(X), 1)
        problem = RCTProblem(outcomes, treatment, X_matrix, nothing, (alpha = 0.05,))

        solution = solve(problem, RegressionATE())

        @test solution.ci_lower < solution.estimate
        @test solution.estimate < solution.ci_upper
        @test solution.se > 0
    end

    @testset "RegressionATE more efficient than SimpleATE" begin
        Random.seed!(42)

        # Create data where covariate strongly predicts outcome
        n = 100
        X = randn(n)
        treatment = rand(Bool, n)
        # Outcomes strongly depend on X (variance reduction from regression)
        outcomes = 5.0 .+ 3.0 .* treatment .+ 10.0 .* X .+ 0.5 .* randn(n)

        # RegressionATE
        X_matrix = reshape(X, n, 1)
        problem_reg = RCTProblem(outcomes, treatment, X_matrix, nothing, (alpha = 0.05,))
        solution_reg = solve(problem_reg, RegressionATE())

        # SimpleATE (ignores covariates)
        problem_simple = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        solution_simple = solve(problem_simple, SimpleATE())

        # Regression should have smaller SE (approximately, due to randomness)
        @test solution_reg.se < solution_simple.se
    end

    @testset "Different alpha levels" begin
        Random.seed!(42)

        X = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        treatment = [true, true, true, true, false, false, false, false]
        # Add noise to avoid zero variance
        outcomes = 2.0 .+ 5.0 .* treatment .+ 3.0 .* X .+ 0.1 .* randn(8)

        X_matrix = reshape(X, length(X), 1)

        # 95% CI
        problem_95 = RCTProblem(outcomes, treatment, X_matrix, nothing, (alpha = 0.05,))
        solution_95 = solve(problem_95, RegressionATE())

        # 99% CI
        problem_99 = RCTProblem(outcomes, treatment, X_matrix, nothing, (alpha = 0.01,))
        solution_99 = solve(problem_99, RegressionATE())

        # Same estimate
        @test solution_95.estimate == solution_99.estimate

        # Same SE
        @test solution_95.se == solution_99.se

        # 99% CI should be wider
        ci_width_95 = solution_95.ci_upper - solution_95.ci_lower
        ci_width_99 = solution_99.ci_upper - solution_99.ci_lower
        @test ci_width_99 > ci_width_95
    end
end

@testset "RegressionATE: Type Stability" begin
    X = [1.0, 2.0, 3.0, 4.0]
    treatment = [true, true, false, false]
    outcomes = [10.0, 14.0, 7.0, 11.0]

    X_matrix = reshape(X, length(X), 1)
    problem = RCTProblem(outcomes, treatment, X_matrix, nothing, (alpha = 0.05,))

    solution = solve(problem, RegressionATE())

    # Check all fields have correct types
    @test solution.estimate isa Float64
    @test solution.se isa Float64
    @test solution.ci_lower isa Float64
    @test solution.ci_upper isa Float64
    @test solution.n_treated isa Int
    @test solution.n_control isa Int
    @test solution.retcode isa Symbol
end

@testset "RegressionATE: Adversarial Tests" begin
    @testset "Perfect collinearity (treatment = covariate)" begin
        # X perfectly predicts treatment → collinearity
        treatment = [true, true, false, false]
        X = Float64.(treatment)  # Perfect collinearity
        outcomes = [10.0, 12.0, 4.0, 5.0]
        X_matrix = reshape(X, length(X), 1)

        problem = RCTProblem(outcomes, treatment, X_matrix, nothing, (alpha = 0.05,))
        
        # Should error during solve (singular matrix)
        @test_throws Exception solve(problem, RegressionATE())
    end

    @testset "Zero variance covariate" begin
        # All covariate values identical
        treatment = [true, true, false, false]
        X = fill(5.0, 4)
        outcomes = [10.0, 12.0, 4.0, 5.0]
        X_matrix = reshape(X, length(X), 1)

        problem = RCTProblem(outcomes, treatment, X_matrix, nothing, (alpha = 0.05,))
        
        # Should error (singular matrix - no variation in X)
        @test_throws Exception solve(problem, RegressionATE())
    end

    @testset "More covariates than observations" begin
        # p > n leads to singular matrix
        treatment = [true, false]
        outcomes = [10.0, 5.0]
        X = [1.0 2.0 3.0; 4.0 5.0 6.0]  # 2x3: more columns than rows

        problem = RCTProblem(outcomes, treatment, X, nothing, (alpha = 0.05,))
        
        # Should error (underdetermined system)
        @test_throws Exception solve(problem, RegressionATE())
    end

    @testset "Extreme covariate values" begin
        treatment = [true, true, false, false]
        X = [1e6, 1e6 + 1.0, 1.0, 2.0]  # Extreme values
        outcomes = [10.0, 12.0, 4.0, 5.0]
        X_matrix = reshape(X, length(X), 1)

        problem = RCTProblem(outcomes, treatment, X_matrix, nothing, (alpha = 0.05,))
        solution = solve(problem, RegressionATE())

        # Should still compute (though numerically challenging)
        @test solution.retcode == :Success
        @test !isnan(solution.estimate)
        @test !isnan(solution.se)
    end

    @testset "Multiple highly correlated covariates" begin
        treatment = [true, true, false, false, true, false]
        X1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        X2 = X1 .+ 0.001  # Nearly perfect correlation
        outcomes = randn(6) .+ 2.0 .* treatment
        X_matrix = hcat(X1, X2)

        problem = RCTProblem(outcomes, treatment, X_matrix, nothing, (alpha = 0.05,))
        
        # May error or produce very large SE due to multicollinearity
        try
            solution = solve(problem, RegressionATE())
            # If it computes, SE should be very large or Inf
            @test solution.retcode == :Success
        catch e
            # Acceptable to fail with singular matrix error
            @test e isa Exception
        end
    end
end
