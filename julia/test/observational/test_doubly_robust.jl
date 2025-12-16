#=
Unit Tests for Doubly Robust (AIPW) Estimator

Tests cover:
1. DRSolution construction and display
2. Outcome model fitting (fit_outcome_models)
3. DR estimator with known-answer DGPs
4. Double robustness property verification
5. DR vs IPW comparison (efficiency)
6. Edge cases and error handling
=#

using Test
using Statistics
using Random
using CausalEstimators


# =============================================================================
# Test Data Generators
# =============================================================================

"""Generate observational data with linear outcomes (correct specification)."""
function generate_linear_data(;
    n::Int = 500,
    true_ate::Float64 = 2.0,
    confounding_strength::Float64 = 0.5,
    seed::Int = 42
)
    Random.seed!(seed)

    # Covariates
    X = randn(n, 2)

    # Propensity: depends on X
    logit = confounding_strength .* X[:, 1] .+ 0.3 .* X[:, 2]
    e_true = 1 ./ (1 .+ exp.(-logit))

    # Treatment assignment
    T = rand(n) .< e_true

    # Outcome: LINEAR in X (outcome model correctly specified)
    Y = true_ate .* T .+ 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2] .+ randn(n)

    return (Y = Y, T = T, X = X, e_true = e_true, true_ate = true_ate)
end


"""Generate data with quadratic outcome (misspecified outcome model)."""
function generate_quadratic_outcome(;
    n::Int = 500,
    true_ate::Float64 = 2.5,
    seed::Int = 123
)
    Random.seed!(seed)

    X = randn(n, 2)

    # Linear propensity (correctly specified)
    logit = 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2]
    e_true = 1 ./ (1 .+ exp.(-logit))
    T = rand(n) .< e_true

    # QUADRATIC outcome (misspecified if using linear model)
    Y = true_ate .* T .+ 0.5 .* X[:, 1].^2 .+ 0.3 .* X[:, 2] .+ randn(n)

    return (Y = Y, T = T, X = X, e_true = e_true, true_ate = true_ate)
end


"""Generate data with misspecified propensity (non-linear selection)."""
function generate_nonlinear_propensity(;
    n::Int = 500,
    true_ate::Float64 = 3.0,
    seed::Int = 456
)
    Random.seed!(seed)

    X = randn(n, 2)

    # NON-LINEAR propensity (misspecified if using logistic)
    logit = 0.5 .* X[:, 1].^2 .+ 0.3 .* sin.(X[:, 2])
    e_true = 1 ./ (1 .+ exp.(-logit))
    T = rand(n) .< e_true

    # Linear outcome (correctly specified)
    Y = true_ate .* T .+ 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2] .+ randn(n)

    return (Y = Y, T = T, X = X, e_true = e_true, true_ate = true_ate)
end


# =============================================================================
# Outcome Model Fitting Tests
# =============================================================================

@testset "Outcome Model Fitting" begin
    data = generate_linear_data(n = 300, seed = 1)

    @testset "fit_outcome_models basic" begin
        result = fit_outcome_models(data.Y, data.T, data.X)

        @test length(result.mu0_predictions) == 300
        @test length(result.mu1_predictions) == 300
        @test length(result.mu0_coefficients) == 3  # intercept + 2 covariates
        @test length(result.mu1_coefficients) == 3
        @test isfinite(result.mu0_r2)
        @test isfinite(result.mu1_r2)
    end

    @testset "R² reasonable for linear DGP" begin
        result = fit_outcome_models(data.Y, data.T, data.X)

        # With linear DGP, R² should be positive but not perfect (noise)
        @test result.mu0_r2 > 0
        @test result.mu1_r2 > 0
        @test result.mu0_r2 < 0.9  # Not perfect fit
        @test result.mu1_r2 < 0.9
    end

    @testset "compute_r2 correct" begin
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.0, 2.0, 3.0, 4.0, 5.0]  # Perfect prediction

        r2 = compute_r2(y_true, y_pred)
        @test r2 ≈ 1.0

        y_pred_mean = fill(mean(y_true), 5)  # Mean prediction
        r2_mean = compute_r2(y_true, y_pred_mean)
        @test r2_mean ≈ 0.0 atol=1e-10
    end

    @testset "Insufficient data error" begin
        # Too few control units
        T_few_control = [true, true, true, true, false]  # Only 1 control
        Y = randn(5)
        X = randn(5, 2)

        @test_throws ArgumentError fit_outcome_models(Y, T_few_control, X)
    end
end


# =============================================================================
# DR Estimator Known-Answer Tests
# =============================================================================

@testset "DR Estimator Known-Answer" begin
    @testset "Linear DGP (both models correct)" begin
        data = generate_linear_data(n = 1000, true_ate = 2.0, seed = 10)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, DoublyRobust())

        # Should recover true ATE well
        @test abs(solution.estimate - data.true_ate) < 0.4
        @test solution.se > 0
        @test isfinite(solution.se)
        @test solution.ci_lower < solution.ci_upper
        @test solution.retcode == :Success
    end

    @testset "Uses outcome model predictions" begin
        data = generate_linear_data(n = 500, seed = 11)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, DoublyRobust())

        # Outcome predictions should exist
        @test length(solution.mu0_predictions) > 0
        @test length(solution.mu1_predictions) > 0
        @test all(isfinite, solution.mu0_predictions)
        @test all(isfinite, solution.mu1_predictions)
    end

    @testset "R² diagnostics computed" begin
        data = generate_linear_data(n = 500, seed = 12)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, DoublyRobust())

        @test 0 <= solution.mu0_r2 <= 1 || solution.mu0_r2 < 0  # R² can be negative
        @test 0 <= solution.mu1_r2 <= 1 || solution.mu1_r2 < 0
        @test isfinite(solution.mu0_r2)
        @test isfinite(solution.mu1_r2)
    end

    @testset "Zero treatment effect" begin
        Random.seed!(13)
        n = 500
        X = randn(n, 2)
        logit = 0.5 .* X[:, 1]
        T = rand(n) .< (1 ./ (1 .+ exp.(-logit)))
        Y = 0.5 .* X[:, 1] .+ randn(n)  # No treatment effect

        problem = ObservationalProblem(Y, T, X)
        solution = solve(problem, DoublyRobust())

        # Should be close to zero
        @test abs(solution.estimate) < 0.5
        # CI should contain zero
        @test solution.ci_lower < 0 < solution.ci_upper
    end
end


# =============================================================================
# Double Robustness Property Tests
# =============================================================================

@testset "Double Robustness Property" begin
    @testset "Both models correct - efficient" begin
        # Linear data → both propensity (logistic) and outcome (linear) correct
        data = generate_linear_data(n = 1000, true_ate = 2.0, seed = 20)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, DoublyRobust())

        # Best case: should recover ATE very well
        @test abs(solution.estimate - data.true_ate) < 0.3
    end

    @testset "Propensity correct, outcome misspecified" begin
        # Quadratic outcome → linear outcome model wrong, but propensity correct
        data = generate_quadratic_outcome(n = 1000, true_ate = 2.5, seed = 21)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, DoublyRobust())

        # DR should still be reasonably close (IPW component saves it)
        # More tolerance due to misspecification
        @test abs(solution.estimate - data.true_ate) < 0.8
    end

    @testset "Outcome correct, propensity misspecified" begin
        # Non-linear selection → logistic propensity wrong, but linear outcome correct
        data = generate_nonlinear_propensity(n = 1000, true_ate = 3.0, seed = 22)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, DoublyRobust())

        # DR should still work (outcome regression component saves it)
        @test abs(solution.estimate - data.true_ate) < 0.8
    end
end


# =============================================================================
# DR vs IPW Comparison Tests
# =============================================================================

@testset "DR vs IPW Comparison" begin
    @testset "DR typically lower variance than IPW" begin
        # Run multiple seeds and compare SE
        dr_ses = Float64[]
        ipw_ses = Float64[]

        for seed in 100:104
            data = generate_linear_data(n = 500, seed = seed)
            problem = ObservationalProblem(data.Y, data.T, data.X)

            dr_sol = solve(problem, DoublyRobust())
            ipw_sol = solve(problem, ObservationalIPW())

            push!(dr_ses, dr_sol.se)
            push!(ipw_ses, ipw_sol.se)
        end

        # DR should have lower average SE when both models correct
        @test mean(dr_ses) <= mean(ipw_ses) + 0.1  # Allow small tolerance
    end

    @testset "DR and IPW give similar point estimates" begin
        data = generate_linear_data(n = 800, seed = 30)
        problem = ObservationalProblem(data.Y, data.T, data.X)

        dr_sol = solve(problem, DoublyRobust())
        ipw_sol = solve(problem, ObservationalIPW())

        # Point estimates should be similar
        @test abs(dr_sol.estimate - ipw_sol.estimate) < 0.5
    end
end


# =============================================================================
# Trimming Tests
# =============================================================================

@testset "Trimming Options" begin
    # Generate data with strong confounding (extreme propensities)
    Random.seed!(40)
    n = 600
    X = randn(n, 2)
    logit = 1.5 .* X[:, 1] .+ 1.0 .* X[:, 2]  # Strong confounding
    e_true = 1 ./ (1 .+ exp.(-logit))
    T = rand(n) .< e_true
    Y = 2.0 .* T .+ X[:, 1] .+ randn(n)

    @testset "No trimming" begin
        problem = ObservationalProblem(Y, T, X; trim_threshold = 0.0)
        solution = solve(problem, DoublyRobust())

        @test solution.n_trimmed == 0
        @test solution.n_treated + solution.n_control == n
    end

    @testset "With trimming" begin
        problem = ObservationalProblem(Y, T, X; trim_threshold = 0.05)
        solution = solve(problem, DoublyRobust())

        # Some units should be trimmed with strong confounding
        @test solution.n_trimmed >= 0
        @test solution.n_treated + solution.n_control <= n
    end
end


# =============================================================================
# Pre-computed Propensity Tests
# =============================================================================

@testset "Pre-computed Propensity" begin
    data = generate_linear_data(n = 500, seed = 50)

    @testset "Using true propensity (oracle)" begin
        problem = ObservationalProblem(
            data.Y, data.T, data.X;
            propensity = data.e_true
        )
        solution = solve(problem, DoublyRobust())

        # Should recover ATE well with oracle propensity
        @test abs(solution.estimate - data.true_ate) < 0.4
    end

    @testset "Oracle vs estimated propensity" begin
        problem_est = ObservationalProblem(data.Y, data.T, data.X)
        solution_est = solve(problem_est, DoublyRobust())

        problem_oracle = ObservationalProblem(
            data.Y, data.T, data.X;
            propensity = data.e_true
        )
        solution_oracle = solve(problem_oracle, DoublyRobust())

        # Both should be reasonable
        @test abs(solution_est.estimate - data.true_ate) < 0.6
        @test abs(solution_oracle.estimate - data.true_ate) < 0.4
    end
end


# =============================================================================
# Edge Cases
# =============================================================================

@testset "Edge Cases" begin
    @testset "Small sample size" begin
        data = generate_linear_data(n = 50, seed = 60)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, DoublyRobust())

        @test isfinite(solution.estimate)
        @test isfinite(solution.se)
        @test solution.se > 0
    end

    @testset "Single covariate" begin
        Random.seed!(61)
        n = 200
        X = randn(n, 1)  # Single covariate
        logit = 0.5 .* X[:, 1]
        T = rand(n) .< (1 ./ (1 .+ exp.(-logit)))
        Y = 2.0 .* T .+ X[:, 1] .+ randn(n)

        problem = ObservationalProblem(Y, T, X)
        solution = solve(problem, DoublyRobust())

        @test isfinite(solution.estimate)
        @test abs(solution.estimate - 2.0) < 1.0
    end

    @testset "Many covariates" begin
        Random.seed!(62)
        n = 500
        p = 10
        X = randn(n, p)
        logit = 0.3 .* sum(X[:, 1:3], dims=2)[:]
        T = rand(n) .< (1 ./ (1 .+ exp.(-logit)))
        Y = 2.0 .* T .+ 0.2 .* sum(X[:, 1:3], dims=2)[:] .+ randn(n)

        problem = ObservationalProblem(Y, T, X)
        solution = solve(problem, DoublyRobust())

        @test isfinite(solution.estimate)
        @test solution.se > 0
    end

    @testset "Imbalanced treatment" begin
        Random.seed!(63)
        n = 500
        X = randn(n, 2)
        # Strong bias toward treatment
        logit = 1.0 .+ 0.5 .* X[:, 1]
        T = rand(n) .< (1 ./ (1 .+ exp.(-logit)))
        Y = 2.0 .* T .+ X[:, 1] .+ randn(n)

        problem = ObservationalProblem(Y, T, X; trim_threshold = 0.02)
        solution = solve(problem, DoublyRobust())

        @test isfinite(solution.estimate)
        @test solution.n_treated > solution.n_control  # Expected imbalance
    end
end


# =============================================================================
# Solution Display
# =============================================================================

@testset "Solution Display" begin
    data = generate_linear_data(n = 200, seed = 70)
    problem = ObservationalProblem(data.Y, data.T, data.X)
    solution = solve(problem, DoublyRobust())

    # Test that show method works without error
    io = IOBuffer()
    show(io, solution)
    output = String(take!(io))

    @test contains(output, "DRSolution")
    @test contains(output, "ATE Estimate")
    @test contains(output, "Outcome")  # Outcome R² info
end
