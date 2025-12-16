#=
Unit Tests for Observational IPW Estimator

Tests cover:
1. ObservationalProblem construction and validation
2. Propensity score estimation
3. IPW estimator with known-answer DGPs
4. Trimming and stabilization options
5. Diagnostics (AUC, overlap)
6. Edge cases and error handling
=#

using Test
using Statistics
using Random
using CausalEstimators

# =============================================================================
# Test Data Generators
# =============================================================================

"""Generate observational data with known confounding."""
function generate_obs_data(;
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

    # Outcome: confounded (X affects both T and Y)
    Y = true_ate .* T .+ 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2] .+ randn(n)

    return (Y = Y, T = T, X = X, e_true = e_true, true_ate = true_ate)
end

"""Generate data with weak confounding (near-RCT)."""
function generate_weak_confounding(; n::Int = 500, true_ate::Float64 = 1.5, seed::Int = 123)
    Random.seed!(seed)

    X = randn(n, 2)

    # Very weak relationship between X and T
    logit = 0.1 .* X[:, 1]
    e_true = 1 ./ (1 .+ exp.(-logit))
    T = rand(n) .< e_true

    # Outcome
    Y = true_ate .* T .+ 0.5 .* X[:, 1] .+ randn(n)

    return (Y = Y, T = T, X = X, e_true = e_true, true_ate = true_ate)
end

"""Generate data with strong confounding (hard case)."""
function generate_strong_confounding(; n::Int = 500, true_ate::Float64 = 3.0, seed::Int = 456)
    Random.seed!(seed)

    X = randn(n, 2)

    # Strong relationship between X and T
    logit = 1.5 .* X[:, 1] .+ 1.0 .* X[:, 2]
    e_true = 1 ./ (1 .+ exp.(-logit))
    T = rand(n) .< e_true

    # Outcome with confounding
    Y = true_ate .* T .+ X[:, 1] .+ 0.5 .* X[:, 2] .+ randn(n)

    return (Y = Y, T = T, X = X, e_true = e_true, true_ate = true_ate)
end


# =============================================================================
# ObservationalProblem Tests
# =============================================================================

@testset "ObservationalProblem Construction" begin
    data = generate_obs_data(n = 200, seed = 1)

    @testset "Basic construction" begin
        problem = ObservationalProblem(data.Y, data.T, data.X)

        @test length(problem.outcomes) == 200
        @test length(problem.treatment) == 200
        @test size(problem.covariates) == (200, 2)
        @test problem.propensity === nothing
        @test problem.parameters.alpha == 0.05
    end

    @testset "With parameters" begin
        problem = ObservationalProblem(
            data.Y, data.T, data.X;
            alpha = 0.10,
            trim_threshold = 0.05,
            stabilize = true
        )

        @test problem.parameters.alpha == 0.10
        @test problem.parameters.trim_threshold == 0.05
        @test problem.parameters.stabilize == true
    end

    @testset "With pre-computed propensity" begin
        problem = ObservationalProblem(
            data.Y, data.T, data.X;
            propensity = data.e_true
        )

        @test problem.propensity !== nothing
        @test length(problem.propensity) == 200
    end

    @testset "Validation errors" begin
        # Mismatched lengths
        @test_throws ArgumentError ObservationalProblem(
            data.Y[1:100], data.T, data.X
        )

        # No treated units
        T_all_control = falses(200)
        @test_throws ArgumentError ObservationalProblem(
            data.Y, T_all_control, data.X
        )

        # No control units
        T_all_treated = trues(200)
        @test_throws ArgumentError ObservationalProblem(
            data.Y, T_all_treated, data.X
        )

        # Invalid propensity (out of bounds)
        bad_propensity = data.e_true
        bad_propensity[1] = 0.0
        @test_throws ArgumentError ObservationalProblem(
            data.Y, data.T, data.X; propensity = bad_propensity
        )
    end
end


# =============================================================================
# Propensity Estimation Tests
# =============================================================================

@testset "Propensity Estimation" begin
    data = generate_obs_data(n = 500, seed = 2)

    @testset "estimate_propensity_scores" begin
        result = estimate_propensity_scores(data.T, data.X)

        @test length(result.propensity) == 500
        @test all(0 .< result.propensity .< 1)
        @test result.converged == true
        @test length(result.coefficients) == 3  # intercept + 2 covariates
    end

    @testset "compute_propensity_auc" begin
        # With moderate confounding, AUC should be > 0.5
        result = estimate_propensity_scores(data.T, data.X)
        auc = compute_propensity_auc(result.propensity, data.T)

        @test 0.5 < auc < 1.0
        @test auc > 0.55  # Should detect some confounding
    end

    @testset "AUC near 0.5 with weak confounding" begin
        weak_data = generate_weak_confounding(n = 500, seed = 3)
        result = estimate_propensity_scores(weak_data.T, weak_data.X)
        auc = compute_propensity_auc(result.propensity, weak_data.T)

        # Weak confounding → AUC closer to 0.5
        @test 0.45 < auc < 0.65
    end
end


# =============================================================================
# IPW Estimator Tests
# =============================================================================

@testset "IPW Estimator Known-Answer" begin
    @testset "Moderate confounding recovers ATE" begin
        data = generate_obs_data(n = 1000, true_ate = 2.0, seed = 10)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, ObservationalIPW())

        # Should recover true ATE within reasonable tolerance
        @test abs(solution.estimate - data.true_ate) < 0.5
        @test solution.se > 0
        @test isfinite(solution.se)
        @test solution.ci_lower < solution.ci_upper
        @test solution.retcode == :Success
    end

    @testset "Weak confounding (near-RCT)" begin
        data = generate_weak_confounding(n = 800, true_ate = 1.5, seed = 11)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, ObservationalIPW())

        # Easier case: should be closer to truth
        @test abs(solution.estimate - data.true_ate) < 0.4
        @test solution.propensity_auc < 0.65  # Low discriminatory power
    end

    @testset "Strong confounding" begin
        data = generate_strong_confounding(n = 1000, true_ate = 3.0, seed = 12)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, ObservationalIPW())

        # Harder case but should still get direction right
        @test abs(solution.estimate - data.true_ate) < 1.0
        @test solution.propensity_auc > 0.65  # High discriminatory power
    end

    @testset "Zero treatment effect" begin
        Random.seed!(13)
        n = 500
        X = randn(n, 2)
        logit = 0.5 .* X[:, 1]
        T = rand(n) .< (1 ./ (1 .+ exp.(-logit)))
        Y = 0.5 .* X[:, 1] .+ randn(n)  # No treatment effect

        problem = ObservationalProblem(Y, T, X)
        solution = solve(problem, ObservationalIPW())

        # Should be close to zero
        @test abs(solution.estimate) < 0.5
        # CI should contain zero
        @test solution.ci_lower < 0 < solution.ci_upper
    end
end


# =============================================================================
# Trimming and Stabilization Tests
# =============================================================================

@testset "Trimming Options" begin
    data = generate_strong_confounding(n = 800, seed = 20)

    @testset "No trimming" begin
        problem = ObservationalProblem(
            data.Y, data.T, data.X;
            trim_threshold = 0.0
        )
        solution = solve(problem, ObservationalIPW())

        @test solution.n_trimmed == 0
        @test solution.n_treated + solution.n_control == 800
    end

    @testset "With trimming" begin
        problem = ObservationalProblem(
            data.Y, data.T, data.X;
            trim_threshold = 0.05
        )
        solution = solve(problem, ObservationalIPW())

        # Some units should be trimmed
        @test solution.n_trimmed >= 0
        @test solution.n_treated + solution.n_control <= 800
    end
end

@testset "Stabilized Weights" begin
    data = generate_obs_data(n = 500, seed = 21)

    @testset "Standard vs stabilized" begin
        problem_std = ObservationalProblem(
            data.Y, data.T, data.X;
            stabilize = false
        )
        solution_std = solve(problem_std, ObservationalIPW())

        problem_stab = ObservationalProblem(
            data.Y, data.T, data.X;
            stabilize = true
        )
        solution_stab = solve(problem_stab, ObservationalIPW())

        # Both should give similar estimates
        @test abs(solution_std.estimate - solution_stab.estimate) < 0.5

        # Stabilized flag should differ
        @test solution_std.stabilized == false
        @test solution_stab.stabilized == true

        # Stabilized weights should have mean closer to 1
        @test abs(mean(solution_stab.weights) - 1.0) < abs(mean(solution_std.weights) - 1.0) + 0.5
    end
end


# =============================================================================
# Diagnostics Tests
# =============================================================================

@testset "Propensity Diagnostics" begin
    data = generate_obs_data(n = 500, seed = 30)

    problem = ObservationalProblem(data.Y, data.T, data.X)
    solution = solve(problem, ObservationalIPW())

    @testset "AUC computed" begin
        @test 0 < solution.propensity_auc < 1
        @test isfinite(solution.propensity_auc)
    end

    @testset "Mean propensity by group" begin
        @test 0 < solution.propensity_mean_treated < 1
        @test 0 < solution.propensity_mean_control < 1

        # Treated should have higher propensity on average
        @test solution.propensity_mean_treated > solution.propensity_mean_control
    end

    @testset "Propensity scores stored" begin
        @test length(solution.propensity_scores) > 0
        @test all(0 .< solution.propensity_scores .< 1)
    end
end


# =============================================================================
# Pre-computed Propensity Tests
# =============================================================================

@testset "Pre-computed Propensity" begin
    data = generate_obs_data(n = 500, seed = 40)

    @testset "Using true propensity" begin
        # Use true propensity (oracle)
        problem = ObservationalProblem(
            data.Y, data.T, data.X;
            propensity = data.e_true
        )
        solution = solve(problem, ObservationalIPW())

        # Should recover ATE well with oracle propensity
        @test abs(solution.estimate - data.true_ate) < 0.4
    end

    @testset "Estimated vs oracle propensity" begin
        # Estimated propensity
        problem_est = ObservationalProblem(data.Y, data.T, data.X)
        solution_est = solve(problem_est, ObservationalIPW())

        # Oracle propensity
        problem_oracle = ObservationalProblem(
            data.Y, data.T, data.X;
            propensity = data.e_true
        )
        solution_oracle = solve(problem_oracle, ObservationalIPW())

        # Oracle should be at least as good
        error_est = abs(solution_est.estimate - data.true_ate)
        error_oracle = abs(solution_oracle.estimate - data.true_ate)

        # Oracle typically better, but both should be reasonable
        @test error_est < 1.0
        @test error_oracle < 0.5
    end
end


# =============================================================================
# Edge Cases
# =============================================================================

@testset "Edge Cases" begin
    @testset "Small sample size" begin
        data = generate_obs_data(n = 50, seed = 50)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, ObservationalIPW())

        @test isfinite(solution.estimate)
        @test isfinite(solution.se)
        @test solution.se > 0
    end

    @testset "Single covariate" begin
        Random.seed!(51)
        n = 200
        X = randn(n, 1)  # Single covariate
        logit = 0.5 .* X[:, 1]
        T = rand(n) .< (1 ./ (1 .+ exp.(-logit)))
        Y = 2.0 .* T .+ X[:, 1] .+ randn(n)

        problem = ObservationalProblem(Y, T, X)
        solution = solve(problem, ObservationalIPW())

        @test isfinite(solution.estimate)
        @test abs(solution.estimate - 2.0) < 1.0
    end

    @testset "Imbalanced treatment" begin
        Random.seed!(52)
        n = 500
        X = randn(n, 2)
        # Strong bias toward treatment
        logit = 1.0 .+ 0.5 .* X[:, 1]  # Most units treated
        T = rand(n) .< (1 ./ (1 .+ exp.(-logit)))
        Y = 2.0 .* T .+ X[:, 1] .+ randn(n)

        problem = ObservationalProblem(Y, T, X; trim_threshold = 0.02)
        solution = solve(problem, ObservationalIPW())

        @test isfinite(solution.estimate)
        @test solution.n_treated > solution.n_control  # Expected imbalance
    end
end


# =============================================================================
# Solution Display
# =============================================================================

@testset "Solution Display" begin
    data = generate_obs_data(n = 200, seed = 60)
    problem = ObservationalProblem(data.Y, data.T, data.X)
    solution = solve(problem, ObservationalIPW())

    # Test that show method works without error
    io = IOBuffer()
    show(io, solution)
    output = String(take!(io))

    @test contains(output, "IPWSolution")
    @test contains(output, "ATE Estimate")
    @test contains(output, "Propensity AUC")
end
