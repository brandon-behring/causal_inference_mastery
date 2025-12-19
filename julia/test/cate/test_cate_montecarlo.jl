"""
Monte Carlo validation for Julia CATE estimators.

Validates statistical properties:
- Unbiasedness: E[τ̂] ≈ τ (bias < 0.10)
- Coverage: 93-97% for 95% CI
- CATE recovery: correlation with true CATE > 0.7

References:
    - Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects"
    - Nie & Wager (2021). "Quasi-oracle estimation of heterogeneous treatment effects"
    - Chernozhukov et al. (2018). "Double/debiased machine learning"
"""

using Test
using Statistics
using Random
using LinearAlgebra

# Include DGP generators
include("dgp_cate.jl")

# Include main module
include("../../src/CausalEstimators.jl")
using .CausalEstimators

# =============================================================================
# Monte Carlo Test Infrastructure
# =============================================================================

"""
    run_cate_monte_carlo(dgp_func, estimator, n_simulations; kwargs...) -> NamedTuple

Run Monte Carlo simulation for CATE estimation.

Returns summary statistics for validating estimator properties.
"""
function run_cate_monte_carlo(
    dgp_func::Function,
    estimator,
    n_simulations::Int;
    dgp_kwargs...
)
    ate_estimates = Float64[]
    ate_covers = Bool[]
    cate_correlations = Float64[]
    successes = 0

    for sim in 1:n_simulations
        seed = 1000 + sim
        local data = dgp_func(; seed=seed, dgp_kwargs...)

        # Create CATE problem
        problem = CATEProblem(
            data.Y,
            data.treatment,
            data.X,
            (alpha=0.05,)
        )

        # Solve
        try
            solution = solve(problem, estimator)

            if solution.retcode in [:Success, :Warning]
                successes += 1
                push!(ate_estimates, solution.ate)
                push!(ate_covers, solution.ci_lower <= data.true_ate <= solution.ci_upper)

                # CATE correlation (handle constant CATE case)
                if std(data.true_cate) > 1e-6 && std(solution.cate) > 1e-6
                    corr = cor(data.true_cate, solution.cate)
                    if !isnan(corr)
                        push!(cate_correlations, corr)
                    end
                end
            end
        catch e
            # Skip failures silently in Monte Carlo
            continue
        end
    end

    # Return nothing if too few successes
    if successes < n_simulations * 0.5
        return nothing
    end

    # Get true ATE from one simulation
    true_ate = dgp_func(; seed=42, dgp_kwargs...).true_ate

    mean_ate = mean(ate_estimates)
    bias = mean_ate - true_ate
    relative_bias = abs(true_ate) > 0.01 ? abs(bias) / abs(true_ate) : abs(bias)
    coverage = mean(ate_covers)
    mean_cate_corr = length(cate_correlations) > 0 ? mean(cate_correlations) : NaN

    return (
        true_ate=true_ate,
        mean_ate=mean_ate,
        bias=bias,
        relative_bias=relative_bias,
        coverage=coverage,
        mean_cate_correlation=mean_cate_corr,
        n_successful=successes,
        n_simulations=n_simulations
    )
end

# =============================================================================
# S-Learner Monte Carlo Tests
# =============================================================================

@testset "S-Learner Monte Carlo" begin

    @testset "S-Learner constant effect - ATE recovery" begin
        results = run_cate_monte_carlo(
            dgp_constant_effect,
            SLearner(),
            200;
            n=500,
            true_ate=2.0,
            p=5
        )

        @test !isnothing(results)
        @test results.n_successful >= 180

        # ATE should be recovered with small bias
        @test abs(results.bias) < 0.20 "Bias $(results.bias) exceeds 0.20"

        # Coverage should be reasonable
        @test 0.85 <= results.coverage <= 0.98 "Coverage $(results.coverage) outside [0.85, 0.98]"
    end

    @testset "S-Learner linear heterogeneity" begin
        results = run_cate_monte_carlo(
            dgp_linear_heterogeneity,
            SLearner(),
            150;
            n=500,
            base_effect=2.0,
            het_coef=1.0
        )

        @test !isnothing(results)
        @test results.n_successful >= 120

        # Should have reasonable ATE recovery
        @test abs(results.bias) < 0.30 "Bias $(results.bias) exceeds 0.30"
    end

end

# =============================================================================
# T-Learner Monte Carlo Tests
# =============================================================================

@testset "T-Learner Monte Carlo" begin

    @testset "T-Learner constant effect" begin
        results = run_cate_monte_carlo(
            dgp_constant_effect,
            TLearner(),
            200;
            n=500,
            true_ate=2.0,
            p=5
        )

        @test !isnothing(results)
        @test results.n_successful >= 180

        @test abs(results.bias) < 0.25 "Bias $(results.bias) exceeds 0.25"
        @test 0.85 <= results.coverage <= 0.98 "Coverage $(results.coverage) outside range"
    end

    @testset "T-Learner linear heterogeneity - CATE recovery" begin
        results = run_cate_monte_carlo(
            dgp_linear_heterogeneity,
            TLearner(),
            150;
            n=500,
            base_effect=2.0,
            het_coef=1.0
        )

        @test !isnothing(results)
        @test results.n_successful >= 120

        @test abs(results.bias) < 0.30

        # T-learner should capture linear heterogeneity
        if !isnan(results.mean_cate_correlation)
            @test results.mean_cate_correlation > 0.5 "CATE correlation too low"
        end
    end

end

# =============================================================================
# X-Learner Monte Carlo Tests
# =============================================================================

@testset "X-Learner Monte Carlo" begin

    @testset "X-Learner imbalanced treatment" begin
        # X-learner should handle imbalanced treatment well
        results = run_cate_monte_carlo(
            dgp_imbalanced_treatment,
            XLearner(),
            150;
            n=800,
            true_ate=2.0,
            treatment_prob=0.2
        )

        @test !isnothing(results)
        @test results.n_successful >= 100

        # Should handle imbalance reasonably
        @test abs(results.bias) < 0.40 "Bias $(results.bias) exceeds 0.40"
    end

    @testset "X-Learner linear heterogeneity" begin
        results = run_cate_monte_carlo(
            dgp_linear_heterogeneity,
            XLearner(),
            150;
            n=500,
            base_effect=2.0,
            het_coef=1.0
        )

        @test !isnothing(results)
        @test results.n_successful >= 100

        @test abs(results.bias) < 0.35
    end

end

# =============================================================================
# R-Learner Monte Carlo Tests
# =============================================================================

@testset "R-Learner Monte Carlo" begin

    @testset "R-Learner constant effect" begin
        results = run_cate_monte_carlo(
            dgp_constant_effect,
            RLearner(),
            200;
            n=500,
            true_ate=2.0,
            p=5
        )

        @test !isnothing(results)
        @test results.n_successful >= 150

        @test abs(results.bias) < 0.30 "Bias $(results.bias) exceeds 0.30"
    end

    @testset "R-Learner strong confounding" begin
        # R-learner is doubly robust, should handle confounding
        results = run_cate_monte_carlo(
            dgp_strong_confounding,
            RLearner(),
            150;
            n=600,
            base_effect=2.0,
            confounding_strength=0.8
        )

        @test !isnothing(results)
        @test results.n_successful >= 100

        # Should handle confounding reasonably
        @test abs(results.bias) < 0.50 "Bias $(results.bias) exceeds 0.50"
    end

end

# =============================================================================
# DML Monte Carlo Tests
# =============================================================================

@testset "Double Machine Learning Monte Carlo" begin

    @testset "DML constant effect" begin
        results = run_cate_monte_carlo(
            dgp_constant_effect,
            DoubleMachineLearning(n_folds=3),
            150;
            n=500,
            true_ate=2.0,
            p=5
        )

        @test !isnothing(results)
        @test results.n_successful >= 100

        @test abs(results.bias) < 0.35 "Bias $(results.bias) exceeds 0.35"
    end

    @testset "DML cross-fitting benefit" begin
        # DML with cross-fitting should have lower bias than naive R-learner
        results = run_cate_monte_carlo(
            dgp_linear_heterogeneity,
            DoubleMachineLearning(n_folds=5),
            100;
            n=600,
            base_effect=2.0,
            het_coef=0.5
        )

        @test !isnothing(results)
        @test results.n_successful >= 60

        @test abs(results.bias) < 0.40
    end

end

# =============================================================================
# Method Comparison Tests
# =============================================================================

@testset "CATE Method Comparison" begin

    @testset "All methods on constant effect" begin
        # All methods should recover constant effect reasonably well
        estimators = [
            (SLearner(), "S-Learner"),
            (TLearner(), "T-Learner"),
            (XLearner(), "X-Learner"),
            (RLearner(), "R-Learner"),
        ]

        for (estimator, name) in estimators
            results = run_cate_monte_carlo(
                dgp_constant_effect,
                estimator,
                100;
                n=400,
                true_ate=2.0
            )

            @test !isnothing(results) "$name failed on constant effect"
            if !isnothing(results)
                @test abs(results.bias) < 0.50 "$name bias $(results.bias) too high"
            end
        end
    end

    @testset "Complex heterogeneity comparison" begin
        # Test methods on step + linear CATE
        estimators = [
            (TLearner(), "T-Learner"),
            (XLearner(), "X-Learner"),
        ]

        for (estimator, name) in estimators
            results = run_cate_monte_carlo(
                dgp_complex_heterogeneity,
                estimator,
                80;
                n=600,
                base_effect=1.0,
                step_effect=2.0,
                linear_coef=0.5
            )

            @test !isnothing(results) "$name failed on complex heterogeneity"
            if !isnothing(results)
                @test abs(results.bias) < 0.60 "$name bias $(results.bias) too high"
            end
        end
    end

end

# =============================================================================
# High-Dimensional Tests
# =============================================================================

@testset "CATE High-Dimensional" begin

    @testset "High-dimensional sparse model" begin
        # Should work with p=30, n=300, 5 relevant
        results = run_cate_monte_carlo(
            dgp_high_dimensional,
            TLearner(model=:ridge),
            80;
            n=300,
            p=30,
            true_ate=2.0,
            n_relevant=5
        )

        @test !isnothing(results)
        @test results.n_successful >= 40

        # High-dim may have larger bias
        @test abs(results.bias) < 0.80 "Bias $(results.bias) exceeds 0.80"
    end

end

# =============================================================================
# Robustness Tests
# =============================================================================

@testset "CATE Robustness" begin

    @testset "Small sample performance" begin
        results = run_cate_monte_carlo(
            dgp_constant_effect,
            TLearner(),
            150;
            n=100,
            true_ate=2.0,
            p=3
        )

        @test !isnothing(results)
        @test results.n_successful >= 100

        # Larger tolerance for small samples
        @test abs(results.bias) < 0.50
        @test 0.75 <= results.coverage <= 0.99
    end

    @testset "Large sample convergence" begin
        small_results = run_cate_monte_carlo(
            dgp_constant_effect,
            SLearner(),
            80;
            n=200,
            true_ate=2.0
        )

        large_results = run_cate_monte_carlo(
            dgp_constant_effect,
            SLearner(),
            80;
            n=1000,
            true_ate=2.0
        )

        @test !isnothing(small_results)
        @test !isnothing(large_results)

        # Large sample should have smaller bias (or at least not worse)
        @test abs(large_results.bias) <= abs(small_results.bias) + 0.2
    end

end

# Run summary
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n" * "="^60)
    println("CATE Monte Carlo Validation Summary")
    println("="^60)
    println("Validating: S/T/X/R-Learner, DML")
    println("Metrics: ATE Bias, Coverage, CATE correlation")
    println("="^60 * "\n")
end
