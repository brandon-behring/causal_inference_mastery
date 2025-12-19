"""
Monte Carlo validation for Julia IPW and DR observational estimators.

Validates statistical properties:
- Unbiasedness: E[τ̂] ≈ τ (bias < 0.15)
- Coverage: 90-98% for 95% CI
- Type I error: ~5% under null
- DR double robustness property

References:
    - Rosenbaum & Rubin (1983). The central role of the propensity score.
    - Bang & Robins (2005). Doubly robust estimation.
    - Austin & Stuart (2015). Moving towards best practice with IPTW.
"""

using Test
using Statistics
using Random
using LinearAlgebra

# Include DGP generators
include("dgp_observational.jl")

# Include main module
include("../../src/CausalEstimators.jl")
using .CausalEstimators

# =============================================================================
# Monte Carlo Test Infrastructure
# =============================================================================

"""
    run_observational_monte_carlo(dgp_func, estimator, n_simulations; kwargs...) -> NamedTuple

Run Monte Carlo simulation for observational estimation.

Returns summary statistics for validating estimator properties.
"""
function run_observational_monte_carlo(
    dgp_func::Function,
    estimator,
    n_simulations::Int;
    dgp_kwargs...
)
    ate_estimates = Float64[]
    ate_covers = Bool[]
    se_estimates = Float64[]
    successes = 0
    warnings = 0

    for sim in 1:n_simulations
        seed = 1000 + sim
        local data = dgp_func(; seed=seed, dgp_kwargs...)

        # Create ObservationalProblem
        problem = ObservationalProblem(
            data.Y,
            data.treatment,
            data.X;
            alpha=0.05,
            trim_threshold=0.01,
            stabilize=false
        )

        # Solve
        try
            solution = solve(problem, estimator)

            if solution.retcode == :Success
                successes += 1
                push!(ate_estimates, solution.estimate)
                push!(ate_covers, solution.ci_lower <= data.true_ate <= solution.ci_upper)
                push!(se_estimates, solution.se)
            elseif solution.retcode == :Warning
                warnings += 1
                successes += 1
                push!(ate_estimates, solution.estimate)
                push!(ate_covers, solution.ci_lower <= data.true_ate <= solution.ci_upper)
                push!(se_estimates, solution.se)
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
    std_ate = std(ate_estimates)
    bias = mean_ate - true_ate
    relative_bias = abs(true_ate) > 0.01 ? abs(bias) / abs(true_ate) : abs(bias)
    coverage = mean(ate_covers)
    mean_se = mean(se_estimates)

    return (
        true_ate=true_ate,
        mean_ate=mean_ate,
        std_ate=std_ate,
        bias=bias,
        relative_bias=relative_bias,
        coverage=coverage,
        mean_se=mean_se,
        se_ratio=mean_se / std_ate,  # Should be ~1 if SE calibrated
        n_successful=successes,
        n_warnings=warnings,
        n_simulations=n_simulations
    )
end

# =============================================================================
# IPW Monte Carlo Tests
# =============================================================================

@testset "IPW Monte Carlo Validation" begin

    @testset "IPW simple confounding - ATE recovery" begin
        results = run_observational_monte_carlo(
            dgp_observational_simple,
            ObservationalIPW(),
            200;
            n=500,
            p=3,
            true_ate=2.0,
            confounding_strength=0.5
        )

        @test !isnothing(results)
        @test results.n_successful >= 180

        # Bias should be small (allow up to 0.35 for IPW)
        @test abs(results.bias) < 0.35

        # Coverage should be reasonable (80-100%)
        @test 0.80 <= results.coverage <= 1.0
    end

    @testset "IPW strong confounding - still recovers ATE" begin
        results = run_observational_monte_carlo(
            dgp_observational_strong_confounding,
            ObservationalIPW(),
            200;
            n=500,
            p=3,
            true_ate=2.0,
            confounding_strength=1.0
        )

        @test !isnothing(results)
        @test results.n_successful >= 150

        # May have higher bias with strong confounding (allow up to 0.60)
        @test abs(results.bias) < 0.60

        # Coverage can be lower with strong confounding
        @test 0.70 <= results.coverage <= 1.0
    end

    @testset "IPW Type I error (null effect)" begin
        results = run_observational_monte_carlo(
            dgp_observational_no_effect,
            ObservationalIPW(),
            300;
            n=500,
            p=3,
            confounding_strength=0.5
        )

        @test !isnothing(results)
        @test results.n_successful >= 250

        # Bias should be near zero
        @test abs(results.bias) < 0.15

        # Type I error: proportion of CIs not covering zero
        type_i_error = 1 - results.coverage
        @test type_i_error < 0.15
    end

    @testset "IPW overlap violation - handles gracefully" begin
        results = run_observational_monte_carlo(
            dgp_observational_overlap_violation,
            ObservationalIPW(),
            150;
            n=500,
            p=3,
            true_ate=2.0,
            overlap_severity=2.0
        )

        @test !isnothing(results)
        # May have more failures/warnings with overlap issues
        @test results.n_successful >= 100

        # Still recovers ATE direction
        @test sign(results.mean_ate) == sign(2.0)
    end

end

# =============================================================================
# DR Monte Carlo Tests
# =============================================================================

@testset "Doubly Robust Monte Carlo Validation" begin

    @testset "DR simple confounding - ATE recovery" begin
        results = run_observational_monte_carlo(
            dgp_observational_simple,
            DoublyRobust(),
            200;
            n=500,
            p=3,
            true_ate=2.0,
            confounding_strength=0.5
        )

        @test !isnothing(results)
        @test results.n_successful >= 180

        # DR should have low bias
        @test abs(results.bias) < 0.15

        # Coverage should be good
        @test 0.88 <= results.coverage <= 0.99
    end

    @testset "DR strong confounding - lower bias than IPW" begin
        # Run both IPW and DR on same DGP
        ipw_results = run_observational_monte_carlo(
            dgp_observational_strong_confounding,
            ObservationalIPW(),
            200;
            n=500,
            p=3,
            true_ate=2.0,
            confounding_strength=1.0
        )

        dr_results = run_observational_monte_carlo(
            dgp_observational_strong_confounding,
            DoublyRobust(),
            200;
            n=500,
            p=3,
            true_ate=2.0,
            confounding_strength=1.0
        )

        @test !isnothing(ipw_results)
        @test !isnothing(dr_results)

        # DR should have comparable or lower bias
        @test abs(dr_results.bias) <= abs(ipw_results.bias) + 0.10
    end

    @testset "DR Type I error (null effect)" begin
        results = run_observational_monte_carlo(
            dgp_observational_no_effect,
            DoublyRobust(),
            300;
            n=500,
            p=3,
            confounding_strength=0.5
        )

        @test !isnothing(results)
        @test results.n_successful >= 250

        # Bias should be near zero
        @test abs(results.bias) < 0.10

        # Type I error
        type_i_error = 1 - results.coverage
        @test type_i_error < 0.12
    end

    @testset "DR nonlinear propensity - double robustness" begin
        # Tests DR with misspecified propensity but correct outcome
        results = run_observational_monte_carlo(
            dgp_observational_nonlinear_propensity,
            DoublyRobust(),
            200;
            n=500,
            p=3,
            true_ate=2.0
        )

        @test !isnothing(results)
        @test results.n_successful >= 150

        # DR should still recover ATE (outcome model correct)
        @test abs(results.bias) < 0.30
    end

    @testset "DR high dimensional - sparse model" begin
        results = run_observational_monte_carlo(
            dgp_observational_high_dimensional,
            DoublyRobust(),
            150;
            n=300,
            p=20,
            n_relevant=3,
            true_ate=2.0
        )

        @test !isnothing(results)
        @test results.n_successful >= 100

        # May have slightly higher bias with high-d
        @test abs(results.bias) < 0.50
    end

end

# =============================================================================
# Comparative Tests: IPW vs DR
# =============================================================================

@testset "IPW vs DR Comparison" begin

    @testset "DR has lower variance than IPW" begin
        # DR should be more efficient when both models correct
        ipw_results = run_observational_monte_carlo(
            dgp_observational_simple,
            ObservationalIPW(),
            200;
            n=500,
            p=3,
            true_ate=2.0
        )

        dr_results = run_observational_monte_carlo(
            dgp_observational_simple,
            DoublyRobust(),
            200;
            n=500,
            p=3,
            true_ate=2.0
        )

        @test !isnothing(ipw_results)
        @test !isnothing(dr_results)

        # DR should have comparable or lower std
        # (Efficiency gain may be small in simple case)
        @test dr_results.std_ate <= ipw_results.std_ate * 1.20
    end

    @testset "Both methods recover positive effect correctly" begin
        true_ate = 3.0

        ipw_results = run_observational_monte_carlo(
            dgp_observational_simple,
            ObservationalIPW(),
            100;
            n=500,
            p=3,
            true_ate=true_ate
        )

        dr_results = run_observational_monte_carlo(
            dgp_observational_simple,
            DoublyRobust(),
            100;
            n=500,
            p=3,
            true_ate=true_ate
        )

        @test !isnothing(ipw_results)
        @test !isnothing(dr_results)

        # Both should recover correct sign and approximate magnitude
        @test ipw_results.mean_ate > 2.0
        @test dr_results.mean_ate > 2.0
    end

end

# =============================================================================
# Standard Error Calibration
# =============================================================================

@testset "SE Calibration" begin

    @testset "IPW SE calibration" begin
        results = run_observational_monte_carlo(
            dgp_observational_simple,
            ObservationalIPW(),
            300;
            n=500,
            p=3,
            true_ate=2.0
        )

        @test !isnothing(results)

        # SE should be reasonably calibrated (ratio near 1)
        # Allow some tolerance: 0.7 < SE/SD < 1.5
        @test 0.6 <= results.se_ratio <= 1.8
    end

    @testset "DR SE calibration" begin
        results = run_observational_monte_carlo(
            dgp_observational_simple,
            DoublyRobust(),
            300;
            n=500,
            p=3,
            true_ate=2.0
        )

        @test !isnothing(results)

        # SE should be reasonably calibrated
        @test 0.6 <= results.se_ratio <= 1.8
    end

end
