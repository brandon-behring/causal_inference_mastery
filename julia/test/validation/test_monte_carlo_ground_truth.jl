"""
Monte Carlo Ground Truth Validation for CausalEstimators.jl

Tests all estimators against datasets with KNOWN treatment effects to validate:
1. Unbiasedness: Mean estimate ≈ true τ (bias < 0.05)
2. Coverage: 95% CI contains true τ in 94-96% of datasets
3. SE accuracy: Empirical SD ≈ mean reported SE

This provides ground truth validation that catches conceptual bugs that both
Python and Julia implementations might share (e.g., incorrect variance formulas,
systematic biases in estimation procedures).

Why Monte Carlo validation is critical:
- Cross-language validation (Python ↔ Julia) only catches implementation bugs
- If both implementations share the same conceptual error, they'll agree but be wrong
- Ground truth validation catches these shared errors

Reference:
- MacKinnon, J. G. (2009). "Bootstrap hypothesis testing"
- Morris, T. P., et al. (2019). "Proposals on Kaplan–Meier plots in medical
  research and a survey of stakeholder views: KMunicate"
"""

using Test
using CausalEstimators
using Random
using Statistics
using PyCall
using Distributions

# Import Python implementations for comparison
py"""
import sys
sys.path.insert(0, '/home/brandon_behring/Claude/causal_inference_mastery/src')
from causal_inference.rct.estimators import simple_ate
from causal_inference.rct.estimators_stratified import stratified_ate
from causal_inference.rct.estimators_regression import regression_adjusted_ate
from causal_inference.rct.estimators_permutation import permutation_test
from causal_inference.rct.estimators_ipw import ipw_ate
"""

# ============================================================================
# Data Generating Processes (DGPs) with Known Treatment Effects
# ============================================================================

"""
Generate RCT data with known constant treatment effect.

Parameters:
- n: Sample size
- τ: True average treatment effect (ATE)
- σ_Y0: Standard deviation of control potential outcomes
- σ_Y1: Standard deviation of treated potential outcomes (for heteroskedasticity)
- seed: Random seed for reproducibility

Returns:
- outcomes: Observed outcomes Y = T*Y(1) + (1-T)*Y(0)
- treatment: Treatment assignment (balanced)
- τ_true: True ATE (returned for clarity)

DGP:
  Y(0) ~ N(0, σ_Y0²)
  Y(1) ~ N(τ, σ_Y1²)
  T ~ Bernoulli(0.5)
  Y = T*Y(1) + (1-T)*Y(0)

True ATE: E[Y(1) - Y(0)] = τ
"""
function dgp_constant_ate(;
    n::Int = 100,
    τ::Float64 = 2.0,
    σ_Y0::Float64 = 1.0,
    σ_Y1::Float64 = 1.0,
    seed::Union{Int,Nothing} = nothing,
)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    n_treated = div(n, 2)
    n_control = n - n_treated

    # Generate potential outcomes
    Y0 = randn(n) .* σ_Y0
    Y1 = Y0 .+ τ .+ randn(n) .* (σ_Y1 - σ_Y0)  # Heteroskedastic treatment effects

    # Treatment assignment (balanced)
    treatment = vcat(fill(true, n_treated), fill(false, n_control))

    # Observed outcomes (fundamental equation of causal inference)
    outcomes = ifelse.(treatment, Y1, Y0)

    return outcomes, treatment, τ
end

"""
Generate stratified RCT data with known constant treatment effect within strata.

Parameters:
- n: Sample size
- n_strata: Number of strata
- τ: True average treatment effect (constant across strata)
- strata_effects: Baseline differences between strata
- seed: Random seed

Returns:
- outcomes, treatment, strata, τ_true

DGP:
  Y(0) = strata_effect[s] + ε
  Y(1) = Y(0) + τ
  T ~ Bernoulli(0.5) within each stratum

True ATE: E[Y(1) - Y(0)] = τ (constant across strata)
"""
function dgp_stratified_ate(;
    n::Int = 100,
    n_strata::Int = 5,
    τ::Float64 = 2.0,
    strata_effects::Vector{Float64} = Float64[-2.0, -1.0, 0.0, 1.0, 2.0],
    seed::Union{Int,Nothing} = nothing,
)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    @assert length(strata_effects) == n_strata

    strata = repeat(1:n_strata, outer = div(n, n_strata))
    strata = vcat(strata, fill(n_strata, n - length(strata)))  # Fill remainder

    # Generate potential outcomes
    Y0 = [strata_effects[s] + randn() for s in strata]
    Y1 = Y0 .+ τ

    # Treatment assignment (balanced within each stratum)
    treatment = Vector{Bool}(undef, n)
    for s in 1:n_strata
        stratum_mask = strata .== s
        n_s = sum(stratum_mask)
        n_treated_s = div(n_s, 2)
        treatment[stratum_mask] =
            vcat(fill(true, n_treated_s), fill(false, n_s - n_treated_s))
    end

    # Observed outcomes
    outcomes = ifelse.(treatment, Y1, Y0)

    return outcomes, treatment, strata, τ
end

"""
Generate RCT data with covariate and known treatment effect.

Parameters:
- n: Sample size
- τ: True ATE
- β_X: Coefficient on covariate X
- seed: Random seed

Returns:
- outcomes, treatment, covariates, τ_true

DGP:
  X ~ N(0, 1)
  Y(0) = β_X * X + ε
  Y(1) = Y(0) + τ
  T ~ Bernoulli(0.5)

True ATE: E[Y(1) - Y(0)] = τ
"""
function dgp_regression_ate(;
    n::Int = 100,
    τ::Float64 = 2.0,
    β_X::Float64 = 0.5,
    seed::Union{Int,Nothing} = nothing,
)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    n_treated = div(n, 2)
    n_control = n - n_treated

    # Covariate
    X = randn(n)
    covariates = reshape(X, n, 1)

    # Potential outcomes
    Y0 = β_X .* X .+ randn(n)
    Y1 = Y0 .+ τ

    # Treatment assignment
    treatment = vcat(fill(true, n_treated), fill(false, n_control))

    # Observed outcomes
    outcomes = ifelse.(treatment, Y1, Y0)

    return outcomes, treatment, covariates, τ
end

"""
Generate RCT data with varying propensity scores and known treatment effect.

Parameters:
- n: Sample size
- τ: True ATE
- propensity_mean: Mean propensity score
- propensity_sd: SD of propensity scores
- seed: Random seed

Returns:
- outcomes, treatment, propensity_scores, τ_true

DGP:
  π_i ~ Beta distributed with mean propensity_mean, sd propensity_sd
  Y(0) ~ N(0, 1)
  Y(1) = Y(0) + τ
  T_i ~ Bernoulli(π_i)

True ATE: E[Y(1) - Y(0)] = τ
"""
function dgp_ipw_ate(;
    n::Int = 100,
    τ::Float64 = 2.0,
    propensity_mean::Float64 = 0.5,
    propensity_sd::Float64 = 0.1,
    seed::Union{Int,Nothing} = nothing,
)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    # Generate propensity scores from Beta distribution
    # Beta(α, β) with mean = α/(α+β), var = αβ/((α+β)²(α+β+1))
    # Solve for α, β given mean and sd
    mean_p = propensity_mean
    var_p = propensity_sd^2
    α = mean_p * (mean_p * (1 - mean_p) / var_p - 1)
    β = (1 - mean_p) * (mean_p * (1 - mean_p) / var_p - 1)

    propensity = rand(Beta(α, β), n)
    propensity = clamp.(propensity, 0.1, 0.9)  # Avoid extreme weights

    # Potential outcomes
    Y0 = randn(n)
    Y1 = Y0 .+ τ

    # Treatment assignment based on propensity
    treatment = Vector{Bool}(rand(n) .< propensity)

    # Observed outcomes
    outcomes = ifelse.(treatment, Y1, Y0)

    return outcomes, treatment, reshape(propensity, n, 1), τ
end

# ============================================================================
# Monte Carlo Validation Functions
# ============================================================================

"""
Run Monte Carlo validation for a single estimator.

Parameters:
- estimator_name: Name of estimator (for display)
- julia_estimator: Julia estimator object
- python_estimator: Python estimator function
- dgp_fn: Data generating process function
- dgp_kwargs: Keyword arguments for DGP
- n_simulations: Number of Monte Carlo simulations
- α: Significance level for confidence intervals
- is_permutation_test: Special handling for PermutationTest (different solution type)

Returns:
- NamedTuple with validation results
"""
function validate_estimator(
    estimator_name::String,
    julia_estimator,
    python_estimator,
    dgp_fn::Function,
    dgp_kwargs::NamedTuple;
    n_simulations::Int = 1000,
    α::Float64 = 0.05,
    is_permutation_test::Bool = false,
)
    println("\n" * "=" ^80)
    println("Monte Carlo Validation: $estimator_name")
    println("=" ^80)
    println("Simulations: $n_simulations")
    println("True ATE: $(dgp_kwargs.τ)")
    println("Significance level: $α")

    # Storage for results
    julia_estimates = Float64[]
    julia_ses = Float64[]
    julia_ci_lowers = Float64[]
    julia_ci_uppers = Float64[]

    python_estimates = Float64[]
    python_ses = Float64[]
    python_ci_lowers = Float64[]
    python_ci_uppers = Float64[]

    τ_true = dgp_kwargs.τ

    for i in 1:n_simulations
        # Generate data with known treatment effect
        dgp_result = dgp_fn(; dgp_kwargs..., seed = i)
        outcomes = dgp_result[1]
        treatment = dgp_result[2]
        extra_data = dgp_result[3:(end - 1)]  # strata, covariates, or propensity
        @assert dgp_result[end] == τ_true  # Verify true ATE

        # Julia estimation
        if length(extra_data) == 0
            # SimpleATE, PermutationTest
            problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = α,))
        elseif isa(extra_data[1], Vector{Int})
            # StratifiedATE
            strata = extra_data[1]
            problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha = α,))
        elseif isa(extra_data[1], Matrix{Float64})
            # RegressionATE or IPWATE
            covariates = extra_data[1]
            problem = RCTProblem(outcomes, treatment, covariates, nothing, (alpha = α,))
        end

        solution = solve(problem, julia_estimator)

        # Handle PermutationTest differently (has observed_statistic instead of estimate)
        if is_permutation_test
            push!(julia_estimates, solution.observed_statistic)
            # Permutation tests don't provide traditional CIs - use dummy values
            # We'll only test bias, not coverage
            push!(julia_ci_lowers, -Inf)
            push!(julia_ci_uppers, Inf)
            push!(julia_ses, std(solution.permutation_distribution))
        else
            push!(julia_estimates, solution.estimate)
            push!(julia_ses, solution.se)
            push!(julia_ci_lowers, solution.ci_lower)
            push!(julia_ci_uppers, solution.ci_upper)
        end

        # Python estimation
        if length(extra_data) == 0
            py_result = python_estimator(outcomes, Int.(treatment))
        elseif isa(extra_data[1], Vector{Int})
            strata = extra_data[1]
            py_result = python_estimator(outcomes, Int.(treatment), strata)
        elseif isa(extra_data[1], Matrix{Float64})
            covariates = extra_data[1]
            py_result = python_estimator(outcomes, Int.(treatment), vec(covariates))
        end

        # Handle Python PermutationTest differently
        if is_permutation_test
            push!(python_estimates, py_result["observed_statistic"])
            # Permutation tests don't provide traditional CIs - use dummy values
            push!(python_ci_lowers, -Inf)
            push!(python_ci_uppers, Inf)
            # SE approximation from permutation distribution (PyCall converts to Julia array)
            perm_dist = py_result["permutation_distribution"]
            push!(python_ses, std(perm_dist))
        else
            push!(python_estimates, py_result["estimate"])
            push!(python_ses, py_result["se"])
            push!(python_ci_lowers, py_result["ci_lower"])
            push!(python_ci_uppers, py_result["ci_upper"])
        end
    end

    # Calculate validation metrics
    function compute_metrics(estimates, ses, ci_lowers, ci_uppers, τ_true)
        bias = mean(estimates) - τ_true
        empirical_sd = std(estimates)
        mean_se = mean(ses)
        se_ratio = empirical_sd / mean_se
        coverage = mean(ci_lowers .<= τ_true .<= ci_uppers)
        return (
            bias = bias,
            empirical_sd = empirical_sd,
            mean_se = mean_se,
            se_ratio = se_ratio,
            coverage = coverage,
        )
    end

    julia_metrics =
        compute_metrics(julia_estimates, julia_ses, julia_ci_lowers, julia_ci_uppers, τ_true)
    python_metrics = compute_metrics(
        python_estimates,
        python_ses,
        python_ci_lowers,
        python_ci_uppers,
        τ_true,
    )

    # Print results
    println("\nJulia Results:")
    println("  Bias: $(round(julia_metrics.bias, digits=4))")
    println("  Empirical SD: $(round(julia_metrics.empirical_sd, digits=4))")
    println("  Mean SE: $(round(julia_metrics.mean_se, digits=4))")
    println("  SE Ratio (SD/SE): $(round(julia_metrics.se_ratio, digits=4))")
    println("  Coverage: $(round(julia_metrics.coverage * 100, digits=2))%")

    println("\nPython Results:")
    println("  Bias: $(round(python_metrics.bias, digits=4))")
    println("  Empirical SD: $(round(python_metrics.empirical_sd, digits=4))")
    println("  Mean SE: $(round(python_metrics.mean_se, digits=4))")
    println("  SE Ratio (SD/SE): $(round(python_metrics.se_ratio, digits=4))")
    println("  Coverage: $(round(python_metrics.coverage * 100, digits=2))%")

    return (julia = julia_metrics, python = python_metrics, τ_true = τ_true)
end

# ============================================================================
# Test Suite
# ============================================================================

@testset "Monte Carlo Ground Truth Validation" begin
    # Validation criteria
    max_bias = 0.05  # |E[τ̂] - τ| < 0.05
    min_coverage = 0.94  # 94% ≤ coverage ≤ 96%
    max_coverage = 0.96
    min_se_ratio = 0.95  # SE should be reasonably accurate (0.95 ≤ SD/SE ≤ 1.05)
    max_se_ratio = 1.05

    n_simulations = 1000
    τ_true = 2.0

    # ========================================================================
    # SimpleATE Validation
    # ========================================================================
    @testset "SimpleATE Ground Truth" begin
        println("\n\n" * "█" ^80)
        println("SIMPLE ATE VALIDATION")
        println("█" ^80)

        results = validate_estimator(
            "SimpleATE",
            SimpleATE(),
            py"simple_ate",
            dgp_constant_ate,
            (n = 100, τ = τ_true, σ_Y0 = 1.0, σ_Y1 = 1.5);
            n_simulations = n_simulations,
        )

        # Julia validation
        @test abs(results.julia.bias) < max_bias
        @test min_coverage <= results.julia.coverage <= max_coverage
        @test min_se_ratio <= results.julia.se_ratio <= max_se_ratio

        # Python validation
        @test abs(results.python.bias) < max_bias
        @test min_coverage <= results.python.coverage <= max_coverage
        @test min_se_ratio <= results.python.se_ratio <= max_se_ratio
    end

    # ========================================================================
    # StratifiedATE Validation
    # ========================================================================
    @testset "StratifiedATE Ground Truth" begin
        println("\n\n" * "█" ^80)
        println("STRATIFIED ATE VALIDATION")
        println("█" ^80)

        results = validate_estimator(
            "StratifiedATE",
            StratifiedATE(),
            py"stratified_ate",
            dgp_stratified_ate,
            (
                n = 100,
                n_strata = 5,
                τ = τ_true,
                strata_effects = Float64[-2.0, -1.0, 0.0, 1.0, 2.0],
            );
            n_simulations = n_simulations,
        )

        # Julia validation
        @test abs(results.julia.bias) < max_bias
        @test min_coverage <= results.julia.coverage <= max_coverage
        @test min_se_ratio <= results.julia.se_ratio <= max_se_ratio

        # Python validation
        @test abs(results.python.bias) < max_bias
        @test min_coverage <= results.python.coverage <= max_coverage
        @test min_se_ratio <= results.python.se_ratio <= max_se_ratio
    end

    # ========================================================================
    # RegressionATE Validation
    # ========================================================================
    @testset "RegressionATE Ground Truth" begin
        println("\n\n" * "█" ^80)
        println("REGRESSION ATE VALIDATION")
        println("█" ^80)

        results = validate_estimator(
            "RegressionATE",
            RegressionATE(),
            py"regression_adjusted_ate",
            dgp_regression_ate,
            (n = 100, τ = τ_true, β_X = 0.5);
            n_simulations = n_simulations,
        )

        # Julia validation
        @test abs(results.julia.bias) < max_bias
        @test min_coverage <= results.julia.coverage <= max_coverage
        # Note: HC3 SE can be slightly conservative, so allow wider range
        @test 0.90 <= results.julia.se_ratio <= 1.10

        # Python validation
        @test abs(results.python.bias) < max_bias
        @test min_coverage <= results.python.coverage <= max_coverage
        @test 0.90 <= results.python.se_ratio <= 1.10
    end

    # ========================================================================
    # PermutationTest Validation
    # ========================================================================
    @testset "PermutationTest Ground Truth" begin
        println("\n\n" * "█" ^80)
        println("PERMUTATION TEST VALIDATION")
        println("█" ^80)

        # Note: PermutationTest doesn't provide traditional CIs
        # We only validate unbiasedness of the observed statistic

        n_sim = 200  # Reduce for permutation test (slower)
        results = validate_estimator(
            "PermutationTest",
            PermutationTest(1000, 42),
            (outcomes, treatment) ->
                py"permutation_test"(outcomes, treatment; n_permutations = 1000, random_seed = 42),
            dgp_constant_ate,
            (n = 50, τ = τ_true, σ_Y0 = 1.0, σ_Y1 = 1.0);
            n_simulations = n_sim,
            is_permutation_test = true,
        )

        # Observed statistic should be unbiased (same as SimpleATE for constant effects)
        @test abs(results.julia.bias) < max_bias
        @test abs(results.python.bias) < max_bias

        # Note: Coverage tests skipped for PermutationTest (no traditional CIs)
        # Future: Could test Type I error rate under H0 (τ=0) instead
    end

    # ========================================================================
    # IPWATE Validation
    # ========================================================================
    @testset "IPWATE Ground Truth" begin
        println("\n\n" * "█" ^80)
        println("IPW ATE VALIDATION")
        println("█" ^80)

        results = validate_estimator(
            "IPWATE",
            IPWATE(),
            py"ipw_ate",
            dgp_ipw_ate,
            (n = 100, τ = τ_true, propensity_mean = 0.5, propensity_sd = 0.1);
            n_simulations = n_simulations,
        )

        # Julia validation
        @test abs(results.julia.bias) < max_bias
        # IPW can have slightly lower coverage with varying propensities (93-96% acceptable)
        @test 0.93 <= results.julia.coverage <= max_coverage
        # IPW can have larger SE variability with varying propensities
        @test 0.90 <= results.julia.se_ratio <= 1.15

        # Python validation
        @test abs(results.python.bias) < max_bias
        @test 0.93 <= results.python.coverage <= max_coverage
        @test 0.90 <= results.python.se_ratio <= 1.15
    end
end
