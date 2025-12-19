#=
Data Generating Processes for Sensitivity Analysis Monte Carlo Validation.

Provides DGPs for validating:
1. E-value formula accuracy
2. Rosenbaum bounds gamma_critical detection

Session 69: Julia Sensitivity Validation
=#

using Random
using Statistics

# =============================================================================
# E-Value DGPs
# =============================================================================

"""
    dgp_evalue_known_rr(; n=500, true_rr=2.0, seed=42)

Generate binary outcome data with known risk ratio.

DGP:
    T ~ Bernoulli(0.5)
    p0 = 0.2 (baseline risk)
    p1 = p0 * true_rr (treated risk)
    Y ~ Bernoulli(p)
"""
function dgp_evalue_known_rr(;
    n::Int=500,
    true_rr::Float64=2.0,
    seed::Int=42,
)
    Random.seed!(seed)

    # Treatment assignment
    treatment = rand(n) .< 0.5

    # Outcome probabilities
    p0 = 0.2
    p1 = min(p0 * true_rr, 0.99)

    probs = ifelse.(treatment, p1, p0)
    outcomes = rand(n) .< probs

    return (
        outcomes=Float64.(outcomes),
        treatment=Float64.(treatment),
        true_rr=true_rr,
    )
end

"""
    dgp_evalue_smd(; n=500, true_smd=0.5, seed=42)

Generate continuous outcome data with known standardized mean difference.

DGP:
    T ~ Bernoulli(0.5)
    Y | T=0 ~ N(0, 1)
    Y | T=1 ~ N(true_smd, 1)
"""
function dgp_evalue_smd(;
    n::Int=500,
    true_smd::Float64=0.5,
    seed::Int=42,
)
    Random.seed!(seed)

    treatment = rand(n) .< 0.5

    outcomes = ifelse.(
        treatment,
        randn(n) .+ true_smd,
        randn(n),
    )

    return (
        outcomes=outcomes,
        treatment=Float64.(treatment),
        true_smd=true_smd,
    )
end

# =============================================================================
# Rosenbaum Bounds DGPs
# =============================================================================

"""
    dgp_matched_pairs_no_confounding(; n_pairs=50, true_effect=2.0, noise_sd=1.0, seed=42)

Generate matched pairs with no hidden confounding.

DGP:
    Y_control ~ N(0, noise_sd)
    Y_treated = Y_control + true_effect + N(0, noise_sd/2)
"""
function dgp_matched_pairs_no_confounding(;
    n_pairs::Int=50,
    true_effect::Float64=2.0,
    noise_sd::Float64=1.0,
    seed::Int=42,
)
    Random.seed!(seed)

    control_outcomes = randn(n_pairs) .* noise_sd
    treated_outcomes = control_outcomes .+ true_effect .+ randn(n_pairs) .* (noise_sd * 0.5)

    return (
        treated=treated_outcomes,
        control=control_outcomes,
        true_effect=true_effect,
    )
end

"""
    dgp_matched_pairs_weak_effect(; n_pairs=50, true_effect=0.3, noise_sd=2.0, seed=42)

Generate matched pairs with weak treatment effect (high noise).
"""
function dgp_matched_pairs_weak_effect(;
    n_pairs::Int=50,
    true_effect::Float64=0.3,
    noise_sd::Float64=2.0,
    seed::Int=42,
)
    Random.seed!(seed)

    control_outcomes = randn(n_pairs) .* noise_sd
    treated_outcomes = control_outcomes .+ true_effect .+ randn(n_pairs) .* (noise_sd * 0.5)

    return (
        treated=treated_outcomes,
        control=control_outcomes,
        true_effect=true_effect,
    )
end

"""
    dgp_matched_pairs_strong_effect(; n_pairs=50, true_effect=5.0, noise_sd=1.0, seed=42)

Generate matched pairs with strong treatment effect.
"""
function dgp_matched_pairs_strong_effect(;
    n_pairs::Int=50,
    true_effect::Float64=5.0,
    noise_sd::Float64=1.0,
    seed::Int=42,
)
    Random.seed!(seed)

    control_outcomes = randn(n_pairs) .* noise_sd
    treated_outcomes = control_outcomes .+ true_effect .+ randn(n_pairs) .* (noise_sd * 0.5)

    return (
        treated=treated_outcomes,
        control=control_outcomes,
        true_effect=true_effect,
    )
end

"""
    dgp_matched_pairs_null_effect(; n_pairs=50, noise_sd=1.0, seed=42)

Generate matched pairs with no treatment effect.
"""
function dgp_matched_pairs_null_effect(;
    n_pairs::Int=50,
    noise_sd::Float64=1.0,
    seed::Int=42,
)
    Random.seed!(seed)

    control_outcomes = randn(n_pairs) .* noise_sd
    treated_outcomes = randn(n_pairs) .* noise_sd  # No effect

    return (
        treated=treated_outcomes,
        control=control_outcomes,
        true_effect=0.0,
    )
end
