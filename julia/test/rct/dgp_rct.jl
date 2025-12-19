"""
Data Generating Processes (DGPs) for RCT Monte Carlo validation.

Provides DGPs for validating Randomized Controlled Trial estimators:
- SimpleATE: Difference in means
- StratifiedATE: Block randomization
- RegressionATE: Covariate adjustment
- IPWATE: Inverse probability weighting

All DGPs have known true ATEs for validation.

References:
    - Imbens & Rubin (2015). "Causal Inference for Statistics, Social, and Biomedical Sciences"
    - Athey & Imbens (2017). "The Econometrics of Randomized Experiments"
"""

using Random
using Statistics

# =============================================================================
# Data Container Types
# =============================================================================

"""
Container for simple RCT simulation data with known ground truth.
"""
struct RCTData{T<:Real}
    outcomes::Vector{T}
    treatment::Union{Vector{Bool}, BitVector}
    true_ate::T
    n_treated::Int
    n_control::Int
end

"""
Container for RCT data with covariates (for RegressionATE).
"""
struct RCTDataWithCovariates{T<:Real}
    outcomes::Vector{T}
    treatment::Union{Vector{Bool}, BitVector}
    covariates::Matrix{T}
    true_ate::T
    n_treated::Int
    n_control::Int
    covariate_names::Vector{Symbol}
end

"""
Container for stratified RCT data (for StratifiedATE).
"""
struct StratifiedRCTData{T<:Real}
    outcomes::Vector{T}
    treatment::Union{Vector{Bool}, BitVector}
    strata::Vector{Int}
    true_ate::T
    n_treated::Int
    n_control::Int
    n_strata::Int
    strata_ate::Vector{T}  # True ATE within each stratum
end

"""
Container for RCT data with propensity scores (for IPWATE).
"""
struct RCTDataWithPropensity{T<:Real}
    outcomes::Vector{T}
    treatment::Union{Vector{Bool}, BitVector}
    true_propensity::Vector{T}  # True P(T=1|X)
    true_ate::T
    n_treated::Int
    n_control::Int
end

# =============================================================================
# Simple RCT DGPs
# =============================================================================

"""
    dgp_rct_simple(; kwargs...) -> RCTData

Simple RCT DGP with constant treatment effect.

DGP:
    Y_i = μ + τ·T_i + ε_i

    where:
    - μ = baseline (default 0)
    - τ = true_ate (treatment effect)
    - T_i ~ Bernoulli(p_treat) (randomized treatment)
    - ε_i ~ N(0, σ²) (idiosyncratic noise)

# Arguments
- `n::Int=200`: Total sample size
- `p_treat::Float64=0.5`: Probability of treatment assignment
- `true_ate::Float64=2.0`: True average treatment effect
- `baseline::Float64=0.0`: Control group mean
- `sigma::Float64=1.0`: Standard deviation of errors
- `seed::Int=42`: Random seed
"""
function dgp_rct_simple(;
    n::Int=200,
    p_treat::Float64=0.5,
    true_ate::Float64=2.0,
    baseline::Float64=0.0,
    sigma::Float64=1.0,
    seed::Int=42
)
    Random.seed!(seed)

    # Randomized treatment assignment
    treatment = rand(n) .< p_treat

    n_treated = sum(treatment)
    n_control = n - n_treated

    # Potential outcomes framework:
    # Y(0) = baseline + ε
    # Y(1) = baseline + τ + ε
    # Observed: Y = T·Y(1) + (1-T)·Y(0)
    errors = randn(n) .* sigma
    outcomes = baseline .+ true_ate .* treatment .+ errors

    return RCTData(
        outcomes,
        treatment,
        true_ate,
        n_treated,
        n_control
    )
end

"""
    dgp_rct_no_effect(; kwargs...) -> RCTData

RCT DGP with true null effect (τ = 0).

Tests Type I error control - estimator should not reject null.
"""
function dgp_rct_no_effect(;
    n::Int=200,
    p_treat::Float64=0.5,
    baseline::Float64=5.0,
    sigma::Float64=1.0,
    seed::Int=42
)
    return dgp_rct_simple(
        n=n,
        p_treat=p_treat,
        true_ate=0.0,
        baseline=baseline,
        sigma=sigma,
        seed=seed
    )
end

"""
    dgp_rct_heteroskedastic(; kwargs...) -> RCTData

RCT DGP with heteroskedastic errors.

Different error variances for treated vs control.
Tests robustness of standard errors.

DGP:
    Y_i = μ + τ·T_i + ε_i
    ε_i ~ N(0, σ_treated²) if T_i = 1
    ε_i ~ N(0, σ_control²) if T_i = 0
"""
function dgp_rct_heteroskedastic(;
    n::Int=200,
    p_treat::Float64=0.5,
    true_ate::Float64=2.0,
    baseline::Float64=0.0,
    sigma_treated::Float64=2.0,
    sigma_control::Float64=1.0,
    seed::Int=42
)
    Random.seed!(seed)

    treatment = rand(n) .< p_treat
    n_treated = sum(treatment)
    n_control = n - n_treated

    # Heteroskedastic errors
    errors = Vector{Float64}(undef, n)
    for i in 1:n
        if treatment[i]
            errors[i] = randn() * sigma_treated
        else
            errors[i] = randn() * sigma_control
        end
    end

    outcomes = baseline .+ true_ate .* treatment .+ errors

    return RCTData(
        outcomes,
        treatment,
        true_ate,
        n_treated,
        n_control
    )
end

"""
    dgp_rct_heavy_tails(; kwargs...) -> RCTData

RCT DGP with heavy-tailed (t-distributed) errors.

Tests robustness to non-normal errors.

DGP:
    Y_i = μ + τ·T_i + ε_i
    ε_i ~ t_df · scale (scaled t-distribution)
"""
function dgp_rct_heavy_tails(;
    n::Int=200,
    p_treat::Float64=0.5,
    true_ate::Float64=2.0,
    baseline::Float64=0.0,
    df::Float64=3.0,  # Degrees of freedom (lower = heavier tails)
    scale::Float64=1.0,
    seed::Int=42
)
    Random.seed!(seed)

    treatment = rand(n) .< p_treat
    n_treated = sum(treatment)
    n_control = n - n_treated

    # t-distributed errors (using normal/chi-square ratio)
    # t_df = Z / sqrt(χ²_df / df)
    z = randn(n)
    chi2 = sum(randn(n, Int(ceil(df))).^2, dims=2)[:] ./ df
    errors = scale .* z ./ sqrt.(chi2)

    outcomes = baseline .+ true_ate .* treatment .+ errors

    return RCTData(
        outcomes,
        treatment,
        true_ate,
        n_treated,
        n_control
    )
end

"""
    dgp_rct_unequal_groups(; kwargs...) -> RCTData

RCT DGP with unequal treatment/control group sizes.

Tests robustness to imbalanced randomization.
"""
function dgp_rct_unequal_groups(;
    n::Int=200,
    p_treat::Float64=0.2,  # Unequal by default
    true_ate::Float64=2.0,
    baseline::Float64=0.0,
    sigma::Float64=1.0,
    seed::Int=42
)
    return dgp_rct_simple(
        n=n,
        p_treat=p_treat,
        true_ate=true_ate,
        baseline=baseline,
        sigma=sigma,
        seed=seed
    )
end

# =============================================================================
# Stratified RCT DGPs (for StratifiedATE)
# =============================================================================

"""
    dgp_rct_stratified(; kwargs...) -> StratifiedRCTData

Stratified RCT DGP with block randomization.

Different baseline outcomes by stratum, constant treatment effect.

DGP:
    Y_i = μ_s(i) + τ·T_i + ε_i

    where:
    - μ_s = stratum-specific baseline
    - τ = constant treatment effect across strata
"""
function dgp_rct_stratified(;
    n_per_stratum::Int=50,
    n_strata::Int=4,
    p_treat::Float64=0.5,
    true_ate::Float64=2.0,
    stratum_means::Vector{Float64}=Float64[0.0, 2.0, 5.0, 8.0],
    sigma::Float64=1.0,
    seed::Int=42
)
    Random.seed!(seed)

    if length(stratum_means) != n_strata
        throw(ArgumentError("Length of stratum_means must equal n_strata"))
    end

    n = n_per_stratum * n_strata

    outcomes = Float64[]
    treatment = Bool[]
    strata = Int[]

    for s in 1:n_strata
        # Block randomization within stratum
        t_stratum = rand(n_per_stratum) .< p_treat
        errors = randn(n_per_stratum) .* sigma
        y_stratum = stratum_means[s] .+ true_ate .* t_stratum .+ errors

        append!(outcomes, y_stratum)
        append!(treatment, t_stratum)
        append!(strata, fill(s, n_per_stratum))
    end

    n_treated = sum(treatment)
    n_control = n - n_treated

    # True ATE is constant across strata in this DGP
    strata_ate = fill(true_ate, n_strata)

    return StratifiedRCTData(
        outcomes,
        treatment,
        strata,
        true_ate,
        n_treated,
        n_control,
        n_strata,
        strata_ate
    )
end

"""
    dgp_rct_stratified_heterogeneous(; kwargs...) -> StratifiedRCTData

Stratified RCT with heterogeneous treatment effects across strata.

DGP:
    Y_i = μ_s(i) + τ_s(i)·T_i + ε_i

    where τ_s varies by stratum.
"""
function dgp_rct_stratified_heterogeneous(;
    n_per_stratum::Int=50,
    n_strata::Int=4,
    p_treat::Float64=0.5,
    stratum_means::Vector{Float64}=Float64[0.0, 2.0, 5.0, 8.0],
    stratum_effects::Vector{Float64}=Float64[1.0, 2.0, 3.0, 4.0],
    sigma::Float64=1.0,
    seed::Int=42
)
    Random.seed!(seed)

    if length(stratum_means) != n_strata || length(stratum_effects) != n_strata
        throw(ArgumentError("Length of stratum vectors must equal n_strata"))
    end

    n = n_per_stratum * n_strata

    outcomes = Float64[]
    treatment = Bool[]
    strata = Int[]

    for s in 1:n_strata
        t_stratum = rand(n_per_stratum) .< p_treat
        errors = randn(n_per_stratum) .* sigma
        y_stratum = stratum_means[s] .+ stratum_effects[s] .* t_stratum .+ errors

        append!(outcomes, y_stratum)
        append!(treatment, t_stratum)
        append!(strata, fill(s, n_per_stratum))
    end

    n_treated = sum(treatment)
    n_control = n - n_treated

    # Overall ATE is weighted average of stratum effects
    true_ate = mean(stratum_effects)  # Equal weights if equal stratum sizes

    return StratifiedRCTData(
        outcomes,
        treatment,
        strata,
        true_ate,
        n_treated,
        n_control,
        n_strata,
        stratum_effects
    )
end

# =============================================================================
# RCT with Covariates (for RegressionATE)
# =============================================================================

"""
    dgp_rct_with_covariates(; kwargs...) -> RCTDataWithCovariates

RCT DGP with covariates that predict outcomes but are balanced by randomization.

DGP:
    Y_i = μ + τ·T_i + X_i'β + ε_i

    where:
    - X_i = covariates (balanced by randomization)
    - β = covariate effects on outcome
    - Covariates reduce residual variance, improving precision
"""
function dgp_rct_with_covariates(;
    n::Int=200,
    p_treat::Float64=0.5,
    true_ate::Float64=2.0,
    baseline::Float64=0.0,
    n_covariates::Int=3,
    covariate_effects::Vector{Float64}=Float64[1.0, 0.5, 0.25],
    sigma::Float64=1.0,
    seed::Int=42
)
    Random.seed!(seed)

    if length(covariate_effects) != n_covariates
        covariate_effects = ones(n_covariates)  # Default to unit effects
    end

    # Randomized treatment
    treatment = rand(n) .< p_treat
    n_treated = sum(treatment)
    n_control = n - n_treated

    # Generate covariates (independent of treatment by randomization)
    covariates = randn(n, n_covariates)

    # Outcome with covariate effects
    covariate_term = covariates * covariate_effects
    errors = randn(n) .* sigma
    outcomes = baseline .+ true_ate .* treatment .+ covariate_term .+ errors

    covariate_names = [Symbol("X$i") for i in 1:n_covariates]

    return RCTDataWithCovariates(
        outcomes,
        treatment,
        covariates,
        true_ate,
        n_treated,
        n_control,
        covariate_names
    )
end

"""
    dgp_rct_high_variance_covariates(; kwargs...) -> RCTDataWithCovariates

RCT where covariates explain most of the outcome variance.

Covariate adjustment should dramatically improve precision.
"""
function dgp_rct_high_variance_covariates(;
    n::Int=200,
    p_treat::Float64=0.5,
    true_ate::Float64=2.0,
    baseline::Float64=0.0,
    n_covariates::Int=3,
    covariate_effects::Vector{Float64}=Float64[3.0, 2.0, 1.0],  # Strong effects
    sigma::Float64=0.5,  # Low residual variance
    seed::Int=42
)
    return dgp_rct_with_covariates(
        n=n,
        p_treat=p_treat,
        true_ate=true_ate,
        baseline=baseline,
        n_covariates=n_covariates,
        covariate_effects=covariate_effects,
        sigma=sigma,
        seed=seed
    )
end

# =============================================================================
# RCT with Propensity Variation (for IPWATE)
# =============================================================================

"""
    dgp_rct_known_propensity(; kwargs...) -> RCTDataWithPropensity

RCT with known (constant) propensity scores.

Simple case for IPWATE validation - propensity is constant p_treat.
"""
function dgp_rct_known_propensity(;
    n::Int=200,
    p_treat::Float64=0.5,
    true_ate::Float64=2.0,
    baseline::Float64=0.0,
    sigma::Float64=1.0,
    seed::Int=42
)
    Random.seed!(seed)

    treatment = rand(n) .< p_treat
    n_treated = sum(treatment)
    n_control = n - n_treated

    errors = randn(n) .* sigma
    outcomes = baseline .+ true_ate .* treatment .+ errors

    # Constant propensity in a pure RCT
    true_propensity = fill(p_treat, n)

    return RCTDataWithPropensity(
        outcomes,
        treatment,
        true_propensity,
        true_ate,
        n_treated,
        n_control
    )
end

"""
    dgp_rct_varying_propensity(; kwargs...) -> RCTDataWithPropensity

RCT with strata-varying propensity (block randomization with different probabilities).

DGP:
    P(T=1|stratum) varies by stratum
    Y_i = μ + τ·T_i + ε_i (homogeneous effect)
"""
function dgp_rct_varying_propensity(;
    n::Int=200,
    stratum_propensities::Vector{Float64}=Float64[0.3, 0.5, 0.7],
    true_ate::Float64=2.0,
    baseline::Float64=0.0,
    sigma::Float64=1.0,
    seed::Int=42
)
    Random.seed!(seed)

    n_strata = length(stratum_propensities)
    n_per_stratum = div(n, n_strata)

    outcomes = Float64[]
    treatment = Bool[]
    true_propensity = Float64[]

    for (s, p) in enumerate(stratum_propensities)
        t_stratum = rand(n_per_stratum) .< p
        errors = randn(n_per_stratum) .* sigma
        y_stratum = baseline .+ true_ate .* t_stratum .+ errors

        append!(outcomes, y_stratum)
        append!(treatment, t_stratum)
        append!(true_propensity, fill(p, n_per_stratum))
    end

    n_treated = sum(treatment)
    n_control = length(treatment) - n_treated

    return RCTDataWithPropensity(
        outcomes,
        treatment,
        true_propensity,
        true_ate,
        n_treated,
        n_control
    )
end
