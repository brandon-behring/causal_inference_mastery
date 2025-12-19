"""
Data Generating Processes (DGPs) for CATE Monte Carlo validation.

Provides DGPs for validating heterogeneous treatment effect estimators:
- S-Learner, T-Learner, X-Learner, R-Learner
- Double Machine Learning (DML)

All DGPs have known true ATE and CATE functions for validation.

References:
    - Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects"
    - Nie & Wager (2021). "Quasi-oracle estimation of heterogeneous treatment effects"
    - Chernozhukov et al. (2018). "Double/debiased machine learning"
"""

using Random
using Statistics
using LinearAlgebra

# =============================================================================
# Data Container Type
# =============================================================================

"""
Container for CATE simulation data with known ground truth.

# Fields
- `Y::Vector{T}`: Outcome variable
- `T::Vector{Bool}`: Binary treatment indicator
- `X::Matrix{T}`: Covariates
- `true_ate::T`: True average treatment effect
- `true_cate::Vector{T}`: True conditional treatment effects τ(X) for each observation
- `propensity::Vector{T}`: True propensity scores P(T=1|X)
- `n::Int`: Sample size
- `p::Int`: Number of covariates
"""
struct CATEData{T<:Real}
    Y::Vector{T}
    treatment::Vector{Bool}
    X::Matrix{T}
    true_ate::T
    true_cate::Vector{T}
    propensity::Vector{T}
    n::Int
    p::Int
end

# =============================================================================
# Constant Effect DGP (Homogeneous)
# =============================================================================

"""
    dgp_constant_effect(; kwargs...) -> CATEData

CATE DGP with constant treatment effect (no heterogeneity).

Model:
    Y(0) = X @ β + ε
    Y(1) = Y(0) + τ (constant)
    T ~ Bernoulli(expit(propensity_strength * X[:, 1]))

# Arguments
- `n::Int=1000`: Sample size
- `true_ate::Float64=2.0`: True constant treatment effect
- `p::Int=5`: Number of covariates
- `propensity_strength::Float64=0.5`: Strength of propensity model
- `seed::Int=42`: Random seed
"""
function dgp_constant_effect(;
    n::Int=1000,
    true_ate::Float64=2.0,
    p::Int=5,
    propensity_strength::Float64=0.5,
    seed::Int=42
)
    Random.seed!(seed)

    # Generate covariates
    X = randn(n, p)

    # Propensity model: logistic on X[:, 1]
    logit = propensity_strength .* X[:, 1]
    propensity = 1.0 ./ (1.0 .+ exp.(-logit))
    T = [rand() < prop for prop in propensity]

    # Outcome model: linear in X
    beta = randn(p) .* 0.5
    Y0 = X * beta .+ randn(n)

    # Constant CATE
    true_cate = fill(true_ate, n)

    # Observed outcome
    Y = Y0 .+ T .* true_ate

    return CATEData(
        Y,
        T,
        X,
        true_ate,
        true_cate,
        propensity,
        n,
        p
    )
end

# =============================================================================
# Linear Heterogeneity DGP
# =============================================================================

"""
    dgp_linear_heterogeneity(; kwargs...) -> CATEData

CATE DGP with linear heterogeneity in first covariate.

Model:
    τ(X) = base_effect + het_coef * X[:, 1]
    Y(0) = X @ β + ε
    Y(1) = Y(0) + τ(X)

# Arguments
- `n::Int=1000`: Sample size
- `base_effect::Float64=2.0`: Intercept of CATE function
- `het_coef::Float64=1.0`: Coefficient on X[:, 1] for heterogeneity
- `p::Int=5`: Number of covariates
- `propensity_strength::Float64=0.5`: Strength of propensity model
- `seed::Int=42`: Random seed
"""
function dgp_linear_heterogeneity(;
    n::Int=1000,
    base_effect::Float64=2.0,
    het_coef::Float64=1.0,
    p::Int=5,
    propensity_strength::Float64=0.5,
    seed::Int=42
)
    Random.seed!(seed)

    # Generate covariates
    X = randn(n, p)

    # Propensity model
    logit = propensity_strength .* X[:, 1]
    propensity = 1.0 ./ (1.0 .+ exp.(-logit))
    T = [rand() < prop for prop in propensity]

    # Outcome model
    beta = randn(p) .* 0.5
    Y0 = X * beta .+ randn(n)

    # Linear CATE
    true_cate = base_effect .+ het_coef .* X[:, 1]
    true_ate = mean(true_cate)

    # Observed outcome
    Y = Y0 .+ T .* true_cate

    return CATEData(
        Y,
        T,
        X,
        true_ate,
        true_cate,
        propensity,
        n,
        p
    )
end

# =============================================================================
# Nonlinear Heterogeneity DGP
# =============================================================================

"""
    dgp_nonlinear_heterogeneity(; kwargs...) -> CATEData

CATE DGP with nonlinear (sinusoidal) heterogeneity.

Model:
    τ(X) = base_effect * (1 + amplitude * sin(X[:, 1]))
    Y(0) = X @ β + ε
    Y(1) = Y(0) + τ(X)

# Arguments
- `n::Int=1000`: Sample size
- `base_effect::Float64=2.0`: Base treatment effect
- `amplitude::Float64=0.5`: Amplitude of sinusoidal modulation
- `p::Int=5`: Number of covariates
- `propensity_strength::Float64=0.5`: Strength of propensity model
- `seed::Int=42`: Random seed
"""
function dgp_nonlinear_heterogeneity(;
    n::Int=1000,
    base_effect::Float64=2.0,
    amplitude::Float64=0.5,
    p::Int=5,
    propensity_strength::Float64=0.5,
    seed::Int=42
)
    Random.seed!(seed)

    # Generate covariates
    X = randn(n, p)

    # Propensity model
    logit = propensity_strength .* X[:, 1]
    propensity = 1.0 ./ (1.0 .+ exp.(-logit))
    T = [rand() < prop for prop in propensity]

    # Outcome model
    beta = randn(p) .* 0.5
    Y0 = X * beta .+ randn(n)

    # Nonlinear CATE
    true_cate = base_effect .* (1.0 .+ amplitude .* sin.(X[:, 1]))
    true_ate = mean(true_cate)

    # Observed outcome
    Y = Y0 .+ T .* true_cate

    return CATEData(
        Y,
        T,
        X,
        true_ate,
        true_cate,
        propensity,
        n,
        p
    )
end

# =============================================================================
# Complex Heterogeneity DGP (Step + Linear)
# =============================================================================

"""
    dgp_complex_heterogeneity(; kwargs...) -> CATEData

CATE DGP with complex heterogeneity (step function + linear).

Model:
    τ(X) = base_effect + step_effect * I(X[:, 0] > 0) + linear_coef * X[:, 1]
    Y(0) = X @ β + ε
    Y(1) = Y(0) + τ(X)

# Arguments
- `n::Int=1000`: Sample size
- `base_effect::Float64=1.0`: Baseline effect
- `step_effect::Float64=2.0`: Additional effect when X[:, 1] > 0
- `linear_coef::Float64=0.5`: Coefficient on X[:, 2]
- `p::Int=5`: Number of covariates (must be >= 2)
- `propensity_strength::Float64=0.5`: Strength of propensity model
- `seed::Int=42`: Random seed
"""
function dgp_complex_heterogeneity(;
    n::Int=1000,
    base_effect::Float64=1.0,
    step_effect::Float64=2.0,
    linear_coef::Float64=0.5,
    p::Int=5,
    propensity_strength::Float64=0.5,
    seed::Int=42
)
    if p < 2
        throw(ArgumentError("p must be >= 2 for complex heterogeneity DGP"))
    end

    Random.seed!(seed)

    # Generate covariates
    X = randn(n, p)

    # Propensity model
    logit = propensity_strength .* X[:, 1]
    propensity = 1.0 ./ (1.0 .+ exp.(-logit))
    T = [rand() < prop for prop in propensity]

    # Outcome model
    beta = randn(p) .* 0.5
    Y0 = X * beta .+ randn(n)

    # Complex CATE (step + linear)
    true_cate = base_effect .+ step_effect .* (X[:, 1] .> 0) .+ linear_coef .* X[:, 2]
    true_ate = mean(true_cate)

    # Observed outcome
    Y = Y0 .+ T .* true_cate

    return CATEData(
        Y,
        T,
        X,
        true_ate,
        true_cate,
        propensity,
        n,
        p
    )
end

# =============================================================================
# High-Dimensional DGP
# =============================================================================

"""
    dgp_high_dimensional(; kwargs...) -> CATEData

CATE DGP with high-dimensional covariates (sparse true model).

Model:
    τ(X) = true_ate + X[:, 1:n_relevant] @ het_coefs
    Y(0) = X[:, 1:n_relevant] @ β + ε
    T ~ Bernoulli(expit(propensity_strength * X[:, 1]))

Only first n_relevant covariates matter; rest are noise.

# Arguments
- `n::Int=500`: Sample size
- `p::Int=50`: Total number of covariates
- `true_ate::Float64=2.0`: Average treatment effect
- `n_relevant::Int=5`: Number of relevant covariates
- `propensity_strength::Float64=0.3`: Strength of propensity model
- `seed::Int=42`: Random seed
"""
function dgp_high_dimensional(;
    n::Int=500,
    p::Int=50,
    true_ate::Float64=2.0,
    n_relevant::Int=5,
    propensity_strength::Float64=0.3,
    seed::Int=42
)
    if n_relevant > p
        throw(ArgumentError("n_relevant must be <= p"))
    end

    Random.seed!(seed)

    # Generate covariates
    X = randn(n, p)

    # Propensity model (only depends on X[:, 1])
    logit = propensity_strength .* X[:, 1]
    propensity = 1.0 ./ (1.0 .+ exp.(-logit))
    T = [rand() < prop for prop in propensity]

    # Outcome model (sparse)
    beta = zeros(p)
    beta[1:n_relevant] = randn(n_relevant) .* 0.5
    Y0 = X * beta .+ randn(n)

    # CATE with sparse heterogeneity
    het_coefs = zeros(p)
    het_coefs[1:n_relevant] = randn(n_relevant) .* 0.3
    true_cate = true_ate .+ X * het_coefs
    actual_ate = mean(true_cate)

    # Observed outcome
    Y = Y0 .+ T .* true_cate

    return CATEData(
        Y,
        T,
        X,
        actual_ate,
        true_cate,
        propensity,
        n,
        p
    )
end

# =============================================================================
# Imbalanced Treatment DGP
# =============================================================================

"""
    dgp_imbalanced_treatment(; kwargs...) -> CATEData

CATE DGP with imbalanced treatment assignment (few treated).

Model:
    τ(X) = true_ate (constant)
    T ~ Bernoulli(treatment_prob)  # Independent of X
    Y(0) = X @ β + ε
    Y(1) = Y(0) + τ

# Arguments
- `n::Int=1000`: Sample size
- `true_ate::Float64=2.0`: True treatment effect
- `treatment_prob::Float64=0.1`: Probability of treatment
- `p::Int=5`: Number of covariates
- `seed::Int=42`: Random seed
"""
function dgp_imbalanced_treatment(;
    n::Int=1000,
    true_ate::Float64=2.0,
    treatment_prob::Float64=0.1,
    p::Int=5,
    seed::Int=42
)
    Random.seed!(seed)

    # Generate covariates
    X = randn(n, p)

    # Constant propensity (imbalanced)
    propensity = fill(treatment_prob, n)
    T = [rand() < treatment_prob for _ in 1:n]

    # Outcome model
    beta = randn(p) .* 0.5
    Y0 = X * beta .+ randn(n)

    # Constant CATE
    true_cate = fill(true_ate, n)

    # Observed outcome
    Y = Y0 .+ T .* true_ate

    return CATEData(
        Y,
        T,
        X,
        true_ate,
        true_cate,
        propensity,
        n,
        p
    )
end

# =============================================================================
# Strong Confounding DGP
# =============================================================================

"""
    dgp_strong_confounding(; kwargs...) -> CATEData

CATE DGP with strong selection on observables (high propensity correlation).

Model:
    T ~ Bernoulli(expit(confounding_strength * (X[:, 1] + X[:, 2])))
    Y(0) = 2*X[:, 1] + X[:, 2] + ε
    τ(X) = base_effect + 0.5*X[:, 1]
    Y(1) = Y(0) + τ(X)

Strong confounding means treatment assignment strongly related to outcomes.

# Arguments
- `n::Int=1000`: Sample size
- `base_effect::Float64=2.0`: Base treatment effect
- `confounding_strength::Float64=1.0`: Strength of confounding
- `p::Int=5`: Number of covariates
- `seed::Int=42`: Random seed
"""
function dgp_strong_confounding(;
    n::Int=1000,
    base_effect::Float64=2.0,
    confounding_strength::Float64=1.0,
    p::Int=5,
    seed::Int=42
)
    Random.seed!(seed)

    # Generate covariates
    X = randn(n, p)

    # Strong selection on X[:, 1] and X[:, 2]
    logit = confounding_strength .* (X[:, 1] .+ X[:, 2])
    propensity = 1.0 ./ (1.0 .+ exp.(-logit))
    T = [rand() < prop for prop in propensity]

    # Outcome model: Y(0) depends on same variables as propensity
    Y0 = 2.0 .* X[:, 1] .+ X[:, 2] .+ randn(n)

    # CATE with heterogeneity on X[:, 1]
    true_cate = base_effect .+ 0.5 .* X[:, 1]
    true_ate = mean(true_cate)

    # Observed outcome
    Y = Y0 .+ T .* true_cate

    return CATEData(
        Y,
        T,
        X,
        true_ate,
        true_cate,
        propensity,
        n,
        p
    )
end
