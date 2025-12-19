"""
Data Generating Processes (DGPs) for Observational Monte Carlo validation.

Provides DGPs for validating IPW and Doubly Robust estimators:
- ObservationalIPW
- DoublyRobust

All DGPs have known true ATE for validation under selection on observables.

Key Assumptions for Identification:
1. Unconfoundedness: Y(0), Y(1) ⟂ T | X
2. Positivity: 0 < P(T=1|X) < 1
3. SUTVA: No interference

References:
    - Rosenbaum & Rubin (1983). The central role of the propensity score.
    - Bang & Robins (2005). Doubly robust estimation.
    - Austin & Stuart (2015). Moving towards best practice with IPTW.
"""

using Random
using Statistics
using LinearAlgebra

# =============================================================================
# Data Container Type
# =============================================================================

"""
Container for observational simulation data with known ground truth.

# Fields
- `Y::Vector{T}`: Observed outcome
- `treatment::Vector{Bool}`: Binary treatment indicator
- `X::Matrix{T}`: Covariates (confounders)
- `true_ate::T`: True average treatment effect
- `propensity::Vector{T}`: True propensity scores P(T=1|X)
- `n::Int`: Sample size
- `p::Int`: Number of covariates
"""
struct ObservationalData{T<:Real}
    Y::Vector{T}
    treatment::Vector{Bool}
    X::Matrix{T}
    true_ate::T
    propensity::Vector{T}
    n::Int
    p::Int
end

# =============================================================================
# Simple Confounded DGP
# =============================================================================

"""
    dgp_observational_simple(; kwargs...) -> ObservationalData

Simple observational DGP with moderate confounding.

Model:
    X ~ N(0, I_p)
    e(X) = expit(confounding_strength * X[:, 1])
    T ~ Bernoulli(e(X))
    Y(0) = β₀ + X[:, 1] + 0.5*X[:, 2] + ε
    Y(1) = Y(0) + true_ate
    Y = Y(0) + T * true_ate

# Arguments
- `n::Int=500`: Sample size
- `p::Int=3`: Number of covariates
- `true_ate::Float64=2.0`: True average treatment effect
- `confounding_strength::Float64=0.5`: Propensity dependence on X[:, 1]
- `seed::Int=42`: Random seed
"""
function dgp_observational_simple(;
    n::Int=500,
    p::Int=3,
    true_ate::Float64=2.0,
    confounding_strength::Float64=0.5,
    seed::Int=42
)
    Random.seed!(seed)

    # Generate covariates
    X = randn(n, p)

    # Propensity model: logistic on X[:, 1]
    logit = confounding_strength .* X[:, 1]
    propensity = 1.0 ./ (1.0 .+ exp.(-logit))

    # Treatment assignment
    T = [rand() < prop for prop in propensity]

    # Ensure both treatment groups exist
    n_treated = sum(T)
    if n_treated == 0 || n_treated == n
        # Force at least one in each group
        T[1] = true
        T[end] = false
    end

    # Outcome model (confounded by X[:, 1])
    Y0 = 1.0 .+ X[:, 1] .+ 0.5 .* X[:, 2] .+ randn(n)

    # Observed outcome
    Y = Y0 .+ T .* true_ate

    return ObservationalData(
        Y,
        T,
        X,
        true_ate,
        propensity,
        n,
        p
    )
end

# =============================================================================
# No Effect DGP (Type I Error Testing)
# =============================================================================

"""
    dgp_observational_no_effect(; kwargs...) -> ObservationalData

DGP with zero treatment effect for Type I error testing.

Model:
    e(X) = expit(0.5 * X[:, 1])
    T ~ Bernoulli(e(X))
    Y = X[:, 1] + 0.5*X[:, 2] + ε  (no treatment effect)

# Arguments
- `n::Int=500`: Sample size
- `p::Int=3`: Number of covariates
- `confounding_strength::Float64=0.5`: Propensity dependence on X[:, 1]
- `seed::Int=42`: Random seed
"""
function dgp_observational_no_effect(;
    n::Int=500,
    p::Int=3,
    confounding_strength::Float64=0.5,
    seed::Int=42
)
    Random.seed!(seed)

    # Generate covariates
    X = randn(n, p)

    # Propensity model
    logit = confounding_strength .* X[:, 1]
    propensity = 1.0 ./ (1.0 .+ exp.(-logit))

    # Treatment assignment
    T = [rand() < prop for prop in propensity]

    # Ensure both groups
    if sum(T) == 0 || sum(T) == n
        T[1] = true
        T[end] = false
    end

    # Outcome: NO treatment effect
    Y = X[:, 1] .+ 0.5 .* X[:, 2] .+ randn(n)

    return ObservationalData(
        Y,
        T,
        X,
        0.0,  # True ATE is zero
        propensity,
        n,
        p
    )
end

# =============================================================================
# Strong Confounding DGP
# =============================================================================

"""
    dgp_observational_strong_confounding(; kwargs...) -> ObservationalData

DGP with strong selection on observables - severe confounding.

Model:
    e(X) = expit(confounding_strength * (X[:, 1] + X[:, 2]))
    T ~ Bernoulli(e(X))
    Y(0) = 2*X[:, 1] + X[:, 2] + ε  (strong dependence on same variables)
    Y(1) = Y(0) + true_ate

# Arguments
- `n::Int=500`: Sample size
- `p::Int=3`: Number of covariates
- `true_ate::Float64=2.0`: True treatment effect
- `confounding_strength::Float64=1.0`: Strength of confounding (>0.5 is strong)
- `seed::Int=42`: Random seed
"""
function dgp_observational_strong_confounding(;
    n::Int=500,
    p::Int=3,
    true_ate::Float64=2.0,
    confounding_strength::Float64=1.0,
    seed::Int=42
)
    Random.seed!(seed)

    # Generate covariates
    X = randn(n, p)

    # Strong propensity dependence on X[:, 1] and X[:, 2]
    logit = confounding_strength .* (X[:, 1] .+ X[:, 2])
    propensity = 1.0 ./ (1.0 .+ exp.(-logit))

    # Treatment assignment
    T = [rand() < prop for prop in propensity]

    # Ensure both groups
    if sum(T) == 0 || sum(T) == n
        T[1] = true
        T[end] = false
    end

    # Outcome: strong dependence on SAME confounders
    Y0 = 2.0 .* X[:, 1] .+ X[:, 2] .+ randn(n)
    Y = Y0 .+ T .* true_ate

    return ObservationalData(
        Y,
        T,
        X,
        true_ate,
        propensity,
        n,
        p
    )
end

# =============================================================================
# Overlap Violation DGP (Positivity Issues)
# =============================================================================

"""
    dgp_observational_overlap_violation(; kwargs...) -> ObservationalData

DGP with near-violations of positivity assumption.

Model:
    e(X) = expit(overlap_severity * X[:, 1])  # Strong propensity extremes
    Propensity clipped to [0.02, 0.98]

Near-separation: treatment strongly determined by X[:, 1].

# Arguments
- `n::Int=500`: Sample size
- `p::Int=3`: Number of covariates
- `true_ate::Float64=2.0`: True treatment effect
- `overlap_severity::Float64=2.0`: Higher = more extreme propensities
- `seed::Int=42`: Random seed
"""
function dgp_observational_overlap_violation(;
    n::Int=500,
    p::Int=3,
    true_ate::Float64=2.0,
    overlap_severity::Float64=2.0,
    seed::Int=42
)
    Random.seed!(seed)

    # Generate covariates
    X = randn(n, p)

    # Extreme propensity dependence
    logit = overlap_severity .* X[:, 1]
    propensity_raw = 1.0 ./ (1.0 .+ exp.(-logit))

    # Clip to avoid complete separation (but keep extreme)
    propensity = clamp.(propensity_raw, 0.02, 0.98)

    # Treatment assignment
    T = [rand() < prop for prop in propensity]

    # Ensure both groups
    if sum(T) == 0 || sum(T) == n
        T[1] = true
        T[end] = false
    end

    # Simple outcome model
    Y0 = X[:, 1] .+ randn(n)
    Y = Y0 .+ T .* true_ate

    return ObservationalData(
        Y,
        T,
        X,
        true_ate,
        propensity,
        n,
        p
    )
end

# =============================================================================
# High-Dimensional Covariates DGP
# =============================================================================

"""
    dgp_observational_high_dimensional(; kwargs...) -> ObservationalData

DGP with high-dimensional covariates (sparse true model).

Model:
    Only first n_relevant covariates matter for propensity and outcome.
    e(X) = expit(0.5 * sum(X[:, 1:n_relevant]))
    Y(0) = sum(X[:, 1:n_relevant] .* β) + ε
    Y(1) = Y(0) + true_ate

# Arguments
- `n::Int=300`: Sample size
- `p::Int=20`: Total covariates (many irrelevant)
- `n_relevant::Int=3`: Number of relevant covariates
- `true_ate::Float64=2.0`: True treatment effect
- `seed::Int=42`: Random seed
"""
function dgp_observational_high_dimensional(;
    n::Int=300,
    p::Int=20,
    n_relevant::Int=3,
    true_ate::Float64=2.0,
    seed::Int=42
)
    if n_relevant > p
        throw(ArgumentError("n_relevant must be <= p"))
    end

    Random.seed!(seed)

    # Generate covariates
    X = randn(n, p)

    # Propensity: only depends on relevant covariates
    relevant_sum = sum(X[:, 1:n_relevant], dims=2) |> vec
    logit = 0.5 .* relevant_sum
    propensity = 1.0 ./ (1.0 .+ exp.(-logit))

    # Treatment assignment
    T = [rand() < prop for prop in propensity]

    # Ensure both groups
    if sum(T) == 0 || sum(T) == n
        T[1] = true
        T[end] = false
    end

    # Outcome: sparse dependence
    beta = zeros(p)
    beta[1:n_relevant] = randn(n_relevant) .* 0.5
    Y0 = X * beta .+ randn(n)
    Y = Y0 .+ T .* true_ate

    return ObservationalData(
        Y,
        T,
        X,
        true_ate,
        propensity,
        n,
        p
    )
end

# =============================================================================
# Misspecified Propensity DGP (Tests DR robustness)
# =============================================================================

"""
    dgp_observational_nonlinear_propensity(; kwargs...) -> ObservationalData

DGP with nonlinear propensity model (tests robustness to misspecification).

Model:
    e(X) = expit(X[:, 1]² - 1)  # Nonlinear in X[:, 1]
    Y(0) = X[:, 1] + ε
    Y(1) = Y(0) + true_ate

A linear propensity model will be misspecified but outcome model correct.

# Arguments
- `n::Int=500`: Sample size
- `p::Int=3`: Number of covariates
- `true_ate::Float64=2.0`: True treatment effect
- `seed::Int=42`: Random seed
"""
function dgp_observational_nonlinear_propensity(;
    n::Int=500,
    p::Int=3,
    true_ate::Float64=2.0,
    seed::Int=42
)
    Random.seed!(seed)

    # Generate covariates
    X = randn(n, p)

    # Nonlinear propensity: quadratic in X[:, 1]
    logit = X[:, 1].^2 .- 1.0
    propensity = 1.0 ./ (1.0 .+ exp.(-logit))

    # Treatment assignment
    T = [rand() < prop for prop in propensity]

    # Ensure both groups
    if sum(T) == 0 || sum(T) == n
        T[1] = true
        T[end] = false
    end

    # Simple linear outcome (correctly specified if using X[:, 1])
    Y0 = X[:, 1] .+ randn(n)
    Y = Y0 .+ T .* true_ate

    return ObservationalData(
        Y,
        T,
        X,
        true_ate,
        propensity,
        n,
        p
    )
end
