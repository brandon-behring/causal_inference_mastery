#=
Neural Network-Style CATE Meta-Learners

Session 155: Julia Neural CATE Parity

Implements neural network-style meta-learners using polynomial feature expansion
and ridge regression to approximate MLP behavior, following the DragonNet pattern.

Methods:
- NeuralSLearner: Single model with polynomial features
- NeuralTLearner: Separate models with polynomial features
- NeuralXLearner: Cross-learner with polynomial features
- NeuralRLearner: Robinson transformation with polynomial features

Key Insight:
Polynomial feature expansion (degree=2) approximates a single hidden layer neural
network by capturing nonlinear relationships and interactions. Ridge regularization
prevents overfitting similar to weight decay in neural networks.

References:
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects"
- Nie & Wager (2021). "Quasi-oracle estimation of heterogeneous treatment effects"
=#

using LinearAlgebra
using Statistics
using Distributions


# =============================================================================
# Estimator Types
# =============================================================================

"""
    NeuralSLearner <: AbstractCATEEstimator

Neural S-Learner using polynomial feature expansion.

Approximates neural network behavior via ridge regression on polynomial features,
providing flexible nonlinear function approximation without neural network overhead.

# Fields
- `degree::Int`: Polynomial degree for feature expansion (default: 2)
- `lambda::Float64`: Ridge regularization parameter (default: 0.01)

# Algorithm
1. Expand X to polynomial features: X_poly = [X, X², X_i*X_j]
2. Augment with treatment: X_aug = [1, X_poly, T]
3. Fit ridge regression: μ̂(X_aug) → Y
4. CATE(x) = μ̂([1, poly(x), 1]) - μ̂([1, poly(x), 0])

# Example
```julia
problem = CATEProblem(Y, T, X, (alpha=0.05,))
solution = solve(problem, NeuralSLearner(degree=2, lambda=0.01))
```
"""
struct NeuralSLearner <: AbstractCATEEstimator
    degree::Int
    lambda::Float64

    function NeuralSLearner(; degree::Int=2, lambda::Float64=0.01)
        if degree < 1
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid NeuralSLearner configuration.\n" *
                "Function: NeuralSLearner\n" *
                "degree must be >= 1, got $degree"
            ))
        end
        if lambda < 0
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid NeuralSLearner configuration.\n" *
                "Function: NeuralSLearner\n" *
                "lambda must be >= 0, got $lambda"
            ))
        end
        new(degree, lambda)
    end
end


"""
    NeuralTLearner <: AbstractCATEEstimator

Neural T-Learner using polynomial feature expansion.

Fits separate models for treated and control groups on polynomial features.

# Fields
- `degree::Int`: Polynomial degree for feature expansion (default: 2)
- `lambda::Float64`: Ridge regularization parameter (default: 0.01)

# Algorithm
1. Expand X to polynomial features
2. Fit μ̂₀(X_poly) on control group
3. Fit μ̂₁(X_poly) on treated group
4. CATE(x) = μ̂₁(poly(x)) - μ̂₀(poly(x))
"""
struct NeuralTLearner <: AbstractCATEEstimator
    degree::Int
    lambda::Float64

    function NeuralTLearner(; degree::Int=2, lambda::Float64=0.01)
        if degree < 1 || lambda < 0
            throw(ArgumentError("Invalid NeuralTLearner parameters"))
        end
        new(degree, lambda)
    end
end


"""
    NeuralXLearner <: AbstractCATEEstimator

Neural X-Learner using polynomial feature expansion.

Two-stage cross-learner with propensity weighting on polynomial features.

# Fields
- `degree::Int`: Polynomial degree for feature expansion (default: 2)
- `lambda::Float64`: Ridge regularization parameter (default: 0.01)

# Algorithm
1. Stage 1: Fit μ̂₀, μ̂₁ on polynomial features (T-learner step)
2. Compute imputed effects:
   - D₁ = Y₁ - μ̂₀(X₁) for treated
   - D₀ = μ̂₁(X₀) - Y₀ for control
3. Stage 2: Fit τ̂₁(X_poly) → D₁, τ̂₀(X_poly) → D₀
4. CATE(x) = ê(x)·τ̂₀(x) + (1-ê(x))·τ̂₁(x)
"""
struct NeuralXLearner <: AbstractCATEEstimator
    degree::Int
    lambda::Float64

    function NeuralXLearner(; degree::Int=2, lambda::Float64=0.01)
        if degree < 1 || lambda < 0
            throw(ArgumentError("Invalid NeuralXLearner parameters"))
        end
        new(degree, lambda)
    end
end


"""
    NeuralRLearner <: AbstractCATEEstimator

Neural R-Learner using polynomial feature expansion.

Robinson transformation with polynomial features for doubly robust estimation.

# Fields
- `degree::Int`: Polynomial degree for feature expansion (default: 2)
- `lambda::Float64`: Ridge regularization parameter (default: 0.01)

# Algorithm
1. Fit ê(X_poly) = P(T=1|X) via logistic ridge
2. Fit m̂(X_poly) = E[Y|X] via ridge regression
3. Residualize: Ỹ = Y - m̂(X), T̃ = T - ê(X)
4. Fit τ(X_poly) by weighted regression minimizing E[(Ỹ - τ(X)·T̃)²]
"""
struct NeuralRLearner <: AbstractCATEEstimator
    degree::Int
    lambda::Float64

    function NeuralRLearner(; degree::Int=2, lambda::Float64=0.01)
        if degree < 1 || lambda < 0
            throw(ArgumentError("Invalid NeuralRLearner parameters"))
        end
        new(degree, lambda)
    end
end


# =============================================================================
# Shared Utilities (reusing from dragonnet.jl where possible)
# =============================================================================

# Note: _create_features, _fit_ridge, _fit_logistic_ridge, _sigmoid are
# defined in dragonnet.jl and accessible in the same module scope


"""
    _neural_ridge_fit(X, y; lambda=0.01) -> coefficients

Fit ridge regression with intercept.
"""
function _neural_ridge_fit(X::Matrix{T}, y::Vector{T}; lambda::Float64=0.01) where {T<:Real}
    n, p = size(X)
    # Add intercept
    X_int = hcat(ones(T, n), X)
    p_int = p + 1

    XtX = X_int' * X_int
    XtX_reg = XtX + T(lambda) * I(p_int)
    coef = XtX_reg \ (X_int' * y)

    return coef
end


"""
    _neural_ridge_predict(X, coef) -> predictions

Predict using ridge coefficients (with intercept).
"""
function _neural_ridge_predict(X::Matrix{T}, coef::Vector{T}) where {T<:Real}
    n = size(X, 1)
    X_int = hcat(ones(T, n), X)
    return X_int * coef
end


"""
    _neural_propensity(X, treatment; lambda=0.01) -> propensity

Estimate propensity scores using logistic ridge on polynomial features.
"""
function _neural_propensity(
    X::Matrix{T},
    treatment::Vector{Bool};
    lambda::Float64 = 0.01
) where {T<:Real}
    n = size(X, 1)
    X_int = hcat(ones(T, n), X)

    # Use _fit_logistic_ridge from dragonnet.jl
    coef = _fit_logistic_ridge(X_int, treatment; lambda=lambda)

    # Predict
    eta = X_int * coef
    propensity = _sigmoid.(eta)

    # Clip for stability
    return clamp.(propensity, T(0.01), T(0.99))
end


# =============================================================================
# Neural S-Learner
# =============================================================================

"""
    solve(problem::CATEProblem, estimator::NeuralSLearner) -> CATESolution

Estimate CATE using Neural S-Learner.
"""
function solve(
    problem::CATEProblem{T,P},
    estimator::NeuralSLearner
)::CATESolution{T,P} where {T<:Real, P<:NamedTuple}
    (; outcomes, treatment, covariates, parameters) = problem
    alpha = get(parameters, :alpha, 0.05)

    n = length(outcomes)

    # Step 1: Create polynomial features
    X_poly = _create_features(covariates; degree=estimator.degree)

    # Step 2: Augment with treatment
    X_aug = hcat(X_poly, T.(treatment))

    # Step 3: Fit ridge regression with intercept
    coef = _neural_ridge_fit(X_aug, outcomes; lambda=estimator.lambda)

    # Step 4: Compute CATE
    # Predict under T=1 and T=0
    X_treat = hcat(X_poly, ones(T, n))
    X_ctrl = hcat(X_poly, zeros(T, n))

    pred_treat = _neural_ridge_predict(X_treat, coef)
    pred_ctrl = _neural_ridge_predict(X_ctrl, coef)

    cate = pred_treat - pred_ctrl

    # Step 5: Compute ATE and SE
    ate = T(mean(cate))
    se = compute_ate_se(cate, treatment, outcomes, covariates)

    # Confidence interval
    z_crit = T(quantile(Normal(), 1 - alpha / 2))
    ci_lower = ate - z_crit * se
    ci_upper = ate + z_crit * se

    return CATESolution{T,P}(
        cate, ate, se, ci_lower, ci_upper,
        :neural_s_learner, :Success, problem
    )
end


# =============================================================================
# Neural T-Learner
# =============================================================================

"""
    solve(problem::CATEProblem, estimator::NeuralTLearner) -> CATESolution

Estimate CATE using Neural T-Learner.
"""
function solve(
    problem::CATEProblem{T,P},
    estimator::NeuralTLearner
)::CATESolution{T,P} where {T<:Real, P<:NamedTuple}
    (; outcomes, treatment, covariates, parameters) = problem
    alpha = get(parameters, :alpha, 0.05)

    n = length(outcomes)

    # Create polynomial features
    X_poly = _create_features(covariates; degree=estimator.degree)

    # Split by treatment
    treat_idx = treatment
    ctrl_idx = .!treatment

    n_treat = sum(treat_idx)
    n_ctrl = sum(ctrl_idx)

    if n_treat < 2 || n_ctrl < 2
        throw(ArgumentError(
            "CRITICAL ERROR: Insufficient samples per group.\n" *
            "Function: NeuralTLearner\n" *
            "n_treated: $n_treat, n_control: $n_ctrl (need >= 2 each)"
        ))
    end

    # Fit separate models
    coef_ctrl = _neural_ridge_fit(X_poly[ctrl_idx, :], outcomes[ctrl_idx];
                                   lambda=estimator.lambda)
    coef_treat = _neural_ridge_fit(X_poly[treat_idx, :], outcomes[treat_idx];
                                    lambda=estimator.lambda)

    # Predict for all units
    pred_ctrl = _neural_ridge_predict(X_poly, coef_ctrl)
    pred_treat = _neural_ridge_predict(X_poly, coef_treat)

    cate = pred_treat - pred_ctrl

    # ATE and SE
    ate = T(mean(cate))
    se = compute_ate_se(cate, treatment, outcomes, covariates)

    z_crit = T(quantile(Normal(), 1 - alpha / 2))
    ci_lower = ate - z_crit * se
    ci_upper = ate + z_crit * se

    return CATESolution{T,P}(
        cate, ate, se, ci_lower, ci_upper,
        :neural_t_learner, :Success, problem
    )
end


# =============================================================================
# Neural X-Learner
# =============================================================================

"""
    solve(problem::CATEProblem, estimator::NeuralXLearner) -> CATESolution

Estimate CATE using Neural X-Learner.
"""
function solve(
    problem::CATEProblem{T,P},
    estimator::NeuralXLearner
)::CATESolution{T,P} where {T<:Real, P<:NamedTuple}
    (; outcomes, treatment, covariates, parameters) = problem
    alpha = get(parameters, :alpha, 0.05)

    n = length(outcomes)

    # Create polynomial features
    X_poly = _create_features(covariates; degree=estimator.degree)

    treat_idx = treatment
    ctrl_idx = .!treatment

    n_treat = sum(treat_idx)
    n_ctrl = sum(ctrl_idx)

    if n_treat < 2 || n_ctrl < 2
        throw(ArgumentError("Insufficient samples per group for X-Learner"))
    end

    # Stage 1: Fit T-learner models (μ₀, μ₁)
    coef_ctrl = _neural_ridge_fit(X_poly[ctrl_idx, :], outcomes[ctrl_idx];
                                   lambda=estimator.lambda)
    coef_treat = _neural_ridge_fit(X_poly[treat_idx, :], outcomes[treat_idx];
                                    lambda=estimator.lambda)

    # Cross-predictions
    mu0_on_treat = _neural_ridge_predict(X_poly[treat_idx, :], coef_ctrl)  # μ̂₀(X₁)
    mu1_on_ctrl = _neural_ridge_predict(X_poly[ctrl_idx, :], coef_treat)   # μ̂₁(X₀)

    # Imputed treatment effects
    D1 = outcomes[treat_idx] - mu0_on_treat  # Y₁ - μ̂₀(X₁) for treated
    D0 = mu1_on_ctrl - outcomes[ctrl_idx]     # μ̂₁(X₀) - Y₀ for control

    # Stage 2: Fit τ models
    coef_tau1 = _neural_ridge_fit(X_poly[treat_idx, :], D1; lambda=estimator.lambda)
    coef_tau0 = _neural_ridge_fit(X_poly[ctrl_idx, :], D0; lambda=estimator.lambda)

    # Predict τ for all units
    tau1_all = _neural_ridge_predict(X_poly, coef_tau1)  # τ̂₁(X)
    tau0_all = _neural_ridge_predict(X_poly, coef_tau0)  # τ̂₀(X)

    # Estimate propensity for weighting
    propensity = _neural_propensity(X_poly, treatment; lambda=estimator.lambda)

    # X-learner combination: CATE(x) = e(x)·τ₀(x) + (1-e(x))·τ₁(x)
    cate = propensity .* tau0_all .+ (one(T) .- propensity) .* tau1_all

    # ATE and SE
    ate = T(mean(cate))
    se = compute_ate_se(cate, treatment, outcomes, covariates)

    z_crit = T(quantile(Normal(), 1 - alpha / 2))
    ci_lower = ate - z_crit * se
    ci_upper = ate + z_crit * se

    return CATESolution{T,P}(
        cate, ate, se, ci_lower, ci_upper,
        :neural_x_learner, :Success, problem
    )
end


# =============================================================================
# Neural R-Learner
# =============================================================================

"""
    solve(problem::CATEProblem, estimator::NeuralRLearner) -> CATESolution

Estimate CATE using Neural R-Learner.

Uses Robinson transformation with polynomial features for doubly robust estimation.
"""
function solve(
    problem::CATEProblem{T,P},
    estimator::NeuralRLearner
)::CATESolution{T,P} where {T<:Real, P<:NamedTuple}
    (; outcomes, treatment, covariates, parameters) = problem
    alpha = get(parameters, :alpha, 0.05)

    n = length(outcomes)

    # Create polynomial features
    X_poly = _create_features(covariates; degree=estimator.degree)
    p_poly = size(X_poly, 2)

    # Step 1: Estimate propensity ê(X)
    propensity = _neural_propensity(X_poly, treatment; lambda=estimator.lambda)

    # Step 2: Estimate outcome m̂(X) = E[Y|X]
    m_coef = _neural_ridge_fit(X_poly, outcomes; lambda=estimator.lambda)
    m_hat = _neural_ridge_predict(X_poly, m_coef)

    # Step 3: Residualize
    Y_resid = outcomes - m_hat
    T_float = T.(treatment)
    T_resid = T_float - propensity

    # Step 4: Estimate τ(X) via weighted regression
    # Minimize: Σ (Ỹᵢ - τ(Xᵢ)·T̃ᵢ)²
    # This is equivalent to weighted regression of Ỹ/T̃ on X with weights T̃²

    weights = T_resid .^ 2

    # Avoid division by near-zero
    valid_idx = abs.(T_resid) .> T(0.05)
    n_valid = sum(valid_idx)

    if n_valid < p_poly + 2
        # Fallback: constant ATE
        ate = T(sum(Y_resid .* T_resid) / (sum(weights) + T(1e-10)))
        cate = fill(ate, n)
    else
        # Weighted regression for heterogeneous τ(X)
        # Transform: pseudo_outcome = Ỹ / T̃
        # Weight by T̃²

        pseudo_outcome = Y_resid[valid_idx] ./ T_resid[valid_idx]
        X_valid = X_poly[valid_idx, :]
        w_valid = weights[valid_idx]

        # Weighted ridge: (X'WX + λI)⁻¹ X'Wy
        n_v = size(X_valid, 1)
        X_int = hcat(ones(T, n_v), X_valid)
        p_int = size(X_int, 2)

        W = Diagonal(w_valid)
        XtWX = X_int' * W * X_int
        XtWy = X_int' * (w_valid .* pseudo_outcome)

        tau_coef = (XtWX + T(estimator.lambda) * I(p_int)) \ XtWy

        # Predict τ for all units
        X_all_int = hcat(ones(T, n), X_poly)
        cate = X_all_int * tau_coef
    end

    # ATE
    ate = T(mean(cate))

    # SE via influence function
    se = compute_influence_se(Y_resid, T_resid, ate)

    z_crit = T(quantile(Normal(), 1 - alpha / 2))
    ci_lower = ate - z_crit * se
    ci_upper = ate + z_crit * se

    return CATESolution{T,P}(
        cate, ate, se, ci_lower, ci_upper,
        :neural_r_learner, :Success, problem
    )
end
