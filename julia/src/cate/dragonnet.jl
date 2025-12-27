#=
DragonNet Implementation for CATE Estimation

DragonNet (Shi et al. 2019) uses a neural network with shared representation
and three output heads: propensity, Y(0), and Y(1).

Two backends:
- :regression (default): Approximates shared representation using ridge regression
  with polynomial/interaction features. Always available.
- :flux: True neural network with Flux.jl (not yet implemented).

Session 152: Julia Neural CATE foundation.

References:
- Shi et al. (2019). "Adapting Neural Networks for the Estimation of
  Treatment Effects." NeurIPS 2019.
=#

using LinearAlgebra
using Statistics
using Random
using Distributions


# =============================================================================
# Internal Model State
# =============================================================================

"""
    DragonNetRegressionModel{T<:Real}

Internal model state for regression-based DragonNet.

Stores fitted coefficients for propensity and outcome models.
"""
mutable struct DragonNetRegressionModel{T<:Real}
    propensity_coef::Union{Nothing, Vector{T}}
    y0_coef::Union{Nothing, Vector{T}}
    y1_coef::Union{Nothing, Vector{T}}
    feature_means::Union{Nothing, Vector{T}}
    feature_stds::Union{Nothing, Vector{T}}
    is_fitted::Bool
end

DragonNetRegressionModel{T}() where {T<:Real} =
    DragonNetRegressionModel{T}(nothing, nothing, nothing, nothing, nothing, false)


# =============================================================================
# Feature Engineering
# =============================================================================

"""
    _create_features(X; degree=2) -> Matrix

Create polynomial features to approximate neural network representation.

Includes: original features, squares, and pairwise interactions.
"""
function _create_features(X::Matrix{T}; degree::Int=2) where {T<:Real}
    n, p = size(X)

    if degree == 1
        return X
    end

    # Start with original features
    features = X

    if degree >= 2
        # Add squared terms
        squared = X .^ 2
        features = hcat(features, squared)

        # Add pairwise interactions (limit to first few to avoid explosion)
        n_interact = min(p, 5)
        for i in 1:n_interact
            for j in (i+1):n_interact
                interaction = X[:, i] .* X[:, j]
                features = hcat(features, interaction)
            end
        end
    end

    return features
end


"""
    _standardize_features(X) -> (X_std, means, stds)

Standardize features to zero mean and unit variance.
"""
function _standardize_features(X::Matrix{T}) where {T<:Real}
    means = vec(mean(X, dims=1))
    stds = vec(std(X, dims=1))

    # Avoid division by zero for constant features
    stds[stds .< T(1e-10)] .= T(1.0)

    X_std = (X .- means') ./ stds'
    return X_std, means, stds
end


"""
    _apply_standardization(X, means, stds) -> X_std

Apply pre-computed standardization to new data.
"""
function _apply_standardization(X::Matrix{T}, means::Vector{T}, stds::Vector{T}) where {T<:Real}
    return (X .- means') ./ stds'
end


# =============================================================================
# Regression Functions
# =============================================================================

# Note: _sigmoid is imported from utils.jl

"""
    _fit_ridge(X, y; lambda=1.0) -> coefficients

Fit ridge regression: β = (X'X + λI)⁻¹X'y
"""
function _fit_ridge(X::Matrix{T}, y::Vector{T}; lambda::Float64=1.0) where {T<:Real}
    n, p = size(X)
    XtX = X' * X
    XtX_reg = XtX + T(lambda) * I(p)
    return XtX_reg \ (X' * y)
end


"""
    _fit_logistic_ridge(X, y; lambda=1.0, max_iter=100) -> coefficients

Fit logistic regression with L2 penalty using IRLS.
"""
function _fit_logistic_ridge(
    X::Matrix{T},
    y::Vector{Bool};
    lambda::Float64 = 1.0,
    max_iter::Int = 100,
    tol::Float64 = 1e-6
) where {T<:Real}
    n, p = size(X)
    beta = zeros(T, p)
    y_float = T.(y)

    for iter in 1:max_iter
        eta = X * beta
        mu = _sigmoid.(eta)
        mu = clamp.(mu, T(1e-10), T(1 - 1e-10))

        W = mu .* (one(T) .- mu)
        z = eta .+ (y_float .- mu) ./ W

        W_sqrt = sqrt.(W)
        X_w = W_sqrt .* X
        z_w = W_sqrt .* z

        # Ridge update
        XtX = X_w' * X_w + T(lambda) * I(p)
        beta_new = XtX \ (X_w' * z_w)

        if norm(beta_new - beta) < tol
            return beta_new
        end
        beta = beta_new
    end

    return beta
end


# =============================================================================
# Model Fitting
# =============================================================================

"""
    _fit_dragonnet_regression!(model, X, T, Y, config)

Fit DragonNet using regression backend.

Fits three models on polynomial features:
1. Propensity model P(T=1|X) via logistic ridge
2. Y(0) model on control units
3. Y(1) model on treated units
"""
function _fit_dragonnet_regression!(
    model::DragonNetRegressionModel{T},
    X::Matrix{T},
    treatment::Vector{Bool},
    Y::Vector{T},
    config::DragonNetConfig
) where {T<:Real}

    # Set random seed if specified
    if !isnothing(config.random_state)
        Random.seed!(config.random_state)
    end

    n = length(Y)

    # Create polynomial features
    X_poly = _create_features(X)

    # Standardize features
    X_std, means, stds = _standardize_features(X_poly)
    model.feature_means = means
    model.feature_stds = stds

    # Add intercept
    X_int = hcat(ones(T, n), X_std)

    # Masks for treatment groups
    treated_mask = treatment
    control_mask = .!treatment

    n_treated = sum(treated_mask)
    n_control = sum(control_mask)

    # Fit propensity model (all data)
    model.propensity_coef = _fit_logistic_ridge(
        X_int,
        treatment;
        lambda = config.alpha,
        max_iter = config.max_iter
    )

    # Fit Y(0) model (control data only)
    if n_control >= 2
        model.y0_coef = _fit_ridge(
            X_int[control_mask, :],
            Y[control_mask];
            lambda = config.alpha
        )
    else
        # Fallback: use mean
        model.y0_coef = zeros(T, size(X_int, 2))
        model.y0_coef[1] = mean(Y[control_mask])
    end

    # Fit Y(1) model (treated data only)
    if n_treated >= 2
        model.y1_coef = _fit_ridge(
            X_int[treated_mask, :],
            Y[treated_mask];
            lambda = config.alpha
        )
    else
        # Fallback: use mean
        model.y1_coef = zeros(T, size(X_int, 2))
        model.y1_coef[1] = mean(Y[treated_mask])
    end

    model.is_fitted = true
    return model
end


# =============================================================================
# Prediction Functions
# =============================================================================

"""
    _prepare_features(model, X) -> X_prepared

Prepare features for prediction (polynomial + standardization + intercept).
"""
function _prepare_features(model::DragonNetRegressionModel{T}, X::Matrix{T}) where {T<:Real}
    if !model.is_fitted
        throw(ErrorException(
            "CRITICAL ERROR: Model not fitted.\n" *
            "Function: _prepare_features\n" *
            "Call _fit_dragonnet_regression! before prediction."
        ))
    end

    n = size(X, 1)
    X_poly = _create_features(X)
    X_std = _apply_standardization(X_poly, model.feature_means, model.feature_stds)
    return hcat(ones(T, n), X_std)
end


"""
    _predict_propensity(model, X) -> Vector

Predict propensity scores P(T=1|X).
"""
function _predict_propensity(model::DragonNetRegressionModel{T}, X::Matrix{T}) where {T<:Real}
    X_int = _prepare_features(model, X)
    eta = X_int * model.propensity_coef
    propensity = _sigmoid.(eta)
    return clamp.(propensity, T(0.01), T(0.99))
end


"""
    _predict_y0(model, X) -> Vector

Predict E[Y|T=0, X].
"""
function _predict_y0(model::DragonNetRegressionModel{T}, X::Matrix{T}) where {T<:Real}
    X_int = _prepare_features(model, X)
    return X_int * model.y0_coef
end


"""
    _predict_y1(model, X) -> Vector

Predict E[Y|T=1, X].
"""
function _predict_y1(model::DragonNetRegressionModel{T}, X::Matrix{T}) where {T<:Real}
    X_int = _prepare_features(model, X)
    return X_int * model.y1_coef
end


"""
    _predict_cate(model, X) -> Vector

Predict CATE = E[Y(1)|X] - E[Y(0)|X].
"""
function _predict_cate(model::DragonNetRegressionModel{T}, X::Matrix{T}) where {T<:Real}
    return _predict_y1(model, X) - _predict_y0(model, X)
end


# =============================================================================
# Flux Backend (Placeholder)
# =============================================================================

"""
    _fit_dragonnet_flux!(...)

Placeholder for Flux.jl neural network backend.
"""
function _fit_dragonnet_flux!(args...)
    throw(ErrorException(
        "CRITICAL ERROR: Flux backend not yet implemented.\n" *
        "Function: _fit_dragonnet_flux!\n" *
        "Use backend = :regression (default) for now.\n" *
        "Flux implementation planned for future sessions."
    ))
end


# =============================================================================
# Main Solve Function
# =============================================================================

"""
    solve(problem::CATEProblem, estimator::Dragonnet) -> CATESolution

Estimate CATE using DragonNet neural architecture.

# Algorithm
1. Build shared representation: φ(X) via polynomial features (regression) or
   hidden layers (Flux)
2. Three output heads:
   - Propensity: P(T=1|φ(X)) - logistic regression/classification
   - Y(0): E[Y|T=0, φ(X)] - ridge/neural regression
   - Y(1): E[Y|T=1, φ(X)] - ridge/neural regression
3. CATE: τ(X) = Ŷ(1) - Ŷ(0)

# Standard Error Estimation
Uses doubly robust influence function:
ψᵢ = (Tᵢ/e(Xᵢ) - (1-Tᵢ)/(1-e(Xᵢ))) × (Yᵢ - Ŷᵢ) + τ̂(Xᵢ)
SE = √(Var(ψ) / n)

# Example
```julia
using CausalEstimators

problem = CATEProblem(Y, T, X, (alpha=0.05,))
solution = solve(problem, Dragonnet())

println("ATE: \$(solution.ate) ± \$(solution.se)")
```

# References
- Shi et al. (2019). "Adapting Neural Networks for the Estimation of
  Treatment Effects." NeurIPS 2019.
"""
function solve(
    problem::CATEProblem{T,P},
    estimator::Dragonnet
)::CATESolution{T,P} where {T<:Real, P<:NamedTuple}
    # Extract from problem
    outcomes = problem.outcomes
    treatment = problem.treatment
    covariates = problem.covariates
    parameters = problem.parameters
    alpha = get(parameters, :alpha, 0.05)

    n = length(outcomes)
    config = estimator.config

    # Select backend
    backend = estimator.backend

    # Create and fit model
    if backend == :regression
        model = DragonNetRegressionModel{T}()
        _fit_dragonnet_regression!(model, covariates, treatment, outcomes, config)
    elseif backend == :flux
        _fit_dragonnet_flux!()  # Will throw informative error
    end

    # Predict CATE
    cate = _predict_cate(model, covariates)

    # Compute ATE
    ate = T(mean(cate))

    # SE estimation using doubly robust influence function
    propensity = _predict_propensity(model, covariates)
    y0_pred = _predict_y0(model, covariates)
    y1_pred = _predict_y1(model, covariates)

    # Doubly robust pseudo-outcome
    # ψ = (T/e - (1-T)/(1-e)) × (Y - T×Y1 - (1-T)×Y0) + (Y1 - Y0)
    T_float = T.(treatment)
    y_pred = T_float .* y1_pred .+ (one(T) .- T_float) .* y0_pred
    residual = outcomes .- y_pred
    weight = T_float ./ propensity .- (one(T) .- T_float) ./ (one(T) .- propensity)
    psi = weight .* residual .+ cate

    # SE = std(ψ) / √n
    se = T(std(psi, corrected=true) / sqrt(n))

    # Ensure SE is positive and finite
    if !isfinite(se) || se <= 0
        se = T(std(cate, corrected=true) / sqrt(n))
    end

    # Confidence interval
    z_crit = T(quantile(Normal(), 1 - alpha / 2))
    ci_lower = ate - z_crit * se
    ci_upper = ate + z_crit * se

    return CATESolution{T,P}(
        cate,
        ate,
        se,
        ci_lower,
        ci_upper,
        :dragonnet,
        :Success,
        problem
    )
end
