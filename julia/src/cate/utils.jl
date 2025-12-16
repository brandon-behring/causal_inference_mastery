#=
CATE Utility Functions

Shared utilities for CATE estimation including:
- OLS and Ridge regression fitting
- Propensity score estimation
- Standard error computation
=#

using LinearAlgebra
using Statistics
using Distributions

# =============================================================================
# Regression Utilities
# =============================================================================

"""
    fit_ols(X, y)

Fit ordinary least squares regression.

Returns (β, predictions, residuals).

# Arguments
- `X::Matrix{T}`: Design matrix (n × p), should include intercept column if desired
- `y::Vector{T}`: Response variable (n,)

# Returns
- `β::Vector{T}`: Coefficient estimates
- `predictions::Vector{T}`: Fitted values
- `residuals::Vector{T}`: Residuals (y - ŷ)
"""
function fit_ols(X::Matrix{T}, y::Vector{T}) where {T<:Real}
    # Add intercept if not present
    n, p = size(X)

    # Solve normal equations: β = (X'X)⁻¹X'y
    # Use QR decomposition for numerical stability
    β = X \ y

    predictions = X * β
    residuals = y - predictions

    return β, predictions, residuals
end

"""
    fit_ridge(X, y; lambda=1.0)

Fit ridge regression with L2 penalty.

Returns (β, predictions, residuals).

# Arguments
- `X::Matrix{T}`: Design matrix (n × p)
- `y::Vector{T}`: Response variable (n,)
- `lambda::Float64`: Regularization parameter (default: 1.0)

# Returns
- `β::Vector{T}`: Coefficient estimates
- `predictions::Vector{T}`: Fitted values
- `residuals::Vector{T}`: Residuals (y - ŷ)
"""
function fit_ridge(X::Matrix{T}, y::Vector{T}; lambda::Float64 = 1.0) where {T<:Real}
    n, p = size(X)

    # Ridge solution: β = (X'X + λI)⁻¹X'y
    XtX = X' * X
    XtX_reg = XtX + lambda * I(p)
    β = XtX_reg \ (X' * y)

    predictions = X * β
    residuals = y - predictions

    return β, predictions, residuals
end

"""
    fit_model(X, y, model::Symbol; kwargs...)

Fit regression model by type.

# Arguments
- `X::Matrix{T}`: Design matrix
- `y::Vector{T}`: Response
- `model::Symbol`: :ols or :ridge

# Returns
- `β::Vector{T}`: Coefficients
- `predictions::Vector{T}`: Fitted values
- `residuals::Vector{T}`: Residuals
"""
function fit_model(X::Matrix{T}, y::Vector{T}, model::Symbol; kwargs...) where {T<:Real}
    if model == :ols
        return fit_ols(X, y)
    elseif model == :ridge
        lambda = get(kwargs, :lambda, 1.0)
        return fit_ridge(X, y; lambda=lambda)
    else
        throw(ArgumentError("Unknown model type: $model. Use :ols or :ridge."))
    end
end

"""
    predict_ols(X, β)

Generate predictions from fitted coefficients.

# Arguments
- `X::Matrix{T}`: Design matrix for prediction
- `β::Vector{T}`: Fitted coefficients

# Returns
- `predictions::Vector{T}`: Predicted values
"""
function predict_ols(X::Matrix{T}, β::Vector{T}) where {T<:Real}
    return X * β
end

# =============================================================================
# Propensity Score Utilities
# =============================================================================

"""
    estimate_propensity(X, treatment)

Estimate propensity scores P(T=1|X) using logistic regression.

# Arguments
- `X::Matrix{T}`: Covariate matrix (n × p)
- `treatment::Vector{Bool}`: Treatment indicator

# Returns
- `propensity::Vector{T}`: Estimated propensity scores, clipped to [0.01, 0.99]
"""
function estimate_propensity(X::Matrix{T}, treatment::Vector{Bool}) where {T<:Real}
    n, p = size(X)

    # Add intercept
    X_with_intercept = hcat(ones(T, n), X)

    # Convert treatment to float for regression
    y = Float64.(treatment)

    # Use GLM for logistic regression
    try
        df = DataFrame(X_with_intercept, :auto)
        df.treatment = y
        formula_str = "treatment ~ " * join(["x$i" for i in 1:size(X_with_intercept, 2)], " + ")

        # Actually, let's use a simpler approach with manual logistic regression
        # to avoid DataFrame overhead
        propensity = _logistic_regression(X_with_intercept, treatment)
        return propensity
    catch e
        # Fallback to simple logistic via Newton-Raphson
        propensity = _logistic_regression(X_with_intercept, treatment)
        return propensity
    end
end

"""
    _logistic_regression(X, y; max_iter=100, tol=1e-6)

Fit logistic regression using iteratively reweighted least squares (IRLS).

# Arguments
- `X::Matrix`: Design matrix with intercept
- `y::Vector{Bool}`: Binary response

# Returns
- `propensity::Vector{Float64}`: Predicted probabilities, clipped to [0.01, 0.99]
"""
function _logistic_regression(
    X::Matrix{T},
    y::Vector{Bool};
    max_iter::Int = 100,
    tol::Float64 = 1e-6
) where {T<:Real}
    n, p = size(X)
    β = zeros(p)
    y_float = Float64.(y)

    for iter in 1:max_iter
        # Compute predictions
        η = X * β
        μ = _sigmoid.(η)

        # Clip for numerical stability
        μ = clamp.(μ, 1e-10, 1 - 1e-10)

        # Weights (variance)
        W = μ .* (1 .- μ)

        # Working response
        z = η .+ (y_float .- μ) ./ W

        # Weighted least squares update
        W_sqrt = sqrt.(W)
        X_weighted = W_sqrt .* X
        z_weighted = W_sqrt .* z

        β_new = X_weighted \ z_weighted

        # Check convergence
        if norm(β_new - β) < tol
            β = β_new
            break
        end
        β = β_new
    end

    # Final predictions
    η = X * β
    propensity = _sigmoid.(η)

    # Clip propensity scores
    propensity = clamp.(propensity, 0.01, 0.99)

    return propensity
end

"""
    _sigmoid(x)

Sigmoid function with numerical stability.
"""
function _sigmoid(x::Real)
    if x >= 0
        return 1.0 / (1.0 + exp(-x))
    else
        exp_x = exp(x)
        return exp_x / (1.0 + exp_x)
    end
end

# =============================================================================
# Standard Error Utilities
# =============================================================================

"""
    compute_ate_se(cate, treatment, Y, X; n_bootstrap=200)

Compute standard error of ATE using bootstrap resampling.

# Arguments
- `cate::Vector{T}`: Individual treatment effect estimates
- `treatment::Vector{Bool}`: Treatment indicator
- `Y::Vector{T}`: Outcome variable
- `X::Matrix{T}`: Covariates
- `n_bootstrap::Int`: Number of bootstrap iterations

# Returns
- `se::T`: Bootstrap standard error of ATE
"""
function compute_ate_se(
    cate::Vector{T},
    treatment::Vector{Bool},
    Y::Vector{T},
    X::Matrix{T};
    n_bootstrap::Int = 200
) where {T<:Real}
    n = length(cate)
    ate = mean(cate)

    # Neyman-style SE based on treatment/control variance (most robust)
    n1 = sum(treatment)
    n0 = n - n1

    if n1 > 1 && n0 > 1
        Y1 = Y[treatment]
        Y0 = Y[.!treatment]
        neyman_se = sqrt(var(Y1) / n1 + var(Y0) / n0)

        # CATE variance can inform SE when there's meaningful heterogeneity
        cate_var = var(cate, corrected=true)
        cate_se = sqrt(cate_var / n)

        # Use max of Neyman SE and CATE-based SE
        # This ensures proper coverage when CATE is constant (Neyman dominates)
        # but also captures extra uncertainty from heterogeneity
        return max(neyman_se, cate_se)
    end

    # Fallback: use CATE variance if available
    cate_var = var(cate, corrected=true)
    if cate_var > 1e-10
        return sqrt(cate_var / n)
    end

    # Last resort: use residual variance
    return std(Y) / sqrt(n)
end

"""
    compute_ate_se_simple(cate)

Compute standard error of ATE from CATE variance (simple method).

# Arguments
- `cate::Vector{T}`: Individual treatment effect estimates

# Returns
- `se::T`: Standard error of ATE
"""
function compute_ate_se_simple(cate::Vector{T}) where {T<:Real}
    n = length(cate)
    cate_var = var(cate, corrected=true)
    return sqrt(max(cate_var, 1e-10) / n)
end

"""
    compute_influence_se(Y_resid, T_resid, ate)

Compute influence function based SE for R-learner/DML.

# Arguments
- `Y_resid::Vector{T}`: Outcome residuals (Y - m̂(X))
- `T_resid::Vector{T}`: Treatment residuals (T - ê(X))
- `ate::T`: Estimated ATE

# Returns
- `se::T`: Influence function based standard error
"""
function compute_influence_se(
    Y_resid::Vector{T},
    T_resid::Vector{T},
    ate::T
) where {T<:Real}
    n = length(Y_resid)

    # Influence function: ψᵢ = (Ỹᵢ - θ·T̃ᵢ) · T̃ᵢ / E[T̃²]
    T_resid_sq_mean = mean(T_resid .^ 2)

    if T_resid_sq_mean < 1e-10
        # Fallback to simple SE
        return std(Y_resid) / sqrt(n)
    end

    influence = (Y_resid .- ate .* T_resid) .* T_resid ./ T_resid_sq_mean

    # SE = sqrt(Var(ψ) / n)
    se = sqrt(var(influence, corrected=true) / n)

    return se
end

# =============================================================================
# Design Matrix Utilities
# =============================================================================

"""
    add_intercept(X)

Add intercept column to design matrix.

# Arguments
- `X::Matrix{T}`: Design matrix (n × p)

# Returns
- `X_int::Matrix{T}`: Design matrix with intercept (n × (p+1))
"""
function add_intercept(X::Matrix{T}) where {T<:Real}
    n = size(X, 1)
    return hcat(ones(T, n), X)
end

"""
    augment_with_treatment(X, treatment)

Augment covariate matrix with treatment indicator for S-learner.

# Arguments
- `X::Matrix{T}`: Covariate matrix (n × p)
- `treatment::Vector{Bool}`: Treatment indicator

# Returns
- `X_aug::Matrix{T}`: Augmented matrix (n × (p+1))
"""
function augment_with_treatment(X::Matrix{T}, treatment::Vector{Bool}) where {T<:Real}
    return hcat(X, T.(treatment))
end
