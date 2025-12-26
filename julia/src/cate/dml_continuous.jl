#=
Double Machine Learning for Continuous Treatment

Extends DML to handle continuous (non-binary) treatments. The key difference
from binary DML is that we model E[D|X] using regression instead of P(T=1|X)
using classification.

Algorithm:
1. Split data into K folds
2. For each fold k:
   - Train outcome model m̂(X) = E[Y|X] on OTHER folds
   - Train treatment model d̂(X) = E[D|X] on OTHER folds (regression, not classification)
   - Predict for fold k (out-of-sample)
3. Compute residuals: Ỹ = Y - m̂(X), D̃ = D - d̂(X)
4. Estimate θ = Cov(Ỹ, D̃) / Var(D̃)

Reference:
- Chernozhukov et al. (2018). "Double/debiased machine learning"
- Colangelo & Lee (2020). "Double Debiased ML with Continuous Treatments"
=#

using Statistics
using LinearAlgebra
using Distributions
using Random

# =============================================================================
# Result Type for Continuous Treatment DML
# =============================================================================

"""
    DMLContinuousResult

Result from DML with continuous treatment.

# Fields
- `cate::Vector{Float64}`: Individual treatment effects τ̂(xᵢ) for each unit
- `ate::Float64`: Average treatment effect (average marginal effect dE[Y]/dD)
- `ate_se::Float64`: Standard error of ATE
- `ci_lower::Float64`: Lower bound of confidence interval
- `ci_upper::Float64`: Upper bound of confidence interval
- `method::Symbol`: :dml_continuous
- `fold_estimates::Vector{Float64}`: Per-fold ATE estimates for stability
- `fold_ses::Vector{Float64}`: Per-fold standard errors
- `outcome_r2::Float64`: R-squared of outcome model (diagnostic)
- `treatment_r2::Float64`: R-squared of treatment model (diagnostic)
- `n::Int`: Number of observations
- `n_folds::Int`: Number of cross-fitting folds
"""
struct DMLContinuousResult
    cate::Vector{Float64}
    ate::Float64
    ate_se::Float64
    ci_lower::Float64
    ci_upper::Float64
    method::Symbol
    fold_estimates::Vector{Float64}
    fold_ses::Vector{Float64}
    outcome_r2::Float64
    treatment_r2::Float64
    n::Int
    n_folds::Int
end

# =============================================================================
# Estimator Type
# =============================================================================

"""
    DMLContinuous <: AbstractCATEEstimator

Double Machine Learning for continuous treatment.

# Fields
- `n_folds::Int`: Number of cross-fitting folds (default: 5)
- `model::Symbol`: Base learner for nuisance models (:ols, :ridge)
- `cate_model::Symbol`: Base learner for CATE model (:ols, :ridge)
"""
struct DMLContinuous <: AbstractCATEEstimator
    n_folds::Int
    model::Symbol
    cate_model::Symbol

    function DMLContinuous(;
        n_folds::Int = 5,
        model::Symbol = :ridge,
        cate_model::Symbol = :ols
    )
        if n_folds < 2
            throw(ArgumentError(
                "CRITICAL ERROR: n_folds must be >= 2.\n" *
                "Function: DMLContinuous\n" *
                "Got: n_folds = $n_folds"
            ))
        end
        if model ∉ (:ols, :ridge)
            throw(ArgumentError(
                "CRITICAL ERROR: Unknown model type.\n" *
                "Function: DMLContinuous\n" *
                "Got: model = $model\n" *
                "Valid options: :ols, :ridge"
            ))
        end
        if cate_model ∉ (:ols, :ridge)
            throw(ArgumentError(
                "CRITICAL ERROR: Unknown cate_model type.\n" *
                "Function: DMLContinuous\n" *
                "Got: cate_model = $cate_model\n" *
                "Valid options: :ols, :ridge"
            ))
        end
        new(n_folds, model, cate_model)
    end
end

# =============================================================================
# Validation
# =============================================================================

"""
    _validate_continuous_inputs(outcomes, treatment, covariates)

Validate inputs for continuous treatment DML.
Unlike binary DML, we do NOT require treatment to be {0, 1}.
We only check for sufficient variation.
"""
function _validate_continuous_inputs(
    outcomes::Vector{T},
    treatment::Vector{T},
    covariates::Matrix{T}
) where {T<:Real}
    n = length(outcomes)

    # Length consistency
    if length(treatment) != n
        throw(ArgumentError(
            "CRITICAL ERROR: Length mismatch.\n" *
            "Function: _validate_continuous_inputs\n" *
            "outcomes: $n, treatment: $(length(treatment))"
        ))
    end

    if size(covariates, 1) != n
        throw(ArgumentError(
            "CRITICAL ERROR: Covariate rows mismatch.\n" *
            "Function: _validate_continuous_inputs\n" *
            "outcomes: $n, covariate rows: $(size(covariates, 1))"
        ))
    end

    # Check for NaN/Inf
    if any(isnan.(outcomes)) || any(isinf.(outcomes))
        throw(ArgumentError(
            "CRITICAL ERROR: NaN or Inf in outcomes.\n" *
            "Function: _validate_continuous_inputs"
        ))
    end

    if any(isnan.(treatment)) || any(isinf.(treatment))
        throw(ArgumentError(
            "CRITICAL ERROR: NaN or Inf in treatment.\n" *
            "Function: _validate_continuous_inputs"
        ))
    end

    if any(isnan.(covariates)) || any(isinf.(covariates))
        throw(ArgumentError(
            "CRITICAL ERROR: NaN or Inf in covariates.\n" *
            "Function: _validate_continuous_inputs"
        ))
    end

    # Check for sufficient treatment variation (key difference from binary)
    treatment_std = std(treatment)
    if treatment_std < 1e-10
        throw(ArgumentError(
            "CRITICAL ERROR: No treatment variation.\n" *
            "Function: _validate_continuous_inputs\n" *
            "Treatment std = $(treatment_std)\n" *
            "Continuous DML requires variation in treatment."
        ))
    end

    # Minimum sample size
    if n < 4
        throw(ArgumentError(
            "CRITICAL ERROR: Insufficient sample size.\n" *
            "Function: _validate_continuous_inputs\n" *
            "n: $n (need at least 4)"
        ))
    end

    return nothing
end

# =============================================================================
# Cross-Fitting
# =============================================================================

"""
    _cross_fit_continuous_nuisance(X, y, D, n_folds, model)

Generate cross-fitted predictions for continuous treatment nuisance models.

Unlike binary DML which uses classification for propensity, we use
regression for both outcome E[Y|X] and treatment E[D|X].

Returns:
- m_hat: Cross-fitted outcome predictions E[Y|X]
- d_hat: Cross-fitted treatment predictions E[D|X]
- fold_info: Vector of (train_idx, test_idx) tuples
"""
function _cross_fit_continuous_nuisance(
    X::Matrix{T},
    y::Vector{T},
    D::Vector{T},
    n_folds::Int,
    model::Symbol
) where {T<:Real}
    n = length(y)
    m_hat = zeros(T, n)
    d_hat = zeros(T, n)
    fold_info = Vector{Tuple{Vector{Int}, Vector{Int}}}()

    # Random permutation for fold assignment (seed for reproducibility)
    Random.seed!(42)
    perm = randperm(n)
    fold_size = div(n, n_folds)

    # Assign each observation to a fold
    fold_indices = zeros(Int, n)
    for k in 1:n_folds
        start_idx = (k - 1) * fold_size + 1
        end_idx = k == n_folds ? n : k * fold_size
        fold_indices[perm[start_idx:end_idx]] .= k
    end

    # Add intercept to covariates
    X_int = hcat(ones(T, n), X)

    for k in 1:n_folds
        train_mask = fold_indices .!= k
        test_mask = fold_indices .== k

        train_idx = findall(train_mask)
        test_idx = findall(test_mask)
        push!(fold_info, (train_idx, test_idx))

        X_train = X_int[train_idx, :]
        X_test = X_int[test_idx, :]
        y_train = y[train_idx]
        D_train = D[train_idx]

        # Fit outcome model: E[Y|X]
        β_m, _, _ = fit_model(X_train, y_train, model)
        m_hat[test_idx] = X_test * β_m

        # Fit treatment model: E[D|X] (REGRESSION, not classification)
        β_d, _, _ = fit_model(X_train, D_train, model)
        d_hat[test_idx] = X_test * β_d
    end

    return m_hat, d_hat, fold_info
end

# =============================================================================
# Influence Function SE
# =============================================================================

"""
    _influence_function_se_continuous(Y_tilde, D_tilde, theta)

Compute standard error using influence function for continuous treatment.

For the partially linear model with continuous D, the influence function is:
ψ_i = (Y_tilde_i - θ * D_tilde_i) * D_tilde_i / E[D_tilde²]

The variance of θ̂ is Var(θ̂) = E[ψ²] / n
"""
function _influence_function_se_continuous(
    Y_tilde::Vector{T},
    D_tilde::Vector{T},
    theta::T
) where {T<:Real}
    n = length(Y_tilde)

    # Denominator: E[D_tilde²]
    D_tilde_sq_mean = mean(D_tilde .^ 2)

    if D_tilde_sq_mean < 1e-10
        # Fallback for degenerate case
        return std(Y_tilde) / sqrt(n)
    end

    # Influence function: ψ_i = (Y_tilde_i - θ * D_tilde_i) * D_tilde_i / E[D_tilde²]
    psi = (Y_tilde .- theta .* D_tilde) .* D_tilde ./ D_tilde_sq_mean

    # Variance of θ̂ = Var(ψ) / n
    var_theta = var(psi, corrected=true) / n

    return sqrt(var_theta)
end

# =============================================================================
# Per-Fold Estimates
# =============================================================================

"""
    _compute_fold_estimates(Y_tilde, D_tilde, fold_info)

Compute per-fold ATE estimates for stability analysis.
"""
function _compute_fold_estimates(
    Y_tilde::Vector{T},
    D_tilde::Vector{T},
    fold_info::Vector{Tuple{Vector{Int}, Vector{Int}}}
) where {T<:Real}
    n_folds = length(fold_info)
    fold_estimates = zeros(T, n_folds)
    fold_ses = zeros(T, n_folds)

    for (i, (_, test_idx)) in enumerate(fold_info)
        Y_fold = Y_tilde[test_idx]
        D_fold = D_tilde[test_idx]

        # Fold-specific estimate
        D_sq_sum = sum(D_fold .^ 2)
        if D_sq_sum > 1e-10
            fold_estimates[i] = sum(Y_fold .* D_fold) / D_sq_sum
            fold_ses[i] = _influence_function_se_continuous(
                Y_fold, D_fold, fold_estimates[i]
            )
        else
            fold_estimates[i] = NaN
            fold_ses[i] = NaN
        end
    end

    return fold_estimates, fold_ses
end

# =============================================================================
# Main Function
# =============================================================================

"""
    dml_continuous(outcomes, treatment, covariates; kwargs...)

Estimate treatment effects using Double ML with continuous treatment.

Implements the partially linear model with K-fold cross-fitting for
continuous (non-binary) treatment effects. This is also known as the
"dose-response" setting.

# Arguments
- `outcomes::Vector`: Outcome variable Y of shape (n,)
- `treatment::Vector`: Continuous treatment D of shape (n,). Can take any real values.
- `covariates::Matrix`: Covariate matrix X of shape (n, p)

# Keyword Arguments
- `n_folds::Int=5`: Number of folds for cross-fitting (must be >= 2)
- `model::Symbol=:ridge`: Model for nuisance estimation (:ols, :ridge)
- `cate_model::Symbol=:ols`: Model for CATE estimation (:ols, :ridge)
- `alpha::Float64=0.05`: Significance level for confidence intervals

# Returns
- `DMLContinuousResult`: Result containing ATE, SE, CATE, diagnostics

# Example
```julia
using CausalEstimators
using Random

Random.seed!(42)
n = 500
X = randn(n, 2)
D = X[:, 1] .+ randn(n)  # Continuous treatment
Y = 1.0 .+ X[:, 1] .+ 2.0 .* D .+ randn(n)  # True effect = 2

result = dml_continuous(Y, D, X)
println("ATE: \$(result.ate) ± \$(result.ate_se)")
```

# Notes
**Key Difference from Binary DML**:
- Binary DML: Uses P(T=1|X) via classification (propensity score)
- Continuous DML: Uses E[D|X] via regression (no propensity)

The treatment effect θ represents the marginal effect: θ = dE[Y|D,X] / dD
"""
function dml_continuous(
    outcomes::AbstractVector,
    treatment::AbstractVector,
    covariates::AbstractMatrix;
    n_folds::Int = 5,
    model::Symbol = :ridge,
    cate_model::Symbol = :ols,
    alpha::Float64 = 0.05
)::DMLContinuousResult
    # Convert inputs
    T = Float64
    outcomes_vec = convert(Vector{T}, outcomes)
    treatment_vec = convert(Vector{T}, treatment)
    covariates_mat = convert(Matrix{T}, covariates)

    # Validate inputs (no binary check)
    _validate_continuous_inputs(outcomes_vec, treatment_vec, covariates_mat)

    n = length(outcomes_vec)
    p = size(covariates_mat, 2)

    # Create estimator (validates n_folds and model)
    estimator = DMLContinuous(n_folds=n_folds, model=model, cate_model=cate_model)

    # =========================================================================
    # Step 1-2: Cross-fit nuisance models
    # =========================================================================
    m_hat, d_hat, fold_info = _cross_fit_continuous_nuisance(
        covariates_mat, outcomes_vec, treatment_vec, n_folds, model
    )

    # Compute R-squared for diagnostics
    ss_total_y = sum((outcomes_vec .- mean(outcomes_vec)) .^ 2)
    ss_resid_y = sum((outcomes_vec .- m_hat) .^ 2)
    outcome_r2 = ss_total_y > 0 ? 1.0 - ss_resid_y / ss_total_y : 0.0

    ss_total_d = sum((treatment_vec .- mean(treatment_vec)) .^ 2)
    ss_resid_d = sum((treatment_vec .- d_hat) .^ 2)
    treatment_r2 = ss_total_d > 0 ? 1.0 - ss_resid_d / ss_total_d : 0.0

    # =========================================================================
    # Step 3: Compute residuals
    # =========================================================================
    Y_tilde = outcomes_vec .- m_hat  # Outcome residual
    D_tilde = treatment_vec .- d_hat  # Treatment residual

    # =========================================================================
    # Step 4: Estimate ATE
    # =========================================================================
    D_tilde_sq_sum = sum(D_tilde .^ 2)

    if D_tilde_sq_sum < 1e-10
        throw(ArgumentError(
            "CRITICAL ERROR: Treatment residuals too small.\n" *
            "Function: dml_continuous\n" *
            "Sum of D_tilde² = $D_tilde_sq_sum\n" *
            "This indicates treatment is almost perfectly predicted by X.\n" *
            "Check that treatment has variation conditional on X."
        ))
    end

    ate = sum(Y_tilde .* D_tilde) / D_tilde_sq_sum

    # =========================================================================
    # Step 5: Estimate CATE(X) - heterogeneous effects
    # =========================================================================
    # For CATE, use weighted regression with D_tilde² weights
    X_int = hcat(ones(T, n), covariates_mat)
    X_transformed = covariates_mat .* D_tilde
    X_with_intercept = hcat(X_transformed, D_tilde)

    # Fit CATE model
    β_cate, _, _ = fit_model(X_with_intercept, Y_tilde, cate_model)

    # Predict CATE for each unit
    X_pred = hcat(covariates_mat, ones(T, n))
    cate = X_pred * β_cate

    # =========================================================================
    # Step 6: Standard error via influence function
    # =========================================================================
    ate_se = _influence_function_se_continuous(Y_tilde, D_tilde, ate)

    # Confidence interval
    z_crit = quantile(Normal(), 1 - alpha / 2)
    ci_lower = ate - z_crit * ate_se
    ci_upper = ate + z_crit * ate_se

    # =========================================================================
    # Step 7: Per-fold estimates for stability analysis
    # =========================================================================
    fold_estimates, fold_ses = _compute_fold_estimates(Y_tilde, D_tilde, fold_info)

    return DMLContinuousResult(
        cate,
        ate,
        ate_se,
        ci_lower,
        ci_upper,
        :dml_continuous,
        fold_estimates,
        fold_ses,
        outcome_r2,
        treatment_r2,
        n,
        n_folds
    )
end
