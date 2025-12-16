#=
Double Machine Learning (DML) Implementation

DML uses K-fold cross-fitting to eliminate regularization bias when using
machine learning methods for nuisance parameter estimation.

Algorithm:
1. Split data into K folds
2. For each fold k:
   - Train nuisance models (propensity ê, outcome m̂) on OTHER folds
   - Predict for fold k using out-of-sample models
3. Compute residuals using cross-fitted predictions
4. Estimate treatment effect via Robinson transformation

Reference:
- Chernozhukov et al. (2018). "Double/debiased machine learning"
=#

"""
    solve(problem::CATEProblem, estimator::DoubleMachineLearning)::CATESolution

Estimate CATE using Double Machine Learning with K-fold cross-fitting.

# Algorithm
1. Randomly partition data into K folds
2. For each fold k:
   - Train propensity model ê(X) on folds {1,...,K}\\{k}
   - Train outcome model m̂(X) on folds {1,...,K}\\{k}
   - Predict ê(Xₖ), m̂(Xₖ) for fold k (out-of-sample)
3. Compute cross-fitted residuals:
   - Ỹ = Y - m̂₋ₖ(X) (outcome residual with cross-fit prediction)
   - T̃ = T - ê₋ₖ(X) (treatment residual with cross-fit prediction)
4. Estimate θ̂ = Σ(Ỹ·T̃) / Σ(T̃²)
5. SE via influence function

# Why Cross-Fitting Matters
Without cross-fitting (standard R-learner), regularization in nuisance
models can introduce bias in the treatment effect estimate. Cross-fitting
eliminates this "regularization bias" by ensuring predictions are always
out-of-sample.

# Arguments
- `problem::CATEProblem`: CATE estimation problem
- `estimator::DoubleMachineLearning`: DML estimator with n_folds and model

# Returns
- `CATESolution`: Results including individual CATE, ATE, SE, and CI

# Example
```julia
using CausalEstimators
using Random

Random.seed!(42)
n = 500
X = randn(n, 5)
propensity_true = 1 ./ (1 .+ exp.(-0.5 .* X[:, 1]))
T = rand(n) .< propensity_true
Y = 1.0 .+ X[:, 1] .+ 2.0 .* T .+ randn(n)

problem = CATEProblem(Y, T, X, (alpha=0.05,))
solution = solve(problem, DoubleMachineLearning(n_folds=5))

println("DML ATE: \$(solution.ate) ± \$(solution.se)")
```
"""
function solve(
    problem::CATEProblem{T,P},
    estimator::DoubleMachineLearning
)::CATESolution{T,P} where {T<:Real, P<:NamedTuple}
    # Extract from problem
    (; outcomes, treatment, covariates, parameters) = problem
    alpha = get(parameters, :alpha, 0.05)
    n_folds = estimator.n_folds

    n = length(outcomes)
    p = size(covariates, 2)

    # =========================================================================
    # Step 1: Create fold indices
    # =========================================================================

    # Random permutation for fold assignment
    perm = randperm(n)
    fold_size = div(n, n_folds)

    # Assign each observation to a fold
    fold_indices = zeros(Int, n)
    for k in 1:n_folds
        start_idx = (k - 1) * fold_size + 1
        end_idx = k == n_folds ? n : k * fold_size
        fold_indices[perm[start_idx:end_idx]] .= k
    end

    # =========================================================================
    # Step 2: Cross-fit nuisance models
    # =========================================================================

    # Storage for cross-fitted predictions
    m_hat_cf = zeros(T, n)  # Cross-fitted outcome predictions
    e_hat_cf = zeros(T, n)  # Cross-fitted propensity predictions

    X_int = add_intercept(covariates)

    for k in 1:n_folds
        # Training set: all folds except k
        train_idx = fold_indices .!= k
        test_idx = fold_indices .== k

        # Training data
        Y_train = outcomes[train_idx]
        T_train = treatment[train_idx]
        X_train = X_int[train_idx, :]
        X_cov_train = covariates[train_idx, :]

        # Test data
        X_test = X_int[test_idx, :]
        X_cov_test = covariates[test_idx, :]

        # --- Fit outcome model m̂(X) on training set ---
        β_m, _, _ = fit_model(X_train, Y_train, estimator.model)
        m_hat_cf[test_idx] = predict_ols(X_test, β_m)

        # --- Fit propensity model ê(X) on training set ---
        e_hat_k = estimate_propensity(X_cov_train, T_train)

        # Predict propensity for test set
        # Need to re-fit logistic on training and predict on test
        e_hat_cf[test_idx] = _predict_propensity(X_cov_train, T_train, X_cov_test)
    end

    # =========================================================================
    # Step 3: Compute cross-fitted residuals
    # =========================================================================

    # Outcome residuals: Ỹ = Y - m̂₋ₖ(X)
    Y_resid = outcomes - m_hat_cf

    # Treatment residuals: T̃ = T - ê₋ₖ(X)
    T_float = T.(treatment)
    T_resid = T_float - e_hat_cf

    # =========================================================================
    # Step 4: Estimate treatment effect
    # =========================================================================

    # θ̂ = Σ(Ỹ·T̃) / Σ(T̃²)
    numerator = sum(Y_resid .* T_resid)
    denominator = sum(T_resid .^ 2)

    if abs(denominator) < 1e-10
        @warn "DML: Near-zero treatment variation after residualization"
        ate = T(0)
    else
        ate = T(numerator / denominator)
    end

    # =========================================================================
    # Step 5: Compute individual CATE
    # =========================================================================

    # For DML with constant effect assumption: CATE = ATE for all
    cate = fill(ate, n)

    # For heterogeneous effects, could fit τ(X) model on residualized data
    # Similar to R-learner but using cross-fitted residuals
    weights = T_resid .^ 2

    if sum(weights) > 1e-10
        valid_idx = abs.(T_resid) .> 0.05
        if sum(valid_idx) > p + 1
            Y_transformed = Y_resid[valid_idx] ./ T_resid[valid_idx]
            X_valid = X_int[valid_idx, :]
            w_valid = weights[valid_idx]

            W = Diagonal(sqrt.(w_valid))
            X_w = W * X_valid
            Y_w = W * Y_transformed

            try
                β_tau = X_w \ Y_w
                cate = X_int * β_tau
            catch
                cate = fill(ate, n)
            end
        end
    end

    # =========================================================================
    # Step 6: Compute SE via influence function
    # =========================================================================

    se = compute_influence_se(Y_resid, T_resid, ate)

    # Confidence interval
    df = n - (p + 1) - n_folds  # Adjust for cross-fitting
    if df < 1
        df = 1
    end
    t_crit = quantile(TDist(df), 1 - alpha / 2)
    ci_lower = ate - t_crit * se
    ci_upper = ate + t_crit * se

    return CATESolution{T,P}(
        cate,
        ate,
        se,
        ci_lower,
        ci_upper,
        :dml,
        :Success,
        problem
    )
end

"""
    _predict_propensity(X_train, T_train, X_test)

Fit propensity model on training data and predict for test data.
"""
function _predict_propensity(
    X_train::Matrix{T},
    T_train::Vector{Bool},
    X_test::Matrix{T}
) where {T<:Real}
    n_train = size(X_train, 1)
    n_test = size(X_test, 1)

    # Add intercept
    X_train_int = add_intercept(X_train)
    X_test_int = add_intercept(X_test)

    # Fit logistic regression via IRLS
    p = size(X_train_int, 2)
    β = zeros(p)
    y_float = Float64.(T_train)

    for iter in 1:100
        η = X_train_int * β
        μ = _sigmoid.(η)
        μ = clamp.(μ, 1e-10, 1 - 1e-10)

        W = μ .* (1 .- μ)
        z = η .+ (y_float .- μ) ./ W

        W_sqrt = sqrt.(W)
        X_w = W_sqrt .* X_train_int
        z_w = W_sqrt .* z

        β_new = X_w \ z_w

        if norm(β_new - β) < 1e-6
            β = β_new
            break
        end
        β = β_new
    end

    # Predict on test set
    η_test = X_test_int * β
    propensity = _sigmoid.(η_test)
    propensity = clamp.(propensity, 0.01, 0.99)

    return propensity
end
