#=
Neural Double Machine Learning with K-Fold Cross-Fitting

Session 155: Julia Neural CATE Parity

Implements Double ML using polynomial feature expansion to approximate
neural network nuisance models. Uses K-fold cross-fitting to eliminate
regularization bias and ensure valid inference.

Key Insight:
Cross-fitting prevents overfitting bias by ensuring predictions are always
made on held-out data. This allows using flexible ML models (approximated
by polynomial features) while maintaining valid asymptotic inference.

References:
- Chernozhukov et al. (2018). "Double/debiased machine learning for treatment
  and structural parameters." The Econometrics Journal 21(1): C1-C68.
=#

using LinearAlgebra
using Statistics
using Distributions
using Random


# =============================================================================
# Estimator Type
# =============================================================================

"""
    NeuralDoubleMachineLearning <: AbstractCATEEstimator

Neural Double ML with K-fold cross-fitting.

Uses polynomial features to approximate neural network nuisance models,
with cross-fitting to eliminate regularization bias.

# Fields
- `n_folds::Int`: Number of cross-fitting folds (default: 5)
- `degree::Int`: Polynomial degree for feature expansion (default: 2)
- `lambda::Float64`: Ridge regularization parameter (default: 0.01)

# Algorithm
1. Split data into K folds
2. For each fold k:
   - Train ĝ(X) = E[Y|X] on data EXCLUDING fold k
   - Train ê(X) = E[T|X] on data EXCLUDING fold k
   - Predict on fold k (out-of-sample)
3. Pool cross-fitted residuals:
   - Ỹ = Y - ĝ(X)
   - T̃ = T - ê(X)
4. Final stage: estimate τ(X) via weighted regression

# Why Cross-Fitting?
Without cross-fitting, regularized models produce biased predictions that
propagate into ATE estimates. Cross-fitting ensures each prediction is
out-of-sample, eliminating this "regularization bias".

# Example
```julia
problem = CATEProblem(Y, T, X, (alpha=0.05,))
solution = solve(problem, NeuralDoubleMachineLearning(n_folds=5, degree=2))
```
"""
struct NeuralDoubleMachineLearning <: AbstractCATEEstimator
    n_folds::Int
    degree::Int
    lambda::Float64

    function NeuralDoubleMachineLearning(;
        n_folds::Int = 5,
        degree::Int = 2,
        lambda::Float64 = 0.01
    )
        if n_folds < 2
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid NeuralDoubleMachineLearning configuration.\n" *
                "Function: NeuralDoubleMachineLearning\n" *
                "n_folds must be >= 2, got $n_folds"
            ))
        end
        if degree < 1
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid NeuralDoubleMachineLearning configuration.\n" *
                "Function: NeuralDoubleMachineLearning\n" *
                "degree must be >= 1, got $degree"
            ))
        end
        if lambda < 0
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid NeuralDoubleMachineLearning configuration.\n" *
                "Function: NeuralDoubleMachineLearning\n" *
                "lambda must be >= 0, got $lambda"
            ))
        end
        new(n_folds, degree, lambda)
    end
end


# =============================================================================
# K-Fold Cross-Fitting Utilities
# =============================================================================

"""
    _create_folds(n, n_folds; shuffle=true, rng=nothing) -> Vector{Tuple}

Create K-fold indices for cross-fitting.

# Returns
Vector of (train_idx, test_idx) tuples.
"""
function _create_folds(
    n::Int,
    n_folds::Int;
    shuffle::Bool = true,
    rng::Union{Nothing, AbstractRNG} = nothing
)
    if isnothing(rng)
        rng = Random.GLOBAL_RNG
    end

    indices = collect(1:n)
    if shuffle
        Random.shuffle!(rng, indices)
    end

    fold_size = n ÷ n_folds
    remainder = n % n_folds

    folds = Vector{Tuple{Vector{Int}, Vector{Int}}}()
    start_idx = 1

    for k in 1:n_folds
        # Distribute remainder across first folds
        end_idx = start_idx + fold_size - 1 + (k <= remainder ? 1 : 0)
        test_idx = indices[start_idx:end_idx]
        train_idx = setdiff(indices, test_idx)

        push!(folds, (train_idx, test_idx))
        start_idx = end_idx + 1
    end

    return folds
end


# =============================================================================
# Cross-Fitted Prediction
# =============================================================================

"""
    _cross_fit_outcome(X_poly, Y, folds, lambda) -> Y_hat

Cross-fitted outcome predictions: ĝ(X) = E[Y|X]
"""
function _cross_fit_outcome(
    X_poly::Matrix{T},
    Y::Vector{T},
    folds::Vector{Tuple{Vector{Int}, Vector{Int}}},
    lambda::Float64
) where {T<:Real}
    n = length(Y)
    Y_hat = zeros(T, n)

    for (train_idx, test_idx) in folds
        # Fit on training data
        X_train = X_poly[train_idx, :]
        Y_train = Y[train_idx]

        coef = _neural_ridge_fit(X_train, Y_train; lambda=lambda)

        # Predict on test fold (out-of-sample)
        X_test = X_poly[test_idx, :]
        Y_hat[test_idx] = _neural_ridge_predict(X_test, coef)
    end

    return Y_hat
end


"""
    _cross_fit_propensity(X_poly, treatment, folds, lambda) -> e_hat

Cross-fitted propensity predictions: ê(X) = P(T=1|X)
"""
function _cross_fit_propensity(
    X_poly::Matrix{T},
    treatment::Vector{Bool},
    folds::Vector{Tuple{Vector{Int}, Vector{Int}}},
    lambda::Float64
) where {T<:Real}
    n = length(treatment)
    e_hat = zeros(T, n)

    for (train_idx, test_idx) in folds
        # Fit propensity on training data
        X_train = X_poly[train_idx, :]
        T_train = treatment[train_idx]

        n_train = size(X_train, 1)
        X_train_int = hcat(ones(T, n_train), X_train)

        coef = _fit_logistic_ridge(X_train_int, T_train; lambda=lambda)

        # Predict on test fold
        n_test = length(test_idx)
        X_test = X_poly[test_idx, :]
        X_test_int = hcat(ones(T, n_test), X_test)

        eta = X_test_int * coef
        e_hat[test_idx] = clamp.(_sigmoid.(eta), T(0.01), T(0.99))
    end

    return e_hat
end


# =============================================================================
# Main Solve Function
# =============================================================================

"""
    solve(problem::CATEProblem, estimator::NeuralDoubleMachineLearning) -> CATESolution

Estimate CATE using Neural Double ML with cross-fitting.

# Cross-Fitting Process
For each fold k:
1. Train outcome model ĝ(X) on folds ≠ k
2. Train propensity model ê(X) on folds ≠ k
3. Predict ĝ(Xₖ), ê(Xₖ) on fold k (out-of-sample)

This ensures all predictions are out-of-sample, eliminating regularization bias.

# Double Robustness
The resulting estimator is doubly robust: consistent if EITHER:
- The outcome model ĝ(X) is correctly specified, OR
- The propensity model ê(X) is correctly specified
"""
function solve(
    problem::CATEProblem{T,P},
    estimator::NeuralDoubleMachineLearning
)::CATESolution{T,P} where {T<:Real, P<:NamedTuple}
    (; outcomes, treatment, covariates, parameters) = problem
    alpha = get(parameters, :alpha, 0.05)

    n = length(outcomes)

    # Validate sample size for cross-fitting
    min_fold_size = n ÷ estimator.n_folds
    if min_fold_size < 5
        throw(ArgumentError(
            "CRITICAL ERROR: Insufficient sample size for cross-fitting.\n" *
            "Function: NeuralDoubleMachineLearning\n" *
            "n=$n with $(estimator.n_folds) folds gives fold size $min_fold_size < 5"
        ))
    end

    # Step 1: Create polynomial features
    X_poly = _create_features(covariates; degree=estimator.degree)
    p_poly = size(X_poly, 2)

    # Step 2: Create K-fold splits
    folds = _create_folds(n, estimator.n_folds; shuffle=true)

    # Step 3: Cross-fit nuisance models
    # Outcome model: ĝ(X) = E[Y|X]
    Y_hat = _cross_fit_outcome(X_poly, outcomes, folds, estimator.lambda)

    # Propensity model: ê(X) = P(T=1|X)
    e_hat = _cross_fit_propensity(X_poly, treatment, folds, estimator.lambda)

    # Step 4: Compute residuals
    Y_resid = outcomes - Y_hat
    T_float = T.(treatment)
    T_resid = T_float - e_hat

    # Step 5: Estimate τ(X) via weighted regression on pooled residuals
    # Minimize: Σ (Ỹᵢ - τ(Xᵢ)·T̃ᵢ)²

    weights = T_resid .^ 2
    valid_idx = abs.(T_resid) .> T(0.05)
    n_valid = sum(valid_idx)

    if n_valid < p_poly + 2
        # Fallback: constant ATE
        ate = T(sum(Y_resid .* T_resid) / (sum(weights) + T(1e-10)))
        cate = fill(ate, n)
    else
        # Heterogeneous CATE via weighted regression
        pseudo_outcome = Y_resid[valid_idx] ./ T_resid[valid_idx]
        X_valid = X_poly[valid_idx, :]
        w_valid = weights[valid_idx]

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

    # Step 6: Compute ATE and SE
    ate = T(mean(cate))

    # SE via influence function
    se = compute_influence_se(Y_resid, T_resid, ate)

    # Ensure SE is positive and reasonable
    if !isfinite(se) || se <= 0
        se = compute_ate_se_simple(cate)
    end

    z_crit = T(quantile(Normal(), 1 - alpha / 2))
    ci_lower = ate - z_crit * se
    ci_upper = ate + z_crit * se

    return CATESolution{T,P}(
        cate, ate, se, ci_lower, ci_upper,
        :neural_dml, :Success, problem
    )
end
