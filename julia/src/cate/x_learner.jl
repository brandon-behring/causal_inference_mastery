#=
X-Learner (Cross-Learner) Implementation

The X-Learner uses propensity-weighted combination of imputed treatment effects.
Particularly effective when treatment groups are imbalanced.

Algorithm:
1. Fit μ₀, μ₁ (T-learner step)
2. Compute imputed effects:
   D₁ = Y₁ - μ̂₀(X₁) for treated
   D₀ = μ̂₁(X₀) - Y₀ for control
3. Fit τ₁(X) → D₁, τ₀(X) → D₀
4. CATE(x) = ê(x)·τ̂₀(x) + (1-ê(x))·τ̂₁(x)

Reference:
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects"
=#

"""
    solve(problem::CATEProblem, estimator::XLearner)::CATESolution

Estimate CATE using X-Learner (Cross-learner approach).

# Algorithm
1. Fit outcome models μ₀, μ₁ (like T-learner)
2. Compute imputed treatment effects:
   - For treated: D₁ᵢ = Y₁ᵢ - μ̂₀(X₁ᵢ) (observed - counterfactual)
   - For control: D₀ᵢ = μ̂₁(X₀ᵢ) - Y₀ᵢ (counterfactual - observed)
3. Fit CATE models: τ̂₁(X) → D₁, τ̂₀(X) → D₀
4. Estimate propensity: ê(X) = P(T=1|X)
5. Combine: τ̂(x) = ê(x)·τ̂₀(x) + (1-ê(x))·τ̂₁(x)

# Arguments
- `problem::CATEProblem`: CATE estimation problem
- `estimator::XLearner`: X-Learner estimator with model choice

# Returns
- `CATESolution`: Results including individual CATE, ATE, SE, and CI

# Notes
The propensity weighting in step 5 gives more weight to:
- τ̂₀ (learned from controls) for high-propensity units (more like treated)
- τ̂₁ (learned from treated) for low-propensity units (more like control)

This adaptive weighting makes X-learner effective for imbalanced groups.
"""
function solve(
    problem::CATEProblem{T,P},
    estimator::XLearner
)::CATESolution{T,P} where {T<:Real, P<:NamedTuple}
    # Extract from problem
    (; outcomes, treatment, covariates, parameters) = problem
    alpha = get(parameters, :alpha, 0.05)

    n = length(outcomes)
    p = size(covariates, 2)

    # =========================================================================
    # Step 1: Split data and fit outcome models (T-learner step)
    # =========================================================================

    treated_idx = treatment
    control_idx = .!treatment

    Y_1 = outcomes[treated_idx]
    Y_0 = outcomes[control_idx]

    X_1 = covariates[treated_idx, :]
    X_0 = covariates[control_idx, :]

    n_1 = sum(treated_idx)
    n_0 = sum(control_idx)

    # Fit outcome models with intercept
    X_1_int = add_intercept(X_1)
    X_0_int = add_intercept(X_0)

    β_1, _, _ = fit_model(X_1_int, Y_1, estimator.model)
    β_0, _, _ = fit_model(X_0_int, Y_0, estimator.model)

    # =========================================================================
    # Step 2: Compute imputed treatment effects
    # =========================================================================

    # For treated: D₁ = Y₁ - μ̂₀(X₁)
    μ_0_at_1 = predict_ols(X_1_int, β_0)  # Counterfactual control outcome for treated
    D_1 = Y_1 - μ_0_at_1

    # For control: D₀ = μ̂₁(X₀) - Y₀
    μ_1_at_0 = predict_ols(X_0_int, β_1)  # Counterfactual treated outcome for control
    D_0 = μ_1_at_0 - Y_0

    # =========================================================================
    # Step 3: Fit CATE models on imputed effects
    # =========================================================================

    # τ̂₁(X) trained on treated imputed effects
    β_tau_1, _, _ = fit_model(X_1_int, D_1, estimator.model)

    # τ̂₀(X) trained on control imputed effects
    β_tau_0, _, _ = fit_model(X_0_int, D_0, estimator.model)

    # =========================================================================
    # Step 4: Estimate propensity scores
    # =========================================================================

    propensity = estimate_propensity(covariates, treatment)

    # =========================================================================
    # Step 5: Combine using propensity weighting
    # =========================================================================

    # Predict CATE from both models for all units
    X_all_int = add_intercept(covariates)

    tau_1_all = predict_ols(X_all_int, β_tau_1)  # CATE from treated model
    tau_0_all = predict_ols(X_all_int, β_tau_0)  # CATE from control model

    # X-learner combination: τ̂(x) = ê(x)·τ̂₀(x) + (1-ê(x))·τ̂₁(x)
    cate = propensity .* tau_0_all .+ (1 .- propensity) .* tau_1_all

    # =========================================================================
    # Step 6: Compute ATE and SE
    # =========================================================================

    ate = mean(cate)
    se = compute_ate_se(cate, treatment, outcomes, covariates)

    # Confidence interval
    df = min(n_0, n_1) - (p + 1)
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
        :x_learner,
        :Success,
        problem
    )
end
