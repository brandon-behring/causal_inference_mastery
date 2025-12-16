#=
R-Learner (Robinson Transformation) Implementation

The R-Learner uses residualization to achieve double robustness.
Named after Robinson (1988)'s partially linear model.

Algorithm:
1. Fit ê(X) = P(T=1|X) [propensity]
2. Fit m̂(X) = E[Y|X] [outcome]
3. Compute residuals: Ỹ = Y - m̂(X), T̃ = T - ê(X)
4. θ̂ = Σ(Ỹ·T̃) / Σ(T̃²)

References:
- Nie & Wager (2021). "Quasi-oracle estimation of heterogeneous treatment effects"
- Robinson (1988). "Root-N-consistent semiparametric regression"
=#

"""
    solve(problem::CATEProblem, estimator::RLearner)::CATESolution

Estimate CATE using R-Learner (Robinson transformation).

# Algorithm
1. Estimate propensity: ê(X) = P(T=1|X)
2. Estimate outcome regression: m̂(X) = E[Y|X]
3. Compute residuals:
   - Outcome: Ỹ = Y - m̂(X)
   - Treatment: T̃ = T - ê(X)
4. Estimate ATE via: θ̂ = Σ(Ỹ·T̃) / Σ(T̃²)
5. SE via influence function

# Double Robustness
The R-learner is doubly robust: consistent if EITHER:
- Propensity model ê(X) is correctly specified, OR
- Outcome model m̂(X) is correctly specified

# Arguments
- `problem::CATEProblem`: CATE estimation problem
- `estimator::RLearner`: R-Learner estimator with model choice

# Returns
- `CATESolution`: Results including individual CATE, ATE, SE, and CI

# Notes
For heterogeneous effects, the R-learner recovers the average effect.
Individual CATE estimates are derived from the residualized regression.
"""
function solve(
    problem::CATEProblem{T,P},
    estimator::RLearner
)::CATESolution{T,P} where {T<:Real, P<:NamedTuple}
    # Extract from problem
    (; outcomes, treatment, covariates, parameters) = problem
    alpha = get(parameters, :alpha, 0.05)

    n = length(outcomes)
    p = size(covariates, 2)

    # =========================================================================
    # Step 1: Estimate propensity scores
    # =========================================================================

    propensity = estimate_propensity(covariates, treatment)

    # =========================================================================
    # Step 2: Estimate outcome regression m̂(X) = E[Y|X]
    # =========================================================================

    X_int = add_intercept(covariates)
    β_m, m_hat, _ = fit_model(X_int, outcomes, estimator.model)

    # =========================================================================
    # Step 3: Compute residuals
    # =========================================================================

    # Outcome residuals: Ỹ = Y - m̂(X)
    Y_resid = outcomes - m_hat

    # Treatment residuals: T̃ = T - ê(X)
    T_float = T.(treatment)
    T_resid = T_float - propensity

    # =========================================================================
    # Step 4: Estimate treatment effect via Robinson transformation
    # =========================================================================

    # θ̂ = Σ(Ỹ·T̃) / Σ(T̃²)
    numerator = sum(Y_resid .* T_resid)
    denominator = sum(T_resid .^ 2)

    if abs(denominator) < 1e-10
        @warn "R-Learner: Near-zero treatment variation after residualization"
        ate = T(0)
    else
        ate = T(numerator / denominator)
    end

    # =========================================================================
    # Step 5: Compute individual CATE
    # =========================================================================

    # For the R-learner with constant treatment effect, CATE = ATE for all
    # For heterogeneous effects, we would fit τ(X) to minimize:
    # Σ(Ỹ - τ(X)·T̃)²
    #
    # With linear model for τ(X) = X'γ:
    # This becomes weighted regression of Ỹ on T̃·X

    # Simple case: constant effect
    cate = fill(ate, n)

    # For heterogeneous CATE, fit weighted regression
    # Weight by T̃² to account for propensity variation
    weights = T_resid .^ 2

    if sum(weights) > 1e-10
        # Weighted regression: minimize Σ wᵢ(Ỹᵢ - τ(Xᵢ)·T̃ᵢ)²
        # Transform: Ỹ/T̃ ≈ τ(X) where we regress on X
        # Only use observations with sufficient T̃

        valid_idx = abs.(T_resid) .> 0.05
        if sum(valid_idx) > p + 1
            Y_transformed = Y_resid[valid_idx] ./ T_resid[valid_idx]
            X_valid = X_int[valid_idx, :]
            w_valid = weights[valid_idx]

            # Weighted OLS
            W = Diagonal(sqrt.(w_valid))
            X_w = W * X_valid
            Y_w = W * Y_transformed

            try
                β_tau = X_w \ Y_w
                cate = X_int * β_tau
            catch
                # Fallback to constant effect
                cate = fill(ate, n)
            end
        end
    end

    # =========================================================================
    # Step 6: Compute SE via influence function
    # =========================================================================

    se = compute_influence_se(Y_resid, T_resid, ate)

    # Confidence interval
    df = n - (p + 1)
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
        :r_learner,
        :Success,
        problem
    )
end
