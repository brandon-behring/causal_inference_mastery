#=
Doubly Robust (AIPW) Estimator for Observational Studies

Implements the Augmented Inverse Probability Weighting estimator that
combines propensity weighting with outcome modeling for double robustness.

Key Property (Double Robustness):
- If propensity model correct → consistent (even if outcome model wrong)
- If outcome model correct → consistent (even if propensity model wrong)
- If both correct → consistent AND efficient
- If both wrong → biased

References:
- Robins, Rotnitzky & Zhao (1994). Estimation of regression coefficients.
- Bang & Robins (2005). Doubly robust estimation.
- Kennedy (2016). Semiparametric theory and empirical processes.
=#

"""
    DoublyRobust <: AbstractObservationalEstimator

Doubly robust (AIPW) estimator for average treatment effect.

Uses both propensity score and outcome modeling, providing consistency
if EITHER model is correctly specified.

# Example
```julia
problem = ObservationalProblem(Y, T, X)
solution = solve(problem, DoublyRobust())
```
"""
struct DoublyRobust <: AbstractObservationalEstimator end


"""
    solve(problem::ObservationalProblem, estimator::DoublyRobust) -> DRSolution

Estimate ATE using doubly robust (AIPW) method.

# Algorithm

1. Estimate propensity scores e(X) via logistic regression
2. Fit outcome models μ₀(X) and μ₁(X) via linear regression
3. (Optional) Trim extreme propensities
4. Compute AIPW estimate:

   τ̂_DR = (1/n) Σᵢ [
       Tᵢ/e(Xᵢ) * (Yᵢ - μ₁(Xᵢ)) + μ₁(Xᵢ)
     - (1-Tᵢ)/(1-e(Xᵢ)) * (Yᵢ - μ₀(Xᵢ)) - μ₀(Xᵢ)
   ]

5. Compute variance via influence function

# Arguments
- `problem::ObservationalProblem`: Problem specification with outcomes, treatment, covariates
- `estimator::DoublyRobust`: Doubly robust estimator

# Returns
- `DRSolution`: Solution with ATE estimate, SE, CI, and diagnostics

# Example
```julia
# Generate observational data with confounding
n = 500
X = randn(n, 2)
logit = 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2]
e_true = 1 ./ (1 .+ exp.(-logit))
T = rand(n) .< e_true
Y = 2.0 .* T .+ 0.5 .* X[:, 1] .+ randn(n)

problem = ObservationalProblem(Y, T, X)
solution = solve(problem, DoublyRobust())

println("ATE: \$(solution.estimate) ± \$(solution.se)")
```

# Notes
- Propensity scores clipped at [1e-6, 1-1e-6] for numerical stability
- Uses t-distribution for CI with n < 50, normal for larger samples
- Influence function variance provides robust standard errors
"""
function solve(
    problem::ObservationalProblem{T},
    estimator::DoublyRobust
)::DRSolution{T} where {T<:Real}

    # Extract data
    outcomes = problem.outcomes
    treatment = problem.treatment
    covariates = problem.covariates
    alpha = problem.parameters.alpha
    trim_threshold = problem.parameters.trim_threshold

    n = length(outcomes)

    # =========================================================================
    # Step 1: Estimate Propensity Scores
    # =========================================================================

    if problem.propensity !== nothing
        # Use provided propensity scores
        propensity = copy(problem.propensity)
    else
        # Estimate via logistic regression
        prop_result = estimate_propensity_scores(treatment, covariates)
        propensity = prop_result.propensity
    end

    # =========================================================================
    # Step 2: Trim Extreme Propensities (Optional)
    # =========================================================================

    n_trimmed = 0
    if trim_threshold > 0
        trim_result = trim_propensities(
            propensity, treatment, outcomes, covariates;
            trim_at = (trim_threshold, 1 - trim_threshold)
        )
        propensity = trim_result.propensity
        treatment = trim_result.treatment
        outcomes = trim_result.outcomes
        covariates = trim_result.covariates
        n_trimmed = trim_result.n_trimmed
        n = length(outcomes)
    end

    # Clip propensity scores for numerical stability
    epsilon = T(1e-6)
    propensity_clipped = clamp.(propensity, epsilon, 1 - epsilon)

    # =========================================================================
    # Step 3: Fit Outcome Models
    # =========================================================================

    outcome_result = fit_outcome_models(outcomes, treatment, covariates)
    mu0_predictions = outcome_result.mu0_predictions
    mu1_predictions = outcome_result.mu1_predictions
    mu0_r2 = outcome_result.mu0_r2
    mu1_r2 = outcome_result.mu1_r2

    # =========================================================================
    # Step 4: Compute Doubly Robust Estimator
    # =========================================================================

    # Convert treatment to numeric for computation
    T_numeric = convert(Vector{T}, treatment)

    # AIPW formula:
    # τ̂_DR = (1/n) Σᵢ [
    #     Tᵢ/e(Xᵢ) * (Yᵢ - μ₁(Xᵢ)) + μ₁(Xᵢ)
    #   - (1-Tᵢ)/(1-e(Xᵢ)) * (Yᵢ - μ₀(Xᵢ)) - μ₀(Xᵢ)
    # ]

    treated_contribution = @. (
        T_numeric / propensity_clipped * (outcomes - mu1_predictions) + mu1_predictions
    )

    control_contribution = @. (
        (1 - T_numeric) / (1 - propensity_clipped) * (outcomes - mu0_predictions) + mu0_predictions
    )

    dr_estimate = mean(treated_contribution .- control_contribution)

    # =========================================================================
    # Step 5: Compute Variance via Influence Function
    # =========================================================================

    # Influence function:
    # IF_i = T/e(X) * (Y - μ₁(X)) + μ₁(X)
    #      - (1-T)/(1-e(X)) * (Y - μ₀(X)) - μ₀(X) - τ̂_DR

    influence_function = treated_contribution .- control_contribution .- dr_estimate

    # Variance: Var(τ̂_DR) = (1/n) * mean(IF_i²)
    variance = mean(influence_function.^2) / n
    se = sqrt(variance)

    # =========================================================================
    # Step 6: Confidence Interval and p-value
    # =========================================================================

    # Use t-distribution for small samples
    if n < 50
        df = n - 2  # Approximate DoF
        critical = quantile(TDist(df), 1 - alpha / 2)
        p_value = 2 * ccdf(TDist(df), abs(dr_estimate / se))
    else
        critical = quantile(Normal(), 1 - alpha / 2)
        p_value = 2 * ccdf(Normal(), abs(dr_estimate / se))
    end

    ci_lower = dr_estimate - critical * se
    ci_upper = dr_estimate + critical * se

    # =========================================================================
    # Step 7: Compute Diagnostics
    # =========================================================================

    propensity_auc = compute_propensity_auc(propensity_clipped, treatment)

    n_treated = sum(treatment)
    n_control = n - n_treated

    # =========================================================================
    # Step 8: Construct Solution
    # =========================================================================

    return DRSolution{T}(
        T(dr_estimate),
        T(se),
        T(ci_lower),
        T(ci_upper),
        T(p_value),
        n_treated,
        n_control,
        n_trimmed,
        propensity_clipped,
        mu0_predictions,
        mu1_predictions,
        propensity_auc,
        mu0_r2,
        mu1_r2,
        :Success,
        problem
    )
end
