"""
Inverse probability weighting (IPW) estimator for ATE.
"""

"""
    IPWATE <: AbstractRCTEstimator

Inverse Probability Weighting (IPW) estimator for average treatment effect.

Reweights units by inverse of propensity scores to create balance. Works for RCTs
with varying assignment probabilities and observational studies (with estimated propensities).

# Mathematical Foundation

IPW creates a pseudo-population where treatment is independent of covariates:

```math
\\hat{\\tau}_{IPW} = \\frac{1}{n}\\sum_{i=1}^n \\left(\\frac{T_i Y_i}{e_i} - \\frac{(1-T_i) Y_i}{1-e_i}\\right)
```

where ``e_i = P(T_i=1|X_i)`` is the propensity score.

Alternatively (Horvitz-Thompson):

```math
\\hat{\\tau}_{IPW} = \\frac{\\sum_i w_i T_i Y_i}{\\sum_i w_i T_i} - \\frac{\\sum_i w_i (1-T_i) Y_i}{\\sum_i w_i (1-T_i)}
```

where ``w_i = 1/e_i`` for treated, ``w_i = 1/(1-e_i)`` for control.

# Variance Estimation

**Horvitz-Thompson Variance Estimator** (Horvitz & Thompson 1952):

We use the weighted variance formula accounting for inverse propensity weights:

```math
Var(\\hat{\\mu}) = \\frac{\\sum_i w_i^2 (Y_i - \\hat{\\mu})^2}{(\\sum_i w_i)^2}
```

for each treatment group separately, then:

```math
Var(\\hat{\\tau}_{IPW}) = Var(\\hat{\\mu}_1) + Var(\\hat{\\mu}_0)
```

**Key properties**:
- **Conservative**: Valid under correct propensity model
- **Accounts for weighting**: Variance increases with weight variability
- **Extreme weights warning**: Large weights (propensity near 0 or 1) inflate variance
- **No homoskedasticity assumption**: Allows heterogeneous treatment effects

**Practical considerations**:
- Trimming propensities near 0/1 may reduce variance (at cost of bias)
- Alternative: augmented IPW (AIPW) for double robustness
- Alternative: stabilized weights to reduce variance

# Usage

```julia
# Simple RCT (constant propensity)
outcomes = [10.0, 12.0, 4.0, 5.0]
treatment = [true, true, false, false]
propensity = [0.5, 0.5, 0.5, 0.5]  # Balanced randomization

# Store propensity in covariates field (will extract in solve())
problem = RCTProblem(outcomes, treatment, hcat(propensity), nothing, (alpha=0.05,))
solution = solve(problem, IPWATE())

# Varying propensity (blocked RCT)
propensity = [0.8, 0.6, 0.6, 0.8]
problem = RCTProblem(outcomes, treatment, hcat(propensity), nothing, (alpha=0.05,))
solution = solve(problem, IPWATE())
```

# Requirements

- Propensity scores must be provided in problem (as first column of covariates)
- Propensity scores must be in (0, 1) exclusive
- For observational studies, propensities should be estimated separately (e.g., logistic regression)

# Benefits

- Valid under correct propensity model (double robustness with outcome model)
- Handles varying assignment probabilities
- Foundation for more advanced methods (AIPW, TMLE)

# Limitations

- Large weights (extreme propensities) increase variance
- Sensitive to propensity model misspecification
- Trimming may be needed for extreme propensities

# References

- Horvitz, D. G., & Thompson, D. J. (1952). A generalization of sampling without
  replacement from a finite universe. *JASA*, 47(260), 663-685.
- Hirano, K., Imbens, G. W., & Ridder, G. (2003). Efficient Estimation of Average
  Treatment Effects Using the Estimated Propensity Score. *Econometrica*, 71(4), 1161-1189.
"""
struct IPWATE <: AbstractRCTEstimator end

"""
    solve(problem::RCTProblem, estimator::IPWATE)::RCTSolution

Compute IPW-adjusted ATE using Horvitz-Thompson weighted means with robust variance.

Reweights units by inverse of propensity scores to create balance. Under correct
propensity model, IPW provides consistent ATE estimates.

# Algorithm

1. Extract propensity scores from covariates (first column)
2. Validate propensity scores in (0, 1) exclusive
3. Compute IPW weights:
   - Treated: w_i = 1 / propensity_i
   - Control: w_i = 1 / (1 - propensity_i)
4. Compute weighted means for treated and control
5. ATE = weighted_mean_treated - weighted_mean_control
6. Compute robust variance accounting for weights
7. Construct confidence interval

# Mathematical Foundation

**IPW weights**: w_i = 1/e_i for treated, w_i = 1/(1-e_i) for control

**Weighted means**:
- μ₁ = Σ(w_i * Y_i * T_i) / Σ(w_i * T_i)
- μ₀ = Σ(w_i * Y_i * (1-T_i)) / Σ(w_i * (1-T_i))

**Robust variance**: Var(μ) = Σ(w_i² * (Y_i - μ)²) / (Σw_i)²

# Validation

- Covariates field must be populated (propensity scores in first column)
- Propensity scores must be in (0, 1) exclusive
- Treatment variation validated in RCTProblem constructor

# Examples

```julia
# Simple RCT (constant propensity = 0.5)
outcomes = [10.0, 12.0, 4.0, 5.0]
treatment = [true, true, false, false]
propensity = [0.5, 0.5, 0.5, 0.5]
problem = RCTProblem(outcomes, treatment, hcat(propensity), nothing, (alpha=0.05,))
solution = solve(problem, IPWATE())

# Varying propensity (blocked RCT)
propensity = [0.8, 0.6, 0.6, 0.8]
problem = RCTProblem(outcomes, treatment, hcat(propensity), nothing, (alpha=0.05,))
solution = solve(problem, IPWATE())
```

# References

- Horvitz & Thompson (1952). Generalization of sampling without replacement.
- Hirano, Imbens, & Ridder (2003). Efficient estimation using propensity scores.
"""
function solve(problem::RCTProblem, estimator::IPWATE)::RCTSolution
    (; outcomes, treatment, covariates, parameters) = problem

    # ============================================================================
    # Extract and Validate Propensity Scores
    # ============================================================================

    # Validate covariates field is populated
    if isnothing(covariates)
        throw(
            ArgumentError(
                "CRITICAL ERROR: Covariates field is nothing.\n" *
                "Function: solve(IPWATE)\n" *
                "IPWATE requires propensity scores in first column of covariates.\n" *
                "For simple RCT with p=0.5, use: covariates = hcat(fill(0.5, n))",
            ),
        )
    end

    # Extract propensity scores (first column)
    propensity = if ndims(covariates) == 1
        covariates  # Already a vector
    else
        covariates[:, 1]  # Extract first column
    end

    # Validate propensity scores in (0, 1) exclusive
    if any(propensity .<= 0) || any(propensity .>= 1)
        throw(
            ArgumentError(
                "CRITICAL ERROR: Propensity scores must be in (0,1) exclusive.\n" *
                "Function: solve(IPWATE)\n" *
                "Propensity scores represent probabilities and cannot be 0, 1, or outside.\n" *
                "Got: min=$(minimum(propensity)), max=$(maximum(propensity))\n" *
                "Valid range: (0, 1) exclusive",
            ),
        )
    end

    n = length(outcomes)
    n_treated = sum(treatment)
    n_control = sum(.!treatment)

    # ============================================================================
    # IPW Weights
    # ============================================================================

    # Treated: w_i = 1 / propensity_i
    # Control: w_i = 1 / (1 - propensity_i)
    weights = [t ? 1 / p : 1 / (1 - p) for (t, p) in zip(treatment, propensity)]

    # ============================================================================
    # Weighted Means
    # ============================================================================

    treated_mask = treatment
    control_mask = .!treatment

    # Treated mean
    weighted_sum_treated = sum(weights[treated_mask] .* outcomes[treated_mask])
    sum_weights_treated = sum(weights[treated_mask])
    mean_treated = weighted_sum_treated / sum_weights_treated

    # Control mean
    weighted_sum_control = sum(weights[control_mask] .* outcomes[control_mask])
    sum_weights_control = sum(weights[control_mask])
    mean_control = weighted_sum_control / sum_weights_control

    # IPW ATE estimate
    ate = mean_treated - mean_control

    # ============================================================================
    # Robust Variance Estimation
    # ============================================================================

    # Variance of weighted mean: Var(μ) = Σ(w_i² * (Y_i - μ)²) / (Σw_i)²

    # Treated variance
    residuals_treated = outcomes[treated_mask] .- mean_treated
    var_treated =
        sum((weights[treated_mask] .^ 2) .* (residuals_treated .^ 2)) /
        (sum_weights_treated^2)

    # Control variance
    residuals_control = outcomes[control_mask] .- mean_control
    var_control =
        sum((weights[control_mask] .^ 2) .* (residuals_control .^ 2)) /
        (sum_weights_control^2)

    # Variance of ATE
    var_ate = var_treated + var_control
    se = sqrt(var_ate)

    # ============================================================================
    # Confidence Interval
    # ============================================================================

    ci_lower, ci_upper = confidence_interval(ate, se, parameters.alpha)

    return RCTSolution(
        estimate = ate,
        se = se,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        n_treated = n_treated,
        n_control = n_control,
        retcode = :Success,
        original_problem = problem,
    )
end
