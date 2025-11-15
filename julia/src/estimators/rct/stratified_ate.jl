"""
Stratified difference-in-means estimator for blocked randomization designs.
"""

"""
    StratifiedATE <: AbstractRCTEstimator

Stratified difference-in-means estimator for blocked randomization designs.

Computes weighted average of stratum-specific ATEs, where weights are
proportional to stratum sizes.

# Mathematical Foundation

For ``S`` strata, the stratified ATE is:

```math
\\hat{\\tau}_{strat} = \\sum_{s=1}^S w_s \\hat{\\tau}_s
```

where ``w_s = n_s / n`` is the weight for stratum ``s`` (proportion of sample),
and ``\\hat{\\tau}_s`` is the simple ATE within stratum ``s``.

Variance:

```math
Var(\\hat{\\tau}_{strat}) = \\sum_{s=1}^S w_s^2 Var(\\hat{\\tau}_s)
```

# Variance Estimation

**Precision-Weighted Variance** (Imbens & Rubin 2015, Ch. 9):
- Each stratum uses Neyman heteroskedasticity-robust variance
- Overall variance = sum of weighted stratum variances (weights squared)
- Accounts for heteroskedasticity both within and between strata
- More efficient than simple ATE when outcomes vary by stratum
- Conservative: valid even if treatment effects vary by stratum

# Usage

```julia
# Blocked randomization data
outcomes = [100.0, 105.0, 10.0, 15.0]  # Two strata with different baselines
treatment = [true, false, true, false]
strata = [1, 1, 2, 2]

problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha=0.05,))
solution = solve(problem, StratifiedATE())
```

# Requirements

- Problem must have `strata` field populated
- Each stratum must have both treated and control units

# References

- Imbens & Rubin (2015), Chapter 9: Stratified Randomized Experiments
"""
struct StratifiedATE <: AbstractRCTEstimator end

"""
    solve(problem::RCTProblem, estimator::StratifiedATE)::RCTSolution

Compute stratified ATE by weighted average across strata.

Removes between-stratum variation, reducing standard errors compared to
simple difference-in-means when outcomes vary by stratum.

# Algorithm

1. For each stratum:
   - Compute stratum-specific ATE (difference-in-means)
   - Compute Neyman variance for stratum
   - Weight by stratum size (n_stratum / n_total)
2. Overall ATE = weighted sum of stratum ATEs
3. Overall variance = sum of (weight² × variance_stratum)

# Validation

- Strata field must be populated (not nothing)
- Each stratum must have both treated and control units
"""
function solve(problem::RCTProblem, estimator::StratifiedATE)::RCTSolution
    (; outcomes, treatment, strata, parameters) = problem

    # Validate strata field is populated
    if isnothing(strata)
        throw(
            ArgumentError(
                "CRITICAL ERROR: Strata field is nothing.\n" *
                "Function: solve(StratifiedATE)\n" *
                "StratifiedATE requires strata to be populated.\n" *
                "Use SimpleATE for unstratified designs.",
            ),
        )
    end

    n = length(outcomes)
    unique_strata = unique(strata)
    n_strata = length(unique_strata)

    # Initialize storage for stratum-level results
    stratum_estimates = Float64[]
    stratum_ses = Float64[]
    stratum_weights = Float64[]

    # Compute ATE for each stratum
    for stratum_id in unique_strata
        # Get data for this stratum
        stratum_mask = strata .== stratum_id
        y_stratum = outcomes[stratum_mask]
        t_stratum = treatment[stratum_mask]

        n_stratum = length(y_stratum)

        # Validate treatment variation within stratum
        treated_mask = t_stratum
        control_mask = .!t_stratum

        if !any(treated_mask)
            throw(
                ArgumentError(
                    "CRITICAL ERROR: No treated units in stratum $stratum_id.\n" *
                    "Function: solve(StratifiedATE)\n" *
                    "Cannot estimate treatment effect without treated group in each stratum.\n" *
                    "Stratum $stratum_id has all units in control.",
                ),
            )
        end

        if !any(control_mask)
            throw(
                ArgumentError(
                    "CRITICAL ERROR: No control units in stratum $stratum_id.\n" *
                    "Function: solve(StratifiedATE)\n" *
                    "Cannot estimate treatment effect without control group in each stratum.\n" *
                    "Stratum $stratum_id has all units treated.",
                ),
            )
        end

        # Stratum-specific ATE
        y1_stratum = y_stratum[treated_mask]
        y0_stratum = y_stratum[control_mask]

        n1_stratum = length(y1_stratum)
        n0_stratum = length(y0_stratum)

        ate_stratum = mean(y1_stratum) - mean(y0_stratum)

        # Stratum-specific Neyman variance
        # When n=1, variance is undefined, so use 0 (following Python implementation)
        var1 = n1_stratum > 1 ? var(y1_stratum) : 0.0
        var0 = n0_stratum > 1 ? var(y0_stratum) : 0.0
        var_stratum = var1 / n1_stratum + var0 / n0_stratum
        se_stratum = sqrt(var_stratum)

        # Weight by stratum size
        weight = n_stratum / n

        push!(stratum_estimates, ate_stratum)
        push!(stratum_ses, se_stratum)
        push!(stratum_weights, weight)
    end

    # Overall stratified ATE (weighted average)
    ate = sum(stratum_estimates .* stratum_weights)

    # Overall variance (sum of weighted variances)
    var_ate = sum((stratum_weights .^ 2) .* (stratum_ses .^ 2))
    se = sqrt(var_ate)

    # Confidence interval
    ci_lower, ci_upper = confidence_interval(ate, se, parameters.alpha)

    # Total counts
    n_treated = sum(treatment)
    n_control = sum(.!treatment)

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
