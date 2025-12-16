#=
Inverse Probability Weighting (IPW) Estimator for Observational Studies

Implements the IPW ATE estimator with:
- Automatic propensity score estimation
- Optional propensity trimming
- Optional stabilized weights
- Robust (sandwich) standard errors
- Comprehensive diagnostics

References:
- Rosenbaum & Rubin (1983). The central role of the propensity score.
- Hirano, Imbens, & Ridder (2003). Efficient estimation of ATE using propensity.
- Austin & Stuart (2015). Moving towards best practice with IPTW.
=#

"""
    ObservationalIPW <: AbstractObservationalEstimator

Inverse Probability Weighting estimator for observational studies.

# Mathematical Formulation

The IPW estimator reweights observations to create a pseudo-population
where treatment is independent of covariates:

## Horvitz-Thompson (HT) Estimator
    τ̂_HT = (1/n) Σᵢ [Tᵢ Yᵢ / e(Xᵢ) - (1-Tᵢ) Yᵢ / (1-e(Xᵢ))]

## Hájek (Normalized) Estimator
    τ̂_Hájek = [Σᵢ Tᵢ Yᵢ / e(Xᵢ)] / [Σᵢ Tᵢ / e(Xᵢ)] -
               [Σᵢ (1-Tᵢ) Yᵢ / (1-e(Xᵢ))] / [Σᵢ (1-Tᵢ) / (1-e(Xᵢ))]

The Hájek estimator (default) is more stable when weights don't sum to n.

# Algorithm

1. Estimate propensity scores e(X) via logistic regression (if not provided)
2. Optionally trim extreme propensities (remove observations outside quantiles)
3. Compute IPW weights (standard or stabilized)
4. Estimate ATE using weighted means
5. Compute robust sandwich standard errors
6. Return solution with diagnostics

# Key Assumptions

1. **Unconfoundedness**: (Y(0), Y(1)) ⟂ T | X
2. **Positivity**: 0 < e(X) < 1 for all X in support
3. **SUTVA**: No interference, no hidden treatments
4. **Correct propensity model**: Logistic specification is correct

# Example
```julia
using CausalEstimators

# Create observational problem
problem = ObservationalProblem(outcomes, treatment, covariates)

# Solve with IPW
solution = solve(problem, ObservationalIPW())

# Access results
println("ATE: \$(solution.estimate) ± \$(solution.se)")
println("Propensity AUC: \$(solution.propensity_auc)")
```

# References
- Hirano, K., Imbens, G. W., & Ridder, G. (2003). Efficient estimation of average
  treatment effects using the estimated propensity score. Econometrica, 71(4), 1161-1189.
"""
struct ObservationalIPW <: AbstractObservationalEstimator end


"""
    solve(problem::ObservationalProblem, estimator::ObservationalIPW)

Estimate ATE using Inverse Probability Weighting.

# Returns
`IPWSolution` with:
- Point estimate
- Robust standard error
- Confidence interval
- Propensity diagnostics (AUC, overlap)
- IPW weights used
"""
function solve(
    problem::ObservationalProblem{T,P},
    estimator::ObservationalIPW
)::IPWSolution{T} where {T<:Real,P<:NamedTuple}
    # Extract data from problem
    (; outcomes, treatment, covariates, propensity, parameters) = problem

    alpha = get(parameters, :alpha, 0.05)
    trim_threshold = get(parameters, :trim_threshold, 0.01)
    stabilize = get(parameters, :stabilize, false)

    n = length(outcomes)
    n_treated_original = sum(treatment)
    n_control_original = n - n_treated_original

    # =========================================================================
    # Step 1: Estimate or validate propensity scores
    # =========================================================================

    if propensity === nothing
        # Estimate propensity scores from covariates
        prop_result = estimate_propensity_scores(treatment, covariates)
        propensity_scores = prop_result.propensity
    else
        propensity_scores = copy(propensity)
    end

    # =========================================================================
    # Step 2: Trim extreme propensities (if threshold > 0)
    # =========================================================================

    n_trimmed = 0
    Y = copy(outcomes)
    T_vec = copy(treatment)
    X = copy(covariates)
    e = copy(propensity_scores)

    if trim_threshold > 0
        trim_result = trim_propensities(
            e, T_vec, Y, X;
            trim_at = (trim_threshold, 1 - trim_threshold)
        )

        Y = trim_result.outcomes
        T_vec = trim_result.treatment
        X = trim_result.covariates
        e = trim_result.propensity
        n_trimmed = trim_result.n_trimmed
    end

    n_effective = length(Y)
    n_treated = sum(T_vec)
    n_control = n_effective - n_treated

    # Check we still have both groups after trimming
    if n_treated == 0
        throw(ArgumentError(
            "No treated units remain after propensity trimming (threshold=$trim_threshold). " *
            "Consider reducing trim_threshold or checking propensity model."
        ))
    end

    if n_control == 0
        throw(ArgumentError(
            "No control units remain after propensity trimming (threshold=$trim_threshold). " *
            "Consider reducing trim_threshold or checking propensity model."
        ))
    end

    # =========================================================================
    # Step 3: Compute IPW weights
    # =========================================================================

    weights = compute_ipw_weights(e, T_vec; stabilize=stabilize)

    # =========================================================================
    # Step 4: Compute IPW estimate (Hájek estimator)
    # =========================================================================

    # Weighted mean for treated
    w_treated = weights[T_vec]
    y_treated = Y[T_vec]
    mu1_hat = sum(w_treated .* y_treated) / sum(w_treated)

    # Weighted mean for control
    w_control = weights[.!T_vec]
    y_control = Y[.!T_vec]
    mu0_hat = sum(w_control .* y_control) / sum(w_control)

    # ATE estimate
    ate_estimate = mu1_hat - mu0_hat

    # =========================================================================
    # Step 5: Compute robust (sandwich) standard error
    # =========================================================================

    # Influence function approach
    # φᵢ = [Tᵢ(Yᵢ - μ₁)/e(Xᵢ) - (1-Tᵢ)(Yᵢ - μ₀)/(1-e(Xᵢ))] + (μ₁ - μ₀)

    influence = zeros(T, n_effective)

    for i in 1:n_effective
        if T_vec[i]
            # Treated
            influence[i] = (Y[i] - mu1_hat) / e[i]
        else
            # Control
            influence[i] = -(Y[i] - mu0_hat) / (1 - e[i])
        end
    end

    # Robust variance (sandwich estimator)
    var_ate = var(influence) / n_effective
    se = sqrt(var_ate)

    # =========================================================================
    # Step 6: Inference
    # =========================================================================

    z_alpha = quantile(Normal(), 1 - alpha / 2)
    ci_lower = ate_estimate - z_alpha * se
    ci_upper = ate_estimate + z_alpha * se

    # Two-sided p-value
    z_stat = ate_estimate / se
    p_value = 2 * (1 - cdf(Normal(), abs(z_stat)))

    # =========================================================================
    # Step 7: Propensity diagnostics
    # =========================================================================

    # AUC for propensity model discriminatory power
    propensity_auc = compute_propensity_auc(e, T_vec)

    # Mean propensity by group
    propensity_mean_treated = mean(e[T_vec])
    propensity_mean_control = mean(e[.!T_vec])

    # =========================================================================
    # Step 8: Determine return code
    # =========================================================================

    retcode = :Success

    # Warnings for potential issues
    if propensity_auc > 0.9
        @warn "Propensity AUC = $(round(propensity_auc, digits=3)) > 0.9 indicates " *
              "near-perfect treatment prediction. Check for positivity violations."
        retcode = :Warning
    end

    if n_trimmed > 0.2 * n
        @warn "Trimmed $n_trimmed observations ($(round(100*n_trimmed/n, digits=1))%). " *
              "High trimming may introduce selection bias."
        retcode = :Warning
    end

    max_weight = maximum(weights)
    if max_weight > 20
        @warn "Maximum IPW weight = $(round(max_weight, digits=1)) is large. " *
              "Consider stabilized weights or additional trimming."
        retcode = :Warning
    end

    # =========================================================================
    # Step 9: Build solution
    # =========================================================================

    return IPWSolution{T}(
        ate_estimate,
        se,
        ci_lower,
        ci_upper,
        p_value,
        n_treated,
        n_control,
        n_trimmed,
        e,
        weights,
        propensity_auc,
        propensity_mean_treated,
        propensity_mean_control,
        stabilize,
        retcode,
        problem
    )
end
