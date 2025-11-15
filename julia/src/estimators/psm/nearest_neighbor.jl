"""
Nearest Neighbor Propensity Score Matching estimator.
"""

"""
    NearestNeighborPSM <: AbstractPSMEstimator

Nearest neighbor propensity score matching estimator.

Estimates average treatment effect (ATE) by:
1. Estimating propensity scores P(T=1|X) via logistic regression
2. Matching each treated unit to M nearest control units by propensity score
3. Computing ATE as average difference between treated and matched control outcomes
4. Computing standard errors via Abadie-Imbens (2006, 2008) analytic variance

# Fields
- `M::Int`: Number of matches per treated unit (default: 1)
- `with_replacement::Bool`: Allow reusing controls (default: false)
- `caliper::Float64`: Maximum propensity score distance (default: Inf, no restriction)
- `variance_method::Symbol`: Variance estimator (:abadie_imbens or :bootstrap, default: :abadie_imbens)

# Method

## Step 1: Propensity Score Estimation
```
logit(e(X)) = β₀ + β'X
e(X) = P(T=1|X) = 1/(1 + exp(-β₀ - β'X))
```

## Step 2: Nearest Neighbor Matching
For each treated unit i, find M controls j with closest |e(Xᵢ) - e(Xⱼ)|

## Step 3: ATE Estimation
```
ATE = (1/N₁) ∑ᵢ∈{T=1} [Yᵢ - (1/M) ∑ⱼ∈Matches(i) Yⱼ]
```

## Step 4: Variance Estimation
- **Abadie-Imbens** (default): Analytic variance accounting for matching uncertainty
- **Bootstrap** (NOT recommended with replacement): Pairs bootstrap (FAILS for with-replacement per Abadie & Imbens 2008)

# Example
```julia
# 1:1 matching without replacement
estimator = NearestNeighborPSM(M=1, with_replacement=false, caliper=0.1)
solution = solve(problem, estimator)

# 2:1 matching with replacement
estimator = NearestNeighborPSM(M=2, with_replacement=true)
solution = solve(problem, estimator)
```

# Notes
- **Caliper matching**: Setting caliper < Inf drops units outside common support
- **With replacement**: More efficient but requires Abadie-Imbens variance (bootstrap FAILS)
- **Without replacement**: Greedy algorithm (match order matters, consider randomizing)
- **M:1 matching**: Larger M reduces variance but may worsen bias

# References
- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in
  observational studies for causal effects. *Biometrika*, 70(1), 41-55.
- Abadie, A., & Imbens, G. W. (2006). Large sample properties of matching estimators for
  average treatment effects. *Econometrica*, 74(1), 235-267.
- Abadie, A., & Imbens, G. W. (2008). On the failure of the bootstrap for matching estimators.
  *Econometrica*, 76(6), 1537-1557.
"""
struct NearestNeighborPSM <: AbstractPSMEstimator
    M::Int
    with_replacement::Bool
    caliper::Float64
    variance_method::Symbol

    function NearestNeighborPSM(;
        M::Int = 1,
        with_replacement::Bool = false,
        caliper::Float64 = Inf,
        variance_method::Symbol = :abadie_imbens,
    )
        # Validate parameters
        if M < 1
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Invalid M.\\n" *
                    "M must be ≥ 1, got M = $M\\n" *
                    "M is the number of matches per treated unit.",
                ),
            )
        end

        if caliper <= 0 && caliper != Inf
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Invalid caliper.\\n" *
                    "caliper must be > 0 or Inf, got caliper = $caliper\\n" *
                    "Typical values: 0.1 (strict), 0.25 (moderate), Inf (no restriction).",
                ),
            )
        end

        if variance_method ∉ [:abadie_imbens, :bootstrap]
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Invalid variance_method.\\n" *
                    "Must be :abadie_imbens or :bootstrap, got :$variance_method\\n" *
                    "Recommended: :abadie_imbens (bootstrap FAILS for with_replacement).",
                ),
            )
        end

        # Warning for bootstrap with replacement
        if with_replacement && variance_method == :bootstrap
            @warn """
            Bootstrap variance is INVALID for matching with replacement.
            See Abadie & Imbens (2008): Bootstrap fails for with-replacement matching.
            Recommend using variance_method = :abadie_imbens instead.
            """
        end

        new(M, with_replacement, caliper, variance_method)
    end
end


"""
    solve(problem::PSMProblem, estimator::NearestNeighborPSM)

Estimate ATE using nearest neighbor propensity score matching.

# Arguments
- `problem::PSMProblem`: Problem specification
- `estimator::NearestNeighborPSM`: Estimator configuration

# Returns
- `solution::PSMSolution`: Solution with ATE, SE, CI, diagnostics

# Method
1. Estimate propensity scores via logistic regression
2. Check common support (overlap in propensity distributions)
3. Match treated units to nearest controls
4. Compute ATE from matched sample
5. Compute variance via Abadie-Imbens or bootstrap
6. Construct confidence intervals

# Retcodes
- `:Success`: Estimation successful
- `:CommonSupportFailed`: Insufficient overlap in propensity scores
- `:MatchingFailed`: No matches found (caliper too restrictive?)
- `:ConvergenceFailed`: Propensity score estimation failed

# Example
```julia
problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
estimator = NearestNeighborPSM(M=1, with_replacement=false, caliper=0.1)
solution = solve(problem, estimator)

if solution.retcode == :Success
    println("ATE: \$(solution.estimate) ± \$(solution.se)")
    println("95% CI: [\$(solution.ci_lower), \$(solution.ci_upper)]")
    println("Matched: \$(solution.n_matched) pairs")
end
```
"""
function solve(problem::PSMProblem, estimator::NearestNeighborPSM)
    # Extract problem data
    outcomes = problem.outcomes
    treatment = problem.treatment
    covariates = problem.covariates
    alpha = haskey(problem.parameters, :alpha) ? problem.parameters.alpha : 0.05

    n = length(outcomes)
    n_treated = sum(treatment)
    n_control = n - n_treated

    # ========================================================================
    # Step 1: Estimate Propensity Scores
    # ========================================================================

    propensity = try
        estimate_propensity(treatment, covariates)
    catch e
        # Propensity estimation failed (convergence, perfect separation, etc.)
        return PSMSolution(
            NaN,
            NaN,
            NaN,
            NaN,
            n_treated,
            n_control,
            0,
            fill(NaN, n),
            Tuple{Int,Int}[],
            (;),  # Empty balance metrics
            :ConvergenceFailed,
            problem,
        )
    end

    # ========================================================================
    # Step 2: Check Common Support
    # ========================================================================

    has_support, support_region, n_outside = check_common_support(propensity, treatment)

    if !has_support
        return PSMSolution(
            NaN,
            NaN,
            NaN,
            NaN,
            n_treated,
            n_control,
            0,
            propensity,
            Tuple{Int,Int}[],
            (; support_region, n_outside),
            :CommonSupportFailed,
            problem,
        )
    end

    # ========================================================================
    # Step 3: Nearest Neighbor Matching
    # ========================================================================

    indices_treated = findall(treatment)
    indices_control = findall(.!treatment)

    matches, distances, n_matched = nearest_neighbor_match(
        propensity[treatment],
        propensity[.!treatment],
        indices_treated,
        indices_control,
        M = estimator.M,
        with_replacement = estimator.with_replacement,
        caliper = estimator.caliper,
    )

    if n_matched == 0
        return PSMSolution(
            NaN,
            NaN,
            NaN,
            NaN,
            n_treated,
            n_control,
            0,
            propensity,
            Tuple{Int,Int}[],
            (; support_region, n_outside),
            :MatchingFailed,
            problem,
        )
    end

    # ========================================================================
    # Step 4: Compute ATE
    # ========================================================================

    ate, treated_outcomes, control_outcomes = compute_ate_from_matches(
        outcomes,
        treatment,
        matches,
    )

    # ========================================================================
    # Step 5: Compute Variance
    # ========================================================================

    variance, se = try
        if estimator.variance_method == :abadie_imbens
            abadie_imbens_variance(
                outcomes,
                treatment,
                covariates,
                propensity,
                matches,
                M = estimator.M,
                with_replacement = estimator.with_replacement,
            )
        else  # :bootstrap
            pairs_bootstrap_variance(
                outcomes,
                treatment,
                propensity,
                matches,
                B = 1000,
                M = estimator.M,
            )
        end
    catch e
        # Variance computation failed (likely due to unmatched units with without-replacement)
        # Cannot quantify uncertainty, so don't report estimate
        # This can happen when caliper is very strict and some units remain unmatched
        return PSMSolution(
            NaN,  # No valid ATE (can't quantify uncertainty)
            NaN,  # No valid SE
            NaN,
            NaN,
            n_treated,
            n_control,
            n_matched,
            propensity,
            Tuple{Int,Int}[],
            (; support_region, n_outside, n_matched_actual=n_matched),
            :MatchingFailed,
            problem,
        )
    end

    # ========================================================================
    # Step 6: Confidence Intervals
    # ========================================================================

    # Normal approximation
    z_crit = quantile(Normal(0, 1), 1 - alpha / 2)
    ci_lower = ate - z_crit * se
    ci_upper = ate + z_crit * se

    # ========================================================================
    # Step 7: Create matched pairs indices
    # ========================================================================

    matched_indices = Tuple{Int,Int}[]
    for i in 1:n_treated
        for j in matches[i]
            push!(matched_indices, (indices_treated[i], j))
        end
    end

    # ========================================================================
    # Step 8: Balance Diagnostics
    # ========================================================================

    # Compute balance metrics (MEDIUM-5: verify balance on ALL covariates)
    balanced, smd_after, vr_after, smd_before, vr_before = check_covariate_balance(
        covariates,
        treatment,
        matched_indices,
        threshold = 0.1,
    )

    # Summary statistics
    balance_stats = balance_summary(smd_after, vr_after, smd_before, vr_before)

    # Comprehensive balance metrics
    balance_metrics = (;
        support_region,
        n_outside,
        mean_distance = mean(vcat(distances...)),
        max_distance = maximum(vcat(distances...)),
        balanced,
        smd_after,
        vr_after,
        smd_before,
        vr_before,
        balance_stats,
    )

    # ========================================================================
    # Return Solution
    # ========================================================================

    return PSMSolution(
        ate,
        se,
        ci_lower,
        ci_upper,
        n_treated,
        n_control,
        n_matched,
        propensity,
        matched_indices,
        balance_metrics,
        :Success,
        problem,
    )
end
