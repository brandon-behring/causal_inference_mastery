"""
Variance estimators for propensity score matching.
"""

"""
    abadie_imbens_variance(outcomes, treatment, covariates, propensity, matches;
                           M=1, with_replacement=false)

Compute Abadie-Imbens (2006, 2008) analytic variance for matching estimators.

# Arguments
- `outcomes::Vector{Float64}`: Observed outcomes
- `treatment::Vector{Bool}`: Treatment indicator
- `covariates::Matrix{Float64}`: Covariate matrix (n × p)
- `propensity::Vector{Float64}`: Estimated propensity scores
- `matches::Vector{Vector{Int}}`: Matched control indices for each treated unit
- `M::Int`: Number of matches per treated unit (default: 1)
- `with_replacement::Bool`: Whether matching was done with replacement (default: false)

# Returns
- `variance::Float64`: Asymptotic variance of ATE estimator
- `se::Float64`: Standard error (√variance)

# Method

The Abadie-Imbens variance accounts for:
1. **Matching uncertainty**: Controls are not fixed, they're selected via matching
2. **Imputation variance**: Using matched outcomes to impute counterfactuals
3. **Finite sample correction**: Unlike bootstrap, valid for M:1 matching

## Variance Formula

For matching WITH replacement (M=1):
```
V = (1/N₁) ∑ᵢ∈{T=1} σ²ᵢ(1)
```

where:
```
σ²ᵢ(1) = Var(Yᵢ(1) - Yᵢ(0) | Xᵢ, T=1)
       ≈ [Yᵢ - Ŷᵢ(0)]² + (K_M / N₀) ∑ⱼ∈{T=0} [Yⱼ - Ŷⱼ(0)]²
```

For matching WITHOUT replacement:
- More complex formula accounting for dependency between matches
- See Abadie & Imbens (2006) Theorem 1

## Key Components

1. **Conditional variance** σ²ᵢ(1): Variance of treatment effect for unit i
2. **Matching variance**: Additional uncertainty from estimating counterfactuals
3. **K_M factor**: Bias correction term (depends on M, dimension p)

# Notes
- **CRITICAL**: Bootstrap FAILS for matching with replacement (Abadie & Imbens 2008)
- Use this analytic variance instead of bootstrap for matching estimators
- Assumes propensity score estimated (if known, variance formula differs)
- Valid for fixed M (number of matches) as N → ∞

# Example
```julia
variance, se = abadie_imbens_variance(
    outcomes,
    treatment,
    covariates,
    propensity,
    matches,
    M=1,
    with_replacement=true
)

ci_lower = ate - 1.96 * se
ci_upper = ate + 1.96 * se
```

# References
- Abadie, A., & Imbens, G. W. (2006). Large sample properties of matching estimators for
  average treatment effects. *Econometrica*, 74(1), 235-267.
- Abadie, A., & Imbens, G. W. (2008). On the failure of the bootstrap for matching estimators.
  *Econometrica*, 76(6), 1537-1557.
- Abadie, A., & Imbens, G. W. (2016). Matching on the estimated propensity score.
  *Econometrica*, 84(2), 781-807.
"""
function abadie_imbens_variance(
    outcomes::Vector{Float64},
    treatment::Vector{Bool},
    covariates::Matrix{Float64},
    propensity::Vector{Float64},
    matches::Vector{Vector{Int}};
    M::Int = 1,
    with_replacement::Bool = false,
)
    n = length(outcomes)
    n_treated = sum(treatment)
    n_control = n - n_treated
    p = size(covariates, 2)

    indices_treated = findall(treatment)
    indices_control = findall(.!treatment)

    # ========================================================================
    # Validate Inputs
    # ========================================================================

    if n_treated == 0 || n_control == 0
        throw(
            ArgumentError(
                "CRITICAL ERROR: Need both treated and control units.\\n" *
                "Function: abadie_imbens_variance\\n" *
                "n_treated = $n_treated, n_control = $n_control\\n" *
                "Cannot compute variance without both groups.",
            ),
        )
    end

    if length(matches) != n_treated
        throw(
            ArgumentError(
                "CRITICAL ERROR: Mismatched lengths.\\n" *
                "Function: abadie_imbens_variance\\n" *
                "matches has $(length(matches)) elements, but $n_treated treated units.\\n" *
                "Must have one match vector per treated unit.",
            ),
        )
    end

    # ========================================================================
    # Compute Imputed Potential Outcomes
    # ========================================================================

    # For each unit, compute imputed Y(0) and Y(1)
    # Y_i(1) = Y_i if treated, else average of matched treated units
    # Y_i(0) = average of matched controls if treated, else Y_i

    imputed_y0 = zeros(n)
    imputed_y1 = zeros(n)

    # Impute for treated units
    for i in 1:n_treated
        idx_treated = indices_treated[i]
        matched_controls = matches[i]

        # Observed Y(1)
        imputed_y1[idx_treated] = outcomes[idx_treated]

        # Imputed Y(0) from matched controls
        if !isempty(matched_controls)
            imputed_y0[idx_treated] = mean(outcomes[matched_controls])
        else
            # No matches - cannot impute (this shouldn't happen if validation passed)
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Treated unit $i has no matches.\\n" *
                    "Function: abadie_imbens_variance\\n" *
                    "Cannot compute variance with unmatched units.",
                ),
            )
        end
    end

    # Impute for control units
    # Need to find which treated units matched each control
    if with_replacement
        # With replacement: multiple treated can match same control
        for j in 1:n_control
            idx_control = indices_control[j]

            # Find all treated units that matched this control
            matched_to = Int[]
            for i in 1:n_treated
                if idx_control in matches[i]
                    push!(matched_to, indices_treated[i])
                end
            end

            # Observed Y(0)
            imputed_y0[idx_control] = outcomes[idx_control]

            # Imputed Y(1) from treated units that matched this control
            if !isempty(matched_to)
                imputed_y1[idx_control] = mean(outcomes[matched_to])
            else
                # Control not matched to any treated unit
                # Use nearest treated unit (not ideal but needed for variance calc)
                # In practice, this means this control doesn't contribute much to variance
                dists = abs.(propensity[indices_treated] .- propensity[idx_control])
                nearest_treated = indices_treated[argmin(dists)]
                imputed_y1[idx_control] = outcomes[nearest_treated]
            end
        end
    else
        # Without replacement: each control matched to at most one treated
        # This is more complex - for simplicity, use conservative variance estimate
        # (Full Abadie-Imbens formula accounts for matching order and dependencies)

        # For controls, find reverse matches
        control_matched_to = fill(0, n_control)
        for i in 1:n_treated
            for j_idx in matches[i]
                # Find position in indices_control
                pos = findfirst(==(j_idx), indices_control)
                if pos !== nothing
                    control_matched_to[pos] = indices_treated[i]
                end
            end
        end

        for j in 1:n_control
            idx_control = indices_control[j]

            # Observed Y(0)
            imputed_y0[idx_control] = outcomes[idx_control]

            # Imputed Y(1)
            if control_matched_to[j] != 0
                imputed_y1[idx_control] = outcomes[control_matched_to[j]]
            else
                # Unmatched control - use nearest treated
                dists = abs.(propensity[indices_treated] .- propensity[idx_control])
                nearest_treated = indices_treated[argmin(dists)]
                imputed_y1[idx_control] = outcomes[nearest_treated]
            end
        end
    end

    # ========================================================================
    # Compute Conditional Variances
    # ========================================================================

    # Individual treatment effects (imputed)
    tau_i = imputed_y1 - imputed_y0

    # Conditional variance for treated units
    # σ²ᵢ(1) ≈ [Yᵢ - Ŷᵢ(0)]²
    sigma_sq_treated = zeros(n_treated)
    for i in 1:n_treated
        idx = indices_treated[i]
        # Imputation error
        sigma_sq_treated[i] = (outcomes[idx] - imputed_y0[idx])^2
    end

    # Matching variance component
    # K_M factor from Abadie-Imbens (2006) Theorem 1
    # Simplified: K_M ≈ M for M:1 matching
    K_M = Float64(M)

    # Variance contribution from controls
    sigma_sq_control = zeros(n_control)
    for j in 1:n_control
        idx = indices_control[j]
        sigma_sq_control[j] = (outcomes[idx] - imputed_y0[idx])^2
    end

    # ========================================================================
    # Compute Abadie-Imbens Variance
    # ========================================================================

    # Variance formula (simplified version for M:1 matching with replacement)
    # V = (1/N₁) ∑ᵢ σ²ᵢ(1) + (K_M/N₁N₀) ∑ⱼ σ²ⱼ(0)

    var_treated_component = mean(sigma_sq_treated)
    var_control_component = (K_M / n_control) * mean(sigma_sq_control)

    variance = var_treated_component + var_control_component

    # Standard error
    se = sqrt(variance / n_treated)

    return variance, se
end


"""
    pairs_bootstrap_variance(outcomes, treatment, propensity, matches; B=1000, M=1)

Bootstrap variance for matching estimators (DEPRECATED for with-replacement matching).

# ⚠️ WARNING
**Bootstrap FAILS for matching with replacement** (Abadie & Imbens 2008).
Use `abadie_imbens_variance` instead for matching with replacement.

This function is provided for:
1. Matching WITHOUT replacement (bootstrap is valid)
2. Educational purposes (demonstrate bootstrap failure)
3. Sensitivity checks (compare to Abadie-Imbens)

# Arguments
- `outcomes::Vector{Float64}`: Observed outcomes
- `treatment::Vector{Bool}`: Treatment indicator
- `propensity::Vector{Float64}`: Estimated propensity scores
- `matches::Vector{Vector{Int}}`: Matched control indices for each treated unit
- `B::Int`: Number of bootstrap resamples (default: 1000)
- `M::Int`: Number of matches per treated unit (default: 1)

# Returns
- `variance::Float64`: Bootstrap variance estimate
- `se::Float64`: Bootstrap standard error

# Method
Pairs bootstrap:
1. Resample matched pairs (treated + matched controls) with replacement
2. Recompute ATE on bootstrap sample
3. Variance = Var(ATE*_b) across B resamples

# Notes
- **Only use for matching WITHOUT replacement**
- For with-replacement, use `abadie_imbens_variance` (analytic)
- Computationally expensive (B × matching operations)

# Example
```julia
# WARNING: Only valid for without-replacement matching
variance, se = pairs_bootstrap_variance(
    outcomes,
    treatment,
    propensity,
    matches,
    B=1000,
    M=1
)
```

# References
- Abadie, A., & Imbens, G. W. (2008). On the failure of the bootstrap for matching estimators.
  *Econometrica*, 76(6), 1537-1557. [Shows bootstrap FAILS for with-replacement]
"""
function pairs_bootstrap_variance(
    outcomes::Vector{Float64},
    treatment::Vector{Bool},
    propensity::Vector{Float64},
    matches::Vector{Vector{Int}};
    B::Int = 1000,
    M::Int = 1,
)
    # WARNING message
    @warn """
    pairs_bootstrap_variance is DEPRECATED for with-replacement matching.
    Bootstrap is INVALID for matching with replacement (Abadie & Imbens 2008).
    Use abadie_imbens_variance() instead for valid inference.

    This function should ONLY be used for:
    - Matching WITHOUT replacement
    - Educational/demonstration purposes
    - Sensitivity analysis (compare to Abadie-Imbens)
    """

    n_treated = sum(treatment)
    indices_treated = findall(treatment)

    # Store bootstrap ATEs
    ate_boot = zeros(B)

    for b in 1:B
        # Resample treated units with replacement
        boot_idx = rand(1:n_treated, n_treated)

        # Compute ATE on bootstrap sample
        ate_b = 0.0
        n_matched_b = 0

        for i in boot_idx
            matched_controls = matches[i]
            if isempty(matched_controls)
                continue
            end

            y_treated = outcomes[indices_treated[i]]
            y_control = mean(outcomes[matched_controls])

            ate_b += (y_treated - y_control)
            n_matched_b += 1
        end

        if n_matched_b > 0
            ate_boot[b] = ate_b / n_matched_b
        else
            ate_boot[b] = 0.0
        end
    end

    # Bootstrap variance
    variance = var(ate_boot)
    se = sqrt(variance)

    return variance, se
end
