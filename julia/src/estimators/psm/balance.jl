"""
Balance diagnostics for propensity score matching.

CRITICAL (MEDIUM-5 concern): Must verify balance on ALL covariates, not subset.
Standard: SMD < 0.1 after matching.
"""

"""
    compute_standardized_mean_difference(x_treated, x_control; pooled=true)

Compute standardized mean difference (SMD) for covariate balance.

# Arguments
- `x_treated::Vector{Float64}`: Covariate values for treated group
- `x_control::Vector{Float64}`: Covariate values for control group
- `pooled::Bool`: Use pooled standard deviation (default: true, recommended)

# Returns
- `smd::Float64`: Standardized mean difference

# Method

**Pooled SMD** (default, recommended):
```
SMD = (mean(X_T) - mean(X_C)) / sqrt((var(X_T) + var(X_C)) / 2)
```

**Unpooled SMD** (alternative):
```
SMD = (mean(X_T) - mean(X_C)) / sqrt(var(X_T))
```

# Interpretation
- |SMD| < 0.1: Good balance (standard threshold)
- 0.1 ≤ |SMD| < 0.2: Acceptable balance
- |SMD| ≥ 0.2: Poor balance (problematic)

# Notes
- Independent of sample size (unlike t-test)
- Scale-free measure (unlike raw mean difference)
- Pooled version recommended for matching (Austin 2009)

# Example
```julia
smd = compute_standardized_mean_difference(x_treated, x_control)
if abs(smd) < 0.1
    println("Good balance achieved")
else
    @warn "Poor balance: SMD = \$smd"
end
```

# References
- Austin, P. C. (2009). Balance diagnostics for comparing the distribution of baseline
  covariates between treatment groups in propensity-score matched samples.
  *Statistics in Medicine*, 28(25), 3083-3107.
- Stuart, E. A. (2010). Matching methods for causal inference: A review and a look forward.
  *Statistical Science*, 25(1), 1-21.
"""
function compute_standardized_mean_difference(
    x_treated::Vector{Float64},
    x_control::Vector{Float64};
    pooled::Bool = true,
)
    mean_t = mean(x_treated)
    mean_c = mean(x_control)
    var_t = var(x_treated)
    var_c = var(x_control)

    # Pooled standard deviation (recommended)
    if pooled
        pooled_std = sqrt((var_t + var_c) / 2)

        # Avoid division by zero
        if pooled_std < 1e-10
            # Both have zero variance
            if abs(mean_t - mean_c) < 1e-10
                return 0.0  # Identical distributions
            else
                # Different means but no variance -> infinite SMD (perfect separation)
                # Return large value as proxy for Inf
                return sign(mean_t - mean_c) * 1e6
            end
        end

        smd = (mean_t - mean_c) / pooled_std
    else
        # Unpooled (standardize by treated variance only)
        if var_t < 1e-10
            # Treated has zero variance
            if abs(mean_t - mean_c) < 1e-10
                return 0.0  # Identical values
            else
                # Different means but no treated variance -> infinite SMD
                return sign(mean_t - mean_c) * 1e6
            end
        end

        smd = (mean_t - mean_c) / sqrt(var_t)
    end

    return smd
end


"""
    compute_variance_ratio(x_treated, x_control)

Compute variance ratio for covariate balance.

# Arguments
- `x_treated::Vector{Float64}`: Covariate values for treated group
- `x_control::Vector{Float64}`: Covariate values for control group

# Returns
- `vr::Float64`: Variance ratio (var_treated / var_control)

# Method
```
VR = var(X_T) / var(X_C)
```

# Interpretation
- VR ≈ 1.0: Good balance
- 0.5 < VR < 2.0: Acceptable balance (some recommend 0.8-1.25)
- VR ≤ 0.5 or VR ≥ 2.0: Poor balance

# Notes
- Complements SMD (checks second moment)
- Sensitive to outliers
- Log scale often used for visualization

# Example
```julia
vr = compute_variance_ratio(x_treated, x_control)
if 0.5 < vr < 2.0
    println("Good variance balance")
else
    @warn "Variance imbalance: VR = \$vr"
end
```

# References
- Rubin, D. B. (2001). Using propensity scores to help design observational studies:
  Application to the tobacco litigation. *Health Services and Outcomes Research
  Methodology*, 2, 169-188.
"""
function compute_variance_ratio(
    x_treated::Vector{Float64},
    x_control::Vector{Float64},
)
    var_t = var(x_treated)
    var_c = var(x_control)

    # Avoid division by zero
    if var_c < 1e-10
        if var_t < 1e-10
            return 1.0  # Both variances near zero
        else
            return Inf  # Only control variance near zero
        end
    end

    return var_t / var_c
end


"""
    check_covariate_balance(covariates, treatment, matched_indices; threshold=0.1)

Check covariate balance after matching for ALL covariates.

CRITICAL (MEDIUM-5): Must verify balance on ALL covariates, not just a subset.

# Arguments
- `covariates::Matrix{Float64}`: Covariate matrix (n × p)
- `treatment::Vector{Bool}`: Treatment indicator
- `matched_indices::Vector{Tuple{Int,Int}}`: Matched pairs (treated_idx, control_idx)
- `threshold::Float64`: SMD threshold for good balance (default: 0.1)

# Returns
- `balanced::Bool`: true if ALL covariates have |SMD| < threshold
- `smd_values::Vector{Float64}`: SMD for each covariate (after matching)
- `vr_values::Vector{Float64}`: Variance ratios for each covariate (after matching)
- `smd_before::Vector{Float64}`: SMD before matching (for comparison)
- `vr_before::Vector{Float64}`: Variance ratios before matching

# Method

1. **Before matching**: Compute SMD and VR for all covariates using full sample
2. **After matching**: Compute SMD and VR using only matched pairs
3. **Balance check**: ALL covariates must have |SMD| < threshold

# Interpretation

**Good balance**:
- ALL |SMD| < 0.1
- Most VR in (0.5, 2.0)

**Poor balance**:
- ANY |SMD| ≥ 0.1
- Many VR outside (0.5, 2.0)

# Example
```julia
balanced, smd, vr, smd_before, vr_before = check_covariate_balance(
    covariates,
    treatment,
    matched_indices,
    threshold=0.1
)

if balanced
    println("✓ All covariates balanced (SMD < 0.1)")
else
    println("✗ Some covariates imbalanced")
    for (j, s) in enumerate(smd)
        if abs(s) ≥ 0.1
            println("  Covariate \$j: SMD = \$s")
        end
    end
end
```

# References
- Austin, P. C. (2009). Balance diagnostics for comparing the distribution of baseline
  covariates between treatment groups in propensity-score matched samples.
  *Statistics in Medicine*, 28(25), 3083-3107.
- Stuart, E. A., & Rubin, D. B. (2008). Matching with multiple control groups with
  adjustment for group differences. *Journal of Educational and Behavioral Statistics*,
  33(3), 279-306.
"""
function check_covariate_balance(
    covariates::Matrix{Float64},
    treatment::AbstractVector{Bool},
    matched_indices::Vector{Tuple{Int,Int}};
    threshold::Float64 = 0.1,
)
    n, p = size(covariates)

    # ========================================================================
    # Validate Inputs
    # ========================================================================

    if length(treatment) != n
        throw(
            ArgumentError(
                "CRITICAL ERROR: Mismatched lengths.\\n" *
                "Function: check_covariate_balance\\n" *
                "covariates has $n rows, treatment has $(length(treatment)) elements.\\n" *
                "Must have same length.",
            ),
        )
    end

    if threshold <= 0
        throw(
            ArgumentError(
                "CRITICAL ERROR: Invalid threshold.\\n" *
                "Function: check_covariate_balance\\n" *
                "threshold must be > 0, got threshold = $threshold\\n" *
                "Standard value: 0.1 (SMD < 0.1 = good balance)",
            ),
        )
    end

    # ========================================================================
    # Balance BEFORE Matching
    # ========================================================================

    indices_treated = findall(treatment)
    indices_control = findall(.!treatment)

    smd_before = zeros(p)
    vr_before = zeros(p)

    for j in 1:p
        x_t = covariates[indices_treated, j]
        x_c = covariates[indices_control, j]

        smd_before[j] = compute_standardized_mean_difference(x_t, x_c)
        vr_before[j] = compute_variance_ratio(x_t, x_c)
    end

    # ========================================================================
    # Balance AFTER Matching
    # ========================================================================

    if isempty(matched_indices)
        # No matches - return NaN for after-matching metrics
        return false, fill(NaN, p), fill(NaN, p), smd_before, vr_before
    end

    # Extract matched treated and control indices
    matched_treated = [pair[1] for pair in matched_indices]
    matched_control = [pair[2] for pair in matched_indices]

    smd_values = zeros(p)
    vr_values = zeros(p)

    for j in 1:p
        x_t_matched = covariates[matched_treated, j]
        x_c_matched = covariates[matched_control, j]

        smd_values[j] = compute_standardized_mean_difference(x_t_matched, x_c_matched)
        vr_values[j] = compute_variance_ratio(x_t_matched, x_c_matched)
    end

    # ========================================================================
    # Check Balance: ALL Covariates
    # ========================================================================

    # CRITICAL (MEDIUM-5): Must check ALL covariates, not subset
    balanced = all(abs.(smd_values) .< threshold)

    return balanced, smd_values, vr_values, smd_before, vr_before
end


"""
    balance_summary(smd_after, vr_after, smd_before, vr_before; threshold=0.1)

Create summary of covariate balance diagnostics.

# Arguments
- `smd_after::Vector{Float64}`: SMD values after matching
- `vr_after::Vector{Float64}`: Variance ratios after matching
- `smd_before::Vector{Float64}`: SMD values before matching
- `vr_before::Vector{Float64}`: Variance ratios before matching
- `threshold::Float64`: SMD threshold for good balance (default: 0.1)

# Returns
- `summary::NamedTuple`: Balance summary with fields:
  - `n_covariates::Int`: Total number of covariates
  - `n_balanced::Int`: Number with |SMD| < threshold (after matching)
  - `n_imbalanced::Int`: Number with |SMD| ≥ threshold (after matching)
  - `max_smd_before::Float64`: Maximum |SMD| before matching
  - `max_smd_after::Float64`: Maximum |SMD| after matching
  - `mean_smd_before::Float64`: Mean |SMD| before matching
  - `mean_smd_after::Float64`: Mean |SMD| after matching
  - `improvement::Float64`: Reduction in mean |SMD|
  - `all_balanced::Bool`: true if ALL covariates balanced

# Example
```julia
summary = balance_summary(smd_after, vr_after, smd_before, vr_before)
println("Balance: \$(summary.n_balanced)/\$(summary.n_covariates) covariates")
println("Improvement: \$(round(summary.improvement*100, digits=1))%")
```
"""
function balance_summary(
    smd_after::Vector{Float64},
    vr_after::Vector{Float64},
    smd_before::Vector{Float64},
    vr_before::Vector{Float64};
    threshold::Float64 = 0.1,
)
    p = length(smd_after)

    n_balanced = count(abs.(smd_after) .< threshold)
    n_imbalanced = p - n_balanced

    max_smd_before = maximum(abs.(smd_before))
    max_smd_after = maximum(abs.(smd_after))

    mean_smd_before = mean(abs.(smd_before))
    mean_smd_after = mean(abs.(smd_after))

    # Improvement: reduction in mean |SMD|
    improvement = (mean_smd_before - mean_smd_after) / mean_smd_before

    all_balanced = n_imbalanced == 0

    return (;
        n_covariates = p,
        n_balanced,
        n_imbalanced,
        max_smd_before,
        max_smd_after,
        mean_smd_before,
        mean_smd_after,
        improvement,
        all_balanced,
    )
end
