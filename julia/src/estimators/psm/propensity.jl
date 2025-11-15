"""
Propensity score estimation via logistic regression.
"""

"""
    estimate_propensity(treatment::AbstractVector{Bool}, covariates::Matrix{Float64})

Estimate propensity scores P(T=1|X) using logistic regression.

# Arguments
- `treatment::Vector{Bool}`: Binary treatment indicator
- `covariates::Matrix{Float64}`: Covariate matrix X (n × p)

# Returns
- `propensity::Vector{Float64}`: Estimated propensity scores in (0, 1)

# Method
Uses maximum likelihood logistic regression:
```math
\\log\\left(\\frac{P(T=1|X)}{1-P(T=1|X)}\\right) = \\beta_0 + \\beta' X
```

Propensity scores are predicted probabilities:
```math
e(X) = P(T=1|X) = \\frac{1}{1 + \\exp(-\\beta_0 - \\beta' X)}
```

# Notes
- Propensity scores guaranteed in (0, 1) by logistic function
- Perfect separation (propensity = 0 or 1) indicates strong confounding
- Check for common support: propensity distributions should overlap

# Example
```julia
treatment = [true, true, false, false]
covariates = [5.0 2.0; 6.0 3.0; 4.5 2.0; 5.2 2.3]
propensity = estimate_propensity(treatment, covariates)
```

# Warnings
- May fail to converge if perfect separation exists
- High-dimensional covariates (p ≈ n) may cause overfitting
- Assumes linear relationship in log-odds (may misspecify)
"""
function estimate_propensity(
    treatment::AbstractVector{Bool},
    covariates::Matrix{Float64},
)::Vector{Float64}
    n, p = size(covariates)

    # ========================================================================
    # Validate Inputs
    # ========================================================================

    if length(treatment) != n
        throw(
            ArgumentError(
                "CRITICAL ERROR: Mismatched lengths.\n" *
                "Function: estimate_propensity\n" *
                "treatment has length $(length(treatment)), covariates has $n rows.\n" *
                "Must have same length.",
            ),
        )
    end

    # Check for perfect separation (all treated have X > all control)
    # This is a heuristic check, not exhaustive
    if p == 1
        x = covariates[:, 1]
        max_control = maximum(x[.!treatment])
        min_treated = minimum(x[treatment])
        if min_treated > max_control
            @warn "Perfect separation detected: min(X|T=1) > max(X|T=0). Propensity scores may be 0 or 1."
        end
    end

    # ========================================================================
    # Fit Logistic Regression
    # ========================================================================

    # Create DataFrame for GLM.jl
    # Note: GLM.jl expects column names, we'll use X1, X2, ..., Xp
    col_names = [Symbol("X$i") for i in 1:p]
    df = DataFrame(covariates, col_names)
    df.treatment = Float64.(treatment)  # Convert Bool to Float64 for GLM

    # Fit logistic regression
    formula = Term(:treatment) ~ sum(Term.(col_names))

    model = try
        glm(formula, df, Binomial(), LogitLink())
    catch e
        if isa(e, ConvergenceException) || contains(string(e), "singular")
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Logistic regression failed to converge.\n" *
                    "Function: estimate_propensity\n" *
                    "Possible causes:\n" *
                    "1. Perfect separation (treatment perfectly predicted by covariates)\n" *
                    "2. Multicollinearity (highly correlated covariates)\n" *
                    "3. Too many covariates relative to sample size (p ≈ n)\n" *
                    "Solutions: Remove collinear covariates, reduce dimensionality, or increase sample size.",
                ),
            )
        else
            rethrow(e)
        end
    end

    # ========================================================================
    # Predict Propensity Scores
    # ========================================================================

    # Predict probabilities P(T=1|X) for all units
    propensity = predict(model, df)

    # Verify propensity scores in (0, 1)
    # GLM.jl logistic link guarantees this, but check anyway
    if any(propensity .<= 0) || any(propensity .>= 1)
        @warn "Propensity scores at boundary detected. " *
              "min=$(minimum(propensity)), max=$(maximum(propensity)). " *
              "This indicates extreme confounding or perfect separation."
    end

    # Ensure strictly in (0, 1) by clamping to avoid division by zero in IPW
    # Use small epsilon = 1e-10 for numerical stability
    eps = 1e-10
    propensity = clamp.(propensity, eps, 1 - eps)

    return propensity
end

"""
    check_common_support(propensity::Vector{Float64}, treatment::AbstractVector{Bool}; threshold::Float64=0.1)

Check for common support (overlap) in propensity score distributions.

Common support means treated and control units have overlapping propensity distributions.
Without overlap, matching is impossible or requires strong extrapolation.

# Arguments
- `propensity::Vector{Float64}`: Estimated propensity scores
- `treatment::Vector{Bool}`: Treatment indicator
- `threshold::Float64`: Minimum overlap proportion (default: 0.1 = 10%)

# Returns
- `has_support::Bool`: true if common support exists
- `support_region::(Float64, Float64)`: Region of overlap (min_overlap, max_overlap)
- `n_outside::Int`: Number of units outside common support

# Method
Common support defined as region where both groups have observations:
```
[max(min(e|T=1), min(e|T=0)), min(max(e|T=1), max(e|T=0))]
```

# Example
```julia
has_support, region, n_outside = check_common_support(propensity, treatment)
if !has_support
    @warn "No common support: \$n_outside units outside overlap region"
end
```

# References
- Crump, R. K., et al. (2009). Dealing with limited overlap in estimation of average treatment effects. *Biometrika*, 96(1), 187-199.
"""
function check_common_support(
    propensity::Vector{Float64},
    treatment::AbstractVector{Bool};
    threshold::Float64 = 0.1,
)
    # Propensity ranges by treatment group
    prop_treated = propensity[treatment]
    prop_control = propensity[.!treatment]

    min_treated = minimum(prop_treated)
    max_treated = maximum(prop_treated)
    min_control = minimum(prop_control)
    max_control = maximum(prop_control)

    # Common support region
    support_min = max(min_treated, min_control)
    support_max = min(max_treated, max_control)

    # Check if overlap exists
    has_support = support_max > support_min

    # Count units outside common support
    n_outside = count(
        (propensity .< support_min) .| (propensity .> support_max),
    )

    # Check if sufficient overlap (at least threshold proportion in overlap)
    n = length(propensity)
    overlap_proportion = 1.0 - (n_outside / n)

    if overlap_proportion < threshold
        has_support = false
    end

    return has_support, (support_min, support_max), n_outside
end
