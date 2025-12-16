#=
Propensity Score Estimation for Observational Studies

Provides logistic regression-based propensity score estimation
with diagnostics (AUC, calibration, overlap checks).

Functions:
- estimate_propensity_scores: Logistic regression propensity estimation
- compute_propensity_auc: Area under ROC curve
- trim_propensities: Remove extreme propensity observations
- stabilize_ipw_weights: Compute stabilized IPW weights

References:
- Rosenbaum & Rubin (1983). The central role of the propensity score.
- Austin & Stuart (2015). Moving towards best practice with IPTW.
=#

"""
    estimate_propensity_scores(treatment, covariates; clip_epsilon=1e-6)

Estimate propensity scores P(T=1|X) using logistic regression.

# Arguments
- `treatment::Vector{Bool}`: Binary treatment indicators
- `covariates::Matrix{T}`: Covariate matrix (n × p)
- `clip_epsilon::Float64`: Clip propensities to [ε, 1-ε] (default: 1e-6)

# Returns
NamedTuple with:
- `propensity::Vector{T}`: Estimated propensity scores
- `coefficients::Vector{T}`: Logistic regression coefficients (intercept + slopes)
- `converged::Bool`: Whether GLM converged
- `log_likelihood::T`: Log-likelihood of fitted model

# Mathematical Model

Logistic regression: logit(P(T=1|X)) = β₀ + β'X

The propensity score is: e(X) = 1 / (1 + exp(-(β₀ + β'X)))

# Example
```julia
X = randn(100, 2)
T = rand(100) .< (1 ./ (1 .+ exp.(-(0.5 .* X[:, 1]))))
result = estimate_propensity_scores(T, X)
propensity = result.propensity
```

# Notes
- Uses GLM.jl for logistic regression
- Clips extreme propensities to avoid division by zero in IPW
- Warns if propensity estimation shows perfect separation
"""
function estimate_propensity_scores(
    treatment::AbstractVector{Bool},
    covariates::AbstractMatrix{T};
    clip_epsilon::Float64 = 1e-6
) where {T<:Real}
    n, p = size(covariates)

    # Build DataFrame for GLM
    df = DataFrame(covariates, :auto)
    df.treatment = Int.(treatment)

    # Create formula: treatment ~ X1 + X2 + ... + Xp
    feature_names = names(df)[1:end-1]
    formula_str = "treatment ~ " * join(feature_names, " + ")
    formula = eval(Meta.parse("@formula($formula_str)"))

    # Fit logistic regression
    try
        model = glm(formula, df, Binomial(), LogitLink())

        # Extract propensity scores
        propensity = predict(model)

        # Check for convergence (GLM.jl uses converged(model))
        converged = try
            GLM.converged(model)
        catch
            true  # Assume converged if method not available
        end

        # Extract coefficients
        coefficients = coef(model)

        # Log-likelihood
        log_likelihood = loglikelihood(model)

        # Clip extreme propensities
        propensity_clipped = clamp.(propensity, clip_epsilon, 1 - clip_epsilon)

        # Warn if many propensities were clipped
        n_clipped = sum((propensity .< clip_epsilon) .| (propensity .> 1 - clip_epsilon))
        if n_clipped > 0
            @warn "Clipped $n_clipped propensity scores to [$clip_epsilon, $(1-clip_epsilon)]. " *
                  "This may indicate perfect separation or extreme confounding."
        end

        return (
            propensity = convert(Vector{T}, propensity_clipped),
            coefficients = convert(Vector{T}, coefficients),
            converged = converged,
            log_likelihood = T(log_likelihood)
        )
    catch e
        # Handle estimation failure
        msg = hasfield(typeof(e), :msg) ? e.msg : string(e)
        throw(ArgumentError(
            "Propensity score estimation failed: $msg. " *
            "Check for perfect separation, collinearity, or insufficient variation."
        ))
    end
end


"""
    compute_propensity_auc(propensity, treatment)

Compute Area Under ROC Curve (AUC) for propensity scores.

AUC measures discriminatory power of propensity model:
- AUC ≈ 0.5: Treatment nearly independent of X (weak confounding)
- AUC > 0.7: Strong relationship between X and treatment (strong confounding)
- AUC ≈ 1.0: Perfect separation (positivity violation)

# Arguments
- `propensity::Vector{T}`: Estimated propensity scores
- `treatment::Vector{Bool}`: True treatment indicators

# Returns
- `auc::T`: Area under the ROC curve

# Algorithm
Uses the Mann-Whitney U statistic formulation:
AUC = P(propensity[T=1] > propensity[T=0])

# Example
```julia
auc = compute_propensity_auc(propensity, treatment)
if auc > 0.8
    @warn "High propensity AUC indicates strong confounding"
end
```
"""
function compute_propensity_auc(
    propensity::AbstractVector{T},
    treatment::AbstractVector{Bool}
) where {T<:Real}
    # Get propensity scores by treatment group
    prop_treated = propensity[treatment]
    prop_control = propensity[.!treatment]

    n1 = length(prop_treated)
    n0 = length(prop_control)

    if n1 == 0 || n0 == 0
        return T(0.5)  # Undefined, return neutral
    end

    # Mann-Whitney U statistic
    # AUC = P(prop_treated > prop_control) + 0.5 * P(prop_treated == prop_control)
    u_count = 0.0
    for p1 in prop_treated
        for p0 in prop_control
            if p1 > p0
                u_count += 1.0
            elseif p1 == p0
                u_count += 0.5
            end
        end
    end

    auc = u_count / (n1 * n0)
    return T(auc)
end


"""
    trim_propensities(propensity, treatment, outcomes, covariates; trim_at=(0.01, 0.99))

Trim observations with extreme propensity scores.

Trimming removes observations where propensity is too close to 0 or 1,
which can cause extreme IPW weights. This is a bias-variance tradeoff:
trimming reduces variance but may introduce selection bias.

# Arguments
- `propensity::Vector{T}`: Propensity scores
- `treatment::Vector{Bool}`: Treatment indicators
- `outcomes::Vector{T}`: Outcome values
- `covariates::Matrix{T}`: Covariate matrix
- `trim_at::Tuple{Float64,Float64}`: Percentile bounds (default: 1st and 99th)

# Returns
NamedTuple with trimmed versions of all arrays and `n_trimmed` count.

# Example
```julia
result = trim_propensities(propensity, treatment, outcomes, covariates; trim_at=(0.05, 0.95))
n_kept = length(result.propensity)
```
"""
function trim_propensities(
    propensity::AbstractVector{T},
    treatment::AbstractVector{Bool},
    outcomes::AbstractVector{T},
    covariates::AbstractMatrix{T};
    trim_at::Tuple{Float64,Float64} = (0.01, 0.99)
) where {T<:Real}
    lower, upper = trim_at

    if lower >= upper
        throw(ArgumentError("trim_at lower ($lower) must be < upper ($upper)"))
    end

    if lower < 0 || upper > 1
        throw(ArgumentError("trim_at values must be in [0, 1]"))
    end

    # Compute percentile bounds
    p_lower = quantile(propensity, lower)
    p_upper = quantile(propensity, upper)

    # Create mask for observations to keep
    keep_mask = (propensity .>= p_lower) .& (propensity .<= p_upper)

    n_trimmed = sum(.!keep_mask)

    # Return trimmed arrays
    return (
        propensity = propensity[keep_mask],
        treatment = treatment[keep_mask],
        outcomes = outcomes[keep_mask],
        covariates = covariates[keep_mask, :],
        n_trimmed = n_trimmed,
        n_kept = sum(keep_mask),
        trim_bounds = (p_lower, p_upper)
    )
end


"""
    compute_ipw_weights(propensity, treatment; stabilize=false)

Compute IPW weights from propensity scores.

# Standard IPW Weights
- Treated: w = 1 / e(X)
- Control: w = 1 / (1 - e(X))

# Stabilized IPW Weights (Hernán & Robins, 2020)
- Treated: w = P(T=1) / e(X)
- Control: w = P(T=0) / (1 - e(X))

Stabilized weights have mean ≈ 1 and typically lower variance.

# Arguments
- `propensity::Vector{T}`: Propensity scores
- `treatment::Vector{Bool}`: Treatment indicators
- `stabilize::Bool`: Use stabilized weights (default: false)

# Returns
- `weights::Vector{T}`: IPW weights

# Example
```julia
weights = compute_ipw_weights(propensity, treatment; stabilize=true)
mean(weights)  # ≈ 1.0 for stabilized weights
```
"""
function compute_ipw_weights(
    propensity::AbstractVector{T},
    treatment::AbstractVector{Bool};
    stabilize::Bool = false
) where {T<:Real}
    n = length(propensity)
    weights = zeros(T, n)

    if stabilize
        # Marginal treatment probability
        p_treat = mean(treatment)

        for i in 1:n
            if treatment[i]
                weights[i] = p_treat / propensity[i]
            else
                weights[i] = (1 - p_treat) / (1 - propensity[i])
            end
        end
    else
        # Standard IPW weights
        for i in 1:n
            if treatment[i]
                weights[i] = 1.0 / propensity[i]
            else
                weights[i] = 1.0 / (1 - propensity[i])
            end
        end
    end

    return weights
end


"""
    check_positivity(propensity, treatment; threshold=0.01)

Check positivity assumption (common support).

Positivity requires: 0 < P(T=1|X) < 1 for all X in support.

In practice, this means propensity scores shouldn't be too extreme
in either treatment group.

# Arguments
- `propensity::Vector{T}`: Propensity scores
- `treatment::Vector{Bool}`: Treatment indicators
- `threshold::Float64`: Warning threshold for extreme propensities

# Returns
NamedTuple with:
- `positivity_ok::Bool`: Whether positivity appears satisfied
- `n_extreme_treated::Int`: Treated units with e(X) > 1-threshold
- `n_extreme_control::Int`: Control units with e(X) < threshold
- `overlap_measure::T`: Measure of common support (0-1)

# Example
```julia
check = check_positivity(propensity, treatment)
if !check.positivity_ok
    @warn "Positivity assumption may be violated"
end
```
"""
function check_positivity(
    propensity::AbstractVector{T},
    treatment::AbstractVector{Bool};
    threshold::Float64 = 0.01
) where {T<:Real}
    prop_treated = propensity[treatment]
    prop_control = propensity[.!treatment]

    # Count extreme propensities
    n_extreme_treated = sum(prop_treated .> 1 - threshold)
    n_extreme_control = sum(prop_control .< threshold)

    # Overlap measure: intersection of propensity distributions
    # Use overlap coefficient (min of densities)
    min_treated = minimum(prop_treated)
    max_treated = maximum(prop_treated)
    min_control = minimum(prop_control)
    max_control = maximum(prop_control)

    # Simple overlap: proportion of common support
    overlap_low = max(min_treated, min_control)
    overlap_high = min(max_treated, max_control)
    total_range = max(max_treated, max_control) - min(min_treated, min_control)

    overlap_measure = total_range > 0 ? max(0, overlap_high - overlap_low) / total_range : T(0)

    positivity_ok = (n_extreme_treated == 0) && (n_extreme_control == 0) && (overlap_measure > 0.5)

    return (
        positivity_ok = positivity_ok,
        n_extreme_treated = n_extreme_treated,
        n_extreme_control = n_extreme_control,
        overlap_measure = T(overlap_measure),
        prop_range_treated = (T(min_treated), T(max_treated)),
        prop_range_control = (T(min_control), T(max_control))
    )
end
