#=
E-Value Sensitivity Analysis

Computes E-values for assessing sensitivity of causal estimates to unmeasured
confounding. The E-value represents the minimum strength of association that
an unmeasured confounder would need with both treatment and outcome to fully
explain away the observed effect.

References:
- VanderWeele, T. J., & Ding, P. (2017). "Sensitivity Analysis in Observational
  Research: Introducing the E-Value." Annals of Internal Medicine.
- VanderWeele, T. J. (2017). "On the Distinction Between Interaction and Effect
  Modification." Epidemiology.
=#

# ============================================================================
# Core E-Value Computation
# ============================================================================

"""
    compute_e_value(rr::Real) -> Float64

Compute the E-value for a given risk ratio.

# Formula
For RR ≥ 1:
    E = RR + √(RR × (RR - 1))

For RR < 1 (protective effect):
    E = 1/RR + √(1/RR × (1/RR - 1))

# Arguments
- `rr`: Risk ratio (must be positive)

# Returns
- E-value (minimum confounding strength to explain away effect)

# Example
```julia
compute_e_value(2.0)  # Returns 3.41 (E = 2 + √2)
compute_e_value(0.5)  # Returns 3.41 (inverts to RR=2)
```
"""
function compute_e_value(rr::Real)::Float64
    rr > 0 || throw(ArgumentError("Risk ratio must be positive, got: $rr"))

    # Handle protective effects (RR < 1) by inverting
    rr_use = rr < 1.0 ? 1.0 / rr : rr

    # E-value formula: E = RR + sqrt(RR * (RR - 1))
    if rr_use ≈ 1.0
        return 1.0  # No effect, no confounding needed
    end

    return rr_use + sqrt(rr_use * (rr_use - 1.0))
end

# ============================================================================
# Effect Type Conversions
# ============================================================================

"""
    smd_to_rr(d::Real) -> Float64

Convert standardized mean difference (Cohen's d) to approximate risk ratio.

Uses VanderWeele (2017) approximation: RR ≈ exp(0.91 × d)

# Arguments
- `d`: Standardized mean difference

# Returns
- Approximate risk ratio

# Note
This approximation assumes a probit link and works well for |d| < 2.
"""
function smd_to_rr(d::Real)::Float64
    # VanderWeele approximation: RR ≈ exp(0.91 * d)
    return exp(0.91 * d)
end

"""
    ate_to_rr(ate::Real, baseline_risk::Real) -> Float64

Convert average treatment effect (risk difference) to risk ratio.

# Arguments
- `ate`: Average treatment effect (risk difference)
- `baseline_risk`: Baseline risk in control group, must be in (0, 1)

# Returns
- Risk ratio: (baseline_risk + ATE) / baseline_risk

# Example
```julia
ate_to_rr(0.1, 0.2)  # Returns 1.5 (30% / 20%)
```
"""
function ate_to_rr(ate::Real, baseline_risk::Real)::Float64
    0 < baseline_risk < 1 || throw(ArgumentError(
        "baseline_risk must be in (0, 1), got: $baseline_risk"
    ))

    new_risk = baseline_risk + ate
    0 ≤ new_risk ≤ 1 || throw(ArgumentError(
        "ATE creates invalid risk: $baseline_risk + $ate = $new_risk"
    ))

    # Avoid division by zero edge case
    if baseline_risk ≈ 0
        return Inf
    end

    return new_risk / baseline_risk
end

"""
    convert_to_rr(estimate::Real, effect_type::EffectType; baseline_risk=nothing) -> Float64

Convert an effect estimate to risk ratio scale.

# Arguments
- `estimate`: The effect estimate
- `effect_type`: Type of effect (RR, OR, HR, SMD, ATE)
- `baseline_risk`: Required for ATE conversion

# Returns
- Effect on risk ratio scale
"""
function convert_to_rr(
    estimate::Real,
    effect_type::EffectType;
    baseline_risk::Union{Nothing,Real}=nothing,
)::Float64
    if effect_type == RR
        estimate > 0 || throw(ArgumentError("Risk ratio must be positive"))
        return Float64(estimate)

    elseif effect_type == OR
        estimate > 0 || throw(ArgumentError("Odds ratio must be positive"))
        # OR approximates RR for rare outcomes; use directly
        return Float64(estimate)

    elseif effect_type == HR
        estimate > 0 || throw(ArgumentError("Hazard ratio must be positive"))
        # HR approximates RR; use directly
        return Float64(estimate)

    elseif effect_type == SMD
        return smd_to_rr(estimate)

    elseif effect_type == ATE
        isnothing(baseline_risk) && throw(ArgumentError(
            "baseline_risk is required for ATE conversion"
        ))
        return ate_to_rr(estimate, baseline_risk)

    else
        throw(ArgumentError("Unknown effect type: $effect_type"))
    end
end

# ============================================================================
# Interpretation Generation
# ============================================================================

"""
    generate_e_value_interpretation(e_value, e_value_ci, rr, effect_type) -> String

Generate human-readable interpretation of E-value results.

# Robustness thresholds (based on literature):
- E ≈ 1.0: Not robust (effect at null)
- E ∈ [1.0, 1.5): Weakly robust
- E ∈ [1.5, 2.0): Moderately robust
- E ∈ [2.0, 3.0): Strongly robust
- E ≥ 3.0: Very strongly robust
"""
function generate_e_value_interpretation(
    e_value::Real,
    e_value_ci::Real,
    rr::Real,
    effect_type::EffectType,
)::String
    # Direction of effect
    direction = rr >= 1.0 ? "harmful" : "protective"

    # Robustness category for point estimate
    robustness = if e_value < 1.0 + 1e-10
        "not robust to unmeasured confounding"
    elseif e_value < 1.5
        "weakly robust to unmeasured confounding"
    elseif e_value < 2.0
        "moderately robust to unmeasured confounding"
    elseif e_value < 3.0
        "strongly robust to unmeasured confounding"
    else
        "very strongly robust to unmeasured confounding"
    end

    # Build interpretation
    effect_str = string(effect_type)
    ci_status = if e_value_ci < 1.0 + 1e-10
        "The confidence interval includes the null, so no confounding strength is needed to shift the CI to include null."
    else
        "For the confidence interval to include the null, unmeasured confounding would need strength ≥ $(round(e_value_ci, digits=2)) on the RR scale."
    end

    return "The observed $direction effect ($effect_str = $(round(rr, digits=3))) is $robustness. " *
           "An unmeasured confounder would need an association with both treatment and outcome of " *
           "at least $(round(e_value, digits=2)) on the RR scale to fully explain away the point estimate. " *
           ci_status
end

# ============================================================================
# Main Solve Implementation
# ============================================================================

"""
    solve(problem::EValueProblem, estimator::EValue) -> EValueSolution

Compute E-value sensitivity analysis for a causal effect estimate.

# Example
```julia
# Risk ratio with CI
problem = EValueProblem(2.5; ci_lower=1.8, ci_upper=3.5, effect_type=:rr)
solution = solve(problem, EValue())

println("E-value: \$(solution.e_value)")
println("E-value for CI: \$(solution.e_value_ci)")
println(solution.interpretation)
```

# Returns
- `EValueSolution` containing:
  - `e_value`: E-value for point estimate
  - `e_value_ci`: E-value for CI bound closest to null
  - `rr_equivalent`: Effect on RR scale
  - `interpretation`: Human-readable summary
"""
function solve(problem::EValueProblem{T,P}, ::EValue) where {T<:Real,P}
    # Convert estimate to RR scale
    rr = convert_to_rr(
        problem.estimate,
        problem.effect_type;
        baseline_risk=problem.baseline_risk,
    )

    # Compute E-value for point estimate
    e_val = compute_e_value(rr)

    # Compute E-value for confidence interval
    e_val_ci = if isnothing(problem.ci_lower) && isnothing(problem.ci_upper)
        # No CI provided, use point estimate E-value
        e_val
    else
        # Convert CI bounds to RR scale
        ci_lo_rr = if !isnothing(problem.ci_lower)
            convert_to_rr(
                problem.ci_lower,
                problem.effect_type;
                baseline_risk=problem.baseline_risk,
            )
        else
            nothing
        end

        ci_hi_rr = if !isnothing(problem.ci_upper)
            convert_to_rr(
                problem.ci_upper,
                problem.effect_type;
                baseline_risk=problem.baseline_risk,
            )
        else
            nothing
        end

        # Check if CI includes null (RR = 1)
        ci_includes_null = false
        if !isnothing(ci_lo_rr) && !isnothing(ci_hi_rr)
            ci_includes_null = ci_lo_rr ≤ 1.0 ≤ ci_hi_rr
        end

        if ci_includes_null
            # CI includes null, no confounding needed to shift to null
            1.0
        else
            # Find bound closest to null (RR = 1)
            bound_closest_to_null = if !isnothing(ci_lo_rr) && !isnothing(ci_hi_rr)
                # Both bounds available
                if rr >= 1.0
                    ci_lo_rr  # For harmful effects, lower bound is closest to 1
                else
                    ci_hi_rr  # For protective effects, upper bound is closest to 1
                end
            elseif !isnothing(ci_lo_rr)
                ci_lo_rr
            else
                ci_hi_rr
            end

            compute_e_value(bound_closest_to_null)
        end
    end

    # Generate interpretation
    interpretation = generate_e_value_interpretation(
        e_val,
        e_val_ci,
        rr,
        problem.effect_type,
    )

    return EValueSolution{T,P}(
        T(e_val),
        T(e_val_ci),
        T(rr),
        problem.effect_type,
        interpretation,
        problem,
    )
end
