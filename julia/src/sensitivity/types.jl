#=
Sensitivity Analysis Types for Causal Inference

Implements types for E-value and Rosenbaum bounds sensitivity analysis
following the SciML Problem-Estimator-Solution pattern.

References:
- VanderWeele, T. J., & Ding, P. (2017). "Sensitivity Analysis in Observational
  Research: Introducing the E-Value." Annals of Internal Medicine.
- Rosenbaum, P. R. (2002). Observational Studies (2nd ed.). Springer.
=#

# ============================================================================
# Abstract Type Hierarchy
# ============================================================================

"""
    AbstractSensitivityProblem{T,P} <: AbstractCausalProblem{T,P}

Abstract type for sensitivity analysis problems.
"""
abstract type AbstractSensitivityProblem{T<:Real,P} <: AbstractCausalProblem{T,P} end

"""
    AbstractSensitivityEstimator <: AbstractCausalEstimator

Abstract type for sensitivity analysis estimators.
"""
abstract type AbstractSensitivityEstimator <: AbstractCausalEstimator end

"""
    AbstractSensitivitySolution <: AbstractCausalSolution

Abstract type for sensitivity analysis solutions.
"""
abstract type AbstractSensitivitySolution <: AbstractCausalSolution end

# ============================================================================
# Effect Types
# ============================================================================

"""
    EffectType

Enumeration of effect measure types for E-value conversion.

# Values
- `RR`: Risk ratio (relative risk)
- `OR`: Odds ratio
- `HR`: Hazard ratio
- `SMD`: Standardized mean difference (Cohen's d)
- `ATE`: Average treatment effect (risk difference)
"""
@enum EffectType begin
    RR   # Risk ratio
    OR   # Odds ratio
    HR   # Hazard ratio
    SMD  # Standardized mean difference
    ATE  # Average treatment effect
end

"""
    effect_type_from_symbol(s::Symbol) -> EffectType

Convert a symbol to an EffectType enum value.

# Examples
```julia
effect_type_from_symbol(:rr)  # Returns RR
effect_type_from_symbol(:smd) # Returns SMD
```
"""
function effect_type_from_symbol(s::Symbol)::EffectType
    mapping = Dict(
        :rr => RR,
        :or => OR,
        :hr => HR,
        :smd => SMD,
        :ate => ATE,
    )
    haskey(mapping, s) || throw(ArgumentError(
        "Invalid effect type: $s. Must be one of: :rr, :or, :hr, :smd, :ate"
    ))
    return mapping[s]
end

# ============================================================================
# E-Value Problem and Solution Types
# ============================================================================

"""
    EValueProblem{T<:Real,P} <: AbstractSensitivityProblem{T,P}

Problem specification for E-value sensitivity analysis.

# Fields
- `estimate::T`: Point estimate of the effect
- `ci_lower::Union{Nothing,T}`: Lower bound of confidence interval
- `ci_upper::Union{Nothing,T}`: Upper bound of confidence interval
- `effect_type::EffectType`: Type of effect measure
- `baseline_risk::Union{Nothing,T}`: Baseline risk (required for ATE)
- `parameters::P`: Additional parameters

# Constructor
```julia
EValueProblem(estimate; ci_lower=nothing, ci_upper=nothing,
              effect_type=:rr, baseline_risk=nothing)
```

# Example
```julia
# Risk ratio of 2.5 with 95% CI [1.8, 3.5]
problem = EValueProblem(2.5; ci_lower=1.8, ci_upper=3.5, effect_type=:rr)
```
"""
struct EValueProblem{T<:Real,P} <: AbstractSensitivityProblem{T,P}
    estimate::T
    ci_lower::Union{Nothing,T}
    ci_upper::Union{Nothing,T}
    effect_type::EffectType
    baseline_risk::Union{Nothing,T}
    parameters::P

    function EValueProblem(
        estimate::T;
        ci_lower::Union{Nothing,T}=nothing,
        ci_upper::Union{Nothing,T}=nothing,
        effect_type::Union{EffectType,Symbol}=RR,
        baseline_risk::Union{Nothing,T}=nothing,
        parameters::P=(;),
    ) where {T<:Real,P}
        # Convert symbol to EffectType if needed
        etype = effect_type isa Symbol ? effect_type_from_symbol(effect_type) : effect_type

        # Validate ATE requires baseline_risk
        if etype == ATE && isnothing(baseline_risk)
            throw(ArgumentError(
                "baseline_risk is required for ATE effect type"
            ))
        end

        # Validate baseline_risk bounds for ATE
        if etype == ATE && !isnothing(baseline_risk)
            if baseline_risk <= 0 || baseline_risk >= 1
                throw(ArgumentError(
                    "baseline_risk must be in (0, 1), got: $baseline_risk"
                ))
            end
            # Check that ATE doesn't create invalid risk
            new_risk = baseline_risk + estimate
            if new_risk < 0 || new_risk > 1
                throw(ArgumentError(
                    "ATE ($estimate) with baseline_risk ($baseline_risk) creates " *
                    "invalid risk ($new_risk). Risk must be in [0, 1]."
                ))
            end
        end

        # Validate CI ordering
        if !isnothing(ci_lower) && !isnothing(ci_upper)
            if ci_lower > ci_upper
                throw(ArgumentError(
                    "ci_lower ($ci_lower) must be ≤ ci_upper ($ci_upper)"
                ))
            end
        end

        new{T,P}(estimate, ci_lower, ci_upper, etype, baseline_risk, parameters)
    end
end

"""
    EValueSolution{T<:Real,P} <: AbstractSensitivitySolution

Solution from E-value sensitivity analysis.

# Fields
- `e_value::T`: E-value for point estimate
- `e_value_ci::T`: E-value for CI bound closest to null
- `rr_equivalent::T`: Effect converted to risk ratio scale
- `effect_type::EffectType`: Original effect type
- `interpretation::String`: Human-readable interpretation
- `original_problem::EValueProblem{T,P}`: Original problem specification
"""
struct EValueSolution{T<:Real,P} <: AbstractSensitivitySolution
    e_value::T
    e_value_ci::T
    rr_equivalent::T
    effect_type::EffectType
    interpretation::String
    original_problem::EValueProblem{T,P}
end

# ============================================================================
# Rosenbaum Bounds Problem and Solution Types
# ============================================================================

"""
    RosenbaumProblem{T<:Real,P} <: AbstractSensitivityProblem{T,P}

Problem specification for Rosenbaum bounds sensitivity analysis.

# Fields
- `treated_outcomes::Vector{T}`: Outcomes for treated units in matched pairs
- `control_outcomes::Vector{T}`: Outcomes for control units in matched pairs
- `gamma_range::Tuple{T,T}`: Range of Γ values to evaluate
- `n_gamma::Int`: Number of Γ values in grid
- `alpha::T`: Significance level
- `parameters::P`: Additional parameters

# Notes
- Treated and control outcomes must have equal length (matched pairs)
- Γ (gamma) represents the maximum odds ratio of differential treatment
  assignment due to unmeasured confounding

# Example
```julia
treated = [10.5, 12.3, 8.7, 15.2, 11.1]
control = [8.2, 10.1, 7.5, 12.8, 9.3]
problem = RosenbaumProblem(treated, control; gamma_range=(1.0, 3.0))
```
"""
struct RosenbaumProblem{T<:Real,P} <: AbstractSensitivityProblem{T,P}
    treated_outcomes::Vector{T}
    control_outcomes::Vector{T}
    gamma_range::Tuple{T,T}
    n_gamma::Int
    alpha::T
    parameters::P

    function RosenbaumProblem(
        treated_outcomes::Vector{T},
        control_outcomes::Vector{T};
        gamma_range::Tuple{<:Real,<:Real}=(1.0, 3.0),
        n_gamma::Int=20,
        alpha::Real=0.05,
        parameters::P=(;),
    ) where {T<:Real,P}
        # Validate equal lengths
        if length(treated_outcomes) != length(control_outcomes)
            throw(ArgumentError(
                "treated_outcomes ($(length(treated_outcomes))) and " *
                "control_outcomes ($(length(control_outcomes))) must have equal length"
            ))
        end

        # Validate non-empty
        if length(treated_outcomes) == 0
            throw(ArgumentError("Outcomes must be non-empty"))
        end

        # Validate gamma_range
        gamma_low, gamma_high = gamma_range
        if gamma_low < 1.0
            throw(ArgumentError(
                "gamma_range lower bound must be ≥ 1.0, got: $gamma_low"
            ))
        end
        if gamma_high <= gamma_low
            throw(ArgumentError(
                "gamma_range upper bound must be > lower bound, got: ($gamma_low, $gamma_high)"
            ))
        end

        # Validate n_gamma
        if n_gamma < 2
            throw(ArgumentError("n_gamma must be ≥ 2, got: $n_gamma"))
        end

        # Validate alpha
        if alpha <= 0 || alpha >= 1
            throw(ArgumentError("alpha must be in (0, 1), got: $alpha"))
        end

        # Convert gamma_range to proper type
        gamma_range_typed = (T(gamma_low), T(gamma_high))

        new{T,P}(
            treated_outcomes,
            control_outcomes,
            gamma_range_typed,
            n_gamma,
            T(alpha),
            parameters,
        )
    end
end

"""
    RosenbaumSolution{T<:Real,P} <: AbstractSensitivitySolution

Solution from Rosenbaum bounds sensitivity analysis.

# Fields
- `gamma_values::Vector{T}`: Array of Γ values evaluated
- `p_upper::Vector{T}`: Upper bound p-values at each Γ
- `p_lower::Vector{T}`: Lower bound p-values at each Γ
- `gamma_critical::Union{Nothing,T}`: Smallest Γ where p_upper > α
- `observed_statistic::T`: Wilcoxon signed-rank T+ statistic
- `n_pairs::Int`: Number of matched pairs (excluding ties)
- `alpha::T`: Significance level used
- `interpretation::String`: Human-readable interpretation
- `original_problem::RosenbaumProblem{T,P}`: Original problem specification
"""
struct RosenbaumSolution{T<:Real,P} <: AbstractSensitivitySolution
    gamma_values::Vector{T}
    p_upper::Vector{T}
    p_lower::Vector{T}
    gamma_critical::Union{Nothing,T}
    observed_statistic::T
    n_pairs::Int
    alpha::T
    interpretation::String
    original_problem::RosenbaumProblem{T,P}
end

# ============================================================================
# Estimator Types
# ============================================================================

"""
    EValue <: AbstractSensitivityEstimator

Estimator for E-value sensitivity analysis.

The E-value represents the minimum strength of association on the risk ratio
scale that an unmeasured confounder would need to have with both the treatment
and outcome to fully explain away the observed association.

# Usage
```julia
problem = EValueProblem(2.5; ci_lower=1.8, ci_upper=3.5, effect_type=:rr)
solution = solve(problem, EValue())
```

# Reference
VanderWeele, T. J., & Ding, P. (2017). "Sensitivity Analysis in Observational
Research: Introducing the E-Value." Annals of Internal Medicine.
"""
struct EValue <: AbstractSensitivityEstimator end

"""
    RosenbaumBounds <: AbstractSensitivityEstimator

Estimator for Rosenbaum bounds sensitivity analysis.

Evaluates how sensitive the conclusion of a matched pairs study is to
unobserved confounding by computing p-value bounds across a range of
Γ (gamma) values, where Γ represents the maximum odds ratio of differential
treatment assignment.

# Usage
```julia
problem = RosenbaumProblem(treated_outcomes, control_outcomes)
solution = solve(problem, RosenbaumBounds())
```

# Reference
Rosenbaum, P. R. (2002). Observational Studies (2nd ed.). Springer.
"""
struct RosenbaumBounds <: AbstractSensitivityEstimator end
