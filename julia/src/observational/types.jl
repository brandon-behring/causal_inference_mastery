#=
Observational Causal Inference Types

Defines Problem-Estimator-Solution types for observational studies with confounding.
Uses inverse probability weighting (IPW) and doubly robust (DR) methods.

Architecture:
- ObservationalProblem: Data + parameters specification
- AbstractObservationalEstimator: Algorithm interface
- IPWSolution / DRSolution: Results with diagnostics

References:
- Rosenbaum & Rubin (1983). The central role of the propensity score.
- Austin & Stuart (2015). Moving towards best practice with IPTW.
- Bang & Robins (2005). Doubly robust estimation.
=#

# =============================================================================
# Abstract Type Hierarchy
# =============================================================================

"""
    AbstractObservationalProblem{T,P} <: AbstractCausalProblem{T,P}

Abstract type for observational study problems with confounding.

All observational problems must specify:
- Outcomes
- Treatment assignment
- Covariates (confounders)
- Optional: pre-computed propensity scores
"""
abstract type AbstractObservationalProblem{T<:Real,P<:NamedTuple} <: AbstractCausalProblem{T,P} end

"""
    AbstractObservationalEstimator <: AbstractCausalEstimator

Abstract type for observational causal effect estimators.

Concrete subtypes: `ObservationalIPW`, `DoublyRobust`
"""
abstract type AbstractObservationalEstimator <: AbstractCausalEstimator end

"""
    AbstractObservationalSolution <: AbstractCausalSolution

Abstract type for observational study solutions.

All solutions include:
- Point estimate
- Standard error
- Confidence interval
- Propensity score diagnostics
- Return code
"""
abstract type AbstractObservationalSolution <: AbstractCausalSolution end


# =============================================================================
# ObservationalProblem
# =============================================================================

"""
    ObservationalProblem{T,P} <: AbstractObservationalProblem{T,P}

Specification for an observational causal inference problem.

# Mathematical Framework

In observational studies, treatment assignment depends on covariates:

    P(T=1|X) = e(X)  (propensity score)

Key assumptions for IPW identification:
1. **Unconfoundedness**: Y(0), Y(1) ⟂ T | X
2. **Positivity**: 0 < e(X) < 1 for all X in support
3. **SUTVA**: No interference, no hidden treatments

# Fields
- `outcomes::Vector{T}`: Observed outcomes Y
- `treatment::Vector{Bool}`: Treatment indicators T ∈ {0,1}
- `covariates::Matrix{T}`: Covariate matrix X (n × p)
- `propensity::Union{Nothing,Vector{T}}`: Pre-computed propensity scores (optional)
- `parameters::P`: Named tuple with estimation parameters

# Parameters NamedTuple
- `alpha::Float64`: Significance level for CI (default: 0.05)
- `trim_threshold::Float64`: Trim propensities outside (ε, 1-ε) (default: 0.01)
- `stabilize::Bool`: Use stabilized weights (default: false)

# Validation
- All arrays must have same length n
- Treatment must be binary (0/1 or false/true)
- Covariates must be n × p matrix (p ≥ 1)
- If propensity provided, must have length n and values in (0,1)
- Must have observations in both treatment groups

# Example
```julia
using CausalEstimators

# Simulated observational data with confounding
n = 500
X = randn(n, 2)
logit = 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2]
e_true = 1 ./ (1 .+ exp.(-logit))
T = rand(n) .< e_true
Y = 2.0 .* T .+ 0.5 .* X[:, 1] .+ randn(n)

# Create problem
problem = ObservationalProblem(
    Y, T, X, nothing,
    (alpha=0.05, trim_threshold=0.01, stabilize=false)
)

# Solve with IPW
solution = solve(problem, ObservationalIPW())
```

# References
- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity
  score in observational studies for causal effects. Biometrika, 70(1), 41-55.
"""
struct ObservationalProblem{T<:Real,P<:NamedTuple} <: AbstractObservationalProblem{T,P}
    outcomes::Vector{T}
    treatment::Vector{Bool}
    covariates::Matrix{T}
    propensity::Union{Nothing,Vector{T}}
    parameters::P

    function ObservationalProblem(
        outcomes::AbstractVector{T},
        treatment::AbstractVector,
        covariates::AbstractMatrix{T},
        propensity::Union{Nothing,AbstractVector{T}},
        parameters::P
    ) where {T<:Real,P<:NamedTuple}
        # Validate dimensions
        n = length(outcomes)

        if length(treatment) != n
            throw(ArgumentError(
                "Treatment length ($(length(treatment))) must match outcomes length ($n)"
            ))
        end

        if size(covariates, 1) != n
            throw(ArgumentError(
                "Covariates rows ($(size(covariates, 1))) must match outcomes length ($n)"
            ))
        end

        if size(covariates, 2) < 1
            throw(ArgumentError(
                "Covariates must have at least 1 column, got $(size(covariates, 2))"
            ))
        end

        # Convert treatment to Bool
        treatment_bool = convert(Vector{Bool}, treatment)

        # Validate treatment has both groups
        n_treated = sum(treatment_bool)
        n_control = n - n_treated

        if n_treated == 0
            throw(ArgumentError("No treated units found. Treatment must be binary with both groups."))
        end

        if n_control == 0
            throw(ArgumentError("No control units found. Treatment must be binary with both groups."))
        end

        # Validate propensity if provided
        if propensity !== nothing
            if length(propensity) != n
                throw(ArgumentError(
                    "Propensity length ($(length(propensity))) must match outcomes length ($n)"
                ))
            end

            # Check propensity bounds
            if any(propensity .<= 0) || any(propensity .>= 1)
                throw(ArgumentError(
                    "Propensity scores must be in (0, 1) exclusive. " *
                    "Got min=$(minimum(propensity)), max=$(maximum(propensity))"
                ))
            end
        end

        # Validate outcomes are finite
        if any(!isfinite, outcomes)
            throw(ArgumentError("Outcomes contain non-finite values (NaN or Inf)"))
        end

        # Validate covariates are finite
        if any(!isfinite, covariates)
            throw(ArgumentError("Covariates contain non-finite values (NaN or Inf)"))
        end

        new{T,P}(
            convert(Vector{T}, outcomes),
            treatment_bool,
            convert(Matrix{T}, covariates),
            propensity === nothing ? nothing : convert(Vector{T}, propensity),
            parameters
        )
    end
end

# Convenience constructor with default parameters
function ObservationalProblem(
    outcomes::AbstractVector{T},
    treatment::AbstractVector,
    covariates::AbstractMatrix{T};
    propensity::Union{Nothing,AbstractVector{T}} = nothing,
    alpha::Float64 = 0.05,
    trim_threshold::Float64 = 0.01,
    stabilize::Bool = false
) where {T<:Real}
    params = (alpha=alpha, trim_threshold=trim_threshold, stabilize=stabilize)
    return ObservationalProblem(outcomes, treatment, covariates, propensity, params)
end


# =============================================================================
# IPWSolution
# =============================================================================

"""
    IPWSolution{T} <: AbstractObservationalSolution

Solution from Inverse Probability Weighting (IPW) estimation.

# Mathematical Formulation

IPW ATE estimator:

    τ̂_IPW = (1/n) Σᵢ [Tᵢ Yᵢ / e(Xᵢ) - (1-Tᵢ) Yᵢ / (1-e(Xᵢ))]

Stabilized IPW (sIPW):

    τ̂_sIPW = [Σᵢ Tᵢ Yᵢ / e(Xᵢ)] / [Σᵢ Tᵢ / e(Xᵢ)] -
              [Σᵢ (1-Tᵢ) Yᵢ / (1-e(Xᵢ))] / [Σᵢ (1-Tᵢ) / (1-e(Xᵢ))]

Robust variance (sandwich estimator):

    Var(τ̂) = (1/n²) Σᵢ φᵢ²

where φᵢ is the influence function.

# Fields
- `estimate::T`: IPW estimate of ATE
- `se::T`: Robust standard error
- `ci_lower::T`: Lower bound of confidence interval
- `ci_upper::T`: Upper bound of confidence interval
- `p_value::T`: Two-sided p-value (H₀: ATE = 0)
- `n_treated::Int`: Number of treated units
- `n_control::Int`: Number of control units
- `n_trimmed::Int`: Number of units trimmed for extreme propensities
- `propensity_scores::Vector{T}`: Estimated (or provided) propensity scores
- `weights::Vector{T}`: IPW weights used
- `propensity_auc::T`: AUC of propensity model (discriminatory power)
- `propensity_mean_treated::T`: Mean propensity among treated
- `propensity_mean_control::T`: Mean propensity among control
- `stabilized::Bool`: Whether stabilized weights were used
- `retcode::Symbol`: `:Success`, `:Warning`, or `:Error`
- `original_problem::ObservationalProblem`: Original problem specification

# Return Codes
- `:Success`: Estimation succeeded without issues
- `:Warning`: Estimation succeeded but with concerns (e.g., extreme weights)
- `:Error`: Estimation failed

# Example
```julia
solution = solve(problem, ObservationalIPW())

println("ATE: \$(solution.estimate) ± \$(solution.se)")
println("95% CI: [\$(solution.ci_lower), \$(solution.ci_upper)]")
println("Propensity AUC: \$(solution.propensity_auc)")
```
"""
struct IPWSolution{T<:Real} <: AbstractObservationalSolution
    estimate::T
    se::T
    ci_lower::T
    ci_upper::T
    p_value::T
    n_treated::Int
    n_control::Int
    n_trimmed::Int
    propensity_scores::Vector{T}
    weights::Vector{T}
    propensity_auc::T
    propensity_mean_treated::T
    propensity_mean_control::T
    stabilized::Bool
    retcode::Symbol
    original_problem::ObservationalProblem{T}
end

# Pretty printing
function Base.show(io::IO, sol::IPWSolution)
    println(io, "IPWSolution")
    println(io, "=" ^ 50)
    println(io, "ATE Estimate:     $(round(sol.estimate, digits=4))")
    println(io, "Std. Error:       $(round(sol.se, digits=4))")
    println(io, "95% CI:           [$(round(sol.ci_lower, digits=4)), $(round(sol.ci_upper, digits=4))]")
    println(io, "p-value:          $(round(sol.p_value, digits=4))")
    println(io, "-" ^ 50)
    println(io, "n_treated:        $(sol.n_treated)")
    println(io, "n_control:        $(sol.n_control)")
    println(io, "n_trimmed:        $(sol.n_trimmed)")
    println(io, "-" ^ 50)
    println(io, "Propensity AUC:   $(round(sol.propensity_auc, digits=4))")
    println(io, "Mean P(T|X) treated: $(round(sol.propensity_mean_treated, digits=4))")
    println(io, "Mean P(T|X) control: $(round(sol.propensity_mean_control, digits=4))")
    println(io, "Stabilized:       $(sol.stabilized)")
    println(io, "Return Code:      $(sol.retcode)")
    println(io, "=" ^ 50)
end
