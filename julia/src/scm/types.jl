#=
Synthetic Control Methods - Type Definitions

Implements the SciML Problem-Estimator-Solution pattern for SCM.

References:
    Abadie, Diamond, & Hainmueller (2010). "Synthetic Control Methods"
    Ben-Michael, Feller, & Rothstein (2021). "Augmented Synthetic Control"
=#

using LinearAlgebra
using Statistics
using Distributions

# =============================================================================
# Abstract Type Hierarchy
# =============================================================================

"""Abstract type for SCM problems."""
abstract type AbstractSCMProblem{T,P} <: AbstractCausalProblem{T,P} end

"""Abstract type for SCM estimators."""
abstract type AbstractSCMEstimator <: AbstractCausalEstimator end

"""Abstract type for SCM solutions."""
abstract type AbstractSCMSolution <: AbstractCausalSolution end

# =============================================================================
# SCMProblem
# =============================================================================

"""
    SCMProblem{T,P}

Problem specification for Synthetic Control Method estimation.

# Fields
- `outcomes::Matrix{T}`: Panel data (n_units × n_periods)
- `treatment::Vector{Bool}`: Treatment indicator (n_units,)
- `treatment_period::Int`: Period when treatment starts (1-indexed)
- `covariates::Union{Nothing,Matrix{T}}`: Optional covariates (n_units × n_covariates)
- `parameters::P`: Named tuple with estimation parameters

# Parameters (in NamedTuple)
- `alpha::Float64`: Significance level for CI (default: 0.05)

# Example
```julia
outcomes = randn(10, 20)  # 10 units, 20 periods
treatment = [true; falses(9)]  # First unit treated
problem = SCMProblem(outcomes, treatment, 10, nothing, (alpha=0.05,))
```
"""
struct SCMProblem{T<:Real,P} <: AbstractSCMProblem{T,P}
    outcomes::Matrix{T}
    treatment::Vector{Bool}
    treatment_period::Int
    covariates::Union{Nothing,Matrix{T}}
    parameters::P

    function SCMProblem(
        outcomes::Matrix{T},
        treatment::Vector{Bool},
        treatment_period::Int,
        covariates::Union{Nothing,Matrix{T}},
        parameters::P,
    ) where {T<:Real,P}
        n_units, n_periods = size(outcomes)

        # Validate dimensions
        if length(treatment) != n_units
            throw(ArgumentError(
                "treatment length ($(length(treatment))) != n_units ($n_units)"
            ))
        end

        # Validate treatment indicator
        n_treated = sum(treatment)
        n_control = n_units - n_treated

        if n_treated == 0
            throw(ArgumentError("No treated units found"))
        end
        if n_control < 2
            throw(ArgumentError(
                "Need at least 2 control units, got $n_control"
            ))
        end

        # Validate treatment period
        if treatment_period < 2
            throw(ArgumentError(
                "treatment_period must be >= 2 (need pre-treatment periods), got $treatment_period"
            ))
        end
        if treatment_period > n_periods
            throw(ArgumentError(
                "treatment_period ($treatment_period) > n_periods ($n_periods)"
            ))
        end

        # Warn about few pre-treatment periods
        n_pre = treatment_period - 1
        if n_pre < 5
            @warn "Only $n_pre pre-treatment periods. SCM works best with >= 5."
        end

        # Check for NaN
        if any(isnan, outcomes)
            throw(ArgumentError("outcomes contains NaN values"))
        end

        # Validate covariates
        if covariates !== nothing
            if size(covariates, 1) != n_units
                throw(ArgumentError(
                    "covariates rows ($(size(covariates, 1))) != n_units ($n_units)"
                ))
            end
            if any(isnan, covariates)
                throw(ArgumentError("covariates contains NaN values"))
            end
        end

        new{T,P}(outcomes, treatment, treatment_period, covariates, parameters)
    end
end

# Convenience constructor without covariates
function SCMProblem(
    outcomes::Matrix{T},
    treatment::Vector{Bool},
    treatment_period::Int,
    parameters::P,
) where {T<:Real,P}
    return SCMProblem(outcomes, treatment, treatment_period, nothing, parameters)
end

# =============================================================================
# Estimator Types
# =============================================================================

"""
    SyntheticControl <: AbstractSCMEstimator

Standard Synthetic Control Method estimator.

# Fields
- `inference::Symbol`: Inference method (:placebo, :bootstrap, :none)
- `n_placebo::Int`: Number of placebo iterations
- `covariate_weight::Float64`: Weight on covariate matching vs outcome matching
"""
Base.@kwdef struct SyntheticControl <: AbstractSCMEstimator
    inference::Symbol = :placebo
    n_placebo::Int = 100
    covariate_weight::Float64 = 1.0

    function SyntheticControl(inference::Symbol, n_placebo::Int, covariate_weight::Float64)
        if inference ∉ (:placebo, :bootstrap, :none)
            throw(ArgumentError(
                "inference must be :placebo, :bootstrap, or :none, got :$inference"
            ))
        end
        if n_placebo < 1
            throw(ArgumentError("n_placebo must be >= 1, got $n_placebo"))
        end
        if covariate_weight < 0
            throw(ArgumentError("covariate_weight must be >= 0"))
        end
        new(inference, n_placebo, covariate_weight)
    end
end

"""
    AugmentedSC <: AbstractSCMEstimator

Augmented Synthetic Control Method (Ben-Michael et al. 2021).

Adds ridge regression bias correction for improved performance
when pre-treatment fit is imperfect.

# Fields
- `inference::Symbol`: Inference method (:jackknife, :bootstrap, :none)
- `lambda::Union{Nothing,Float64}`: Ridge penalty (nothing = CV selection)
"""
Base.@kwdef struct AugmentedSC <: AbstractSCMEstimator
    inference::Symbol = :jackknife
    lambda::Union{Nothing,Float64} = nothing

    function AugmentedSC(inference::Symbol, lambda::Union{Nothing,Float64})
        if inference ∉ (:jackknife, :bootstrap, :none)
            throw(ArgumentError(
                "inference must be :jackknife, :bootstrap, or :none"
            ))
        end
        if lambda !== nothing && lambda < 0
            throw(ArgumentError("lambda must be >= 0"))
        end
        new(inference, lambda)
    end
end

# =============================================================================
# SCMSolution
# =============================================================================

"""
    SCMSolution{T,P}

Solution container for Synthetic Control Method estimation.

# Fields
- `estimate::T`: Average treatment effect on treated (ATT)
- `se::T`: Standard error
- `ci_lower::T`, `ci_upper::T`: Confidence interval bounds
- `p_value::T`: P-value from placebo distribution
- `weights::Vector{T}`: Synthetic control weights for donors
- `pre_rmse::T`: Pre-treatment RMSE
- `pre_r_squared::T`: Pre-treatment R²
- `n_treated::Int`, `n_control::Int`: Unit counts
- `n_pre_periods::Int`, `n_post_periods::Int`: Period counts
- `synthetic_control::Vector{T}`: Counterfactual series
- `treated_series::Vector{T}`: Observed treated series
- `gap::Vector{T}`: Period-by-period effects
- `retcode::Symbol`: :Success, :Warning, or :Error
- `original_problem::SCMProblem{T,P}`: Original problem for reproducibility
"""
struct SCMSolution{T<:Real,P} <: AbstractSCMSolution
    estimate::T
    se::T
    ci_lower::T
    ci_upper::T
    p_value::T
    weights::Vector{T}
    pre_rmse::T
    pre_r_squared::T
    n_treated::Int
    n_control::Int
    n_pre_periods::Int
    n_post_periods::Int
    synthetic_control::Vector{T}
    treated_series::Vector{T}
    gap::Vector{T}
    retcode::Symbol
    original_problem::SCMProblem{T,P}

    function SCMSolution(;
        estimate::T,
        se::T,
        ci_lower::T,
        ci_upper::T,
        p_value::T,
        weights::Vector{T},
        pre_rmse::T,
        pre_r_squared::T,
        n_treated::Int,
        n_control::Int,
        n_pre_periods::Int,
        n_post_periods::Int,
        synthetic_control::Vector{T},
        treated_series::Vector{T},
        gap::Vector{T},
        retcode::Symbol,
        original_problem::SCMProblem{T,P},
    ) where {T<:Real,P}
        # Validate retcode
        if retcode ∉ (:Success, :Warning, :Error)
            throw(ArgumentError("retcode must be :Success, :Warning, or :Error"))
        end

        # Validate dimensions
        if length(weights) != n_control
            throw(ArgumentError(
                "weights length ($(length(weights))) != n_control ($n_control)"
            ))
        end

        n_periods = n_pre_periods + n_post_periods
        if length(synthetic_control) != n_periods
            throw(ArgumentError(
                "synthetic_control length ($(length(synthetic_control))) != n_periods ($n_periods)"
            ))
        end

        new{T,P}(
            estimate, se, ci_lower, ci_upper, p_value,
            weights, pre_rmse, pre_r_squared,
            n_treated, n_control, n_pre_periods, n_post_periods,
            synthetic_control, treated_series, gap,
            retcode, original_problem
        )
    end
end

# =============================================================================
# Pretty Printing
# =============================================================================

function Base.show(io::IO, problem::SCMProblem{T,P}) where {T,P}
    n_units, n_periods = size(problem.outcomes)
    n_treated = sum(problem.treatment)
    n_control = n_units - n_treated
    n_pre = problem.treatment_period - 1
    n_post = n_periods - problem.treatment_period + 1

    print(io, "SCMProblem{$T}")
    print(io, "\n  Units: $n_treated treated, $n_control control")
    print(io, "\n  Periods: $n_pre pre-treatment, $n_post post-treatment")
    if problem.covariates !== nothing
        print(io, "\n  Covariates: $(size(problem.covariates, 2))")
    end
    print(io, "\n  α = $(problem.parameters.alpha)")
end

function Base.show(io::IO, sol::SCMSolution{T,P}) where {T,P}
    print(io, "SCMSolution{$T}")
    print(io, "\n  ATT: $(round(sol.estimate, digits=4))")
    print(io, " (SE: $(round(sol.se, digits=4)))")
    print(io, "\n  95% CI: [$(round(sol.ci_lower, digits=4)), $(round(sol.ci_upper, digits=4))]")
    print(io, "\n  p-value: $(round(sol.p_value, digits=4))")
    print(io, "\n  Pre-treatment fit: R² = $(round(sol.pre_r_squared, digits=3)), RMSE = $(round(sol.pre_rmse, digits=4))")
    print(io, "\n  Donors: $(sol.n_control) ($(sum(sol.weights .> 0.01)) with weight > 1%)")
    print(io, "\n  Status: $(sol.retcode)")
end

function Base.show(io::IO, est::SyntheticControl)
    print(io, "SyntheticControl(inference=:$(est.inference), n_placebo=$(est.n_placebo))")
end

function Base.show(io::IO, est::AugmentedSC)
    λ_str = est.lambda === nothing ? "CV" : string(est.lambda)
    print(io, "AugmentedSC(inference=:$(est.inference), λ=$λ_str)")
end
