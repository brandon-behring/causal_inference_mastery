"""
Propensity Score Matching (PSM) problem definition.
"""

"""
    PSMProblem <: AbstractPSMProblem

Problem definition for propensity score matching estimation.

Propensity score matching estimates treatment effects by matching treated and control
units with similar propensity scores (probability of treatment given covariates).

# Fields
- `outcomes::Vector{Float64}`: Observed outcomes Y
- `treatment::Vector{Bool}`: Binary treatment indicator (true = treated, false = control)
- `covariates::Matrix{Float64}`: Covariate matrix X (n × p), used for propensity estimation
- `parameters::NamedTuple`: Additional parameters (alpha, matching_type, caliper, etc.)

# Requirements
- Treatment must be binary (no missing values)
- Covariates required (cannot be nothing)
- Sufficient treated AND control units for matching
- No NaN/Inf in outcomes or covariates
- Treatment variation exists (not all treated or all control)

# Example
```julia
outcomes = [10.0, 12.0, 4.0, 5.0, 11.0, 3.0]
treatment = [true, true, true, false, false, false]
covariates = [5.0 2.0; 6.0 3.0; 5.5 2.5; 4.5 2.0; 5.2 2.3; 4.8 1.9]

problem = PSMProblem(
    outcomes,
    treatment,
    covariates,
    (alpha = 0.05, matching_type = :nearest_neighbor)
)
```

# Notes
- Propensity scores will be estimated from covariates during solve()
- Common support (overlap in propensity distributions) checked during matching
- Units outside common support may be dropped depending on matching method
"""
struct PSMProblem{T<:Real,P} <: AbstractPSMProblem{T,P}
    outcomes::Vector{T}
    treatment::Vector{Bool}
    covariates::Matrix{T}
    parameters::P

    function PSMProblem(
        outcomes::Vector{T},
        treatment::AbstractVector{Bool},
        covariates::Matrix{T},
        parameters::P,
    ) where {T<:Real,P}

        # ====================================================================
        # Validate Inputs
        # ====================================================================

        n = length(outcomes)

        # Check lengths match
        if length(treatment) != n
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Mismatched lengths.\n" *
                    "Function: PSMProblem constructor\n" *
                    "outcomes has length $n, treatment has length $(length(treatment))\n" *
                    "All inputs must have same length.",
                ),
            )
        end

        if size(covariates, 1) != n
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Mismatched covariate rows.\n" *
                    "Function: PSMProblem constructor\n" *
                    "outcomes has length $n, covariates has $(size(covariates, 1)) rows\n" *
                    "Covariate matrix must have n rows.",
                ),
            )
        end

        # Check for empty inputs
        if n == 0
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Empty inputs.\n" *
                    "Function: PSMProblem constructor\n" *
                    "Cannot estimate treatment effects with zero observations.\n" *
                    "Provide non-empty outcome and treatment vectors.",
                ),
            )
        end

        # Check for NaN/Inf in outcomes
        if any(isnan, outcomes) || any(isinf, outcomes)
            throw(
                ArgumentError(
                    "CRITICAL ERROR: NaN or Inf in outcomes.\n" *
                    "Function: PSMProblem constructor\n" *
                    "Outcomes contain $(count(isnan, outcomes)) NaN and $(count(isinf, outcomes)) Inf values.\n" *
                    "Remove or impute missing/invalid outcome values.",
                ),
            )
        end

        # Check for NaN/Inf in covariates
        if any(isnan, covariates) || any(isinf, covariates)
            throw(
                ArgumentError(
                    "CRITICAL ERROR: NaN or Inf in covariates.\n" *
                    "Function: PSMProblem constructor\n" *
                    "Covariates contain $(count(isnan, covariates)) NaN and $(count(isinf, covariates)) Inf values.\n" *
                    "Remove or impute missing/invalid covariate values before PSM.",
                ),
            )
        end

        # Check treatment variation
        n_treated = sum(treatment)
        n_control = n - n_treated

        if n_treated == 0
            throw(
                ArgumentError(
                    "CRITICAL ERROR: No treated units.\n" *
                    "Function: PSMProblem constructor\n" *
                    "All $n units are in control group.\n" *
                    "Cannot estimate treatment effect without treated units.",
                ),
            )
        end

        if n_control == 0
            throw(
                ArgumentError(
                    "CRITICAL ERROR: No control units.\n" *
                    "Function: PSMProblem constructor\n" *
                    "All $n units are treated.\n" *
                    "Cannot estimate treatment effect without control units for matching.",
                ),
            )
        end

        # Check sufficient sample size for matching
        min_group_size = 2
        if n_treated < min_group_size
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Insufficient treated units.\n" *
                    "Function: PSMProblem constructor\n" *
                    "Only $n_treated treated unit(s), need at least $min_group_size for matching.\n" *
                    "Increase sample size or use RCT methods.",
                ),
            )
        end

        if n_control < min_group_size
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Insufficient control units.\n" *
                    "Function: PSMProblem constructor\n" *
                    "Only $n_control control unit(s), need at least $min_group_size for matching.\n" *
                    "Increase sample size or use RCT methods.",
                ),
            )
        end

        # Validate alpha parameter if present
        if haskey(parameters, :alpha)
            α = parameters.alpha
            if !(0 < α < 1)
                throw(
                    ArgumentError(
                        "CRITICAL ERROR: Invalid significance level.\n" *
                        "Function: PSMProblem constructor\n" *
                        "alpha must be in (0, 1), got alpha = $α\n" *
                        "Common values: 0.01, 0.05, 0.10",
                    ),
                )
            end
        end

        # Convert treatment to Vector{Bool} if needed (e.g., from BitVector)
        treatment_vec = treatment isa Vector{Bool} ? treatment : Vector{Bool}(treatment)

        new{T,P}(outcomes, treatment_vec, covariates, parameters)
    end
end

"""
    PSMSolution <: AbstractCausalSolution

Solution from propensity score matching estimation.

# Fields
- `estimate::Float64`: ATE estimate (average treatment effect)
- `se::Float64`: Standard error of ATE estimate
- `ci_lower::Float64`: Lower confidence interval bound
- `ci_upper::Float64`: Upper confidence interval bound
- `n_treated::Int`: Number of treated units used in estimation
- `n_control::Int`: Number of control units used in estimation
- `n_matched::Int`: Number of successfully matched pairs/units
- `propensity_scores::Vector{Float64}`: Estimated propensity scores for all units
- `matched_indices::Vector{Tuple{Int,Int}}`: Indices of matched pairs (treated, control)
- `balance_metrics::NamedTuple`: Balance diagnostics (SMD, variance ratios, etc.)
- `retcode::Symbol`: Return code (:Success, :BalanceFailed, :CommonSupportFailed, etc.)
- `original_problem::PSMProblem`: Original problem definition
"""
struct PSMSolution <: AbstractCausalSolution
    estimate::Float64
    se::Float64
    ci_lower::Float64
    ci_upper::Float64
    n_treated::Int
    n_control::Int
    n_matched::Int
    propensity_scores::Vector{Float64}
    matched_indices::Vector{Tuple{Int,Int}}
    balance_metrics::NamedTuple
    retcode::Symbol
    original_problem::PSMProblem
end
