"""
Abstract type hierarchy for causal inference problems.

Following SciML pattern: three-level hierarchy (Universal → Method → Concrete).
"""

# Level 1: Universal causal inference base
abstract type AbstractCausalProblem{T,P} end
abstract type AbstractCausalEstimator end
abstract type AbstractCausalSolution end

# Level 2: Method-specific (RCT, DiD, IV, RDD, PSM, etc.)
abstract type AbstractRCTProblem{T,P} <: AbstractCausalProblem{T,P} end
abstract type AbstractRCTEstimator <: AbstractCausalEstimator end
abstract type AbstractRCTSolution <: AbstractCausalSolution end

abstract type AbstractPSMProblem{T,P} <: AbstractCausalProblem{T,P} end
abstract type AbstractPSMEstimator <: AbstractCausalEstimator end
abstract type AbstractPSMSolution <: AbstractCausalSolution end

"""
    RCTProblem{T<:Real,P}

Problem specification for Randomized Controlled Trial (RCT) estimation.

Immutable struct following SciML pattern. Contains all data needed for treatment effect estimation.

# Type Parameters
- `T<:Real`: Numeric type for outcomes (Float64, Float32, BigFloat)
- `P`: Parameter type (typically NamedTuple with fields like `alpha`)

# Fields
- `outcomes::Vector{T}`: Observed outcomes for all units
- `treatment::Vector{Bool}`: Treatment assignment (true=treated, false=control)
- `covariates::Union{Nothing,Matrix{T}}`: Optional covariates for adjustment (n × p matrix)
- `strata::Union{Nothing,Vector{Int}}`: Optional stratification indicators
- `parameters::P`: Estimation parameters (e.g., `(alpha=0.05,)` for 95% CI)

# Constructor Behavior

Inner constructor validates inputs and fails fast following Brandon's principle:
- Checks lengths match
- Checks for NaN/Inf values
- Validates treatment is binary
- Validates treatment variation exists
- Validates covariate/strata dimensions

# Examples

```julia
# Simple unstratified RCT
outcomes = [10.0, 12.0, 4.0, 5.0]
treatment = [true, true, false, false]
problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))

# Stratified RCT (block randomization)
strata = [1, 1, 2, 2]  # Two blocks
problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha=0.05,))

# RCT with covariate adjustment
X = [5.0 2.0; 6.0 3.0; 5.5 2.5; 4.5 2.0]  # n × 2 covariate matrix
problem = RCTProblem(outcomes, treatment, X, nothing, (alpha=0.05,))
```

# References
- Imbens & Rubin (2015), Chapter 6: The Randomized Experiment
"""
struct RCTProblem{T<:Real,P} <: AbstractRCTProblem{T,P}
    outcomes::Vector{T}
    treatment::Vector{Bool}
    covariates::Union{Nothing,Matrix{T}}
    strata::Union{Nothing,Vector{Int}}
    parameters::P

    function RCTProblem(
        outcomes::Vector{T},
        treatment::Vector{Bool},
        covariates::Union{Nothing,Matrix{T}},
        strata::Union{Nothing,Vector{Int}},
        parameters::P,
    ) where {T<:Real,P}
        # Validate inputs (fail fast - Brandon's principle)
        validate_rct_inputs(outcomes, treatment, covariates, strata)

        new{T,P}(outcomes, treatment, covariates, strata, parameters)
    end
end

"""
    remake(problem::RCTProblem; kwargs...)

Create modified copy of RCT problem (for sensitivity analysis).

Following SciML `remake()` pattern for immutable structs.

# Arguments
- `outcomes`: New outcomes (default: original)
- `treatment`: New treatment (default: original)
- `covariates`: New covariates (default: original)
- `strata`: New strata (default: original)
- `parameters`: New parameters (default: original)

# Examples

```julia
# Change alpha for different CI
original = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))
new_problem = remake(original, parameters=(alpha=0.01,))

# Add covariates to existing problem
with_covariates = remake(original, covariates=X)
```
"""
function remake(
    problem::RCTProblem;
    outcomes=problem.outcomes,
    treatment=problem.treatment,
    covariates=problem.covariates,
    strata=problem.strata,
    parameters=problem.parameters,
)
    return RCTProblem(outcomes, treatment, covariates, strata, parameters)
end
