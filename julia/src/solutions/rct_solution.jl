"""
Solution types for RCT estimation problems.

Following SciML pattern: Solutions contain results + metadata + original problem for reproducibility.
"""

"""
    RCTSolution{T<:Real,P}

Solution type for RCT estimation problems.

Immutable struct containing point estimate, uncertainty quantification, and metadata.

# Type Parameters
- `T<:Real`: Numeric type matching outcomes (Float64, Float32, BigFloat)
- `P`: Parameter type from original problem

# Fields
- `estimate::T`: Point estimate of average treatment effect (ATE)
- `se::T`: Standard error of the estimate
- `ci_lower::T`: Lower confidence interval bound
- `ci_upper::T`: Upper confidence interval bound
- `n_treated::Int`: Number of treated units
- `n_control::Int`: Number of control units
- `retcode::Symbol`: Return code indicating estimation status
- `original_problem::RCTProblem{T,P}`: Original problem for full reproducibility

# Return Codes

- `:Success`: Estimation completed successfully
- `:Warning`: Estimation completed but with warnings (check carefully)
- `:Error`: Estimation failed (should not occur - constructor would throw instead)

# Examples

```julia
problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))
solution = solve(problem, SimpleATE())

# Access results
solution.estimate      # Point estimate
solution.se            # Standard error
solution.ci_lower      # Lower 95% CI
solution.ci_upper      # Upper 95% CI

# Check status
solution.retcode == :Success  # true if succeeded

# Reproducibility: access original inputs
solution.original_problem.outcomes
solution.original_problem.parameters.alpha
```

# Design Rationale

Including `original_problem` enables:
1. Full reproducibility (inputs + outputs in one object)
2. Sensitivity analysis via `remake()`
3. Debugging (inspect inputs that led to solution)
4. Documentation (what parameters were used?)

# References
- SciML pattern: https://docs.sciml.ai/
"""
struct RCTSolution{T<:Real,P} <: AbstractRCTSolution
    estimate::T
    se::T
    ci_lower::T
    ci_upper::T
    n_treated::Int
    n_control::Int
    retcode::Symbol
    original_problem::RCTProblem{T,P}

    function RCTSolution(;
        estimate::T,
        se::T,
        ci_lower::T,
        ci_upper::T,
        n_treated::Int,
        n_control::Int,
        retcode::Symbol,
        original_problem::RCTProblem{T,P},
    ) where {T<:Real,P}
        # Validate retcode
        valid_codes = (:Success, :Warning, :Error)
        if !(retcode in valid_codes)
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Invalid return code.\n" *
                    "Function: RCTSolution constructor\n" *
                    "Valid codes: $(valid_codes)\n" *
                    "Got: $(retcode)",
                ),
            )
        end

        # Validate counts are non-negative
        if n_treated < 0 || n_control < 0
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Negative sample sizes.\n" *
                    "Function: RCTSolution constructor\n" *
                    "Got: n_treated=$(n_treated), n_control=$(n_control)",
                ),
            )
        end

        new{T,P}(estimate, se, ci_lower, ci_upper, n_treated, n_control, retcode, original_problem)
    end
end

# Custom show method for nice display
function Base.show(io::IO, solution::RCTSolution)
    println(io, "RCTSolution")
    println(io, "  Status: $(solution.retcode)")
    println(io, "  ATE estimate: $(round(solution.estimate, digits=4))")
    println(io, "  Standard error: $(round(solution.se, digits=4))")
    alpha = solution.original_problem.parameters.alpha
    ci_level = Int(round((1 - alpha) * 100))
    println(io, "  $(ci_level)% CI: [$(round(solution.ci_lower, digits=4)), $(round(solution.ci_upper, digits=4))]")
    println(io, "  Sample: n_treated=$(solution.n_treated), n_control=$(solution.n_control)")
end
