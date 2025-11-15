"""
Solution type for permutation test (Fisher exact test / randomization inference).

Unlike standard RCT solutions with SE/CI, permutation tests provide exact p-values
under the sharp null hypothesis without distributional assumptions.
"""

"""
    PermutationTestSolution{T<:Real,P}

Solution type for permutation test (randomization inference).

Immutable struct containing p-value, observed statistic, permutation distribution, and metadata.

# Type Parameters
- `T<:Real`: Numeric type matching outcomes (Float64, Float32, BigFloat)
- `P`: Parameter type from original problem

# Fields
- `p_value::Float64`: P-value from permutation test
- `observed_statistic::T`: Observed test statistic (difference-in-means)
- `permutation_distribution::Vector{T}`: Test statistics from all permutations
- `n_permutations::Int`: Number of permutations performed
- `alternative::String`: Alternative hypothesis ("two-sided", "greater", "less")
- `n_treated::Int`: Number of treated units
- `n_control::Int`: Number of control units
- `retcode::Symbol`: Return code indicating estimation status
- `original_problem::RCTProblem{T,P}`: Original problem for full reproducibility

# Return Codes

- `:Success`: Permutation test completed successfully
- `:Warning`: Test completed but with warnings (check carefully)
- `:Error`: Test failed (should not occur - constructor would throw instead)

# Examples

```julia
problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))
solution = solve(problem, PermutationTest(n_permutations=1000, random_seed=42))

# Access results
solution.p_value                    # P-value from permutation test
solution.observed_statistic         # Observed difference-in-means (ATE)
solution.permutation_distribution   # Array of permuted statistics
solution.n_permutations             # Number of permutations performed

# Check significance
solution.p_value < 0.05  # true if significant at 5% level

# Reproducibility: access original inputs
solution.original_problem.outcomes
solution.original_problem.parameters.alpha
```

# Design Rationale

Including `permutation_distribution` enables:
1. Visual diagnostics (histogram of null distribution)
2. Custom p-value calculations (different alternatives)
3. Effect size contextualization (where does observed sit in null?)
4. Sensitivity analysis

Including `original_problem` enables:
1. Full reproducibility (inputs + outputs in one object)
2. Sensitivity analysis via `remake()`
3. Debugging (inspect inputs that led to solution)

# References
- Fisher, R. A. (1935). The Design of Experiments.
- Rosenbaum, P. R. (2002). Observational Studies (Chapter 2).
"""
struct PermutationTestSolution{T<:Real,P} <: AbstractRCTSolution
    p_value::Float64
    observed_statistic::T
    permutation_distribution::Vector{T}
    n_permutations::Int
    alternative::String
    n_treated::Int
    n_control::Int
    retcode::Symbol
    original_problem::RCTProblem{T,P}

    function PermutationTestSolution(;
        p_value::Float64,
        observed_statistic::T,
        permutation_distribution::Vector{T},
        n_permutations::Int,
        alternative::String,
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
                    "Function: PermutationTestSolution constructor\n" *
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
                    "Function: PermutationTestSolution constructor\n" *
                    "Got: n_treated=$(n_treated), n_control=$(n_control)",
                ),
            )
        end

        # Validate n_permutations matches distribution length
        if n_permutations != length(permutation_distribution)
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Permutation count mismatch.\n" *
                    "Function: PermutationTestSolution constructor\n" *
                    "Expected: n_permutations = length(permutation_distribution)\n" *
                    "Got: n_permutations=$(n_permutations), length=$(length(permutation_distribution))",
                ),
            )
        end

        # Validate alternative
        valid_alternatives = ["two-sided", "greater", "less"]
        if !(alternative in valid_alternatives)
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Invalid alternative hypothesis.\n" *
                    "Function: PermutationTestSolution constructor\n" *
                    "Expected: alternative in $(valid_alternatives)\n" *
                    "Got: alternative='$(alternative)'",
                ),
            )
        end

        # Validate p_value in [0, 1]
        if !(0.0 <= p_value <= 1.0)
            throw(
                ArgumentError(
                    "CRITICAL ERROR: P-value out of range.\n" *
                    "Function: PermutationTestSolution constructor\n" *
                    "Expected: p_value in [0, 1]\n" *
                    "Got: p_value=$(p_value)",
                ),
            )
        end

        new{T,P}(
            p_value,
            observed_statistic,
            permutation_distribution,
            n_permutations,
            alternative,
            n_treated,
            n_control,
            retcode,
            original_problem,
        )
    end
end

# Custom show method for nice display
function Base.show(io::IO, solution::PermutationTestSolution)
    println(io, "PermutationTestSolution")
    println(io, "  Status: $(solution.retcode)")
    println(io, "  Observed statistic: $(round(solution.observed_statistic, digits=4))")
    println(io, "  P-value: $(round(solution.p_value, digits=4)) ($(solution.alternative))")
    println(io, "  Permutations: $(solution.n_permutations)")
    alpha = solution.original_problem.parameters.alpha
    significance = solution.p_value < alpha ? "Significant" : "Not significant"
    println(io, "  Significance: $significance at α=$(alpha)")
    println(io, "  Sample: n_treated=$(solution.n_treated), n_control=$(solution.n_control)")
end
