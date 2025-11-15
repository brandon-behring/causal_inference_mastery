"""
Permutation test for randomization inference (Fisher exact test).
"""

"""
    PermutationTest <: AbstractRCTEstimator

Permutation test for randomization inference (Fisher exact p-values).

Computes exact or Monte Carlo p-values under the sharp null hypothesis of no
treatment effect for any unit.

# Mathematical Foundation

Under Fisher's sharp null (``Y_i(1) = Y_i(0)`` for all ``i``), the treatment
assignment is the only source of randomness. The permutation distribution is:

```math
\\{\\hat{\\tau}^{(b)} : b \\in \\text{all possible randomizations}\\}
```

P-value:

```math
p = \\frac{1}{B} \\sum_{b=1}^B \\mathbb{1}(|\\hat{\\tau}^{(b)}| \\geq |\\hat{\\tau}_{obs}|)
```

where ``B`` is the number of permutations (all possible or Monte Carlo sample).

# Fields

- `n_permutations::Union{Nothing,Int}`: Number of permutations
  - `nothing`: Exact test (enumerate all possible randomizations)
  - `Int`: Monte Carlo test (random sample of permutations)
- `random_seed::Union{Nothing,Int}`: Random seed for reproducibility
- `alternative::String`: Alternative hypothesis ("two-sided", "greater", "less")

# Usage

```julia
# Small sample: exact test
outcomes = [10.0, 12.0, 11.0, 4.0, 5.0, 3.0]
treatment = [true, true, true, false, false, false]
problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))

estimator = PermutationTest(nothing, 42)  # exact test
solution = solve(problem, estimator)

# Large sample: Monte Carlo
estimator = PermutationTest(1000, 42)  # 1000 permutations
solution = solve(problem, estimator)
```

# Returns

Solution includes:
- `estimate`: Observed ATE
- `se`: Standard deviation of permutation distribution
- `p_value`: Two-sided permutation p-value (stored in retcode or separate field)

# Computational Cost

- Exact test: ``O(\\binom{n}{n_1})`` - feasible for n < 20
- Monte Carlo: ``O(B \\cdot n)`` - linear in permutations

# References

- Fisher, R. A. (1935). *The Design of Experiments*.
- Rosenbaum, P. R. (2002). *Observational Studies*. Springer. Chapter 2.
- Imbens & Rubin (2015), Chapter 5: Randomization Inference.
"""
struct PermutationTest <: AbstractRCTEstimator
    n_permutations::Union{Nothing,Int}
    random_seed::Union{Nothing,Int}
    alternative::String

    function PermutationTest(
        n_permutations::Union{Nothing,Int},
        random_seed::Union{Nothing,Int},
        alternative::String,
    )
        # Validate n_permutations if provided
        if !isnothing(n_permutations) && n_permutations <= 0
            throw(
                ArgumentError(
                    "n_permutations must be positive if provided, got $(n_permutations)",
                ),
            )
        end

        # Validate alternative
        valid_alternatives = ["two-sided", "greater", "less"]
        if !(alternative in valid_alternatives)
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Invalid alternative hypothesis.\n" *
                    "Expected: alternative in $(valid_alternatives)\n" *
                    "Got: alternative='$(alternative)'",
                ),
            )
        end

        new(n_permutations, random_seed, alternative)
    end
end

# Convenience constructors with defaults
PermutationTest() = PermutationTest(nothing, nothing, "two-sided")  # exact test, no seed, two-sided
PermutationTest(n_permutations::Int) = PermutationTest(n_permutations, nothing, "two-sided")  # Monte Carlo, no seed, two-sided
PermutationTest(n_permutations::Int, random_seed::Int) = PermutationTest(n_permutations, random_seed, "two-sided")  # Monte Carlo with seed, two-sided

"""
    solve(problem::RCTProblem, estimator::PermutationTest)::PermutationTestSolution

Perform permutation test for treatment effect (Fisher exact test / randomization inference).

Under the sharp null hypothesis of no treatment effect for any unit, treatment
assignment is the only source of randomness. We compute the exact distribution
of the test statistic under all possible randomizations.

# Algorithm

1. Compute observed test statistic (difference-in-means)
2. Generate permutation distribution:
   - If `n_permutations === nothing`: Exact test (enumerate all C(n, n1) combinations)
   - Else: Monte Carlo test (randomly sample n_permutations)
3. Compute p-value based on alternative hypothesis:
   - "two-sided": proportion with |stat| >= |observed|
   - "greater": proportion with stat >= observed
   - "less": proportion with stat <= observed

# Mathematical Foundation

**Sharp null hypothesis**: Y_i(1) = Y_i(0) for all i (no effect for any unit)

Under this null, treatment assignments are the only randomness. The observed
outcomes are fixed, so we can compute the exact distribution of the test
statistic by permuting treatment assignments.

**Test statistic**: τ̂ = ȳ₁ - ȳ₀ (difference-in-means)

**Exact p-value**: Proportion of all C(n, n₁) permutations with statistic as extreme as observed

**Monte Carlo p-value**: Proportion of randomly sampled permutations with statistic as extreme

# Validation

- Problem must have valid treatment variation (validated in RCTProblem constructor)
- If n_permutations provided, must be > 0

# Examples

```julia
# Exact test (small sample)
problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))
solution = solve(problem, PermutationTest())

# Monte Carlo test (larger sample)
solution = solve(problem, PermutationTest(n_permutations=10000, random_seed=42))

# Check significance
solution.p_value < 0.05  # true if significant
```

# References

- Fisher, R. A. (1935). The Design of Experiments.
- Rosenbaum, P. R. (2002). Observational Studies, Chapter 2.
"""
function solve(problem::RCTProblem, estimator::PermutationTest)::PermutationTestSolution
    (; outcomes, treatment, parameters) = problem
    (; n_permutations, random_seed, alternative) = estimator

    n = length(outcomes)
    n_treated = sum(treatment)
    n_control = sum(.!treatment)

    # ============================================================================
    # Helper function: compute test statistic (difference-in-means)
    # ============================================================================
    function compute_test_statistic(y::Vector{Float64}, t::AbstractVector{Bool})::Float64
        y1 = y[t]
        y0 = y[.!t]
        return mean(y1) - mean(y0)
    end

    # ============================================================================
    # Observed Test Statistic
    # ============================================================================
    observed_statistic = compute_test_statistic(outcomes, treatment)

    # ============================================================================
    # Permutation Distribution
    # ============================================================================
    if isnothing(n_permutations)
        # Exact test: enumerate all C(n, n_treated) permutations
        permutation_stats = Float64[]
        indices = 1:n

        # Enumerate all combinations of n_treated indices (treated positions)
        for treated_indices in combinations(indices, n_treated)
            t_perm = falses(n)
            t_perm[treated_indices] .= true
            stat = compute_test_statistic(outcomes, t_perm)
            push!(permutation_stats, stat)
        end

        permutation_distribution = permutation_stats
        n_perms_performed = length(permutation_stats)
    else
        # Monte Carlo: randomly sample permutations
        if !isnothing(random_seed)
            Random.seed!(random_seed)
        end

        permutation_stats = Float64[]
        for _ in 1:n_permutations
            # Randomly permute treatment assignments
            t_perm = treatment[randperm(n)]
            stat = compute_test_statistic(outcomes, t_perm)
            push!(permutation_stats, stat)
        end

        permutation_distribution = permutation_stats
        n_perms_performed = n_permutations
    end

    # ============================================================================
    # P-value Calculation
    # ============================================================================
    if alternative == "two-sided"
        # P-value = proportion of permutations with |stat| >= |observed|
        p_value = mean(abs.(permutation_distribution) .>= abs(observed_statistic))
    elseif alternative == "greater"
        # P-value = proportion of permutations with stat >= observed
        p_value = mean(permutation_distribution .>= observed_statistic)
    else  # alternative == "less"
        # P-value = proportion of permutations with stat <= observed
        p_value = mean(permutation_distribution .<= observed_statistic)
    end

    return PermutationTestSolution(
        p_value = p_value,
        observed_statistic = observed_statistic,
        permutation_distribution = permutation_distribution,
        n_permutations = n_perms_performed,
        alternative = alternative,
        n_treated = n_treated,
        n_control = n_control,
        retcode = :Success,
        original_problem = problem,
    )
end
