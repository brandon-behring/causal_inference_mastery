#=
Rosenbaum Bounds Sensitivity Analysis

Implements sensitivity analysis for matched pairs studies using Wilcoxon
signed-rank test under Γ-confounding. Evaluates how sensitive conclusions
are to unobserved bias in treatment assignment.

References:
- Rosenbaum, P. R. (1987). "Sensitivity Analysis for Certain Permutation
  Inferences in Matched Observational Studies." Biometrika.
- Rosenbaum, P. R. (2002). Observational Studies (2nd ed.). Springer.
- Rosenbaum, P. R. (2010). Design of Observational Studies. Springer.
=#

using Distributions: Normal, cdf

# ============================================================================
# Wilcoxon Signed-Rank Statistics
# ============================================================================

"""
    compute_signed_rank_statistic(differences::Vector{<:Real}) -> Tuple{Float64, Vector{Float64}, Vector{Int}}

Compute Wilcoxon signed-rank test statistic and related quantities.

# Arguments
- `differences`: Vector of paired differences (treated - control)

# Returns
- `T_plus`: Sum of ranks for positive differences
- `ranks`: Ranks of absolute differences (excluding zeros)
- `signs`: Signs of non-zero differences (+1 or -1)

# Algorithm
1. Remove zero differences (ties)
2. Rank absolute values of remaining differences
3. Sum ranks where original difference was positive
"""
function compute_signed_rank_statistic(
    differences::Vector{T},
) where {T<:Real}
    # Remove zeros (ties)
    nonzero_mask = .!isapprox.(differences, 0.0)
    diffs_nonzero = differences[nonzero_mask]

    n = length(diffs_nonzero)
    if n == 0
        return 0.0, Float64[], Int[]
    end

    # Compute absolute values and signs
    abs_diffs = abs.(diffs_nonzero)
    signs = sign.(diffs_nonzero)

    # Rank the absolute differences (average ties)
    ranks = StatsBase.ordinalrank(abs_diffs)  # 1, 2, ..., n

    # Handle ties by averaging ranks
    # Group by absolute value, compute average rank for each group
    unique_vals = unique(abs_diffs)
    rank_map = Dict{eltype(abs_diffs), Float64}()
    for v in unique_vals
        mask = abs_diffs .== v
        avg_rank = mean(ranks[mask])
        rank_map[v] = avg_rank
    end
    ranks_float = [rank_map[v] for v in abs_diffs]

    # T+ = sum of ranks for positive differences
    T_plus = sum(ranks_float[i] for i in 1:n if signs[i] > 0; init=0.0)

    return T_plus, ranks_float, Int.(signs)
end

# ============================================================================
# P-Value Bounds at Given Γ
# ============================================================================

"""
    compute_bounds_at_gamma(ranks, signs, gamma) -> Tuple{Float64, Float64, Float64, Float64}

Compute expectation and variance bounds for T+ at a given Γ level.

Under Γ-confounding, each pair's probability of the treated unit having
a larger outcome is bounded:

    p_i ∈ [1/(1+Γ), Γ/(1+Γ)]

# Arguments
- `ranks`: Vector of ranks for non-zero differences
- `signs`: Vector of signs (+1 or -1)
- `gamma`: Γ level (≥ 1)

# Returns
- `E_upper`: Upper bound on E[T+]
- `E_lower`: Lower bound on E[T+]
- `Var_upper`: Variance at upper probability
- `Var_lower`: Variance at lower probability
"""
function compute_bounds_at_gamma(
    ranks::Vector{Float64},
    signs::Vector{Int},
    gamma::Real,
)::Tuple{Float64,Float64,Float64,Float64}
    n = length(ranks)
    if n == 0
        return 0.0, 0.0, 0.0, 0.0
    end

    # Probability bounds under Γ-confounding
    p_high = gamma / (1.0 + gamma)  # Upper bound on P(treated > control)
    p_low = 1.0 / (1.0 + gamma)     # Lower bound on P(treated > control)

    # Expectation bounds: E[T+] = Σ rank_i × P(D_i > 0)
    # Upper bound: use p_high for positive diffs, (1 - p_low) = p_high for negative
    # For Wilcoxon test under Γ, worst case is all pairs at extreme probability
    E_upper = sum(ranks) * p_high
    E_lower = sum(ranks) * p_low

    # Variance: Var[T+] = Σ rank_i² × p × (1-p)
    sum_ranks_sq = sum(r^2 for r in ranks)
    Var_upper = sum_ranks_sq * p_high * (1.0 - p_high)
    Var_lower = sum_ranks_sq * p_low * (1.0 - p_low)

    return E_upper, E_lower, Var_upper, Var_lower
end

"""
    normal_approximation_p(observed, expectation, variance; upper_tail=true) -> Float64

Compute p-value using normal approximation with continuity correction.

# Arguments
- `observed`: Observed test statistic
- `expectation`: Expected value under null
- `variance`: Variance under null
- `upper_tail`: If true, compute P(T ≥ observed); else P(T ≤ observed)

# Returns
- P-value from normal approximation
"""
function normal_approximation_p(
    observed::Real,
    expectation::Real,
    variance::Real;
    upper_tail::Bool=true,
)::Float64
    if variance ≤ 0
        return upper_tail ? 0.0 : 1.0
    end

    sd = sqrt(variance)

    # Continuity correction
    correction = upper_tail ? -0.5 : 0.5
    z = (observed + correction - expectation) / sd

    # P-value from standard normal
    if upper_tail
        return 1.0 - cdf(Normal(0, 1), z)
    else
        return cdf(Normal(0, 1), z)
    end
end

# ============================================================================
# Interpretation Generation
# ============================================================================

"""
    generate_rosenbaum_interpretation(gamma_critical, n_pairs, alpha) -> String

Generate human-readable interpretation of Rosenbaum bounds results.

# Robustness categories:
- Γ* ∈ [1.0, 1.2): Very sensitive
- Γ* ∈ [1.2, 1.5): Sensitive
- Γ* ∈ [1.5, 2.0): Moderately robust
- Γ* ∈ [2.0, 3.0): Reasonably robust
- Γ* ≥ 3.0: Quite robust
- Γ* = nothing: Robust to all tested Γ values
"""
function generate_rosenbaum_interpretation(
    gamma_critical::Union{Nothing,Real},
    gamma_max::Real,
    n_pairs::Int,
    alpha::Real,
)::String
    alpha_pct = round(alpha * 100, digits=0)

    if isnothing(gamma_critical)
        return "The treatment effect is robust to all tested Γ values " *
               "(up to Γ = $(round(gamma_max, digits=1))). " *
               "Even under substantial hidden bias, the effect remains " *
               "statistically significant at α = $alpha_pct%. " *
               "Analysis based on $n_pairs matched pairs."
    end

    # Robustness interpretation
    robustness = if gamma_critical < 1.2
        "very sensitive to unmeasured confounding"
    elseif gamma_critical < 1.5
        "sensitive to unmeasured confounding"
    elseif gamma_critical < 2.0
        "moderately robust to unmeasured confounding"
    elseif gamma_critical < 3.0
        "reasonably robust to unmeasured confounding"
    else
        "quite robust to unmeasured confounding"
    end

    gamma_str = round(gamma_critical, digits=2)

    return "The treatment effect is $robustness. " *
           "At Γ = $gamma_str, the upper bound p-value exceeds " *
           "α = $alpha_pct%, meaning an unobserved confounder with odds ratio " *
           "of $gamma_str for treatment assignment could potentially explain " *
           "the observed effect. Analysis based on $n_pairs matched pairs."
end

# ============================================================================
# Main Solve Implementation
# ============================================================================

"""
    solve(problem::RosenbaumProblem, estimator::RosenbaumBounds) -> RosenbaumSolution

Compute Rosenbaum bounds sensitivity analysis for matched pairs.

# Algorithm
1. Compute paired differences and Wilcoxon signed-rank statistic T+
2. For each Γ in gamma_range:
   - Compute upper/lower bounds on E[T+] and Var[T+]
   - Compute upper/lower p-values using normal approximation
3. Find critical Γ (smallest where p_upper > α)
4. Generate interpretation

# Example
```julia
treated = [10.5, 12.3, 8.7, 15.2, 11.1]
control = [8.2, 10.1, 7.5, 12.8, 9.3]
problem = RosenbaumProblem(treated, control; gamma_range=(1.0, 3.0))
solution = solve(problem, RosenbaumBounds())

println("Critical Γ: \$(solution.gamma_critical)")
println(solution.interpretation)
```

# Returns
- `RosenbaumSolution` containing:
  - `gamma_values`: Grid of Γ values
  - `p_upper`, `p_lower`: P-value bounds at each Γ
  - `gamma_critical`: Smallest Γ where p_upper > α
  - `interpretation`: Human-readable summary
"""
function solve(problem::RosenbaumProblem{T,P}, ::RosenbaumBounds) where {T<:Real,P}
    # Compute paired differences
    differences = problem.treated_outcomes .- problem.control_outcomes

    # Compute Wilcoxon signed-rank statistic
    T_plus, ranks, signs = compute_signed_rank_statistic(differences)

    # Number of non-zero pairs
    n_pairs = length(ranks)

    # Handle edge case: all differences are zero
    if n_pairs == 0
        gamma_grid = collect(range(
            problem.gamma_range[1],
            problem.gamma_range[2],
            length=problem.n_gamma,
        ))
        return RosenbaumSolution{T,P}(
            T.(gamma_grid),
            ones(T, problem.n_gamma),  # p_upper = 1 (no effect)
            ones(T, problem.n_gamma),  # p_lower = 1
            T(1.0),                    # gamma_critical = 1.0
            T(0.0),                    # observed_statistic = 0
            0,                         # n_pairs = 0
            problem.alpha,
            "All paired differences are zero. No treatment effect detected.",
            problem,
        )
    end

    # Create gamma grid
    gamma_grid = collect(range(
        problem.gamma_range[1],
        problem.gamma_range[2],
        length=problem.n_gamma,
    ))

    # Compute p-value bounds at each gamma
    p_upper = Vector{Float64}(undef, problem.n_gamma)
    p_lower = Vector{Float64}(undef, problem.n_gamma)

    for (i, gamma) in enumerate(gamma_grid)
        E_upper, E_lower, Var_upper, Var_lower = compute_bounds_at_gamma(ranks, signs, gamma)

        # Use max variance for conservative inference
        Var_max = max(Var_upper, Var_lower)

        # Upper p-value: worst case (assume T+ is at its expected upper bound)
        # We test H0: no effect. Under Γ-confounding, the expectation of T+
        # can be as high as E_upper. P-value is P(T+ ≥ observed | H0, Γ)
        p_upper[i] = normal_approximation_p(T_plus, E_upper, Var_max; upper_tail=true)

        # Lower p-value: best case (T+ expected at lower bound)
        p_lower[i] = normal_approximation_p(T_plus, E_lower, Var_max; upper_tail=true)
    end

    # Find critical gamma (first where p_upper exceeds alpha)
    gamma_critical = nothing
    for (i, p) in enumerate(p_upper)
        if p > problem.alpha
            gamma_critical = gamma_grid[i]
            break
        end
    end

    # Generate interpretation
    interpretation = generate_rosenbaum_interpretation(
        gamma_critical,
        problem.gamma_range[2],
        n_pairs,
        problem.alpha,
    )

    return RosenbaumSolution{T,P}(
        T.(gamma_grid),
        T.(p_upper),
        T.(p_lower),
        isnothing(gamma_critical) ? nothing : T(gamma_critical),
        T(T_plus),
        n_pairs,
        problem.alpha,
        interpretation,
        problem,
    )
end
