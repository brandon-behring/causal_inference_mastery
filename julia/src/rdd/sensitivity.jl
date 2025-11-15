"""
Sensitivity Analysis for Regression Discontinuity Design.

Implements 6 key sensitivity checks:
1. Bandwidth sensitivity: How estimates change with h
2. Placebo cutoffs: Test at fake cutoffs
3. Balance testing: Covariates balanced at cutoff?
4. Donut RDD: Exclude observations near cutoff
5. Kernel sensitivity: Compare different kernels (already in types)
6. Permutation test: Randomization inference

Following Cattaneo, Idrobo, Titiunik (2019) best practices.
"""

using Statistics
using StatsBase
using Distributions
using Random

"""
    bandwidth_sensitivity(problem::RDDProblem, estimator::SharpRDD;
                          bandwidths::AbstractVector{<:Real}=Float64[])
                          -> DataFrame

Test sensitivity of RDD estimates to bandwidth choice.

# Method
Estimates treatment effect at multiple bandwidths around the optimal bandwidth.
Plots should show stability - if estimates vary wildly, results are not robust.

# Arguments
- `problem`: RDD problem specification
- `estimator`: Sharp RDD estimator
- `bandwidths`: Vector of bandwidths to test (default: 0.5h, 0.75h, h, 1.25h, 1.5h, 2h)

# Returns
DataFrame with columns:
- `bandwidth`: Bandwidth value
- `estimate`: Treatment effect estimate
- `se`: Standard error
- `ci_lower`, `ci_upper`: 95% confidence interval
- `n_eff_left`, `n_eff_right`: Effective sample sizes
- `p_value`: Two-sided p-value

# Interpretation
- **Robust**: Estimates stable across bandwidths (variations within CI)
- **Fragile**: Estimates jump significantly - caution interpreting results
- **Optimal**: CCT/IK bandwidth usually performs best

# Examples
```julia
problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))
estimator = SharpRDD()

# Test sensitivity
results = bandwidth_sensitivity(problem, estimator)

# Visual check: plot(results.bandwidth, results.estimate, ribbon=1.96*results.se)
```

# References
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2019). *A Practical Introduction
  to Regression Discontinuity Designs: Foundations*. Cambridge University Press.
  (Section 4.4: Sensitivity to Bandwidth)
"""
function bandwidth_sensitivity(
    problem::RDDProblem{T},
    estimator::SharpRDD;
    bandwidths::AbstractVector{<:Real}=Float64[]
) where {T<:Real}

    # Get optimal bandwidth
    h_main, h_bias = _select_bandwidths(problem, estimator.bandwidth_method)

    # Default bandwidths: 0.5h, 0.75h, h, 1.25h, 1.5h, 2h
    if isempty(bandwidths)
        bandwidths = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0] .* h_main
    end

    # Estimate at each bandwidth
    results = []
    for h in bandwidths
        # Create custom estimator with fixed bandwidth
        est_custom = SharpRDD(
            bandwidth_method=estimator.bandwidth_method,
            kernel=estimator.kernel,
            run_density_test=false,  # Skip McCrary for speed
            polynomial_order=estimator.polynomial_order
        )

        # Estimate with custom bandwidth
        τ, se = _local_linear_rdd(
            problem.outcomes,
            problem.running_var,
            problem.cutoff,
            h,
            estimator.kernel,
            problem.covariates
        )

        # Compute CI
        z_crit = quantile(Normal(0, 1), 1 - problem.parameters.alpha/2)
        ci_lower = τ - z_crit * se
        ci_upper = τ + z_crit * se
        p_value = 2 * (1 - cdf(Normal(0, 1), abs(τ / se)))

        # Effective sample sizes
        n_left, n_right = _effective_sample_sizes(problem.running_var, problem.cutoff, h)

        push!(results, (
            bandwidth = h,
            estimate = τ,
            se = se,
            ci_lower = ci_lower,
            ci_upper = ci_upper,
            p_value = p_value,
            n_eff_left = n_left,
            n_eff_right = n_right
        ))
    end

    return DataFrame(results)
end

"""
    placebo_test(problem::RDDProblem, estimator::SharpRDD;
                 cutoffs::AbstractVector{<:Real}=Float64[],
                 n_placebos::Int=10) -> DataFrame

Test for discontinuities at placebo (fake) cutoffs.

# Method
If RDD is valid, there should be no discontinuity at cutoffs away from the
true treatment threshold. Finding significant effects at placebo cutoffs
suggests confounding or model misspecification.

# Arguments
- `problem`: RDD problem specification
- `estimator`: Sharp RDD estimator
- `cutoffs`: Specific placebo cutoffs to test (optional)
- `n_placebos`: Number of placebo cutoffs if not specified (default: 10)

# Returns
DataFrame with:
- `cutoff`: Placebo cutoff value
- `estimate`, `se`, `ci_lower`, `ci_upper`, `p_value`
- `significant`: Boolean (p < 0.05)

# Interpretation
- **Valid RDD**: No significant effects at placebo cutoffs
- **Problematic**: Multiple significant placebos suggest confounding
- **Rule of thumb**: Expect ~5% false positives (α=0.05)

# Examples
```julia
# Test at 10 random placebo cutoffs
results = placebo_test(problem, estimator)

# Count false positives
sum(results.significant)  # Should be ≈ 0-1 out of 10
```

# References
- Imbens & Lemieux (2008). "Regression discontinuity designs: A guide to practice."
  *Journal of Econometrics*, 142(2), 615-635. (Section 6.3)
"""
function placebo_test(
    problem::RDDProblem{T},
    estimator::SharpRDD;
    cutoffs::AbstractVector{<:Real}=Float64[],
    n_placebos::Int=10
) where {T<:Real}

    x = problem.running_var
    c_true = problem.cutoff

    # Generate placebo cutoffs if not specified
    if isempty(cutoffs)
        # Use percentiles excluding area near true cutoff
        x_sorted = sort(x)
        n = length(x)

        # Exclude middle 20% around true cutoff
        left_idx = findfirst(x_sorted .> quantile(x_sorted, 0.1))
        right_idx = findlast(x_sorted .< quantile(x_sorted, 0.9))

        # Sample placebo cutoffs
        n_left = div(n_placebos, 2)
        n_right = n_placebos - n_left

        cutoffs_left = quantile(x_sorted[1:left_idx], range(0.2, 0.8, length=n_left))
        cutoffs_right = quantile(x_sorted[right_idx:end], range(0.2, 0.8, length=n_right))

        cutoffs = vcat(cutoffs_left, cutoffs_right)
    end

    results = []
    for c_placebo in cutoffs
        # Skip if too close to true cutoff
        if abs(c_placebo - c_true) < 0.1 * std(x)
            continue
        end

        # Create placebo treatment
        treatment_placebo = x .>= c_placebo

        # Create placebo problem
        problem_placebo = RDDProblem(
            problem.outcomes,
            problem.running_var,
            treatment_placebo,
            c_placebo,
            problem.covariates,
            problem.parameters
        )

        # Estimate (suppress warnings)
        est_placebo = SharpRDD(
            bandwidth_method=estimator.bandwidth_method,
            kernel=estimator.kernel,
            run_density_test=false,  # Skip McCrary for placebos
            polynomial_order=estimator.polynomial_order
        )

        try
            solution = solve(problem_placebo, est_placebo)

            push!(results, (
                cutoff = c_placebo,
                estimate = solution.estimate,
                se = solution.se,
                ci_lower = solution.ci_lower,
                ci_upper = solution.ci_upper,
                p_value = solution.p_value,
                significant = solution.p_value < problem.parameters.alpha
            ))
        catch e
            # Skip if estimation fails (e.g., insufficient data)
            @warn "Placebo test failed at cutoff $c_placebo: $e"
            continue
        end
    end

    return DataFrame(results)
end

"""
    balance_test(problem::RDDProblem) -> DataFrame

Test for covariate balance at the cutoff.

# Method
In a valid RDD, covariates should be smooth through the cutoff (no discontinuity).
Finding discontinuities suggests non-random sorting or confounding.

# Arguments
- `problem`: RDD problem with covariates (nothing if no covariates)

# Returns
DataFrame with one row per covariate:
- `covariate`: Column index
- `estimate`: Discontinuity in covariate
- `se`, `ci_lower`, `ci_upper`, `p_value`
- `balanced`: Boolean (p > 0.05)

# Interpretation
- **Balanced**: No significant discontinuities (p > 0.05)
- **Imbalanced**: Significant discontinuities suggest confounding
- **Note**: Some imbalance expected by chance (5% false positive rate)

# Examples
```julia
problem_with_covs = RDDProblem(y, x, treatment, 0.0, covariates, (alpha=0.05,))
balance = balance_test(problem_with_covs)

# Check balance
all(balance.balanced)  # Should be true
```

# References
- Lee & Lemieux (2010). "Regression Discontinuity Designs in Economics."
  *Journal of Economic Literature*, 48(2), 281-355. (Section 5.1)
"""
function balance_test(problem::RDDProblem{T}) where {T<:Real}
    if isnothing(problem.covariates)
        throw(ArgumentError("No covariates in problem. Cannot test balance."))
    end

    X = problem.covariates
    x = problem.running_var
    treatment = problem.treatment
    c = problem.cutoff
    p = size(X, 2)

    results = []
    for j in 1:p
        # Create problem with j-th covariate as outcome
        problem_cov = RDDProblem(
            X[:, j],
            x,
            treatment,
            c,
            nothing,  # No covariates in balance test
            problem.parameters
        )

        # Estimate discontinuity in covariate
        estimator = SharpRDD(
            bandwidth_method=CCTBandwidth(),
            kernel=TriangularKernel(),
            run_density_test=false,
            polynomial_order=1
        )

        solution = solve(problem_cov, estimator)

        push!(results, (
            covariate = j,
            estimate = solution.estimate,
            se = solution.se,
            ci_lower = solution.ci_lower,
            ci_upper = solution.ci_upper,
            p_value = solution.p_value,
            balanced = solution.p_value > problem.parameters.alpha
        ))
    end

    return DataFrame(results)
end

"""
    donut_rdd(problem::RDDProblem, estimator::SharpRDD;
              hole_radius::Real=0.0) -> RDDSolution

Donut RDD: Exclude observations very close to the cutoff.

# Method
Excludes observations within `hole_radius` of cutoff to test robustness to:
- Manipulation precisely at cutoff
- Measurement error in running variable
- Heaping at round numbers

If estimates change substantially with donut, results may be fragile.

# Arguments
- `problem`: RDD problem
- `estimator`: Sharp RDD estimator
- `hole_radius`: Radius of donut hole (exclude |x - c| < r)

# Returns
RDDSolution with donut RDD estimate

# Interpretation
- **Robust**: Estimate similar to baseline (within CI)
- **Fragile**: Estimate changes significantly - investigate manipulation

# Examples
```julia
# Baseline
baseline = solve(problem, SharpRDD())

# Donut with 0.1 radius
donut = donut_rdd(problem, SharpRDD(), hole_radius=0.1)

# Compare
abs(baseline.estimate - donut.estimate) < 1.96 * sqrt(baseline.se^2 + donut.se^2)
```

# References
- Barreca, Guldi, Lindo, & Waddell (2011). "Saving Babies? Revisiting the effect
  of very low birth weight classification." *QJE*, 126(4), 2117-2123.
"""
function donut_rdd(
    problem::RDDProblem{T},
    estimator::SharpRDD;
    hole_radius::Real=0.0
) where {T<:Real}

    if hole_radius <= 0
        throw(ArgumentError("hole_radius must be positive (got $hole_radius)"))
    end

    x = problem.running_var
    c = problem.cutoff

    # Exclude observations in donut hole
    keep_idx = abs.(x .- c) .>= hole_radius

    if sum(keep_idx) < 40
        @warn "Donut RDD: Very few observations remaining ($(sum(keep_idx))). " *
              "Results may be unreliable."
    end

    # Create donut problem
    problem_donut = RDDProblem(
        problem.outcomes[keep_idx],
        problem.running_var[keep_idx],
        problem.treatment[keep_idx],
        problem.cutoff,
        isnothing(problem.covariates) ? nothing : problem.covariates[keep_idx, :],
        problem.parameters
    )

    # Estimate
    return solve(problem_donut, estimator)
end

"""
    permutation_test(problem::RDDProblem, estimator::SharpRDD;
                     n_permutations::Int=1000, rng::AbstractRNG=Random.GLOBAL_RNG)
                     -> (estimate::Real, p_value::Real, null_distribution::Vector{Real})

Randomization inference for RDD via permutation test.

# Method
1. Compute observed treatment effect
2. Randomly permute treatment assignment (respecting cutoff structure)
3. Compute treatment effect under permutation
4. Repeat n_permutations times
5. p-value = proportion of permuted effects ≥ |observed|

More robust to distributional assumptions than asymptotic inference.

# Arguments
- `problem`: RDD problem
- `estimator`: Sharp RDD estimator
- `n_permutations`: Number of permutations (default: 1000)
- `rng`: Random number generator

# Returns
- `estimate`: Observed treatment effect
- `p_value`: Two-sided permutation p-value
- `null_distribution`: Vector of permuted treatment effects

# Interpretation
- **Significant**: p_value < 0.05
- **Robust**: Permutation p-value similar to asymptotic p-value
- **Conservative**: Permutation test tends to be slightly conservative

# Examples
```julia
# Run permutation test
estimate, p_value, null_dist = permutation_test(problem, SharpRDD(), n_permutations=1000)

# Visual check
histogram(null_dist)
vline!([estimate], label="Observed")
```

# References
- Canay & Kamat (2018). "Approximate permutation tests and induced order statistics
  in the regression discontinuity design." *Review of Economic Studies*, 85(3), 1577-1608.
"""
function permutation_test(
    problem::RDDProblem{T},
    estimator::SharpRDD;
    n_permutations::Int=1000,
    rng::AbstractRNG=Random.GLOBAL_RNG
) where {T<:Real}

    # Observed estimate
    solution_obs = solve(problem, estimator)
    τ_obs = solution_obs.estimate

    # Permutation distribution
    τ_perm = zeros(n_permutations)

    x = problem.running_var
    c = problem.cutoff
    n = length(x)

    for i in 1:n_permutations
        # Permute treatment assignment
        # Note: This is a simplified permutation - maintains cutoff but randomizes within sides
        treatment_perm = copy(problem.treatment)

        # Permute within left side
        left_idx = findall(x .< c)
        if length(left_idx) > 1
            treatment_perm[left_idx] = shuffle(rng, treatment_perm[left_idx])
        end

        # Permute within right side
        right_idx = findall(x .>= c)
        if length(right_idx) > 1
            treatment_perm[right_idx] = shuffle(rng, treatment_perm[right_idx])
        end

        # Create permuted problem
        problem_perm = RDDProblem(
            problem.outcomes,
            problem.running_var,
            treatment_perm,
            problem.cutoff,
            problem.covariates,
            problem.parameters
        )

        # Estimate (suppress warnings and skip McCrary)
        est_perm = SharpRDD(
            bandwidth_method=estimator.bandwidth_method,
            kernel=estimator.kernel,
            run_density_test=false,
            polynomial_order=estimator.polynomial_order
        )

        try
            solution_perm = solve(problem_perm, est_perm)
            τ_perm[i] = solution_perm.estimate
        catch
            # If estimation fails, use 0 (conservative)
            τ_perm[i] = 0.0
        end
    end

    # Two-sided p-value
    p_value = mean(abs.(τ_perm) .>= abs(τ_obs))

    return (estimate=τ_obs, p_value=p_value, null_distribution=τ_perm)
end
