"""
Bunching estimator implementation.

Implements the Saez (2010) bunching estimator with bootstrap inference.

References:
- Saez (2010) - Original bunching methodology
- Chetty et al. (2011) - Integration constraint
- Kleven (2016) - Bunching review
"""

using Statistics
using Random

"""
    solve(problem::BunchingProblem, estimator::SaezBunching) -> BunchingSolution

Estimate bunching using Saez (2010) methodology.

# Arguments
- `problem::BunchingProblem`: Bunching problem specification
- `estimator::SaezBunching`: Estimator configuration

# Returns
- `BunchingSolution`: Full solution with estimates and inference

# Example
```julia
data = randn(1000) .* 15 .+ 50
problem = BunchingProblem(data, 50.0, 5.0; t1_rate=0.25, t2_rate=0.35)
estimator = SaezBunching(n_bins=50, polynomial_order=7, n_bootstrap=200)
solution = solve(problem, estimator)
```
"""
function solve(
    problem::BunchingProblem{T},
    estimator::SaezBunching,
) where {T<:Real}

    data = problem.data
    kink_point = problem.kink_point
    n_obs = length(data)

    # Estimate counterfactual
    cf_result = estimate_counterfactual(
        problem,
        estimator.n_bins;
        polynomial_order=estimator.polynomial_order,
    )

    # Compute point estimates
    excess_mass, excess_count, h0 = compute_excess_mass(cf_result)

    # Compute elasticity if rates provided
    elasticity = if !isnothing(problem.t1_rate) && !isnothing(problem.t2_rate)
        compute_elasticity(excess_mass, problem.t1_rate, problem.t2_rate)
    else
        T(NaN)
    end

    # Bootstrap inference
    excess_mass_bootstrap = Vector{T}(undef, estimator.n_bootstrap)
    elasticity_bootstrap = Vector{T}(undef, estimator.n_bootstrap)

    for b in 1:estimator.n_bootstrap
        # Resample data with replacement
        boot_indices = rand(1:n_obs, n_obs)
        boot_data = data[boot_indices]

        # Create bootstrap problem
        boot_problem = BunchingProblem(
            boot_data,
            kink_point,
            problem.bunching_width;
            t1_rate=problem.t1_rate,
            t2_rate=problem.t2_rate,
        )

        # Estimate counterfactual on bootstrap sample
        try
            boot_cf = estimate_counterfactual(
                boot_problem,
                estimator.n_bins;
                polynomial_order=estimator.polynomial_order,
            )

            boot_excess, _, _ = compute_excess_mass(boot_cf)
            excess_mass_bootstrap[b] = boot_excess

            if !isnothing(problem.t1_rate) && !isnothing(problem.t2_rate)
                elasticity_bootstrap[b] = compute_elasticity(
                    boot_excess, problem.t1_rate, problem.t2_rate
                )
            else
                elasticity_bootstrap[b] = T(NaN)
            end
        catch
            # Bootstrap iteration failed - use NaN
            excess_mass_bootstrap[b] = T(NaN)
            elasticity_bootstrap[b] = T(NaN)
        end
    end

    # Compute standard errors from bootstrap (excluding NaN)
    valid_excess = filter(isfinite, excess_mass_bootstrap)
    excess_mass_se = if length(valid_excess) >= 10
        std(valid_excess)
    else
        T(NaN)
    end

    valid_elasticity = filter(isfinite, elasticity_bootstrap)
    elasticity_se = if length(valid_elasticity) >= 10
        std(valid_elasticity)
    else
        T(NaN)
    end

    # Check convergence
    convergence = length(valid_excess) >= estimator.n_bootstrap * 0.8
    message = if convergence
        "Estimation converged successfully"
    else
        "Warning: $(estimator.n_bootstrap - length(valid_excess)) bootstrap iterations failed"
    end

    return BunchingSolution(
        excess_mass,
        excess_mass_se,
        excess_count,
        elasticity,
        elasticity_se,
        kink_point,
        cf_result.bunching_region,
        cf_result,
        problem.t1_rate,
        problem.t2_rate,
        n_obs,
        estimator.n_bootstrap,
        convergence,
        message,
    )
end

"""
    bunching_confidence_interval(solution::BunchingSolution; level::Float64=0.95)

Compute confidence interval for excess mass.

# Arguments
- `solution::BunchingSolution`: Solution from solve()
- `level::Float64=0.95`: Confidence level

# Returns
- `(lower, upper)`: Confidence interval bounds
"""
function bunching_confidence_interval(
    solution::BunchingSolution{T};
    level::Float64=0.95,
) where {T<:Real}
    (level <= 0 || level >= 1) && throw(ArgumentError("level must be in (0, 1)"))

    z = quantile_normal((1 + level) / 2)
    lower = solution.excess_mass - z * solution.excess_mass_se
    upper = solution.excess_mass + z * solution.excess_mass_se

    return (lower, upper)
end

"""
    elasticity_confidence_interval(solution::BunchingSolution; level::Float64=0.95)

Compute confidence interval for elasticity.

# Arguments
- `solution::BunchingSolution`: Solution from solve()
- `level::Float64=0.95`: Confidence level

# Returns
- `(lower, upper)`: Confidence interval bounds (NaN if elasticity not computed)
"""
function elasticity_confidence_interval(
    solution::BunchingSolution{T};
    level::Float64=0.95,
) where {T<:Real}
    (level <= 0 || level >= 1) && throw(ArgumentError("level must be in (0, 1)"))

    if !isfinite(solution.elasticity)
        return (T(NaN), T(NaN))
    end

    z = quantile_normal((1 + level) / 2)
    lower = solution.elasticity - z * solution.elasticity_se
    upper = solution.elasticity + z * solution.elasticity_se

    return (lower, upper)
end

# Helper: Standard normal quantile (approximation)
function quantile_normal(p::Float64)
    # Rational approximation for standard normal quantile
    # Accurate to ~1e-4 for p in (0.01, 0.99)
    if p <= 0.5
        t = sqrt(-2 * log(p))
    else
        t = sqrt(-2 * log(1 - p))
    end

    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    z = t - (c0 + c1 * t + c2 * t^2) / (1 + d1 * t + d2 * t^2 + d3 * t^3)

    return p > 0.5 ? z : -z
end
