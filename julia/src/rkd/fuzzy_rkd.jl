"""
Fuzzy Regression Kink Design (RKD) estimator.

Implements 2SLS-style estimation for fuzzy kinks where treatment is stochastic
but E[D|X] has a kink at the cutoff.

Key Insight:
- Sharp RKD: D = f(X) with known kink (deterministic)
- Fuzzy RKD: E[D|X] has a kink at cutoff (stochastic)

References:
- Card et al. (2015) - Generalized RKD inference
- Dong (2018) - RKD theory and practice
"""

"""
    solve(problem::RKDProblem, estimator::FuzzyRKD) -> FuzzyRKDSolution

Estimate Fuzzy RKD treatment effect using 2SLS.

# Method
1. First Stage: Estimate kink in E[D|X] at cutoff
2. Reduced Form: Estimate kink in E[Y|X] at cutoff
3. 2SLS Estimate: τ = reduced_form_kink / first_stage_kink
4. Standard error via delta method

# Arguments
- `problem`: RKDProblem with outcomes, running_var, treatment, cutoff
- `estimator`: FuzzyRKD estimator with options

# Returns
- FuzzyRKDSolution with LATE estimate, first stage diagnostics, etc.

# Identification
Fuzzy RKD requires:
1. A kink in E[D|X] at the cutoff (first stage)
2. Smooth behavior of all other determinants at cutoff
3. No manipulation of running variable around cutoff
4. Monotonicity: the kink affects D in the same direction for all units

The estimator identifies a LATE - the effect for "compliers" whose treatment
changes due to the kink.

# Example
```julia
# Generate fuzzy RKD data
X = randn(1000)
D_expected = [x < 0 ? 0.5*x : 1.5*x for x in X]
D = D_expected .+ 0.5*randn(1000)  # Add noise for fuzzy
Y = 2.0 .* D .+ randn(1000)

problem = RKDProblem(Y, X, D, 0.0)
solution = solve(problem, FuzzyRKD())

println("LATE: \$(solution.estimate)")
println("First stage F: \$(solution.first_stage_f_stat)")
```
"""
function solve(
    problem::RKDProblem{T,P},
    estimator::FuzzyRKD
) where {T<:Real,P<:NamedTuple}

    y = problem.outcomes
    x = problem.running_var
    d = problem.treatment
    cutoff = problem.cutoff
    alpha = estimator.alpha
    polynomial_order = estimator.polynomial_order

    n = length(y)

    # Select bandwidth
    if isnothing(estimator.bandwidth)
        bandwidth = select_rkd_bandwidth(problem, :ik)
    else
        bandwidth = estimator.bandwidth
    end

    # Get kernel function
    kernel = get_rkd_kernel(estimator.kernel)

    # Compute kernel weights
    u = (x .- cutoff) ./ bandwidth
    weights = [rkd_kernel_function(kernel, ui) for ui in u]

    # Split by cutoff
    left_mask = (x .< cutoff) .& (weights .> 0)
    right_mask = (x .>= cutoff) .& (weights .> 0)

    n_left = sum(left_mask)
    n_right = sum(right_mask)

    # Check minimum observations
    min_obs = polynomial_order + 2
    if n_left < min_obs || n_right < min_obs
        return FuzzyRKDSolution(
            estimate=T(NaN),
            se=T(NaN),
            ci_lower=T(NaN),
            ci_upper=T(NaN),
            t_stat=T(NaN),
            p_value=T(NaN),
            bandwidth=T(bandwidth),
            kernel=estimator.kernel,
            n_eff_left=n_left,
            n_eff_right=n_right,
            first_stage_slope_left=T(NaN),
            first_stage_slope_right=T(NaN),
            first_stage_kink=T(NaN),
            reduced_form_slope_left=T(NaN),
            reduced_form_slope_right=T(NaN),
            reduced_form_kink=T(NaN),
            first_stage_f_stat=T(NaN),
            weak_first_stage=true,
            polynomial_order=polynomial_order,
            retcode=:Failure,
            message="Insufficient observations: left=$n_left, right=$n_right, need $min_obs each"
        )
    end

    # =========================================================================
    # First Stage: Estimate kink in E[D|X]
    # =========================================================================
    d_coef_left, d_vcov_left = fit_weighted_polynomial(
        d[left_mask], x[left_mask], cutoff, weights[left_mask], polynomial_order
    )
    d_coef_right, d_vcov_right = fit_weighted_polynomial(
        d[right_mask], x[right_mask], cutoff, weights[right_mask], polynomial_order
    )

    # Extract first stage slopes
    fs_slope_left = d_coef_left[2]
    fs_slope_right = d_coef_right[2]
    fs_kink = fs_slope_right - fs_slope_left

    # First stage variance
    var_fs_kink = d_vcov_left[2, 2] + d_vcov_right[2, 2]

    # First stage F-statistic
    if var_fs_kink > 0
        fs_f_stat = fs_kink^2 / var_fs_kink
    else
        fs_f_stat = isfinite(fs_kink) && fs_kink != 0 ? T(Inf) : T(0)
    end

    weak_first_stage = fs_f_stat < 10

    # Check for no kink in first stage
    if abs(fs_kink) < 1e-10
        return FuzzyRKDSolution(
            estimate=T(NaN),
            se=T(NaN),
            ci_lower=T(NaN),
            ci_upper=T(NaN),
            t_stat=T(NaN),
            p_value=T(NaN),
            bandwidth=T(bandwidth),
            kernel=estimator.kernel,
            n_eff_left=n_left,
            n_eff_right=n_right,
            first_stage_slope_left=T(fs_slope_left),
            first_stage_slope_right=T(fs_slope_right),
            first_stage_kink=T(fs_kink),
            reduced_form_slope_left=T(NaN),
            reduced_form_slope_right=T(NaN),
            reduced_form_kink=T(NaN),
            first_stage_f_stat=T(fs_f_stat),
            weak_first_stage=true,
            polynomial_order=polynomial_order,
            retcode=:Failure,
            message="No kink detected in first stage (Δslope_D ≈ 0)"
        )
    end

    # =========================================================================
    # Reduced Form: Estimate kink in E[Y|X]
    # =========================================================================
    y_coef_left, y_vcov_left = fit_weighted_polynomial(
        y[left_mask], x[left_mask], cutoff, weights[left_mask], polynomial_order
    )
    y_coef_right, y_vcov_right = fit_weighted_polynomial(
        y[right_mask], x[right_mask], cutoff, weights[right_mask], polynomial_order
    )

    # Extract reduced form slopes
    rf_slope_left = y_coef_left[2]
    rf_slope_right = y_coef_right[2]
    rf_kink = rf_slope_right - rf_slope_left

    # Reduced form variance
    var_rf_kink = y_vcov_left[2, 2] + y_vcov_right[2, 2]

    # =========================================================================
    # 2SLS Estimate: τ = Reduced Form / First Stage
    # =========================================================================
    estimate = rf_kink / fs_kink

    # Standard error via delta method
    # Var(τ) = (1/fs_kink²) * [Var(rf_kink) + τ² * Var(fs_kink)]
    var_estimate = (1 / fs_kink^2) * (var_rf_kink + estimate^2 * var_fs_kink)
    se = sqrt(max(var_estimate, 0))

    # Inference
    if se > 0 && isfinite(se)
        t_stat = estimate / se
        df = n_left + n_right - 2 * (polynomial_order + 1)
        df = max(df, 1)
        p_value = 2 * (1 - cdf(TDist(df), abs(t_stat)))
        t_crit = quantile(TDist(df), 1 - alpha / 2)
        ci_lower = estimate - t_crit * se
        ci_upper = estimate + t_crit * se
    else
        t_stat = isfinite(estimate) && estimate != 0 ? Inf * sign(estimate) : T(0)
        p_value = estimate != 0 ? T(0) : T(1)
        ci_lower = estimate
        ci_upper = estimate
    end

    # Determine return code and message
    retcode = :Success
    message = "Estimation completed successfully"

    if weak_first_stage
        retcode = :Warning
        message = "Weak first stage (F=$(round(fs_f_stat, digits=2)) < 10). LATE may be biased."
    elseif n_left < 30 || n_right < 30
        retcode = :Warning
        message = "Small sample warning: left=$n_left, right=$n_right"
    end

    return FuzzyRKDSolution(
        estimate=T(estimate),
        se=T(se),
        ci_lower=T(ci_lower),
        ci_upper=T(ci_upper),
        t_stat=T(t_stat),
        p_value=T(p_value),
        bandwidth=T(bandwidth),
        kernel=estimator.kernel,
        n_eff_left=n_left,
        n_eff_right=n_right,
        first_stage_slope_left=T(fs_slope_left),
        first_stage_slope_right=T(fs_slope_right),
        first_stage_kink=T(fs_kink),
        reduced_form_slope_left=T(rf_slope_left),
        reduced_form_slope_right=T(rf_slope_right),
        reduced_form_kink=T(rf_kink),
        first_stage_f_stat=T(fs_f_stat),
        weak_first_stage=weak_first_stage,
        polynomial_order=polynomial_order,
        retcode=retcode,
        message=message
    )
end
