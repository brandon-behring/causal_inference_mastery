"""
Sharp Regression Kink Design (RKD) estimator.

Implements local polynomial regression to estimate the kink effect:
τ = Δslope(Y) / Δslope(D)

References:
- Card et al. (2015) - Generalized RKD inference
- Dong (2018) - RKD theory and practice
"""

"""
    solve(problem::RKDProblem, estimator::SharpRKD) -> RKDSolution

Estimate Sharp RKD treatment effect.

# Method
1. Select bandwidth (auto or user-specified)
2. Fit local polynomial regression on each side of cutoff
3. Estimate slope change (kink) in outcome and treatment
4. Compute τ = kink_Y / kink_D via Wald estimator
5. Standard error via delta method

# Arguments
- `problem`: RKDProblem with outcomes, running_var, treatment, cutoff
- `estimator`: SharpRKD estimator with options

# Returns
- RKDSolution with estimate, SE, CI, and diagnostics

# Example
```julia
# Generate RKD data
X = randn(1000)
D = [x < 0 ? 0.5*x : 1.5*x for x in X]  # kink = 1.0 at X=0
Y = 2.0 .* D .+ randn(1000)  # true effect = 2.0

problem = RKDProblem(Y, X, D, 0.0)
solution = solve(problem, SharpRKD())

println(solution.estimate)  # ≈ 2.0
```
"""
function solve(
    problem::RKDProblem{T,P},
    estimator::SharpRKD
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
        return RKDSolution(
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
            outcome_slope_left=T(NaN),
            outcome_slope_right=T(NaN),
            outcome_kink=T(NaN),
            treatment_slope_left=T(NaN),
            treatment_slope_right=T(NaN),
            treatment_kink=T(NaN),
            polynomial_order=polynomial_order,
            retcode=:Failure,
            message="Insufficient observations: left=$n_left, right=$n_right, need $min_obs each"
        )
    end

    # =========================================================================
    # Fit local polynomial on each side for OUTCOME
    # =========================================================================
    y_coef_left, y_vcov_left = fit_weighted_polynomial(
        y[left_mask], x[left_mask], cutoff, weights[left_mask], polynomial_order
    )
    y_coef_right, y_vcov_right = fit_weighted_polynomial(
        y[right_mask], x[right_mask], cutoff, weights[right_mask], polynomial_order
    )

    # Extract slopes (coefficient on linear term)
    y_slope_left = y_coef_left[2]
    y_slope_right = y_coef_right[2]
    y_kink = y_slope_right - y_slope_left

    var_y_kink = y_vcov_left[2, 2] + y_vcov_right[2, 2]

    # =========================================================================
    # Fit local polynomial on each side for TREATMENT
    # =========================================================================
    d_coef_left, d_vcov_left = fit_weighted_polynomial(
        d[left_mask], x[left_mask], cutoff, weights[left_mask], polynomial_order
    )
    d_coef_right, d_vcov_right = fit_weighted_polynomial(
        d[right_mask], x[right_mask], cutoff, weights[right_mask], polynomial_order
    )

    # Extract slopes
    d_slope_left = d_coef_left[2]
    d_slope_right = d_coef_right[2]
    d_kink = d_slope_right - d_slope_left

    var_d_kink = d_vcov_left[2, 2] + d_vcov_right[2, 2]

    # =========================================================================
    # Compute RKD estimate: τ = Δslope_Y / Δslope_D
    # =========================================================================
    if abs(d_kink) < 1e-10
        return RKDSolution(
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
            outcome_slope_left=T(y_slope_left),
            outcome_slope_right=T(y_slope_right),
            outcome_kink=T(y_kink),
            treatment_slope_left=T(d_slope_left),
            treatment_slope_right=T(d_slope_right),
            treatment_kink=T(d_kink),
            polynomial_order=polynomial_order,
            retcode=:Failure,
            message="No kink detected in treatment (Δslope_D ≈ 0)"
        )
    end

    estimate = y_kink / d_kink

    # Delta method for SE: Var(Y/D) ≈ (1/D²) * [Var(Y) + τ² * Var(D)]
    var_estimate = (1 / d_kink^2) * (var_y_kink + estimate^2 * var_d_kink)
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

    # Determine return code
    retcode = :Success
    message = "Estimation completed successfully"

    if n_left < 30 || n_right < 30
        retcode = :Warning
        message = "Small sample warning: left=$n_left, right=$n_right"
    end

    return RKDSolution(
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
        outcome_slope_left=T(y_slope_left),
        outcome_slope_right=T(y_slope_right),
        outcome_kink=T(y_kink),
        treatment_slope_left=T(d_slope_left),
        treatment_slope_right=T(d_slope_right),
        treatment_kink=T(d_kink),
        polynomial_order=polynomial_order,
        retcode=retcode,
        message=message
    )
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    fit_weighted_polynomial(
        y::AbstractVector,
        x::AbstractVector,
        cutoff::Real,
        weights::AbstractVector,
        order::Int
    ) -> (coefficients, variance_covariance)

Fit weighted local polynomial regression centered at cutoff.

Returns coefficient vector [intercept, slope, ...] and robust variance-covariance matrix.
"""
function fit_weighted_polynomial(
    y::AbstractVector{T},
    x::AbstractVector{T},
    cutoff::T,
    weights::AbstractVector{T},
    order::Int
) where {T<:Real}

    n = length(y)

    # Center x at cutoff
    x_centered = x .- cutoff

    # Build design matrix [1, x, x², ...]
    X_design = zeros(T, n, order + 1)
    for i in 1:n
        for p in 0:order
            X_design[i, p+1] = x_centered[i]^p
        end
    end

    # Weighted least squares
    W = Diagonal(weights)
    XtWX = X_design' * W * X_design
    XtWy = X_design' * W * y

    # Solve for coefficients
    coef = try
        XtWX \ XtWy
    catch
        pinv(XtWX) * XtWy
    end

    # Residuals
    residuals = y - X_design * coef

    # Degrees of freedom
    df = sum(weights .> 0) - (order + 1)
    df = max(df, 1)

    # Robust variance-covariance (sandwich estimator)
    XtWX_inv = try
        inv(XtWX)
    catch
        pinv(XtWX)
    end

    # Meat of sandwich: X'W * diag(residuals²) * W * X
    meat = X_design' * W * Diagonal(residuals.^2) * W * X_design

    # Sandwich: (X'WX)^{-1} * meat * (X'WX)^{-1}
    vcov = XtWX_inv * meat * XtWX_inv

    return coef, vcov
end
