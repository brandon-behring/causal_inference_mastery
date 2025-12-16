#=
Synthetic Control Methods - Main Estimator

Implements solve(::SCMProblem, ::SyntheticControl) following SciML pattern.
=#

using LinearAlgebra
using Statistics
using Distributions

"""
    solve(problem::SCMProblem, estimator::SyntheticControl)

Estimate treatment effect using Synthetic Control Method.

# Example
```julia
outcomes = randn(10, 20)  # 10 units, 20 periods
treatment = [true; falses(9)]
problem = SCMProblem(outcomes, treatment, 10, (alpha=0.05,))
solution = solve(problem, SyntheticControl())
```
"""
function solve(
    problem::SCMProblem{T,P},
    estimator::SyntheticControl,
) where {T<:Real,P}
    outcomes = problem.outcomes
    treatment = problem.treatment
    treatment_period = problem.treatment_period
    covariates = problem.covariates
    alpha = problem.parameters.alpha

    n_units, n_periods = size(outcomes)
    n_pre = treatment_period - 1
    n_post = n_periods - treatment_period + 1

    # Identify treated and control units
    treated_mask = treatment
    control_mask = .!treatment
    n_treated = sum(treated_mask)
    n_control = sum(control_mask)

    # Extract outcomes
    treated_outcomes = outcomes[treated_mask, :]  # (n_treated, n_periods)
    control_outcomes = outcomes[control_mask, :]  # (n_control, n_periods)

    # Split into pre/post
    # Julia is 1-indexed, so pre = 1:(treatment_period-1), post = treatment_period:end
    treated_pre = treated_outcomes[:, 1:n_pre]  # (n_treated, n_pre)
    control_pre = control_outcomes[:, 1:n_pre]  # (n_control, n_pre)

    # Average treated if multiple (standard approach)
    treated_pre_avg = vec(mean(treated_pre, dims=1))  # (n_pre,)
    treated_avg = vec(mean(treated_outcomes, dims=1))  # (n_periods,)

    # Handle covariates
    cov_treated = nothing
    cov_control = nothing
    if covariates !== nothing
        cov_treated = vec(mean(covariates[treated_mask, :], dims=1))
        cov_control = covariates[control_mask, :]
    end

    # Compute optimal weights
    weights, converged = compute_scm_weights(
        treated_pre_avg,
        control_pre;
        covariates_treated=cov_treated,
        covariates_control=cov_control,
        covariate_weight=estimator.covariate_weight,
    )

    # Compute synthetic control series
    synthetic_series = vec(control_outcomes' * weights)  # (n_periods,)

    # Compute gap (treatment effect by period)
    gap = treated_avg .- synthetic_series

    # Pre-treatment fit
    pre_rmse, pre_r_squared = compute_pre_treatment_fit(
        treated_pre_avg, control_pre, weights
    )

    # Post-treatment effect (ATT)
    post_gap = gap[treatment_period:end]
    estimate = mean(post_gap)

    # Inference
    if estimator.inference == :placebo
        se, p_value = _placebo_inference(
            control_outcomes,
            treatment_period,
            estimate,
            estimator.n_placebo,
        )
    elseif estimator.inference == :bootstrap
        se, p_value = _bootstrap_inference(
            treated_pre_avg,
            control_pre,
            treated_avg[treatment_period:end],
            control_outcomes[:, treatment_period:end],
            estimate,
            estimator.n_placebo,
        )
    else  # :none
        se = T(NaN)
        p_value = T(NaN)
    end

    # Confidence interval
    z = quantile(Normal(), 1 - alpha / 2)
    ci_lower = isnan(se) ? T(NaN) : estimate - z * se
    ci_upper = isnan(se) ? T(NaN) : estimate + z * se

    # Determine retcode
    retcode = :Success
    if !converged
        retcode = :Warning
        @warn "Weight optimization did not converge"
    end
    if pre_r_squared < 0.85
        retcode = :Warning
        @warn "Poor pre-treatment fit: R² = $(round(pre_r_squared, digits=3))"
    end

    return SCMSolution(
        estimate=estimate,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        weights=weights,
        pre_rmse=pre_rmse,
        pre_r_squared=pre_r_squared,
        n_treated=n_treated,
        n_control=n_control,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        synthetic_control=synthetic_series,
        treated_series=treated_avg,
        gap=gap,
        retcode=retcode,
        original_problem=problem,
    )
end
