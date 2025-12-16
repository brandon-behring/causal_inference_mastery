#=
Augmented Synthetic Control Method (ASCM)

Implements Ben-Michael, Feller, & Rothstein (2021) augmented estimator.

The ASCM adds a ridge regression bias correction:
    τ̂_ASCM = τ̂_SCM + (m̂₁ - ∑ᵢ wᵢm̂ᵢ)
where m̂ is the ridge regression prediction of post from pre.
=#

using LinearAlgebra
using Statistics
using Distributions

"""
    solve(problem::SCMProblem, estimator::AugmentedSC)

Estimate treatment effect using Augmented Synthetic Control Method.

Adds ridge regression bias correction for improved performance
when pre-treatment fit is imperfect.
"""
function solve(
    problem::SCMProblem{T,P},
    estimator::AugmentedSC,
) where {T<:Real,P}
    outcomes = problem.outcomes
    treatment = problem.treatment
    treatment_period = problem.treatment_period
    alpha = problem.parameters.alpha

    n_units, n_periods = size(outcomes)
    n_pre = treatment_period - 1
    n_post = n_periods - treatment_period + 1

    # Identify treated and control
    treated_mask = treatment
    control_mask = .!treatment
    n_treated = sum(treated_mask)
    n_control = sum(control_mask)

    # Extract outcomes
    treated_outcomes = outcomes[treated_mask, :]
    control_outcomes = outcomes[control_mask, :]

    # Split pre/post
    treated_pre = treated_outcomes[:, 1:n_pre]
    control_pre = control_outcomes[:, 1:n_pre]
    treated_post = treated_outcomes[:, treatment_period:end]
    control_post = control_outcomes[:, treatment_period:end]

    # Average treated
    treated_pre_avg = vec(mean(treated_pre, dims=1))
    treated_post_avg = vec(mean(treated_post, dims=1))
    treated_avg = vec(mean(treated_outcomes, dims=1))

    # Step 1: Compute SCM weights
    weights, converged = compute_scm_weights(treated_pre_avg, control_pre)

    # Step 2: Fit ridge regression outcome model
    lambda = estimator.lambda
    if lambda === nothing
        lambda = _select_lambda_cv(control_pre, control_post)
    end

    ridge_coef = _fit_ridge_outcome_model(control_pre, control_post, lambda)

    # Step 3: Compute predictions
    m_treated = _predict_ridge(treated_pre_avg, ridge_coef)  # (n_post,)
    m_control = hcat([_predict_ridge(control_pre[i, :], ridge_coef) for i in 1:n_control]...)'  # (n_control, n_post)
    m_synthetic = vec(m_control' * weights)  # (n_post,)

    # Augmentation term
    augmentation = m_treated .- m_synthetic

    # SCM synthetic post
    scm_synthetic_post = vec(control_post' * weights)

    # Augmented synthetic post
    augmented_post = scm_synthetic_post .+ augmentation

    # Full series
    scm_synthetic = vec(control_outcomes' * weights)
    augmented_series = vcat(scm_synthetic[1:n_pre], augmented_post)

    # Gap and estimate
    gap = treated_avg .- augmented_series
    estimate = mean(gap[treatment_period:end])

    # Pre-treatment fit (using SCM, not augmented)
    pre_rmse, pre_r_squared = compute_pre_treatment_fit(
        treated_pre_avg, control_pre, weights
    )

    # Inference
    if estimator.inference == :jackknife
        se = _jackknife_se(
            control_pre, control_post,
            treated_pre_avg, treated_post_avg,
            weights, lambda
        )
    elseif estimator.inference == :bootstrap
        se = _ascm_bootstrap_se(
            control_pre, control_post,
            treated_pre_avg, treated_post_avg,
            lambda, 200
        )
    else  # :none
        se = T(NaN)
    end

    # CI and p-value
    z = quantile(Normal(), 1 - alpha / 2)
    ci_lower = isnan(se) ? T(NaN) : estimate - z * se
    ci_upper = isnan(se) ? T(NaN) : estimate + z * se

    if !isnan(se) && se > 0
        z_stat = estimate / se
        p_value = 2 * (1 - cdf(Normal(), abs(z_stat)))
    else
        p_value = T(NaN)
    end

    retcode = converged ? :Success : :Warning

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
        synthetic_control=augmented_series,
        treated_series=treated_avg,
        gap=gap,
        retcode=retcode,
        original_problem=problem,
    )
end

# =============================================================================
# Ridge Regression Helpers
# =============================================================================

function _fit_ridge_outcome_model(
    X_pre::Matrix{T},
    Y_post::Matrix{T},
    lambda::Float64,
) where {T<:Real}
    n_control, n_pre = size(X_pre)
    n_post = size(Y_post, 2)

    # Add intercept
    X = hcat(ones(T, n_control), X_pre)  # (n_control, n_pre + 1)

    # Ridge: β = (X'X + λI)⁻¹ X'Y
    XtX = X' * X
    I_mat = Matrix{T}(I, n_pre + 1, n_pre + 1)
    I_mat[1, 1] = zero(T)  # Don't penalize intercept
    reg = XtX + lambda * I_mat

    coef = reg \ (X' * Y_post)  # (n_pre + 1, n_post)

    return coef
end

function _predict_ridge(x_pre::Vector{T}, coef::Matrix{T}) where {T<:Real}
    x = vcat(one(T), x_pre)  # Add intercept
    return vec(x' * coef)
end

function _select_lambda_cv(
    X_pre::Matrix{T},
    Y_post::Matrix{T};
    lambdas::Vector{Float64} = 10.0 .^ range(-2, 4, length=20),
    n_folds::Int = 5,
) where {T<:Real}
    n_control = size(X_pre, 1)

    if n_control < n_folds
        n_folds = n_control  # Leave-one-out
    end

    # Fold assignment
    fold_ids = mod.(0:(n_control-1), n_folds) .+ 1

    cv_errors = zeros(length(lambdas))

    for (i, lam) in enumerate(lambdas)
        fold_errors = Float64[]

        for fold in 1:n_folds
            train_mask = fold_ids .!= fold
            test_mask = fold_ids .== fold

            X_train = X_pre[train_mask, :]
            Y_train = Y_post[train_mask, :]
            X_test = X_pre[test_mask, :]
            Y_test = Y_post[test_mask, :]

            try
                coef = _fit_ridge_outcome_model(X_train, Y_train, lam)

                # Predict on test
                for j in 1:size(X_test, 1)
                    pred = _predict_ridge(X_test[j, :], coef)
                    err = mean((Y_test[j, :] .- pred) .^ 2)
                    push!(fold_errors, err)
                end
            catch
                push!(fold_errors, Inf)
            end
        end

        cv_errors[i] = mean(fold_errors)
    end

    best_idx = argmin(cv_errors)
    return lambdas[best_idx]
end

# =============================================================================
# Inference Helpers
# =============================================================================

function _jackknife_se(
    control_pre::Matrix{T},
    control_post::Matrix{T},
    treated_pre::Vector{T},
    treated_post::Vector{T},
    weights::Vector{T},
    lambda::Float64,
) where {T<:Real}
    n_control = size(control_pre, 1)

    jackknife_estimates = T[]

    for i in 1:n_control
        # Leave out unit i
        mask = trues(n_control)
        mask[i] = false

        loo_control_pre = control_pre[mask, :]
        loo_control_post = control_post[mask, :]
        loo_weights = weights[mask]
        loo_weights ./= sum(loo_weights)  # Renormalize

        try
            # Refit ridge
            coef = _fit_ridge_outcome_model(loo_control_pre, loo_control_post, lambda)

            # Predictions
            m_treated = _predict_ridge(treated_pre, coef)
            m_control = hcat([_predict_ridge(loo_control_pre[j, :], coef) for j in 1:(n_control-1)]...)'
            m_synthetic = vec(m_control' * loo_weights)

            # SCM part
            scm_synthetic_post = vec(loo_control_post' * loo_weights)

            # Augmented estimate
            augmented_post = scm_synthetic_post .+ (m_treated .- m_synthetic)
            est = mean(treated_post .- augmented_post)
            push!(jackknife_estimates, est)
        catch
            continue
        end
    end

    if length(jackknife_estimates) < 2
        return T(NaN)
    end

    # Jackknife SE formula
    n = length(jackknife_estimates)
    mean_est = mean(jackknife_estimates)
    se = sqrt((n - 1) / n * sum((jackknife_estimates .- mean_est) .^ 2))

    return se
end

function _ascm_bootstrap_se(
    control_pre::Matrix{T},
    control_post::Matrix{T},
    treated_pre::Vector{T},
    treated_post::Vector{T},
    lambda::Float64,
    n_bootstrap::Int,
) where {T<:Real}
    n_control = size(control_pre, 1)

    bootstrap_estimates = T[]

    for _ in 1:n_bootstrap
        # Resample control units
        idx = rand(1:n_control, n_control)
        boot_control_pre = control_pre[idx, :]
        boot_control_post = control_post[idx, :]

        try
            # Recompute weights
            weights, _ = compute_scm_weights(treated_pre, boot_control_pre)

            # Refit ridge
            coef = _fit_ridge_outcome_model(boot_control_pre, boot_control_post, lambda)

            # Predictions
            m_treated = _predict_ridge(treated_pre, coef)
            m_control = hcat([_predict_ridge(boot_control_pre[j, :], coef) for j in 1:n_control]...)'
            m_synthetic = vec(m_control' * weights)

            # SCM part
            scm_synthetic_post = vec(boot_control_post' * weights)

            # Augmented estimate
            augmented_post = scm_synthetic_post .+ (m_treated .- m_synthetic)
            est = mean(treated_post .- augmented_post)
            push!(bootstrap_estimates, est)
        catch
            continue
        end
    end

    if length(bootstrap_estimates) < 2
        return T(NaN)
    end

    return std(bootstrap_estimates, corrected=true)
end
