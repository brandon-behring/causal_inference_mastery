#=
Synthetic Control Methods - Inference

Implements:
- In-space placebo tests
- Block bootstrap SE
=#

using Statistics
using Random

"""
    _placebo_inference(control_outcomes, treatment_period, observed_effect, n_placebo)

Compute SE and p-value using in-space placebo tests.

For each control unit, pretend it was treated and compute the
"placebo effect" using remaining controls as donors.
"""
function _placebo_inference(
    control_outcomes::Matrix{T},
    treatment_period::Int,
    observed_effect::T,
    n_placebo::Int,
) where {T<:Real}
    n_control, n_periods = size(control_outcomes)
    n_pre = treatment_period - 1

    # Limit placebo iterations to available controls
    n_placebo = min(n_placebo, n_control)

    placebo_effects = T[]

    for i in 1:n_placebo
        # Treat control unit i as "treated"
        pseudo_treated = control_outcomes[i, :]
        pseudo_control_mask = trues(n_control)
        pseudo_control_mask[i] = false
        pseudo_control = control_outcomes[pseudo_control_mask, :]

        if sum(pseudo_control_mask) < 2
            continue  # Need at least 2 donors
        end

        # Pre-treatment data
        pseudo_treated_pre = pseudo_treated[1:n_pre]
        pseudo_control_pre = pseudo_control[:, 1:n_pre]

        try
            # Compute weights
            weights, _ = compute_scm_weights(pseudo_treated_pre, pseudo_control_pre)

            # Compute pseudo effect
            pseudo_synthetic = vec(pseudo_control' * weights)
            pseudo_gap = pseudo_treated .- pseudo_synthetic
            pseudo_effect = mean(pseudo_gap[treatment_period:end])
            push!(placebo_effects, pseudo_effect)
        catch
            # Skip if optimization fails
            continue
        end
    end

    if length(placebo_effects) < 2
        return T(NaN), T(NaN)
    end

    # SE from placebo distribution
    se = std(placebo_effects, corrected=true)

    # Two-sided p-value
    n_extreme = sum(abs.(placebo_effects) .>= abs(observed_effect))
    p_value = (n_extreme + 1) / (length(placebo_effects) + 1)

    return se, T(p_value)
end

"""
    _bootstrap_inference(treated_pre, control_pre, treated_post, control_post,
                         observed_effect, n_bootstrap)

Compute SE and p-value using block bootstrap.

Resamples time periods in the pre-treatment period and recomputes weights.
"""
function _bootstrap_inference(
    treated_pre::Vector{T},
    control_pre::Matrix{T},
    treated_post::Vector{T},
    control_post::Matrix{T},
    observed_effect::T,
    n_bootstrap::Int,
) where {T<:Real}
    n_pre = length(treated_pre)
    n_control = size(control_pre, 1)
    n_post = length(treated_post)

    bootstrap_effects = T[]

    for _ in 1:n_bootstrap
        # Resample pre-treatment periods
        pre_idx = rand(1:n_pre, n_pre)
        boot_treated_pre = treated_pre[pre_idx]
        boot_control_pre = control_pre[:, pre_idx]

        try
            # Compute weights on resampled data
            weights, _ = compute_scm_weights(boot_treated_pre, boot_control_pre)

            # Apply weights to original post-treatment (no resampling)
            boot_synthetic_post = vec(control_post' * weights)
            boot_gap_post = treated_post .- boot_synthetic_post
            boot_effect = mean(boot_gap_post)
            push!(bootstrap_effects, boot_effect)
        catch
            continue
        end
    end

    if length(bootstrap_effects) < 2
        return T(NaN), T(NaN)
    end

    se = std(bootstrap_effects, corrected=true)

    # Two-sided p-value (centered at observed)
    n_extreme = sum(abs.(bootstrap_effects .- observed_effect) .>= abs(observed_effect))
    p_value = (n_extreme + 1) / (length(bootstrap_effects) + 1)

    return se, T(p_value)
end
