#=
Dynamic Double Machine Learning for Time Series/Panel Treatment Effects.

Implements the Dynamic DML estimator from Lewis & Syrgkanis (2021).

Reference:
Lewis, G., & Syrgkanis, V. (2021). Double/Debiased Machine Learning for
Dynamic Treatment Effects via g-Estimation. arXiv:2002.07285.
=#

module DynamicDMLEstimator

using ..DynamicTypes
using ..CrossFitting
using ..GEstimation
using ..HACInference
using Random

export dynamic_dml, dynamic_dml_panel, simulate_dynamic_dgp

"""
    dynamic_dml(outcomes, treatments, states; kwargs...)

Estimate dynamic treatment effects using Double/Debiased Machine Learning.

# Arguments
- `outcomes`: Outcome variable Y, shape (T,)
- `treatments`: Treatment variable(s), shape (T,) or (T, n_treatments)
- `states`: Covariate/state variables X, shape (T, p)
- `max_lag`: Maximum treatment lag (default: 5)
- `n_folds`: Number of cross-fitting folds (default: 5)
- `cross_fitting`: Strategy - :blocked, :rolling, :panel, :progressive (default: :blocked)
- `nuisance_model`: :ridge, :rf, :gb (default: :ridge)
- `hac_kernel`: :bartlett or :qs (default: :bartlett)
- `hac_bandwidth`: HAC bandwidth (default: Newey-West optimal)
- `alpha`: Significance level (default: 0.05)
- `discount_factor`: For cumulative effect (default: 0.99)
- `unit_id`: Unit identifiers for panel data

# Returns
- `DynamicDMLResult`: Results with effects, SEs, CIs, and diagnostics
"""
function dynamic_dml(
    outcomes::AbstractVector{<:Real},
    treatments::AbstractVecOrMat{<:Real},
    states::AbstractMatrix{<:Real};
    max_lag::Int=5,
    n_folds::Int=5,
    cross_fitting::Symbol=:blocked,
    nuisance_model::Symbol=:ridge,
    hac_kernel::Symbol=:bartlett,
    hac_bandwidth::Union{Int,Nothing}=nothing,
    alpha::Float64=0.05,
    discount_factor::Float64=0.99,
    unit_id::Union{AbstractVector{<:Integer},Nothing}=nothing
)
    # Validate cross-fitting choice
    if cross_fitting == :panel && isnothing(unit_id)
        error("CRITICAL ERROR: unit_id required for panel cross-fitting.")
    end

    # Create data structure
    data = TimeSeriesPanelData(
        Float64.(outcomes),
        treatments,
        Float64.(states);
        unit_id=isnothing(unit_id) ? nothing : Int.(unit_id)
    )

    # Validate max_lag
    if max_lag < 0
        error("max_lag must be non-negative, got $max_lag")
    end
    if max_lag >= length(outcomes)
        error("max_lag ($max_lag) must be less than number of observations ($(length(outcomes)))")
    end

    # Get lagged data
    Y, T_lagged, X, valid_mask = get_lagged_data(data, max_lag)
    n_valid = length(Y)
    n_treatments = size(T_lagged, 3)

    # Check sufficient observations
    if n_valid < 10
        error("Insufficient observations after trimming. n=$n_valid, need at least 10.")
    end

    # Set up cross-validator
    if cross_fitting == :panel && !isnothing(unit_id)
        cv = get_cross_validator(:panel; n_folds=n_folds)
        valid_unit_id = Int.(unit_id)[valid_mask]
        splits = split_indices(cv, n_valid, valid_unit_id)
    else
        cv = get_cross_validator(cross_fitting; n_samples=n_valid, n_folds=n_folds)
        splits = split_indices(cv, n_valid)
    end

    # Storage for fold results
    fold_thetas = Matrix{Float64}[]
    fold_influences = Matrix{Float64}[]
    fold_n_test = Int[]
    all_nuisance_r2 = Dict{Symbol,Vector{Float64}}(
        :outcome_r2 => Float64[],
        :propensity_r2 => Float64[]
    )

    # Cross-fitting loop
    for (train_idx, test_idx) in splits
        # Create masks
        train_mask = falses(n_valid)
        test_mask = falses(n_valid)
        train_mask[train_idx] .= true
        test_mask[test_idx] .= true

        # Run sequential g-estimation
        theta_fold, influence_fold, nuisance_r2_fold = sequential_g_estimation(
            Y, T_lagged, X, train_mask, test_mask;
            nuisance_model=nuisance_model
        )

        push!(fold_thetas, theta_fold)
        push!(fold_influences, influence_fold)
        push!(fold_n_test, length(test_idx))

        # Accumulate nuisance R² values
        append!(all_nuisance_r2[:outcome_r2], nuisance_r2_fold[:outcome_r2])
        append!(all_nuisance_r2[:propensity_r2], nuisance_r2_fold[:propensity_r2])
    end

    # Aggregate across folds
    theta, influence = aggregate_fold_estimates(fold_thetas, fold_influences, fold_n_test)

    # Compute HAC standard errors
    if isnothing(hac_bandwidth)
        hac_bandwidth = optimal_bandwidth(n_valid)
    end

    # Standard errors for each lag
    theta_se = zeros(max_lag + 1)
    for h in 0:max_lag
        se_h = influence_function_se(influence[:, h + 1]; bandwidth=hac_bandwidth, kernel=hac_kernel)
        theta_se[h + 1] = se_h
    end

    # Confidence intervals
    theta_1d = theta[:, 1]
    ci_lower, ci_upper = confidence_interval(theta_1d, theta_se; alpha=alpha, method=:normal)

    # Cumulative effect
    cumulative, weights = compute_cumulative_effect(theta; discount_factor=discount_factor)
    cumulative_influence = compute_cumulative_influence(influence, weights)
    cumulative_se = influence_function_se(cumulative_influence; bandwidth=hac_bandwidth, kernel=hac_kernel)

    # Build result
    return DynamicDMLResult(
        theta_1d,
        theta_se,
        ci_lower,
        ci_upper,
        cumulative,
        cumulative_se,
        influence,
        all_nuisance_r2,
        "dynamic_dml",
        max_lag,
        n_folds,
        hac_bandwidth,
        String(hac_kernel),
        n_valid,
        alpha,
        discount_factor
    )
end

"""
    dynamic_dml_panel(outcomes, treatments, states, unit_id; kwargs...)

Convenience function for panel data dynamic DML.
Uses panel cross-fitting (split by unit).
"""
function dynamic_dml_panel(
    outcomes::AbstractVector{<:Real},
    treatments::AbstractVecOrMat{<:Real},
    states::AbstractMatrix{<:Real},
    unit_id::AbstractVector{<:Integer};
    max_lag::Int=5,
    n_folds::Int=5,
    nuisance_model::Symbol=:ridge,
    hac_kernel::Symbol=:bartlett,
    hac_bandwidth::Union{Int,Nothing}=nothing,
    alpha::Float64=0.05,
    discount_factor::Float64=0.99
)
    return dynamic_dml(
        outcomes, treatments, states;
        max_lag=max_lag,
        n_folds=n_folds,
        cross_fitting=:panel,
        nuisance_model=nuisance_model,
        hac_kernel=hac_kernel,
        hac_bandwidth=hac_bandwidth,
        alpha=alpha,
        discount_factor=discount_factor,
        unit_id=unit_id
    )
end

"""
    simulate_dynamic_dgp(; kwargs...)

Simulate data from a dynamic treatment effect DGP.

# Arguments
- `n_obs`: Number of observations (default: 500)
- `n_lags`: Number of lagged effects (default: 3)
- `true_effects`: True effects at each lag (default: [2.0, 1.0, 0.5])
- `n_covariates`: Number of covariates (default: 3)
- `treatment_prob`: Base treatment probability (default: 0.5)
- `confounding_strength`: Confounding strength (default: 0.3)
- `noise_scale`: Outcome noise scale (default: 1.0)
- `seed`: Random seed (default: nothing)

# Returns
- `Y`: Outcomes
- `D`: Treatments
- `X`: Covariates
- `true_effects`: True effects used
"""
function simulate_dynamic_dgp(;
    n_obs::Int=500,
    n_lags::Int=3,
    true_effects::Union{Vector{Float64},Nothing}=nothing,
    n_covariates::Int=3,
    treatment_prob::Float64=0.5,
    confounding_strength::Float64=0.3,
    noise_scale::Float64=1.0,
    seed::Union{Int,Nothing}=nothing
)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    # Default true effects
    if isnothing(true_effects)
        true_effects = [2.0, 1.0, 0.5][1:min(n_lags, 3)]
        if n_lags > 3
            append!(true_effects, zeros(n_lags - 3))
        end
    else
        n_lags = length(true_effects)
    end

    # Generate autocorrelated covariates
    X = zeros(n_obs, n_covariates)
    X[1, :] = randn(n_covariates)
    for t in 2:n_obs
        X[t, :] = 0.5 .* X[t-1, :] .+ sqrt(0.75) .* randn(n_covariates)
    end

    # Generate treatment (confounded by covariates)
    propensity = treatment_prob .+ confounding_strength .* X[:, 1]
    propensity = clamp.(propensity, 0.1, 0.9)
    D = Float64.(rand(n_obs) .< propensity)

    # Generate outcomes with dynamic treatment effects
    covariate_effects = [1.0, 0.5, 0.2]
    if n_covariates < 3
        covariate_effects = covariate_effects[1:n_covariates]
    elseif n_covariates > 3
        covariate_effects = vcat(covariate_effects, zeros(n_covariates - 3))
    end

    Y = zeros(n_obs)
    for t in 1:n_obs
        Y[t] = sum(X[t, :] .* covariate_effects)

        for (h, theta_h) in enumerate(true_effects)
            if t >= h
                Y[t] += theta_h * D[t - h + 1]
            end
        end

        Y[t] += noise_scale * randn()
    end

    return Y, D, X, true_effects
end

end  # module DynamicDMLEstimator
