"""
Data Generating Processes (DGPs) for DiD Monte Carlo validation.

Provides DGPs for validating Difference-in-Differences estimators:
- Classic 2×2 DiD
- Event study designs

All DGPs have known true effects for validation.

References:
    - Bertrand, Duflo, Mullainathan (2004). "How much should we trust differences-in-differences estimates?"
    - Goodman-Bacon (2021). "Difference-in-Differences with Variation in Treatment Timing"
"""

using Random
using Statistics

# =============================================================================
# Data Container Types
# =============================================================================

"""
Container for 2×2 DiD simulation data with known ground truth.
"""
struct DiDData{T<:Real}
    outcomes::Vector{T}
    treatment::Vector{Bool}
    post::Vector{Bool}
    unit_id::Vector{Int}
    time::Vector{Int}
    true_att::T
    n_treated::Int
    n_control::Int
    n_periods::Int
end

"""
Container for event study simulation data.
"""
struct EventStudyData{T<:Real}
    outcomes::Vector{T}
    treatment::Vector{Bool}
    time::Vector{Int}
    unit_id::Vector{Int}
    event_time::Vector{Int}
    true_pretrend_effects::Dict{Int, T}
    true_post_effects::Dict{Int, T}
    treatment_time::Int
    n_treated::Int
    n_control::Int
end

# =============================================================================
# Classic 2×2 DiD DGPs
# =============================================================================

"""
    dgp_did_2x2_simple(; kwargs...) -> DiDData

Simple 2×2 DiD DGP with known treatment effect.

DGP:
    Y_it = α_i + λ_t + τ·D_i·Post_t + ε_it

    where:
    - α_i ~ N(0, unit_fe_sigma²) : unit fixed effects
    - λ_t ~ N(0, time_fe_sigma²) : time fixed effects
    - τ = true_att : treatment effect on the treated
    - D_i = 1 for treated units, 0 for control
    - Post_t = 1 for post-treatment periods, 0 for pre
    - ε_it ~ N(0, σ²) : idiosyncratic errors

# Arguments
- `n_treated::Int=50`: Number of treated units
- `n_control::Int=50`: Number of control units
- `n_pre::Int=1`: Number of pre-treatment periods
- `n_post::Int=1`: Number of post-treatment periods
- `true_att::Float64=2.0`: True average treatment effect on treated
- `sigma::Float64=1.0`: Standard deviation of idiosyncratic errors
- `unit_fe_sigma::Float64=1.0`: Standard deviation of unit fixed effects
- `time_fe_sigma::Float64=0.5`: Standard deviation of time fixed effects
- `seed::Int=42`: Random seed
"""
function dgp_did_2x2_simple(;
    n_treated::Int=50,
    n_control::Int=50,
    n_pre::Int=1,
    n_post::Int=1,
    true_att::Float64=2.0,
    sigma::Float64=1.0,
    unit_fe_sigma::Float64=1.0,
    time_fe_sigma::Float64=0.5,
    seed::Int=42
)
    Random.seed!(seed)

    n_units = n_treated + n_control
    n_periods = n_pre + n_post

    # Unit and time indices
    unit_id = repeat(1:n_units, inner=n_periods)
    time = repeat(1:n_periods, outer=n_units)

    # Treatment indicator (unit-level, then expanded)
    treatment_unit = Bool[i <= n_treated for i in 1:n_units]
    treatment = repeat(treatment_unit, inner=n_periods)

    # Post indicator (time-level, then expanded)
    post_time = Bool[t > n_pre for t in 1:n_periods]
    post = repeat(post_time, outer=n_units)

    # Fixed effects
    unit_fe = randn(n_units) .* unit_fe_sigma
    time_fe = randn(n_periods) .* time_fe_sigma

    # Expand fixed effects to observation level
    unit_fe_obs = unit_fe[unit_id]
    time_fe_obs = time_fe[time]

    # Idiosyncratic errors
    epsilon = randn(n_units * n_periods) .* sigma

    # Outcome: Y = α_i + λ_t + τ·D·Post + ε
    outcomes = unit_fe_obs .+ time_fe_obs .+ true_att .* treatment .* post .+ epsilon

    return DiDData(
        outcomes,
        treatment,
        post,
        unit_id,
        time,
        true_att,
        n_treated,
        n_control,
        n_periods
    )
end

"""
    dgp_did_2x2_heteroskedastic(; kwargs...) -> DiDData

2×2 DiD DGP with heteroskedastic errors.

Different error variances for treated vs control groups.
Tests robustness of standard errors.

DGP:
    Y_it = α_i + λ_t + τ·D_i·Post_t + ε_it
    ε_it ~ N(0, σ_treated²) if D_i = 1
    ε_it ~ N(0, σ_control²) if D_i = 0
"""
function dgp_did_2x2_heteroskedastic(;
    n_treated::Int=50,
    n_control::Int=50,
    n_pre::Int=1,
    n_post::Int=1,
    true_att::Float64=2.0,
    sigma_treated::Float64=2.0,
    sigma_control::Float64=1.0,
    seed::Int=42
)
    Random.seed!(seed)

    n_units = n_treated + n_control
    n_periods = n_pre + n_post

    unit_id = repeat(1:n_units, inner=n_periods)
    time = repeat(1:n_periods, outer=n_units)

    treatment_unit = Bool[i <= n_treated for i in 1:n_units]
    treatment = repeat(treatment_unit, inner=n_periods)

    post_time = Bool[t > n_pre for t in 1:n_periods]
    post = repeat(post_time, outer=n_units)

    # Fixed effects
    unit_fe = randn(n_units)
    time_fe = randn(n_periods) .* 0.5
    unit_fe_obs = unit_fe[unit_id]
    time_fe_obs = time_fe[time]

    # Heteroskedastic errors
    sigma = ifelse.(Vector{Bool}(treatment), sigma_treated, sigma_control)
    epsilon = randn(length(sigma)) .* sigma

    # Outcome
    outcomes = unit_fe_obs .+ time_fe_obs .+ true_att .* treatment .* post .+ epsilon

    return DiDData(
        outcomes,
        treatment,
        post,
        unit_id,
        time,
        true_att,
        n_treated,
        n_control,
        n_periods
    )
end

"""
    dgp_did_2x2_serial_correlation(; kwargs...) -> DiDData

2×2 DiD DGP with AR(1) serial correlation within units.

Critical test for cluster-robust SE validity (Bertrand et al. 2004).

DGP:
    Y_it = α_i + λ_t + τ·D_i·Post_t + u_it
    u_it = ρ·u_{i,t-1} + ε_it
    ε_it ~ N(0, σ²)

Serial correlation within units requires cluster-robust SEs.
Naive SEs will be too small, leading to over-rejection.
"""
function dgp_did_2x2_serial_correlation(;
    n_treated::Int=50,
    n_control::Int=50,
    n_pre::Int=5,
    n_post::Int=5,
    true_att::Float64=2.0,
    rho::Float64=0.5,
    sigma::Float64=1.0,
    seed::Int=42
)
    Random.seed!(seed)

    n_units = n_treated + n_control
    n_periods = n_pre + n_post

    unit_id = repeat(1:n_units, inner=n_periods)
    time = repeat(1:n_periods, outer=n_units)

    treatment_unit = Bool[i <= n_treated for i in 1:n_units]
    treatment = repeat(treatment_unit, inner=n_periods)

    post_time = Bool[t > n_pre for t in 1:n_periods]
    post = repeat(post_time, outer=n_units)

    # Fixed effects
    unit_fe = randn(n_units)
    time_fe = randn(n_periods) .* 0.5

    # AR(1) errors within each unit
    errors = zeros(n_units * n_periods)
    for i in 1:n_units
        start_idx = (i - 1) * n_periods + 1
        end_idx = i * n_periods

        # Generate AR(1) process
        innovations = randn(n_periods) .* sigma
        u = zeros(n_periods)
        u[1] = innovations[1]
        for t in 2:n_periods
            u[t] = rho * u[t-1] + innovations[t]
        end
        errors[start_idx:end_idx] = u
    end

    # Outcome
    unit_fe_obs = unit_fe[unit_id]
    time_fe_obs = time_fe[time]
    outcomes = unit_fe_obs .+ time_fe_obs .+ true_att .* treatment .* post .+ errors

    return DiDData(
        outcomes,
        treatment,
        post,
        unit_id,
        time,
        true_att,
        n_treated,
        n_control,
        n_periods
    )
end

"""
    dgp_did_2x2_no_effect(; kwargs...) -> DiDData

2×2 DiD DGP with true null effect (τ = 0).

Tests that estimator does not spuriously detect effects.
"""
function dgp_did_2x2_no_effect(;
    n_treated::Int=50,
    n_control::Int=50,
    n_pre::Int=1,
    n_post::Int=1,
    sigma::Float64=1.0,
    seed::Int=42
)
    return dgp_did_2x2_simple(
        n_treated=n_treated,
        n_control=n_control,
        n_pre=n_pre,
        n_post=n_post,
        true_att=0.0,
        sigma=sigma,
        seed=seed
    )
end

# =============================================================================
# Event Study DGPs
# =============================================================================

"""
    dgp_event_study_null_pretrends(; kwargs...) -> EventStudyData

Event study DGP with true null pre-trends.

All pre-treatment effects are exactly zero (parallel trends hold).
Post-treatment effect is constant.

DGP:
    Y_it = α_i + λ_t + Σ_k β_k·D_i·1{t - g = k} + ε_it

    where:
    - β_k = 0 for k < 0 (true parallel trends)
    - β_k = true_effect for k >= 0 (constant post-treatment)
"""
function dgp_event_study_null_pretrends(;
    n_treated::Int=100,
    n_control::Int=100,
    n_pre::Int=5,
    n_post::Int=5,
    treatment_time::Int=6,  # 1-indexed: periods 1-5 are pre, 6-10 are post
    true_effect::Float64=2.0,
    sigma::Float64=1.0,
    seed::Int=42
)
    Random.seed!(seed)

    n_units = n_treated + n_control
    n_periods = n_pre + n_post

    unit_id = repeat(1:n_units, inner=n_periods)
    time = repeat(1:n_periods, outer=n_units)

    # Treatment indicator (unit-level)
    treatment_unit = Bool[i <= n_treated for i in 1:n_units]
    treatment = repeat(treatment_unit, inner=n_periods)

    # Event time: k = t - treatment_time
    event_time = time .- treatment_time

    # Fixed effects
    unit_fe = randn(n_units)
    time_fe = randn(n_periods) .* 0.5
    unit_fe_obs = unit_fe[unit_id]
    time_fe_obs = time_fe[time]

    # Treatment effect: 0 for k < 0, true_effect for k >= 0
    effect = ifelse.(Vector{Bool}(treatment) .& (event_time .>= 0), true_effect, 0.0)

    # Errors
    epsilon = randn(n_units * n_periods) .* sigma

    # Outcome
    outcomes = unit_fe_obs .+ time_fe_obs .+ effect .+ epsilon

    # Event time range
    min_event = -n_pre
    max_event = n_post - 1

    # True effects
    true_pretrend_effects = Dict(k => 0.0 for k in min_event:-1)
    true_post_effects = Dict(k => true_effect for k in 0:max_event)

    return EventStudyData(
        outcomes,
        treatment,
        time,
        unit_id,
        event_time,
        true_pretrend_effects,
        true_post_effects,
        treatment_time,
        n_treated,
        n_control
    )
end

"""
    dgp_event_study_violated_pretrends(; kwargs...) -> EventStudyData

Event study DGP with violated pre-trends (anticipation effects).

Pre-treatment effects increase linearly toward treatment.
Tests detection of parallel trends violations.

DGP:
    β_k = pretrend_slope × k for k < 0 (linear pre-trend)
    β_k = true_effect for k >= 0
"""
function dgp_event_study_violated_pretrends(;
    n_treated::Int=100,
    n_control::Int=100,
    n_pre::Int=5,
    n_post::Int=5,
    treatment_time::Int=6,
    true_effect::Float64=2.0,
    pretrend_slope::Float64=0.3,
    sigma::Float64=1.0,
    seed::Int=42
)
    Random.seed!(seed)

    n_units = n_treated + n_control
    n_periods = n_pre + n_post

    unit_id = repeat(1:n_units, inner=n_periods)
    time = repeat(1:n_periods, outer=n_units)

    treatment_unit = Bool[i <= n_treated for i in 1:n_units]
    treatment = repeat(treatment_unit, inner=n_periods)

    event_time = time .- treatment_time

    unit_fe = randn(n_units)
    time_fe = randn(n_periods) .* 0.5
    unit_fe_obs = unit_fe[unit_id]
    time_fe_obs = time_fe[time]

    # Treatment effect with pre-trends
    effect = zeros(length(unit_id))

    # Pre-treatment: linear trend
    pre_mask = Vector{Bool}(treatment) .& (event_time .< 0)
    effect[pre_mask] = pretrend_slope .* event_time[pre_mask]

    # Post-treatment: constant effect
    post_mask = Vector{Bool}(treatment) .& (event_time .>= 0)
    effect[post_mask] .= true_effect

    epsilon = randn(n_units * n_periods) .* sigma
    outcomes = unit_fe_obs .+ time_fe_obs .+ effect .+ epsilon

    min_event = -n_pre
    max_event = n_post - 1

    true_pretrend_effects = Dict(k => pretrend_slope * k for k in min_event:-1)
    true_post_effects = Dict(k => true_effect for k in 0:max_event)

    return EventStudyData(
        outcomes,
        treatment,
        time,
        unit_id,
        event_time,
        true_pretrend_effects,
        true_post_effects,
        treatment_time,
        n_treated,
        n_control
    )
end

"""
    dgp_event_study_dynamic(; kwargs...) -> EventStudyData

Event study DGP with dynamic (time-varying) treatment effects.

Effects grow over time since treatment.
Tests event study estimation of time-varying effects.

DGP:
    β_k = effect_base + effect_growth × k for k >= 0
"""
function dgp_event_study_dynamic(;
    n_treated::Int=100,
    n_control::Int=100,
    n_pre::Int=5,
    n_post::Int=5,
    treatment_time::Int=6,
    effect_base::Float64=1.0,
    effect_growth::Float64=0.5,
    sigma::Float64=1.0,
    seed::Int=42
)
    Random.seed!(seed)

    n_units = n_treated + n_control
    n_periods = n_pre + n_post

    unit_id = repeat(1:n_units, inner=n_periods)
    time = repeat(1:n_periods, outer=n_units)

    treatment_unit = Bool[i <= n_treated for i in 1:n_units]
    treatment = repeat(treatment_unit, inner=n_periods)

    event_time = time .- treatment_time

    unit_fe = randn(n_units)
    time_fe = randn(n_periods) .* 0.5
    unit_fe_obs = unit_fe[unit_id]
    time_fe_obs = time_fe[time]

    # Dynamic treatment effects
    effect = zeros(length(unit_id))
    post_mask = Vector{Bool}(treatment) .& (event_time .>= 0)
    effect[post_mask] = effect_base .+ effect_growth .* event_time[post_mask]

    epsilon = randn(n_units * n_periods) .* sigma
    outcomes = unit_fe_obs .+ time_fe_obs .+ effect .+ epsilon

    min_event = -n_pre
    max_event = n_post - 1

    true_pretrend_effects = Dict(k => 0.0 for k in min_event:-1)
    true_post_effects = Dict(k => effect_base + effect_growth * k for k in 0:max_event)

    return EventStudyData(
        outcomes,
        treatment,
        time,
        unit_id,
        event_time,
        true_pretrend_effects,
        true_post_effects,
        treatment_time,
        n_treated,
        n_control
    )
end
