#=
Data Generating Processes (DGPs) for SCM Monte Carlo validation.

Provides DGPs for validating Synthetic Control Method estimators:
- SyntheticControl (Abadie et al. 2010)
- AugmentedSC (Ben-Michael et al. 2021)

All DGPs have known true ATT for validation.

References:
    Abadie, Diamond, Hainmueller (2010). "Synthetic Control Methods"
    Ben-Michael, Feller, Rothstein (2021). "Augmented Synthetic Control"
=#

using Random
using Statistics
using LinearAlgebra

# =============================================================================
# Data Container
# =============================================================================

"""
Container for SCM simulation data with known ground truth.

# Fields
- `outcomes::Matrix{T}`: Panel data (n_units × n_periods)
- `treatment::Vector{Bool}`: Treatment indicator
- `treatment_period::Int`: Period when treatment starts (1-indexed)
- `true_att::T`: Known ground truth ATT
- `n_units::Int`: Total units
- `n_control::Int`: Control units
- `n_pre_periods::Int`: Pre-treatment periods
- `n_post_periods::Int`: Post-treatment periods
- `dgp_type::String`: Description of DGP
- `expected_fit::String`: Expected fit quality
- `true_weights::Union{Nothing,Vector{T}}`: True weights (if known)
"""
struct SCMData{T<:Real}
    outcomes::Matrix{T}
    treatment::Vector{Bool}
    treatment_period::Int
    true_att::T
    n_units::Int
    n_control::Int
    n_pre_periods::Int
    n_post_periods::Int
    dgp_type::String
    expected_fit::String
    true_weights::Union{Nothing,Vector{T}}
end

# =============================================================================
# Perfect Match DGP
# =============================================================================

"""
    dgp_scm_perfect_match(; kwargs...) -> SCMData

SCM DGP with perfect pre-treatment match.

One control unit exactly matches the treated unit's counterfactual,
so weights should concentrate on it.

Expected:
    - Weight on control 1 ≈ 1.0
    - pre_rmse ≈ 0
    - Bias < 0.05
"""
function dgp_scm_perfect_match(;
    n_control::Int=10,
    n_pre::Int=10,
    n_post::Int=5,
    true_att::Float64=2.0,
    sigma::Float64=0.5,
    seed::Int=42,
)
    Random.seed!(seed)

    n_units = n_control + 1
    n_periods = n_pre + n_post
    treatment_period = n_pre + 1  # 1-indexed

    # Common trajectory for treated and control 1
    alpha_treated = 10.0
    beta_treated = 0.3
    t = collect(1:n_periods)

    # Initialize outcomes
    outcomes = zeros(Float64, n_units, n_periods)

    # Treated unit (row 1): trajectory + treatment effect post
    treated_trajectory = alpha_treated .+ beta_treated .* t
    treatment_effect = zeros(n_periods)
    treatment_effect[treatment_period:end] .= true_att
    outcomes[1, :] = treated_trajectory .+ treatment_effect .+ randn(n_periods) .* sigma

    # Control 1 (row 2): Same trajectory, no treatment
    outcomes[2, :] = treated_trajectory .+ randn(n_periods) .* sigma

    # Other controls (rows 3+): Different trajectories
    for i in 3:n_units
        alpha_i = alpha_treated + rand() * 6 - 3  # [-3, 3]
        beta_i = beta_treated + rand() * 0.4 - 0.2  # [-0.2, 0.2]
        outcomes[i, :] = alpha_i .+ beta_i .* t .+ randn(n_periods) .* sigma
    end

    # Treatment indicator: first unit treated
    treatment = Bool[i == 1 for i in 1:n_units]

    # True weights: all weight on control 1 (index 1 in control array)
    true_weights = zeros(n_control)
    true_weights[1] = 1.0

    return SCMData(
        outcomes,
        treatment,
        treatment_period,
        true_att,
        n_units,
        n_control,
        n_pre,
        n_post,
        "perfect_match",
        "perfect",
        true_weights,
    )
end

# =============================================================================
# Good Fit DGP
# =============================================================================

"""
    dgp_scm_good_fit(; kwargs...) -> SCMData

SCM DGP with good but imperfect pre-treatment match.

Treated unit's counterfactual is a weighted average of controls.

Expected:
    - pre_rmse < 0.5
    - pre_r_squared > 0.9
    - Bias < 0.10
    - Coverage 93-97%
"""
function dgp_scm_good_fit(;
    n_control::Int=20,
    n_pre::Int=10,
    n_post::Int=5,
    true_att::Float64=2.0,
    sigma::Float64=1.0,
    seed::Int=42,
)
    Random.seed!(seed)

    n_units = n_control + 1
    n_periods = n_pre + n_post
    treatment_period = n_pre + 1

    t = collect(1:n_periods)

    # Generate control outcomes
    control_outcomes = zeros(Float64, n_control, n_periods)
    for i in 1:n_control
        alpha_i = 10.0 + rand() * 4 - 2
        beta_i = 0.3 + rand() * 0.2 - 0.1
        gamma_i = 0.5 + rand() * 0.6 - 0.3
        control_outcomes[i, :] = (
            alpha_i .+ beta_i .* t .+
            gamma_i .* sin.(2π .* t ./ n_pre) .+
            randn(n_periods) .* sigma
        )
    end

    # Generate true weights (sparse Dirichlet)
    raw_weights = rand(n_control) .^ 2  # Sparse
    true_weights = raw_weights ./ sum(raw_weights)

    # Treated counterfactual = weighted average of controls
    treated_counterfactual = vec(control_outcomes' * true_weights)

    # Treated observed = counterfactual + treatment effect post
    treatment_effect = zeros(n_periods)
    treatment_effect[treatment_period:end] .= true_att
    treated_outcome = (
        treated_counterfactual .+ treatment_effect .+
        randn(n_periods) .* sigma .* 0.5
    )

    # Stack: treated first, then controls
    outcomes = vcat(treated_outcome', control_outcomes)

    treatment = Bool[i == 1 for i in 1:n_units]

    return SCMData(
        outcomes,
        treatment,
        treatment_period,
        true_att,
        n_units,
        n_control,
        n_pre,
        n_post,
        "good_fit",
        "good",
        true_weights,
    )
end

# =============================================================================
# Moderate Fit DGP
# =============================================================================

"""
    dgp_scm_moderate_fit(; kwargs...) -> SCMData

SCM DGP with moderate pre-treatment fit.

Treated unit's counterfactual is noisy combination of controls.

Expected:
    - pre_rmse: 0.3-1.0
    - Bias < 0.50
"""
function dgp_scm_moderate_fit(;
    n_control::Int=15,
    n_pre::Int=10,
    n_post::Int=5,
    true_att::Float64=2.0,
    sigma::Float64=1.0,
    seed::Int=42,
)
    Random.seed!(seed)

    n_units = n_control + 1
    n_periods = n_pre + n_post
    treatment_period = n_pre + 1

    t = collect(1:n_periods)

    # Controls spread across a range
    control_outcomes = zeros(Float64, n_control, n_periods)
    for i in 1:n_control
        alpha_i = 10.0 + rand() * 6 - 3
        beta_i = 0.2 + rand() * 0.2 - 0.1
        control_outcomes[i, :] = alpha_i .+ beta_i .* t .+ randn(n_periods) .* sigma
    end

    # Noisy combination of controls
    raw_weights = rand(n_control) .^ 2
    true_weights = raw_weights ./ sum(raw_weights)
    treated_counterfactual = vec(control_outcomes' * true_weights)

    # Add extra noise to create moderate mismatch
    treated_counterfactual .+= randn(n_periods) .* sigma .* 0.8

    # Add treatment effect
    treatment_effect = zeros(n_periods)
    treatment_effect[treatment_period:end] .= true_att
    treated_outcome = treated_counterfactual .+ treatment_effect

    outcomes = vcat(treated_outcome', control_outcomes)
    treatment = Bool[i == 1 for i in 1:n_units]

    return SCMData(
        outcomes,
        treatment,
        treatment_period,
        true_att,
        n_units,
        n_control,
        n_pre,
        n_post,
        "moderate_fit",
        "moderate",
        nothing,  # No exact weights due to noise
    )
end

# =============================================================================
# Poor Fit DGP (ASCM should outperform)
# =============================================================================

"""
    dgp_scm_poor_fit(; kwargs...) -> SCMData

SCM DGP with poor pre-treatment fit (treated outside convex hull).

ASCM should outperform basic SCM in this scenario.

Expected:
    - pre_rmse: 1.0-2.0
    - SCM bias: 0.2-0.5
    - ASCM bias: 0.1-0.25
"""
function dgp_scm_poor_fit(;
    n_control::Int=10,
    n_pre::Int=8,
    n_post::Int=5,
    true_att::Float64=2.0,
    sigma::Float64=1.0,
    seed::Int=42,
)
    Random.seed!(seed)

    n_units = n_control + 1
    n_periods = n_pre + n_post
    treatment_period = n_pre + 1

    t = collect(1:n_periods)

    # Controls clustered around base level
    base_alpha = 10.0
    base_beta = 0.2
    control_outcomes = zeros(Float64, n_control, n_periods)
    for i in 1:n_control
        alpha_i = base_alpha + rand() * 4 - 2
        beta_i = base_beta + rand() * 0.16 - 0.08
        control_outcomes[i, :] = alpha_i .+ beta_i .* t .+ randn(n_periods) .* sigma
    end

    # Treated: Higher intercept than all controls (outside convex hull)
    treated_alpha = base_alpha + 3.0
    treated_beta = base_beta + 0.05
    treated_counterfactual = (
        treated_alpha .+ treated_beta .* t .+
        randn(n_periods) .* sigma .* 0.5
    )

    # Add treatment effect
    treatment_effect = zeros(n_periods)
    treatment_effect[treatment_period:end] .= true_att
    treated_outcome = treated_counterfactual .+ treatment_effect

    outcomes = vcat(treated_outcome', control_outcomes)
    treatment = Bool[i == 1 for i in 1:n_units]

    return SCMData(
        outcomes,
        treatment,
        treatment_period,
        true_att,
        n_units,
        n_control,
        n_pre,
        n_post,
        "poor_fit",
        "poor",
        nothing,
    )
end

# =============================================================================
# Few Controls DGP
# =============================================================================

"""
    dgp_scm_few_controls(; kwargs...) -> SCMData

SCM DGP with few control units.

Realistic scenario: Many comparative case studies have limited donors.

Expected:
    - Placebo inference limited
    - SE may be less accurate
"""
function dgp_scm_few_controls(;
    n_control::Int=5,
    n_pre::Int=10,
    n_post::Int=5,
    true_att::Float64=2.0,
    sigma::Float64=0.8,
    seed::Int=42,
)
    Random.seed!(seed)

    n_units = n_control + 1
    n_periods = n_pre + n_post
    treatment_period = n_pre + 1

    t = collect(1:n_periods)

    # Controls: Spread trajectories
    control_outcomes = zeros(Float64, n_control, n_periods)
    for i in 1:n_control
        alpha_i = 10.0 + (i - 1) * 1.5
        beta_i = 0.2 + rand() * 0.1 - 0.05
        control_outcomes[i, :] = alpha_i .+ beta_i .* t .+ randn(n_periods) .* sigma
    end

    # Treated: Interpolates between controls
    raw_weights = rand(n_control)
    true_weights = raw_weights ./ sum(raw_weights)
    treated_counterfactual = vec(control_outcomes' * true_weights)

    treatment_effect = zeros(n_periods)
    treatment_effect[treatment_period:end] .= true_att
    treated_outcome = (
        treated_counterfactual .+ treatment_effect .+
        randn(n_periods) .* sigma .* 0.3
    )

    outcomes = vcat(treated_outcome', control_outcomes)
    treatment = Bool[i == 1 for i in 1:n_units]

    return SCMData(
        outcomes,
        treatment,
        treatment_period,
        true_att,
        n_units,
        n_control,
        n_pre,
        n_post,
        "few_controls",
        "good",
        true_weights,
    )
end

# =============================================================================
# Many Controls DGP
# =============================================================================

"""
    dgp_scm_many_controls(; kwargs...) -> SCMData

SCM DGP with many control units.

More donors improve placebo inference precision.
"""
function dgp_scm_many_controls(;
    n_control::Int=50,
    n_pre::Int=10,
    n_post::Int=5,
    true_att::Float64=2.0,
    sigma::Float64=1.0,
    seed::Int=42,
)
    Random.seed!(seed)

    n_units = n_control + 1
    n_periods = n_pre + n_post
    treatment_period = n_pre + 1

    t = collect(1:n_periods)

    # Controls: Diverse trajectories
    control_outcomes = zeros(Float64, n_control, n_periods)
    for i in 1:n_control
        alpha_i = 10.0 + rand() * 8 - 4
        beta_i = 0.3 + rand() * 0.3 - 0.15
        gamma_i = rand() - 0.5
        control_outcomes[i, :] = (
            alpha_i .+ beta_i .* t .+
            gamma_i .* sin.(2π .* t ./ n_pre) .+
            randn(n_periods) .* sigma
        )
    end

    # Very sparse weights
    raw_weights = rand(n_control) .^ 3
    true_weights = raw_weights ./ sum(raw_weights)
    treated_counterfactual = vec(control_outcomes' * true_weights)

    treatment_effect = zeros(n_periods)
    treatment_effect[treatment_period:end] .= true_att
    treated_outcome = (
        treated_counterfactual .+ treatment_effect .+
        randn(n_periods) .* sigma .* 0.3
    )

    outcomes = vcat(treated_outcome', control_outcomes)
    treatment = Bool[i == 1 for i in 1:n_units]

    return SCMData(
        outcomes,
        treatment,
        treatment_period,
        true_att,
        n_units,
        n_control,
        n_pre,
        n_post,
        "many_controls",
        "good",
        true_weights,
    )
end

# =============================================================================
# Short Pre-Period DGP
# =============================================================================

"""
    dgp_scm_short_pre_period(; kwargs...) -> SCMData

SCM DGP with short pre-treatment period.

Tests behavior with limited pre-treatment data.
This triggers warning: "Only N pre-treatment periods."
"""
function dgp_scm_short_pre_period(;
    n_control::Int=15,
    n_pre::Int=3,
    n_post::Int=5,
    true_att::Float64=2.0,
    sigma::Float64=1.0,
    seed::Int=42,
)
    Random.seed!(seed)

    n_units = n_control + 1
    n_periods = n_pre + n_post
    treatment_period = n_pre + 1

    t = collect(1:n_periods)

    # Controls
    control_outcomes = zeros(Float64, n_control, n_periods)
    for i in 1:n_control
        alpha_i = 10.0 + rand() * 4 - 2
        beta_i = 0.3 + rand() * 0.2 - 0.1
        control_outcomes[i, :] = alpha_i .+ beta_i .* t .+ randn(n_periods) .* sigma
    end

    # Treated: Combination of controls
    raw_weights = rand(n_control) .^ 1.5
    true_weights = raw_weights ./ sum(raw_weights)
    treated_counterfactual = vec(control_outcomes' * true_weights)

    treatment_effect = zeros(n_periods)
    treatment_effect[treatment_period:end] .= true_att
    treated_outcome = (
        treated_counterfactual .+ treatment_effect .+
        randn(n_periods) .* sigma .* 0.5
    )

    outcomes = vcat(treated_outcome', control_outcomes)
    treatment = Bool[i == 1 for i in 1:n_units]

    return SCMData(
        outcomes,
        treatment,
        treatment_period,
        true_att,
        n_units,
        n_control,
        n_pre,
        n_post,
        "short_pre_period",
        "moderate",
        true_weights,
    )
end

# =============================================================================
# Null Effect DGP
# =============================================================================

"""
    dgp_scm_null_effect(; kwargs...) -> SCMData

SCM DGP with zero treatment effect.

Used to test Type I error calibration.

Expected:
    - estimate ≈ 0
    - p_value > 0.05 in ~95% of simulations
"""
function dgp_scm_null_effect(;
    n_control::Int=15,
    n_pre::Int=10,
    n_post::Int=5,
    sigma::Float64=1.0,
    seed::Int=42,
)
    Random.seed!(seed)

    n_units = n_control + 1
    n_periods = n_pre + n_post
    treatment_period = n_pre + 1
    true_att = 0.0  # NULL EFFECT

    t = collect(1:n_periods)

    # Controls
    control_outcomes = zeros(Float64, n_control, n_periods)
    for i in 1:n_control
        alpha_i = 10.0 + rand() * 4 - 2
        beta_i = 0.3 + rand() * 0.2 - 0.1
        control_outcomes[i, :] = alpha_i .+ beta_i .* t .+ randn(n_periods) .* sigma
    end

    # Treated: NO TREATMENT EFFECT
    raw_weights = rand(n_control) .^ 2
    true_weights = raw_weights ./ sum(raw_weights)
    treated_counterfactual = vec(control_outcomes' * true_weights)
    treated_outcome = treated_counterfactual .+ randn(n_periods) .* sigma .* 0.5

    outcomes = vcat(treated_outcome', control_outcomes)
    treatment = Bool[i == 1 for i in 1:n_units]

    return SCMData(
        outcomes,
        treatment,
        treatment_period,
        true_att,
        n_units,
        n_control,
        n_pre,
        n_post,
        "null_effect",
        "good",
        true_weights,
    )
end
