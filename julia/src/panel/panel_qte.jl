"""
    Panel Quantile Treatment Effects via RIF Regression

Implements RIF-OLS approach (Firpo, Fortin, Lemieux 2009) adapted for panel data
with Mundlak projection and clustered standard errors.

# Key Features
- RIF (Recentered Influence Function) transformation for unconditional QTE
- Mundlak projection: includes time-means X̄ᵢ as covariates
- Clustered standard errors at unit level
- Supports balanced and unbalanced panels

# References
- Firpo, S., Fortin, N., & Lemieux, T. (2009). "Unconditional Quantile Regressions."
- Mundlak, Y. (1978). "On the pooling of time series and cross section data."
"""

using Statistics
using LinearAlgebra
using Distributions

# Include types if not already included
if !@isdefined(PanelData)
    include("types.jl")
end

# =============================================================================
# Result Types
# =============================================================================

"""
    PanelQTEResult

Result from Panel RIF-QTE estimation.

# Fields
- `qte::Float64`: Quantile treatment effect at quantile τ.
- `qte_se::Float64`: Standard error (clustered by unit).
- `ci_lower::Float64`: Lower bound of confidence interval.
- `ci_upper::Float64`: Upper bound of confidence interval.
- `quantile::Float64`: Quantile τ ∈ (0, 1) at which effect is estimated.
- `n_obs::Int`: Total number of observations.
- `n_units::Int`: Number of unique units.
- `outcome_quantile::Float64`: Estimated pooled quantile q̂_τ.
- `density_at_quantile::Float64`: Kernel density estimate f̂_Y(q̂_τ).
- `bandwidth::Float64`: Kernel bandwidth used.
- `method::Symbol`: Estimation method.
"""
struct PanelQTEResult
    qte::Float64
    qte_se::Float64
    ci_lower::Float64
    ci_upper::Float64
    quantile::Float64
    n_obs::Int
    n_units::Int
    outcome_quantile::Float64
    density_at_quantile::Float64
    bandwidth::Float64
    method::Symbol
end

"""
    PanelQTEBandResult

Result from Panel RIF-QTE estimation at multiple quantiles.

# Fields
- `quantiles::Vector{Float64}`: Array of quantiles τ.
- `qtes::Vector{Float64}`: QTE estimates at each quantile.
- `qte_ses::Vector{Float64}`: Standard errors at each quantile.
- `ci_lowers::Vector{Float64}`: Lower CI bounds.
- `ci_uppers::Vector{Float64}`: Upper CI bounds.
- `n_obs::Int`: Total number of observations.
- `n_units::Int`: Number of unique units.
- `method::Symbol`: Estimation method.
"""
struct PanelQTEBandResult
    quantiles::Vector{Float64}
    qtes::Vector{Float64}
    qte_ses::Vector{Float64}
    ci_lowers::Vector{Float64}
    ci_uppers::Vector{Float64}
    n_obs::Int
    n_units::Int
    method::Symbol
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    silverman_bandwidth(y)

Compute Silverman's rule-of-thumb bandwidth for kernel density.

h = 1.06 × σ̂_Y × n^(-0.2)
"""
function silverman_bandwidth(y::Vector{T}) where T
    n = length(y)
    sigma = std(y; corrected=true)

    # Use IQR-based estimate if data is heavy-tailed
    iqr = quantile(y, 0.75) - quantile(y, 0.25)
    sigma_robust = min(sigma, iqr / 1.34)  # 1.34 ≈ 2 * Φ^{-1}(0.75)

    # Silverman's rule
    h = 1.06 * sigma_robust * n^(-0.2)

    return max(h, 1e-10)  # Avoid zero bandwidth
end

"""
    kernel_density_at_quantile(y, q_tau, bandwidth)

Estimate kernel density at a specific point using Gaussian kernel.
"""
function kernel_density_at_quantile(y::Vector{T}, q_tau::Real, bandwidth::Real) where T
    n = length(y)
    h = bandwidth

    # Normalized distances
    u = (y .- q_tau) ./ h

    # Gaussian kernel
    kernel_vals = exp.(-0.5 .* u.^2) ./ sqrt(2π)

    # Density estimate
    f_hat = mean(kernel_vals) / h

    return max(f_hat, 1e-10)  # Avoid division by zero
end

"""
    compute_rif(y, tau, q_tau, f_q)

Compute Recentered Influence Function for a quantile.

RIF(Y; q_τ) = q_τ + (τ - I(Y ≤ q_τ)) / f_Y(q_τ)
"""
function compute_rif(y::Vector{T}, tau::Real, q_tau::Real, f_q::Real) where T
    indicator = Float64.(y .<= q_tau)
    rif = q_tau .+ (tau .- indicator) ./ f_q
    return rif
end

"""
    panel_clustered_se_qte(residuals, D, unit_id)

Compute clustered standard error for QTE coefficient.

Clusters at the unit level to account for within-unit correlation.
"""
function panel_clustered_se_qte(
    residuals::Vector{T},
    D::Vector{T},
    unit_id::Vector{Int}
) where T
    n = length(residuals)
    unique_units = unique(unit_id)
    n_units_total = length(unique_units)

    # Influence function contribution for D coefficient
    psi = residuals .* D

    # Aggregate within clusters
    cluster_sums = zeros(n_units_total)
    for (i, unit) in enumerate(unique_units)
        mask = unit_id .== unit
        cluster_sums[i] = sum(psi[mask])
    end

    # Clustered variance with small-sample correction
    var_clustered = sum(cluster_sums .^ 2) / (n^2)

    # Correction for number of clusters
    correction = n_units_total > 1 ? n_units_total / (n_units_total - 1) : 1.0
    var_clustered *= correction

    return sqrt(var_clustered)
end

# =============================================================================
# Main Functions
# =============================================================================

"""
    panel_rif_qte(panel; tau=0.5, alpha=0.05, include_covariates=true)

Estimate panel quantile treatment effect via RIF regression.

Uses the Recentered Influence Function (Firpo et al. 2009) with
Mundlak projection to control for unobserved unit effects.

# Arguments
- `panel::PanelData`: Panel data with outcomes, treatment, covariates, unit_id, time.
- `tau::Float64`: The quantile τ ∈ (0, 1) at which to estimate the effect.
- `alpha::Float64`: Significance level for confidence interval.
- `include_covariates::Bool`: Include covariates and their unit means (Mundlak).

# Returns
- `PanelQTEResult`: Result containing QTE estimate, SE, CI, and diagnostics.

# Example
```julia
using Random
Random.seed!(42)

n_units, n_periods = 50, 10
n_obs_total = n_units * n_periods
unit_id = repeat(1:n_units, inner=n_periods)
time = repeat(1:n_periods, outer=n_units)
X = randn(n_obs_total, 2)
D = Float64.(rand(n_obs_total) .< 0.5)
Y = X[:, 1] .+ 2.0 .* D .+ randn(n_obs_total)

panel = PanelData(Y, D, X, unit_id, time)
result = panel_rif_qte(panel; tau=0.5)
println("Median QTE: \$(round(result.qte, digits=3)) ± \$(round(result.qte_se, digits=3))")
```

# Notes
The RIF-OLS approach transforms outcomes using:
    RIF(Y; q_τ) = q_τ + (τ - I(Y ≤ q_τ)) / f_Y(q_τ)

Then regresses RIF on [1, X, X̄, D] where X̄ are unit means (Mundlak).
The coefficient on D estimates the unconditional QTE.
"""
function panel_rif_qte(
    panel::PanelData{T};
    tau::Float64=0.5,
    alpha::Float64=0.05,
    include_covariates::Bool=true
) where T
    # ========================================================================
    # INPUT VALIDATION
    # ========================================================================

    (0 < tau < 1) || error(
        "CRITICAL ERROR: Invalid quantile.\n" *
        "Function: panel_rif_qte\n" *
        "Expected: tau in (0, 1), Got: $tau"
    )

    Y = panel.outcomes
    D = panel.treatment
    X = panel.covariates
    unit_id = panel.unit_id

    n_obs_total = n_obs(panel)
    n_units_total = n_units(panel)

    # Warn for extreme quantiles
    if tau < 0.05 || tau > 0.95
        n_tail = Int(round(n_obs_total * min(tau, 1 - tau)))
        if n_tail < 20
            @warn "Extreme quantile τ=$tau with only ~$n_tail observations in the tail."
        end
    end

    # ========================================================================
    # MUNDLAK PROJECTION
    # ========================================================================

    if include_covariates
        X_bar = compute_unit_means(panel)  # n_obs × p
        X_augmented = hcat(X, X_bar)  # n_obs × 2p
    else
        X_augmented = nothing
    end

    # ========================================================================
    # COMPUTE RIF
    # ========================================================================

    # Pooled quantile
    q_tau = Statistics.quantile(Y, tau)

    # Kernel density at quantile
    h = silverman_bandwidth(Y)
    f_q = kernel_density_at_quantile(Y, q_tau, h)

    # RIF transformation
    rif = compute_rif(Y, tau, q_tau, f_q)

    # Warn if density is very sparse
    if f_q < 0.01
        @warn "Sparse density at quantile (f̂(q_$tau) = $(round(f_q, digits=4))). " *
              "RIF estimates may be unstable."
    end

    # ========================================================================
    # OLS REGRESSION: RIF ~ [1, X, X̄, D]
    # ========================================================================

    # Build design matrix
    if include_covariates
        Z = hcat(ones(n_obs_total), X_augmented, D)
    else
        Z = hcat(ones(n_obs_total), D)
    end

    # OLS via normal equations
    ZtZ = Z' * Z
    ZtRIF = Z' * rif

    # Try to solve; fallback to pseudoinverse if singular
    beta = try
        ZtZ \ ZtRIF
    catch
        pinv(ZtZ) * ZtRIF
    end

    # QTE is the coefficient on D (last column)
    qte = beta[end]

    # ========================================================================
    # CLUSTERED STANDARD ERROR
    # ========================================================================

    # OLS residuals
    residuals = rif .- Z * beta

    # Clustered SE at unit level
    se = panel_clustered_se_qte(residuals, D, unit_id)

    # Confidence interval
    z_crit = Distributions.quantile(Normal(), 1 - alpha / 2)
    ci_lower = qte - z_crit * se
    ci_upper = qte + z_crit * se

    # ========================================================================
    # RETURN RESULT
    # ========================================================================

    return PanelQTEResult(
        qte,
        se,
        ci_lower,
        ci_upper,
        tau,
        n_obs_total,
        n_units_total,
        q_tau,
        f_q,
        h,
        :panel_rif_qte
    )
end

"""
    panel_rif_qte_band(panel; quantiles=nothing, alpha=0.05, include_covariates=true)

Estimate panel QTE across multiple quantiles.

# Arguments
- `panel::PanelData`: Panel data structure.
- `quantiles::Vector{Float64}`: Quantiles to estimate. Default: [0.1, 0.25, 0.5, 0.75, 0.9].
- `alpha::Float64`: Significance level for confidence intervals.
- `include_covariates::Bool`: Include covariates and their unit means.

# Returns
- `PanelQTEBandResult`: Arrays of estimates across quantiles.

# Example
```julia
result = panel_rif_qte_band(panel; quantiles=[0.1, 0.5, 0.9])
for (q, qte, se) in zip(result.quantiles, result.qtes, result.qte_ses)
    println("τ=\$q: QTE=\$(round(qte, digits=3)) ± \$(round(se, digits=3))")
end
```
"""
function panel_rif_qte_band(
    panel::PanelData{T};
    quantiles::Union{Nothing, Vector{Float64}}=nothing,
    alpha::Float64=0.05,
    include_covariates::Bool=true
) where T
    # Default quantiles
    if quantiles === nothing
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    end

    n_quantiles = length(quantiles)

    # Validate
    all(q -> 0 < q < 1, quantiles) || error(
        "CRITICAL ERROR: Quantiles must be in (0, 1).\n" *
        "Function: panel_rif_qte_band\n" *
        "Got: $quantiles"
    )

    # Estimate at each quantile
    qtes = zeros(n_quantiles)
    qte_ses = zeros(n_quantiles)
    ci_lowers = zeros(n_quantiles)
    ci_uppers = zeros(n_quantiles)

    for (i, q) in enumerate(quantiles)
        result = panel_rif_qte(panel; tau=q, alpha=alpha, include_covariates=include_covariates)
        qtes[i] = result.qte
        qte_ses[i] = result.qte_se
        ci_lowers[i] = result.ci_lower
        ci_uppers[i] = result.ci_upper
    end

    return PanelQTEBandResult(
        quantiles,
        qtes,
        qte_ses,
        ci_lowers,
        ci_uppers,
        n_obs(panel),
        n_units(panel),
        :panel_rif_qte_band
    )
end

"""
    panel_unconditional_qte(panel; tau=0.5, n_bootstrap=1000, alpha=0.05,
                            cluster_bootstrap=true, random_state=nothing)

Estimate simple unconditional QTE for panel data.

Computes the difference in quantiles between treated and control groups:
    QTE(τ) = Q_τ(Y | D=1) - Q_τ(Y | D=0)

This is a baseline estimator that doesn't control for covariates
but uses cluster bootstrap for valid inference.

# Arguments
- `panel::PanelData`: Panel data structure.
- `tau::Float64`: The quantile τ ∈ (0, 1).
- `n_bootstrap::Int`: Number of bootstrap replications.
- `alpha::Float64`: Significance level.
- `cluster_bootstrap::Bool`: Resample clusters rather than observations.
- `random_state::Union{Nothing, Int}`: Random seed.

# Returns
- `PanelQTEResult`: Result containing QTE estimate, SE, CI.

# Notes
This estimator does NOT control for confounding via covariates or unit effects.
For panel data with confounding, use `panel_rif_qte` which includes
Mundlak projection and covariate adjustment.
"""
function panel_unconditional_qte(
    panel::PanelData{T};
    tau::Float64=0.5,
    n_bootstrap::Int=1000,
    alpha::Float64=0.05,
    cluster_bootstrap::Bool=true,
    random_state::Union{Nothing, Int}=nothing
) where T
    if random_state !== nothing
        Random.seed!(random_state)
    end

    Y = panel.outcomes
    D = panel.treatment
    unit_id = panel.unit_id

    n_obs_total = n_obs(panel)
    n_units_total = n_units(panel)

    # Validate
    (0 < tau < 1) || error(
        "CRITICAL ERROR: Invalid quantile.\n" *
        "Function: panel_unconditional_qte\n" *
        "Expected: tau in (0, 1), Got: $tau"
    )

    # ========================================================================
    # POINT ESTIMATE
    # ========================================================================

    y_treated = Y[D .== 1]
    y_control = Y[D .== 0]

    (length(y_treated) >= 2 && length(y_control) >= 2) || error(
        "CRITICAL ERROR: Insufficient observations.\n" *
        "Function: panel_unconditional_qte\n" *
        "Treated: $(length(y_treated)), Control: $(length(y_control))"
    )

    q_treated = Statistics.quantile(y_treated, tau)
    q_control = Statistics.quantile(y_control, tau)
    qte = q_treated - q_control

    # Use q_tau as the pooled quantile for consistency
    q_tau = Statistics.quantile(Y, tau)
    h = silverman_bandwidth(Y)
    f_q = kernel_density_at_quantile(Y, q_tau, h)

    # ========================================================================
    # CLUSTER BOOTSTRAP FOR SE
    # ========================================================================

    bootstrap_estimates = zeros(n_bootstrap)
    unique_units = unique(unit_id)

    for b in 1:n_bootstrap
        if cluster_bootstrap
            # Resample clusters
            boot_units = rand(unique_units, length(unique_units))

            # Gather observations from resampled clusters
            boot_idx = Int[]
            for unit in boot_units
                unit_idx = findall(==(unit), unit_id)
                append!(boot_idx, unit_idx)
            end
        else
            # Simple observation-level bootstrap
            boot_idx = rand(1:n_obs_total, n_obs_total)
        end

        Y_boot = Y[boot_idx]
        D_boot = D[boot_idx]

        y_t_boot = Y_boot[D_boot .== 1]
        y_c_boot = Y_boot[D_boot .== 0]

        if length(y_t_boot) > 0 && length(y_c_boot) > 0
            q_t = Statistics.quantile(y_t_boot, tau)
            q_c = Statistics.quantile(y_c_boot, tau)
            bootstrap_estimates[b] = q_t - q_c
        else
            bootstrap_estimates[b] = NaN
        end
    end

    # Remove NaN bootstrap samples
    valid_bootstrap = filter(!isnan, bootstrap_estimates)

    length(valid_bootstrap) >= 10 || error(
        "CRITICAL ERROR: Too few valid bootstrap samples.\n" *
        "Function: panel_unconditional_qte\n" *
        "Valid: $(length(valid_bootstrap))"
    )

    se = std(valid_bootstrap; corrected=true)
    ci_lower = Statistics.quantile(valid_bootstrap, alpha / 2)
    ci_upper = Statistics.quantile(valid_bootstrap, 1 - alpha / 2)

    return PanelQTEResult(
        qte,
        se,
        ci_lower,
        ci_upper,
        tau,
        n_obs_total,
        n_units_total,
        q_tau,
        f_q,
        h,
        :panel_unconditional_qte
    )
end
