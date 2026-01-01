"""
Proxy SVAR (External Instrument SVAR)

Session 164: Structural VAR identification using external instruments.

Stock & Watson (2012), Mertens & Ravn (2013).

The proxy SVAR approach uses external instruments (proxies) that are:
1. Relevant: Correlated with the target structural shock
2. Exogenous: Uncorrelated with other structural shocks

This achieves point identification of one column of B0_inv without
imposing the recursive ordering assumptions of Cholesky identification.

Algorithm:
1. Estimate reduced-form VAR -> residuals u_t
2. First stage: Regress u_target on z_t -> fitted values u_hat_target
3. Second stage: Regress u_i on u_hat_target for each i -> beta_i
4. Stack coefficients to form first column of B0_inv
5. Complete B0_inv via variance decomposition

Key insight: Cov(z_t, u_t) = B0_inv * Cov(z_t, eps_t)
If z correlated only with eps_1: Cov(z_t, u_t) is proportional to first column of B0_inv

References:
- Stock & Watson (2012): "Disentangling the Channels of the 2007-09 Recession"
- Mertens & Ravn (2013): "The Dynamic Effects of Personal and Corporate Income Tax Changes"
- Gertler & Karadi (2015): "Monetary Policy Surprises, Credit Costs, and Economic Activity"
"""
module ProxySVAR

using LinearAlgebra
using Statistics
using Random
using Distributions

using ..TimeSeriesTypes: VARResult
using ..SVARTypes: SVARResult, IRFResult, IdentificationMethod, PROXY
using ..SVAR: cholesky_svar, vma_coefficients

export ProxySVARResult
export proxy_svar, weak_instrument_diagnostics, compute_irf_from_proxy
export first_stage_regression, compute_proxy_impact_column
export complete_impact_matrix, proxy_bootstrap


"""
    ProxySVARResult

Result from Proxy SVAR (External Instrument) estimation.

Proxy SVAR identifies structural shocks using external instruments
correlated with the target shock but uncorrelated with other shocks.

# Fields
- `var_result::VARResult`: Underlying reduced-form VAR estimation
- `B0_inv::Matrix{Float64}`: Impact matrix (n_vars x n_vars)
- `B0::Matrix{Float64}`: Inverse of B0_inv
- `structural_shocks::Matrix{Float64}`: Structural shock series
- `identification::IdentificationMethod`: Always PROXY
- `target_shock_idx::Int`: Index of the shock being identified (1-indexed)
- `target_residual_idx::Int`: Index of the VAR residual used in first stage
- `instrument::Vector{Float64}`: External instrument used for identification
- `first_stage_f_stat::Float64`: F-statistic from first-stage regression
- `first_stage_r2::Float64`: R-squared from first-stage regression
- `is_weak_instrument::Bool`: True if F < threshold (Stock-Yogo rule: F < 10)
- `reliability_ratio::Float64`: Ratio of signal to total variance
- `impact_column::Vector{Float64}`: Identified column of B0_inv
- `impact_column_se::Vector{Float64}`: Standard errors for impact_column
- `impact_column_ci_lower::Vector{Float64}`: Lower confidence bounds
- `impact_column_ci_upper::Vector{Float64}`: Upper confidence bounds
- `n_restrictions::Int`: Number of restrictions (n_vars - 1 for proxy SVAR)
- `is_just_identified::Bool`: True for single-instrument proxy SVAR
- `is_over_identified::Bool`: False for single instrument
- `alpha::Float64`: Significance level for confidence intervals
- `bootstrap_se::Union{Vector{Float64},Nothing}`: Bootstrap standard errors
- `n_bootstrap::Int`: Number of bootstrap replications (0 if no bootstrap)

# Example
```julia
result = proxy_svar(var_result, instrument, target_shock_idx=1)
println("F-stat: ", result.first_stage_f_stat)
println("Impact column: ", result.impact_column)
```
"""
struct ProxySVARResult
    var_result::VARResult
    B0_inv::Matrix{Float64}
    B0::Matrix{Float64}
    structural_shocks::Matrix{Float64}
    identification::IdentificationMethod

    # Proxy-specific fields
    target_shock_idx::Int
    target_residual_idx::Int
    instrument::Vector{Float64}
    first_stage_f_stat::Float64
    first_stage_r2::Float64
    is_weak_instrument::Bool
    reliability_ratio::Float64

    # Impact column inference
    impact_column::Vector{Float64}
    impact_column_se::Vector{Float64}
    impact_column_ci_lower::Vector{Float64}
    impact_column_ci_upper::Vector{Float64}

    # Metadata
    n_restrictions::Int
    is_just_identified::Bool
    is_over_identified::Bool
    alpha::Float64

    # Optional bootstrap
    bootstrap_se::Union{Vector{Float64},Nothing}
    n_bootstrap::Int
end

# Keyword constructor
function ProxySVARResult(;
    var_result::VARResult,
    B0_inv::Matrix{Float64},
    B0::Matrix{Float64},
    structural_shocks::Matrix{Float64},
    identification::IdentificationMethod=PROXY,
    target_shock_idx::Int=1,
    target_residual_idx::Int=1,
    instrument::Vector{Float64}=Float64[],
    first_stage_f_stat::Float64=0.0,
    first_stage_r2::Float64=0.0,
    is_weak_instrument::Bool=false,
    reliability_ratio::Float64=0.0,
    impact_column::Vector{Float64}=Float64[],
    impact_column_se::Vector{Float64}=Float64[],
    impact_column_ci_lower::Vector{Float64}=Float64[],
    impact_column_ci_upper::Vector{Float64}=Float64[],
    n_restrictions::Int=0,
    is_just_identified::Bool=true,
    is_over_identified::Bool=false,
    alpha::Float64=0.05,
    bootstrap_se::Union{Vector{Float64},Nothing}=nothing,
    n_bootstrap::Int=0
)
    ProxySVARResult(
        var_result, B0_inv, B0, structural_shocks, identification,
        target_shock_idx, target_residual_idx, instrument,
        first_stage_f_stat, first_stage_r2, is_weak_instrument, reliability_ratio,
        impact_column, impact_column_se, impact_column_ci_lower, impact_column_ci_upper,
        n_restrictions, is_just_identified, is_over_identified, alpha,
        bootstrap_se, n_bootstrap
    )
end

# Accessors
n_vars(r::ProxySVARResult) = size(r.B0_inv, 1)
lags(r::ProxySVARResult) = r.var_result.lags
n_obs(r::ProxySVARResult) = r.var_result.n_obs_effective
var_names(r::ProxySVARResult) = r.var_result.var_names
has_bootstrap_se(r::ProxySVARResult) = r.bootstrap_se !== nothing

"""
    get_structural_coefficient(r::ProxySVARResult, shock_var::Int, response_var::Int)

Get contemporaneous impact of structural shock on variable.
"""
function get_structural_coefficient(r::ProxySVARResult, shock_var::Int, response_var::Int)
    r.B0_inv[response_var, shock_var]
end

function Base.show(io::IO, r::ProxySVARResult)
    weak_str = r.is_weak_instrument ? ", WEAK" : ""
    print(io, "ProxySVARResult(n_vars=$(n_vars(r)), lags=$(lags(r)), ",
          "F=$(round(r.first_stage_f_stat, digits=2))$(weak_str))")
end


# =============================================================================
# First-Stage Regression
# =============================================================================

"""
    first_stage_regression(y::Vector{Float64}, z::Vector{Float64})

First-stage regression for proxy SVAR: y = gamma_0 + gamma_1 * z + v

# Arguments
- `y`: Dependent variable (target residual)
- `z`: Instrument

# Returns
- `fitted`: Predicted y values
- `f_stat`: F-statistic for instrument strength
- `r2`: R-squared
- `residuals`: First-stage residuals
"""
function first_stage_regression(
    y::Vector{Float64},
    z::Vector{Float64}
)::Tuple{Vector{Float64}, Float64, Float64, Vector{Float64}}
    n = length(y)

    # Design matrix with intercept
    X = hcat(ones(n), z)

    # OLS: beta = (X'X)^{-1} X'y
    XtX = X' * X
    Xty = X' * y
    beta = XtX \ Xty

    # Fitted values
    fitted = X * beta

    # Residuals
    residuals = y - fitted

    # R-squared
    ss_res = sum(residuals.^2)
    ss_tot = sum((y .- mean(y)).^2)
    r2 = ss_tot > 0 ? 1 - ss_res / ss_tot : 0.0

    # F-statistic: F = (R^2 / (1-R^2)) * ((n-k) / (k-1))
    # For simple regression with intercept: k=2, so (k-1)=1
    k = 2
    if r2 < 1.0
        f_stat = (r2 / (1 - r2)) * ((n - k) / (k - 1))
    else
        f_stat = Inf
    end

    return fitted, f_stat, r2, residuals
end


# =============================================================================
# Proxy Impact Column Computation
# =============================================================================

"""
    compute_proxy_impact_column(residuals, instrument, target_residual_idx)

Compute impact column via 2SLS on VAR residuals.

# Arguments
- `residuals::Matrix{Float64}`: Shape (n_obs, n_vars) VAR residuals
- `instrument::Vector{Float64}`: External instrument
- `target_residual_idx::Int`: Which residual is instrumented (1-indexed)

# Returns
- `impact_column`: First column of B0_inv (shape: n_vars)
- `f_stat`: First-stage F-statistic
- `r2`: First-stage R-squared
- `standard_errors`: Delta-method standard errors
- `first_stage_residuals`: Residuals from first-stage regression
"""
function compute_proxy_impact_column(
    residuals::Matrix{Float64},
    instrument::Vector{Float64},
    target_residual_idx::Int
)::Tuple{Vector{Float64}, Float64, Float64, Vector{Float64}, Vector{Float64}}
    n_obs, n_vars_local = size(residuals)

    # Target residual (the one we're instrumenting)
    u_target = residuals[:, target_residual_idx]

    # First stage: regress u_target on z
    fitted, f_stat, r2, first_stage_resid = first_stage_regression(u_target, instrument)

    # Second stage: regress each u_i on fitted values
    impact_column = zeros(n_vars_local)
    standard_errors = zeros(n_vars_local)

    # Design matrix for second stage
    X_second = hcat(ones(n_obs), fitted)

    for i in 1:n_vars_local
        u_i = residuals[:, i]

        # OLS: u_i = a + b * fitted + error
        XtX = X_second' * X_second
        Xty = X_second' * u_i
        beta = XtX \ Xty

        impact_column[i] = beta[2]  # Coefficient on fitted values

        # Standard error
        resid_second = u_i - X_second * beta
        sigma2 = sum(resid_second.^2) / (n_obs - 2)
        var_beta = sigma2 * inv(XtX)
        standard_errors[i] = sqrt(var_beta[2, 2])
    end

    # Normalize: scale so target shock element = 1
    scale = impact_column[target_residual_idx]
    if abs(scale) < 1e-10
        @warn "Target coefficient near zero in second stage. Using unit scaling."
        scale = 1.0
    end

    impact_column = impact_column / scale
    standard_errors = standard_errors / abs(scale)

    return impact_column, f_stat, r2, standard_errors, first_stage_resid
end


# =============================================================================
# Complete Impact Matrix
# =============================================================================

"""
    complete_impact_matrix(impact_column, sigma_u, target_shock_idx)

Complete B0_inv given identified first column.

Uses variance decomposition: Sigma_u = B0_inv * B0_inv'

Given b_1 (target column), solve for remaining columns using
Cholesky decomposition approach.

# Arguments
- `impact_column`: Identified column of B0_inv (shape: n_vars)
- `sigma_u`: Reduced-form residual covariance (n_vars x n_vars)
- `target_shock_idx`: Index where impact_column should be placed (1-indexed)

# Returns
Complete impact matrix B0_inv (n_vars x n_vars)
"""
function complete_impact_matrix(
    impact_column::Vector{Float64},
    sigma_u::Matrix{Float64},
    target_shock_idx::Int
)::Matrix{Float64}
    n_vars_local = length(impact_column)

    # Start with Cholesky as baseline
    L = cholesky(Symmetric(sigma_u)).L

    # Compute scaling for impact column
    sigma_u_inv = inv(sigma_u)
    shock_variance = impact_column' * sigma_u_inv * impact_column

    if shock_variance <= 0
        @warn "Non-positive shock variance. Using Cholesky fallback."
        return Matrix(L)
    end

    # Scale impact column so shock has unit variance
    scaled_impact = impact_column / sqrt(shock_variance)

    # Build B0_inv by placing identified column and using Cholesky for rest
    B0_inv = Matrix(L)
    B0_inv[:, target_shock_idx] = scaled_impact

    # Orthogonalize remaining columns with respect to identified column
    for j in 1:n_vars_local
        if j != target_shock_idx
            # Project out the identified column component
            proj = (scaled_impact' * B0_inv[:, j]) * scaled_impact
            B0_inv[:, j] = B0_inv[:, j] - proj

            # Re-normalize to preserve variance contribution
            col_norm = norm(B0_inv[:, j])
            if col_norm > 1e-10
                orig_norm = norm(L[:, j])
                B0_inv[:, j] = B0_inv[:, j] * (orig_norm / col_norm)
            end
        end
    end

    return B0_inv
end


# =============================================================================
# Bootstrap Inference
# =============================================================================

"""
    proxy_bootstrap(residuals, instrument, target_residual_idx, n_bootstrap, rng)

Bootstrap inference for proxy SVAR.

Uses residual resampling to compute bootstrap distribution of impact coefficients.

# Arguments
- `residuals`: VAR residuals (n_obs x n_vars)
- `instrument`: External instrument
- `target_residual_idx`: Which residual is instrumented (1-indexed)
- `n_bootstrap`: Number of bootstrap replications
- `rng`: Random number generator

# Returns
Bootstrap standard errors for impact column
"""
function proxy_bootstrap(
    residuals::Matrix{Float64},
    instrument::Vector{Float64},
    target_residual_idx::Int,
    n_bootstrap::Int,
    rng::AbstractRNG
)::Vector{Float64}
    n_obs, n_vars_local = size(residuals)
    impact_boots = zeros(n_bootstrap, n_vars_local)
    valid_count = 0

    for b in 1:n_bootstrap
        # Resample indices
        boot_idx = rand(rng, 1:n_obs, n_obs)

        # Resample residuals and instrument
        resid_boot = residuals[boot_idx, :]
        inst_boot = instrument[boot_idx]

        # Compute impact column for bootstrap sample
        try
            impact_b, _, _, _, _ = compute_proxy_impact_column(
                resid_boot,
                inst_boot,
                target_residual_idx
            )
            impact_boots[b, :] = impact_b
            valid_count += 1
        catch e
            # If bootstrap sample fails, fill with NaN
            impact_boots[b, :] .= NaN
        end
    end

    # Compute standard errors, ignoring NaN samples
    bootstrap_se = zeros(n_vars_local)
    for i in 1:n_vars_local
        vals = filter(!isnan, impact_boots[:, i])
        if length(vals) > 1
            bootstrap_se[i] = std(vals, corrected=true)
        else
            bootstrap_se[i] = NaN
        end
    end

    return bootstrap_se
end


# =============================================================================
# Main Estimation Function
# =============================================================================

"""
    proxy_svar(var_result, instrument; kwargs...)

Proxy SVAR identification via external instruments.

Identifies structural shocks using external instruments (proxies)
following Stock & Watson (2012) and Mertens & Ravn (2013).

# Arguments
- `var_result::VARResult`: Estimated reduced-form VAR
- `instrument::Vector{Float64}`: External instrument z_t

# Keyword Arguments
- `target_shock_idx::Int=1`: Which structural shock to identify (1-indexed)
- `target_residual_idx::Union{Int,Nothing}=nothing`: Which VAR residual is instrumented
- `alpha::Float64=0.05`: Significance level for confidence intervals
- `bootstrap_se::Bool=false`: Whether to compute bootstrap standard errors
- `n_bootstrap::Int=500`: Number of bootstrap replications
- `weak_instrument_threshold::Float64=10.0`: F-statistic threshold for weak instrument
- `seed::Union{Int,Nothing}=nothing`: Random seed for reproducibility

# Returns
`ProxySVARResult`: Estimation result with identified impact column and diagnostics

# Raises
- Error if instrument length doesn't match n_obs_effective
- Error if instrument has zero variance (constant)
- Error if target indices are out of bounds

# Example
```julia
var_result = var_estimate(data, lags=4)
result = proxy_svar(var_result, instrument, target_shock_idx=1)
println("F-stat: ", result.first_stage_f_stat)
if result.is_weak_instrument
    println("Warning: Weak instrument detected!")
end
```
"""
function proxy_svar(
    var_result::VARResult,
    instrument::Vector{Float64};
    target_shock_idx::Int=1,
    target_residual_idx::Union{Int,Nothing}=nothing,
    alpha::Float64=0.05,
    bootstrap_se::Bool=false,
    n_bootstrap::Int=500,
    weak_instrument_threshold::Float64=10.0,
    seed::Union{Int,Nothing}=nothing
)::ProxySVARResult

    n_obs_eff = var_result.n_obs_effective
    n_vars_local = length(var_result.var_names)

    # Input validation
    if length(instrument) != n_obs_eff
        error("Instrument length ($(length(instrument))) must match " *
              "VAR n_obs_effective ($n_obs_eff). " *
              "If VAR has $(var_result.lags) lags, instrument should be trimmed accordingly.")
    end

    if var(instrument) < 1e-10
        error("Instrument has near-zero variance (constant). " *
              "Cannot compute first-stage regression.")
    end

    if any(isnan, instrument)
        error("Instrument contains NaN values. " *
              "Please handle missing values before calling proxy_svar.")
    end

    if target_shock_idx < 1 || target_shock_idx > n_vars_local
        error("target_shock_idx ($target_shock_idx) out of bounds. " *
              "Must be in [1, $n_vars_local].")
    end

    if target_residual_idx === nothing
        target_residual_idx = target_shock_idx
    end

    if target_residual_idx < 1 || target_residual_idx > n_vars_local
        error("target_residual_idx ($target_residual_idx) out of bounds. " *
              "Must be in [1, $n_vars_local].")
    end

    # Get VAR residuals
    residuals = var_result.residuals  # (n_obs_effective, n_vars)
    sigma_u = var_result.sigma  # (n_vars, n_vars)

    # Compute proxy impact column via 2SLS
    impact_column, f_stat, r2, impact_se, first_stage_residuals = compute_proxy_impact_column(
        residuals,
        instrument,
        target_residual_idx
    )

    # Check for weak instrument
    is_weak = f_stat < weak_instrument_threshold
    if is_weak
        @warn "Weak instrument detected: F-statistic ($(round(f_stat, digits=2))) < " *
              "threshold ($weak_instrument_threshold). " *
              "Estimates may be biased. Consider using robust inference."
    end

    # Compute reliability ratio
    reliability = r2

    # Complete the impact matrix
    B0_inv = complete_impact_matrix(
        impact_column,
        sigma_u,
        target_shock_idx
    )

    # Compute B0 and structural shocks
    B0 = inv(B0_inv)
    structural_shocks = Matrix((B0 * residuals')')

    # Confidence intervals (delta method)
    z_crit = quantile(Normal(), 1 - alpha / 2)
    ci_lower = impact_column - z_crit * impact_se
    ci_upper = impact_column + z_crit * impact_se

    # Optional bootstrap
    bootstrap_se_result = nothing
    n_boot_actual = 0
    if bootstrap_se
        rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)

        bootstrap_se_result = proxy_bootstrap(
            residuals,
            instrument,
            target_residual_idx,
            n_bootstrap,
            rng
        )
        n_boot_actual = n_bootstrap

        # Update CIs with bootstrap
        ci_lower = impact_column - z_crit * bootstrap_se_result
        ci_upper = impact_column + z_crit * bootstrap_se_result
    end

    return ProxySVARResult(
        var_result=var_result,
        B0_inv=B0_inv,
        B0=B0,
        structural_shocks=structural_shocks,
        identification=PROXY,
        target_shock_idx=target_shock_idx,
        target_residual_idx=target_residual_idx,
        instrument=instrument,
        first_stage_f_stat=f_stat,
        first_stage_r2=r2,
        is_weak_instrument=is_weak,
        reliability_ratio=reliability,
        impact_column=impact_column,
        impact_column_se=impact_se,
        impact_column_ci_lower=ci_lower,
        impact_column_ci_upper=ci_upper,
        n_restrictions=n_vars_local - 1,
        is_just_identified=true,
        is_over_identified=false,
        alpha=alpha,
        bootstrap_se=bootstrap_se_result,
        n_bootstrap=n_boot_actual
    )
end


# =============================================================================
# Weak Instrument Diagnostics
# =============================================================================

"""
    weak_instrument_diagnostics(f_stat, n_obs; kwargs...)

Comprehensive weak instrument diagnostics for proxy SVAR.

# Arguments
- `f_stat::Float64`: First-stage F-statistic
- `n_obs::Int`: Number of observations

# Keyword Arguments
- `n_instruments::Int=1`: Number of instruments
- `alpha::Float64=0.05`: Significance level

# Returns
Dict with diagnostic results:
- `f_stat`: First-stage F-statistic
- `is_weak`: Boolean for weak instrument (F < 10)
- `is_very_weak`: Boolean for very weak instrument (F < 5)
- `stock_yogo_critical_15pct`: Stock-Yogo critical value for 15% size
- `stock_yogo_critical_values`: Dict of all critical values
- `interpretation`: String interpretation
- `recommended_inference`: Recommended inference method

# Notes
Stock & Yogo (2005) critical values for single endogenous variable:
- For 10% maximal IV size: F > 16.38
- For 15% maximal IV size: F > 8.96
- For 20% maximal IV size: F > 6.66
- For 25% maximal IV size: F > 5.53

Rule of thumb: F > 10 is "strong enough" for reliable inference.

# Example
```julia
diag = weak_instrument_diagnostics(f_stat=8.5, n_obs=200)
println(diag["interpretation"])
```
"""
function weak_instrument_diagnostics(
    f_stat::Float64,
    n_obs::Int;
    n_instruments::Int=1,
    alpha::Float64=0.05
)::Dict{String,Any}

    # Stock-Yogo critical values for single instrument, single endogenous
    stock_yogo = Dict(
        0.10 => 16.38,  # 10% maximal size
        0.15 => 8.96,   # 15% maximal size
        0.20 => 6.66,   # 20% maximal size
        0.25 => 5.53    # 25% maximal size
    )

    is_weak = f_stat < 10.0
    is_very_weak = f_stat < 5.0

    # Find Stock-Yogo threshold for 15% size
    sy_critical = get(stock_yogo, 0.15, 8.96)

    # Interpretation
    if f_stat >= 16.38
        interpretation = "Strong instrument (F=$(round(f_stat, digits=2)) >= 16.38). " *
                         "Standard inference is reliable."
        recommended = "standard"
    elseif f_stat >= 10.0
        interpretation = "Moderate instrument strength (F=$(round(f_stat, digits=2))). " *
                         "Standard inference likely OK, but consider robust methods."
        recommended = "standard_with_caution"
    elseif f_stat >= 5.0
        interpretation = "Weak instrument (F=$(round(f_stat, digits=2)) < 10). " *
                         "Standard errors biased. Use weak-instrument robust methods."
        recommended = "anderson_rubin"
    else
        interpretation = "Very weak instrument (F=$(round(f_stat, digits=2)) < 5). " *
                         "Estimates unreliable. Consider alternative instruments."
        recommended = "reconsider_instrument"
    end

    return Dict{String,Any}(
        "f_stat" => f_stat,
        "is_weak" => is_weak,
        "is_very_weak" => is_very_weak,
        "stock_yogo_critical_15pct" => sy_critical,
        "stock_yogo_critical_values" => stock_yogo,
        "interpretation" => interpretation,
        "recommended_inference" => recommended,
        "n_obs" => n_obs,
        "n_instruments" => n_instruments
    )
end


# =============================================================================
# IRF Computation
# =============================================================================

"""
    compute_irf_from_proxy(result; horizons=20)

Compute impulse responses from proxy SVAR result.

# Arguments
- `result::ProxySVARResult`: Proxy SVAR estimation result

# Keyword Arguments
- `horizons::Int=20`: Maximum horizon

# Returns
IRF array of shape (n_vars, n_vars, horizons+1)
irf[i, j, h+1] = response of var i to shock j at horizon h

# Notes
Only the column corresponding to target_shock_idx is directly identified.
Other columns are based on the completed B0_inv matrix.
"""
function compute_irf_from_proxy(
    result::ProxySVARResult;
    horizons::Int=20
)::Array{Float64,3}

    # Get VMA coefficients
    Phi = vma_coefficients(result.var_result, horizons)  # (n_vars, n_vars, horizons+1)

    n_vars_local = n_vars(result)

    # Compute structural IRF: IRF_h = Phi_h * B0_inv
    irf = zeros(n_vars_local, n_vars_local, horizons + 1)
    for h in 0:horizons
        irf[:, :, h + 1] = Phi[:, :, h + 1] * result.B0_inv
    end

    return irf
end


"""
    to_irf_result(r::ProxySVARResult; horizons=20)

Convert ProxySVARResult to standard IRFResult for compatibility.
"""
function to_irf_result(r::ProxySVARResult; horizons::Int=20)::IRFResult
    irf = compute_irf_from_proxy(r, horizons=horizons)

    IRFResult(
        irf=irf,
        irf_lower=nothing,
        irf_upper=nothing,
        horizons=horizons,
        cumulative=false,
        orthogonalized=true,
        var_names=r.var_result.var_names,
        alpha=r.alpha,
        n_bootstrap=0
    )
end


end # module
