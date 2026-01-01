"""
Local Projections (Jordà 2005) for Impulse Response Estimation

Session 160: Julia parity for Local Projections.

Alternative to VAR-based IRF. More robust to misspecification.

Algorithm
---------
For each horizon h = 0, 1, ..., H:
    Y_{t+h} = α_h + β_h · Shock_t + γ_h · Controls_t + ε_{t+h}

    1. Run regression of Y_{t+h} on shock and controls
    2. β_h is the impulse response at horizon h
    3. HAC standard errors for Newey-West inference

Key advantages over VAR-based IRF:
- More robust to VAR misspecification (lag length, omitted variables)
- Allows for nonlinear dynamics and state dependence
- Direct estimation avoids compound error from VAR inversion

Key disadvantages:
- Less efficient when VAR is correctly specified
- Requires more data (separate regression for each horizon)
- Overlapping residuals require HAC correction

References
----------
Jordà (2005). "Estimation and Inference of Impulse Responses by Local
Projections." American Economic Review 95(1): 161-182.
"""
module LocalProjections

using LinearAlgebra
using Statistics
using Distributions

using ..TimeSeriesTypes
using ..SVARTypes

export LocalProjectionResult
export local_projection_irf, state_dependent_lp, lp_to_irf_result


"""
    LocalProjectionResult

Result from Local Projection impulse response estimation.

# Fields
- `irf`: (n_vars, n_vars, horizons+1) impulse response matrix
- `se`: (n_vars, n_vars, horizons+1) HAC standard errors
- `ci_lower`: Lower confidence band
- `ci_upper`: Upper confidence band
- `horizons`: Maximum horizon
- `n_obs`: Number of observations used
- `lags`: Number of control lags
- `alpha`: Significance level
- `method`: Estimation method ("cholesky" or "external")
- `var_names`: Variable names
- `hac_kernel`: HAC kernel used
- `hac_bandwidth`: Bandwidth for HAC
"""
struct LocalProjectionResult
    irf::Array{Float64,3}
    se::Array{Float64,3}
    ci_lower::Array{Float64,3}
    ci_upper::Array{Float64,3}
    horizons::Int
    n_obs::Int
    lags::Int
    alpha::Float64
    method::String
    var_names::Vector{String}
    hac_kernel::String
    hac_bandwidth::Int
end

function LocalProjectionResult(;
    irf::Array{Float64,3},
    se::Array{Float64,3},
    ci_lower::Array{Float64,3},
    ci_upper::Array{Float64,3},
    horizons::Int,
    n_obs::Int,
    lags::Int,
    alpha::Float64=0.05,
    method::String="cholesky",
    var_names::Vector{String},
    hac_kernel::String="bartlett",
    hac_bandwidth::Int=0
)
    LocalProjectionResult(irf, se, ci_lower, ci_upper, horizons, n_obs, lags,
                         alpha, method, var_names, hac_kernel, hac_bandwidth)
end

n_vars(r::LocalProjectionResult) = size(r.irf, 1)
has_confidence_bands(r::LocalProjectionResult) = r.ci_lower !== nothing && r.ci_upper !== nothing

function get_response(r::LocalProjectionResult, response_var::Int, shock_var::Int)
    r.irf[response_var, shock_var, :]
end

function get_response(r::LocalProjectionResult, response_var::Int, shock_var::Int, horizon::Int)
    r.irf[response_var, shock_var, horizon + 1]  # +1 for Julia 1-indexing
end

function is_significant(r::LocalProjectionResult, response_var::Int, shock_var::Int, horizon::Int)
    lower = r.ci_lower[response_var, shock_var, horizon + 1]
    upper = r.ci_upper[response_var, shock_var, horizon + 1]
    return (lower > 0) || (upper < 0)
end

function Base.show(io::IO, r::LocalProjectionResult)
    print(io, "LocalProjectionResult(n_vars=$(n_vars(r)), horizons=$(r.horizons), ",
          "lags=$(r.lags), method='$(r.method)')")
end


"""
    local_projection_irf(data; horizons=20, lags=4, shock_type="cholesky",
                         external_shock=nothing, alpha=0.05, var_names=nothing,
                         cumulative=false, hac_kernel="bartlett", hac_bandwidth=nothing)

Estimate impulse response functions using Local Projections.

# Arguments
- `data::Matrix{Float64}`: Time series data, shape (n_obs, n_vars)
- `horizons::Int`: Maximum horizon (0 to horizons inclusive)
- `lags::Int`: Number of lagged controls to include
- `shock_type::String`: "cholesky" or "external"
- `external_shock::Union{Vector{Float64}, Nothing}`: External shock series
- `alpha::Float64`: Significance level for confidence bands
- `var_names::Union{Vector{String}, Nothing}`: Variable names
- `cumulative::Bool`: If true, compute cumulative IRF
- `hac_kernel::String`: Kernel for Newey-West HAC
- `hac_bandwidth::Union{Int, Nothing}`: Bandwidth for HAC

# Returns
- `LocalProjectionResult`: Impulse responses with HAC standard errors

# Example
```julia
using Random
Random.seed!(42)
n = 200
data = zeros(n, 2)
for t in 2:n
    data[t, 1] = 0.5 * data[t-1, 1] + randn()
    data[t, 2] = 0.3 * data[t-1, 1] + 0.4 * data[t-1, 2] + randn()
end
lp = local_projection_irf(data, horizons=10, lags=2)
println("IRF[2,1,5] = ", round(lp.irf[2, 1, 6], digits=4))
```
"""
function local_projection_irf(data::Matrix{Float64};
                              horizons::Int=20,
                              lags::Int=4,
                              shock_type::String="cholesky",
                              external_shock::Union{Vector{Float64}, Nothing}=nothing,
                              alpha::Float64=0.05,
                              var_names::Union{Vector{String}, Nothing}=nothing,
                              cumulative::Bool=false,
                              hac_kernel::String="bartlett",
                              hac_bandwidth::Union{Int, Nothing}=nothing)

    n_obs, n_vars = size(data)

    if horizons < 0
        error("horizons must be >= 0, got $horizons")
    end

    if lags < 1
        error("lags must be >= 1, got $lags")
    end

    if n_obs <= lags + horizons
        error("Insufficient observations: n_obs=$n_obs, need > lags + horizons = $(lags + horizons)")
    end

    if shock_type == "external" && external_shock === nothing
        error("external_shock must be provided when shock_type='external'")
    end

    if !(shock_type in ("cholesky", "external"))
        error("shock_type must be 'cholesky' or 'external', got '$shock_type'")
    end

    if var_names === nothing
        var_names = ["var_$i" for i in 1:n_vars]
    end

    if length(var_names) != n_vars
        error("var_names length ($(length(var_names))) != n_vars ($n_vars)")
    end

    # Default HAC bandwidth: Newey-West rule
    if hac_bandwidth === nothing
        hac_bandwidth = max(1, floor(Int, 4 * (n_obs / 100)^(2/9)))
    end

    # Storage for results
    irf = zeros(n_vars, n_vars, horizons + 1)
    se = zeros(n_vars, n_vars, horizons + 1)

    if shock_type == "cholesky"
        # Compute Cholesky-orthogonalized shocks
        shocks = compute_cholesky_shocks(data, lags)

        # For each response and shock variable
        for shock_idx in 1:n_vars
            for response_idx in 1:n_vars
                for h in 0:horizons
                    beta, se_beta = lp_regression_single(
                        data, response_idx, shocks[:, shock_idx],
                        h, lags, hac_kernel, hac_bandwidth
                    )
                    irf[response_idx, shock_idx, h + 1] = beta
                    se[response_idx, shock_idx, h + 1] = se_beta
                end
            end
        end
    else  # external shock
        if length(external_shock) != n_obs
            error("external_shock length ($(length(external_shock))) != n_obs ($n_obs)")
        end

        for response_idx in 1:n_vars
            for h in 0:horizons
                beta, se_beta = lp_regression_single_external(
                    data, response_idx, external_shock,
                    h, lags, hac_kernel, hac_bandwidth
                )
                irf[response_idx, 1, h + 1] = beta
                se[response_idx, 1, h + 1] = se_beta
            end
        end
    end

    # Cumulative IRF
    if cumulative
        irf = cumsum(irf, dims=3)
        se = sqrt.(cumsum(se.^2, dims=3))
    end

    # Confidence bands
    z_crit = quantile(Normal(), 1 - alpha / 2)
    ci_lower = irf .- z_crit .* se
    ci_upper = irf .+ z_crit .* se

    return LocalProjectionResult(
        irf=irf,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        horizons=horizons,
        n_obs=n_obs - lags - horizons,
        lags=lags,
        alpha=alpha,
        method=shock_type,
        var_names=var_names,
        hac_kernel=hac_kernel,
        hac_bandwidth=hac_bandwidth
    )
end


"""
Compute Cholesky-orthogonalized shocks from data.
"""
function compute_cholesky_shocks(data::Matrix{Float64}, lags::Int)
    n_obs, n_vars = size(data)
    T = n_obs - lags

    # Build VAR regression matrices
    Y = data[lags+1:end, :]  # (T, n_vars)

    # Design matrix: constant + lagged values
    n_cols = 1 + n_vars * lags
    X = ones(T, n_cols)

    for j in 1:lags
        X[:, 2 + (j-1)*n_vars : 1 + j*n_vars] = data[lags+1-j:n_obs-j, :]
    end

    # OLS: β = (X'X)^{-1} X'Y
    XtX_inv = inv(X' * X)
    beta = XtX_inv * X' * Y

    # Residuals
    residuals = Y - X * beta

    # Covariance matrix
    Sigma = residuals' * residuals / T

    # Cholesky decomposition: Σ = PP'
    Sigma_reg = Sigma + 1e-6 * I  # Small regularization
    P = cholesky(Hermitian(Sigma_reg)).L

    # Orthogonalized shocks: ε = P^{-1} u
    P_inv = inv(P)
    shocks = (P_inv * residuals')'

    return shocks
end


"""
Run single LP regression for one horizon (Cholesky shocks).
"""
function lp_regression_single(data::Matrix{Float64}, response_idx::Int,
                              shock::Vector{Float64}, horizon::Int,
                              lags::Int, hac_kernel::String, hac_bandwidth::Int)
    n_obs, n_vars = size(data)
    shock_len = length(shock)
    T = shock_len - horizon

    if T <= lags + 2
        return 0.0, Inf
    end

    # Response variable at t+h
    Y = data[lags + horizon + 1 : lags + horizon + T, response_idx]

    # Shock at time t
    shock_t = shock[1:T]

    # Build control matrix: constant + shock + lagged Y values
    n_controls = 1 + n_vars * lags
    X = ones(T, 1 + n_controls)
    X[:, 2] = shock_t

    for j in 1:lags
        X[:, 3 + (j-1)*n_vars : 2 + j*n_vars] = data[lags - j + 1 : lags - j + T, :]
    end

    # OLS estimation
    XtX = X' * X
    XtX_inv = try
        inv(XtX)
    catch
        inv(XtX + 1e-8 * I)
    end

    beta_hat = XtX_inv * X' * Y
    beta = beta_hat[2]  # Shock coefficient

    # Residuals for HAC
    residuals = Y - X * beta_hat

    # HAC standard error
    se = newey_west_se(X, residuals, XtX_inv, hac_kernel, hac_bandwidth)
    se_beta = se[2]

    return beta, se_beta
end


"""
Run single LP regression for one horizon (external shock).
"""
function lp_regression_single_external(data::Matrix{Float64}, response_idx::Int,
                                       shock::Vector{Float64}, horizon::Int,
                                       lags::Int, hac_kernel::String, hac_bandwidth::Int)
    n_obs, n_vars = size(data)
    T = n_obs - lags - horizon

    if T <= lags + 2
        return 0.0, Inf
    end

    # Response variable at t+h
    Y = data[lags + horizon + 1 : lags + horizon + T, response_idx]

    # Shock at time t
    shock_t = shock[lags + 1 : lags + T]

    # Build control matrix
    n_controls = 1 + n_vars * lags
    X = ones(T, 1 + n_controls)
    X[:, 2] = shock_t

    for j in 1:lags
        X[:, 3 + (j-1)*n_vars : 2 + j*n_vars] = data[lags - j + 1 : lags - j + T, :]
    end

    # OLS
    XtX = X' * X
    XtX_inv = try
        inv(XtX)
    catch
        inv(XtX + 1e-8 * I)
    end

    beta_hat = XtX_inv * X' * Y
    beta = beta_hat[2]

    residuals = Y - X * beta_hat
    se = newey_west_se(X, residuals, XtX_inv, hac_kernel, hac_bandwidth)
    se_beta = se[2]

    return beta, se_beta
end


"""
Compute Newey-West HAC standard errors.
"""
function newey_west_se(X::Matrix{Float64}, residuals::Vector{Float64},
                       XtX_inv::Matrix{Float64}, kernel::String, bandwidth::Int)
    T, k = size(X)
    bandwidth = min(bandwidth, T - 2)

    # Omega = sum of weighted autocovariances
    xu = X .* residuals
    Omega = xu' * xu / T

    # Add lagged terms
    for j in 1:bandwidth
        if kernel == "bartlett"
            w = 1 - j / (bandwidth + 1)
        else  # quadratic spectral
            z = 6 * π * j / (5 * bandwidth)
            w = 3 / z^2 * (sin(z) / z - cos(z))
        end

        Gamma_j = xu[j+1:end, :]' * xu[1:end-j, :] / T
        Omega += w * (Gamma_j + Gamma_j')
    end

    # HAC variance: (X'X)^{-1} Ω (X'X)^{-1}
    V = XtX_inv * Omega * XtX_inv

    # Standard errors
    se = sqrt.(max.(diag(V), 1e-10))

    return se
end


"""
    state_dependent_lp(data, state_indicator; horizons=20, lags=4,
                       shock_type="cholesky", alpha=0.05, var_names=nothing)

State-dependent Local Projections (Auerbach & Gorodnichenko 2012).

Estimates separate impulse responses depending on a state indicator.

# Arguments
- `data::Matrix{Float64}`: Time series data
- `state_indicator::Vector{Float64}`: Binary state (1=high, 0=low)
- Other arguments same as `local_projection_irf`

# Returns
- `Dict` with keys: 'high_state_irf', 'low_state_irf', 'difference', 'diff_significant'
"""
function state_dependent_lp(data::Matrix{Float64}, state_indicator::Vector{Float64};
                           horizons::Int=20, lags::Int=4,
                           shock_type::String="cholesky",
                           alpha::Float64=0.05,
                           var_names::Union{Vector{String}, Nothing}=nothing)
    n_obs, n_vars = size(data)

    if length(state_indicator) != n_obs
        error("state_indicator length must equal n_obs ($n_obs)")
    end

    if var_names === nothing
        var_names = ["var_$i" for i in 1:n_vars]
    end

    # Compute Cholesky shocks
    if shock_type == "cholesky"
        shocks = compute_cholesky_shocks(data, lags)
    else
        error("External shocks not yet supported for state-dependent LP")
    end

    # Storage
    irf_high = zeros(n_vars, n_vars, horizons + 1)
    irf_low = zeros(n_vars, n_vars, horizons + 1)
    se_high = zeros(n_vars, n_vars, horizons + 1)
    se_low = zeros(n_vars, n_vars, horizons + 1)

    # State indicator aligned with shocks
    state_aligned = state_indicator[lags + 1 : end]

    for shock_idx in 1:n_vars
        for response_idx in 1:n_vars
            for h in 0:horizons
                betas, ses = lp_state_regression(
                    data, response_idx, shocks[:, shock_idx],
                    state_aligned, h, lags
                )
                irf_low[response_idx, shock_idx, h + 1] = betas[1]
                irf_high[response_idx, shock_idx, h + 1] = betas[2]
                se_low[response_idx, shock_idx, h + 1] = ses[1]
                se_high[response_idx, shock_idx, h + 1] = ses[2]
            end
        end
    end

    z_crit = quantile(Normal(), 1 - alpha / 2)

    diff = irf_high .- irf_low
    se_diff = sqrt.(se_high.^2 .+ se_low.^2)
    diff_significant = abs.(diff) .> z_crit .* se_diff

    return Dict(
        "high_state_irf" => LocalProjectionResult(
            irf=irf_high,
            se=se_high,
            ci_lower=irf_high .- z_crit .* se_high,
            ci_upper=irf_high .+ z_crit .* se_high,
            horizons=horizons,
            n_obs=n_obs - lags - horizons,
            lags=lags,
            alpha=alpha,
            method=shock_type,
            var_names=var_names,
            hac_kernel="bartlett",
            hac_bandwidth=0
        ),
        "low_state_irf" => LocalProjectionResult(
            irf=irf_low,
            se=se_low,
            ci_lower=irf_low .- z_crit .* se_low,
            ci_upper=irf_low .+ z_crit .* se_low,
            horizons=horizons,
            n_obs=n_obs - lags - horizons,
            lags=lags,
            alpha=alpha,
            method=shock_type,
            var_names=var_names,
            hac_kernel="bartlett",
            hac_bandwidth=0
        ),
        "difference" => diff,
        "diff_significant" => diff_significant
    )
end


"""
Run state-dependent LP regression with interaction terms.
"""
function lp_state_regression(data::Matrix{Float64}, response_idx::Int,
                             shock::Vector{Float64}, state::Vector{Float64},
                             horizon::Int, lags::Int)
    n_obs, n_vars = size(data)
    shock_len = length(shock)
    T = shock_len - horizon

    if T <= lags + 4
        return (0.0, 0.0), (Inf, Inf)
    end

    # Response at t+h
    Y = data[lags + horizon + 1 : lags + horizon + T, response_idx]

    # Shock and state at t
    shock_t = shock[1:T]
    state_t = state[1:T]

    # Design matrix with interactions
    n_controls = n_vars * lags
    X = zeros(T, 4 + n_controls)
    X[:, 1] .= 1.0
    X[:, 2] = state_t
    X[:, 3] = shock_t .* (1.0 .- state_t)  # Low state
    X[:, 4] = shock_t .* state_t           # High state

    for j in 1:lags
        X[:, 5 + (j-1)*n_vars : 4 + j*n_vars] = data[lags - j + 1 : lags - j + T, :]
    end

    # OLS
    XtX = X' * X
    XtX_inv = try
        inv(XtX)
    catch
        inv(XtX + 1e-8 * I)
    end

    beta_hat = XtX_inv * X' * Y
    residuals = Y - X * beta_hat

    # HAC SEs
    bandwidth = max(1, floor(Int, 4 * (T / 100)^(2/9)))
    se = newey_west_se(X, residuals, XtX_inv, "bartlett", bandwidth)

    return (beta_hat[3], beta_hat[4]), (se[3], se[4])
end


"""
    lp_to_irf_result(lp_result::LocalProjectionResult)

Convert LocalProjectionResult to IRFResult for API compatibility.
"""
function lp_to_irf_result(lp_result::LocalProjectionResult)
    IRFResult(
        irf=lp_result.irf,
        irf_lower=lp_result.ci_lower,
        irf_upper=lp_result.ci_upper,
        horizons=lp_result.horizons,
        cumulative=false,
        orthogonalized=true,
        var_names=lp_result.var_names,
        alpha=lp_result.alpha,
        n_bootstrap=0
    )
end


end # module
