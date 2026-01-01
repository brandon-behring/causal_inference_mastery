#=
Time-Varying Parameter VAR (TVP-VAR) Estimation.

Session 165: Kalman filter estimation for VAR with time-varying coefficients.

The TVP-VAR model (Primiceri 2005, Cogley & Sargent 2005):

State-Space Representation:
    Measurement:  Y_t = X_t β_t + ε_t,     ε_t ~ N(0, Σ)
    Transition:   β_t = β_{t-1} + w_t,     w_t ~ N(0, Q)

Where:
    Y_t: k×1 observation vector (VAR endogenous variables)
    X_t: k×(k²p+k) regressor matrix (lagged Y, intercepts)
    β_t: (k²p+k)×1 time-varying coefficient vector
    Σ: k×k observation covariance
    Q: (k²p+k)×(k²p+k) state transition covariance

The Kalman filter recursion:
    1. Predict: β_{t|t-1} = β_{t-1|t-1}, P_{t|t-1} = P_{t-1|t-1} + Q
    2. Update: v_t = Y_t - X_t β_{t|t-1} (innovation)
               F_t = X_t P_{t|t-1} X_t' + Σ (innovation covariance)
               K_t = P_{t|t-1} X_t' F_t⁻¹ (Kalman gain)
               β_{t|t} = β_{t|t-1} + K_t v_t
               P_{t|t} = (I - K_t X_t) P_{t|t-1} (Joseph form for stability)
    3. Log-likelihood: ℓ = -0.5 Σ_t [k log(2π) + log|F_t| + v_t' F_t⁻¹ v_t]

References:
- Primiceri (2005). "Time Varying Structural VARs and Monetary Policy."
- Cogley & Sargent (2005). "Drifts and Volatilities."
=#

module TVPVAR

using LinearAlgebra
using Statistics
using Distributions

using ..TimeSeriesTypes: VARResult
using ..VAR: var_estimate, build_var_matrices

export TVPVARResult
export tvp_var_estimate, tvp_var_smooth
export compute_tvp_irf, compute_tvp_irf_all_times
export check_tvp_stability, check_tvp_stability_all_times
export coefficient_change_test


"""
    TVPVARResult

Result from Time-Varying Parameter VAR estimation.

Primiceri (2005) style TVP-VAR with Kalman filter estimation.
Coefficients follow random walk: β_t = β_{t-1} + w_t, w_t ~ N(0, Q).

# Fields
- `coefficients_filtered::Array{Float64,3}`: Shape (T, n_vars, n_params_per_eq).
    Filtered coefficient estimates β_{t|t}.
- `coefficients_smoothed::Union{Array{Float64,3},Nothing}`: Smoothed estimates β_{t|T}.
- `covariance_filtered::Array{Float64,3}`: Shape (T, state_dim, state_dim).
    Filtered state covariance P_{t|t}.
- `covariance_smoothed::Union{Array{Float64,3},Nothing}`: Smoothed covariance P_{t|T}.
- `innovations::Matrix{Float64}`: Shape (T, n_vars). Innovation sequence v_t.
- `innovation_covariance::Array{Float64,3}`: Shape (T, n_vars, n_vars). F_t.
- `kalman_gain::Array{Float64,3}`: Shape (T, state_dim, n_vars). K_t.
- `sigma::Matrix{Float64}`: Observation covariance Σ.
- `Q::Matrix{Float64}`: State transition covariance Q.
- `log_likelihood::Float64`: Total log-likelihood.
- `aic::Float64`: Akaike Information Criterion.
- `bic::Float64`: Bayesian Information Criterion.
- `lags::Int`: Number of VAR lags p.
- `n_vars::Int`: Number of endogenous variables k.
- `n_obs::Int`: Number of time observations T.
- `n_obs_effective::Int`: Effective observations (T - lags).
- `state_dim::Int`: State dimension (k × (k*p + 1)).
- `var_names::Vector{String}`: Variable names.
- `initialization::String`: Initialization method used.
"""
struct TVPVARResult
    coefficients_filtered::Array{Float64,3}
    coefficients_smoothed::Union{Array{Float64,3},Nothing}
    covariance_filtered::Array{Float64,3}
    covariance_smoothed::Union{Array{Float64,3},Nothing}
    innovations::Matrix{Float64}
    innovation_covariance::Array{Float64,3}
    kalman_gain::Array{Float64,3}
    sigma::Matrix{Float64}
    Q::Matrix{Float64}
    log_likelihood::Float64
    aic::Float64
    bic::Float64
    lags::Int
    n_vars::Int
    n_obs::Int
    n_obs_effective::Int
    state_dim::Int
    var_names::Vector{String}
    initialization::String
end

"""Number of parameters per equation (n_vars*lags + 1)."""
n_params_per_eq(result::TVPVARResult) = result.n_vars * result.lags + 1

"""Whether smoothed estimates are available."""
has_smoothed(result::TVPVARResult) = !isnothing(result.coefficients_smoothed)


"""
    get_coefficients_at_time(result, t; smoothed=true)

Get coefficient matrix at specific time.

# Arguments
- `result::TVPVARResult`: Estimated TVP-VAR
- `t::Int`: Time index (1-indexed, Julia convention)
- `smoothed::Bool`: Use smoothed (true) or filtered (false) estimates

# Returns
- `Matrix{Float64}`: Shape (n_vars, n_params_per_eq) coefficient matrix at time t
"""
function get_coefficients_at_time(
    result::TVPVARResult,
    t::Int;
    smoothed::Bool=true
)
    if t < 1 || t > result.n_obs_effective
        error("Time index $t out of bounds [1, $(result.n_obs_effective)]")
    end

    if smoothed && has_smoothed(result)
        return result.coefficients_smoothed[t, :, :]
    end
    return result.coefficients_filtered[t, :, :]
end


"""
    get_lag_matrix_at_time(result, t, lag; smoothed=true)

Get coefficient matrix for specific lag at specific time.

# Arguments
- `result::TVPVARResult`: Estimated TVP-VAR
- `t::Int`: Time index (1-indexed)
- `lag::Int`: Lag number (1 to lags)
- `smoothed::Bool`: Use smoothed or filtered estimates

# Returns
- `Matrix{Float64}`: Shape (n_vars, n_vars) coefficient matrix A_lag at time t
"""
function get_lag_matrix_at_time(
    result::TVPVARResult,
    t::Int,
    lag::Int;
    smoothed::Bool=true
)
    if lag < 1 || lag > result.lags
        error("Lag must be between 1 and $(result.lags), got $lag")
    end

    coef = get_coefficients_at_time(result, t; smoothed=smoothed)
    # Coefficients ordered: [intercept, lag1_var1, lag1_var2, ..., lagp_vark]
    start_idx = 2 + (lag - 1) * result.n_vars
    end_idx = start_idx + result.n_vars - 1
    return coef[:, start_idx:end_idx]
end


"""
    get_intercepts_at_time(result, t; smoothed=true)

Get intercept vector at specific time.
"""
function get_intercepts_at_time(
    result::TVPVARResult,
    t::Int;
    smoothed::Bool=true
)
    coef = get_coefficients_at_time(result, t; smoothed=smoothed)
    return coef[:, 1]
end


"""
    coefficient_trajectory(result, equation_idx, coef_idx; smoothed=true)

Get trajectory of a single coefficient over time.

# Arguments
- `result::TVPVARResult`: Estimated TVP-VAR
- `equation_idx::Int`: Which equation (1-indexed)
- `coef_idx::Int`: Which coefficient in that equation (1-indexed)
- `smoothed::Bool`: Use smoothed or filtered estimates

# Returns
- `Vector{Float64}`: Shape (T,) coefficient values over time
"""
function coefficient_trajectory(
    result::TVPVARResult,
    equation_idx::Int,
    coef_idx::Int;
    smoothed::Bool=true
)
    if smoothed && has_smoothed(result)
        return result.coefficients_smoothed[:, equation_idx, coef_idx]
    end
    return result.coefficients_filtered[:, equation_idx, coef_idx]
end


"""
    tvp_var_estimate(data; kwargs...)

Estimate Time-Varying Parameter VAR via Kalman filter.

# Arguments
- `data::Matrix{Float64}`: Shape (T, n_vars). Time series data.
- `lags::Int=1`: Number of VAR lags p.
- `Q_init::Union{Matrix{Float64},Nothing}=nothing`: State transition covariance.
- `Q_scale::Float64=0.001`: Scale factor for Q when Q_init is nothing.
- `sigma_init::Union{Matrix{Float64},Nothing}=nothing`: Observation covariance.
- `initialization::Symbol=:ols`: :ols, :diffuse, or :custom.
- `beta_init::Union{Vector{Float64},Nothing}=nothing`: Initial state (for :custom).
- `P_init::Union{Matrix{Float64},Nothing}=nothing`: Initial covariance (for :custom).
- `diffuse_scale::Float64=1e6`: Scale for diffuse initialization.
- `smooth::Bool=true`: Whether to run RTS smoother.
- `var_names::Union{Vector{String},Nothing}=nothing`: Variable names.

# Returns
- `TVPVARResult`: Estimated TVP-VAR model.

# Example
```julia
data = randn(200, 2)
result = tvp_var_estimate(data; lags=1)
println("Log-likelihood: ", result.log_likelihood)
```
"""
function tvp_var_estimate(
    data::AbstractMatrix{<:Real};
    lags::Int=1,
    Q_init::Union{AbstractMatrix{<:Real},Nothing}=nothing,
    Q_scale::Float64=0.001,
    sigma_init::Union{AbstractMatrix{<:Real},Nothing}=nothing,
    initialization::Symbol=:ols,
    beta_init::Union{AbstractVector{<:Real},Nothing}=nothing,
    P_init::Union{AbstractMatrix{<:Real},Nothing}=nothing,
    diffuse_scale::Float64=1e6,
    smooth::Bool=true,
    var_names::Union{Vector{String},Nothing}=nothing
)
    data = Float64.(data)

    if ndims(data) != 2
        error("Data must be 2D, got $(ndims(data))D")
    end

    n_obs, n_vars = size(data)

    if lags < 1
        error("lags must be >= 1, got $lags")
    end

    n_params_per_eq_val = n_vars * lags + 1  # Include intercept
    state_dim = n_vars * n_params_per_eq_val
    n_obs_effective = n_obs - lags

    if n_obs_effective < 10
        error("Insufficient observations. Need at least $(lags + 10), got $n_obs")
    end

    if any(isnan, data)
        error("Data contains NaN values")
    end

    # Variable names
    if isnothing(var_names)
        var_names = ["var_$i" for i in 1:n_vars]
    elseif length(var_names) != n_vars
        error("var_names length ($(length(var_names))) must match n_vars ($n_vars)")
    end

    # Build design matrices
    Y, X_full = build_var_matrices(data, lags, true)  # include_constant=true

    # Convert to TVP-VAR format
    X_tvp = _build_tvp_regressor_matrix(Y, X_full, n_vars, n_params_per_eq_val)

    # Initialize Kalman filter
    if initialization == :ols
        beta_init_vec, P_init_mat, sigma = _initialize_from_ols(
            data, lags, n_vars, n_params_per_eq_val, state_dim
        )
    elseif initialization == :diffuse
        beta_init_vec = zeros(state_dim)
        P_init_mat = Matrix{Float64}(I, state_dim, state_dim) * diffuse_scale
        _, _, sigma = _initialize_from_ols(data, lags, n_vars, n_params_per_eq_val, state_dim)
    elseif initialization == :custom
        if isnothing(beta_init) || isnothing(P_init)
            error("beta_init and P_init must be provided for custom initialization")
        end
        beta_init_vec = Float64.(vec(beta_init))
        P_init_mat = Float64.(P_init)

        if length(beta_init_vec) != state_dim
            error("beta_init has wrong size. Expected $state_dim, got $(length(beta_init_vec))")
        end
        if size(P_init_mat) != (state_dim, state_dim)
            error("P_init has wrong shape. Expected ($state_dim, $state_dim), got $(size(P_init_mat))")
        end

        if !isnothing(sigma_init)
            sigma = Float64.(sigma_init)
        else
            _, _, sigma = _initialize_from_ols(data, lags, n_vars, n_params_per_eq_val, state_dim)
        end
    else
        error("initialization must be :ols, :diffuse, or :custom, got $initialization")
    end

    # Override sigma if provided
    if !isnothing(sigma_init)
        sigma = Float64.(sigma_init)
        if size(sigma) != (n_vars, n_vars)
            error("sigma_init has wrong shape. Expected ($n_vars, $n_vars), got $(size(sigma))")
        end
    end

    # State transition covariance Q
    if !isnothing(Q_init)
        Q = Float64.(Q_init)
        if size(Q) != (state_dim, state_dim)
            error("Q_init has wrong shape. Expected ($state_dim, $state_dim), got $(size(Q))")
        end
        eigvals_Q = eigvals(Symmetric(Q))
        if any(eigvals_Q .< -1e-10)
            error("Q_init must be positive semi-definite")
        end
    else
        Q = Matrix{Float64}(I, state_dim, state_dim) * Q_scale
    end

    # Run Kalman filter
    beta_filt, P_filt, beta_pred, P_pred, innovations, innovation_cov, kalman_gains, log_lik =
        _kalman_filter(Y, X_tvp, beta_init_vec, P_init_mat, Q, sigma, n_vars)

    # Reshape filtered coefficients: (T, state_dim) -> (T, n_vars, n_params_per_eq)
    coef_filtered = reshape(beta_filt, n_obs_effective, n_vars, n_params_per_eq_val)

    # RTS smoother
    if smooth
        beta_smooth, P_smooth = _rts_smoother(beta_filt, P_filt, beta_pred, P_pred, Q)
        coef_smoothed = reshape(beta_smooth, n_obs_effective, n_vars, n_params_per_eq_val)
    else
        coef_smoothed = nothing
        P_smooth = nothing
    end

    # Information criteria
    n_params_effective = state_dim + n_vars * (n_vars + 1) ÷ 2
    aic = -2 * log_lik + 2 * n_params_effective
    bic = -2 * log_lik + n_params_effective * log(n_obs_effective)

    return TVPVARResult(
        coef_filtered,
        coef_smoothed,
        P_filt,
        P_smooth,
        innovations,
        innovation_cov,
        kalman_gains,
        sigma,
        Q,
        log_lik,
        aic,
        bic,
        lags,
        n_vars,
        n_obs,
        n_obs_effective,
        state_dim,
        var_names,
        String(initialization)
    )
end


"""
    compute_tvp_irf(result, t; kwargs...)

Compute impulse response function at specific time point.

# Arguments
- `result::TVPVARResult`: Estimated TVP-VAR
- `t::Int`: Time index (1-indexed)
- `horizons::Int=20`: Number of IRF horizons
- `shock_idx::Int=1`: Which structural shock (1-indexed)
- `shock_size::Float64=1.0`: Size of shock
- `smoothed::Bool=true`: Use smoothed or filtered coefficients
- `orthogonalize::Bool=true`: Use Cholesky orthogonalization

# Returns
- `Matrix{Float64}`: Shape (n_vars, horizons+1). IRF for all variables.
"""
function compute_tvp_irf(
    result::TVPVARResult,
    t::Int;
    horizons::Int=20,
    shock_idx::Int=1,
    shock_size::Float64=1.0,
    smoothed::Bool=true,
    orthogonalize::Bool=true
)
    if t < 1 || t > result.n_obs_effective
        error("Time index $t out of bounds [1, $(result.n_obs_effective)]")
    end

    if shock_idx < 1 || shock_idx > result.n_vars
        error("shock_idx $shock_idx out of bounds [1, $(result.n_vars)]")
    end

    if horizons < 0
        error("horizons must be >= 0, got $horizons")
    end

    n_vars = result.n_vars
    lags = result.lags

    # Get coefficient matrix at time t
    coef = get_coefficients_at_time(result, t; smoothed=smoothed)

    # Impact matrix
    if orthogonalize
        try
            P = cholesky(Symmetric(result.sigma)).L
        catch
            # Regularize if not positive definite
            eigvals_s, eigvecs_s = eigen(Symmetric(result.sigma))
            eigvals_s = max.(eigvals_s, 1e-10)
            sigma_reg = eigvecs_s * Diagonal(eigvals_s) * eigvecs_s'
            P = cholesky(Symmetric(sigma_reg)).L
        end
    else
        P = Matrix{Float64}(I, n_vars, n_vars)
    end

    # Extract lag matrices A_1, ..., A_p
    A_matrices = Matrix{Float64}[]
    for lag in 1:lags
        A_lag = get_lag_matrix_at_time(result, t, lag; smoothed=smoothed)
        push!(A_matrices, A_lag)
    end

    # Compute IRF
    irf = zeros(n_vars, horizons + 1)

    # Initial shock: unit shock to variable shock_idx
    shock = zeros(n_vars)
    shock[shock_idx] = shock_size

    # h=0: immediate impact
    irf[:, 1] = P * shock

    # Compute VMA coefficients iteratively
    Phi = [P]  # Φ_0 = P

    for h in 1:horizons
        Phi_h = zeros(n_vars, n_vars)
        for j in 1:min(h, lags)
            Phi_h += Phi[h - j + 1] * A_matrices[j]
        end
        push!(Phi, Phi_h)
        irf[:, h + 1] = Phi_h * shock
    end

    return irf
end


"""
    compute_tvp_irf_all_times(result; kwargs...)

Compute IRF at all time points (time-varying IRF).

# Returns
- `Array{Float64,3}`: Shape (T, n_vars, horizons+1). Time-varying IRF.
"""
function compute_tvp_irf_all_times(
    result::TVPVARResult;
    horizons::Int=20,
    shock_idx::Int=1,
    shock_size::Float64=1.0,
    smoothed::Bool=true,
    orthogonalize::Bool=true
)
    T = result.n_obs_effective
    n_vars = result.n_vars

    irf_all = zeros(T, n_vars, horizons + 1)

    for t in 1:T
        irf_all[t, :, :] = compute_tvp_irf(
            result, t;
            horizons=horizons,
            shock_idx=shock_idx,
            shock_size=shock_size,
            smoothed=smoothed,
            orthogonalize=orthogonalize
        )
    end

    return irf_all
end


"""
    check_tvp_stability(coefficients, lags, n_vars)

Check VAR stability at a time point.

A VAR is stable if all eigenvalues of the companion matrix
are inside the unit circle.

# Returns
- `is_stable::Bool`: True if all eigenvalues inside unit circle
- `eigenvalues::Vector{ComplexF64}`: Eigenvalues of companion matrix
"""
function check_tvp_stability(
    coefficients::AbstractMatrix{<:Real},
    lags::Int,
    n_vars::Int
)
    companion_dim = n_vars * lags
    companion = zeros(companion_dim, companion_dim)

    # First n_vars rows: A_1, A_2, ..., A_p
    for lag in 1:lags
        start_col = (lag - 1) * n_vars + 1
        end_col = lag * n_vars
        # Skip intercept at column 1
        coef_start = 2 + (lag - 1) * n_vars
        coef_end = 1 + lag * n_vars
        companion[1:n_vars, start_col:end_col] = coefficients[:, coef_start:coef_end]
    end

    # Identity blocks below
    if lags > 1
        companion[(n_vars + 1):end, 1:(n_vars * (lags - 1))] = Matrix{Float64}(I, n_vars * (lags - 1), n_vars * (lags - 1))
    end

    eigenvalues = eigvals(companion)
    is_stable = all(abs.(eigenvalues) .< 1.0)

    return is_stable, eigenvalues
end


"""
    check_tvp_stability_all_times(result; smoothed=true)

Check VAR stability at all time points.

# Returns
- `is_stable::Vector{Bool}`: Shape (T,) boolean vector
- `max_eigenvalue_modulus::Vector{Float64}`: Maximum eigenvalue modulus at each time
"""
function check_tvp_stability_all_times(
    result::TVPVARResult;
    smoothed::Bool=true
)
    T = result.n_obs_effective
    is_stable = Vector{Bool}(undef, T)
    max_mod = Vector{Float64}(undef, T)

    for t in 1:T
        coef = get_coefficients_at_time(result, t; smoothed=smoothed)
        stable, eigvals_t = check_tvp_stability(coef, result.lags, result.n_vars)
        is_stable[t] = stable
        max_mod[t] = maximum(abs.(eigvals_t))
    end

    return is_stable, max_mod
end


"""
    coefficient_change_test(result, equation_idx, coef_idx; smoothed=true)

Test for significant coefficient change over time.

Uses a variance ratio test comparing actual coefficient variance
to expected variance under constant coefficients.

# Returns
- `variance_ratio::Float64`: Ratio of observed to expected coefficient variance
- `p_value::Float64`: Approximate p-value (chi-squared based)
"""
function coefficient_change_test(
    result::TVPVARResult,
    equation_idx::Int,
    coef_idx::Int;
    smoothed::Bool=true
)
    trajectory = coefficient_trajectory(result, equation_idx, coef_idx; smoothed=smoothed)

    # Observed variance
    obs_var = var(trajectory; corrected=true)

    # Expected variance under constant β
    n_params_per_eq_val = result.n_vars * result.lags + 1
    coef_flat_idx = (equation_idx - 1) * n_params_per_eq_val + coef_idx

    if smoothed && has_smoothed(result)
        P = result.covariance_smoothed
    else
        P = result.covariance_filtered
    end

    expected_var = mean(P[:, coef_flat_idx, coef_flat_idx])

    if expected_var < 1e-10
        return 0.0, 1.0
    end

    variance_ratio = obs_var / expected_var

    # Under null of constant coefficients, variance ratio follows chi-squared
    T = result.n_obs_effective
    test_stat = variance_ratio * T
    p_value = 1.0 - cdf(Chisq(T - 1), test_stat)

    return variance_ratio, p_value
end


# =============================================================================
# Helper Functions
# =============================================================================

"""Build TVP-VAR regressor matrices (block-diagonal format)."""
function _build_tvp_regressor_matrix(
    Y::Matrix{Float64},
    X_full::Matrix{Float64},
    n_vars::Int,
    n_params_per_eq::Int
)
    T = size(Y, 1)
    state_dim = n_vars * n_params_per_eq

    X_tvp = zeros(T, n_vars, state_dim)

    for t in 1:T
        for eq in 1:n_vars
            start_col = (eq - 1) * n_params_per_eq + 1
            end_col = eq * n_params_per_eq
            X_tvp[t, eq, start_col:end_col] = X_full[t, :]
        end
    end

    return X_tvp
end


"""Kalman filter for TVP-VAR."""
function _kalman_filter(
    Y::Matrix{Float64},
    X::Array{Float64,3},
    beta_init::Vector{Float64},
    P_init::Matrix{Float64},
    Q::Matrix{Float64},
    sigma::Matrix{Float64},
    n_vars::Int
)
    T = size(Y, 1)
    state_dim = length(beta_init)

    # Storage
    beta_filt = zeros(T, state_dim)
    P_filt = zeros(T, state_dim, state_dim)
    beta_pred = zeros(T, state_dim)
    P_pred = zeros(T, state_dim, state_dim)
    innovations = zeros(T, n_vars)
    innovation_cov = zeros(T, n_vars, n_vars)
    kalman_gains = zeros(T, state_dim, n_vars)

    log_lik = 0.0
    log_2pi = log(2π)

    # Initialize
    beta_curr = copy(beta_init)
    P_curr = copy(P_init)

    for t in 1:T
        # === Prediction step ===
        beta_pred_t = beta_curr  # Random walk
        P_pred_t = P_curr + Q

        beta_pred[t, :] = beta_pred_t
        P_pred[t, :, :] = P_pred_t

        # === Update step ===
        X_t = X[t, :, :]  # Shape: (n_vars, state_dim)

        # Innovation
        y_pred = X_t * beta_pred_t
        v_t = Y[t, :] - y_pred
        innovations[t, :] = v_t

        # Innovation covariance
        F_t = X_t * P_pred_t * X_t' + sigma
        innovation_cov[t, :, :] = F_t

        # Kalman gain (try-catch returns value to ensure F_inv is in scope)
        F_inv = try
            F_t \ I(n_vars)
        catch
            F_t_reg = F_t + I(n_vars) * 1e-6
            F_t_reg \ I(n_vars)
        end

        K_t = P_pred_t * X_t' * F_inv
        kalman_gains[t, :, :] = K_t

        # State update
        beta_filt_t = beta_pred_t + K_t * v_t
        beta_filt[t, :] = beta_filt_t

        # Covariance update (Joseph form)
        P_filt_t = _joseph_form_update(P_pred_t, K_t, X_t, sigma)
        P_filt[t, :, :] = P_filt_t

        # Log-likelihood contribution
        sign_det, logdet = logabsdet(F_t)
        if sign_det <= 0
            logdet = sum(log.(max.(diag(F_t), 1e-10)))
        end

        quad_form = dot(v_t, F_inv * v_t)
        log_lik_t = -0.5 * (n_vars * log_2pi + logdet + quad_form)
        log_lik += log_lik_t

        # Update for next iteration
        beta_curr = beta_filt_t
        P_curr = P_filt_t
    end

    return beta_filt, P_filt, beta_pred, P_pred, innovations, innovation_cov, kalman_gains, log_lik
end


"""Rauch-Tung-Striebel backward smoother."""
function _rts_smoother(
    beta_filt::Matrix{Float64},
    P_filt::Array{Float64,3},
    beta_pred::Matrix{Float64},
    P_pred::Array{Float64,3},
    Q::Matrix{Float64}
)
    T, state_dim = size(beta_filt)

    beta_smooth = zeros(T, state_dim)
    P_smooth = zeros(T, state_dim, state_dim)

    # Initialize at T (last time point)
    beta_smooth[T, :] = beta_filt[T, :]
    P_smooth[T, :, :] = P_filt[T, :, :]

    # Backward recursion
    for t in (T - 1):-1:1
        P_pred_next = P_pred[t + 1, :, :]

        try
            J_t = P_filt[t, :, :] * (P_pred_next \ I(state_dim))'
        catch
            P_pred_reg = P_pred_next + I(state_dim) * 1e-6
            J_t = P_filt[t, :, :] * (P_pred_reg \ I(state_dim))'
        end

        beta_smooth[t, :] = beta_filt[t, :] + J_t * (beta_smooth[t + 1, :] - beta_pred[t + 1, :])
        P_smooth[t, :, :] = P_filt[t, :, :] + J_t * (P_smooth[t + 1, :, :] - P_pred_next) * J_t'
    end

    return beta_smooth, P_smooth
end


"""Joseph form covariance update for numerical stability."""
function _joseph_form_update(
    P_pred::Matrix{Float64},
    K::Matrix{Float64},
    X_t::Matrix{Float64},
    sigma::Matrix{Float64}
)
    state_dim = size(P_pred, 1)
    I_mat = Matrix{Float64}(I, state_dim, state_dim)

    # (I - K X)
    I_KX = I_mat - K * X_t

    # Joseph form
    P_filt = I_KX * P_pred * I_KX' + K * sigma * K'

    # Ensure symmetry
    P_filt = 0.5 * (P_filt + P_filt')

    return P_filt
end


"""Initialize TVP-VAR from OLS VAR estimates."""
function _initialize_from_ols(
    data::Matrix{Float64},
    lags::Int,
    n_vars::Int,
    n_params_per_eq::Int,
    state_dim::Int
)
    # Estimate OLS VAR
    var_result = var_estimate(data; lags=lags, include_constant=true)

    # Vectorize coefficients (row by row)
    beta_init = vec(var_result.coefficients)

    # Initial covariance: scaled identity
    resid_var = var(vec(var_result.residuals))
    P_init = Matrix{Float64}(I, state_dim, state_dim) * resid_var * 10

    # Observation covariance from OLS
    sigma = var_result.sigma

    return beta_init, P_init, sigma
end


end  # module TVPVAR
