"""
Cointegration Tests for Time Series

Session 147: Johansen and Engle-Granger cointegration tests for Julia parity.

Cointegration occurs when a linear combination of non-stationary I(1) series
is stationary. This implies a long-run equilibrium relationship.

The Johansen procedure tests for the number of cointegrating relationships
(cointegration rank r) using both trace and maximum eigenvalue statistics.

References:
- Johansen (1988). "Statistical analysis of cointegration vectors."
- Johansen (1991). "Estimation and hypothesis testing of cointegration vectors."
- MacKinnon-Haug-Michelis (1999). "Numerical distribution functions for
  unit root and cointegration tests."
"""
module Cointegration

using LinearAlgebra
using Statistics

using ..TimeSeriesTypes
using ..Stationarity: adf_test

export johansen_test, engle_granger_test


# Critical values for Johansen tests (MacKinnon-Haug-Michelis 1999)
# Format: JOHANSEN_CV[det_order][n_vars][rank] = Dict("90%" => ..., "95%" => ..., "99%" => ...)

# Trace statistic critical values
const JOHANSEN_TRACE_CV = Dict(
    # det_order = 0 (restricted constant, no trend - most common)
    0 => Dict(
        2 => Dict(
            0 => Dict("90%" => 13.43, "95%" => 15.49, "99%" => 19.94),
            1 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
        3 => Dict(
            0 => Dict("90%" => 27.07, "95%" => 29.80, "99%" => 35.46),
            1 => Dict("90%" => 13.43, "95%" => 15.49, "99%" => 19.94),
            2 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
        4 => Dict(
            0 => Dict("90%" => 44.49, "95%" => 47.86, "99%" => 54.68),
            1 => Dict("90%" => 27.07, "95%" => 29.80, "99%" => 35.46),
            2 => Dict("90%" => 13.43, "95%" => 15.49, "99%" => 19.94),
            3 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
        5 => Dict(
            0 => Dict("90%" => 65.82, "95%" => 69.82, "99%" => 77.82),
            1 => Dict("90%" => 44.49, "95%" => 47.86, "99%" => 54.68),
            2 => Dict("90%" => 27.07, "95%" => 29.80, "99%" => 35.46),
            3 => Dict("90%" => 13.43, "95%" => 15.49, "99%" => 19.94),
            4 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
        6 => Dict(
            0 => Dict("90%" => 91.11, "95%" => 95.75, "99%" => 104.96),
            1 => Dict("90%" => 65.82, "95%" => 69.82, "99%" => 77.82),
            2 => Dict("90%" => 44.49, "95%" => 47.86, "99%" => 54.68),
            3 => Dict("90%" => 27.07, "95%" => 29.80, "99%" => 35.46),
            4 => Dict("90%" => 13.43, "95%" => 15.49, "99%" => 19.94),
            5 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
    ),
    # det_order = -1 (no constant, no trend)
    -1 => Dict(
        2 => Dict(
            0 => Dict("90%" => 7.52, "95%" => 9.16, "99%" => 12.97),
            1 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
        3 => Dict(
            0 => Dict("90%" => 17.98, "95%" => 20.26, "99%" => 25.08),
            1 => Dict("90%" => 7.52, "95%" => 9.16, "99%" => 12.97),
            2 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
        4 => Dict(
            0 => Dict("90%" => 32.09, "95%" => 35.19, "99%" => 41.20),
            1 => Dict("90%" => 17.98, "95%" => 20.26, "99%" => 25.08),
            2 => Dict("90%" => 7.52, "95%" => 9.16, "99%" => 12.97),
            3 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
        5 => Dict(
            0 => Dict("90%" => 50.52, "95%" => 54.07, "99%" => 61.27),
            1 => Dict("90%" => 32.09, "95%" => 35.19, "99%" => 41.20),
            2 => Dict("90%" => 17.98, "95%" => 20.26, "99%" => 25.08),
            3 => Dict("90%" => 7.52, "95%" => 9.16, "99%" => 12.97),
            4 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
        6 => Dict(
            0 => Dict("90%" => 72.77, "95%" => 76.97, "99%" => 85.34),
            1 => Dict("90%" => 50.52, "95%" => 54.07, "99%" => 61.27),
            2 => Dict("90%" => 32.09, "95%" => 35.19, "99%" => 41.20),
            3 => Dict("90%" => 17.98, "95%" => 20.26, "99%" => 25.08),
            4 => Dict("90%" => 7.52, "95%" => 9.16, "99%" => 12.97),
            5 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
    ),
    # det_order = 1 (unrestricted constant)
    1 => Dict(
        2 => Dict(
            0 => Dict("90%" => 17.98, "95%" => 20.26, "99%" => 25.08),
            1 => Dict("90%" => 7.52, "95%" => 9.16, "99%" => 12.97),
        ),
        3 => Dict(
            0 => Dict("90%" => 32.09, "95%" => 35.19, "99%" => 41.20),
            1 => Dict("90%" => 17.98, "95%" => 20.26, "99%" => 25.08),
            2 => Dict("90%" => 7.52, "95%" => 9.16, "99%" => 12.97),
        ),
        4 => Dict(
            0 => Dict("90%" => 50.52, "95%" => 54.07, "99%" => 61.27),
            1 => Dict("90%" => 32.09, "95%" => 35.19, "99%" => 41.20),
            2 => Dict("90%" => 17.98, "95%" => 20.26, "99%" => 25.08),
            3 => Dict("90%" => 7.52, "95%" => 9.16, "99%" => 12.97),
        ),
        5 => Dict(
            0 => Dict("90%" => 72.77, "95%" => 76.97, "99%" => 85.34),
            1 => Dict("90%" => 50.52, "95%" => 54.07, "99%" => 61.27),
            2 => Dict("90%" => 32.09, "95%" => 35.19, "99%" => 41.20),
            3 => Dict("90%" => 17.98, "95%" => 20.26, "99%" => 25.08),
            4 => Dict("90%" => 7.52, "95%" => 9.16, "99%" => 12.97),
        ),
        6 => Dict(
            0 => Dict("90%" => 98.42, "95%" => 103.18, "99%" => 112.74),
            1 => Dict("90%" => 72.77, "95%" => 76.97, "99%" => 85.34),
            2 => Dict("90%" => 50.52, "95%" => 54.07, "99%" => 61.27),
            3 => Dict("90%" => 32.09, "95%" => 35.19, "99%" => 41.20),
            4 => Dict("90%" => 17.98, "95%" => 20.26, "99%" => 25.08),
            5 => Dict("90%" => 7.52, "95%" => 9.16, "99%" => 12.97),
        ),
    ),
)

# Maximum eigenvalue critical values
const JOHANSEN_MAX_EIGEN_CV = Dict(
    # det_order = 0 (restricted constant)
    0 => Dict(
        2 => Dict(
            0 => Dict("90%" => 12.30, "95%" => 14.26, "99%" => 18.52),
            1 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
        3 => Dict(
            0 => Dict("90%" => 18.89, "95%" => 21.13, "99%" => 25.86),
            1 => Dict("90%" => 12.30, "95%" => 14.26, "99%" => 18.52),
            2 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
        4 => Dict(
            0 => Dict("90%" => 25.12, "95%" => 27.58, "99%" => 32.72),
            1 => Dict("90%" => 18.89, "95%" => 21.13, "99%" => 25.86),
            2 => Dict("90%" => 12.30, "95%" => 14.26, "99%" => 18.52),
            3 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
        5 => Dict(
            0 => Dict("90%" => 31.24, "95%" => 33.88, "99%" => 39.37),
            1 => Dict("90%" => 25.12, "95%" => 27.58, "99%" => 32.72),
            2 => Dict("90%" => 18.89, "95%" => 21.13, "99%" => 25.86),
            3 => Dict("90%" => 12.30, "95%" => 14.26, "99%" => 18.52),
            4 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
        6 => Dict(
            0 => Dict("90%" => 37.28, "95%" => 40.07, "99%" => 45.87),
            1 => Dict("90%" => 31.24, "95%" => 33.88, "99%" => 39.37),
            2 => Dict("90%" => 25.12, "95%" => 27.58, "99%" => 32.72),
            3 => Dict("90%" => 18.89, "95%" => 21.13, "99%" => 25.86),
            4 => Dict("90%" => 12.30, "95%" => 14.26, "99%" => 18.52),
            5 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
    ),
    # det_order = -1 (no constant)
    -1 => Dict(
        2 => Dict(
            0 => Dict("90%" => 6.69, "95%" => 8.18, "99%" => 11.65),
            1 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
        3 => Dict(
            0 => Dict("90%" => 12.78, "95%" => 14.59, "99%" => 18.78),
            1 => Dict("90%" => 6.69, "95%" => 8.18, "99%" => 11.65),
            2 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
        4 => Dict(
            0 => Dict("90%" => 18.63, "95%" => 20.78, "99%" => 25.42),
            1 => Dict("90%" => 12.78, "95%" => 14.59, "99%" => 18.78),
            2 => Dict("90%" => 6.69, "95%" => 8.18, "99%" => 11.65),
            3 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
        5 => Dict(
            0 => Dict("90%" => 24.16, "95%" => 26.53, "99%" => 31.73),
            1 => Dict("90%" => 18.63, "95%" => 20.78, "99%" => 25.42),
            2 => Dict("90%" => 12.78, "95%" => 14.59, "99%" => 18.78),
            3 => Dict("90%" => 6.69, "95%" => 8.18, "99%" => 11.65),
            4 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
        6 => Dict(
            0 => Dict("90%" => 29.57, "95%" => 32.12, "99%" => 37.61),
            1 => Dict("90%" => 24.16, "95%" => 26.53, "99%" => 31.73),
            2 => Dict("90%" => 18.63, "95%" => 20.78, "99%" => 25.42),
            3 => Dict("90%" => 12.78, "95%" => 14.59, "99%" => 18.78),
            4 => Dict("90%" => 6.69, "95%" => 8.18, "99%" => 11.65),
            5 => Dict("90%" => 2.71, "95%" => 3.84, "99%" => 6.63),
        ),
    ),
    # det_order = 1 (unrestricted constant)
    1 => Dict(
        2 => Dict(
            0 => Dict("90%" => 12.78, "95%" => 14.59, "99%" => 18.78),
            1 => Dict("90%" => 6.69, "95%" => 8.18, "99%" => 11.65),
        ),
        3 => Dict(
            0 => Dict("90%" => 18.63, "95%" => 20.78, "99%" => 25.42),
            1 => Dict("90%" => 12.78, "95%" => 14.59, "99%" => 18.78),
            2 => Dict("90%" => 6.69, "95%" => 8.18, "99%" => 11.65),
        ),
        4 => Dict(
            0 => Dict("90%" => 24.16, "95%" => 26.53, "99%" => 31.73),
            1 => Dict("90%" => 18.63, "95%" => 20.78, "99%" => 25.42),
            2 => Dict("90%" => 12.78, "95%" => 14.59, "99%" => 18.78),
            3 => Dict("90%" => 6.69, "95%" => 8.18, "99%" => 11.65),
        ),
        5 => Dict(
            0 => Dict("90%" => 29.57, "95%" => 32.12, "99%" => 37.61),
            1 => Dict("90%" => 24.16, "95%" => 26.53, "99%" => 31.73),
            2 => Dict("90%" => 18.63, "95%" => 20.78, "99%" => 25.42),
            3 => Dict("90%" => 12.78, "95%" => 14.59, "99%" => 18.78),
            4 => Dict("90%" => 6.69, "95%" => 8.18, "99%" => 11.65),
        ),
        6 => Dict(
            0 => Dict("90%" => 34.87, "95%" => 37.61, "99%" => 43.27),
            1 => Dict("90%" => 29.57, "95%" => 32.12, "99%" => 37.61),
            2 => Dict("90%" => 24.16, "95%" => 26.53, "99%" => 31.73),
            3 => Dict("90%" => 18.63, "95%" => 20.78, "99%" => 25.42),
            4 => Dict("90%" => 12.78, "95%" => 14.59, "99%" => 18.78),
            5 => Dict("90%" => 6.69, "95%" => 8.18, "99%" => 11.65),
        ),
    ),
)


"""
    johansen_test(data; lags=1, det_order=0, alpha=0.05)

Johansen cointegration test.

Tests for cointegration rank in a VAR system. Determines the number r
of cointegrating relationships among n variables.

# Arguments
- `data::Matrix{Float64}`: Shape (n_obs, n_vars) multivariate time series
- `lags::Int`: Number of VAR lags (p). The VECM uses p-1 lagged differences.
- `det_order::Int`: Deterministic terms specification:
  - -1: No constant, no trend
  - 0: Restricted constant (most common)
  - 1: Unrestricted constant
- `alpha::Float64`: Significance level for rank determination

# Returns
- `JohansenResult`: Test results including rank, test statistics, eigenvalues,
  and cointegrating vectors

# Example
```julia
using Random
Random.seed!(42)
n = 200
trend = cumsum(randn(n))  # Common trend
y1 = trend + randn(n) * 0.5
y2 = 0.5 * trend + randn(n) * 0.5
data = hcat(y1, y2)
result = johansen_test(data, lags=2)
println("Cointegration rank: \$(result.rank)")
```

# Notes
The Johansen procedure works by reformulating the VAR in VECM form:
    ΔY_t = Π Y_{t-1} + Γ_1 ΔY_{t-1} + ... + Γ_{p-1} ΔY_{t-p+1} + ε_t

where Π = αβ' contains the long-run information:
- α: Adjustment coefficients (speed of adjustment to equilibrium)
- β: Cointegrating vectors (long-run relationships)

The rank of Π equals the number of cointegrating relationships.

# References
Johansen (1988, 1991). Statistical analysis of cointegration vectors.
"""
function johansen_test(
    data::AbstractMatrix{<:Real};
    lags::Int=1,
    det_order::Int=0,
    alpha::Float64=0.05,
)
    data = Float64.(data)
    n_obs, n_vars = size(data)

    if n_vars < 2
        error("Need at least 2 variables for cointegration, got $n_vars")
    end

    if n_vars > 6
        error("Critical values only available for up to 6 variables, got $n_vars")
    end

    if n_obs < 3 * n_vars + lags
        error("Insufficient observations ($n_obs) for $n_vars variables and $lags lags")
    end

    if !(det_order in [-1, 0, 1])
        error("det_order must be -1, 0, or 1, got $det_order")
    end

    if lags < 1
        error("lags must be >= 1, got $lags")
    end

    # Step 1: Compute first differences and levels
    dy = diff(data, dims=1)  # ΔY_t (T-1, n_vars)
    y_lag = data[1:end-1, :]  # Y_{t-1} (T-1, n_vars)

    # Effective sample size
    T = size(dy, 1) - lags + 1

    # Trim to account for lags
    dy_trimmed = dy[lags:end, :]  # (T, n_vars)
    y_lag_trimmed = y_lag[lags:end, :]  # (T, n_vars)

    # Step 2: Build regressors for lagged differences
    X_parts = Matrix{Float64}[]

    # Add lagged differences ΔY_{t-1}, ..., ΔY_{t-p+1}
    for i in 1:(lags-1)
        lag_diff = dy[lags-i:end-i, :]
        push!(X_parts, lag_diff)
    end

    # Add deterministic terms
    if det_order >= 0
        # Constant
        push!(X_parts, ones(T, 1))
    end

    X = if !isempty(X_parts)
        hcat(X_parts...)
    else
        nothing
    end

    # Step 3: Reduced rank regression via canonical correlations
    if X !== nothing && size(X, 2) > 0
        # Residuals from regressing ΔY on X
        beta0 = X \ dy_trimmed
        R0 = dy_trimmed - X * beta0

        # Residuals from regressing Y_{t-1} on X
        beta1 = X \ y_lag_trimmed
        R1 = y_lag_trimmed - X * beta1
    else
        # No regressors - use centered data
        R0 = dy_trimmed .- mean(dy_trimmed, dims=1)
        R1 = y_lag_trimmed .- mean(y_lag_trimmed, dims=1)
    end

    # Step 4: Form moment matrices
    S00 = R0' * R0 / T
    S11 = R1' * R1 / T
    S01 = R0' * R1 / T
    S10 = R1' * R0 / T

    # Step 5: Solve generalized eigenvalue problem
    S00_inv = try
        inv(S00)
    catch
        pinv(S00)
    end

    S11_inv = try
        inv(S11)
    catch
        pinv(S11)
    end

    # Matrix for eigenvalue problem
    M = S11_inv * S10 * S00_inv * S01

    # Solve eigenvalue problem
    eig_result = eigen(M)
    eigenvalues = real.(eig_result.values)
    eigenvectors = real.(eig_result.vectors)

    # Sort in descending order
    sort_idx = sortperm(eigenvalues, rev=true)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Ensure eigenvalues are in [0, 1]
    eigenvalues = clamp.(eigenvalues, 0, 1 - 1e-10)

    # Step 6: Compute trace and max eigenvalue statistics
    trace_stats = zeros(n_vars)
    max_eigen_stats = zeros(n_vars)

    for r in 0:(n_vars-1)
        # Trace statistic: -T * sum(ln(1 - λ_i)) for i = r+1 to n
        trace_stats[r+1] = -T * sum(log.(1 .- eigenvalues[r+1:end]))

        # Max eigenvalue statistic: -T * ln(1 - λ_{r+1})
        max_eigen_stats[r+1] = -T * log(1 - eigenvalues[r+1])
    end

    # Step 7: Get critical values and determine rank
    trace_crit = zeros(n_vars)
    max_eigen_crit = zeros(n_vars)
    trace_pvalues = zeros(n_vars)
    max_eigen_pvalues = zeros(n_vars)

    # Get critical value level string
    cv_level = if alpha <= 0.01
        "99%"
    elseif alpha <= 0.05
        "95%"
    else
        "90%"
    end

    for r in 0:(n_vars-1)
        trace_cv_dict = get(get(get(JOHANSEN_TRACE_CV, det_order, Dict()), n_vars, Dict()), r, Dict())
        max_cv_dict = get(get(get(JOHANSEN_MAX_EIGEN_CV, det_order, Dict()), n_vars, Dict()), r, Dict())

        trace_crit[r+1] = get(trace_cv_dict, cv_level, NaN)
        max_eigen_crit[r+1] = get(max_cv_dict, cv_level, NaN)

        # Approximate p-values
        trace_pvalues[r+1] = johansen_pvalue(trace_stats[r+1], trace_cv_dict)
        max_eigen_pvalues[r+1] = johansen_pvalue(max_eigen_stats[r+1], max_cv_dict)
    end

    # Determine rank by trace test (sequential testing)
    rank = 0
    for r in 0:(n_vars-1)
        if trace_stats[r+1] > trace_crit[r+1]
            rank = r + 1
        else
            break
        end
    end

    # Step 8: Extract adjustment coefficients
    beta = eigenvectors

    # Compute adjustment coefficients
    alpha_mat = try
        beta_S11_beta_inv = inv(beta' * S11 * beta)
        S01 * beta * beta_S11_beta_inv
    catch
        zeros(n_vars, n_vars)
    end

    JohansenResult(
        rank,
        trace_stats,
        trace_crit,
        trace_pvalues,
        max_eigen_stats,
        max_eigen_crit,
        max_eigen_pvalues,
        eigenvalues,
        beta,
        alpha_mat,
        lags,
        T,
        n_vars,
        det_order,
        alpha,
    )
end


"""Compute approximate p-value for Johansen statistic."""
function johansen_pvalue(stat::Float64, cv_dict::Dict)
    if isempty(cv_dict)
        return NaN
    end

    cv_90 = get(cv_dict, "90%", NaN)
    cv_95 = get(cv_dict, "95%", NaN)
    cv_99 = get(cv_dict, "99%", NaN)

    if isnan(cv_90)
        return NaN
    end

    if stat <= cv_90
        return 0.15  # Above 10%
    elseif stat <= cv_95
        return 0.10 - 0.05 * (stat - cv_90) / (cv_95 - cv_90)
    elseif stat <= cv_99
        return 0.05 - 0.04 * (stat - cv_95) / (cv_99 - cv_95)
    else
        return 0.005
    end
end


"""
    engle_granger_test(y, x; alpha=0.05)

Engle-Granger two-step cointegration test.

Simpler alternative to Johansen for bivariate case.

# Arguments
- `y::Vector{Float64}`: Dependent variable (1D)
- `x::AbstractVecOrMat{Float64}`: Independent variable(s) (1D or 2D)
- `alpha::Float64`: Significance level

# Returns
- `EngleGrangerResult`: Contains cointegrating regression coefficients, residuals,
  ADF test on residuals, and cointegration decision

# Example
```julia
using Random
Random.seed!(42)
n = 200
trend = cumsum(randn(n))
y = trend + randn(n) * 0.3
x = 2 * trend + randn(n) * 0.3
result = engle_granger_test(y, x)
println("Cointegrated: \$(result.is_cointegrated)")
```

# References
Engle & Granger (1987). "Co-integration and error correction:
Representation, estimation, and testing." Econometrica 55: 251-276.
"""
function engle_granger_test(
    y::AbstractVector{<:Real},
    x::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}};
    alpha::Float64=0.05,
)
    y = Float64.(vec(y))
    x = Float64.(x)

    if ndims(x) == 1
        x = reshape(x, :, 1)
    end

    n = length(y)

    if size(x, 1) != n
        error("Length mismatch: y ($n) vs x ($(size(x, 1)))")
    end

    # Step 1: Cointegrating regression
    # y_t = β_0 + β_1 x_t + u_t
    X = hcat(ones(n), x)
    beta = X \ y
    residuals = y - X * beta

    # Step 2: Test residuals for stationarity
    adf_result = adf_test(residuals; regression="n", alpha=alpha)

    # Cointegration critical values (MacKinnon 1991)
    coint_cv = Dict("1%" => -3.90, "5%" => -3.34, "10%" => -3.04)

    # Determine cointegration based on cointegration-specific CVs
    cv = if alpha <= 0.01
        coint_cv["1%"]
    elseif alpha <= 0.05
        coint_cv["5%"]
    else
        coint_cv["10%"]
    end

    is_cointegrated = adf_result.statistic < cv

    EngleGrangerResult(beta, residuals, adf_result, coint_cv, is_cointegrated)
end

end # module
