"""
Stationarity Tests for Time Series

Session 147: ADF, KPSS, and Phillips-Perron tests for Julia parity.

The ADF test examines:
    H0: Series has a unit root (non-stationary)
    H1: Series is stationary

The KPSS test examines:
    H0: Series is trend-stationary
    H1: Series has unit root (non-stationary)

The PP test examines (like ADF):
    H0: Series has a unit root (non-stationary)
    H1: Series is stationary
"""
module Stationarity

using LinearAlgebra
using Statistics

using ..TimeSeriesTypes

export adf_test, kpss_test, phillips_perron_test, confirmatory_stationarity_test
export difference_series, check_stationarity


# ADF critical values (MacKinnon 1994, 2010) - Asymptotic values
const ADF_CRITICAL_VALUES = Dict(
    "n" => Dict("1%" => -2.566, "5%" => -1.941, "10%" => -1.617),
    "c" => Dict("1%" => -3.430, "5%" => -2.862, "10%" => -2.567),
    "ct" => Dict("1%" => -3.960, "5%" => -3.410, "10%" => -3.127),
)

# KPSS critical values (Kwiatkowski et al. 1992, Table 1)
# Note: KPSS rejects stationarity if stat > CV (opposite of ADF)
const KPSS_CRITICAL_VALUES = Dict(
    "c" => Dict("10%" => 0.347, "5%" => 0.463, "2.5%" => 0.574, "1%" => 0.739),
    "ct" => Dict("10%" => 0.119, "5%" => 0.146, "2.5%" => 0.176, "1%" => 0.216),
)


"""
    adf_test(series; max_lags=nothing, regression="c", alpha=0.05, autolag="aic")

Augmented Dickey-Fuller test for unit root.

Tests H0: series has unit root (non-stationary)
vs H1: series is stationary.

# Arguments
- `series::Vector{Float64}`: 1D time series
- `max_lags::Union{Int,Nothing}`: Maximum lags. If nothing, uses Schwert (1989) rule.
- `regression::String`: Type of regression:
  - "n": no constant, no trend
  - "c": constant only (default)
  - "ct": constant and trend
- `alpha::Float64`: Significance level for determining stationarity
- `autolag::String`: Lag selection method ("aic", "bic", or "fixed")

# Returns
- `ADFResult`: Test results including statistic, p-value, and stationarity decision

# Example
```julia
using Random
Random.seed!(42)
y_stat = randn(200)
result = adf_test(y_stat)
println("Stationary: \$(result.is_stationary)")
```
"""
function adf_test(
    series::AbstractVector{<:Real};
    max_lags::Union{Int,Nothing}=nothing,
    regression::String="c",
    alpha::Float64=0.05,
    autolag::String="aic",
)
    series = Float64.(series)
    n = length(series)

    if n < 10
        error("Series too short for ADF test (n=$n, need >= 10)")
    end

    if !(regression in ["n", "c", "ct"])
        error("regression must be 'n', 'c', or 'ct', got '$regression'")
    end

    # Default max_lags (Schwert 1989)
    if max_lags === nothing
        max_lags = Int(ceil(12 * (n / 100)^0.25))
    end
    max_lags = min(max_lags, n ÷ 3 - 1)

    # Select optimal lag
    if autolag in ["aic", "bic"]
        optimal_lag = select_adf_lag(series, max_lags, regression, autolag)
    else
        optimal_lag = max_lags
    end

    # Run ADF test with selected lag
    adf_stat, n_used = adf_statistic(series, optimal_lag, regression)

    # Get critical values
    critical_values = get(ADF_CRITICAL_VALUES, regression, ADF_CRITICAL_VALUES["c"])

    # Compute approximate p-value
    p_value = adf_pvalue(adf_stat, regression, n_used)

    # Determine stationarity (reject unit root if stat < critical value)
    is_stationary = p_value < alpha

    ADFResult(
        statistic=adf_stat,
        p_value=p_value,
        lags=optimal_lag,
        n_obs=n,
        critical_values=critical_values,
        is_stationary=is_stationary,
        regression=regression,
        alpha=alpha,
    )
end


"""Compute ADF test statistic."""
function adf_statistic(series::Vector{Float64}, lags::Int, regression::String)
    n = length(series)

    # First difference
    dy = diff(series)

    # Lagged level
    y_lag = series[lags+1:end-1]

    # Trimmed differences
    n_obs = length(dy) - lags
    dy_trimmed = dy[lags+1:end]

    # Build design matrix
    X_parts = Vector{Float64}[]

    # Constant
    if regression in ["c", "ct"]
        push!(X_parts, ones(n_obs))
    end

    # Time trend
    if regression == "ct"
        trend = Float64.(lags+1:lags+n_obs)
        push!(X_parts, trend)
    end

    # Lagged level (y_{t-1})
    push!(X_parts, y_lag[1:n_obs])

    # Lagged differences
    for i in 1:lags
        dy_lag = dy[lags-i+1:end-i]
        if length(dy_lag) > n_obs
            dy_lag = dy_lag[1:n_obs]
        end
        push!(X_parts, dy_lag)
    end

    X = hcat(X_parts...)
    y = dy_trimmed[1:n_obs]

    # OLS estimation
    beta = X \ y
    residuals = y - X * beta

    # Standard error of gamma coefficient
    gamma_idx = if regression == "n"
        1
    elseif regression == "c"
        2
    else  # ct
        3
    end

    # Compute variance-covariance matrix
    mse = sum(residuals.^2) / (n_obs - size(X, 2))
    XtX_inv = inv(X' * X)
    se_gamma = sqrt(mse * XtX_inv[gamma_idx, gamma_idx])

    # t-statistic for gamma
    gamma = beta[gamma_idx]
    adf_stat = if se_gamma > 0
        gamma / se_gamma
    else
        0.0
    end

    return adf_stat, n_obs
end


"""Select optimal lag for ADF test using information criterion."""
function select_adf_lag(series::Vector{Float64}, max_lags::Int, regression::String, criterion::String)
    best_lag = 0
    best_ic = Inf

    for lag in 0:max_lags
        try
            dy = diff(series)
            n_obs = length(dy) - lag
            if n_obs < 5
                continue
            end

            y_lag = series[lag+1:end-1][1:n_obs]
            dy_trimmed = dy[lag+1:end][1:n_obs]

            X_parts = Vector{Float64}[]
            if regression in ["c", "ct"]
                push!(X_parts, ones(n_obs))
            end
            if regression == "ct"
                push!(X_parts, Float64.(lag+1:lag+n_obs))
            end
            push!(X_parts, y_lag)
            for i in 1:lag
                dy_lag = dy[lag-i+1:end-i][1:n_obs]
                push!(X_parts, dy_lag)
            end

            X = hcat(X_parts...)
            beta = X \ dy_trimmed
            residuals = dy_trimmed - X * beta
            rss = sum(residuals.^2)
            sigma2 = rss / n_obs
            k = size(X, 2)

            ic = if criterion == "aic"
                n_obs * log(sigma2) + 2 * k
            else  # bic
                n_obs * log(sigma2) + k * log(n_obs)
            end

            if ic < best_ic
                best_ic = ic
                best_lag = lag
            end
        catch
            continue
        end
    end

    return best_lag
end


"""Compute approximate p-value for ADF statistic."""
function adf_pvalue(tau::Float64, regression::String, n::Int)
    cv = get(ADF_CRITICAL_VALUES, regression, ADF_CRITICAL_VALUES["c"])
    cv_1 = cv["1%"]
    cv_5 = cv["5%"]
    cv_10 = cv["10%"]

    if tau < -10
        return 0.0001
    elseif tau > 0
        return 0.99
    elseif tau <= cv_1
        return 0.005
    elseif tau <= cv_5
        return 0.01 + 0.04 * (tau - cv_1) / (cv_5 - cv_1)
    elseif tau <= cv_10
        return 0.05 + 0.05 * (tau - cv_5) / (cv_10 - cv_5)
    else
        return 0.10 + 0.45 * (1 - exp(-(tau - cv_10) * 0.5))
    end
end


"""
    kpss_test(series; regression="c", lags=nothing, alpha=0.05)

KPSS test for stationarity.

Tests H0: series is trend-stationary (stationary around deterministic trend)
vs H1: series has unit root (non-stationary).

IMPORTANT: Opposite null hypothesis from ADF test!
- KPSS: H0 = stationary (low stat = stationary)
- ADF: H0 = unit root (low stat = stationary)

# Arguments
- `series::Vector{Float64}`: 1D time series
- `regression::String`: Type of regression:
  - "c": constant only (level stationarity)
  - "ct": constant and trend (trend stationarity)
- `lags::Union{Int,Nothing}`: Lags for Newey-West. If nothing, uses Schwert rule.
- `alpha::Float64`: Significance level for determining stationarity

# Returns
- `KPSSResult`: Test results including statistic, p-value, and stationarity decision

# Example
```julia
using Random
Random.seed!(42)
y_stat = randn(200)
result = kpss_test(y_stat)
println("Stationary: \$(result.is_stationary)")  # Should be True
```

# Notes
Use KPSS with ADF for confirmatory testing:
- ADF rejects + KPSS fails to reject -> stationary
- ADF fails to reject + KPSS rejects -> non-stationary
- Both reject or both fail to reject -> inconclusive

# References
Kwiatkowski et al. (1992). "Testing the null hypothesis of stationarity
against the alternative of a unit root." J. Econometrics 54: 159-178.
"""
function kpss_test(
    series::AbstractVector{<:Real};
    regression::String="c",
    lags::Union{Int,Nothing}=nothing,
    alpha::Float64=0.05,
)
    series = Float64.(series)
    n = length(series)

    if n < 10
        error("Series too short for KPSS test (n=$n, need >= 10)")
    end

    if !(regression in ["c", "ct"])
        error("regression must be 'c' or 'ct', got '$regression'")
    end

    # Default lags (Schwert rule)
    if lags === nothing
        lags = Int(ceil(4 * (n / 100)^0.25))
    end

    # Step 1: Fit OLS regression to get residuals
    X = if regression == "c"
        ones(n, 1)
    else  # ct
        t = Float64.(1:n)
        hcat(ones(n), t)
    end

    beta = X \ series
    residuals = series - X * beta

    # Step 2: Compute partial sums S_t = sum_{i=1}^{t} e_i
    partial_sums = cumsum(residuals)

    # Step 3: Compute Newey-West long-run variance with Bartlett kernel
    gamma_0 = sum(residuals.^2) / n

    gamma_sum = 0.0
    for s in 1:lags
        weight = 1 - s / (lags + 1)  # Bartlett kernel
        gamma_s = sum(residuals[s+1:end] .* residuals[1:end-s]) / n
        gamma_sum += 2 * weight * gamma_s
    end

    long_run_var = gamma_0 + gamma_sum

    # Ensure positive variance
    if long_run_var <= 0
        long_run_var = gamma_0
    end

    # Step 4: Compute KPSS statistic
    kpss_stat = sum(partial_sums.^2) / (n^2 * long_run_var)

    # Step 5: Get critical values and compute p-value
    critical_values = KPSS_CRITICAL_VALUES[regression]
    p_value = kpss_pvalue(kpss_stat, regression)

    # Determine stationarity (fail to reject H0 if stat < critical value)
    is_stationary = p_value >= alpha

    KPSSResult(
        statistic=kpss_stat,
        p_value=p_value,
        lags=lags,
        n_obs=n,
        critical_values=critical_values,
        is_stationary=is_stationary,
        regression=regression,
        alpha=alpha,
    )
end


"""Compute approximate p-value for KPSS statistic."""
function kpss_pvalue(stat::Float64, regression::String)
    cv = KPSS_CRITICAL_VALUES[regression]

    if stat <= cv["10%"]
        return 0.15  # Conservative upper bound
    elseif stat <= cv["5%"]
        return 0.10 - 0.05 * (stat - cv["10%"]) / (cv["5%"] - cv["10%"])
    elseif stat <= cv["2.5%"]
        return 0.05 - 0.025 * (stat - cv["5%"]) / (cv["2.5%"] - cv["5%"])
    elseif stat <= cv["1%"]
        return 0.025 - 0.015 * (stat - cv["2.5%"]) / (cv["1%"] - cv["2.5%"])
    else
        return 0.005
    end
end


"""
    phillips_perron_test(series; regression="c", lags=nothing, alpha=0.05)

Phillips-Perron test for unit root.

Tests H0: series has unit root (non-stationary)
vs H1: series is stationary.

Like ADF but uses Newey-West HAC correction instead of augmented lags.
Robust to heteroskedasticity and autocorrelation of unknown form.

# Arguments
- `series::Vector{Float64}`: 1D time series
- `regression::String`: Type of regression:
  - "n": no constant, no trend
  - "c": constant only (default)
  - "ct": constant and trend
- `lags::Union{Int,Nothing}`: Lags for Newey-West. If nothing, uses Schwert rule.
- `alpha::Float64`: Significance level for determining stationarity

# Returns
- `PPResult`: Test results including Z_t statistic, p-value, and stationarity decision

# Example
```julia
using Random
Random.seed!(42)
y_stat = randn(200)
result = phillips_perron_test(y_stat)
println("Stationary: \$(result.is_stationary)")
```

# Notes
PP test has same null hypothesis and critical values as ADF test.
Advantage: robust to general heteroskedasticity without specifying lag structure.
Disadvantage: may have worse size properties in small samples.

# References
Phillips & Perron (1988). "Testing for a unit root in time series
regression." Biometrika 75(2): 335-346.
"""
function phillips_perron_test(
    series::AbstractVector{<:Real};
    regression::String="c",
    lags::Union{Int,Nothing}=nothing,
    alpha::Float64=0.05,
)
    series = Float64.(series)
    n = length(series)

    if n < 10
        error("Series too short for PP test (n=$n, need >= 10)")
    end

    if !(regression in ["n", "c", "ct"])
        error("regression must be 'n', 'c', or 'ct', got '$regression'")
    end

    # Default lags for Newey-West
    if lags === nothing
        lags = Int(ceil(4 * (n / 100)^0.25))
    end

    # Step 1: Run simple Dickey-Fuller regression (no augmentation)
    dy = diff(series)  # First difference
    y_lag = series[1:end-1]  # y_{t-1}
    T = length(dy)

    # Build design matrix
    X_parts = Vector{Float64}[]

    if regression in ["c", "ct"]
        push!(X_parts, ones(T))
    end

    if regression == "ct"
        trend = Float64.(1:T)
        push!(X_parts, trend)
    end

    push!(X_parts, y_lag)

    X = hcat(X_parts...)

    # OLS estimation
    beta = X \ dy
    residuals = dy - X * beta

    # Get rho coefficient position
    rho_idx = if regression == "n"
        1
    elseif regression == "c"
        2
    else  # ct
        3
    end

    rho_hat = beta[rho_idx]

    # Step 2: Compute short-run and long-run variance
    s2 = sum(residuals.^2) / (T - size(X, 2))

    # Newey-West long-run variance with Bartlett kernel
    gamma_0 = sum(residuals.^2) / T

    gamma_sum = 0.0
    for j in 1:lags
        weight = 1 - j / (lags + 1)
        gamma_j = sum(residuals[j+1:end] .* residuals[1:end-j]) / T
        gamma_sum += 2 * weight * gamma_j
    end

    lambda_sq = gamma_0 + gamma_sum

    # Ensure positive variance
    if lambda_sq <= 0
        lambda_sq = gamma_0
    end

    # Step 3: Compute PP Z_t statistic
    XtX_inv = inv(X' * X)
    se_rho = sqrt(s2 * XtX_inv[rho_idx, rho_idx])
    t_rho = rho_hat / se_rho

    s = sqrt(s2)
    lambda_hat = sqrt(lambda_sq)

    # Standard error of regression (for the x'x term)
    sum_y_lag_sq = sum((y_lag .- mean(y_lag)).^2)

    # PP Z_t statistic
    ratio = s / lambda_hat
    correction = (T * (lambda_sq - s2)) / (2 * lambda_hat * se_rho * sqrt(sum_y_lag_sq))

    z_t = ratio * t_rho - correction

    # PP Z_rho statistic (alternative form)
    z_rho = T * rho_hat - (T^2 * (lambda_sq - s2)) / (2 * sum_y_lag_sq)

    # Step 4: Get critical values and p-value
    critical_values = get(ADF_CRITICAL_VALUES, regression, ADF_CRITICAL_VALUES["c"])
    p_value = adf_pvalue(z_t, regression, T)

    # Determine stationarity
    is_stationary = p_value < alpha

    PPResult(
        statistic=z_t,
        p_value=p_value,
        lags=lags,
        n_obs=n,
        critical_values=critical_values,
        is_stationary=is_stationary,
        regression=regression,
        alpha=alpha,
        rho_stat=z_rho,
    )
end


"""
    confirmatory_stationarity_test(series; regression="c", alpha=0.05)

Run ADF and KPSS tests together for confirmatory analysis.

Combines opposite-null tests for stronger inference:
- ADF: H0 = unit root
- KPSS: H0 = stationary

# Arguments
- `series::Vector{Float64}`: 1D time series
- `regression::String`: Type of regression ("c" or "ct")
- `alpha::Float64`: Significance level

# Returns
- `ConfirmatoryResult`: Contains ADF result, KPSS result, and interpretation

# Example
```julia
using Random
Random.seed!(42)
y = randn(200)
result = confirmatory_stationarity_test(y)
println(result.interpretation)
```
"""
function confirmatory_stationarity_test(
    series::AbstractVector{<:Real};
    regression::String="c",
    alpha::Float64=0.05,
)
    adf_result = adf_test(series; regression=regression, alpha=alpha)
    kpss_result = kpss_test(series; regression=regression, alpha=alpha)

    # Interpret combined results
    adf_rejects = adf_result.is_stationary  # Rejects unit root -> stationary
    kpss_rejects = !kpss_result.is_stationary  # Rejects stationarity -> non-stationary

    interpretation, conclusion = if adf_rejects && !kpss_rejects
        "Stationary (ADF rejects unit root, KPSS fails to reject stationarity)", "stationary"
    elseif !adf_rejects && kpss_rejects
        "Non-stationary (ADF fails to reject unit root, KPSS rejects stationarity)", "non-stationary"
    elseif adf_rejects && kpss_rejects
        "Inconclusive (both tests reject - possible fractional integration)", "inconclusive"
    else
        "Inconclusive (neither test rejects - low power or near unit root)", "inconclusive"
    end

    ConfirmatoryResult(adf_result, kpss_result, interpretation, conclusion)
end


"""
    difference_series(series; order=1)

Difference a time series.

# Arguments
- `series::Vector{Float64}`: 1D time series
- `order::Int`: Order of differencing (1 = first difference, 2 = second difference, etc.)

# Returns
- `Vector{Float64}`: Differenced series (length = n - order)

# Example
```julia
y = [1.0, 3.0, 6.0, 10.0, 15.0]
difference_series(y, order=1)  # [2.0, 3.0, 4.0, 5.0]
difference_series(y, order=2)  # [1.0, 1.0, 1.0]
```
"""
function difference_series(series::AbstractVector{<:Real}; order::Int=1)
    series = Float64.(series)

    if order < 1
        error("order must be >= 1, got $order")
    end

    if order >= length(series)
        error("order ($order) must be less than series length ($(length(series)))")
    end

    result = series
    for _ in 1:order
        result = diff(result)
    end

    return result
end


"""
    check_stationarity(data; var_names=nothing, alpha=0.05, regression="c")

Check stationarity for multiple series.

# Arguments
- `data::Matrix{Float64}`: Shape (n_obs, n_vars) multivariate time series
- `var_names::Union{Vector{String},Nothing}`: Variable names
- `alpha::Float64`: Significance level
- `regression::String`: Regression type for ADF test

# Returns
- `Dict{String, ADFResult}`: Mapping from variable name to ADF result

# Example
```julia
using Random
Random.seed!(42)
data = hcat(randn(200), cumsum(randn(200)))
results = check_stationarity(data, var_names=["stat", "nonstat"])
```
"""
function check_stationarity(
    data::AbstractMatrix{<:Real};
    var_names::Union{Vector{String},Nothing}=nothing,
    alpha::Float64=0.05,
    regression::String="c",
)
    data = Float64.(data)
    n_obs, n_vars = size(data)

    if var_names === nothing
        var_names = ["var_$i" for i in 1:n_vars]
    end

    results = Dict{String, ADFResult}()
    for (i, name) in enumerate(var_names)
        results[name] = adf_test(data[:, i]; regression=regression, alpha=alpha)
    end

    return results
end

end # module
