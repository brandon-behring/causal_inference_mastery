#=
HAC (Heteroskedasticity and Autocorrelation Consistent) Inference.

Provides HAC-robust variance estimation for time series data,
with support for influence function-based inference in DML settings.

Reference:
Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
heteroskedasticity and autocorrelation consistent covariance matrix.
=#

module HACInference

using Statistics
using LinearAlgebra
using Distributions

export newey_west_variance, influence_function_se, optimal_bandwidth
export confidence_interval

"""
    newey_west_variance(scores::AbstractMatrix; bandwidth=nothing, kernel=:bartlett)

Compute Newey-West HAC variance estimator for influence scores.

# Arguments
- `scores`: Influence scores, shape (n_obs, n_params)
- `bandwidth`: Truncation lag. Default: Newey-West optimal
- `kernel`: :bartlett or :qs (quadratic spectral)

# Returns
- Variance-covariance matrix
"""
function newey_west_variance(
    scores::AbstractVecOrMat{<:Real};
    bandwidth::Union{Int,Nothing}=nothing,
    kernel::Symbol=:bartlett
)
    # Ensure 2D
    S = scores isa AbstractVector ? reshape(scores, :, 1) : scores
    T, k = size(S)

    # Default bandwidth: Newey-West optimal
    if isnothing(bandwidth)
        bandwidth = floor(Int, 4 * (T / 100)^(2/9))
    end
    bandwidth = min(bandwidth, T - 2)

    # Demean scores
    S_demeaned = S .- mean(S, dims=1)

    # Lag 0: Γ₀
    Omega = (S_demeaned' * S_demeaned) / T

    # Add lagged terms
    for j in 1:bandwidth
        if kernel == :bartlett
            w = 1 - j / (bandwidth + 1)
        else  # quadratic spectral
            z = 6 * π * j / (5 * bandwidth)
            if z < 1e-10
                w = 1.0
            else
                w = 3 / z^2 * (sin(z) / z - cos(z))
            end
        end

        # Autocovariance at lag j
        Gamma_j = (S_demeaned[(j+1):end, :]' * S_demeaned[1:(end-j), :]) / T

        # Add both j and -j (symmetric)
        Omega += w * (Gamma_j + Gamma_j')
    end

    # Return scalar for 1D input
    if k == 1
        return Omega[1, 1]
    end

    return Omega
end

"""
    influence_function_se(influence_scores; bandwidth=nothing, kernel=:bartlett)

Compute standard errors from influence function scores with HAC.

# Arguments
- `influence_scores`: Influence scores, shape (n_obs,) or (n_obs, n_params)
- `bandwidth`: HAC bandwidth
- `kernel`: HAC kernel

# Returns
- Standard errors
"""
function influence_function_se(
    influence_scores::AbstractVecOrMat{<:Real};
    bandwidth::Union{Int,Nothing}=nothing,
    kernel::Symbol=:bartlett
)
    S = influence_scores isa AbstractVector ? reshape(influence_scores, :, 1) : influence_scores
    n = size(S, 1)

    V = newey_west_variance(S; bandwidth=bandwidth, kernel=kernel)

    if V isa Number
        return sqrt(V / n)
    else
        return sqrt.(diag(V) ./ n)
    end
end

"""
    optimal_bandwidth(n::Int; method=:nw)

Compute optimal HAC bandwidth.

# Arguments
- `n`: Number of observations
- `method`: :nw (Newey-West) or :andrews
"""
function optimal_bandwidth(n::Int; method::Symbol=:nw)
    return floor(Int, 4 * (n / 100)^(2/9))
end

"""
    confidence_interval(estimate, se; alpha=0.05, method=:normal, df=nothing)

Compute confidence intervals.

# Arguments
- `estimate`: Point estimates
- `se`: Standard errors
- `alpha`: Significance level
- `method`: :normal or :t
- `df`: Degrees of freedom for t-distribution

# Returns
- (ci_lower, ci_upper) tuple
"""
function confidence_interval(
    estimate::AbstractVector{<:Real},
    se::AbstractVector{<:Real};
    alpha::Float64=0.05,
    method::Symbol=:normal,
    df::Union{Int,Nothing}=nothing
)
    if method == :normal
        z = quantile(Normal(), 1 - alpha / 2)
    else
        isnothing(df) && error("df required for t-distribution")
        z = quantile(TDist(df), 1 - alpha / 2)
    end

    ci_lower = estimate .- z .* se
    ci_upper = estimate .+ z .* se

    return ci_lower, ci_upper
end

end  # module HACInference
