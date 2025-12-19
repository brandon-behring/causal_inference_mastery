"""
Bandwidth selection for Regression Kink Design (RKD).

RKD requires larger bandwidths than RDD because we're estimating derivatives
(slopes) rather than levels. The optimal rate is n^{-1/9} vs RDD's n^{-1/5}.

References:
- Card et al. (2015) - RKD bandwidth considerations
- Calonico, Cattaneo, & Titiunik (2014) - CCT bandwidth
- Imbens & Kalyanaraman (2012) - IK bandwidth
"""

"""
    rkd_ik_bandwidth(y::AbstractVector, x::AbstractVector, cutoff::Real) -> Float64

Compute IK-style bandwidth adapted for RKD estimation.

RKD bandwidth differs from RDD bandwidth because:
1. We estimate derivatives (slopes) not levels
2. This requires more data, so bandwidth is larger
3. Optimal rate is n^{-1/9} instead of n^{-1/5}

# Arguments
- `y`: Outcome variable
- `x`: Running variable
- `cutoff`: Kink point

# Returns
- Optimal bandwidth for RKD estimation

# Formula
The bandwidth follows:
```math
h_{RKD} = C * σ / f(c) * n^{-1/9}
```

where:
- σ = std(y) is outcome standard deviation
- f(c) = density at cutoff (estimated)
- C = constant (empirically calibrated)
"""
function rkd_ik_bandwidth(
    y::AbstractVector{T},
    x::AbstractVector{T},
    cutoff::T
) where {T<:Real}
    n = length(x)

    # Estimate standard deviation of outcome
    sigma_y = std(y)

    # Estimate density at cutoff using histogram-based approach
    x_range = maximum(x) - minimum(x)
    n_bins = max(10, Int(ceil(sqrt(n))))
    bin_width = x_range / n_bins

    # Count observations near cutoff
    near_cutoff = abs.(x .- cutoff) .< bin_width
    f_cutoff = sum(near_cutoff) / (n * bin_width)
    f_cutoff = max(f_cutoff, 0.01)  # Lower bound to avoid division issues

    # IK-style constant
    c_ik = 2.702

    # RKD rate: n^{-1/9} (slower than RDD's n^{-1/5})
    h = c_ik * (sigma_y / f_cutoff) * n^(-1/9)

    # Bound by data range
    h = min(h, 0.5 * x_range)
    h = max(h, 0.1 * x_range / sqrt(n))

    return Float64(h)
end

"""
    rkd_rot_bandwidth(x::AbstractVector, cutoff::Real) -> Float64

Rule-of-thumb bandwidth for RKD.

Simple bandwidth based on data spread and sample size.

# Formula
```math
h_{ROT} = 1.5 * IQR(x) / n^{1/9}
```
"""
function rkd_rot_bandwidth(
    x::AbstractVector{T},
    cutoff::T
) where {T<:Real}
    n = length(x)

    # Interquartile range
    q25 = quantile(x, 0.25)
    q75 = quantile(x, 0.75)
    iqr = q75 - q25

    # Silverman-style bandwidth adapted for RKD
    sigma = min(std(x), iqr / 1.34)

    # n^{-1/9} rate for RKD
    h = 1.5 * sigma * n^(-1/9)

    # Bound by data range
    x_range = maximum(x) - minimum(x)
    h = min(h, 0.5 * x_range)
    h = max(h, 0.05 * x_range)

    return Float64(h)
end

"""
    select_rkd_bandwidth(
        problem::RKDProblem,
        method::Symbol=:ik
    ) -> Float64

Select optimal bandwidth for RKD estimation.

# Arguments
- `problem`: RKDProblem with data
- `method`: Bandwidth selection method
  - `:ik` (default): IK-style adapted for RKD
  - `:rot`: Rule-of-thumb

# Returns
- Selected bandwidth value
"""
function select_rkd_bandwidth(
    problem::RKDProblem{T,P},
    method::Symbol=:ik
) where {T<:Real,P<:NamedTuple}
    y = problem.outcomes
    x = problem.running_var
    cutoff = problem.cutoff

    if method == :ik
        return rkd_ik_bandwidth(y, x, cutoff)
    elseif method == :rot
        return rkd_rot_bandwidth(x, cutoff)
    else
        throw(ArgumentError("Unknown bandwidth method: $method. Use :ik or :rot"))
    end
end
