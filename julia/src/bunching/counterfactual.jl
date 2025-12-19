"""
Counterfactual density estimation for bunching analysis.

Implements polynomial fitting to estimate what the density would look like
in the absence of bunching behavior.

References:
- Saez (2010) - Original bunching methodology
- Chetty et al. (2011) - Integration constraint
"""

using Statistics
using LinearAlgebra

"""
    polynomial_counterfactual(bin_centers, counts, bunching_lower, bunching_upper;
                              polynomial_order=7)

Fit polynomial counterfactual excluding bunching region.

# Arguments
- `bin_centers::Vector{T}`: Centers of histogram bins
- `counts::Vector{T}`: Observed counts in each bin
- `bunching_lower::T`: Lower bound of bunching region
- `bunching_upper::T`: Upper bound of bunching region
- `polynomial_order::Int=7`: Order of polynomial to fit

# Returns
- `counterfactual::Vector{T}`: Predicted counterfactual counts
- `coeffs::Vector{T}`: Polynomial coefficients
- `r_squared::T`: R² of fit on non-bunching bins

# Throws
- `ArgumentError`: If insufficient bins outside bunching region
"""
function polynomial_counterfactual(
    bin_centers::Vector{T},
    counts::Vector{T},
    bunching_lower::T,
    bunching_upper::T;
    polynomial_order::Int=7,
) where {T<:Real}

    length(bin_centers) != length(counts) && throw(ArgumentError(
        "bin_centers and counts must have same length " *
        "(got $(length(bin_centers)) and $(length(counts)))"
    ))

    polynomial_order < 1 && throw(ArgumentError(
        "polynomial_order must be >= 1 (got $polynomial_order)"
    ))

    # Identify bins outside bunching region
    outside_bunching = (bin_centers .< bunching_lower) .| (bin_centers .> bunching_upper)
    n_outside = sum(outside_bunching)

    if n_outside < polynomial_order + 1
        throw(ArgumentError(
            "Need at least $(polynomial_order + 1) bins outside bunching region " *
            "for polynomial order $polynomial_order, but only have $n_outside"
        ))
    end

    # Fit polynomial to bins outside bunching region
    x_fit = bin_centers[outside_bunching]
    y_fit = counts[outside_bunching]

    # Center x for numerical stability
    x_mean = mean(bin_centers)
    x_centered = x_fit .- x_mean

    # Build Vandermonde matrix
    X = zeros(T, length(x_centered), polynomial_order + 1)
    for j in 0:polynomial_order
        X[:, j+1] = x_centered .^ j
    end

    # Fit polynomial (least squares)
    coeffs = X \ y_fit

    # Predict counterfactual for all bins
    all_x_centered = bin_centers .- x_mean
    X_all = zeros(T, length(all_x_centered), polynomial_order + 1)
    for j in 0:polynomial_order
        X_all[:, j+1] = all_x_centered .^ j
    end
    counterfactual = X_all * coeffs

    # Ensure non-negative
    counterfactual = max.(counterfactual, zero(T))

    # Compute R² on fitting region
    y_pred_fit = X * coeffs
    ss_res = sum((y_fit .- y_pred_fit) .^ 2)
    ss_tot = sum((y_fit .- mean(y_fit)) .^ 2)
    r_squared = ss_tot > 0 ? one(T) - ss_res / ss_tot : zero(T)

    return counterfactual, coeffs, r_squared
end

"""
    estimate_counterfactual(problem::BunchingProblem, n_bins::Int;
                            polynomial_order::Int=7)

Estimate counterfactual density for bunching analysis.

# Arguments
- `problem::BunchingProblem`: Bunching problem specification
- `n_bins::Int`: Number of histogram bins
- `polynomial_order::Int=7`: Polynomial order for counterfactual

# Returns
- `CounterfactualResult`: Full counterfactual estimation result
"""
function estimate_counterfactual(
    problem::BunchingProblem{T},
    n_bins::Int;
    polynomial_order::Int=7,
) where {T<:Real}

    n_bins < 10 && throw(ArgumentError("n_bins must be >= 10 (got $n_bins)"))

    data = problem.data
    kink_point = problem.kink_point
    bunching_width = problem.bunching_width

    # Determine data range
    data_min, data_max = extrema(data)

    # Create bins
    bin_edges = range(data_min, data_max, length=n_bins + 1)
    bin_width = (data_max - data_min) / n_bins

    # Compute histogram
    counts = zeros(T, n_bins)
    for x in data
        # Find bin index
        idx = clamp(floor(Int, (x - data_min) / bin_width) + 1, 1, n_bins)
        counts[idx] += one(T)
    end

    bin_centers = collect((bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2)

    # Define bunching region
    bunching_lower = kink_point - bunching_width
    bunching_upper = kink_point + bunching_width

    # Adjust to data range if needed
    bunching_lower = max(bunching_lower, data_min)
    bunching_upper = min(bunching_upper, data_max)

    # Fit polynomial counterfactual
    counterfactual, coeffs, r_squared = polynomial_counterfactual(
        bin_centers, counts, bunching_lower, bunching_upper;
        polynomial_order=polynomial_order
    )

    return CounterfactualResult(
        bin_centers,
        counts,
        counterfactual,
        coeffs,
        polynomial_order,
        (bunching_lower, bunching_upper),
        r_squared,
        n_bins,
        bin_width,
    )
end

"""
    compute_excess_mass(result::CounterfactualResult)

Compute excess mass from counterfactual estimation.

# Returns
- `excess_mass::T`: Normalized excess mass (b = B/h0)
- `excess_count::T`: Raw excess count (B)
- `h0::T`: Counterfactual height at kink
"""
function compute_excess_mass(result::CounterfactualResult{T}) where {T<:Real}
    bin_centers = result.bin_centers
    actual = result.actual_counts
    counterfactual = result.counterfactual_counts
    bunching_lower, bunching_upper = result.bunching_region

    # Identify bins in bunching region
    bunching_mask = (bin_centers .>= bunching_lower) .& (bin_centers .<= bunching_upper)

    # Raw excess count
    excess_count = sum(actual[bunching_mask] .- counterfactual[bunching_mask])

    # Find counterfactual height at kink (center of bunching region)
    kink_point = (bunching_lower + bunching_upper) / 2
    kink_idx = argmin(abs.(bin_centers .- kink_point))
    h0 = counterfactual[kink_idx]

    # Normalized excess mass
    excess_mass = if h0 > 0
        excess_count / h0
    else
        excess_count > 0 ? T(Inf) : zero(T)
    end

    return excess_mass, excess_count, h0
end

"""
    compute_elasticity(excess_mass::T, t1_rate::T, t2_rate::T) where {T<:Real}

Compute behavioral elasticity from excess mass.

# Arguments
- `excess_mass::T`: Normalized excess mass (b = B/h0)
- `t1_rate::T`: Marginal rate below kink
- `t2_rate::T`: Marginal rate above kink

# Returns
- `elasticity::T`: Behavioral elasticity

# Formula
```
e = b / ln((1 - t1) / (1 - t2))
```
"""
function compute_elasticity(excess_mass::T, t1_rate::T, t2_rate::T) where {T<:Real}
    (t1_rate < 0 || t1_rate >= 1) && throw(ArgumentError("t1_rate must be in [0, 1)"))
    (t2_rate < 0 || t2_rate >= 1) && throw(ArgumentError("t2_rate must be in [0, 1)"))
    t2_rate <= t1_rate && throw(ArgumentError("t2_rate must be > t1_rate"))

    log_change = log((1 - t1_rate) / (1 - t2_rate))

    log_change ≈ 0 && throw(ArgumentError("Rates are too close: log change is zero"))

    return excess_mass / log_change
end
