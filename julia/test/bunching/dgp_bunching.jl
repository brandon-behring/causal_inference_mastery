"""
Data Generating Processes (DGPs) for Bunching Monte Carlo validation.

Provides DGPs for validating the bunching estimator (Saez 2010):
- Simple bunching with known excess mass
- No effect (Type I error testing)
- With known elasticity
- Asymmetric/offset bunching
- Diffuse bunching (optimization frictions)
- Large and small sample variants

References:
    - Saez (2010). Do taxpayers bunch at kink points?
    - Chetty et al. (2011). Adjustment costs and frictions.
    - Kleven (2016). Bunching estimation review.
"""

using Random
using Statistics

# =============================================================================
# Data Container Type
# =============================================================================

"""
Container for bunching simulation data with known ground truth.

# Fields
- `data::Vector{T}`: Observed data with bunching
- `true_excess_mass::T`: True excess mass b = B/h0
- `kink_point::T`: Location of the kink
- `bunching_width::T`: Recommended bunching region half-width
- `t1_rate::Union{Nothing,T}`: Marginal rate below kink (optional)
- `t2_rate::Union{Nothing,T}`: Marginal rate above kink (optional)
- `true_elasticity::Union{Nothing,T}`: True elasticity (optional)
- `n::Int`: Sample size
"""
struct BunchingData{T<:Real}
    data::Vector{T}
    true_excess_mass::T
    kink_point::T
    bunching_width::T
    t1_rate::Union{Nothing,T}
    t2_rate::Union{Nothing,T}
    true_elasticity::Union{Nothing,T}
    n::Int
end

# =============================================================================
# Simple Bunching DGP
# =============================================================================

"""
    dgp_bunching_simple(; kwargs...) -> BunchingData

Simple bunching DGP with known excess mass.

Creates a mixture of:
1. Background counterfactual (normal distribution)
2. Bunchers concentrated at kink

# Arguments
- `n::Int=1000`: Sample size
- `kink_point::Float64=50.0`: Location of the kink
- `true_excess_mass::Float64=2.0`: Target excess mass b = B/h0
- `counterfactual_std::Float64=15.0`: Standard deviation of counterfactual
- `bunching_std::Float64=1.0`: Standard deviation of buncher concentration
- `seed::Int=42`: Random seed
"""
function dgp_bunching_simple(;
    n::Int=1000,
    kink_point::Float64=50.0,
    true_excess_mass::Float64=2.0,
    counterfactual_std::Float64=15.0,
    bunching_std::Float64=1.0,
    seed::Int=42
)
    Random.seed!(seed)

    # Counterfactual height at kink (for normal centered at kink)
    h0_per_unit = 1 / (sqrt(2π) * counterfactual_std)

    # Buncher fraction - approximate to get target excess mass
    buncher_fraction = min(0.3, true_excess_mass * h0_per_unit)
    n_bunchers = Int(floor(n * buncher_fraction))
    n_background = n - n_bunchers

    # Generate background (counterfactual)
    background = randn(n_background) .* counterfactual_std .+ kink_point

    # Generate bunchers (concentrated at kink)
    bunchers = randn(n_bunchers) .* bunching_std .+ kink_point

    # Combine
    data = vcat(background, bunchers)

    # Recommended bunching width
    bunching_width = 3 * bunching_std

    return BunchingData(
        data,
        true_excess_mass,
        kink_point,
        bunching_width,
        nothing,
        nothing,
        nothing,
        n
    )
end

# =============================================================================
# No Effect DGP (Type I Error Testing)
# =============================================================================

"""
    dgp_bunching_no_effect(; kwargs...) -> BunchingData

DGP with no bunching (null effect) for Type I error testing.

Pure counterfactual distribution - no excess mass at kink.

# Arguments
- `n::Int=1000`: Sample size
- `kink_point::Float64=50.0`: Location of the kink
- `counterfactual_std::Float64=15.0`: Standard deviation
- `seed::Int=42`: Random seed
"""
function dgp_bunching_no_effect(;
    n::Int=1000,
    kink_point::Float64=50.0,
    counterfactual_std::Float64=15.0,
    seed::Int=42
)
    Random.seed!(seed)

    # Pure counterfactual - no bunching
    data = randn(n) .* counterfactual_std .+ kink_point

    bunching_width = 5.0

    return BunchingData(
        data,
        0.0,  # True excess mass is zero
        kink_point,
        bunching_width,
        nothing,
        nothing,
        nothing,
        n
    )
end

# =============================================================================
# Uniform Counterfactual DGP
# =============================================================================

"""
    dgp_bunching_uniform(; kwargs...) -> BunchingData

Bunching DGP with uniform counterfactual (simpler h0 calculation).

# Arguments
- `n::Int=1000`: Sample size
- `kink_point::Float64=50.0`: Kink location
- `data_range::Tuple{Float64,Float64}=(20.0, 80.0)`: (min, max) of data range
- `buncher_fraction::Float64=0.15`: Fraction of bunchers
- `bunching_std::Float64=1.0`: Standard deviation of bunching
- `seed::Int=42`: Random seed
"""
function dgp_bunching_uniform(;
    n::Int=1000,
    kink_point::Float64=50.0,
    data_range::Tuple{Float64,Float64}=(20.0, 80.0),
    buncher_fraction::Float64=0.15,
    bunching_std::Float64=1.0,
    seed::Int=42
)
    Random.seed!(seed)

    n_bunchers = Int(floor(n * buncher_fraction))
    n_background = n - n_bunchers

    # Background (uniform)
    background = rand(n_background) .* (data_range[2] - data_range[1]) .+ data_range[1]

    # Bunchers
    bunchers = randn(n_bunchers) .* bunching_std .+ kink_point

    data = vcat(background, bunchers)

    # Approximate excess mass for uniform counterfactual
    bunching_width = 3 * bunching_std
    true_excess_mass = buncher_fraction / (1 - buncher_fraction) * (
        (data_range[2] - data_range[1]) / (2 * bunching_width)
    )

    return BunchingData(
        data,
        true_excess_mass,
        kink_point,
        bunching_width,
        nothing,
        nothing,
        nothing,
        n
    )
end

# =============================================================================
# Bunching with Known Elasticity
# =============================================================================

"""
    dgp_bunching_with_elasticity(; kwargs...) -> BunchingData

Bunching DGP with known elasticity for formula validation.

Uses the bunching-elasticity relationship:
    e = b / ln((1-t1)/(1-t2))

# Arguments
- `n::Int=1000`: Sample size
- `kink_point::Float64=50000.0`: Kink location (e.g., tax threshold)
- `t1_rate::Float64=0.20`: Marginal rate below kink
- `t2_rate::Float64=0.30`: Marginal rate above kink
- `true_elasticity::Float64=0.25`: Target behavioral elasticity
- `counterfactual_std::Float64=12000.0`: Counterfactual standard deviation
- `seed::Int=42`: Random seed
"""
function dgp_bunching_with_elasticity(;
    n::Int=1000,
    kink_point::Float64=50000.0,
    t1_rate::Float64=0.20,
    t2_rate::Float64=0.30,
    true_elasticity::Float64=0.25,
    counterfactual_std::Float64=12000.0,
    seed::Int=42
)
    Random.seed!(seed)

    # Calculate required excess mass from elasticity
    # e = b / ln((1-t1)/(1-t2))
    # b = e * ln((1-t1)/(1-t2))
    log_rate_change = log((1 - t1_rate) / (1 - t2_rate))
    true_excess_mass = true_elasticity * log_rate_change

    # Counterfactual height at kink
    h0_per_unit = 1 / (sqrt(2π) * counterfactual_std)

    # Buncher fraction
    buncher_fraction = min(0.25, true_excess_mass * h0_per_unit)
    n_bunchers = Int(floor(n * buncher_fraction))
    n_background = n - n_bunchers

    # Generate data
    bunching_std = counterfactual_std * 0.05  # Tight bunching
    background = randn(n_background) .* counterfactual_std .+ kink_point
    bunchers = randn(n_bunchers) .* bunching_std .+ kink_point

    data = vcat(background, bunchers)
    bunching_width = 3 * bunching_std

    return BunchingData(
        data,
        true_excess_mass,
        kink_point,
        bunching_width,
        t1_rate,
        t2_rate,
        true_elasticity,
        n
    )
end

# =============================================================================
# Asymmetric Bunching DGP
# =============================================================================

"""
    dgp_bunching_asymmetric(; kwargs...) -> BunchingData

Bunching DGP where bunchers are offset from kink.

Tests robustness when bunching is not perfectly centered at kink
(e.g., due to rounding, optimization frictions).

# Arguments
- `n::Int=1000`: Sample size
- `kink_point::Float64=50.0`: Kink location
- `buncher_fraction::Float64=0.15`: Fraction of bunchers
- `bunching_std::Float64=1.0`: Standard deviation of bunching
- `bunching_offset::Float64=-1.0`: Offset of bunching center from kink
- `counterfactual_std::Float64=15.0`: Counterfactual standard deviation
- `seed::Int=42`: Random seed
"""
function dgp_bunching_asymmetric(;
    n::Int=1000,
    kink_point::Float64=50.0,
    buncher_fraction::Float64=0.15,
    bunching_std::Float64=1.0,
    bunching_offset::Float64=-1.0,
    counterfactual_std::Float64=15.0,
    seed::Int=42
)
    Random.seed!(seed)

    n_bunchers = Int(floor(n * buncher_fraction))
    n_background = n - n_bunchers

    # Background
    background = randn(n_background) .* counterfactual_std .+ kink_point

    # Bunchers offset from kink
    bunching_center = kink_point + bunching_offset
    bunchers = randn(n_bunchers) .* bunching_std .+ bunching_center

    data = vcat(background, bunchers)

    # Approximate excess mass
    h0_per_unit = 1 / (sqrt(2π) * counterfactual_std)
    true_excess_mass = buncher_fraction / h0_per_unit

    bunching_width = 3 * bunching_std + abs(bunching_offset)

    return BunchingData(
        data,
        true_excess_mass,
        kink_point,
        bunching_width,
        nothing,
        nothing,
        nothing,
        n
    )
end

# =============================================================================
# Diffuse Bunching DGP (Optimization Frictions)
# =============================================================================

"""
    dgp_bunching_diffuse(; kwargs...) -> BunchingData

Bunching DGP with diffuse bunching (optimization frictions).

Per Chetty et al. (2011), adjustment costs create diffuse bunching
rather than sharp bunching at the kink.

# Arguments
- `n::Int=1000`: Sample size
- `kink_point::Float64=50.0`: Kink location
- `buncher_fraction::Float64=0.15`: Fraction of bunchers
- `bunching_std::Float64=5.0`: Standard deviation (larger = more diffuse)
- `counterfactual_std::Float64=15.0`: Counterfactual standard deviation
- `seed::Int=42`: Random seed
"""
function dgp_bunching_diffuse(;
    n::Int=1000,
    kink_point::Float64=50.0,
    buncher_fraction::Float64=0.15,
    bunching_std::Float64=5.0,
    counterfactual_std::Float64=15.0,
    seed::Int=42
)
    Random.seed!(seed)

    n_bunchers = Int(floor(n * buncher_fraction))
    n_background = n - n_bunchers

    background = randn(n_background) .* counterfactual_std .+ kink_point
    bunchers = randn(n_bunchers) .* bunching_std .+ kink_point

    data = vcat(background, bunchers)

    # Excess mass harder to detect with diffuse bunching
    h0_per_unit = 1 / (sqrt(2π) * counterfactual_std)
    true_excess_mass = buncher_fraction / h0_per_unit

    bunching_width = 2 * bunching_std

    return BunchingData(
        data,
        true_excess_mass,
        kink_point,
        bunching_width,
        nothing,
        nothing,
        nothing,
        n
    )
end

# =============================================================================
# Large Sample DGP
# =============================================================================

"""
    dgp_bunching_large_sample(; kwargs...) -> BunchingData

Large sample bunching data for precision testing.

# Arguments
- `n::Int=10000`: Large sample size
- `kink_point::Float64=50.0`: Kink location
- `true_excess_mass::Float64=1.5`: Target excess mass
- `counterfactual_std::Float64=15.0`: Counterfactual standard deviation
- `bunching_std::Float64=1.0`: Bunching standard deviation
- `seed::Int=42`: Random seed
"""
function dgp_bunching_large_sample(;
    n::Int=10000,
    kink_point::Float64=50.0,
    true_excess_mass::Float64=1.5,
    counterfactual_std::Float64=15.0,
    bunching_std::Float64=1.0,
    seed::Int=42
)
    return dgp_bunching_simple(
        n=n,
        kink_point=kink_point,
        true_excess_mass=true_excess_mass,
        counterfactual_std=counterfactual_std,
        bunching_std=bunching_std,
        seed=seed
    )
end

# =============================================================================
# Small Sample DGP
# =============================================================================

"""
    dgp_bunching_small_sample(; kwargs...) -> BunchingData

Small sample bunching data.

Tests behavior with limited data where estimates are noisier.

# Arguments
- `n::Int=200`: Small sample size
- `kink_point::Float64=50.0`: Kink location
- `true_excess_mass::Float64=2.5`: Target excess mass (larger for detectability)
- `counterfactual_std::Float64=15.0`: Counterfactual standard deviation
- `bunching_std::Float64=1.5`: Bunching standard deviation
- `seed::Int=42`: Random seed
"""
function dgp_bunching_small_sample(;
    n::Int=200,
    kink_point::Float64=50.0,
    true_excess_mass::Float64=2.5,
    counterfactual_std::Float64=15.0,
    bunching_std::Float64=1.5,
    seed::Int=42
)
    return dgp_bunching_simple(
        n=n,
        kink_point=kink_point,
        true_excess_mass=true_excess_mass,
        counterfactual_std=counterfactual_std,
        bunching_std=bunching_std,
        seed=seed
    )
end
