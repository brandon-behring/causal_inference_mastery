"""
Type definitions for bunching estimation.

Follows SciML Problem-Estimator-Solution pattern:
- BunchingProblem: Data and parameters
- BunchingEstimator: Algorithm specification
- BunchingSolution: Results

References:
- Saez (2010) - Original bunching methodology
- Chetty et al. (2011) - Integration constraint
- Kleven (2016) - Bunching review
"""

using Statistics

# =============================================================================
# Problem Type
# =============================================================================

"""
    BunchingProblem{T<:Real}

Bunching estimation problem specification.

# Fields
- `data::Vector{T}`: Observed data (e.g., reported income)
- `kink_point::T`: Location of the kink
- `bunching_width::T`: Half-width of bunching region
- `t1_rate::Union{Nothing,T}`: Marginal rate below kink (for elasticity)
- `t2_rate::Union{Nothing,T}`: Marginal rate above kink (for elasticity)

# Example
```julia
data = randn(1000) .* 15 .+ 50
problem = BunchingProblem(data, 50.0, 5.0)
```
"""
struct BunchingProblem{T<:Real}
    data::Vector{T}
    kink_point::T
    bunching_width::T
    t1_rate::Union{Nothing,T}
    t2_rate::Union{Nothing,T}

    function BunchingProblem(
        data::Vector{T},
        kink_point::T,
        bunching_width::T;
        t1_rate::Union{Nothing,T}=nothing,
        t2_rate::Union{Nothing,T}=nothing,
    ) where {T<:Real}
        # Validation
        isempty(data) && throw(ArgumentError("data cannot be empty"))
        any(!isfinite, data) && throw(ArgumentError("data contains non-finite values"))
        bunching_width <= 0 && throw(ArgumentError("bunching_width must be positive"))

        if !isnothing(t1_rate) && !isnothing(t2_rate)
            (t1_rate < 0 || t1_rate >= 1) && throw(ArgumentError("t1_rate must be in [0, 1)"))
            (t2_rate < 0 || t2_rate >= 1) && throw(ArgumentError("t2_rate must be in [0, 1)"))
            t2_rate <= t1_rate && throw(ArgumentError("t2_rate must be > t1_rate"))
        end

        new{T}(data, kink_point, bunching_width, t1_rate, t2_rate)
    end
end

# =============================================================================
# Estimator Types
# =============================================================================

"""
    SaezBunching

Saez (2010) bunching estimator with polynomial counterfactual.

# Fields
- `n_bins::Int`: Number of histogram bins
- `polynomial_order::Int`: Order of polynomial for counterfactual (default=7)
- `n_bootstrap::Int`: Bootstrap iterations for SE (default=200)

# Example
```julia
estimator = SaezBunching(n_bins=50, polynomial_order=7)
solution = solve(problem, estimator)
```
"""
struct SaezBunching
    n_bins::Int
    polynomial_order::Int
    n_bootstrap::Int

    function SaezBunching(;
        n_bins::Int=50,
        polynomial_order::Int=7,
        n_bootstrap::Int=200,
    )
        n_bins < 10 && throw(ArgumentError("n_bins must be >= 10"))
        polynomial_order < 1 && throw(ArgumentError("polynomial_order must be >= 1"))
        n_bootstrap < 10 && throw(ArgumentError("n_bootstrap must be >= 10"))
        new(n_bins, polynomial_order, n_bootstrap)
    end
end

# =============================================================================
# Solution Types
# =============================================================================

"""
    CounterfactualResult{T<:Real}

Result from counterfactual density estimation.

# Fields
- `bin_centers::Vector{T}`: Centers of histogram bins
- `actual_counts::Vector{T}`: Observed counts
- `counterfactual_counts::Vector{T}`: Estimated counterfactual
- `polynomial_coeffs::Vector{T}`: Fitted polynomial coefficients
- `polynomial_order::Int`: Order of polynomial
- `bunching_region::Tuple{T,T}`: (lower, upper) bounds
- `r_squared::T`: R² of polynomial fit
- `n_bins::Int`: Number of bins
- `bin_width::T`: Width of each bin
"""
struct CounterfactualResult{T<:Real}
    bin_centers::Vector{T}
    actual_counts::Vector{T}
    counterfactual_counts::Vector{T}
    polynomial_coeffs::Vector{T}
    polynomial_order::Int
    bunching_region::Tuple{T,T}
    r_squared::T
    n_bins::Int
    bin_width::T
end

"""
    BunchingSolution{T<:Real}

Solution from bunching estimation.

# Fields
- `excess_mass::T`: Normalized excess mass (b = B/h0)
- `excess_mass_se::T`: Standard error of excess mass
- `excess_mass_count::T`: Raw excess count (B)
- `elasticity::T`: Behavioral elasticity (NaN if rates not provided)
- `elasticity_se::T`: Standard error of elasticity
- `kink_point::T`: Location of kink
- `bunching_region::Tuple{T,T}`: (lower, upper) bounds
- `counterfactual::CounterfactualResult{T}`: Full counterfactual result
- `t1_rate::Union{Nothing,T}`: Rate below kink
- `t2_rate::Union{Nothing,T}`: Rate above kink
- `n_obs::Int`: Number of observations
- `n_bootstrap::Int`: Bootstrap iterations
- `convergence::Bool`: Whether estimation succeeded
- `message::String`: Status message

# Example
```julia
solution = solve(problem, SaezBunching())
println("Excess mass: ", solution.excess_mass)
println("Elasticity: ", solution.elasticity)
```
"""
struct BunchingSolution{T<:Real}
    excess_mass::T
    excess_mass_se::T
    excess_mass_count::T
    elasticity::T
    elasticity_se::T
    kink_point::T
    bunching_region::Tuple{T,T}
    counterfactual::CounterfactualResult{T}
    t1_rate::Union{Nothing,T}
    t2_rate::Union{Nothing,T}
    n_obs::Int
    n_bootstrap::Int
    convergence::Bool
    message::String
end

# =============================================================================
# Display Methods
# =============================================================================

function Base.show(io::IO, p::BunchingProblem)
    print(io, "BunchingProblem(n=$(length(p.data)), kink=$(p.kink_point), width=$(p.bunching_width))")
end

function Base.show(io::IO, ::MIME"text/plain", p::BunchingProblem)
    println(io, "BunchingProblem")
    println(io, "  Observations: ", length(p.data))
    println(io, "  Kink point: ", p.kink_point)
    println(io, "  Bunching width: ", p.bunching_width)
    if !isnothing(p.t1_rate)
        println(io, "  Tax rates: ", p.t1_rate, " → ", p.t2_rate)
    end
end

function Base.show(io::IO, s::BunchingSolution)
    print(io, "BunchingSolution(b=$(round(s.excess_mass, digits=3)), e=$(round(s.elasticity, digits=3)))")
end

function Base.show(io::IO, ::MIME"text/plain", s::BunchingSolution)
    println(io, "BunchingSolution")
    println(io, "─" ^ 40)
    println(io, "  Excess mass (b):     ", round(s.excess_mass, digits=4),
            " (SE: ", round(s.excess_mass_se, digits=4), ")")
    println(io, "  Excess count (B):    ", round(s.excess_mass_count, digits=2))
    if isfinite(s.elasticity)
        println(io, "  Elasticity (e):      ", round(s.elasticity, digits=4),
                " (SE: ", round(s.elasticity_se, digits=4), ")")
    end
    println(io, "─" ^ 40)
    println(io, "  Kink point:          ", s.kink_point)
    println(io, "  Bunching region:     ", s.bunching_region)
    println(io, "  Observations:        ", s.n_obs)
    println(io, "  R² (counterfactual): ", round(s.counterfactual.r_squared, digits=4))
    println(io, "  Convergence:         ", s.convergence ? "✓" : "✗")
end
