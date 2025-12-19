# ============================================================================
# CLR Critical Values and Conditional P-Value Functions
# ============================================================================
#
# Implementation of conditional p-value calculations for the Conditional
# Likelihood Ratio (CLR) test following:
#   - Moreira (2003) "A Conditional Likelihood Ratio Test for Structural Models"
#   - Andrews, Moreira, Stock (2006) "Optimal Two-Sided Invariant Similar Tests"
#
# The CLR test is the optimal weak-IV-robust test with better power than
# Anderson-Rubin while maintaining correct size under weak instruments.
#
# ============================================================================

using Distributions
using QuadGK  # For numerical integration
using SpecialFunctions  # For beta function

"""
    cond_pvalue(m::Real, qT::Real, k::Int, df::Int; eps::Float64=0.02)

Compute the conditional p-value for the CLR test.

The conditional p-value is: P(LR > m | QT = qT)

# Arguments
- `m`: LR test statistic value
- `qT`: Conditioning value of the QT statistic
- `k`: Number of instruments (k ≥ 1)
- `df`: Degrees of freedom (n - p - L, where p = exogenous, L = endogenous)
- `eps`: Numerical stability parameter for k ≥ 4 case (default 0.02)

# Returns
- `pval`: Conditional p-value in [0, 1]

# Theory
For k=1 instruments, the LR statistic follows an F distribution.
For k≥2, the p-value is computed via numerical integration following
Andrews, Moreira, Stock (2006).

# References
- Moreira, M. J. (2003). Econometrica 71, 1027-1048.
- Andrews, D. W. K., Moreira, M. J., Stock, J. H. (2006). Econometrica 74, 715-752.
"""
function cond_pvalue(m::Real, qT::Real, k::Int, df::Int; eps::Float64=0.02)
    # Input validation
    k < 1 && throw(ArgumentError("Number of instruments k must be ≥ 1, got $k"))
    df < 1 && throw(ArgumentError("Degrees of freedom df must be ≥ 1, got $df"))

    # Handle edge cases
    m ≤ 0 && return 1.0
    qT < 0 && return 1.0  # QT should be non-negative

    # Case k = 1: Simple F distribution
    if k == 1
        # Under H₀, LR ~ F(1, df)
        return 1.0 - cdf(FDist(1, df), m)
    end

    # Case k = 2: Integration over [0, π/2]
    if k == 2
        result, _ = quadgk(0, π/2, rtol=1e-8) do x
            pdf(Chisq(k), (qT + m) / (1 + qT * sin(x)^2 / m))
        end
        return (2/π) * result
    end

    # Case k = 3: Integration over [0, 1]
    if k == 3
        result, _ = quadgk(0, 1, rtol=1e-8) do x
            pdf(Chisq(k), (qT + m) / (1 + qT * x^2 / m))
        end
        return result
    end

    # Case k ≥ 4: Split integration for numerical stability
    # Integrand: pdf(χ²_k, (qT + m)/(1 + qT*x²/m)) * (1-x²)^((k-3)/2)
    k_local = k  # Capture k for closure

    # Region 1: [0, 1-eps] - well-behaved
    result1, _ = quadgk(0, 1-eps, rtol=1e-8) do x
        arg = (qT + m) / (1 + qT * x^2 / m)
        weight = (1 - x^2)^((k_local-3)/2)
        pdf(Chisq(k_local), arg) * weight
    end

    # Region 2: [1-eps, 1] - looser tolerance near boundary
    result2, _ = quadgk(1-eps, 1, rtol=1e-6) do x
        arg = (qT + m) / (1 + qT * x^2 / m)
        weight = (1 - x^2)^((k_local-3)/2)
        pdf(Chisq(k_local), arg) * weight
    end

    # Normalization constant: Beta((k-1)/2, 1/2) = Γ((k-1)/2)*Γ(1/2) / Γ(k/2)
    # For the integral of (1-x²)^((k-3)/2) over [0,1]
    norm_const = beta((k-1)/2, 0.5) / 2  # Factor of 2 from [0,1] vs [-1,1]

    return (result1 + result2) / norm_const
end


"""
    clr_critical_value(qT::Real, k::Int, df::Int, alpha::Float64=0.05;
                       tol::Float64=1e-6, maxiter::Int=100)

Compute the CLR critical value by inverting the conditional p-value function.

Finds C such that: cond_pvalue(maxEigen - C, qT, k, df) = alpha

# Arguments
- `qT`: Conditioning value of QT statistic
- `k`: Number of instruments
- `df`: Degrees of freedom
- `alpha`: Significance level (default 0.05)
- `tol`: Tolerance for root-finding (default 1e-6)
- `maxiter`: Maximum iterations for bisection (default 100)

# Returns
- `C`: Critical value for the CLR test

# Notes
The critical value depends on qT, making the test conditional.
This is computed via bisection search since cond_pvalue is monotonic in m.
"""
function clr_critical_value(qT::Real, k::Int, df::Int, alpha::Float64=0.05;
                            tol::Float64=1e-6, maxiter::Int=100)
    # Input validation
    0 < alpha < 1 || throw(ArgumentError("alpha must be in (0,1), got $alpha"))

    # For k=1, use simple F critical value
    if k == 1
        return quantile(FDist(1, df), 1 - alpha)
    end

    # Initial bounds for bisection
    # Lower bound: at m=0, p-value=1 > alpha
    # Upper bound: need to find m where p-value < alpha
    lower = 0.0
    upper = quantile(Chisq(k), 1 - alpha/10)  # Start with approximate χ² quantile

    # Expand upper bound if needed
    while cond_pvalue(upper, qT, k, df) > alpha && upper < 1000
        upper *= 2
    end

    if upper >= 1000
        @warn "Could not find upper bound for CLR critical value (qT=$qT, k=$k)"
        return upper
    end

    # Bisection search
    for _ in 1:maxiter
        mid = (lower + upper) / 2
        pval = cond_pvalue(mid, qT, k, df)

        if abs(pval - alpha) < tol
            return mid
        elseif pval > alpha
            lower = mid
        else
            upper = mid
        end

        if upper - lower < tol
            return mid
        end
    end

    @warn "CLR critical value search did not converge (qT=$qT, k=$k)"
    return (lower + upper) / 2
end


"""
    clr_test_pvalue(lr_stat::Real, qT::Real, k::Int, df::Int)

Compute the p-value for a CLR test statistic.

This is a convenience wrapper around cond_pvalue.

# Arguments
- `lr_stat`: The computed LR test statistic
- `qT`: The conditioning QT statistic value
- `k`: Number of instruments
- `df`: Degrees of freedom (n - p - L)

# Returns
- `pval`: Two-sided p-value for the CLR test
"""
function clr_test_pvalue(lr_stat::Real, qT::Real, k::Int, df::Int)
    return cond_pvalue(lr_stat, qT, k, df)
end


# ============================================================================
# Precomputed Critical Value Tables (for efficiency)
# ============================================================================

"""
Precomputed CLR critical values for common configurations.
Format: CLR_CRITICAL_VALUES[(k, alpha)] returns function qT -> critical_value

These are computed via simulation/integration and cached for efficiency.
For qT values not in table, interpolation or direct computation is used.
"""
const CLR_CRITICAL_VALUES_CACHE = Dict{Tuple{Int,Float64,Int}, Float64}()

"""
    get_clr_critical_value_cached(qT::Real, k::Int, df::Int, alpha::Float64)

Get CLR critical value, using cache if available.
"""
function get_clr_critical_value_cached(qT::Real, k::Int, df::Int, alpha::Float64)
    # Round qT to nearest 0.1 for caching
    qT_rounded = round(qT, digits=1)
    key = (k, alpha, Int(qT_rounded * 10))

    if haskey(CLR_CRITICAL_VALUES_CACHE, key)
        return CLR_CRITICAL_VALUES_CACHE[key]
    end

    # Compute and cache
    cv = clr_critical_value(qT, k, df, alpha)
    CLR_CRITICAL_VALUES_CACHE[key] = cv
    return cv
end
