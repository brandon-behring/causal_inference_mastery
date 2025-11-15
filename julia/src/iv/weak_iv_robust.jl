"""
Weak IV Robust Inference for Instrumental Variables

Implements inference procedures that remain valid even when instruments are weak
(first-stage F-statistic < 10). Unlike 2SLS/LIML/GMM which can have poor finite-sample
properties with weak instruments, these tests provide correct Type I error control
regardless of instrument strength.

# Implemented Tests

1. **Anderson-Rubin (AR) Test**: Tests H₀: β = β₀ using reduced-form F-statistic
   - Valid for any instrument strength
   - Conservative (lower power than Wald test when instruments are strong)
   - Can be inverted to form confidence sets

2. **Conditional Likelihood Ratio (CLR) Test** (Moreira, 2003): Tests H₀: β = β₀
   - Optimal power properties under weak instruments
   - Conditionally unbiased and similar
   - More powerful than AR test in many settings

# References

- Anderson, T. W., & Rubin, H. (1949). "Estimation of the parameters of a single
  equation in a complete system of stochastic equations." *Annals of Mathematical
  Statistics*, 20(1), 46-63.

- Moreira, M. J. (2003). "A conditional likelihood ratio test for structural models."
  *Econometrica*, 71(4), 1027-1048.

- Andrews, D. W., Moreira, M. J., & Stock, J. H. (2006). "Optimal two-sided invariant
  similar tests for instrumental variables regression." *Econometrica*, 74(3), 715-752.

- Stock, J. H., & Yogo, M. (2005). "Testing for Weak Instruments in Linear IV
  Regression." In *Identification and Inference for Econometric Models* (pp. 80-108).
  Cambridge University Press.
"""

using LinearAlgebra
using Distributions
using Statistics

"""
    AndersonRubin(; alpha=0.05)

Anderson-Rubin test for weak IV robust inference.

Tests H₀: β = β₀ by examining whether the reduced-form residuals have
zero projection onto the instruments. The test statistic is:

    AR(β₀) = (n-K-p) / K * (Y - β₀*D)'P_Z(Y - β₀*D) / [(Y - β₀*D)'M_Z(Y - β₀*D)]

Under H₀, AR ~ F(K, n-K-p) regardless of instrument strength.

# Fields
- `alpha`: Significance level for testing (default: 0.05)
- `grid_size`: Number of points for confidence set grid search (default: 100)

# Advantages
- Valid for any instrument strength (including very weak instruments)
- Exact finite-sample size control under homoskedasticity
- Simple to compute

# Disadvantages
- Conservative (low power) when instruments are strong
- Confidence sets can be unbounded with very weak instruments
- Requires specifying null value β₀ (not a point estimator)

# Usage

```julia
# Test specific null hypothesis
estimator = AndersonRubin(alpha=0.05)
result = solve(problem, estimator, β₀=2.0)

# Construct confidence set
ci = ar_confidence_set(problem, estimator)
```
"""
struct AndersonRubin <: AbstractIVEstimator
    alpha::Float64
    grid_size::Int

    function AndersonRubin(; alpha::Float64=0.05, grid_size::Int=100)
        if !(0 < alpha < 1)
            throw(ArgumentError("alpha must be in (0,1), got $alpha"))
        end
        if grid_size < 10
            throw(ArgumentError("grid_size must be at least 10, got $grid_size"))
        end
        new(alpha, grid_size)
    end
end

"""
    ConditionalLR(; alpha=0.05)

Conditional Likelihood Ratio (CLR) test for weak IV robust inference (Moreira, 2003).

More powerful than Anderson-Rubin test while maintaining validity under weak instruments.
The test statistic conditions on a sufficient statistic for the nuisance parameter
(reduced-form coefficients on instruments).

# Fields
- `alpha`: Significance level for testing (default: 0.05)
- `grid_size`: Number of points for confidence set grid search (default: 100)

# Advantages
- Optimal power properties under weak instruments
- More powerful than AR test in most settings
- Exact finite-sample distribution (tables or simulation)

# Disadvantages
- More complex to compute than AR test
- Critical values require numerical computation or tables
- Implementation complexity

# Usage

```julia
estimator = ConditionalLR(alpha=0.05)
result = solve(problem, estimator, β₀=2.0)
```

# Note
This is a simplified implementation. Full implementation would require:
- Critical value tables or simulation
- Andrews-Moreira-Stock (2006) extensions
- Conditional inference machinery
"""
struct ConditionalLR <: AbstractIVEstimator
    alpha::Float64
    grid_size::Int

    function ConditionalLR(; alpha::Float64=0.05, grid_size::Int=100)
        if !(0 < alpha < 1)
            throw(ArgumentError("alpha must be in (0,1), got $alpha"))
        end
        if grid_size < 10
            throw(ArgumentError("grid_size must be at least 10, got $grid_size"))
        end
        new(alpha, grid_size)
    end
end

"""
    ar_test_statistic(y, d, Z, X, β₀)

Compute Anderson-Rubin test statistic for H₀: β = β₀.

# Arguments
- `y`: Outcome vector (n × 1)
- `d`: Treatment vector (n × 1)
- `Z`: Instrument matrix (n × K)
- `X`: Covariate matrix (n × p) or nothing
- `β₀`: Null hypothesis value

# Returns
- `ar_stat`: AR test statistic ~ F(K, n-K-p-1) under H₀
- `p_value`: P-value for one-sided upper-tail test (reject for large AR)
"""
function ar_test_statistic(
    y::Vector{T},
    d::Vector{T},
    Z::Matrix{T},
    X::Union{Matrix{T}, Nothing},
    β₀::T
) where {T<:Real}
    n = length(y)
    K = size(Z, 2)
    p = isnothing(X) ? 0 : size(X, 2)

    # Reduced-form residual: u = Y - β₀*D
    u = y - β₀ * d

    # Build full instrument matrix: [1 Z X]
    Z_full = isnothing(X) ? hcat(ones(T, n), Z) : hcat(ones(T, n), Z, X)

    # Projection matrices
    P_Z = Z_full * ((Z_full' * Z_full) \ Z_full')
    M_Z = I - P_Z

    # AR statistic: (n-K-p-1)/K * u'P_Z u / u'M_Z u
    numerator = u' * P_Z * u
    denominator = u' * M_Z * u

    ar_stat = T((n - K - p - 1) / K) * (numerator / denominator)

    # P-value from F distribution (one-sided upper-tail test)
    # AR test rejects H₀: β = β₀ for large values of AR statistic
    df1 = K
    df2 = n - K - p - 1
    p_value = T(1 - cdf(FDist(df1, df2), ar_stat))

    return ar_stat, p_value
end

"""
    ar_confidence_set(problem::IVProblem, estimator::AndersonRubin; grid_min=-10, grid_max=10)

Construct confidence set for β by inverting Anderson-Rubin test.

# Arguments
- `problem`: IVProblem specification
- `estimator`: AndersonRubin estimator
- `grid_min`: Minimum value for grid search (default: -10)
- `grid_max`: Maximum value for grid search (default: 10)

# Returns
Named tuple with:
- `ci_set`: Vector of β values in confidence set
- `ci_lower`: Lower bound (minimum of set)
- `ci_upper`: Upper bound (maximum of set)
- `is_bounded`: Whether confidence set is bounded
"""
function ar_confidence_set(
    problem::IVProblem{T,P},
    estimator::AndersonRubin;
    grid_min::Real=-10.0,
    grid_max::Real=10.0
) where {T<:Real, P<:NamedTuple}
    y = problem.outcomes
    d = problem.treatment
    Z = problem.instruments
    X = problem.covariates

    # Grid search over β values
    β_grid = range(grid_min, grid_max, length=estimator.grid_size)
    in_set = falses(estimator.grid_size)

    for (i, β₀) in enumerate(β_grid)
        _, p_value = ar_test_statistic(y, d, Z, X, β₀)
        in_set[i] = p_value > estimator.alpha
    end

    # Extract confidence set
    ci_set = β_grid[in_set]

    if isempty(ci_set)
        # Empty set (rare, indicates very weak instruments)
        return (ci_set=T[], ci_lower=NaN, ci_upper=NaN, is_bounded=false)
    end

    ci_lower = minimum(ci_set)
    ci_upper = maximum(ci_set)

    # Check if bounded (should be continuous interval)
    is_bounded = (ci_upper - ci_lower) < (grid_max - grid_min) * 0.9

    return (ci_set=collect(ci_set), ci_lower=ci_lower, ci_upper=ci_upper, is_bounded=is_bounded)
end

"""
    solve(problem::IVProblem, estimator::AndersonRubin, β₀::Real)

Perform Anderson-Rubin test for H₀: β = β₀.

Returns IVSolution with:
- `estimate`: Grid search point estimate (center of confidence set)
- `se`: Half-width of confidence set (not a standard error)
- `ci_lower`, `ci_upper`: Confidence set bounds
- `p_value`: P-value for testing H₀: β = β₀

# Note
Anderson-Rubin is a testing procedure, not a point estimator. The "estimate"
returned is the midpoint of the confidence set for convenience.
"""
function solve(
    problem::IVProblem{T,P},
    estimator::AndersonRubin,
    β₀::Real=0.0
) where {T<:Real, P<:NamedTuple}
    y = problem.outcomes
    d = problem.treatment
    Z = problem.instruments
    X = problem.covariates
    alpha = problem.parameters.alpha

    n = length(y)
    K = size(Z, 2)
    p = isnothing(X) ? 0 : size(X, 2)

    # ========================================================================
    # Test H₀: β = β₀
    # ========================================================================

    ar_stat, p_value_test = ar_test_statistic(y, d, Z, X, T(β₀))

    # ========================================================================
    # Construct Confidence Set
    # ========================================================================

    # Estimate TSLS first to get reasonable grid range
    tsls_sol = solve(problem, TSLS())
    grid_min = tsls_sol.estimate - 5 * tsls_sol.se
    grid_max = tsls_sol.estimate + 5 * tsls_sol.se

    cs = ar_confidence_set(problem, estimator; grid_min=grid_min, grid_max=grid_max)

    # Point estimate: midpoint of confidence set
    β_hat = isnan(cs.ci_lower) ? tsls_sol.estimate : (cs.ci_lower + cs.ci_upper) / 2

    # "SE": half-width of confidence set (not a true standard error)
    se_ar = isnan(cs.ci_lower) ? Inf : (cs.ci_upper - cs.ci_lower) / 2

    # ========================================================================
    # Weak IV Diagnostics
    # ========================================================================

    fstat, _ = first_stage_fstat(d, Z, X)
    cd_stat = cragg_donald_stat(d, Z, X)
    is_weak, weak_warning = weak_iv_warning(fstat, cd_stat, K)

    # ========================================================================
    # Diagnostics
    # ========================================================================

    diagnostics = (
        ar_statistic=ar_stat,
        ci_set_bounded=cs.is_bounded,
        ci_set_size=length(cs.ci_set),
        tsls_estimate=tsls_sol.estimate,
        tsls_se=tsls_sol.se,
        first_stage_fstat=fstat,
        cragg_donald=cd_stat,
        n_instruments=K,
        n_covariates=p,
    )

    # ========================================================================
    # Return Solution
    # ========================================================================

    return IVSolution(
        β_hat,
        se_ar,
        cs.ci_lower,
        cs.ci_upper,
        p_value_test,
        n,
        K,
        p,
        fstat,
        nothing,  # No overidentification test for AR
        is_weak,
        "Anderson-Rubin",
        alpha,
        diagnostics,
    )
end

"""
    solve(problem::IVProblem, estimator::ConditionalLR, β₀::Real)

Perform Conditional Likelihood Ratio test for H₀: β = β₀.

**Note**: This is a placeholder implementation. Full CLR test requires:
- Critical value tables from Moreira (2003)
- Andrews-Moreira-Stock (2006) extensions
- Sophisticated conditional inference machinery

Current implementation uses simplified AR-like statistic as approximation.
"""
function solve(
    problem::IVProblem{T,P},
    estimator::ConditionalLR,
    β₀::Real=0.0
) where {T<:Real, P<:NamedTuple}
    # Placeholder: Use AR test as base
    # Full implementation would compute CLR statistic and use proper critical values
    ar_est = AndersonRubin(alpha=estimator.alpha, grid_size=estimator.grid_size)
    ar_sol = solve(problem, ar_est, β₀)

    # Return with CLR name but AR implementation
    # TODO: Implement true CLR test with Moreira (2003) critical values
    return IVSolution(
        ar_sol.estimate,
        ar_sol.se,
        ar_sol.ci_lower,
        ar_sol.ci_upper,
        ar_sol.p_value,
        ar_sol.n,
        ar_sol.n_instruments,
        ar_sol.n_covariates,
        ar_sol.first_stage_fstat,
        ar_sol.overid_pvalue,
        ar_sol.weak_iv_warning,
        "CLR (Simplified)",
        ar_sol.alpha,
        (
            clr_statistic=ar_sol.diagnostics.ar_statistic,  # Placeholder
            ar_approximation=true,
            note="Full CLR implementation requires Moreira (2003) critical values",
            first_stage_fstat=ar_sol.diagnostics.first_stage_fstat,
            cragg_donald=ar_sol.diagnostics.cragg_donald,
            n_instruments=ar_sol.diagnostics.n_instruments,
            n_covariates=ar_sol.diagnostics.n_covariates,
        ),
    )
end
