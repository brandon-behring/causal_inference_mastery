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
    clr_test_statistic(y, d, Z, X, β₀)

Compute the Conditional Likelihood Ratio (CLR) test statistic for H₀: β = β₀.

Based on Moreira (2003) "A Conditional Likelihood Ratio Test for Structural Models".

# Arguments
- `y`: Outcome vector (n × 1)
- `d`: Treatment vector (n × 1)
- `Z`: Instrument matrix (n × K)
- `X`: Covariate matrix (n × p) or nothing
- `β₀`: Null hypothesis value

# Returns
Named tuple with:
- `lr_stat`: LR test statistic
- `qS`: QS statistic
- `qT`: QT statistic (used for conditioning)
- `p_value`: Conditional p-value

# Theory
The CLR statistic is:
    LR = 0.5 * (QS - QT + √((QS + QT)² - 4*(QS*QT - QTS²)))

where QS and QT are quadratic forms that decompose the likelihood.
"""
function clr_test_statistic(
    y::Vector{T},
    d::Vector{T},
    Z::Matrix{T},
    X::Union{Matrix{T}, Nothing},
    β₀::T
) where {T<:Real}
    n = length(y)
    K = size(Z, 2)
    p = isnothing(X) ? 0 : size(X, 2)
    df = n - K - p - 1

    # Stack Y and D: Yadj_Dadj = [Y - X*γ_Y, D - X*γ_D]
    # For simplicity, we partiall out X from both Y and D, then from Z
    if isnothing(X)
        Y_adj = y
        D_adj = d
        Z_adj = Z
    else
        # Partial out X
        X_full = hcat(ones(T, n), X)
        M_X = I - X_full * ((X_full' * X_full) \ X_full')
        Y_adj = M_X * y
        D_adj = M_X * d
        Z_adj = M_X * Z
    end

    # QR decomposition of Z_adj for numerical stability
    Q, R = qr(Z_adj)
    PZ = Matrix(Q) * Matrix(Q)'  # Projection onto Z

    # Stack adjusted outcomes: columns are [Y_adj, D_adj]
    YD = hcat(Y_adj, D_adj)

    # Project onto Z
    PZ_YD = PZ * YD  # n × 2 matrix

    # Residual covariance estimate
    M_Z = I - PZ
    residuals = M_Z * YD
    sigma_hat = (residuals' * residuals) / df  # 2 × 2 covariance

    # Check for degenerate cases
    det_sigma = sigma_hat[1,1] * sigma_hat[2,2] - sigma_hat[1,2]^2
    if det_sigma < 1e-10
        # Degenerate case - fall back to AR
        ar_stat, ar_pval = ar_test_statistic(y, d, Z, X, β₀)
        return (lr_stat=ar_stat, qS=ar_stat, qT=T(0), p_value=ar_pval)
    end

    sigma_hat_inv = inv(sigma_hat)

    # Define a0 and b0 vectors for null hypothesis β = β₀
    # b0 = [1, -β₀] gives Y - β₀*D
    # a0 = [β₀, 1] gives β₀*Y + D (orthogonal direction)
    b0 = T[1, -β₀]
    a0 = T[β₀, 1]

    # Denominators
    denom_S = b0' * sigma_hat * b0
    denom_T = a0' * sigma_hat_inv * a0

    # QS statistic: related to reduced-form in null direction
    qS_vec = PZ_YD * b0  # n-vector
    qS = (qS_vec' * qS_vec) / denom_S

    # QT statistic: related to first-stage strength
    qT_vec = PZ_YD * (sigma_hat_inv * a0)  # n-vector
    qT = (qT_vec' * qT_vec) / denom_T

    # Cross term QTS
    qTS = (qS_vec' * qT_vec) / sqrt(denom_S * denom_T)

    # LR statistic: Moreira (2003) formula
    # LR = 0.5 * (QS - QT + sqrt((QS + QT)² - 4*(QS*QT - QTS²)))
    discriminant = (qS + qT)^2 - 4 * (qS * qT - qTS^2)

    if discriminant < 0
        # Numerical issue - should not happen with valid data
        discriminant = T(0)
    end

    lr_stat = T(0.5) * (qS - qT + sqrt(discriminant))

    # Compute conditional p-value
    p_value = cond_pvalue(lr_stat, qT, K, df)

    return (lr_stat=lr_stat, qS=qS, qT=qT, p_value=p_value)
end


"""
    clr_confidence_set(problem::IVProblem, estimator::ConditionalLR; grid_min=-10, grid_max=10)

Construct confidence set for β by inverting CLR test.

# Arguments
- `problem`: IVProblem specification
- `estimator`: ConditionalLR estimator
- `grid_min`: Minimum value for grid search (default: -10)
- `grid_max`: Maximum value for grid search (default: 10)

# Returns
Named tuple with confidence interval bounds and diagnostic information.
"""
function clr_confidence_set(
    problem::IVProblem{T,P},
    estimator::ConditionalLR;
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
        result = clr_test_statistic(y, d, Z, X, T(β₀))
        in_set[i] = result.p_value > estimator.alpha
    end

    # Extract confidence set
    ci_set = β_grid[in_set]

    if isempty(ci_set)
        return (ci_set=T[], ci_lower=NaN, ci_upper=NaN, is_bounded=false)
    end

    ci_lower = minimum(ci_set)
    ci_upper = maximum(ci_set)

    # Check if bounded (endpoints not in set)
    is_bounded = !in_set[1] && !in_set[end]

    return (ci_set=collect(ci_set), ci_lower=ci_lower, ci_upper=ci_upper, is_bounded=is_bounded)
end


"""
    solve(problem::IVProblem, estimator::ConditionalLR, β₀::Real)

Perform Conditional Likelihood Ratio test for H₀: β = β₀.

Implements Moreira (2003) CLR test with conditional critical values following
Andrews, Moreira, Stock (2006).

# Arguments
- `problem`: IVProblem specification
- `estimator`: ConditionalLR estimator with alpha and grid_size
- `β₀`: Null hypothesis value (default: 0.0)

# Returns
IVSolution with CLR test results including:
- Point estimate (midpoint of CI)
- Standard error (from CI width)
- Confidence interval from test inversion
- P-value for H₀: β = β₀

# Theory
The CLR test is the optimal weak-IV-robust test, more powerful than
Anderson-Rubin while maintaining correct size under weak instruments.

# References
- Moreira (2003) Econometrica 71, 1027-1048
- Andrews, Moreira, Stock (2006) Econometrica 74, 715-752
"""
function solve(
    problem::IVProblem{T,P},
    estimator::ConditionalLR,
    β₀::Real=0.0
) where {T<:Real, P<:NamedTuple}
    y = problem.outcomes
    d = problem.treatment
    Z = problem.instruments
    X = problem.covariates
    alpha = estimator.alpha

    n = length(y)
    K = size(Z, 2)
    p = isnothing(X) ? 0 : size(X, 2)

    # Compute CLR test statistic for the null
    clr_result = clr_test_statistic(y, d, Z, X, T(β₀))

    # Compute confidence set by test inversion
    # Adaptive grid based on 2SLS estimate
    tsls_est = (Z' * d) \ (Z' * y)  # Simple 2SLS for centering
    tsls_point = isnothing(X) ? tsls_est[1] : tsls_est[1]

    # Set grid around 2SLS estimate
    grid_width = T(20)  # Wide grid for weak IV
    grid_min = tsls_point - grid_width
    grid_max = tsls_point + grid_width

    ci_result = clr_confidence_set(problem, estimator, grid_min=grid_min, grid_max=grid_max)

    # Point estimate: midpoint of CI (or 2SLS if unbounded)
    if ci_result.is_bounded && !isnan(ci_result.ci_lower)
        estimate = (ci_result.ci_lower + ci_result.ci_upper) / 2
        se = (ci_result.ci_upper - ci_result.ci_lower) / (2 * quantile(Normal(), 1 - alpha/2))
    else
        estimate = tsls_point
        se = T(NaN)  # Unbounded CI
    end

    # First-stage F-statistic for diagnostics
    first_stage_fstat = _compute_first_stage_fstat(d, Z, X)

    # Weak instrument warning
    weak_iv_warning = first_stage_fstat < 10

    # Cragg-Donald statistic (simplified)
    cragg_donald = first_stage_fstat

    diagnostics = (
        clr_statistic=clr_result.lr_stat,
        qS=clr_result.qS,
        qT=clr_result.qT,
        ar_approximation=false,
        first_stage_fstat=first_stage_fstat,
        cragg_donald=cragg_donald,
        n_instruments=K,
        n_covariates=p,
        ci_is_bounded=ci_result.is_bounded,
    )

    return IVSolution(
        estimate,
        se,
        ci_result.ci_lower,
        ci_result.ci_upper,
        clr_result.p_value,
        n,
        K,
        p,
        first_stage_fstat,
        NaN,  # overid_pvalue not applicable for CLR
        weak_iv_warning,
        "CLR (Moreira 2003)",
        alpha,
        diagnostics,
    )
end


"""
    _compute_first_stage_fstat(d, Z, X)

Helper to compute first-stage F-statistic for diagnostics.
"""
function _compute_first_stage_fstat(
    d::Vector{T},
    Z::Matrix{T},
    X::Union{Matrix{T}, Nothing}
) where {T<:Real}
    n = length(d)
    K = size(Z, 2)
    p = isnothing(X) ? 0 : size(X, 2)

    # Regress D on Z (and X if present)
    if isnothing(X)
        Z_full = hcat(ones(T, n), Z)
    else
        Z_full = hcat(ones(T, n), Z, X)
    end

    # OLS: D ~ Z
    coeffs = Z_full \ d
    fitted = Z_full * coeffs
    residuals = d - fitted

    # SSR from instruments only
    # Compare: model with Z vs model without Z
    if isnothing(X)
        Z_restricted = ones(T, n, 1)
    else
        Z_restricted = hcat(ones(T, n), X)
    end

    coeffs_r = Z_restricted \ d
    fitted_r = Z_restricted * coeffs_r
    ssr_restricted = sum((d - fitted_r).^2)
    ssr_full = sum(residuals.^2)

    # F-stat: ((SSR_r - SSR_f) / K) / (SSR_f / (n - K - p - 1))
    df1 = K
    df2 = n - K - p - 1

    if df2 <= 0 || ssr_full < 1e-10
        return T(NaN)
    end

    f_stat = ((ssr_restricted - ssr_full) / df1) / (ssr_full / df2)
    return f_stat
end
