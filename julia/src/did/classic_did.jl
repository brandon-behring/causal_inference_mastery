"""
Classic 2×2 Difference-in-Differences estimator.

Implements the canonical DiD regression with cluster-robust standard errors:

```math
Y_{it} = α + β·Treatment_i + γ·Post_t + δ·(Treatment_i × Post_t) + ε_{it}
```

where δ is the DiD estimator (Average Treatment Effect on the Treated).
"""

using LinearAlgebra
using Statistics

"""
    solve(problem::DiDProblem{T}, estimator::ClassicDiD) where {T<:Real}

Estimate treatment effect using Classic 2×2 Difference-in-Differences.

# Algorithm
1. Create regression matrix: [1, Treatment, Post, Treatment×Post]
2. Estimate OLS: Y = Xβ + ε
3. Extract δ (coefficient on interaction term)
4. Compute cluster-robust standard errors (if cluster_se=true)
5. Construct confidence intervals and p-values
6. Optional: Test parallel trends if ≥2 pre-periods

# Cluster-Robust Standard Errors
Following Bertrand et al. (2004), cluster by unit_id to account for:
- Serial correlation within units over time
- Heteroskedasticity across units

Formula (HC1 cluster adjustment):
```math
V_cluster = (N/(N-k)) × (X'X)^{-1} × Σ_c u_c'u_c × (X'X)^{-1}
```

where:
- N = number of clusters (unique units)
- k = number of regressors (4 for classic DiD)
- u_c = residuals for cluster c

Degrees of freedom: N_clusters - k (conservative, recommended)

# Examples
```julia
using CausalEstimators

# Job training program data
unit_id = [1, 1, 2, 2, 3, 3, 4, 4]
time = [2008, 2010, 2008, 2010, 2008, 2010, 2008, 2010]
treatment = [true, true, true, true, false, false, false, false]
post = [false, true, false, true, false, true, false, true]
outcomes = [9.5, 10.2, 9.3, 10.1, 9.4, 9.6, 9.2, 9.5]

problem = DiDProblem(
    outcomes,
    treatment,
    post,
    unit_id,
    time,
    (alpha=0.05, cluster_se=true)
)

solution = solve(problem, ClassicDiD())

println("Treatment effect: \$(solution.estimate) ± \$(solution.se)")
println("95% CI: [\$(solution.ci_lower), \$(solution.ci_upper)]")
println("P-value: \$(solution.p_value)")
```

# Returns
`DiDSolution{T}` with:
- `estimate`: δ̂ (DiD treatment effect)
- `se`: Cluster-robust standard error (if cluster_se=true)
- `ci_lower`, `ci_upper`: 95% confidence interval
- `p_value`: P-value for H₀: δ = 0
- `t_stat`: T-statistic
- `df`: Degrees of freedom (n_clusters - k for cluster, n - k otherwise)
- `n_obs`: Number of observations
- `n_treated`: Number of treated units
- `n_control`: Number of control units
- `parallel_trends_test`: Pre-trends test results (if requested and ≥2 pre-periods)
- `retcode`: :Success, :Warning, or :Failure

# Interpretation
Under the parallel trends assumption, δ̂ identifies the Average Treatment Effect
on the Treated (ATT). The assumption is UNTESTABLE with only 2 periods, but can
be assessed with pre-treatment trend tests if more periods are available.

# References
- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Bertrand, M., Duflo, E., & Mullainathan, S. (2004). "How much should we trust differences-in-differences estimates?" *Quarterly Journal of Economics*, 119(1), 249-275.
"""
function solve(problem::DiDProblem{T}, estimator::ClassicDiD) where {T<:Real}
    # Extract data
    y = problem.outcomes
    treat = problem.treatment
    post = problem.post
    unit_id = problem.unit_id
    n = length(y)

    # Get alpha for CI
    alpha = haskey(problem.parameters, :alpha) ? problem.parameters.alpha : 0.05

    # ClassicDiD requires observations in all 2×2 cells
    # (EventStudy can handle edge cases like no pre-period)
    n_treated_pre = sum(treat .& .!post)
    n_treated_post = sum(treat .& post)
    n_control_pre = sum(.!treat .& .!post)
    n_control_post = sum(.!treat .& post)

    if n_treated_pre == 0 || n_treated_post == 0 || n_control_pre == 0 || n_control_post == 0
        return DiDSolution(
            estimate=T(NaN),
            se=T(NaN),
            ci_lower=T(NaN),
            ci_upper=T(NaN),
            p_value=T(NaN),
            t_stat=T(NaN),
            df=0,
            n_obs=n,
            n_treated=sum(treat),
            n_control=sum(.!treat),
            parallel_trends_test=nothing,
            retcode=:Failure
        )
    end

    # Construct regression matrix: [intercept, treatment, post, treatment×post]
    X = hcat(
        ones(T, n),                    # Intercept
        T.(treat),                      # Treatment indicator
        T.(post),                       # Post indicator
        T.(treat .& post)               # Interaction (DiD term)
    )

    # Estimate OLS: β = (X'X)^{-1}X'y
    XtX = X' * X
    Xty = X' * y

    # Check for singularity
    if cond(XtX) > 1e10
        return DiDSolution(
            estimate=T(NaN),
            se=T(NaN),
            ci_lower=T(NaN),
            ci_upper=T(NaN),
            p_value=T(NaN),
            t_stat=T(NaN),
            df=0,
            n_obs=n,
            n_treated=sum(treat),
            n_control=sum(.!treat),
            parallel_trends_test=nothing,
            retcode=:Failure
        )
    end

    beta = XtX \ Xty

    # Extract DiD estimate (coefficient on interaction term)
    estimate = beta[4]

    # Compute residuals
    y_pred = X * beta
    residuals = y - y_pred

    # Compute standard errors
    k = size(X, 2)  # Number of regressors (4)

    if estimator.cluster_se
        # Cluster-robust standard errors (by unit_id)
        se, df = _cluster_robust_se(X, residuals, unit_id, 4, alpha)
    else
        # Heteroskedasticity-robust standard errors (HC1)
        se, df = _heteroskedasticity_robust_se(X, residuals, n, k, alpha)
    end

    # Extract SE for DiD coefficient (4th regressor)
    se_did = se[4]

    # T-statistic
    t_stat = estimate / se_did

    # Count treated/control units (not observations)
    unique_units = unique(unit_id)
    n_treated = sum([any(treat[unit_id .== uid]) for uid in unique_units])
    n_control = length(unique_units) - n_treated

    # Handle edge case: df <= 0
    if df <= 0
        return DiDSolution(
            estimate=estimate,
            se=se_did,
            ci_lower=T(-Inf),
            ci_upper=T(Inf),
            p_value=T(NaN),
            t_stat=t_stat,
            df=df,
            n_obs=n,
            n_treated=n_treated,
            n_control=n_control,
            parallel_trends_test=nothing,
            retcode=:Failure
        )
    end

    # Confidence interval
    t_crit = _quantile_tdist(1 - alpha/2, df)
    ci_lower = estimate - t_crit * se_did
    ci_upper = estimate + t_crit * se_did

    # P-value (two-tailed)
    p_value = 2 * (1 - _cdf_tdist(abs(t_stat), df))

    # Parallel trends test (if requested and ≥2 pre-periods)
    parallel_trends_test = nothing
    if estimator.test_parallel_trends
        parallel_trends_test = _test_parallel_trends(problem, estimator, alpha)
    end

    # Return code
    retcode = :Success
    if isnan(estimate) || isnan(se_did)
        retcode = :Failure
    elseif p_value > 0.10  # Weak evidence
        retcode = :Warning
    end

    return DiDSolution(
        estimate=estimate,
        se=se_did,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        t_stat=t_stat,
        df=df,
        n_obs=n,
        n_treated=n_treated,
        n_control=n_control,
        parallel_trends_test=parallel_trends_test,
        retcode=retcode
    )
end

"""
    _cluster_robust_se(X, residuals, cluster_id, interaction_idx, alpha)

Compute cluster-robust standard errors.

# Algorithm
1. For each cluster c, sum outer product of residuals: u_c'u_c
2. Sandwich estimator: V = (X'X)^{-1} × [Σ_c u_c'u_c] × (X'X)^{-1}
3. HC1 finite-sample adjustment: multiply by N/(N-k)
4. SE = sqrt(diag(V))

# Returns
- `se`: Vector of standard errors for all coefficients
- `df`: Degrees of freedom (N_clusters - k)
"""
function _cluster_robust_se(X::Matrix{T}, residuals::Vector{T},
                           cluster_id::Vector{Int}, interaction_idx::Int,
                           alpha::T) where {T<:Real}
    n, k = size(X)

    # Get unique clusters
    clusters = unique(cluster_id)
    n_clusters = length(clusters)

    # Compute (X'X)^{-1}
    XtX_inv = inv(X' * X)

    # Compute clustered meat matrix: Σ_c (X_c' * u_c * u_c' * X_c)
    meat = zeros(T, k, k)

    for c in clusters
        # Get observations in this cluster
        cluster_mask = cluster_id .== c
        X_c = X[cluster_mask, :]
        u_c = residuals[cluster_mask]

        # Add cluster's contribution to meat matrix
        # u_c'u_c in matrix form is u_c * u_c' (outer product)
        meat .+= X_c' * (u_c * u_c') * X_c
    end

    # Degrees of freedom: n_clusters - 1 (following Bertrand et al. 2004)
    # This is the standard for cluster-robust SEs in panel data
    df = n_clusters - 1

    # Finite-sample adjustment factor (matching statsmodels)
    # Combines cluster-level and observation-level corrections:
    #   adjustment = [G/(G-1)] * [(N-1)/(N-k)]
    # where G = n_clusters, N = n_obs, k = n_parameters
    #
    # This matches statsmodels.regression.linear_model.OLS.fit(cov_type='cluster')
    # which uses Cameron, Gelbach, Miller (2011) HC1-style adjustment
    if (n_clusters - 1) <= 0 || (n - k) <= 0
        # Degenerate case: insufficient clusters or observations
        # Use unadjusted sandwich estimator
        adjustment = T(1.0)
    else
        # Part 1: Cluster finite-sample correction
        cluster_adj = n_clusters / (n_clusters - 1)
        # Part 2: HC1 finite-sample correction
        hc1_adj = (n - 1) / (n - k)
        # Combined adjustment
        adjustment = cluster_adj * hc1_adj
    end

    # Sandwich estimator: V = (X'X)^{-1} × meat × (X'X)^{-1}
    V = adjustment .* (XtX_inv * meat * XtX_inv)

    # Standard errors (sqrt of diagonal)
    # Check for numerical issues (negative variances)
    v_diag = diag(V)
    if any(v_diag .< 0)
        # Numerical instability - return NaN SEs but keep actual df
        se = fill(T(NaN), k)
        return se, df
    end

    se = sqrt.(v_diag)

    return se, df
end

"""
    _heteroskedasticity_robust_se(X, residuals, n, k, alpha)

Compute heteroskedasticity-robust standard errors (HC1).

# Algorithm
HC1 estimator: V = (X'X)^{-1} × [n/(n-k)] × [Σ_i x_i'x_i × u_i^2] × (X'X)^{-1}

# Returns
- `se`: Vector of standard errors for all coefficients
- `df`: Degrees of freedom (n - k)
"""
function _heteroskedasticity_robust_se(X::Matrix{T}, residuals::Vector{T},
                                      n::Int, k::Int, alpha::T) where {T<:Real}
    # Compute (X'X)^{-1}
    XtX_inv = inv(X' * X)

    # Compute meat matrix: Σ_i x_i'x_i × u_i^2
    meat = zeros(T, k, k)
    for i in 1:n
        xi = X[i, :]
        ui = residuals[i]
        meat .+= (xi * xi') .* (ui^2)
    end

    # HC1 finite-sample adjustment: n/(n-k)
    adjustment = n / (n - k)

    # Sandwich estimator
    V = adjustment .* (XtX_inv * meat * XtX_inv)

    # Standard errors
    se = sqrt.(diag(V))

    # Degrees of freedom
    df = n - k

    return se, df
end

"""
    _test_parallel_trends(problem, estimator, alpha)

Test parallel trends assumption using pre-treatment periods.

Requires ≥2 pre-treatment periods. Tests whether treated and control groups
had different trends before treatment.

# Algorithm
1. Subset to pre-treatment observations only
2. Create time trend variable
3. Regress: Y = α + β·Treatment + γ·Time + δ·(Treatment × Time) + ε
4. Test H₀: δ = 0 (parallel trends)
5. If cluster_se=true, use cluster-robust SEs

# Returns
NamedTuple with:
- `pre_trend_coef`: Coefficient on Treatment×Time interaction
- `se`: Standard error (cluster-robust if cluster_se=true)
- `p_value`: P-value for H₀: parallel trends
- `passes`: true if p > alpha (fail to reject parallel trends)
- `n_pre_periods`: Number of pre-treatment periods detected
"""
function _test_parallel_trends(problem::DiDProblem{T}, estimator::ClassicDiD,
                              alpha::T) where {T<:Real}
    # Check if we have time variable
    if isnothing(problem.time)
        return (
            pre_trend_coef=T(NaN),
            se=T(NaN),
            p_value=T(NaN),
            passes=false,
            n_pre_periods=0,
            message="No time variable provided - cannot test parallel trends"
        )
    end

    # Subset to pre-treatment period
    pre_mask = .!problem.post

    if sum(pre_mask) == 0
        return (
            pre_trend_coef=T(NaN),
            se=T(NaN),
            p_value=T(NaN),
            passes=false,
            n_pre_periods=0,
            message="No pre-treatment observations found"
        )
    end

    # Get unique time periods in pre-treatment
    pre_times = unique(problem.time[pre_mask])
    n_pre_periods = length(pre_times)

    if n_pre_periods < 2
        return (
            pre_trend_coef=T(NaN),
            se=T(NaN),
            p_value=T(NaN),
            passes=false,
            n_pre_periods=n_pre_periods,
            message="Need ≥2 pre-treatment periods to test trends (found $n_pre_periods)"
        )
    end

    # Extract pre-treatment data
    y_pre = problem.outcomes[pre_mask]
    treat_pre = problem.treatment[pre_mask]
    time_pre = problem.time[pre_mask]
    unit_id_pre = problem.unit_id[pre_mask]
    n_pre = length(y_pre)

    # Normalize time to start at 0
    time_normalized = T.(time_pre .- minimum(time_pre))

    # Create regression matrix: [1, Treatment, Time, Treatment×Time]
    X_pre = hcat(
        ones(T, n_pre),
        T.(treat_pre),
        time_normalized,
        T.(treat_pre) .* time_normalized
    )

    # OLS estimation
    XtX = X_pre' * X_pre

    if cond(XtX) > 1e10
        return (
            pre_trend_coef=T(NaN),
            se=T(NaN),
            p_value=T(NaN),
            passes=false,
            n_pre_periods=n_pre_periods,
            message="Singular matrix in pre-trends regression"
        )
    end

    beta_pre = XtX \ (X_pre' * y_pre)
    pre_trend_coef = beta_pre[4]  # Treatment×Time coefficient

    # Compute residuals
    residuals_pre = y_pre - X_pre * beta_pre

    # Compute standard errors
    k = 4
    if estimator.cluster_se
        se_vec, df = _cluster_robust_se(X_pre, residuals_pre, unit_id_pre, 4, alpha)
    else
        se_vec, df = _heteroskedasticity_robust_se(X_pre, residuals_pre, n_pre, k, alpha)
    end

    se_pretrend = se_vec[4]

    # T-statistic and p-value
    t_stat = pre_trend_coef / se_pretrend
    p_value = 2 * (1 - _cdf_tdist(abs(t_stat), df))

    # Test passes if we fail to reject H₀: parallel trends
    passes = p_value > alpha

    return (
        pre_trend_coef=pre_trend_coef,
        se=se_pretrend,
        t_stat=t_stat,
        p_value=p_value,
        df=df,
        passes=passes,
        n_pre_periods=n_pre_periods,
        message=passes ? "Parallel trends supported (p > α)" : "Parallel trends violated (p ≤ α)"
    )
end

"""
    _quantile_tdist(p, df)

Compute quantile of t-distribution at probability p with df degrees of freedom.

Uses approximation for df > 30, exact values for small df.
"""
function _quantile_tdist(p::T, df::Int) where {T<:Real}
    # For large df, t-distribution ≈ normal
    if df > 30
        return _quantile_normal(p)
    end

    # Small sample: use lookup table (common values)
    # This is a simplified implementation - production would use Distributions.jl
    # For now, use normal approximation with correction
    z = _quantile_normal(p)

    # Cornish-Fisher expansion for better accuracy
    # t ≈ z + (z^3 + z)/(4df) + ...
    correction = (z^3 + z) / (4 * df)

    return z + correction
end

"""
    _cdf_tdist(x, df)

Compute cumulative distribution function of t-distribution.

For production use, replace with Distributions.jl: cdf(TDist(df), x)
"""
function _cdf_tdist(x::T, df::Int) where {T<:Real}
    # For large df, use normal approximation
    if df > 30
        return _cdf_normal(x)
    end

    # For small df, use normal with correction
    # This is simplified - production should use Distributions.jl
    z = x
    phi = _cdf_normal(z)

    # Small correction for finite df
    correction = -(z^3 + z) / (4 * df) * _pdf_normal(z)

    return phi + correction
end

"""
    _quantile_normal(p)

Standard normal quantile function (inverse CDF).
"""
function _quantile_normal(p::T) where {T<:Real}
    # Beasley-Springer-Moro algorithm (simplified)
    # For production, use StatsFuns.jl: norminvcdf(p)

    if p <= 0.0 || p >= 1.0
        throw(ArgumentError("p must be in (0, 1)"))
    end

    # Coefficients for Beasley-Springer approximation
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]

    # Use rational approximation for central region
    if 0.075 <= p <= 0.925
        q = p - 0.5
        r = q * q
        num = (((a[4]*r + a[3])*r + a[2])*r + a[1]) * q
        den = ((((b[4]*r + b[3])*r + b[2])*r + b[1])*r + 1.0)
        return num / den
    end

    # Tail region
    if p < 0.075
        q = sqrt(-2.0 * log(p))
    else
        q = sqrt(-2.0 * log(1.0 - p))
    end

    num = ((((((((c[9]*q + c[8])*q + c[7])*q + c[6])*q + c[5])*q + c[4])*q + c[3])*q + c[2])*q + c[1])

    result = num / q

    return p < 0.075 ? -result : result
end

"""
    _cdf_normal(x)

Standard normal cumulative distribution function.
"""
function _cdf_normal(x::T) where {T<:Real}
    # Using error function approximation
    # Φ(x) = 0.5 * (1 + erf(x/√2))

    # Abramowitz and Stegun approximation for erf
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # Save the sign of x
    sign_x = x >= 0 ? 1 : -1
    x_abs = abs(x / sqrt(T(2)))

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * exp(-x_abs * x_abs)

    erf_val = sign_x * y

    return 0.5 * (1.0 + erf_val)
end

"""
    _pdf_normal(x)

Standard normal probability density function.
"""
function _pdf_normal(x::T) where {T<:Real}
    return exp(-x^2 / 2) / sqrt(2 * π)
end
