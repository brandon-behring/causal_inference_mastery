"""
Event Study (Dynamic DiD) estimator with leads and lags.

Implements Two-Way Fixed Effects (TWFE) regression with event-time indicators:

```math
Y_{it} = α_i + λ_t + Σ_{k≠-1} β_k·1{EventTime_{it} = k} + ε_{it}
```

where:
- α_i = unit fixed effects
- λ_t = time fixed effects
- k = event time (periods relative to treatment)
- k = -1 is omitted for normalization

This allows estimation of:
- Dynamic treatment effects (β_k for k ≥ 0)
- Pre-treatment trends (β_k for k < -1)
- Joint test of parallel trends (F-test: β_{-2} = β_{-3} = ... = 0)
"""

using LinearAlgebra
using Statistics

"""
    _empty_parallel_trends_test(T, message)

Create empty parallel_trends_test NamedTuple for failure cases.
"""
function _empty_parallel_trends_test(::Type{T}, message::String) where {T<:Real}
    return (
        coefficients_pre=T[],
        coefficients_post=T[],
        se_pre=T[],
        se_post=T[],
        event_times_pre=Int[],
        event_times_post=Int[],
        f_stat=T(NaN),
        f_pvalue=T(NaN),
        f_df1=0,
        f_df2=0,
        pre_trends_pass=false,
        message=message
    )
end

"""
    solve(problem::DiDProblem{T}, estimator::EventStudy) where {T<:Real}

Estimate dynamic treatment effects using Event Study design.

# Algorithm
1. Compute event time for each observation (periods relative to treatment)
2. Create lead/lag indicator variables (omit period -1 for normalization)
3. Demean within units (unit fixed effects)
4. Demean within time periods (time fixed effects)
5. Estimate OLS on demeaned data
6. Compute cluster-robust SEs
7. Joint F-test for pre-treatment coefficients (parallel trends)

# Event Time Calculation
For each unit i, event time is:
```
EventTime_{it} = t - T_i
```
where T_i is the period when unit i receives treatment.

Example:
- Treatment occurs in period 2010
- Period 2008 → EventTime = -2 (2 periods before)
- Period 2010 → EventTime = 0 (treatment period)
- Period 2012 → EventTime = 2 (2 periods after)

# Two-Way Fixed Effects (TWFE)
Instead of including dummy variables for each unit and time period,
we demean the data:

1. Unit FE: Ỹ_it = Y_it - Ȳ_i (subtract unit mean)
2. Time FE: Ỹ_it = Ỹ_it - Ȳ_t (subtract time mean)

This is equivalent to including unit and time dummies but more efficient.

# Examples
```julia
using CausalEstimators

# Minimum wage increase: multiple time periods
# Treatment: Minimum wage increase in 2010 for some states
unit_id = repeat(1:4, inner=5)  # 4 states, 5 periods each
time = repeat(2008:2012, outer=4)
treatment = [true, true, false, false]  # States 1-2 treated
post = time .>= 2010
outcomes = randn(20) .+ 2 .* (treatment .& post)

problem = DiDProblem(
    outcomes,
    repeat(treatment, inner=5),
    post,
    unit_id,
    time,
    (alpha=0.05, cluster_se=true)
)

solution = solve(problem, EventStudy())

println("Pre-treatment coefficients: \$(solution.coefficients_pre)")
println("Post-treatment coefficients: \$(solution.coefficients_post)")
println("Pre-trends F-test p-value: \$(solution.pre_trends_test.p_value)")
```

# Returns
`DiDSolution{T}` with additional fields via `parallel_trends_test`:
- `coefficients_pre`: Lead coefficients (k < 0, excluding k = -1)
- `coefficients_post`: Lag coefficients (k ≥ 0)
- `se_pre`: Standard errors for pre-treatment coefficients
- `se_post`: Standard errors for post-treatment coefficients
- `event_times`: Event time values used
- `f_stat`: Joint F-statistic for pre-trends test
- `f_pvalue`: P-value for joint test
- `pre_trends_pass`: true if joint test p > α

# Interpretation
- **Pre-treatment coefficients** (k < -1): Should be ≈ 0 if parallel trends holds
- **Treatment effect** (k = 0): Immediate impact in treatment period
- **Post-treatment effects** (k > 0): Persistence or dynamics of treatment

# References
- Freyaldenhoven, S., Hansen, C., & Shapiro, J. M. (2019). "Pre-event trends in the panel event-study design." *American Economic Review*, 109(9), 3307-3338.
- Borusyak, K., & Jaravel, X. (2017). "Revisiting event study designs." Available at SSRN 2826228.
"""
function solve(problem::DiDProblem{T}, estimator::EventStudy) where {T<:Real}
    # Require time variable for event study
    if isnothing(problem.time)
        return DiDSolution(
            estimate=T(NaN),
            se=T(NaN),
            ci_lower=T(NaN),
            ci_upper=T(NaN),
            p_value=T(NaN),
            t_stat=T(NaN),
            df=0,
            n_obs=length(problem.outcomes),
            n_treated=sum(problem.treatment),
            n_control=sum(.!problem.treatment),
            parallel_trends_test=nothing,
            retcode=:Failure
        )
    end

    # Get alpha for CI
    alpha = haskey(problem.parameters, :alpha) ? problem.parameters.alpha : 0.05

    # Compute event time for each observation
    event_time_result = _compute_event_time(problem)

    if event_time_result.retcode != :Success
        return DiDSolution(
            estimate=T(NaN),
            se=T(NaN),
            ci_lower=T(NaN),
            ci_upper=T(NaN),
            p_value=T(NaN),
            t_stat=T(NaN),
            df=0,
            n_obs=length(problem.outcomes),
            n_treated=sum(problem.treatment),
            n_control=sum(.!problem.treatment),
            parallel_trends_test=_empty_parallel_trends_test(T, event_time_result.message),
            retcode=:Failure
        )
    end

    event_times = event_time_result.event_times
    unique_event_times = sort(unique(event_times[.!isnan.(event_times)]))

    # Check for control units (needed for identification)
    unique_units = unique(problem.unit_id)
    n_treated_units = length(event_time_result.treatment_periods)
    n_control_units = length(unique_units) - n_treated_units

    if n_control_units == 0
        return DiDSolution(
            estimate=T(NaN),
            se=T(NaN),
            ci_lower=T(NaN),
            ci_upper=T(NaN),
            p_value=T(NaN),
            t_stat=T(NaN),
            df=0,
            n_obs=length(problem.outcomes),
            n_treated=n_treated_units,
            n_control=0,
            parallel_trends_test=_empty_parallel_trends_test(T, "No control units: cannot estimate treatment effects"),
            retcode=:Failure
        )
    end

    # Determine leads and lags to include
    if isnothing(estimator.n_leads)
        # Auto-detect: maximum lead available
        leads = Int.(filter(x -> x < -1, unique_event_times))
        n_leads = length(leads)
    else
        n_leads = estimator.n_leads
        leads = collect(-(n_leads+1):-2)  # Fixed: for n_leads=3 → [-4, -3, -2]
    end

    if isnothing(estimator.n_lags)
        # Auto-detect: maximum lag available
        lags = Int.(filter(x -> x >= 0, unique_event_times))
        n_lags = length(lags)
    else
        n_lags = estimator.n_lags
        lags = collect(0:(n_lags-1))  # Fixed: for n_lags=2 → [0, 1]
    end

    # Create indicator variables for each event time (omit -1)
    event_time_indicators = _create_event_time_indicators(
        event_times, leads, lags, estimator.omit_period
    )

    if size(event_time_indicators, 2) == 0
        return DiDSolution(
            estimate=T(NaN),
            se=T(NaN),
            ci_lower=T(NaN),
            ci_upper=T(NaN),
            p_value=T(NaN),
            t_stat=T(NaN),
            df=0,
            n_obs=length(problem.outcomes),
            n_treated=sum(problem.treatment),
            n_control=sum(.!problem.treatment),
            parallel_trends_test=_empty_parallel_trends_test(T, "No valid event time indicators created"),
            retcode=:Failure
        )
    end

    # Apply two-way fixed effects (demean by unit and time)
    y_demeaned, X_demeaned = _apply_twfe(
        problem.outcomes,
        event_time_indicators,
        problem.unit_id,
        problem.time
    )

    # Estimate OLS on demeaned data
    XtX = X_demeaned' * X_demeaned
    k = size(X_demeaned, 2)

    # Check for singularity using both rank and condition number
    # rank(XtX) < k catches perfect collinearity
    # cond(XtX) > 1e10 catches near-singularity
    if rank(XtX) < k || cond(XtX) > 1e10
        return DiDSolution(
            estimate=T(NaN),
            se=T(NaN),
            ci_lower=T(NaN),
            ci_upper=T(NaN),
            p_value=T(NaN),
            t_stat=T(NaN),
            df=0,
            n_obs=length(problem.outcomes),
            n_treated=sum(problem.treatment),
            n_control=sum(.!problem.treatment),
            parallel_trends_test=_empty_parallel_trends_test(T, "Singular matrix: perfect collinearity or insufficient variation"),
            retcode=:Failure
        )
    end

    # Solve OLS with safety net for singular exceptions
    beta = try
        XtX \ (X_demeaned' * y_demeaned)
    catch e
        if e isa LinearAlgebra.SingularException
            return DiDSolution(
                estimate=T(NaN),
                se=T(NaN),
                ci_lower=T(NaN),
                ci_upper=T(NaN),
                p_value=T(NaN),
                t_stat=T(NaN),
                df=0,
                n_obs=length(problem.outcomes),
                n_treated=sum(problem.treatment),
                n_control=sum(.!problem.treatment),
                parallel_trends_test=_empty_parallel_trends_test(T, "Singular matrix in TWFE regression (matrix inversion failed)"),
                retcode=:Failure
            )
        end
        rethrow(e)  # Re-throw unexpected errors
    end

    # Compute residuals
    residuals = y_demeaned - X_demeaned * beta

    # Compute cluster-robust SEs
    n = length(problem.outcomes)
    k = length(beta)

    if estimator.cluster_se
        se, df = _cluster_robust_se(
            X_demeaned, residuals, problem.unit_id, 1, alpha
        )
        # Fall back to heteroskedasticity-robust if insufficient clusters
        if any(isnan.(se))
            se, df = _heteroskedasticity_robust_se(
                X_demeaned, residuals, n, k, alpha
            )
        end
    else
        se, df = _heteroskedasticity_robust_se(
            X_demeaned, residuals, n, k, alpha
        )
    end

    # Split coefficients into pre and post
    n_pre_coefs = length(leads)
    n_post_coefs = length(lags)

    coefs_pre = beta[1:n_pre_coefs]
    coefs_post = beta[(n_pre_coefs+1):(n_pre_coefs+n_post_coefs)]

    se_pre = se[1:n_pre_coefs]
    se_post = se[(n_pre_coefs+1):(n_pre_coefs+n_post_coefs)]

    # Average treatment effect (mean of post-treatment coefficients)
    ate = mean(coefs_post)
    se_ate = sqrt(sum(se_post.^2)) / length(se_post)  # Approximate SE

    # Confidence interval for ATE
    t_crit = _quantile_tdist(1 - alpha/2, df)
    ci_lower = ate - t_crit * se_ate
    ci_upper = ate + t_crit * se_ate

    # T-statistic and p-value for ATE
    t_stat = ate / se_ate
    p_value = 2 * (1 - _cdf_tdist(abs(t_stat), df))

    # Joint F-test for pre-treatment coefficients (parallel trends)
    pre_trends_test_result = _joint_f_test_pretrends(
        coefs_pre, se_pre, n, k, problem.unit_id, estimator.cluster_se, alpha
    )

    # Count treated/control units
    unique_units = unique(problem.unit_id)
    n_treated = sum([any(problem.treatment[problem.unit_id .== uid]) for uid in unique_units])
    n_control = length(unique_units) - n_treated

    # Construct parallel_trends_test NamedTuple with all event study info
    parallel_trends_test = (
        coefficients_pre=coefs_pre,
        coefficients_post=coefs_post,
        se_pre=se_pre,
        se_post=se_post,
        event_times_pre=leads,
        event_times_post=lags,
        f_stat=pre_trends_test_result.f_stat,
        f_pvalue=pre_trends_test_result.p_value,
        f_df1=pre_trends_test_result.df1,
        f_df2=pre_trends_test_result.df2,
        pre_trends_pass=pre_trends_test_result.p_value > alpha,
        message=pre_trends_test_result.message
    )

    # Return code
    retcode = :Success
    if isnan(ate) || isnan(se_ate) || se_ate == 0
        # NaN or zero SE indicates numerical failure
        retcode = :Failure
    elseif !pre_trends_test_result.passes
        # Non-NaN estimates but pre-trends violated (statistical warning)
        retcode = :Warning
    end

    return DiDSolution(
        estimate=ate,
        se=se_ate,
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
    _compute_event_time(problem::DiDProblem{T}) where {T<:Real}

Compute event time (periods relative to treatment) for each observation.

For treated units, event time = current period - treatment period.
For control units (never treated), event time = NaN.

# Returns
NamedTuple with:
- `event_times`: Vector of event times
- `treatment_periods`: Dict mapping unit_id → treatment period
- `retcode`: :Success or :Failure
- `message`: Status message
"""
function _compute_event_time(problem::DiDProblem{T}) where {T<:Real}
    n = length(problem.outcomes)
    event_times = fill(T(NaN), n)

    # Find treatment period for each treated unit
    treatment_periods = Dict{Int,Int}()

    for uid in unique(problem.unit_id)
        unit_mask = problem.unit_id .== uid
        is_treated = any(problem.treatment[unit_mask])

        if is_treated
            # Find first period where post=true for this unit
            unit_times = problem.time[unit_mask]
            unit_post = problem.post[unit_mask]

            # Treatment period is first period where post=true
            post_periods = unit_times[unit_post]

            if length(post_periods) == 0
                # Treated but never observed in post period - invalid
                return (
                    event_times=event_times,
                    treatment_periods=treatment_periods,
                    retcode=:Failure,
                    message="Treated unit $uid never observed in post period"
                )
            end

            treatment_period = minimum(post_periods)
            treatment_periods[uid] = treatment_period

            # Compute event time for all observations of this unit
            for i in findall(unit_mask)
                event_times[i] = problem.time[i] - treatment_period
            end
        end
        # Control units keep NaN event times
    end

    if length(treatment_periods) == 0
        return (
            event_times=event_times,
            treatment_periods=treatment_periods,
            retcode=:Failure,
            message="No treated units found with valid treatment periods"
        )
    end

    return (
        event_times=event_times,
        treatment_periods=treatment_periods,
        retcode=:Success,
        message="Event times computed successfully"
    )
end

"""
    _create_event_time_indicators(event_times, leads, lags, omit_period)

Create indicator variables for each event time period.

# Arguments
- `event_times`: Vector of event times for each observation
- `leads`: Vector of lead periods to include (e.g., [-3, -2])
- `lags`: Vector of lag periods to include (e.g., [0, 1, 2, 3])
- `omit_period`: Period to omit for normalization (default: -1)

# Returns
Matrix where each column is an indicator for one event time.
Columns ordered: leads (ascending), then lags (ascending).
"""
function _create_event_time_indicators(event_times::Vector{T},
                                      leads::Vector{Int},
                                      lags::Vector{Int},
                                      omit_period::Int) where {T<:Real}
    n = length(event_times)

    # Remove omit_period from leads/lags if present
    leads_filtered = filter(x -> x != omit_period, leads)
    lags_filtered = filter(x -> x != omit_period, lags)

    all_periods = [leads_filtered; lags_filtered]
    n_indicators = length(all_periods)

    if n_indicators == 0
        return Matrix{T}(undef, n, 0)
    end

    # Create indicator matrix
    X = zeros(T, n, n_indicators)

    for (col_idx, period) in enumerate(all_periods)
        for i in 1:n
            if !isnan(event_times[i]) && event_times[i] == period
                X[i, col_idx] = 1.0
            end
        end
    end

    return X
end

"""
    _apply_twfe(y, X, unit_id, time)

Apply Two-Way Fixed Effects by demeaning.

# Algorithm
1. Demean within units: y_tilde = y - unit_mean
2. Demean within time: y_tilde = y_tilde - time_mean
3. Apply same transformation to X

This is equivalent to including unit and time fixed effects but more efficient.

# Returns
- `y_demeaned`: Demeaned outcome
- `X_demeaned`: Demeaned covariates
"""
function _apply_twfe(y::Vector{T}, X::Matrix{T},
                    unit_id::Vector{Int}, time::Vector{Int}) where {T<:Real}
    n = length(y)
    k = size(X, 2)

    # Initialize demeaned arrays
    y_demeaned = copy(y)
    X_demeaned = copy(X)

    # Demean by unit
    for uid in unique(unit_id)
        unit_mask = unit_id .== uid
        y_unit_mean = mean(y[unit_mask])
        X_unit_mean = mean(X[unit_mask, :], dims=1)

        y_demeaned[unit_mask] .-= y_unit_mean
        X_demeaned[unit_mask, :] .-= X_unit_mean
    end

    # Demean by time
    for t in unique(time)
        time_mask = time .== t
        y_time_mean = mean(y_demeaned[time_mask])
        X_time_mean = mean(X_demeaned[time_mask, :], dims=1)

        y_demeaned[time_mask] .-= y_time_mean
        X_demeaned[time_mask, :] .-= X_time_mean
    end

    return y_demeaned, X_demeaned
end

"""
    _joint_f_test_pretrends(coefs_pre, se_pre, n, k, cluster_id, cluster_se, alpha)

Joint F-test for pre-treatment coefficients.

Tests H₀: β_{-2} = β_{-3} = ... = β_{-m} = 0 (parallel trends).

# Algorithm
For cluster-robust:
- Use Wald test with cluster-robust variance matrix
- F = (β' V^{-1} β) / m, where m = number of pre-treatment coefficients

For heteroskedasticity-robust:
- F = (β' V^{-1} β) / m with HC1 variance

# Returns
NamedTuple with:
- `f_stat`: F-statistic
- `p_value`: P-value
- `df1`: Numerator degrees of freedom (m)
- `df2`: Denominator degrees of freedom
- `passes`: true if p > alpha
- `message`: Interpretation
"""
function _joint_f_test_pretrends(coefs_pre::Vector{T}, se_pre::Vector{T},
                                n::Int, k::Int, cluster_id::Vector{Int},
                                cluster_se::Bool, alpha::T) where {T<:Real}
    m = length(coefs_pre)

    if m == 0
        return (
            f_stat=T(NaN),
            p_value=T(NaN),
            df1=0,
            df2=0,
            passes=true,
            message="No pre-treatment periods to test"
        )
    end

    # Construct variance matrix (diagonal from standard errors)
    # Note: This is a simplification - full implementation would need
    # covariance matrix, not just diagonal
    V = diagm(se_pre.^2)

    # Check for zero standard errors (singular V matrix)
    # This happens when outcome has no variation
    if any(se_pre .== 0.0)
        return (
            f_stat=T(NaN),
            p_value=T(NaN),
            df1=m,
            df2=0,
            passes=false,
            message="Cannot test pre-trends: zero standard errors (no variation in outcome)"
        )
    end

    # Wald statistic: W = β' V^{-1} β
    V_inv = inv(V)
    wald_stat = coefs_pre' * V_inv * coefs_pre

    # Convert to F-statistic: F = W / m
    f_stat = wald_stat / m

    # Degrees of freedom
    df1 = m  # Numerator DF (number of restrictions)

    if cluster_se
        n_clusters = length(unique(cluster_id))
        df2 = n_clusters - k  # Denominator DF (conservative)
    else
        df2 = n - k
    end

    # P-value from F-distribution
    p_value = _pvalue_fdist(f_stat, df1, df2)

    # Test passes if we fail to reject parallel trends
    passes = p_value > alpha

    message = passes ? "Parallel trends supported (joint F-test p > α)" :
                      "Parallel trends violated (joint F-test p ≤ α)"

    return (
        f_stat=f_stat,
        p_value=p_value,
        df1=df1,
        df2=df2,
        passes=passes,
        message=message
    )
end

"""
    _pvalue_fdist(f_stat, df1, df2)

Compute p-value for F-distribution.

For production, use Distributions.jl: ccdf(FDist(df1, df2), f_stat)

This is a simplified approximation.
"""
function _pvalue_fdist(f_stat::T, df1::Int, df2::Int) where {T<:Real}
    if f_stat <= 0
        return T(1.0)
    end

    # For large df2, F ≈ χ²/df1
    # Use chi-squared approximation
    chi2_stat = f_stat * df1

    # Approximate p-value using normal approximation to chi-squared
    # For large df, χ² ≈ N(df, 2df)
    z = (chi2_stat - df1) / sqrt(2 * df1)
    p_value = 1 - _cdf_normal(z)

    return max(T(0.0), min(T(1.0), p_value))
end
