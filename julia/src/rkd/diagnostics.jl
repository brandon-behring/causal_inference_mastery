"""
Diagnostics for Regression Kink Design (RKD).

Implements validation tests for RKD assumptions:
1. Density smoothness - no bunching at the kink
2. Covariate smoothness - predetermined covariates continuous at kink
3. First stage strength - sufficient variation in treatment at kink

References:
- Card et al. (2015) - Generalized RKD inference
- McCrary (2008) - Manipulation test (adapted for kinks)
"""

# =============================================================================
# Result Types
# =============================================================================

"""
    DensitySmoothnessResult

Result of density smoothness test at kink.

# Fields
- `slope_left::Float64`: Estimated density slope on left of kink
- `slope_right::Float64`: Estimated density slope on right of kink
- `slope_difference::Float64`: Difference in slopes
- `se::Float64`: Standard error of the slope difference
- `t_stat::Float64`: T-statistic
- `p_value::Float64`: P-value (H0: slopes are equal)
- `n_bins::Int`: Number of bins used
- `interpretation::String`: Human-readable interpretation
"""
struct DensitySmoothnessResult
    slope_left::Float64
    slope_right::Float64
    slope_difference::Float64
    se::Float64
    t_stat::Float64
    p_value::Float64
    n_bins::Int
    interpretation::String
end

"""
    CovariateSmoothnessResult

Result of covariate smoothness test.

# Fields
- `covariate_name::String`: Name of the covariate tested
- `slope_left::Float64`: Estimated covariate slope on left of kink
- `slope_right::Float64`: Estimated covariate slope on right of kink
- `slope_difference::Float64`: Difference in slopes
- `se::Float64`: Standard error
- `t_stat::Float64`: T-statistic
- `p_value::Float64`: P-value (H0: smooth at kink)
- `is_smooth::Bool`: Whether the covariate passes the smoothness test
"""
struct CovariateSmoothnessResult
    covariate_name::String
    slope_left::Float64
    slope_right::Float64
    slope_difference::Float64
    se::Float64
    t_stat::Float64
    p_value::Float64
    is_smooth::Bool
end

"""
    FirstStageTestResult

Result of first stage strength test.

# Fields
- `kink_estimate::Float64`: Estimated kink in treatment
- `se::Float64`: Standard error
- `f_stat::Float64`: F-statistic (kink_estimate² / se²)
- `p_value::Float64`: P-value for H0: no kink
- `is_strong::Bool`: Whether first stage is strong (F > 10)
- `interpretation::String`: Human-readable interpretation
"""
struct FirstStageTestResult
    kink_estimate::Float64
    se::Float64
    f_stat::Float64
    p_value::Float64
    is_strong::Bool
    interpretation::String
end

"""
    RKDDiagnosticsSummary

Comprehensive RKD diagnostics summary.

# Fields
- `density_test::DensitySmoothnessResult`: Density smoothness test result
- `first_stage_test::FirstStageTestResult`: First stage test result
- `covariate_tests::Vector{CovariateSmoothnessResult}`: Covariate test results
- `density_smooth::Bool`: Whether density is smooth
- `first_stage_strong::Bool`: Whether first stage is strong
- `covariates_smooth::Bool`: Whether all covariates are smooth
- `all_pass::Bool`: Whether all diagnostics pass
"""
struct RKDDiagnosticsSummary
    density_test::DensitySmoothnessResult
    first_stage_test::FirstStageTestResult
    covariate_tests::Vector{CovariateSmoothnessResult}
    density_smooth::Bool
    first_stage_strong::Bool
    covariates_smooth::Bool
    all_pass::Bool
end

# =============================================================================
# Diagnostic Functions
# =============================================================================

"""
    density_smoothness_test(x, cutoff; n_bins=20, bandwidth=nothing)

Test for smoothness in the density of the running variable at the kink.

Unlike McCrary's test for RDD (which tests for a jump in density level),
this tests for a kink in the density function - i.e., whether the
derivative of the density changes at the cutoff.

# Arguments
- `x`: Running variable
- `cutoff`: Kink point
- `n_bins`: Number of bins on each side (default: 20)
- `bandwidth`: Bandwidth for density estimation (nothing = Silverman's rule)

# Returns
- `DensitySmoothnessResult`: Test results

# Notes
H0: The density is smooth at the kink (no bunching)
H1: There is a kink in the density (bunching/manipulation)

A significant result (p < 0.05) suggests potential manipulation.
"""
function density_smoothness_test(
    x::AbstractVector{T},
    cutoff::Real;
    n_bins::Int=20,
    bandwidth::Union{Nothing,Real}=nothing
) where {T<:Real}

    # Silverman's rule for bandwidth
    if isnothing(bandwidth)
        sigma = std(x)
        iqr = quantile(x, 0.75) - quantile(x, 0.25)
        bandwidth = 0.9 * min(sigma, iqr / 1.34) * length(x)^(-0.2)
    end

    # Split data
    x_left = x[x .< cutoff]
    x_right = x[x .>= cutoff]

    if length(x_left) < 20 || length(x_right) < 20
        return DensitySmoothnessResult(
            NaN, NaN, NaN, NaN, NaN, 1.0, n_bins,
            "Insufficient data for density test"
        )
    end

    # Create bins
    left_edges = range(minimum(x_left), cutoff, length=n_bins + 1)
    right_edges = range(cutoff, maximum(x_right), length=n_bins + 1)

    # Bin counts
    left_counts = [sum((x_left .>= left_edges[i]) .& (x_left .< left_edges[i+1]))
                   for i in 1:n_bins]
    right_counts = [sum((x_right .>= right_edges[i]) .& (x_right .< right_edges[i+1]))
                    for i in 1:n_bins]

    # Bin centers
    left_centers = [(left_edges[i] + left_edges[i+1]) / 2 for i in 1:n_bins]
    right_centers = [(right_edges[i] + right_edges[i+1]) / 2 for i in 1:n_bins]

    # Convert to density
    left_widths = [left_edges[i+1] - left_edges[i] for i in 1:n_bins]
    right_widths = [right_edges[i+1] - right_edges[i] for i in 1:n_bins]
    left_density = left_counts ./ (left_widths .* length(x_left))
    right_density = right_counts ./ (right_widths .* length(x_right))

    # Filter out zero counts
    left_valid = left_density .> 0
    right_valid = right_density .> 0

    if sum(left_valid) < 3 || sum(right_valid) < 3
        return DensitySmoothnessResult(
            NaN, NaN, NaN, NaN, NaN, 1.0, n_bins,
            "Insufficient non-zero bins for density test"
        )
    end

    # Fit slopes to log density
    x_left_c = (left_centers[left_valid] .- cutoff)
    y_left = log.(left_density[left_valid])

    x_right_c = (right_centers[right_valid] .- cutoff)
    y_right = log.(right_density[right_valid])

    # Simple OLS for slopes
    slope_left, se_left = _fit_ols_slope(x_left_c, y_left)
    slope_right, se_right = _fit_ols_slope(x_right_c, y_right)

    # Test for difference in slopes
    slope_diff = slope_right - slope_left
    se_diff = sqrt(se_left^2 + se_right^2)

    if se_diff > 0
        t_stat = slope_diff / se_diff
        df = sum(left_valid) + sum(right_valid) - 4
        df = max(df, 1)
        p_value = 2 * (1 - cdf(TDist(df), abs(t_stat)))
    else
        t_stat = slope_diff != 0 ? Inf * sign(slope_diff) : 0.0
        p_value = slope_diff != 0 ? 0.0 : 1.0
    end

    # Interpretation
    if p_value < 0.01
        interpretation = "Strong evidence of density kink (p=$(round(p_value, digits=4))). Potential manipulation."
    elseif p_value < 0.05
        interpretation = "Evidence of density kink (p=$(round(p_value, digits=4))). Possible manipulation."
    elseif p_value < 0.10
        interpretation = "Weak evidence of density kink (p=$(round(p_value, digits=4))). Caution advised."
    else
        interpretation = "No evidence of density kink (p=$(round(p_value, digits=4))). Smoothness assumption supported."
    end

    return DensitySmoothnessResult(
        slope_left, slope_right, slope_diff, se_diff, t_stat, p_value, n_bins, interpretation
    )
end

"""
    covariate_smoothness_test(x, covariates, cutoff; covariate_names=nothing, bandwidth=nothing)

Test for smoothness of predetermined covariates at the kink.

If the RKD is valid, predetermined covariates should vary smoothly
at the kink - they should not exhibit a kink in their relationship
with the running variable.

# Arguments
- `x`: Running variable
- `covariates`: Matrix of k covariates to test (n × k)
- `cutoff`: Kink point
- `covariate_names`: Names for each covariate (optional)
- `bandwidth`: Bandwidth for local regression (optional)

# Returns
- `Vector{CovariateSmoothnessResult}`: Test results for each covariate
"""
function covariate_smoothness_test(
    x::AbstractVector{T},
    covariates::AbstractMatrix{T},
    cutoff::Real;
    covariate_names::Union{Nothing,Vector{String}}=nothing,
    bandwidth::Union{Nothing,Real}=nothing
) where {T<:Real}

    n, k = size(covariates)

    if isnothing(covariate_names)
        covariate_names = ["Covariate_$i" for i in 1:k]
    end

    if isnothing(bandwidth)
        sigma = std(x)
        iqr = quantile(x, 0.75) - quantile(x, 0.25)
        bandwidth = 1.5 * min(sigma, iqr / 1.34) * n^(-0.2)
    end

    results = CovariateSmoothnessResult[]

    for j in 1:k
        cov = covariates[:, j]
        name = j <= length(covariate_names) ? covariate_names[j] : "Covariate_$j"

        # Filter to bandwidth region
        in_bw = abs.(x .- cutoff) .<= bandwidth
        x_bw = x[in_bw]
        cov_bw = cov[in_bw]

        left_mask = x_bw .< cutoff
        right_mask = x_bw .>= cutoff

        if sum(left_mask) < 5 || sum(right_mask) < 5
            push!(results, CovariateSmoothnessResult(
                name, NaN, NaN, NaN, NaN, NaN, 1.0, true
            ))
            continue
        end

        # Fit slopes on each side
        x_left = x_bw[left_mask] .- cutoff
        cov_left = cov_bw[left_mask]
        x_right = x_bw[right_mask] .- cutoff
        cov_right = cov_bw[right_mask]

        slope_left, se_left = _fit_ols_slope(x_left, cov_left)
        slope_right, se_right = _fit_ols_slope(x_right, cov_right)

        slope_diff = slope_right - slope_left
        se_diff = sqrt(se_left^2 + se_right^2)

        if se_diff > 0
            t_stat = slope_diff / se_diff
            df = sum(left_mask) + sum(right_mask) - 4
            df = max(df, 1)
            p_value = 2 * (1 - cdf(TDist(df), abs(t_stat)))
        else
            t_stat = slope_diff != 0 ? Inf * sign(slope_diff) : 0.0
            p_value = slope_diff != 0 ? 0.0 : 1.0
        end

        is_smooth = p_value >= 0.05

        push!(results, CovariateSmoothnessResult(
            name, slope_left, slope_right, slope_diff, se_diff, t_stat, p_value, is_smooth
        ))
    end

    return results
end

# Single covariate version
function covariate_smoothness_test(
    x::AbstractVector{T},
    covariate::AbstractVector{T},
    cutoff::Real;
    covariate_name::String="Covariate",
    bandwidth::Union{Nothing,Real}=nothing
) where {T<:Real}
    covariates = reshape(covariate, :, 1)
    return covariate_smoothness_test(
        x, covariates, cutoff;
        covariate_names=[covariate_name],
        bandwidth=bandwidth
    )
end

"""
    first_stage_test(d, x, cutoff; bandwidth=nothing, polynomial_order=2)

Test the strength of the first stage in Fuzzy RKD.

Tests whether there is a significant kink in E[D|X] at the cutoff,
which is required for identification in Fuzzy RKD.

# Arguments
- `d`: Treatment variable
- `x`: Running variable
- `cutoff`: Kink point
- `bandwidth`: Bandwidth for local regression (optional)
- `polynomial_order`: Order of local polynomial (default: 2)

# Returns
- `FirstStageTestResult`: Test results including F-statistic

# Notes
Rule of thumb: F > 10 indicates a strong first stage.
F < 10 suggests weak instrument concerns - the LATE may be biased.
"""
function first_stage_test(
    d::AbstractVector{T},
    x::AbstractVector{T},
    cutoff::Real;
    bandwidth::Union{Nothing,Real}=nothing,
    polynomial_order::Int=2
) where {T<:Real}

    n = length(d)

    if isnothing(bandwidth)
        sigma = std(x)
        iqr = quantile(x, 0.75) - quantile(x, 0.25)
        bandwidth = 1.5 * min(sigma, iqr / 1.34) * n^(-0.2)
    end

    # Filter to bandwidth region
    in_bw = abs.(x .- cutoff) .<= bandwidth
    x_bw = x[in_bw]
    d_bw = d[in_bw]

    left_mask = x_bw .< cutoff
    right_mask = x_bw .>= cutoff

    if sum(left_mask) < 5 || sum(right_mask) < 5
        return FirstStageTestResult(
            NaN, NaN, NaN, 1.0, false, "Insufficient data for first stage test"
        )
    end

    # Fit slopes on each side
    x_left = x_bw[left_mask] .- cutoff
    d_left = d_bw[left_mask]
    x_right = x_bw[right_mask] .- cutoff
    d_right = d_bw[right_mask]

    slope_left, se_left = _fit_ols_slope(x_left, d_left)
    slope_right, se_right = _fit_ols_slope(x_right, d_right)

    kink = slope_right - slope_left
    se = sqrt(se_left^2 + se_right^2)

    if se > 0
        f_stat = (kink / se)^2
        df2 = sum(left_mask) + sum(right_mask) - 4
        df2 = max(df2, 1)
        p_value = 1 - cdf(FDist(1, df2), f_stat)
    else
        f_stat = kink != 0 ? Inf : 0.0
        p_value = kink != 0 ? 0.0 : 1.0
    end

    is_strong = f_stat >= 10

    if f_stat >= 10
        interpretation = "Strong first stage (F=$(round(f_stat, digits=2)) ≥ 10). Identification is reliable."
    elseif f_stat >= 5
        interpretation = "Moderate first stage (F=$(round(f_stat, digits=2))). Some weak instrument concern."
    else
        interpretation = "Weak first stage (F=$(round(f_stat, digits=2)) < 5). LATE may be severely biased."
    end

    return FirstStageTestResult(kink, se, f_stat, p_value, is_strong, interpretation)
end

"""
    rkd_diagnostics(y, x, d, cutoff; covariates=nothing, covariate_names=nothing, bandwidth=nothing)

Run comprehensive RKD diagnostics and return summary.

# Arguments
- `y`: Outcome variable
- `x`: Running variable
- `d`: Treatment variable
- `cutoff`: Kink point
- `covariates`: Predetermined covariates to test (optional)
- `covariate_names`: Names for covariates (optional)
- `bandwidth`: Bandwidth for tests (optional)

# Returns
- `RKDDiagnosticsSummary`: Comprehensive diagnostics summary
"""
function rkd_diagnostics(
    y::AbstractVector{T},
    x::AbstractVector{T},
    d::AbstractVector{T},
    cutoff::Real;
    covariates::Union{Nothing,AbstractMatrix{T}}=nothing,
    covariate_names::Union{Nothing,Vector{String}}=nothing,
    bandwidth::Union{Nothing,Real}=nothing
) where {T<:Real}

    # Density smoothness
    density_result = density_smoothness_test(x, cutoff; bandwidth=bandwidth)

    # First stage
    fs_result = first_stage_test(d, x, cutoff; bandwidth=bandwidth)

    # Covariate smoothness
    if !isnothing(covariates)
        cov_results = covariate_smoothness_test(
            x, covariates, cutoff;
            covariate_names=covariate_names,
            bandwidth=bandwidth
        )
    else
        cov_results = CovariateSmoothnessResult[]
    end

    # Summary
    density_smooth = density_result.p_value >= 0.05
    first_stage_strong = fs_result.is_strong
    covariates_smooth = isempty(cov_results) || all(c.is_smooth for c in cov_results)
    all_pass = density_smooth && first_stage_strong && covariates_smooth

    return RKDDiagnosticsSummary(
        density_result, fs_result, cov_results,
        density_smooth, first_stage_strong, covariates_smooth, all_pass
    )
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    _fit_ols_slope(x, y) -> (slope, se)

Fit simple OLS and return slope and its standard error.
"""
function _fit_ols_slope(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:Real}
    n = length(x)
    if n < 3
        return NaN, NaN
    end

    # OLS: y = a + b*x
    x_mean = mean(x)
    y_mean = mean(y)

    Sxx = sum((x .- x_mean).^2)
    Sxy = sum((x .- x_mean) .* (y .- y_mean))

    if Sxx < 1e-10
        return NaN, NaN
    end

    slope = Sxy / Sxx
    intercept = y_mean - slope * x_mean

    # Residuals
    residuals = y .- (intercept .+ slope .* x)
    residual_var = sum(residuals.^2) / (n - 2)

    # SE of slope
    se_slope = sqrt(residual_var / Sxx)

    return slope, se_slope
end

# Display methods
function Base.show(io::IO, result::DensitySmoothnessResult)
    println(io, "DensitySmoothnessResult")
    println(io, "  Slope left: $(round(result.slope_left, digits=4))")
    println(io, "  Slope right: $(round(result.slope_right, digits=4))")
    println(io, "  Difference: $(round(result.slope_difference, digits=4))")
    println(io, "  p-value: $(round(result.p_value, digits=4))")
    println(io, "  $(result.interpretation)")
end

function Base.show(io::IO, result::FirstStageTestResult)
    println(io, "FirstStageTestResult")
    println(io, "  Kink estimate: $(round(result.kink_estimate, digits=4))")
    println(io, "  F-statistic: $(round(result.f_stat, digits=2))")
    println(io, "  Strong: $(result.is_strong)")
    println(io, "  $(result.interpretation)")
end

function Base.show(io::IO, summary::RKDDiagnosticsSummary)
    println(io, "RKDDiagnosticsSummary")
    println(io, "  Density smooth: $(summary.density_smooth)")
    println(io, "  First stage strong: $(summary.first_stage_strong)")
    println(io, "  Covariates smooth: $(summary.covariates_smooth)")
    println(io, "  All pass: $(summary.all_pass)")
end
