"""
Bootstrap Inference for Impulse Response Functions

Session 147: Bootstrap and MBB confidence bands for IRF/FEVD.

Provides:
- Residual bootstrap for IRF
- Moving Block Bootstrap (MBB) for time-dependent data
- Joint confidence bands (Bonferroni, Sup-t, Simes)
- Bootstrap FEVD

References:
- Kunsch (1989). "The jackknife and bootstrap for general stationary observations."
- Lütkepohl (2005). "New Introduction to Multiple Time Series Analysis."
"""
module BootstrapIRF

using LinearAlgebra
using Statistics
using Random

using ..TimeSeriesTypes
using ..SVARTypes
using ..VAR: var_estimate
using ..SVAR: cholesky_svar, structural_vma_coefficients

export bootstrap_irf, moving_block_bootstrap_irf
export joint_confidence_bands, moving_block_bootstrap_irf_joint
export bootstrap_fevd


"""
    bootstrap_irf(data, svar_result; horizons=20, n_bootstrap=500, alpha=0.05,
                  cumulative=false, method="residual", seed=nothing)

Bootstrap confidence bands for IRF.

# Arguments
- `data::Matrix{Float64}`: Original time series data (n_obs, n_vars)
- `svar_result::SVARResult`: Structural VAR estimation from original data
- `horizons::Int`: Maximum horizon for IRF
- `n_bootstrap::Int`: Number of bootstrap replications
- `alpha::Float64`: Significance level (e.g., 0.05 for 95% CI)
- `cumulative::Bool`: If true, compute cumulative IRF
- `method::String`: Bootstrap method ("residual" or "wild")
- `seed`: Random seed for reproducibility

# Returns
- `IRFResult`: IRF with confidence bands

# Example
```julia
using Random
Random.seed!(42)
data = randn(200, 3)
var_result = var_estimate(data, lags=2)
svar_result = cholesky_svar(var_result)
irf_ci = bootstrap_irf(data, svar_result, horizons=20, n_bootstrap=500)
```
"""
function bootstrap_irf(
    data::AbstractMatrix{<:Real},
    svar_result::SVARResult;
    horizons::Int=20,
    n_bootstrap::Int=500,
    alpha::Float64=0.05,
    cumulative::Bool=false,
    method::String="residual",
    seed::Union{Int,Nothing}=nothing,
)
    if n_bootstrap < 2
        error("n_bootstrap must be >= 2, got $n_bootstrap")
    end

    if alpha <= 0 || alpha >= 1
        error("alpha must be in (0, 1), got $alpha")
    end

    if !(method in ["residual", "wild"])
        error("method must be 'residual' or 'wild', got '$method'")
    end

    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)

    data = Float64.(data)
    n_obs, n_vars = size(data)
    lags = svar_result.lags
    n_effective = n_obs - lags

    # Get residuals from the VAR
    residuals = svar_result.var_residuals

    # Storage for bootstrap IRFs
    irf_boots = zeros(n_bootstrap, n_vars, n_vars, horizons + 1)

    for b in 1:n_bootstrap
        # Generate bootstrap residuals
        if method == "residual"
            # Resample residuals with replacement
            indices = rand(rng, 1:n_effective, n_effective)
            resid_boot = residuals[indices, :]
        else  # wild bootstrap
            # Rademacher distribution
            signs = rand(rng, [-1.0, 1.0], n_effective)
            resid_boot = residuals .* signs
        end

        # Reconstruct bootstrap data
        data_boot = reconstruct_var_data(data, svar_result, resid_boot, lags)

        # Re-estimate VAR and SVAR
        try
            var_boot = var_estimate(data_boot; lags=lags, var_names=svar_result.var_names)
            svar_boot = cholesky_svar(var_boot; ordering=svar_result.ordering)

            # Compute IRF
            irf_boot = structural_vma_coefficients(svar_boot, horizons)
            if cumulative
                irf_boot = cumsum(irf_boot, dims=3)
            end

            irf_boots[b, :, :, :] = irf_boot
        catch
            # Use NaN for failed bootstrap
            irf_boots[b, :, :, :] .= NaN
        end
    end

    # Compute percentile confidence intervals
    lower_pct = alpha / 2
    upper_pct = 1 - alpha / 2

    irf_lower = zeros(n_vars, n_vars, horizons + 1)
    irf_upper = zeros(n_vars, n_vars, horizons + 1)

    for i in 1:n_vars, j in 1:n_vars, h in 1:(horizons+1)
        samples = filter(!isnan, irf_boots[:, i, j, h])
        if !isempty(samples)
            irf_lower[i, j, h] = quantile(samples, lower_pct)
            irf_upper[i, j, h] = quantile(samples, upper_pct)
        else
            irf_lower[i, j, h] = NaN
            irf_upper[i, j, h] = NaN
        end
    end

    # Point estimate from original
    irf_point = structural_vma_coefficients(svar_result, horizons)
    if cumulative
        irf_point = cumsum(irf_point, dims=3)
    end

    IRFResult(
        irf=irf_point,
        irf_lower=irf_lower,
        irf_upper=irf_upper,
        horizons=horizons,
        cumulative=cumulative,
        orthogonalized=true,
        var_names=svar_result.var_names,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
    )
end


"""Reconstruct time series from VAR fitted values and bootstrap residuals."""
function reconstruct_var_data(
    original_data::Matrix{Float64},
    svar_result::SVARResult,
    bootstrap_residuals::Matrix{Float64},
    lags::Int,
)
    n_obs, n_vars = size(original_data)

    # Start with original initial values
    data_boot = zeros(n_obs, n_vars)
    data_boot[1:lags, :] = original_data[1:lags, :]

    # Reconstruct using fitted values + bootstrap residuals
    # Get intercepts from first column of coefficients
    intercepts = svar_result.var_coefficients[:, 1]

    for t in (lags+1):n_obs
        y_t = copy(intercepts)

        for lag in 1:lags
            # Get lag matrix: columns (1 + (lag-1)*n_vars + 1) to (1 + lag*n_vars)
            start_col = 2 + (lag - 1) * n_vars
            end_col = 1 + lag * n_vars
            A_lag = svar_result.var_coefficients[:, start_col:end_col]
            y_t += A_lag * data_boot[t - lag, :]
        end

        # Add bootstrap residual
        resid_idx = t - lags
        y_t += bootstrap_residuals[resid_idx, :]

        data_boot[t, :] = y_t
    end

    return data_boot
end


"""
    moving_block_bootstrap_irf(data, svar_result; horizons=20, n_bootstrap=500,
                                block_length=nothing, alpha=0.05, cumulative=false, seed=nothing)

Moving Block Bootstrap (MBB) confidence bands for IRF.

MBB preserves temporal dependence structure by resampling blocks of consecutive
observations, making it more appropriate for time series than i.i.d. bootstrap.

# Arguments
- `data::Matrix{Float64}`: Original time series data (n_obs, n_vars)
- `svar_result::SVARResult`: Structural VAR estimation from original data
- `horizons::Int`: Maximum horizon for IRF
- `n_bootstrap::Int`: Number of bootstrap replications
- `block_length::Union{Int,Nothing}`: Length of blocks. Default: T^(1/3).
- `alpha::Float64`: Significance level
- `cumulative::Bool`: If true, compute cumulative IRF
- `seed`: Random seed for reproducibility

# Returns
- `IRFResult`: IRF with confidence bands

# Notes
The block bootstrap (Kunsch 1989, Liu & Singh 1992) preserves within-block
dependence while achieving consistency for weakly dependent processes.
The default block length l = T^(1/3) is optimal for variance estimation.

# Example
```julia
irf_mbb = moving_block_bootstrap_irf(data, svar_result, horizons=20)
```
"""
function moving_block_bootstrap_irf(
    data::AbstractMatrix{<:Real},
    svar_result::SVARResult;
    horizons::Int=20,
    n_bootstrap::Int=500,
    block_length::Union{Int,Nothing}=nothing,
    alpha::Float64=0.05,
    cumulative::Bool=false,
    seed::Union{Int,Nothing}=nothing,
)
    if n_bootstrap < 2
        error("n_bootstrap must be >= 2, got $n_bootstrap")
    end

    if alpha <= 0 || alpha >= 1
        error("alpha must be in (0, 1), got $alpha")
    end

    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)

    data = Float64.(data)
    n_obs, n_vars = size(data)
    lags = svar_result.lags
    n_effective = n_obs - lags

    # Default block length: T^(1/3)
    if block_length === nothing
        block_length = max(1, Int(ceil(n_effective^(1/3))))
    end

    if block_length > n_effective
        error("block_length ($block_length) exceeds effective sample size ($n_effective)")
    end

    # Storage for bootstrap IRFs
    irf_boots = zeros(n_bootstrap, n_vars, n_vars, horizons + 1)

    for b in 1:n_bootstrap
        # Generate MBB sample
        data_boot = moving_block_sample(data, block_length, rng)

        try
            var_boot = var_estimate(data_boot; lags=lags, var_names=svar_result.var_names)
            svar_boot = cholesky_svar(var_boot; ordering=svar_result.ordering)

            irf_boot = structural_vma_coefficients(svar_boot, horizons)
            if cumulative
                irf_boot = cumsum(irf_boot, dims=3)
            end

            irf_boots[b, :, :, :] = irf_boot
        catch
            irf_boots[b, :, :, :] .= NaN
        end
    end

    # Compute percentile confidence intervals
    lower_pct = alpha / 2
    upper_pct = 1 - alpha / 2

    irf_lower = zeros(n_vars, n_vars, horizons + 1)
    irf_upper = zeros(n_vars, n_vars, horizons + 1)

    for i in 1:n_vars, j in 1:n_vars, h in 1:(horizons+1)
        samples = filter(!isnan, irf_boots[:, i, j, h])
        if !isempty(samples)
            irf_lower[i, j, h] = quantile(samples, lower_pct)
            irf_upper[i, j, h] = quantile(samples, upper_pct)
        else
            irf_lower[i, j, h] = NaN
            irf_upper[i, j, h] = NaN
        end
    end

    # Point estimate from original
    irf_point = structural_vma_coefficients(svar_result, horizons)
    if cumulative
        irf_point = cumsum(irf_point, dims=3)
    end

    IRFResult(
        irf=irf_point,
        irf_lower=irf_lower,
        irf_upper=irf_upper,
        horizons=horizons,
        cumulative=cumulative,
        orthogonalized=true,
        var_names=svar_result.var_names,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
    )
end


"""Generate a moving block bootstrap sample."""
function moving_block_sample(
    data::Matrix{Float64},
    block_length::Int,
    rng::AbstractRNG,
)
    n_obs, n_vars = size(data)

    # Number of blocks needed
    n_blocks = Int(ceil(n_obs / block_length))

    # Valid starting positions
    max_start = n_obs - block_length
    if max_start < 0
        return copy(data)
    end

    # Sample block starting positions with replacement
    block_starts = rand(rng, 0:max_start, n_blocks)

    # Concatenate blocks
    blocks = Matrix{Float64}[]
    for start in block_starts
        push!(blocks, data[start+1:start+block_length, :])
    end

    data_boot = vcat(blocks...)

    # Trim to original length
    return data_boot[1:n_obs, :]
end


"""
    joint_confidence_bands(irf_boots; alpha=0.05, method="bonferroni")

Compute joint confidence bands for IRF across all horizons.

Pointwise confidence bands have inflated Type I error when making simultaneous
inference across multiple horizons. Joint bands correct for multiple comparisons.

# Arguments
- `irf_boots::Array{Float64,4}`: Bootstrap IRF samples (n_bootstrap, n_vars, n_vars, horizons+1)
- `alpha::Float64`: Family-wise significance level
- `method::String`: Correction method:
  - "bonferroni": Bonferroni correction (conservative)
  - "sup": Supremum-based bands using max deviation
  - "simes": Simes procedure (less conservative)

# Returns
- `Tuple{Array{Float64,3}, Array{Float64,3}}`: (irf_lower, irf_upper) joint bands

# Example
```julia
lower, upper = joint_confidence_bands(irf_boots, alpha=0.05, method="bonferroni")
```
"""
function joint_confidence_bands(
    irf_boots::Array{Float64,4};
    alpha::Float64=0.05,
    method::String="bonferroni",
)
    if !(method in ["bonferroni", "sup", "simes"])
        error("method must be 'bonferroni', 'sup', or 'simes', got '$method'")
    end

    n_bootstrap, n_vars, _, n_horizons = size(irf_boots)
    horizons = n_horizons - 1

    n_tests = n_horizons

    if method == "bonferroni"
        # Bonferroni: use α/H for each horizon
        alpha_adj = alpha / n_tests
        lower_pct = alpha_adj / 2
        upper_pct = 1 - alpha_adj / 2

        irf_lower = zeros(n_vars, n_vars, n_horizons)
        irf_upper = zeros(n_vars, n_vars, n_horizons)

        for i in 1:n_vars, j in 1:n_vars, h in 1:n_horizons
            samples = filter(!isnan, irf_boots[:, i, j, h])
            if !isempty(samples)
                irf_lower[i, j, h] = quantile(samples, lower_pct)
                irf_upper[i, j, h] = quantile(samples, upper_pct)
            else
                irf_lower[i, j, h] = NaN
                irf_upper[i, j, h] = NaN
            end
        end

    elseif method == "sup"
        # Sup-t bands based on max deviation
        irf_point = zeros(n_vars, n_vars, n_horizons)
        irf_se = zeros(n_vars, n_vars, n_horizons)

        for i in 1:n_vars, j in 1:n_vars, h in 1:n_horizons
            samples = filter(!isnan, irf_boots[:, i, j, h])
            if !isempty(samples)
                irf_point[i, j, h] = median(samples)
                irf_se[i, j, h] = max(std(samples), 1e-10)
            end
        end

        # Compute max deviation for each bootstrap sample
        max_devs = zeros(n_bootstrap)
        for b in 1:n_bootstrap
            if any(isnan, irf_boots[b, :, :, :])
                max_devs[b] = NaN
                continue
            end
            z = abs.(irf_boots[b, :, :, :] .- irf_point) ./ irf_se
            max_devs[b] = maximum(filter(!isnan, z))
        end

        # Critical value at (1-α) quantile
        valid_devs = filter(!isnan, max_devs)
        c_alpha = isempty(valid_devs) ? Inf : quantile(valid_devs, 1 - alpha)

        irf_lower = irf_point .- c_alpha .* irf_se
        irf_upper = irf_point .+ c_alpha .* irf_se

    elseif method == "simes"
        # Simes procedure with progressive adjustment
        alphas_adj = [(i * alpha / n_tests) for i in 1:n_tests]

        irf_lower = zeros(n_vars, n_vars, n_horizons)
        irf_upper = zeros(n_vars, n_vars, n_horizons)

        for i in 1:n_vars, j in 1:n_vars
            # Sort horizons by variance
            variances = zeros(n_horizons)
            for h in 1:n_horizons
                samples = filter(!isnan, irf_boots[:, i, j, h])
                variances[h] = isempty(samples) ? 0.0 : var(samples)
            end
            sorted_h = sortperm(variances, rev=true)

            for (rank, h) in enumerate(sorted_h)
                alpha_h = alphas_adj[rank]
                lower_pct = alpha_h / 2
                upper_pct = 1 - alpha_h / 2

                samples = filter(!isnan, irf_boots[:, i, j, h])
                if !isempty(samples)
                    irf_lower[i, j, h] = quantile(samples, lower_pct)
                    irf_upper[i, j, h] = quantile(samples, upper_pct)
                else
                    irf_lower[i, j, h] = NaN
                    irf_upper[i, j, h] = NaN
                end
            end
        end
    end

    return irf_lower, irf_upper
end


"""
    moving_block_bootstrap_irf_joint(data, svar_result; horizons=20, n_bootstrap=500,
                                      block_length=nothing, alpha=0.05, cumulative=false,
                                      joint_method="bonferroni", seed=nothing)

Moving Block Bootstrap IRF with joint confidence bands.

Combines MBB with multiple testing correction for valid simultaneous
inference across all horizons.

# Example
```julia
irf_joint = moving_block_bootstrap_irf_joint(data, svar_result, horizons=20)
```
"""
function moving_block_bootstrap_irf_joint(
    data::AbstractMatrix{<:Real},
    svar_result::SVARResult;
    horizons::Int=20,
    n_bootstrap::Int=500,
    block_length::Union{Int,Nothing}=nothing,
    alpha::Float64=0.05,
    cumulative::Bool=false,
    joint_method::String="bonferroni",
    seed::Union{Int,Nothing}=nothing,
)
    if n_bootstrap < 2
        error("n_bootstrap must be >= 2, got $n_bootstrap")
    end

    if alpha <= 0 || alpha >= 1
        error("alpha must be in (0, 1), got $alpha")
    end

    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)

    data = Float64.(data)
    n_obs, n_vars = size(data)
    lags = svar_result.lags
    n_effective = n_obs - lags

    # Default block length: T^(1/3)
    if block_length === nothing
        block_length = max(1, Int(ceil(n_effective^(1/3))))
    end

    # Storage for bootstrap IRFs
    irf_boots = zeros(n_bootstrap, n_vars, n_vars, horizons + 1)

    for b in 1:n_bootstrap
        data_boot = moving_block_sample(data, block_length, rng)

        try
            var_boot = var_estimate(data_boot; lags=lags, var_names=svar_result.var_names)
            svar_boot = cholesky_svar(var_boot; ordering=svar_result.ordering)

            irf_boot = structural_vma_coefficients(svar_boot, horizons)
            if cumulative
                irf_boot = cumsum(irf_boot, dims=3)
            end

            irf_boots[b, :, :, :] = irf_boot
        catch
            irf_boots[b, :, :, :] .= NaN
        end
    end

    # Compute joint confidence bands
    irf_lower, irf_upper = joint_confidence_bands(irf_boots; alpha=alpha, method=joint_method)

    # Point estimate from original
    irf_point = structural_vma_coefficients(svar_result, horizons)
    if cumulative
        irf_point = cumsum(irf_point, dims=3)
    end

    IRFResult(
        irf=irf_point,
        irf_lower=irf_lower,
        irf_upper=irf_upper,
        horizons=horizons,
        cumulative=cumulative,
        orthogonalized=true,
        var_names=svar_result.var_names,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
    )
end


"""
    bootstrap_fevd(data, svar_result; horizons=20, n_bootstrap=500, alpha=0.05, seed=nothing)

Bootstrap confidence bands for FEVD.

# Arguments
- `data::Matrix{Float64}`: Original time series data
- `svar_result::SVARResult`: Structural VAR estimation
- `horizons::Int`: Maximum horizon for FEVD
- `n_bootstrap::Int`: Number of bootstrap replications
- `alpha::Float64`: Significance level
- `seed`: Random seed

# Returns
- `Tuple{FEVDResult, Array{Float64,4}, Array{Float64,4}}`: (fevd_point, fevd_lower, fevd_upper)
"""
function bootstrap_fevd(
    data::AbstractMatrix{<:Real},
    svar_result::SVARResult;
    horizons::Int=20,
    n_bootstrap::Int=500,
    alpha::Float64=0.05,
    seed::Union{Int,Nothing}=nothing,
)
    if n_bootstrap < 2
        error("n_bootstrap must be >= 2, got $n_bootstrap")
    end

    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)

    data = Float64.(data)
    n_obs, n_vars = size(data)
    lags = svar_result.lags
    n_effective = n_obs - lags

    residuals = svar_result.var_residuals

    # Storage for bootstrap FEVDs
    fevd_boots = zeros(n_bootstrap, n_vars, n_vars, horizons + 1)

    for b in 1:n_bootstrap
        # Residual bootstrap
        indices = rand(rng, 1:n_effective, n_effective)
        resid_boot = residuals[indices, :]

        data_boot = reconstruct_var_data(data, svar_result, resid_boot, lags)

        try
            var_boot = var_estimate(data_boot; lags=lags, var_names=svar_result.var_names)
            svar_boot = cholesky_svar(var_boot; ordering=svar_result.ordering)

            # Compute FEVD
            Psi = structural_vma_coefficients(svar_boot, horizons)

            for h in 0:horizons
                cumsum_squared = zeros(n_vars, n_vars)
                for k in 0:h
                    cumsum_squared += Psi[:, :, k+1].^2
                end

                total_mse = sum(cumsum_squared, dims=2)[:, 1]

                for i in 1:n_vars
                    if total_mse[i] > 1e-12
                        fevd_boots[b, i, :, h+1] = cumsum_squared[i, :] / total_mse[i]
                    else
                        fevd_boots[b, i, :, h+1] .= 1.0 / n_vars
                    end
                end
            end
        catch
            fevd_boots[b, :, :, :] .= NaN
        end
    end

    # Compute confidence intervals
    lower_pct = alpha / 2
    upper_pct = 1 - alpha / 2

    fevd_lower = zeros(n_vars, n_vars, horizons + 1)
    fevd_upper = zeros(n_vars, n_vars, horizons + 1)

    for i in 1:n_vars, j in 1:n_vars, h in 1:(horizons+1)
        samples = filter(!isnan, fevd_boots[:, i, j, h])
        if !isempty(samples)
            fevd_lower[i, j, h] = quantile(samples, lower_pct)
            fevd_upper[i, j, h] = quantile(samples, upper_pct)
        else
            fevd_lower[i, j, h] = NaN
            fevd_upper[i, j, h] = NaN
        end
    end

    # Point estimate from original (using existing compute_fevd logic)
    Psi_point = structural_vma_coefficients(svar_result, horizons)
    fevd_point = zeros(n_vars, n_vars, horizons + 1)

    for h in 0:horizons
        cumsum_squared = zeros(n_vars, n_vars)
        for k in 0:h
            cumsum_squared += Psi_point[:, :, k+1].^2
        end

        total_mse = sum(cumsum_squared, dims=2)[:, 1]

        for i in 1:n_vars
            if total_mse[i] > 1e-12
                fevd_point[i, :, h+1] = cumsum_squared[i, :] / total_mse[i]
            else
                fevd_point[i, :, h+1] .= 1.0 / n_vars
            end
        end
    end

    fevd_result = FEVDResult(
        fevd=fevd_point,
        horizons=horizons,
        var_names=svar_result.var_names,
    )

    return fevd_result, fevd_lower, fevd_upper
end

end # module
