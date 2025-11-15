"""
Generalized Method of Moments (GMM) estimator for instrumental variables.

Implements Hansen (1982) GMM framework with optimal weighting and
heteroskedasticity-robust inference.

# References
- Hansen, L. P. (1982). "Large Sample Properties of Generalized Method of Moments
  Estimators." *Econometrica*, 50(4), 1029-1054.
- Baum, C. F., Schaffer, M. E., & Stillman, S. (2003). "Instrumental Variables and
  GMM: Estimation and Testing." *Stata Journal*, 3(1), 1-31.
- Newey, W. K., & West, K. D. (1987). "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
  *Econometrica*, 55(3), 703-708.
"""

using LinearAlgebra
using Statistics
using Distributions

"""
    GMM <: AbstractIVEstimator

Generalized Method of Moments estimator for IV regression.

# Fields
- `weighting::Symbol`: Weighting matrix type (:identity, :optimal, :hac)
- `max_iterations::Int`: Maximum iterations for iterative GMM (default: 10)
- `kernel::Symbol`: HAC kernel type (:bartlett, :parzen) for time series
- `bandwidth::Union{Int,Nothing}`: HAC bandwidth (auto-selected if nothing)

# Weighting Methods

**Identity weighting** (`:identity`):
- W = I_K (identity matrix)
- Equivalent to 2SLS
- No efficiency gains

**Optimal weighting** (`:optimal`):
- W = (Z'ΩZ/n)⁻¹ where Ω = diag(û²)
- Two-step efficient GMM (Hansen 1982)
- Asymptotically efficient under heteroskedasticity

**HAC weighting** (`:hac`):
- W = S⁻¹ with Newey-West HAC covariance
- Robust to heteroskedasticity and autocorrelation
- For time series data

# Method

GMM minimizes:
```
Q(β) = (Z'û)'W(Z'û)
```

where û = Y - βD and W is the weighting matrix.

## Two-Step Efficient GMM

**Step 1**: Initial estimate with W = I
- β̂₁ = (D'P_Z D)⁻¹ D'P_Z Y (equivalent to 2SLS)
- Obtain residuals û₁

**Step 2**: Optimal weighting
- Construct W = (Z'Ω̂Z/n)⁻¹ where Ω̂ = diag(û₁²)
- Re-estimate: β̂_GMM minimizing Q(β)

## Asymptotic Variance

V = (D'ZWZ'D/n)⁻¹

Standard errors: se = sqrt(diag(V)/n)

## Hansen J Test

Overidentification test (K > L):
```
J = n * (Z'û)'W(Z'û) ~ χ²(K - L)
```

Under H₀ (instruments exogenous), J has chi-squared distribution.
Reject if J > critical value → suggests instrument invalidity.

# Properties

**Asymptotic Efficiency**: With optimal weighting, GMM is efficient.

**Robust Inference**: Handles heteroskedasticity (optimal) and autocorrelation (HAC).

**Overidentification**: Hansen J test checks instrument validity.

**Relationship to 2SLS**: GMM with identity weighting = 2SLS.

# Example
```julia
problem = IVProblem(outcomes, treatment, instruments, covariates, (alpha=0.05,))

# Two-step efficient GMM (default)
estimator = GMM(weighting=:optimal)
solution = solve(problem, estimator)

# Check overidentification
if solution.n_instruments > 1
    println("Hansen J: \$(solution.diagnostics.hansen_j)")
    println("J p-value: \$(solution.overid_pvalue)")

    if solution.overid_pvalue < 0.05
        @warn "Instruments may be invalid (Hansen J rejects)"
    end
end

# HAC for time series
estimator_hac = GMM(weighting=:hac, kernel=:bartlett, bandwidth=4)
solution_hac = solve(problem, estimator_hac)
```

# References
- Hansen (1982): GMM framework
- Newey & West (1987): HAC covariance
- Baum et al. (2003): Practical guide
"""
struct GMM <: AbstractIVEstimator
    weighting::Symbol
    max_iterations::Int
    kernel::Symbol
    bandwidth::Union{Int,Nothing}

    function GMM(;
        weighting::Symbol = :optimal,
        max_iterations::Int = 10,
        kernel::Symbol = :bartlett,
        bandwidth::Union{Int,Nothing} = nothing,
    )
        # Validate weighting method
        valid_weightings = [:identity, :optimal, :hac]
        if !(weighting in valid_weightings)
            throw(
                ArgumentError(
                    "Invalid weighting method: $weighting\n" *
                    "Valid options: :identity, :optimal, :hac",
                ),
            )
        end

        # Validate kernel
        valid_kernels = [:bartlett, :parzen]
        if !(kernel in valid_kernels)
            throw(
                ArgumentError(
                    "Invalid HAC kernel: $kernel\n" * "Valid options: :bartlett, :parzen",
                ),
            )
        end

        # Validate iterations
        if max_iterations < 1
            throw(ArgumentError("max_iterations must be >= 1, got $max_iterations"))
        end

        # Validate bandwidth
        if !isnothing(bandwidth) && bandwidth < 1
            throw(ArgumentError("bandwidth must be >= 1, got $bandwidth"))
        end

        new(weighting, max_iterations, kernel, bandwidth)
    end
end


"""
    solve(problem::IVProblem, estimator::GMM)

Estimate treatment effect using Generalized Method of Moments.

# Arguments
- `problem::IVProblem`: IV problem specification
- `estimator::GMM`: GMM estimator configuration

# Returns
- `solution::IVSolution`: Solution with estimate, SE, CI, Hansen J test

# Algorithm

1. **Step 1 - Initial Estimate**:
   - Use identity weighting (W = I)
   - Equivalent to 2SLS: β̂₁ = (D'P_Z D)⁻¹ D'P_Z Y
   - Obtain residuals û₁

2. **Step 2 - Optimal Weighting**:
   - Construct W based on estimator.weighting
   - :identity → W = I (stop at step 1)
   - :optimal → W = (Z'Ω̂Z/n)⁻¹ where Ω̂ = diag(û₁²)
   - :hac → W = S⁻¹ with Newey-West HAC matrix

3. **Step 3 - Efficient GMM**:
   - Re-estimate with optimal W
   - β̂_GMM minimizes (Z'û)'W(Z'û)

4. **Inference**:
   - Asymptotic variance: V = (D'ZWZ'D/n)⁻¹
   - Standard errors: se = sqrt(diag(V)/n)
   - Confidence intervals

5. **Hansen J Test**:
   - J = n * (Z'û)'W(Z'û)
   - p-value from χ²(K - L)

# Example
```julia
problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
solution = solve(problem, GMM(weighting=:optimal))

println("GMM estimate: \$(solution.estimate)")
println("Hansen J p-value: \$(solution.overid_pvalue)")
```
"""
function solve(problem::IVProblem{T,P}, estimator::GMM) where {T<:Real,P<:NamedTuple}
    # Extract problem data
    y = problem.outcomes
    d = problem.treatment
    Z = problem.instruments
    X = problem.covariates
    alpha = problem.parameters.alpha

    n = length(y)
    K = size(Z, 2)
    p = isnothing(X) ? 0 : size(X, 2)

    # ========================================================================
    # Weak IV Diagnostics
    # ========================================================================

    fstat, fstat_pval = first_stage_fstat(d, Z, X)
    cd_stat = cragg_donald_stat(d, Z, X)
    is_weak, weak_warning = weak_iv_warning(fstat, cd_stat, K)

    # ========================================================================
    # Step 1: Initial Estimate (Identity Weighting = 2SLS)
    # ========================================================================

    # Build instrument matrix: [1 Z X]
    if isnothing(X)
        Z_full = hcat(ones(T, n), Z)
    else
        Z_full = hcat(ones(T, n), Z, X)
    end

    # Projection matrix: P_Z = Z_full(Z_full'Z_full)^(-1)Z_full'
    P_Z = Z_full * ((Z_full' * Z_full) \ Z_full')

    # Initial estimate: β̂₁ = (D'P_Z D)^(-1) D'P_Z Y
    β_init = (d' * P_Z * d) \ (d' * P_Z * y)

    # Initial residuals
    û_init = y - β_init * d

    # ========================================================================
    # Step 2-3: Compute Weighting and Estimate
    # ========================================================================

    if estimator.weighting == :identity
        # Identity weighting = 2SLS exactly
        # Return TSLS solution directly to guarantee exact matching
        tsls_solution = solve(problem, TSLS())

        # Return with GMM-Identity name but all TSLS values
        # IVSolution fields (in order): estimate, se, ci_lower, ci_upper, p_value,
        #   n, n_instruments, n_covariates, first_stage_fstat, overid_pvalue,
        #   weak_iv_warning, estimator_name, alpha, diagnostics
        return IVSolution(
            tsls_solution.estimate,
            tsls_solution.se,
            tsls_solution.ci_lower,
            tsls_solution.ci_upper,
            tsls_solution.p_value,
            tsls_solution.n,
            tsls_solution.n_instruments,
            tsls_solution.n_covariates,
            tsls_solution.first_stage_fstat,
            tsls_solution.overid_pvalue,
            tsls_solution.weak_iv_warning,
            "GMM-Identity",
            tsls_solution.alpha,
            (
                weighting=:identity,
                hansen_j=nothing,  # Hansen J not computed for identity weighting (uses Sargan test in overid_pvalue)
                gmm_iterations=1,
                initial_estimate=tsls_solution.estimate,
                cragg_donald=tsls_solution.diagnostics.cragg_donald,
                n_instruments=K,
                n_covariates=p,
            ),
        )
    end

    # Compute optimal weighting matrix
    W = if estimator.weighting == :optimal
        # Optimal weighting: W = (Z'ΩZ/n)^(-1)
        Omega = Diagonal(û_init .^ 2)
        S = (Z_full' * Omega * Z_full) / n
        inv(S)
    else  # :hac
        # HAC weighting (Newey-West)
        bw = isnothing(estimator.bandwidth) ? _auto_bandwidth(n) : estimator.bandwidth
        S_hac = _newey_west_cov(Z_full, û_init, bw, estimator.kernel)
        inv(S_hac)
    end

    # GMM estimation with optimal weighting
    # For GMM: minimize (Z'û)'W(Z'û)
    # FOC: D'Z W Z'û = 0
    # Solution: β̂ = (D'Z W Z'D)^(-1) D'Z W Z'Y

    # Numerator: D'Z W Z'Y
    numerator = d' * Z_full * W * Z_full' * y

    # Denominator: D'Z W Z'D
    denominator = d' * Z_full * W * Z_full' * d

    # GMM estimate
    β_gmm = numerator / denominator

    # GMM residuals
    û_gmm = y - β_gmm * d

    # ========================================================================
    # Inference
    # ========================================================================

    # Asymptotic variance for GMM: V = n * (D'ZWZ'D)^{-1}
    # Derivation: Var(β̂) = (1/n) * (G'WG)^{-1} where G = -Z'D/n
    # So V = (1/n) * ((D'Z/n)'W(D'Z/n))^{-1} = n * (D'ZWZ'D)^{-1}
    DZW = d' * Z_full * W * Z_full' * d
    V = n / DZW

    se_gmm = sqrt(V)

    # Confidence interval
    z_crit = quantile(Normal(0, 1), 1 - alpha / 2)
    ci_lower = β_gmm - z_crit * se_gmm
    ci_upper = β_gmm + z_crit * se_gmm

    # P-value for H₀: β = 0
    z_stat = β_gmm / se_gmm
    p_value = 2 * (1 - cdf(Normal(0, 1), abs(z_stat)))

    # ========================================================================
    # Hansen J Overidentification Test
    # ========================================================================

    overid_pvalue, J_stat = if K > 1  # Overidentified (K instruments > L=1 endogenous)
        # Hansen J statistic: J = (1/n) * û'Z W Z'û
        # where Z includes all instruments (intercept + excluded instruments + covariates)
        moment_full = Z_full' * û_gmm  # Dimension: (K+1+p) × 1
        J = (moment_full' * W * moment_full) / n

        # Degrees of freedom: K - L where:
        # K = number of excluded instruments (from problem.instruments)
        # L = number of endogenous variables = 1
        df_overid = K - 1

        # P-value from chi-squared distribution
        p_j = 1 - cdf(Chisq(df_overid), J)

        (p_j, J)
    else
        # Exactly identified (K = L = 1): no overidentification test
        (nothing, nothing)
    end

    # ========================================================================
    # Diagnostics
    # ========================================================================

    diagnostics = (
        weighting=estimator.weighting,
        hansen_j=J_stat,
        gmm_iterations=1,  # Two-step GMM (fixed at 1 iteration)
        initial_estimate=β_init,
        cragg_donald=cd_stat,
        n_instruments=K,
        n_covariates=p,
    )

    # ========================================================================
    # Return Solution
    # ========================================================================

    estimator_name = if estimator.weighting == :optimal
        "GMM-Optimal"
    else
        "GMM-HAC"
    end

    return IVSolution(
        β_gmm,
        se_gmm,
        ci_lower,
        ci_upper,
        p_value,
        n,
        K,
        p,
        fstat,
        overid_pvalue,
        is_weak,
        estimator_name,
        alpha,
        diagnostics,
    )
end


"""
    _auto_bandwidth(n::Int) → Int

Automatic bandwidth selection for HAC covariance (Newey-West rule).

Uses the rule: bw = floor(4 * (n/100)^(2/9))

# Arguments
- `n::Int`: Sample size

# Returns
- `bw::Int`: Optimal bandwidth
"""
function _auto_bandwidth(n::Int)
    return max(1, floor(Int, 4 * (n / 100)^(2 / 9)))
end


"""
    _newey_west_cov(Z, û, bandwidth, kernel) → Matrix

Compute Newey-West HAC covariance matrix.

Handles heteroskedasticity and autocorrelation in time series data.

# Arguments
- `Z::Matrix`: Instrument matrix (n × K)
- `û::Vector`: Residuals (n × 1)
- `bandwidth::Int`: Lag bandwidth
- `kernel::Symbol`: Kernel type (:bartlett or :parzen)

# Returns
- `S::Matrix`: HAC covariance matrix (K × K)

# Algorithm

S = Σ₀ + Σⱼ₌₁ᴸ w(j, L) * (Γⱼ + Γⱼ')

where:
- Γⱼ = (1/n) Σᵢ₌ⱼ₊₁ⁿ (Zᵢûᵢ)(Zᵢ₋ⱼûᵢ₋ⱼ)' (autocovariance at lag j)
- w(j, L) = kernel weight
- L = bandwidth

**Bartlett kernel**: w(j, L) = 1 - j/(L+1)
**Parzen kernel**: w(j, L) = 1 - 6(j/(L+1))² + 6|j/(L+1)|³ for j ≤ L/2

# References
- Newey & West (1987): HAC covariance estimation
"""
function _newey_west_cov(
    Z::Matrix{T},
    û::Vector{T},
    bandwidth::Int,
    kernel::Symbol,
) where {T<:Real}
    n, K = size(Z)

    # Variance component (lag 0)
    Zu = Z .* û  # Element-wise multiplication
    Gamma_0 = (Zu' * Zu) / n

    # Initialize HAC matrix
    S = copy(Gamma_0)

    # Add autocovariance components (lags 1 to bandwidth)
    for j in 1:bandwidth
        # Autocovariance at lag j
        Gamma_j = (Zu[(j+1):end, :]' * Zu[1:(end-j), :]) / n

        # Kernel weight
        w = _hac_kernel_weight(j, bandwidth, kernel)

        # Add weighted autocovariance (symmetric)
        S .+= w * (Gamma_j + Gamma_j')
    end

    return S
end


"""
    _hac_kernel_weight(j, L, kernel) → Float64

Compute HAC kernel weight.

# Arguments
- `j::Int`: Lag
- `L::Int`: Bandwidth
- `kernel::Symbol`: Kernel type (:bartlett or :parzen)

# Returns
- `w::Float64`: Kernel weight

# Kernels

**Bartlett** (Newey-West default):
- w(j) = 1 - j/(L+1)
- Linear decay

**Parzen** (Gallant 1987):
- w(j) = 1 - 6(j/(L+1))² + 6|j/(L+1)|³  for j ≤ L/2
- w(j) = 2(1 - j/(L+1))³  for L/2 < j ≤ L
- Smoother decay
"""
function _hac_kernel_weight(j::Int, L::Int, kernel::Symbol)
    if kernel == :bartlett
        # Bartlett (Newey-West) kernel
        return 1.0 - j / (L + 1)
    elseif kernel == :parzen
        # Parzen (Gallant) kernel
        ratio = j / (L + 1)
        if j <= L / 2
            return 1.0 - 6 * ratio^2 + 6 * abs(ratio)^3
        else
            return 2.0 * (1.0 - ratio)^3
        end
    else
        throw(ArgumentError("Unknown kernel: $kernel"))
    end
end
