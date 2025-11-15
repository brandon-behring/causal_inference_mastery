"""
Limited Information Maximum Likelihood (LIML) estimator for instrumental variables.

LIML is a k-class estimator with k chosen to minimize the eigenvalue criterion.
It has better finite-sample properties than 2SLS, especially with weak instruments.

# References
- Anderson, T. W., & Rubin, H. (1949). "Estimation of the Parameters of a Single
  Equation in a Complete System of Stochastic Equations." *Annals of Mathematical
  Statistics*, 20(1), 46-63.
- Fuller, W. A. (1977). "Some Properties of a Modification of the Limited Information
  Estimator." *Econometrica*, 45(4), 939-953.
- Stock, J. H., Wright, J. H., & Yogo, M. (2002). "A Survey of Weak Instruments and
  Weak Identification in Generalized Method of Moments." *Journal of Business &
  Economic Statistics*, 20(4), 518-529.
"""

using LinearAlgebra
using Statistics
using Distributions

"""
    LIML <: AbstractIVEstimator

Limited Information Maximum Likelihood estimator (k-class with k = λ_min).

# Fields
- `robust::Bool`: Use heteroskedasticity-robust standard errors (default: true)
- `fuller::Float64`: Fuller's modification parameter (default: 0.0, set to 1.0 for Fuller-LIML)

# Method

LIML is a k-class estimator where k is chosen optimally:

## K-Class Estimator

For k-class estimator:
```
β̂_k = (D'M_k D)^(-1) D'M_k Y
```

where M_k = M_X - k(M_X - P_Z) and:
- M_X = I - X(X'X)^(-1)X' (residual maker for exogenous variables)
- P_Z = Z(Z'Z)^(-1)Z' (projection onto instruments)

## LIML Choice of k

LIML sets k = λ_min, the smallest eigenvalue of:
```
(Y D)'M_X(Y D) / (Y D)'M_Z(Y D)
```

where M_Z = I - [Z X]([Z X]'[Z X])^(-1)[Z X]'.

## Fuller's Modification

Fuller (1977) proposes k_Fuller = k_LIML - α/(n - K - p - 1) with α = 1:
- Reduces bias in finite samples
- Better MSE properties
- Set `fuller=1.0` to use Fuller-LIML

# Properties

**Median-Unbiased**: LIML is median-unbiased even with weak instruments.

**Finite-Sample Robustness**: Better properties than 2SLS with weak IVs:
- Smaller bias
- Better MSE
- More reliable inference

**Asymptotic Equivalence**: With strong instruments, LIML ≈ 2SLS asymptotically.

**Trade-off**: Higher variance than 2SLS but lower bias with weak IVs.

# Example
```julia
problem = IVProblem(outcomes, treatment, instruments, covariates, (alpha=0.05,))

# Standard LIML
estimator = LIML(robust=true)
solution = solve(problem, estimator)

# Fuller-LIML (often preferred)
estimator_fuller = LIML(robust=true, fuller=1.0)
solution_fuller = solve(problem, estimator_fuller)

if solution.weak_iv_warning
    println("With weak IVs, LIML preferred over 2SLS")
    println("LIML k = \$(solution.diagnostics.k_liml)")
end
```

# References
- Stock, J. H., & Yogo, M. (2005). Testing for weak instruments.
- Hausman, J. A., Newey, W. K., Woutersen, T., Chao, J. C., & Swanson, N. R. (2012).
  "Instrumental Variable Estimation with Heteroskedasticity and Many Instruments."
  *Quantitative Economics*, 3(2), 211-255.
"""
struct LIML <: AbstractIVEstimator
    robust::Bool
    fuller::Float64

    function LIML(; robust::Bool = true, fuller::Float64 = 0.0)
        if fuller < 0.0
            throw(ArgumentError("Fuller parameter must be non-negative, got $fuller"))
        end
        new(robust, fuller)
    end
end


"""
    solve(problem::IVProblem, estimator::LIML)

Estimate treatment effect using Limited Information Maximum Likelihood.

# Arguments
- `problem::IVProblem`: IV problem specification
- `estimator::LIML`: LIML estimator configuration

# Returns
- `solution::IVSolution`: Solution with estimate, SE, CI, diagnostics

# Algorithm

1. **Compute k**: Solve eigenvalue problem for LIML k
   - k = λ_min of (Y D)'M_X(Y D) / (Y D)'M_Z(Y D)
   - Fuller modification: k → k - α/(n - K - p - 1) if fuller > 0

2. **K-Class Estimation**: Compute β̂_k
   - Transform data using k
   - Regress transformed Y on transformed D

3. **Standard Errors**: Robust or classical
   - Adjust for k-class weighting

4. **Diagnostics**: k value, comparison to 2SLS

# Returns

Solution contains:
- `estimate`: LIML estimate of treatment effect
- `se`: Standard error (robust or classical)
- `ci_lower`, `ci_upper`: (1-α)% confidence interval
- `diagnostics.k_liml`: LIML k value
- `diagnostics.fuller_alpha`: Fuller parameter used

# Example
```julia
problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
solution = solve(problem, LIML())

println("LIML estimate: \$(solution.estimate)")
println("LIML k: \$(solution.diagnostics.k_liml)")
```
"""
function solve(problem::IVProblem{T,P}, estimator::LIML) where {T<:Real,P<:NamedTuple}
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
    # Compute LIML k
    # ========================================================================

    # Build matrices
    # W = [Y D] (n × 2 for L=1 endogenous variable)
    W = hcat(y, d)

    # Instruments matrix: [1 Z X] or [1 Z]
    if isnothing(X)
        Z_full = hcat(ones(T, n), Z)
    else
        Z_full = hcat(ones(T, n), Z, X)
    end

    # Exogenous variables matrix: [1 X] or [1]
    if isnothing(X)
        X_full = ones(T, n, 1)
    else
        X_full = hcat(ones(T, n), X)
    end

    # Projection matrices
    # P_Z = Z_full(Z_full'Z_full)^(-1)Z_full'
    P_Z = Z_full * ((Z_full' * Z_full) \ Z_full')

    # M_Z = I - P_Z (residual maker for instruments)
    M_Z = I - P_Z

    # M_X = I - X_full(X_full'X_full)^(-1)X_full'
    M_X = I - X_full * ((X_full' * X_full) \ X_full')

    # Compute k as smallest eigenvalue of:
    # (W'M_X W)^(-1) (W'M_Z W)
    # Equivalently: solve generalized eigenvalue problem
    # (W'M_Z W)v = λ(W'M_X W)v

    A = W' * M_Z * W  # 2×2 matrix (changed from P_Z to M_Z)
    B = W' * M_X * W  # 2×2 matrix

    # Solve generalized eigenvalue problem Av = λBv
    # Equivalent to: B^(-1)Av = λv
    eigenvalues = eigvals(B \ A)

    # LIML k = minimum eigenvalue
    k_liml = minimum(real.(eigenvalues))

    # Fuller's modification: k → k - α/(n - K - p - 1)
    if estimator.fuller > 0.0
        df = n - K - p - 1
        if df <= 0
            throw(
                ArgumentError(
                    "Insufficient degrees of freedom for Fuller modification\n" *
                    "n=$n, K=$K, p=$p → df=$df ≤ 0\n" *
                    "Need n > K + p + 1",
                ),
            )
        end
        k_fuller = k_liml - estimator.fuller / df
        k = k_fuller
    else
        k = k_liml
    end

    # ========================================================================
    # K-Class Estimation
    # ========================================================================

    # K-class estimator: β̂_k = (D'M_k D)^(-1) D'M_k Y
    # where M_k = M_X - k(M_X - P_Z) = (1-k)M_X + kP_Z

    # Transformed instruments: M_k = (1-k)M_X + kP_Z
    # Apply to D and Y
    D_transformed = ((1 - k) * M_X + k * P_Z) * d
    Y_transformed = ((1 - k) * M_X + k * P_Z) * y

    # Add intercept (if needed - depends on whether X_full already has it)
    # For simplicity, regress transformed Y on transformed D without intercept
    # (transformation already accounts for intercept via M_X)

    # β̂_k = (D_t'D_t)^(-1) D_t'Y_t
    β_liml = (D_transformed' * D_transformed) \ (D_transformed' * Y_transformed)

    # Residuals
    u = y - β_liml * d

    # ========================================================================
    # Standard Errors
    # ========================================================================

    if estimator.robust
        # Robust standard error for k-class estimator
        # V = σ̂² (D'M_k D)^(-1)
        # where σ̂² accounts for heteroskedasticity

        # Fitted values
        y_fitted = β_liml * d
        residuals = y - y_fitted

        # Heteroskedasticity-robust variance
        # For k-class: V = (D'M_k D)^(-1) D'M_k Ω M_k D (D'M_k D)^(-1)
        Omega = Diagonal(residuals .^ 2)

        # M_k = (1-k)M_X + kP_Z
        M_k = (1 - k) * M_X + k * P_Z

        # Bread: (D'M_k D)^(-1)
        bread_inv = d' * M_k * d
        bread = 1 / bread_inv

        # Meat: D'M_k Ω M_k D
        meat = d' * M_k * Omega * M_k * d

        # Sandwich
        V = bread * meat * bread

        se_liml = sqrt(V)

    else
        # Classical standard error
        σ_sq = sum(u .^ 2) / (n - 1)  # Residual variance
        V = σ_sq / (d' * ((1 - k) * M_X + k * P_Z) * d)
        se_liml = sqrt(V)
    end

    # ========================================================================
    # Inference
    # ========================================================================

    z_crit = quantile(Normal(0, 1), 1 - alpha / 2)
    ci_lower = β_liml - z_crit * se_liml
    ci_upper = β_liml + z_crit * se_liml

    z_stat = β_liml / se_liml
    p_value = 2 * (1 - cdf(Normal(0, 1), abs(z_stat)))

    # ========================================================================
    # Overidentification Test
    # ========================================================================

    overid_pvalue = if K > 1
        # Sargan test with LIML residuals
        if isnothing(X)
            Z_overid = Z
        else
            Z_overid = hcat(Z, X)
        end

        Z_overid = hcat(ones(T, n), Z_overid)

        β_overid = (Z_overid' * Z_overid) \ (Z_overid' * u)
        u_overid = u - Z_overid * β_overid

        TSS = sum((u .- mean(u)) .^ 2)
        RSS = sum(u_overid .^ 2)
        R_sq = 1 - RSS / TSS

        sargan_stat = n * R_sq
        df_overid = K - 1

        1 - cdf(Chisq(df_overid), sargan_stat)
    else
        nothing
    end

    # ========================================================================
    # Diagnostics
    # ========================================================================

    diagnostics = (
        k_liml=k_liml,
        k_used=k,
        fuller_alpha=estimator.fuller,
        cragg_donald=cd_stat,
        robust_se=estimator.robust,
        n_instruments=K,
        n_covariates=p,
    )

    # ========================================================================
    # Return Solution
    # ========================================================================

    estimator_name = estimator.fuller > 0.0 ? "Fuller-LIML" : "LIML"

    return IVSolution(
        β_liml,
        se_liml,
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
