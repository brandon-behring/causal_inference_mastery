"""
Two-Stage Least Squares (2SLS) estimator for instrumental variables.

Implements the classic Theil (1953) and Basmann (1957) two-stage procedure
with robust inference following White (1980) and Stock & Watson (2008).

# References
- Theil, H. (1953). "Repeated Least Squares Applied to Complete Equation Systems."
  Central Planning Bureau, The Hague.
- Basmann, R. L. (1957). "A Generalized Classical Method of Linear Estimation of
  Coefficients in a Structural Equation." *Econometrica*, 25(1), 77-83.
- White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator."
  *Econometrica*, 48(4), 817-838.
- Stock, J. H., & Watson, M. W. (2008). "Heteroskedasticity-Robust Standard Errors
  for Fixed Effects Panel Data Regression." *Econometrica*, 76(1), 155-174.
"""

using LinearAlgebra
using Statistics
using Distributions

"""
    TSLS <: AbstractIVEstimator

Two-Stage Least Squares estimator.

# Fields
- `robust::Bool`: Use heteroskedasticity-robust standard errors (default: true)

# Method

## Stage 1: First-Stage Regression
Regress endogenous variable D on instruments Z and exogenous covariates X:
```
D = Z'π + X'γ + ε
```

Obtain predicted values D̂ = Z'π̂ + X'γ̂

## Stage 2: Second-Stage Regression
Regress outcome Y on predicted D̂ and exogenous covariates X:
```
Y = β₀ + βD̂ + X'δ + u
```

The coefficient β is the 2SLS estimate of the treatment effect.

## Standard Errors

**Robust (default)**: Heteroskedasticity-robust (White 1980):
```
V_robust = (X̂'X̂)^(-1) X̂'ΩX̂ (X̂'X̂)^(-1)
```
where Ω = diag(û²) and û are second-stage residuals.

**Non-robust**: Homoskedastic (classical):
```
V_classic = σ̂² (X̂'X̂)^(-1)
```
where σ̂² = û'û / (n - k).

# Properties

**Consistency**: Under valid instruments, 2SLS is consistent for β.

**Asymptotic Normality**: √n(β̂ - β) → N(0, V) as n → ∞.

**Weak IV Bias**: With weak instruments (F < 10), 2SLS is biased toward OLS.
- Use LIML for better finite-sample properties
- Use Anderson-Rubin test for robust inference

# Example
```julia
problem = IVProblem(outcomes, treatment, instruments, covariates, (alpha=0.05,))
estimator = TSLS(robust=true)  # Heteroskedasticity-robust SEs
solution = solve(problem, estimator)

if solution.weak_iv_warning
    @warn "Weak instruments detected. Consider LIML or weak IV robust inference."
end
```

# References
- Stock, J. H., & Watson, M. W. (2015). *Introduction to Econometrics* (3rd ed.).
  Pearson. Chapter 12.
- Angrist, J. D., & Pischke, J. S. (2008). *Mostly Harmless Econometrics*.
  Princeton University Press. Chapter 4.
"""
struct TSLS <: AbstractIVEstimator
    robust::Bool

    function TSLS(; robust::Bool = true)
        new(robust)
    end
end


"""
    solve(problem::IVProblem, estimator::TSLS)

Estimate treatment effect using Two-Stage Least Squares.

# Arguments
- `problem::IVProblem`: IV problem specification
- `estimator::TSLS`: 2SLS estimator configuration

# Returns
- `solution::IVSolution`: Solution with estimate, SE, CI, diagnostics

# Algorithm

1. **First Stage**: Regress D on [1 Z X]
   - Obtain fitted values D̂
   - Compute first-stage F-statistic
   - Check for weak instruments

2. **Second Stage**: Regress Y on [1 D̂ X]
   - Obtain 2SLS estimate β̂
   - Compute robust or classical standard errors
   - Construct confidence intervals

3. **Diagnostics**:
   - First-stage F-statistic
   - Cragg-Donald statistic
   - Weak IV warning
   - Overidentification test (if K > L)

# Returns

Solution contains:
- `estimate`: 2SLS estimate of treatment effect
- `se`: Standard error (robust or classical)
- `ci_lower`, `ci_upper`: (1-α)% confidence interval
- `p_value`: Two-sided p-value for H₀: β = 0
- `first_stage_fstat`: First-stage F-statistic
- `weak_iv_warning`: True if F < 10 or CD < Stock-Yogo threshold
- `overid_pvalue`: Sargan test p-value (if K > L)
- `diagnostics`: Additional diagnostic information

# Example
```julia
# Simple 2SLS
problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
solution = solve(problem, TSLS())

println("2SLS estimate: \$(solution.estimate)")
println("Robust SE: \$(solution.se)")
println("95% CI: [\$(solution.ci_lower), \$(solution.ci_upper)]")
println("First-stage F: \$(solution.first_stage_fstat)")

if solution.weak_iv_warning
    println("⚠️  Weak instruments detected!")
end
```
"""
function solve(problem::IVProblem{T,P}, estimator::TSLS) where {T<:Real,P<:NamedTuple}
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
    # Stage 1: First-Stage Regression
    # ========================================================================

    # Build first-stage regressor matrix: [1 Z X]
    if isnothing(X)
        X_fs = hcat(ones(T, n), Z)
    else
        X_fs = hcat(ones(T, n), Z, X)
    end

    # First-stage: D ~ 1 + Z + X
    # β_fs = (X_fs'X_fs)^(-1) X_fs'D
    β_fs = (X_fs' * X_fs) \ (X_fs' * d)

    # Fitted values: D̂ = X_fs * β_fs
    d_fitted = X_fs * β_fs

    # First-stage residuals
    ε_fs = d - d_fitted

    # ========================================================================
    # Weak IV Diagnostics
    # ========================================================================

    # First-stage F-statistic
    fstat, fstat_pval = first_stage_fstat(d, Z, X)

    # Cragg-Donald statistic
    cd_stat = cragg_donald_stat(d, Z, X)

    # Check for weak instruments
    is_weak, weak_warning = weak_iv_warning(fstat, cd_stat, K)

    # ========================================================================
    # Stage 2: Second-Stage Regression
    # ========================================================================

    # Build second-stage regressor matrix: [1 D̂ X]
    if isnothing(X)
        X_ss = hcat(ones(T, n), d_fitted)
    else
        X_ss = hcat(ones(T, n), d_fitted, X)
    end

    # Second-stage: Y ~ 1 + D̂ + X
    # β_ss = (X_ss'X_ss)^(-1) X_ss'Y
    β_ss = (X_ss' * X_ss) \ (X_ss' * y)

    # 2SLS estimate is coefficient on D̂ (second column, after intercept)
    β_tsls = β_ss[2]

    # Second-stage fitted values and residuals
    y_fitted = X_ss * β_ss
    u_ss = y - y_fitted

    # ========================================================================
    # Standard Errors
    # ========================================================================

    if estimator.robust
        # Heteroskedasticity-robust standard errors (White 1980)
        # Need to adjust for use of fitted D̂ instead of actual D

        # Correct second-stage design matrix: [1 Z X] (instruments, not D̂)
        if isnothing(X)
            X_iv = hcat(ones(T, n), Z)
        else
            X_iv = hcat(ones(T, n), Z, X)
        end

        # Projection matrix: P_Z = X_iv (X_iv'X_iv)^(-1) X_iv'
        P_Z = X_iv * ((X_iv' * X_iv) \ X_iv')

        # Project second-stage regressors onto instruments
        if isnothing(X)
            X_proj = hcat(ones(T, n), d_fitted)
        else
            X_proj = hcat(ones(T, n), d_fitted, X)
        end

        # Robust variance matrix: V = (X̂'X̂)^(-1) X̂'ΩX̂ (X̂'X̂)^(-1)
        # where Ω = diag(u²) and X̂ = P_Z * X_proj

        # Meat of sandwich: X̂'ΩX̂
        Omega = Diagonal(u_ss .^ 2)
        meat = X_proj' * Omega * X_proj

        # Bread of sandwich: (X̂'X̂)^(-1)
        bread_inv = X_proj' * X_proj
        bread = inv(bread_inv)

        # Sandwich: V = bread * meat * bread
        V = bread * meat * bread

        # Standard error for treatment effect (2nd parameter, after intercept)
        se_tsls = sqrt(V[2, 2])

    else
        # Classical homoskedastic standard errors
        # σ² = u'u / (n - k)
        k_ss = size(X_ss, 2)  # Number of second-stage regressors
        σ_sq = sum(u_ss .^ 2) / (n - k_ss)

        # V = σ² (X_ss'X_ss)^(-1)
        V = σ_sq * inv(X_ss' * X_ss)

        # Standard error for treatment effect
        se_tsls = sqrt(V[2, 2])
    end

    # ========================================================================
    # Inference
    # ========================================================================

    # Critical value for (1-α)% CI
    z_crit = T(quantile(Normal(0, 1), 1 - alpha / 2))

    # Confidence interval
    ci_lower = β_tsls - z_crit * se_tsls
    ci_upper = β_tsls + z_crit * se_tsls

    # P-value for H₀: β = 0
    z_stat = β_tsls / se_tsls
    p_value = T(2 * (1 - cdf(Normal(0, 1), abs(z_stat))))

    # ========================================================================
    # Overidentification Test (if K > L=1)
    # ========================================================================

    overid_pvalue = if K > 1
        # Sargan test statistic: nR² from regression of û on Z
        if isnothing(X)
            Z_overid = Z
        else
            Z_overid = hcat(Z, X)
        end

        # Add intercept
        Z_overid = hcat(ones(T, n), Z_overid)

        # Regress second-stage residuals on instruments
        β_overid = (Z_overid' * Z_overid) \ (Z_overid' * u_ss)
        u_overid = u_ss - Z_overid * β_overid

        # R² from this regression
        TSS = sum((u_ss .- mean(u_ss)) .^ 2)
        RSS = sum(u_overid .^ 2)
        R_sq = 1 - RSS / TSS

        # Sargan statistic: nR² ~ χ²(K - L)
        sargan_stat = n * R_sq
        df_overid = K - 1  # K instruments - L=1 endogenous variables

        # P-value from chi-squared distribution
        1 - cdf(Chisq(df_overid), sargan_stat)
    else
        # Exactly identified (K = L = 1): no overidentification test
        nothing
    end

    # ========================================================================
    # Diagnostics
    # ========================================================================

    diagnostics = (
        first_stage_coef=β_fs[2:(1+K)],  # Coefficients on instruments (excluding intercept)
        first_stage_se=sqrt.(diag(inv(X_fs' * X_fs) * sum(ε_fs .^ 2) / (n - size(X_fs, 2)))),
        cragg_donald=cd_stat,
        robust_se=estimator.robust,
        second_stage_coef=β_ss,
        n_instruments=K,
        n_covariates=p,
    )

    # ========================================================================
    # Return Solution
    # ========================================================================

    return IVSolution(
        β_tsls,
        se_tsls,
        ci_lower,
        ci_upper,
        p_value,
        n,
        K,
        p,
        fstat,
        overid_pvalue,
        is_weak,
        "2SLS",
        alpha,
        diagnostics,
    )
end
