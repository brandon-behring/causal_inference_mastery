"""
Regression-adjusted ATE estimator using ANCOVA.
"""

"""
    RegressionATE <: AbstractRCTEstimator

Regression-adjusted estimator for average treatment effect (ANCOVA).

Uses covariate adjustment to improve precision. Under randomization, this provides
more efficient estimates than simple difference-in-means.

# Mathematical Foundation

Estimate ATE via OLS regression:

```math
Y_i = \\alpha + \\tau T_i + \\beta' X_i + \\epsilon_i
```

The coefficient ``\\tau`` is the ATE estimate.

# Variance Estimation

**HC3 Heteroskedasticity-Robust Standard Errors** (Long & Ervin 2000):

We use HC3 ("Jackknife") robust standard errors specifically because:
- **Best small-sample properties**: HC3 has lowest bias in small samples (n < 250)
- **Conservative**: More conservative than HC0, HC1, HC2 (protects Type I error)
- **Leverage adjustment**: Down-weights high-leverage observations appropriately
- **Recommended**: Long & Ervin (2000) recommend HC3 for small samples

HC3 variance for coefficient ``j``:

```math
Var(\\hat{\\beta}_j) = (X'X)^{-1} \\left[\\sum_i \\frac{e_i^2}{(1-h_i)^2} x_i x_i'\\right] (X'X)^{-1}
```

where ``e_i`` is residual, ``h_i`` is leverage (diagonal of hat matrix).

**Alternative estimators**:
- HC0: No finite-sample correction (biased downward in small samples)
- HC1: df correction only (still biased in small samples)
- HC2: Less conservative than HC3 (higher Type I error)

**References**:
- Long, J. S., & Ervin, L. H. (2000). Using Heteroscedasticity Consistent Standard
  Errors in the Linear Regression Model. *The American Statistician*, 54(3), 217-224.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance
  matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305-325.

# Usage

```julia
# RCT with covariates
outcomes = [10.0, 12.0, 4.0, 5.0]
treatment = [true, true, false, false]
X = [5.0 2.0; 6.0 3.0; 5.5 2.5; 4.5 2.0]  # n × p covariate matrix

problem = RCTProblem(outcomes, treatment, X, nothing, (alpha=0.05,))
solution = solve(problem, RegressionATE())
```

# Requirements

- Problem must have `covariates` field populated
- Covariate matrix must be full rank

# Benefits

- Increased precision (smaller standard errors than SimpleATE)
- Valid under randomization even if model is misspecified
- Particularly helpful when covariates predict outcomes

# References

- Imbens & Rubin (2015), Chapter 7: Regression Methods
- Freedman, D. A. (2008). On regression adjustments in experiments with several treatments.
  *Annals of Applied Statistics*, 2(1), 176-196.
"""
struct RegressionATE <: AbstractRCTEstimator end

"""
    solve(problem::RCTProblem, estimator::RegressionATE)::RCTSolution

Compute regression-adjusted ATE using ANCOVA with HC3 robust SE.

Fits regression Y = α + τT + β'X + ε and extracts treatment coefficient τ as ATE.
Under randomization, this is unbiased and more efficient than simple difference-in-means
when covariates predict outcomes.

# Algorithm

1. Build design matrix [1, T, X]
2. Solve OLS normal equations: β = (X'X)^{-1} X'Y
3. Extract τ (treatment coefficient, index 2)
4. Compute residuals
5. Compute HC3 heteroskedasticity-robust standard errors
6. Extract SE for τ
7. Construct confidence interval

# Validation

- Covariates field must be populated (not nothing)
- Design matrix must be full rank (non-singular X'X)
"""
function solve(problem::RCTProblem, estimator::RegressionATE)::RCTSolution
    (; outcomes, treatment, covariates, parameters) = problem

    # Validate covariates field is populated
    if isnothing(covariates)
        throw(
            ArgumentError(
                "CRITICAL ERROR: Covariates field is nothing.\n" *
                "Function: solve(RegressionATE)\n" *
                "RegressionATE requires covariates to be populated.\n" *
                "Use SimpleATE for designs without covariates.",
            ),
        )
    end

    n = length(outcomes)

    # Ensure covariates is a matrix (n × p)
    X_covariates = if ndims(covariates) == 1
        reshape(covariates, n, 1)
    else
        covariates
    end

    # Build design matrix: [intercept, treatment, covariates]
    # Convert boolean treatment to Float64 for matrix operations
    treatment_col = Float64.(treatment)

    X_design = hcat(
        ones(n),           # Intercept
        treatment_col,     # Treatment indicator
        X_covariates       # Covariates
    )

    # Solve OLS: β = (X'X)^{-1} X'Y
    XtX = X_design' * X_design
    XtY = X_design' * outcomes

    coef = try
        XtX \ XtY  # Solve linear system
    catch e
        if isa(e, LinearAlgebra.SingularException)
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Singular design matrix.\n" *
                    "Function: solve(RegressionATE)\n" *
                    "This may be due to perfect collinearity in covariates.\n" *
                    "Check for duplicate or linearly dependent columns.",
                ),
            )
        else
            rethrow(e)
        end
    end

    # Extract coefficients
    # coef = [intercept, tau, beta_cov_1, ..., beta_cov_p]
    tau = coef[2]  # ATE (treatment coefficient)

    # Compute residuals
    y_fitted = X_design * coef
    residuals = outcomes - y_fitted

    # Heteroskedasticity-robust standard errors (HC3)
    se_all = robust_se_hc3(residuals, X_design)
    se = se_all[2]  # SE for treatment coefficient

    # Confidence interval
    ci_lower, ci_upper = confidence_interval(tau, se, parameters.alpha)

    # Total counts
    n_treated = sum(treatment)
    n_control = sum(.!treatment)

    return RCTSolution(
        estimate = tau,
        se = se,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        n_treated = n_treated,
        n_control = n_control,
        retcode = :Success,
        original_problem = problem,
    )
end
