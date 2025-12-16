"""
Simple difference-in-means estimator for average treatment effect.

Implements the simplest RCT estimator with Neyman heteroskedasticity-robust variance.
"""

"""
    SimpleATE <: AbstractRCTEstimator

Simple difference-in-means estimator for average treatment effect in randomized experiments.

# Mathematical Foundation

Under randomization, the average treatment effect (ATE) is identified by:

```math
\\tau = \\mathbb{E}[Y(1) - Y(0)] = \\mathbb{E}[Y|T=1] - \\mathbb{E}[Y|T=0]
```

The sample estimator is:

```math
\\hat{\\tau} = \\bar{Y}_1 - \\bar{Y}_0
```

Standard error uses Neyman heteroskedasticity-robust variance:

```math
SE(\\hat{\\tau}) = \\sqrt{\\frac{s_1^2}{n_1} + \\frac{s_0^2}{n_0}}
```

where ``s_t^2`` is the sample variance for treatment group ``t \\in \\{0,1\\}``.

# Variance Estimation

**Neyman Conservative Variance** (Neyman 1923):
- Allows heteroskedasticity across treatment groups (``\\sigma_1^2 \\neq \\sigma_0^2``)
- Does NOT assume equal variances (unlike pooled t-test)
- Conservative: valid even with heterogeneous treatment effects
- Equivalent to Welch's t-test variance (unequal variances)

# Usage

```julia
using CausalEstimators

# Create problem
outcomes = [10.0, 12.0, 4.0, 5.0]
treatment = [true, true, false, false]
problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))

# Estimate ATE
solution = solve(problem, SimpleATE())

# Extract results
solution.estimate  # Point estimate
solution.se        # Standard error
solution.ci_lower  # Lower 95% CI bound
solution.ci_upper  # Upper 95% CI bound
```

# Returns

Returns `RCTSolution` with fields:
- `estimate::Float64`: Point estimate of ATE
- `se::Float64`: Standard error (Neyman variance)
- `ci_lower::Float64`: Lower confidence bound
- `ci_upper::Float64`: Upper confidence bound
- `n_treated::Int`: Number of treated units
- `n_control::Int`: Number of control units
- `retcode::Symbol`: `:Success` if estimation succeeded
- `original_problem::RCTProblem`: Original problem for reproducibility

# References

- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social,
  and Biomedical Sciences*. Cambridge University Press. Chapter 6.
- Neyman, J. (1923). On the Application of Probability Theory to Agricultural
  Experiments. *Statistical Science*, 5(4), 465-472.
"""
struct SimpleATE <: AbstractRCTEstimator end

"""
    solve(problem::RCTProblem, estimator::SimpleATE)

Estimate average treatment effect using simple difference-in-means.

# Algorithm

1. Split outcomes by treatment group
2. Compute means: `μ₁ = mean(Y₁)`, `μ₀ = mean(Y₀)`
3. ATE estimate: `τ̂ = μ₁ - μ₀`
4. Standard error: Neyman heteroskedasticity-robust variance
5. Confidence interval: t-distribution with Satterthwaite degrees of freedom

# Returns

`RCTSolution` with:
- Point estimate of ATE
- Neyman robust standard error
- Confidence interval
- Sample sizes
- Success status

# Examples

```julia
outcomes = [10.0, 12.0, 4.0, 5.0]
treatment = [true, true, false, false]
problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))
solution = solve(problem, SimpleATE())
```
"""
function solve(problem::RCTProblem, estimator::SimpleATE)::RCTSolution
    # Extract from problem (destructuring)
    (; outcomes, treatment, parameters) = problem

    # Split by treatment group
    treated_mask = treatment
    control_mask = .!treatment

    y1 = outcomes[treated_mask]
    y0 = outcomes[control_mask]

    # Sample sizes
    n1 = length(y1)
    n0 = length(y0)

    # Point estimate: difference in means
    μ1 = mean(y1)
    μ0 = mean(y0)
    ate = μ1 - μ0

    # Standard error: Neyman heteroskedasticity-robust variance
    var1 = var(y1)
    var0 = var(y0)
    var_ate = var1 / n1 + var0 / n0
    se = sqrt(var_ate)

    # Confidence interval using t-distribution with Satterthwaite df
    # (matches Python implementation, better for small samples)
    df = satterthwaite_df(var1, n1, var0, n0)
    ci_lower, ci_upper = confidence_interval_t(ate, se, parameters.alpha, df)

    # Build solution
    return RCTSolution(
        estimate = ate,
        se = se,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        n_treated = n1,
        n_control = n0,
        retcode = :Success,
        original_problem = problem,
    )
end
