# CausalEstimators.jl User Guide

**Version**: 0.1.0 (Phase 2 Complete - RCT + PSM)
**Last Updated**: 2025-11-14

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Estimator Guide](#estimator-guide)
4. [API Reference](#api-reference)
5. [Advanced Usage](#advanced-usage)
6. [Performance Tips](#performance-tips)
7. [Troubleshooting](#troubleshooting)
8. [Contributing](#contributing)

---

## Installation

### From Source (Development)

```julia
using Pkg
Pkg.develop(path="/path/to/CausalEstimators.jl")
```

### Dependencies

CausalEstimators.jl requires:
- Julia 1.9+
- Standard library: Statistics, LinearAlgebra, Random
- External: StatsBase, Distributions, GLM, DataFrames, Combinatorics

All dependencies are installed automatically.

---

## Quick Start

### Example 1: Simple RCT Analysis

```julia
using CausalEstimators

# Your data
outcomes = [10.0, 12.0, 11.0, 4.0, 5.0, 3.0]
treatment = [true, true, true, false, false, false]

# Create problem
problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))

# Estimate ATE
solution = solve(problem, SimpleATE())

# View results
println("ATE: $(solution.estimate) ± $(solution.se)")
println("95% CI: [$(solution.ci_lower), $(solution.ci_upper)]")
println("Sample: $(solution.n_treated) treated, $(solution.n_control) control")
```

**Output**:
```
ATE: 7.0 ± 0.816
95% CI: [5.40, 8.60]
Sample: 3 treated, 3 control
```

### Example 2: Stratified RCT (Block Randomization)

```julia
using CausalEstimators

# Data with strata (blocks)
outcomes = [10.0, 12.0, 4.0, 5.0, 11.0, 13.0, 3.0, 6.0]
treatment = [true, false, true, false, true, false, true, false]
strata = [1, 1, 1, 1, 2, 2, 2, 2]  # Two blocks

# Create problem
problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha=0.05,))

# Estimate ATE with stratification
solution = solve(problem, StratifiedATE())

println("ATE: $(solution.estimate) ± $(solution.se)")
println("95% CI: [$(solution.ci_lower), $(solution.ci_upper)]")
```

### Example 3: ANCOVA (Regression Adjustment)

```julia
using CausalEstimators

# Data with pre-treatment covariate
outcomes = [10.0, 12.0, 11.0, 4.0, 5.0, 3.0]
treatment = [true, true, true, false, false, false]
baseline_score = [5.0, 6.0, 5.5, 4.5, 5.0, 4.0]
covariates = reshape(baseline_score, :, 1)  # n × 1 matrix

# Create problem
problem = RCTProblem(outcomes, treatment, covariates, nothing, (alpha=0.05,))

# Estimate ATE with regression adjustment
solution = solve(problem, RegressionATE())

println("ATE: $(solution.estimate) ± $(solution.se)")
println("95% CI: [$(solution.ci_lower), $(solution.ci_upper)]")
```

### Example 4: Propensity Score Matching

```julia
using CausalEstimators

# Observational data (not randomized)
outcomes = randn(200) .+ treatment .* 5.0  # True ATE = 5.0
treatment = rand(Bool, 200)
confounders = randn(200, 3)  # 3 confounding variables

# Create PSM problem
problem = PSMProblem(outcomes, treatment, confounders, (alpha=0.05,))

# Estimate ATE with 1:1 nearest neighbor matching
estimator = NearestNeighborPSM(M=1, with_replacement=false, caliper=0.25)
solution = solve(problem, estimator)

if solution.retcode == :Success
    println("ATE: $(solution.estimate) ± $(solution.se)")
    println("95% CI: [$(solution.ci_lower), $(solution.ci_upper)]")
    println("Matched: $(solution.n_matched) pairs")

    # Check balance
    balance = solution.balance_metrics
    println("Balanced: $(balance.balanced)")
    println("Mean SMD after: $(balance.balance_stats.mean_smd_after)")
else
    println("Matching failed: $(solution.retcode)")
end
```

---

## Estimator Guide

### When to Use Which Estimator?

| Estimator | Use When | Pros | Cons | Example |
|-----------|----------|------|------|---------|
| **SimpleATE** | Pure randomization, no covariates | Unbiased, simple, fast | Less precise if covariates predict outcomes | Medical trial with simple randomization |
| **StratifiedATE** | Block randomization | Variance reduction, respects design | Requires strata variable | Multi-site trial (site = stratum) |
| **RegressionATE** | Have baseline covariates | Variance reduction, flexible | Assumes linearity | Trial with baseline measurements |
| **PermutationTest** | Small sample, non-normal outcomes | Distribution-free, exact | Slow for large n | n<20 trial with outliers |
| **IPWATE** | Varying treatment propensity | Handles imbalance | Sensitive to extreme weights | Unequal cluster sizes |
| **NearestNeighborPSM** | Observational data (no randomization) | Intuitive, transparent | Assumes no hidden confounding | Retrospective study |

### Decision Tree

```
Is this a randomized experiment (RCT)?
│
├─ YES (RCT):
│  │
│  ├─ Do you have baseline covariates?
│  │  ├─ YES: Use RegressionATE (ANCOVA)
│  │  └─ NO: ──> Was it block-randomized?
│  │             ├─ YES: Use StratifiedATE
│  │             └─ NO: Use SimpleATE
│  │
│  └─ Is sample size small (n < 20)?
│     ├─ YES: Consider PermutationTest (distribution-free)
│     └─ NO: Use parametric methods (SimpleATE, StratifiedATE, RegressionATE)
│
└─ NO (Observational Study):
   │
   ├─ Do you have confounders measured?
   │  ├─ YES: Use NearestNeighborPSM
   │  └─ NO: ⚠️ Cannot estimate causal effect (omitted variable bias)
   │
   └─ Check assumptions:
      - ✅ Unconfoundedness: All confounders measured
      - ✅ Positivity: All units have chance of treatment
      - ✅ SUTVA: No interference between units
```

---

## API Reference

### Problem Construction

#### RCTProblem

```julia
RCTProblem(
    outcomes::Vector{T},
    treatment::Vector{Bool},
    covariates::Union{Nothing,Matrix{T}},
    strata::Union{Nothing,Vector{Int}},
    parameters::NamedTuple
) where T<:Real
```

**Fields**:
- `outcomes`: Observed outcomes for all units (length n)
- `treatment`: Treatment indicator (true=treated, false=control, length n)
- `covariates`: Optional covariates (n × p matrix, or nothing)
- `strata`: Optional stratification indicators (length n, positive integers)
- `parameters`: Estimation parameters (e.g., `(alpha=0.05,)`)

**Validation**:
- All vectors must have same length
- No NaN or Inf values
- At least one treated and one control unit
- Covariates must have n rows, p ≥ 1 columns
- Strata must have treatment variation within each stratum

#### PSMProblem

```julia
PSMProblem(
    outcomes::Vector{T},
    treatment::Vector{Bool},
    covariates::Matrix{T},
    parameters::NamedTuple
) where T<:Real
```

**Fields**:
- `outcomes`: Observed outcomes
- `treatment`: Treatment indicator
- `covariates`: Confounding variables (n × p matrix, **required** for PSM)
- `parameters`: `(alpha=0.05,)` for confidence level

### RCT Estimators

#### SimpleATE

```julia
SimpleATE()
```

Difference-in-means with Neyman variance.

**Method**: ATE = mean(Y[T=1]) - mean(Y[T=0])

**Requires**: Nothing (works with minimal data)

#### StratifiedATE

```julia
StratifiedATE()
```

Block randomization estimator.

**Method**: Weighted average of within-stratum ATEs

**Requires**: `problem.strata` must be provided

#### RegressionATE

```julia
RegressionATE()
```

ANCOVA (regression adjustment).

**Method**: ATE = coefficient of T in `Y ~ T + X`

**Requires**: `problem.covariates` must be provided

#### PermutationTest

```julia
PermutationTest(n_permutations::Union{Nothing,Int}=nothing, seed::Union{Nothing,Int}=nothing)
```

Fisher's exact test via permutation.

**Parameters**:
- `n_permutations`: Number of permutations (default: exact if n ≤ 12, else 10,000 Monte Carlo)
- `seed`: Random seed for reproducibility (Monte Carlo only)

**Method**: p-value = proportion of permutations with |ATE*| ≥ |ATE_obs|

**Requires**: Nothing

#### IPWATE

```julia
IPWATE(trim_propensity::Tuple{Float64,Float64}=(0.01, 0.99))
```

Inverse probability weighting (Horvitz-Thompson).

**Parameters**:
- `trim_propensity`: Bounds for propensity scores (default: [0.01, 0.99])

**Method**: ATE = (1/n) × Σ[Y×T/e(X) - Y×(1-T)/(1-e(X))]

**Requires**: `problem.covariates` must be provided (for propensity estimation)

### PSM Estimators

#### NearestNeighborPSM

```julia
NearestNeighborPSM(;
    M::Int=1,
    with_replacement::Bool=false,
    caliper::Float64=Inf,
    variance_method::Symbol=:abadie_imbens
)
```

Nearest neighbor propensity score matching.

**Parameters**:
- `M`: Number of control matches per treated unit (default: 1)
- `with_replacement`: Allow reuse of controls (default: false)
- `caliper`: Maximum propensity score distance (default: Inf = no caliper)
- `variance_method`: `:abadie_imbens` (default, recommended) or `:bootstrap`

**Method**:
1. Estimate propensity scores via logistic regression
2. Match treated to M nearest controls by propensity score
3. Compute ATE as mean difference in matched sample
4. Variance via Abadie-Imbens (2006, 2008) or pairs bootstrap

**Requires**: `problem.covariates` must be provided

**⚠️ Warning**: Bootstrap variance is INVALID for `with_replacement=true`

### Solution Interface

All solutions have the following fields:

#### RCTSolution

```julia
solution.estimate    # Point estimate (ATE)
solution.se          # Standard error
solution.ci_lower    # Lower 95% CI (or 1-alpha)
solution.ci_upper    # Upper 95% CI
solution.n_treated   # Number treated
solution.n_control   # Number control
solution.retcode     # :Success (or error code)
```

#### PermutationTestSolution

```julia
solution.observed_statistic  # Observed test statistic
solution.p_value             # Two-sided p-value
solution.n_permutations      # Number of permutations
solution.alternative         # "two-sided", "greater", "less"
solution.retcode             # :Success
```

#### PSMSolution

```julia
solution.estimate           # ATE
solution.se                 # Standard error
solution.ci_lower           # Lower CI
solution.ci_upper           # Upper CI
solution.n_matched          # Number of matched pairs (or treated units)
solution.n_dropped          # Units dropped (no match or outside common support)
solution.retcode            # :Success, :MatchingFailed, :CommonSupportFailed
solution.propensity_scores  # Vector of estimated propensities
solution.matched_indices    # Vector of (treated, control) index pairs
solution.balance_metrics    # Balance diagnostics (SMD, VR)
```

---

## Advanced Usage

### Sensitivity Analysis with remake()

```julia
# Original problem
problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))

# Solve
solution_95 = solve(problem, SimpleATE())

# Try different confidence level
problem_99 = remake(problem, parameters=(alpha=0.01,))
solution_99 = solve(problem_99, SimpleATE())

# Compare CI widths
println("95% CI width: $(solution_95.ci_upper - solution_95.ci_lower)")
println("99% CI width: $(solution_99.ci_upper - solution_99.ci_lower)")
```

### Comparing Multiple Estimators

```julia
# Create problem once
problem = RCTProblem(outcomes, treatment, covariates, strata, (alpha=0.05,))

# Solve with all applicable estimators
simple = solve(problem, SimpleATE())
stratified = solve(problem, StratifiedATE())  # Uses strata
regression = solve(problem, RegressionATE())  # Uses covariates

# Compare
println("Simple ATE: $(simple.estimate) ± $(simple.se)")
println("Stratified ATE: $(stratified.estimate) ± $(stratified.se)")
println("Regression ATE: $(regression.estimate) ± $(regression.se)")
```

### Handling PSM Failures

```julia
estimator = NearestNeighborPSM(M=1, with_replacement=false, caliper=0.1)
solution = solve(problem, estimator)

if solution.retcode == :Success
    println("✅ Matching succeeded")
    println("ATE: $(solution.estimate)")
    println("Matched: $(solution.n_matched) pairs")

elseif solution.retcode == :CommonSupportFailed
    println("⚠️ No common support region found")
    println("Try relaxing caliper or trimming extreme propensities")

elseif solution.retcode == :MatchingFailed
    println("⚠️ Matching failed (too few viable matches)")
    println("Dropped units: $(solution.n_dropped)")
    println("Try with_replacement=true or larger caliper")

else
    println("❌ Unknown error: $(solution.retcode)")
end
```

### Balance Diagnostics

```julia
solution = solve(problem, NearestNeighborPSM(M=1, with_replacement=false))

if solution.retcode == :Success
    balance = solution.balance_metrics

    println("Overall balance: $(balance.balanced)")
    println("Number of covariates: $(balance.balance_stats.n_covariates)")
    println("Covariates balanced: $(balance.balance_stats.n_balanced)")
    println("Mean SMD before: $(balance.balance_stats.mean_smd_before)")
    println("Mean SMD after: $(balance.balance_stats.mean_smd_after)")
    println("Improvement: $(round(balance.balance_stats.improvement * 100, digits=1))%")

    # Covariate-specific balance
    for j in 1:length(balance.smd_after)
        smd = balance.smd_after[j]
        vr = balance.vr_after[j]
        status = abs(smd) < 0.1 ? "✅" : "⚠️"
        println("Covariate $j: SMD = $(round(smd, digits=3)), VR = $(round(vr, digits=2)) $status")
    end
end
```

---

## Performance Tips

### 1. Type Stability

CausalEstimators.jl is 100% type-stable. Ensure your data is type-stable:

```julia
# ✅ Good (type-stable)
outcomes = Float64[10.0, 12.0, 4.0, 5.0]
treatment = Bool[true, true, false, false]

# ❌ Bad (type-unstable)
outcomes = [10, 12.0, 4, 5.0]  # Mixed Int and Float64
treatment = [1, 1, 0, 0]       # Int instead of Bool
```

### 2. Estimator Selection

For large datasets (n > 10,000):

| Estimator | Speed | Notes |
|-----------|-------|-------|
| SimpleATE | ⚡⚡⚡ Fastest | 0.02ms for n=10,000 |
| StratifiedATE | ⚡⚡⚡ Fast | 0.13ms for n=10,000 |
| IPWATE | ⚡⚡ Fast | 0.12ms for n=10,000 |
| RegressionATE | ⚡ Moderate | 1.5ms for n=10,000 (p=1) |
| PermutationTest | 🐢 Slow | 131ms for n=10,000 (1,000 perms) |
| NearestNeighborPSM | ⚡ Moderate | Depends on M and caliper |

### 3. PSM Configuration

For faster PSM:
- Use `with_replacement=true` (more robust, avoids partial matches)
- Use smaller caliper (drops bad matches early)
- Use smaller `M` (fewer matches per treated unit)

For better balance:
- Use larger `M` (2:1 or 3:1 matching)
- Check `balance_metrics.balanced` flag
- Iterate on caliper size

---

## Troubleshooting

### Common Errors

#### "Mismatched lengths"

```
ArgumentError: CRITICAL ERROR: Mismatched lengths.
```

**Solution**: Ensure all vectors have same length:
```julia
length(outcomes) == length(treatment)  # Must be true
```

#### "No treatment variation"

```
ArgumentError: CRITICAL ERROR: No treatment variation.
```

**Solution**: At least one treated and one control unit required:
```julia
any(treatment)        # At least one true
any(.!treatment)      # At least one false
```

#### "Unsupported problem-estimator combination"

```
ArgumentError: CRITICAL ERROR: Unsupported problem-estimator combination.
```

**Solution**:
- `StratifiedATE` requires `problem.strata` to be provided
- `RegressionATE` requires `problem.covariates` to be provided
- `IPWATE` requires `problem.covariates` to be provided
- `NearestNeighborPSM` requires PSMProblem (not RCTProblem)

#### "Common support failed" (PSM)

```
solution.retcode == :CommonSupportFailed
```

**Solution**:
- Relax caliper: try `caliper=0.5` or `caliper=Inf`
- Check propensity scores: `solution.propensity_scores`
- Trim extreme propensities: remove units with e(X) < 0.01 or > 0.99

#### "Matching failed" (PSM)

```
solution.retcode == :MatchingFailed
```

**Solution**:
- Use `with_replacement=true`
- Relax caliper
- Check `solution.n_dropped` to see how many units failed to match

### Performance Issues

#### PermutationTest too slow

```julia
# Instead of exact test (all permutations)
solution = solve(problem, PermutationTest())

# Use Monte Carlo with fewer permutations
solution = solve(problem, PermutationTest(1000, 42))
```

#### PSM too slow

```julia
# Use caliper to prune search space
estimator = NearestNeighborPSM(M=1, with_replacement=false, caliper=0.2)

# Or use with_replacement (faster matching)
estimator = NearestNeighborPSM(M=1, with_replacement=true)
```

---

## Contributing

### Reporting Issues

Open an issue with:
1. Minimal reproducible example (MWE)
2. Expected behavior
3. Actual behavior
4. Julia version and `Pkg.status("CausalEstimators")`

### Adding New Estimators

Follow the SciML pattern:
1. Define abstract types (if new method family)
2. Create problem struct (if needed)
3. Create estimator struct
4. Implement `solve(problem, estimator)`
5. Add tests (unit + integration + golden reference)

See `docs/PHASE_1_RCT_COMPLETE.md` for detailed examples.

### Code Style

- Follow SciML formatting (92-char lines, 4-space indents)
- Never fail silently (explicit errors)
- Immutability by default
- Type-annotate all parameters and returns
- Comprehensive docstrings with examples

---

## References

### Textbooks

- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.
- Rosenbaum, P. R. (2017). *Observation and Experiment: An Introduction to Causal Inference*. Harvard University Press.
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.

### Methods Papers

**Propensity Score Matching**:
- Abadie, A., & Imbens, G. W. (2006). "Large Sample Properties of Matching Estimators for Average Treatment Effects." *Econometrica*, 74(1), 235-267.
- Abadie, A., & Imbens, G. W. (2008). "On the Failure of the Bootstrap for Matching Estimators." *Econometrica*, 76(6), 1537-1557.
- Austin, P. C. (2009). "Balance diagnostics for comparing the distribution of baseline covariates between treatment groups in propensity-score matched samples." *Statistics in Medicine*, 28(25), 3083-3107.

**SciML Ecosystem**:
- Rackauckas, C., & Nie, Q. (2017). "DifferentialEquations.jl–a performant and feature-rich ecosystem for solving differential equations in Julia." *Journal of Open Research Software*, 5(1).
- SciML Documentation: https://docs.sciml.ai/

---

**Version**: 0.1.0 | **Last Updated**: 2025-11-14 | **License**: MIT
