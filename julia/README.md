# CausalEstimators.jl

**Production-quality causal inference estimators for Julia**

[![Tests](https://img.shields.io/badge/tests-460%2F460%20passing-brightgreen)](test/)
[![Phase](https://img.shields.io/badge/phase-2%20complete-blue)]()
[![Julia](https://img.shields.io/badge/julia-1.9%2B-purple)](https://julialang.org/)

---

## Overview

CausalEstimators.jl implements state-of-the-art causal inference methods following the [SciML](https://docs.sciml.ai/) Problem-Estimator-Solution architecture. The package provides production-ready estimators for randomized controlled trials (RCTs) and observational studies with comprehensive testing and cross-language validation.

**Design Philosophy**:
- **Immutable data structures** - Problems and solutions are immutable for reproducibility
- **Fail fast** - Input validation at construction, never fail silently
- **Type stable** - 100% type-stable for performance (8-98x faster than Python)
- **Comprehensive testing** - 460 tests with cross-language validation (rtol < 1e-10)
- **SciML integration** - Universal `solve(problem, estimator)` interface

---

## Features

### Randomized Controlled Trials (RCTs)

| Estimator | Method | Use Case |
|-----------|--------|----------|
| **SimpleATE** | Difference-in-means | Pure randomization |
| **StratifiedATE** | Block randomization | Stratified/clustered designs |
| **RegressionATE** | ANCOVA | Baseline covariate adjustment |
| **PermutationTest** | Fisher exact test | Small samples, distribution-free |
| **IPWATE** | Inverse probability weighting | Varying treatment propensity |

### Observational Studies (PSM)

| Estimator | Method | Use Case |
|-----------|--------|----------|
| **NearestNeighborPSM** | Propensity score matching | Observational data with confounders |

**Features**:
- M:1 matching (configurable)
- With/without replacement
- Caliper matching for common support
- Abadie-Imbens (2006, 2008) variance estimator
- Comprehensive balance diagnostics (SMD, variance ratios)

---

## Installation

### From Source (Development)

```julia
using Pkg
Pkg.develop(path="/path/to/CausalEstimators.jl")
```

### Requirements
- Julia 1.9+
- Dependencies: StatsBase, Distributions, GLM, DataFrames, Combinatorics

All dependencies installed automatically.

---

## Quick Start

### Example 1: Simple RCT

```julia
using CausalEstimators

# Your data
outcomes = [10.0, 12.0, 11.0, 4.0, 5.0, 3.0]
treatment = [true, true, true, false, false, false]

# Create problem
problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))

# Estimate ATE
solution = solve(problem, SimpleATE())

# Results
println("ATE: $(solution.estimate) ± $(solution.se)")
println("95% CI: [$(solution.ci_lower), $(solution.ci_upper)]")
```

**Output**:
```
ATE: 7.0 ± 0.816
95% CI: [5.40, 8.60]
```

### Example 2: ANCOVA (Regression Adjustment)

```julia
using CausalEstimators

outcomes = [10.0, 12.0, 11.0, 4.0, 5.0, 3.0]
treatment = [true, true, true, false, false, false]
baseline = [5.0, 6.0, 5.5, 4.5, 5.0, 4.0]
covariates = reshape(baseline, :, 1)  # n × 1 matrix

# Create problem with covariates
problem = RCTProblem(outcomes, treatment, covariates, nothing, (alpha=0.05,))

# Regression-adjusted ATE
solution = solve(problem, RegressionATE())

println("Adjusted ATE: $(solution.estimate) ± $(solution.se)")
```

### Example 3: Propensity Score Matching

```julia
using CausalEstimators

# Observational data (not randomized)
n = 200
treatment = rand(Bool, n)
confounders = randn(n, 3)  # 3 confounding variables
outcomes = randn(n) .+ treatment .* 5.0  # True ATE = 5.0

# PSM problem
problem = PSMProblem(outcomes, treatment, confounders, (alpha=0.05,))

# 1:1 nearest neighbor matching
estimator = NearestNeighborPSM(M=1, with_replacement=false, caliper=0.25)
solution = solve(problem, estimator)

if solution.retcode == :Success
    println("ATE: $(solution.estimate) ± $(solution.se)")
    println("Matched: $(solution.n_matched) pairs")

    # Check balance
    balance = solution.balance_metrics
    println("Balanced: $(balance.balanced)")
    println("Mean SMD: $(balance.balance_stats.mean_smd_after)")
end
```

---

## Documentation

**Comprehensive guides** in `docs/`:

- **[USER_GUIDE.md](docs/USER_GUIDE.md)** (648 lines) - Complete user documentation
  - Installation and quick start
  - Estimator guide with decision tree
  - API reference
  - Advanced usage and troubleshooting
  - Performance tips

- **[PHASE_0_FOUNDATION.md](docs/PHASE_0_FOUNDATION.md)** (506 lines) - Architecture
  - SciML Problem-Estimator-Solution pattern
  - Type hierarchy and design philosophy
  - Foundation implementation (880 lines)

- **[PHASE_1_RCT_COMPLETE.md](docs/PHASE_1_RCT_COMPLETE.md)** (365 lines) - RCT Estimators
  - 5 RCT estimators detailed
  - 199 tests + 46 cross-language validation tests
  - Performance benchmarks

- **[PHASE_2_PSM_COMPLETE.md](docs/PHASE_2_PSM_COMPLETE.md)** (325 lines) - PSM Estimators
  - Propensity score matching implementation
  - Balance diagnostics
  - 214 tests with Monte Carlo validation

---

## Testing

**Test Suite**: 460 tests (100% passing)

```julia
using Pkg
Pkg.test("CausalEstimators")
```

**Test Breakdown**:
- Foundation (Phase 0): 27 tests
- RCT estimators (Phase 1): 199 tests
- PSM estimators (Phase 2): 214 tests
- Cross-language validation: 46 tests (vs Python, rtol < 1e-10)

**Coverage**: ~95% for core estimators, 100% for error handling

---

## Performance

**Benchmarks** (n=10,000):

| Estimator | Time (ms) | Memory (KB) | Speedup vs Python |
|-----------|-----------|-------------|-------------------|
| SimpleATE | 0.018 | 79.7 | **98x faster** |
| StratifiedATE | 0.133 | 178.8 | **71x faster** |
| RegressionATE | 1.552 | 3,286 | **8x faster** |
| IPWATE | 0.123 | 788.6 | **53x faster** |

**Type Stability**: 100% (all estimators fully type-stable)

---

## Development Roadmap

**Completed** (Phases 0-2):
- ✅ **Phase 0**: Foundation and SciML architecture
- ✅ **Phase 1**: RCT estimators (SimpleATE, StratifiedATE, RegressionATE, PermutationTest, IPWATE)
- ✅ **Phase 2**: Propensity Score Matching (NearestNeighborPSM with balance diagnostics)

**Planned** (Phase 3+):
- ⏳ **Phase 3**: Regression Discontinuity Design (RDD)
  - Sharp RDD with local linear regression
  - Bandwidth selection (IK, CCT optimal bandwidth)
  - Fuzzy RDD support
  - Cross-validation with R's `rdrobust`

- 🔮 **Phase 4**: Difference-in-Differences (DiD)
  - Two-way fixed effects
  - Parallel trends testing
  - Event study plots

- 🔮 **Phase 5**: Instrumental Variables (IV)
  - 2SLS, LIML, GMM estimators
  - Weak instrument diagnostics
  - Overidentification tests

- 🔮 **Cross-cutting improvements**:
  - Plotting infrastructure (Phase 3.5) - foundation complete
  - Sensitivity analysis framework
  - Power analysis tools
  - LaTeX table export

---

## Architecture

**SciML Problem-Estimator-Solution Pattern**:

```julia
# 1. Define problem (immutable data specification)
problem = RCTProblem(outcomes, treatment, covariates, strata, parameters)

# 2. Choose estimator (algorithm specification)
estimator = SimpleATE()  # or StratifiedATE(), RegressionATE(), etc.

# 3. Solve (universal interface)
solution = solve(problem, estimator)

# 4. Access results
solution.estimate      # Point estimate
solution.se            # Standard error
solution.ci_lower      # Lower confidence bound
solution.ci_upper      # Upper confidence bound
solution.retcode       # :Success
```

**Benefits**:
- Same problem can be solved with different estimators
- Immutable data prevents accidental modification
- Type stability for performance
- Extensible via multiple dispatch

---

## Estimator Selection

**Decision Tree**:

```
Is this a randomized experiment (RCT)?
│
├─ YES (RCT):
│  │
│  ├─ Have baseline covariates?
│  │  ├─ YES: Use RegressionATE (ANCOVA for variance reduction)
│  │  └─ NO: ──> Was it block-randomized?
│  │             ├─ YES: Use StratifiedATE
│  │             └─ NO: Use SimpleATE
│  │
│  └─ Small sample (n < 20)?
│     └─ Consider PermutationTest (distribution-free)
│
└─ NO (Observational Study):
   │
   └─ Have confounders measured?
      ├─ YES: Use NearestNeighborPSM
      └─ NO: ⚠️ Cannot estimate causal effect (omitted variable bias)
```

---

## Cross-Language Validation

**Golden Reference Tests**: 46 tests comparing Julia vs Python implementations

- **Tolerance**: rtol < 1e-10 (matches Python to 10 decimal places)
- **Estimators validated**: SimpleATE, StratifiedATE, RegressionATE, PermutationTest, IPWATE
- **Python libraries**: `statsmodels`, `scipy.stats`
- **All tests passing**: 46/46 (100%)

---

## Contributing

### Reporting Issues

Open an issue with:
1. Minimal reproducible example (MWE)
2. Expected vs actual behavior
3. Julia version and `Pkg.status("CausalEstimators")`

### Adding New Estimators

Follow the SciML pattern:
1. Define abstract types (if new method family)
2. Create problem and estimator structs
3. Implement `solve(problem, estimator)`
4. Add comprehensive tests (unit + integration + golden reference)
5. Document in phase completion summary

**Example structure**:
```julia
# Define types
struct MyProblem{T,P} <: AbstractCausalProblem{T,P}
    # fields...
end

struct MyEstimator <: AbstractCausalEstimator end

# Implement solve
function solve(problem::MyProblem, estimator::MyEstimator)
    # estimation logic...
    return MySolution(...)
end
```

### Code Style

- **SciML formatting**: 92-char lines, 4-space indents
- **Never fail silently**: Explicit errors with diagnostic messages
- **Immutability by default**: Return new data unless marked `!`
- **Type annotations**: All parameters and returns
- **Comprehensive docstrings**: With examples and references

---

## References

### Textbooks

- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.
- Rosenbaum, P. R. (2017). *Observation and Experiment: An Introduction to Causal Inference*. Harvard University Press.
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.

### Key Papers

**Propensity Score Matching**:
- Abadie, A., & Imbens, G. W. (2006). "Large Sample Properties of Matching Estimators for Average Treatment Effects." *Econometrica*, 74(1), 235-267.
- Abadie, A., & Imbens, G. W. (2008). "On the Failure of the Bootstrap for Matching Estimators." *Econometrica*, 76(6), 1537-1557.
- Austin, P. C. (2009). "Balance diagnostics for comparing the distribution of baseline covariates between treatment groups in propensity-score matched samples." *Statistics in Medicine*, 28(25), 3083-3107.

**SciML Ecosystem**:
- Rackauckas, C., & Nie, Q. (2017). "DifferentialEquations.jl–a performant and feature-rich ecosystem for solving differential equations in Julia." *Journal of Open Research Software*, 5(1).
- SciML Documentation: https://docs.sciml.ai/

---

## License

MIT License

---

## Acknowledgments

Built following [SciML](https://sciml.ai/) design patterns for scientific machine learning in Julia.

**Status**: Phases 0-2 complete (460/460 tests passing) | **Version**: 0.1.0
