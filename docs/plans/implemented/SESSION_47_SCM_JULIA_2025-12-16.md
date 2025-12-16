# Session 47: Julia Synthetic Control Methods

**Date**: 2025-12-16
**Duration**: ~5 hours
**Status**: ✅ COMPLETE

---

## Objective

Port Python SCM (Session 46) to Julia following SciML Problem-Estimator-Solution pattern for cross-language validation.

---

## Deliverables

### Module Structure

```
julia/src/scm/
├── types.jl              # SCMProblem, SCMSolution, estimator types (~220 lines)
├── weights.jl            # Simplex-constrained optimization (~120 lines)
├── synthetic_control.jl  # solve(::SCMProblem, ::SyntheticControl) (~180 lines)
├── inference.jl          # Placebo tests, bootstrap SE (~160 lines)
└── augmented_scm.jl      # solve(::SCMProblem, ::AugmentedSC) (~140 lines)

julia/test/scm/
├── runtests.jl           # SafeTestsets runner
├── test_types.jl         # Problem/Solution construction tests
├── test_synthetic_control.jl  # Known-answer + adversarial
├── test_inference.jl     # Placebo test validation
└── test_augmented.jl     # ASCM tests

tests/validation/cross_language/
└── test_python_julia_scm.py  # Python↔Julia parity tests
```

### Type Hierarchy

```
AbstractCausalProblem{T,P}
  └─ AbstractSCMProblem{T,P}
       └─ SCMProblem{T,P}

AbstractCausalEstimator
  └─ AbstractSCMEstimator
       ├─ SyntheticControl
       └─ AugmentedSC

AbstractCausalSolution
  └─ AbstractSCMSolution
       └─ SCMSolution{T,P}
```

### Key Types

```julia
struct SCMProblem{T<:Real,P} <: AbstractSCMProblem{T,P}
    outcomes::Matrix{T}           # (n_units, n_periods)
    treatment::Vector{Bool}       # Which units treated
    treatment_period::Int         # When treatment starts (1-indexed)
    covariates::Union{Nothing,Matrix{T}}
    parameters::P
end

struct SyntheticControl <: AbstractSCMEstimator
    inference::Symbol             # :placebo, :bootstrap, :none
    n_placebo::Int
    covariate_weight::Float64
end

struct AugmentedSC <: AbstractSCMEstimator
    inference::Symbol             # :jackknife, :none
    lambda::Union{Nothing,Float64}  # Ridge regularization
end

struct SCMSolution{T<:Real,P} <: AbstractSCMSolution
    estimate::T                   # ATT
    se::Union{Nothing,T}
    ci_lower::T
    ci_upper::T
    p_value::Union{Nothing,T}
    weights::Vector{T}
    pre_rmse::T
    pre_r_squared::T
    n_treated::Int
    n_control::Int
    n_pre_periods::Int
    n_post_periods::Int
    synthetic_control::Vector{T}
    treated_series::Vector{T}
    gap::Vector{T}
    retcode::Symbol
    original_problem::SCMProblem{T,P}
end
```

---

## Test Results

| File | Tests | Status |
|------|-------|--------|
| `test_types.jl` | 26 | ✅ Pass |
| `test_synthetic_control.jl` | 31 | ✅ Pass |
| `test_inference.jl` | 17 | ✅ Pass |
| `test_augmented.jl` | 26 | ✅ Pass |
| **Total Julia** | **100** | ✅ Pass |

### Cross-Language Tests

| Test Class | Tests | Status |
|------------|-------|--------|
| `TestSyntheticControlParity` | 5 | ✅ Pass |
| `TestAugmentedSCMParity` | 3 | ✅ Pass |
| `TestSCMIntegration` | 2 | ✅ Pass |
| **Total** | **10** | ✅ Pass |

---

## Example Usage

```julia
using CausalEstimators

# Panel data: 10 units × 20 periods
outcomes = randn(10, 20) .+ 10
treatment = Vector{Bool}([true; falses(9)])
treatment_period = 11  # Julia 1-indexed

# Create problem
problem = SCMProblem(outcomes, treatment, treatment_period)

# Solve with SyntheticControl
solution = solve(problem, SyntheticControl(inference=:placebo, n_placebo=50))
println("ATT: $(solution.estimate) (p=$(solution.p_value))")
println("Pre-RMSE: $(solution.pre_rmse)")
println("Weights: $(solution.weights)")

# Augmented SCM
ascm = solve(problem, AugmentedSC(inference=:jackknife))
println("ASCM ATT: $(ascm.estimate)")
```

---

## Cross-Language Validation

### Python Interface Wrappers

Added to `tests/validation/cross_language/julia_interface.py`:

```python
def julia_synthetic_control(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    treatment_period: int,  # 1-indexed for Julia
    covariates: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    inference: str = "placebo",
    n_placebo: int = 100,
    covariate_weight: float = 1.0,
) -> Dict[str, Union[float, int, np.ndarray, str, None]]:
    ...

def julia_augmented_scm(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    treatment_period: int,  # 1-indexed for Julia
    covariates: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    inference: str = "jackknife",
    lambda_ridge: Optional[float] = None,
) -> Dict[str, Union[float, int, np.ndarray, str, None]]:
    ...
```

### Parity Tolerances

| Metric | Tolerance | Rationale |
|--------|-----------|-----------|
| ATE | rtol=0.10 | Same optimization, minor numerics |
| SE | rtol=0.20 | Placebo variation |
| Weights correlation | r > 0.90 | Same objective, solver differences |
| Pre-RMSE | rtol=0.20 | Should match closely |

---

## Key Fixes

### 1. BitVector → Vector{Bool}

**Problem**: `falses(n)` returns `BitVector`, constructor expected `Vector{Bool}`.

**Fix**: Explicit conversion in tests:
```julia
treatment = Vector{Bool}(falses(10))
treatment[1] = true
```

### 2. Known-Answer DGP

**Problem**: Constant time series caused optimization issues, pre-R² = 0.

**Fix**: Added realistic time trends and noise:
```julia
# Before (failed)
control_outcomes[i, :] .= 10.0 + i

# After (works)
control_outcomes[i, :] .= 10.0 .+ i .+ base_trend .+ 0.1 .* randn(n_periods)
```

### 3. Python `lambda` Keyword

**Problem**: `lambda` is reserved in Python, can't use as kwarg.

**Fix**: Used `jl.seval()` to construct estimator:
```python
if lambda_ridge is not None:
    estimator = jl.seval(f"AugmentedSC(inference=:{inference}, lambda={lambda_ridge})")
else:
    estimator = jl.seval(f"AugmentedSC(inference=:{inference})")
```

---

## Module Updates

### CausalEstimators.jl Exports

```julia
# SCM Types
export SCMProblem
export SyntheticControl, AugmentedSC
export SCMSolution
export compute_scm_weights, compute_pre_treatment_fit
```

### Test Runner Update

```julia
# julia/test/runtests.jl
@safetestset "SCM Estimators" begin
    include("scm/runtests.jl")
end
```

---

## Performance Notes

- Weight optimization: ~0.5ms for 10 units × 20 periods
- Placebo inference: ~50ms for 50 placebos
- Jackknife SE: ~20ms for 10-unit leave-one-out

---

## Next Steps

- Session 48: Documentation update (consolidate Sessions 38-47)
- Future: SCM Monte Carlo validation
- Future: SCM sensitivity analysis integration

---

## Commits

```
(pending commit after Session 47 completion)
```
