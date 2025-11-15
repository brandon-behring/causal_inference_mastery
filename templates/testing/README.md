# Test Infrastructure Templates

**Purpose**: Reduce boilerplate when implementing the six-layer validation architecture for new phases.

**Created**: 2024-11-14
**Based on**: Phase 1 RCT Foundation (Grade: A+, 98/100)

---

## Available Templates

### 1. Known-Answer Test Template
**File**: `known_answer_test_template.jl`
**Purpose**: Test estimators against analytically verifiable scenarios

**What it provides**:
- 4 standard known-answer scenarios (zero effect, constant effect, zero variance, large sample)
- Template structure for adding estimator-specific scenarios
- Usage instructions and philosophy

**When to use**: First layer of validation for any new estimator

**Time savings**: ~30 minutes per estimator

---

### 2. Adversarial Test Template
**File**: `adversarial_test_template.jl`
**Purpose**: Test estimators against edge cases, invalid inputs, boundary conditions

**What it provides**:
- 12 standard adversarial scenarios:
  - Edge cases (n=1, n=2, all treated/control)
  - Numerical stability (NaN, Inf, extreme values)
  - Invalid inputs (mismatched lengths, empty arrays)
  - Boundary conditions (zero variance)
- Estimator-specific adversarial test guide
- Adversarial testing philosophy

**When to use**: Second layer of validation for any new estimator

**Time savings**: ~45 minutes per estimator (10+ adversarial tests)

**Target**: Minimum 10 adversarial tests per estimator

---

### 3. Monte Carlo Validation Template
**File**: `monte_carlo_validation_template.jl`
**Purpose**: Verify statistical properties (bias, coverage, SE accuracy) with known ground truth

**What it provides**:
- DGP with known treatment effect (`dgp_constant_ate`)
- Monte Carlo validation function (`validate_monte_carlo`)
- 6 standard Monte Carlo tests:
  - Unbiasedness (bias < 0.05)
  - Coverage (94-96% for α=0.05)
  - SE accuracy (< 10% error)
  - Large sample consistency
  - Heteroskedasticity robustness
  - Multi-alpha coverage (0.01, 0.05, 0.10)
- Monte Carlo validation philosophy

**When to use**: Third layer of validation for any new estimator

**Time savings**: ~1 hour per estimator

**Target**: All estimators must pass bias, coverage, and SE accuracy tests

---

## How to Use Templates

### Step 1: Copy Template to Test Directory

```bash
# Example: Creating tests for new DiD estimator
cp templates/testing/known_answer_test_template.jl \
   test/estimators/did/test_twfe_known_answer.jl
```

### Step 2: Find and Replace Placeholders

Replace these placeholders in the template:
- `[EstimatorName]` → Your estimator name (e.g., `TWFE`, `SimpleATE`)
- `[ProblemType]` → Your problem type (e.g., `DiDProblem`, `RCTProblem`)
- `[Module]` → Your module name (e.g., `DiD`, `RCT`)

**Example**:
```julia
# Before (template)
using CausalInference.[Module]
problem = [ProblemType](outcomes, treatment, nothing, nothing, (alpha=0.05,))
solution = solve(problem, [EstimatorName]())

# After (customized for DiD)
using CausalInference.DiD
problem = DiDProblem(outcomes, treatment, time_periods, (alpha=0.05,))
solution = solve(problem, TWFE())
```

### Step 3: Add Estimator-Specific Tests

Each template has comments like:
```julia
# ========================================================================
# ESTIMATOR-SPECIFIC TESTS
# ========================================================================
# Examples:
# - For IPWATE: Propensity scores near 0/1
# - For StratifiedATE: Stratum with all treated units
```

Add tests specific to your estimator's assumptions and failure modes.

### Step 4: Verify Tests Work

```bash
# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run specific test file
julia --project=. test/estimators/did/test_twfe_known_answer.jl
```

---

## Six-Layer Validation Architecture

These templates cover layers 1-3. Complete architecture:

1. ✅ **Known-Answer Tests** → `known_answer_test_template.jl`
2. ✅ **Adversarial Tests** → `adversarial_test_template.jl`
3. ✅ **Monte Carlo Validation** → `monte_carlo_validation_template.jl`
4. ⚠️ **PyCall Cross-Validation** → See Phase 1 `test/validation/test_python_parity.jl` for example
5. ⚠️ **R Triangulation** → See Phase 1 `validation/r_scripts/validate_rct.R` for example
6. ⚠️ **Golden Reference Tests** → Dataset-specific, implement per phase

Layers 4-6 are method-specific and harder to templatize.

---

## Quality Standards

All tests must meet these standards (from `PHASE_COMPLETION_STANDARDS.md`):

### Known-Answer Tests
- ✅ At least 2 tests per estimator
- ✅ Analytically verifiable results
- ✅ Cover zero effect, constant effect scenarios
- ✅ Deterministic (use `Random.seed!()`)

### Adversarial Tests
- ✅ Minimum 10 tests per estimator
- ✅ Cover edge cases, numerical stability, invalid inputs, boundary conditions
- ✅ Test both error cases (should throw) and graceful degradation (should work)
- ✅ Descriptive test names (not `test1`, `test2`)

### Monte Carlo Validation
- ✅ DGP with KNOWN treatment effect
- ✅ N=10,000 simulations for reliable estimates
- ✅ Bias < 0.05
- ✅ Coverage 94-96% for α=0.05
- ✅ SE accuracy < 10%
- ✅ Multi-alpha coverage (0.01, 0.05, 0.10)

---

## Time Estimates

Using templates vs writing from scratch:

| Layer | From Scratch | With Template | Savings |
|-------|--------------|---------------|---------|
| Known-Answer | 1.0 hours | 0.5 hours | 0.5 hours |
| Adversarial | 1.5 hours | 0.75 hours | 0.75 hours |
| Monte Carlo | 2.0 hours | 1.0 hours | 1.0 hours |
| **Total** | **4.5 hours** | **2.25 hours** | **2.25 hours** |

**Per estimator savings**: ~2.25 hours
**Per phase savings** (assuming 3-5 estimators): ~7-11 hours

---

## Examples from Phase 1

See Phase 1 test files for complete examples:

**Known-Answer Tests**:
- `test/estimators/rct/test_simple_ate.jl:10-50` - Zero effect, constant effect
- `test/estimators/rct/test_ipw_ate.jl:15-60` - Balanced propensity, extreme propensity

**Adversarial Tests**:
- `test/estimators/rct/test_simple_ate.jl:150-300` - 16 adversarial tests
- `test/estimators/rct/test_stratified_ate.jl:200-450` - 20 adversarial tests

**Monte Carlo Validation**:
- `test/validation/test_monte_carlo_ground_truth.jl` - Complete Monte Carlo suite for all 5 RCT estimators

---

## Customization Guide

### For Different Methods

**RCT estimators**: Templates work out-of-the-box

**DiD estimators**:
- Add time period dimension to DGP
- Test parallel trends assumption
- Add adversarial test for single time period

**IV estimators**:
- Add instruments to DGP
- Test weak instrument scenarios (F < 10)
- Add Monte Carlo test for varying instrument strength

**RDD estimators**:
- Add running variable to DGP
- Test manipulation with McCrary test
- Add adversarial test for all units on one side of cutoff

**CATE estimators**:
- Generate heterogeneous treatment effects in DGP
- Test recovery of known τ(x) function
- Add adversarial test for no heterogeneity (CATE = ATE)

### For Different Languages

**Python**: Adapt templates to pytest format:
- `@testset "Name" begin ... end` → `def test_name():`
- `@test` → `assert`
- `@test_throws` → `with pytest.raises(ValueError):`

**R**: Adapt templates to testthat format:
- `@testset "Name" begin ... end` → `test_that("Name", { ... })`
- `@test` → `expect_true()`, `expect_equal()`
- `@test_throws` → `expect_error()`

---

## Maintenance

**When to update templates**:
- New adversarial test categories discovered (add to template)
- Better Monte Carlo DGPs developed (add as alternative)
- New validation layers added (create new template)

**How to update**:
1. Edit template in `templates/testing/`
2. Document change in this README
3. Update `PHASE_COMPLETION_STANDARDS.md` if standards change

---

## References

- **Phase 1 RCT Foundation**: Complete working examples
- **PHASE_COMPLETION_STANDARDS.md**: Mandatory validation requirements
- **METHODOLOGICAL_CONCERNS.md**: Phase-specific concerns to test for

---

**Last Updated**: 2024-11-14
**Next Review**: After Phase 2 completion
