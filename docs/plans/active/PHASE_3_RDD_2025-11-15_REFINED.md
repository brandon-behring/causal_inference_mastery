# Phase 3: Regression Discontinuity Design (RDD) - Refined Implementation Plan

**Created**: 2025-11-15
**Status**: NOT_STARTED
**Estimated Duration**: 28-32 hours
**Target Completion**: Week 3-4
**Dependencies**: Phases 0-2 complete

---

## Executive Summary

Implement Regression Discontinuity Design (RDD) estimators with optimal bandwidth selection, manipulation testing, and comprehensive validation matching Phase 1-2 A+ quality standards.

**Key Focus Areas**:
1. **Sharp RDD** - Deterministic treatment assignment at cutoff (primary focus)
2. **Fuzzy RDD** - Probabilistic treatment assignment (secondary, if time permits)
3. **Bandwidth Selection** - IK (2012) and CCT (2014) methods
4. **Manipulation Testing** - McCrary density test (2008) for validity
5. **Sensitivity Analysis** - Bandwidth robustness and donut-hole tests
6. **Cross-Language Validation** - Compare to R's `rdrobust` package (Calonico et al.)

**Methodological Foundation**:
- Local polynomial regression (linear and quadratic)
- MSE-optimal vs coverage error-optimal bandwidth
- Robust bias-corrected confidence intervals (CCT 2014)
- Falsification tests at placebo cutoffs

---

## Research Summary: What is RDD?

### Core Concept

**Regression Discontinuity Design** exploits sharp thresholds in treatment assignment to estimate causal effects.

**Example**: Students with test scores ≥ 70 receive tutoring (treated), < 70 do not (control).
- **Assumption**: At cutoff (score = 70), students are nearly identical
- **Effect**: Difference in outcomes at cutoff = causal effect of tutoring
- **Challenge**: Estimate effect locally around cutoff, not globally

### Sharp vs Fuzzy RDD

**Sharp RDD**:
- Treatment probability jumps from 0 to 1 at cutoff
- P(T=1 | X=c⁺) = 1, P(T=1 | X=c⁻) = 0
- Effect = lim[E(Y|X=c⁺) - E(Y|X=c⁻)]

**Fuzzy RDD**:
- Treatment probability discontinuous but not 0→1 jump
- P(T=1 | X=c⁺) > P(T=1 | X=c⁻), but neither is 0 or 1
- Effect = [lim E(Y|X=c⁺) - lim E(Y|X=c⁻)] / [lim E(T|X=c⁺) - lim E(T|X=c⁻)]
- Essentially IV design (running variable as instrument)

**Phase 3 Scope**: Sharp RDD primary, Fuzzy RDD if time permits (move to Phase 3B if needed)

### Local Polynomial Regression

**Goal**: Estimate E(Y|X) locally around cutoff using weighted regression

**Local Linear** (recommended, p=1):
```
minimize: Σ Kₕ(Xᵢ - c) · [Yᵢ - α - β(Xᵢ - c)]²
```
- α = outcome at cutoff
- β = slope
- Kₕ = kernel function with bandwidth h

**Local Quadratic** (p=2):
```
minimize: Σ Kₕ(Xᵢ - c) · [Yᵢ - α - β₁(Xᵢ - c) - β₂(Xᵢ - c)²]²
```

**Kernel Functions**:
- **Triangular**: K(u) = (1 - |u|) · 1{|u| ≤ 1} (CCT default, optimal MSE)
- **Epanechnikov**: K(u) = 0.75(1 - u²) · 1{|u| ≤ 1}
- **Uniform**: K(u) = 0.5 · 1{|u| ≤ 1} (simple, less efficient)

**Phase 3 Scope**: Local linear primary, triangular kernel (matches `rdrobust`)

### Bandwidth Selection Methods

**Imbens-Kalyanaraman (IK, 2012)**:
- MSE-optimal bandwidth for local linear regression
- Closed-form solution based on density, variance, curvature
- Fast, widely used
- **Issue**: Optimized for point estimation, not CI coverage

**Calonico-Cattaneo-Titiunik (CCT, 2014)**:
- Coverage error-optimal bandwidth for bias-corrected CIs
- Accounts for bias in CI construction
- Two bandwidths: main (h) and bias estimation (b)
- **Advantage**: Valid inference with correct coverage
- **Default in `rdrobust`**

**Phase 3 Implementation**: Both IK and CCT, document differences

### Manipulation Testing

**McCrary Density Test (2008)**:
- Tests for discontinuity in density of running variable at cutoff
- **Null**: f(X|X=c⁺) = f(X|X=c⁻) (no manipulation)
- **Alternative**: Discontinuity in density (manipulation detected)
- **Method**: Local linear density estimation on each side, compare difference

**Why it matters**:
- If units can precisely manipulate X to cross cutoff → selection bias
- Example: Students retaking test to get ≥ 70
- RDD invalid if manipulation detected

**Phase 3 Implementation**: Implement McCrary test, warn if p < 0.05

### Sensitivity Analysis

**Bandwidth Sensitivity**:
- Report results at h, 0.5h, 2h
- Plot RDD estimate vs bandwidth
- Should be relatively stable

**Donut-Hole Tests**:
- Drop observations very close to cutoff (e.g., |X - c| < δ)
- If manipulation only affects units at exact cutoff, estimates should be stable
- Large change → manipulation concern

**Placebo Cutoffs**:
- Run RDD at fake cutoffs (c ± offset)
- Should find NO effect (falsification test)
- Effect at placebo → violation of continuity assumption

**Phase 3 Implementation**: All three sensitivity tests

---

## Current State

**Phase 2 Status**: COMPLETE (assumed for planning)
- 4 PSM estimators implemented
- Abadie-Imbens SE for with-replacement matching
- Balance diagnostics comprehensive (SMD < 0.1 on ALL covariates)
- Six-layer validation passing

**Phase 3 Status**: Not started
- No RDD estimators implemented
- No bandwidth selection infrastructure
- No manipulation testing
- No Lee (2008) dataset loaded

---

## Target State

**Deliverables** (Julia + Python):

1. **2 RDD Estimators** (Sharp RDD primary focus):
   - Sharp RDD with local linear regression
   - Sharp RDD with local quadratic regression (optional, time permitting)
   
2. **2 Bandwidth Selection Methods**:
   - Imbens-Kalyanaraman (IK 2012) - MSE-optimal
   - Calonico-Cattaneo-Titiunik (CCT 2014) - coverage error-optimal

3. **Kernel Functions**:
   - Triangular kernel (primary, CCT default)
   - Uniform kernel (comparison)

4. **Manipulation Testing**:
   - McCrary density test (2008)
   - Visual density plot around cutoff
   - Automatic warnings if p < 0.05

5. **Sensitivity Analysis Suite**:
   - Bandwidth sensitivity (h, 0.5h, 2h)
   - Donut-hole tests (drop |X - c| < δ)
   - Placebo cutoff tests (c ± offset)

6. **Validation**: All 6 layers passing
   - Known-answer tests (2+ per estimator)
   - Adversarial tests (10+ per estimator, 20+ total)
   - Monte Carlo validation (bias < 0.05, coverage 94-96%)
   - Python-Julia cross-validation (rtol < 1e-10)
   - R triangulation (vs `rdrobust` package)
   - Golden reference (Lee 2008 or simulated)

7. **Documentation**: 8-section docstrings for all estimators with bandwidth method justification

**Expected File Count**: ~22 files created
- 2 Julia RDD estimators (sharp_rdd_linear.jl, sharp_rdd_quadratic.jl)
- 2 Python RDD estimators
- 2 Bandwidth selection modules (ik_bandwidth.jl, cct_bandwidth.jl)
- 2 Kernel functions module
- 1 McCrary test module
- 1 Sensitivity analysis module
- 4 Julia unit test files
- 3 Validation test files (Monte Carlo, Python parity, R validation)
- 1 R validation script
- Documentation files
- Example notebooks

---

## Detailed Implementation Plan

### Phase 3.1: Foundation & Problem Structure (4-5 hours)

**Objective**: Create RDD problem type and basic infrastructure following SciML pattern

#### Tasks:

- [ ] **3.1.1**: Review methodological concerns
  - McCrary density test (METHODOLOGICAL_CONCERNS.md #8)
  - Bandwidth sensitivity (METHODOLOGICAL_CONCERNS.md #9)
  - Research rdrobust package behavior

- [ ] **3.1.2**: Create RDD problem type (`RDDProblem`)
  - Fields: outcomes (Y), running_variable (X), cutoff (c), treatment (T, for fuzzy)
  - Validation: X not missing, sufficient observations near cutoff, cutoff within X range
  - SciML pattern compliance: similar to PSMProblem from Phase 2

- [ ] **3.1.3**: Implement kernel functions (`src/estimators/rdd/kernels.jl`)
  - Triangular kernel (CCT default)
  - Uniform kernel (comparison)
  - Epanechnikov kernel (optional, for reference)
  - Unit tests: kernel integrates to 1, non-negative, symmetric

- [ ] **3.1.4**: Known-answer tests for kernels
  - Triangular at u=0 → K=1, at u=1 → K=0
  - Uniform at |u|<1 → K=0.5, else K=0
  - Integral of kernel = 1

**Expected Output**: RDD infrastructure ready, kernel functions tested

**Time Estimate**: 4-5 hours

---

### Phase 3.2: IK Bandwidth Selection (5-6 hours)

**Objective**: Implement Imbens-Kalyanaraman MSE-optimal bandwidth

#### Tasks:

- [ ] **3.2.1**: Implement IK bandwidth selector (`src/estimators/rdd/ik_bandwidth.jl`)
  - Estimate density at cutoff (f(c))
  - Estimate variance on each side (σ²₊, σ²₋)
  - Estimate second derivative of CEF (m''(c))
  - Closed-form h_IK formula from IK (2012) paper

- [ ] **3.2.2**: Known-answer tests for IK
  - Uniform density, constant variance → analytic h_IK
  - Very smooth outcome (m''(c) ≈ 0) → larger bandwidth
  - High curvature (large m''(c)) → smaller bandwidth
  - High variance → larger bandwidth

- [ ] **3.2.3**: Adversarial tests for IK (10+)
  - Running variable with single unique value (no variation)
  - All X on one side of cutoff (no common support)
  - Very sparse data near cutoff (n < 10 in window)
  - Extreme outliers in Y near cutoff
  - Perfect separation (all Y=1 above cutoff, Y=0 below)
  - Zero variance in outcomes
  - Cutoff at min(X) or max(X) boundary
  - Non-monotonic relationship (U-shaped)
  - Bunching at cutoff (manipulation signal)
  - Very large sample (n > 10,000, ensure numerical stability)

- [ ] **3.2.4**: Document IK limitations
  - Optimized for point estimation MSE, NOT CI coverage
  - May produce CIs with under-coverage (< 95%)
  - Use CCT for valid inference

**Expected Output**: IK bandwidth selector working with 10+ adversarial tests

**Time Estimate**: 5-6 hours

---

### Phase 3.3: Sharp RDD with Local Linear Regression (6-7 hours)

**Objective**: Implement sharp RDD estimator with local linear regression

#### Tasks:

- [ ] **3.3.1**: Implement sharp RDD estimator (`src/estimators/rdd/sharp_rdd_linear.jl`)
  - Local linear regression on each side of cutoff
  - Weighted by kernel K_h(X_i - c)
  - Treatment effect = α₊ - α₋ (intercepts at cutoff)
  - Standard errors: conventional (biased) and robust bias-corrected (CCT 2014)

- [ ] **3.3.2**: Implement conventional SE
  - Asymptotic variance formula for local polynomial
  - Assumes correct bandwidth choice
  - Known to produce under-coverage

- [ ] **3.3.3**: Implement robust bias-corrected SE (CCT 2014)
  - Bias estimation bandwidth (b > h)
  - Bias correction term
  - Robust variance with bias uncertainty
  - Valid CIs with correct coverage

- [ ] **3.3.4**: Known-answer tests (2+)
  - Linear DGP with discontinuity: Y = X + τ·1{X ≥ c}
    - True effect = τ
    - Local linear should recover exactly (zero bias)
  - Quadratic DGP: Y = X² + τ·1{X ≥ c}
    - Local linear has bias (linear approximation to quadratic)
    - Document bias, verify SE captures uncertainty

- [ ] **3.3.5**: Adversarial tests for Sharp RDD (10+)
  - n=1 (insufficient data)
  - All X on one side of cutoff
  - Running variable exactly at cutoff for all units
  - Zero variance in Y
  - Perfect separation (all treated Y=100, all control Y=0)
  - Extreme outcome at cutoff (Y=1e10 at X=c)
  - Bandwidth larger than range(X) (uses all data)
  - Bandwidth approaching zero (uses no data)
  - Non-monotonic outcome (U-shaped around cutoff)
  - Sparse data (only 2-3 observations within bandwidth)

- [ ] **3.3.6**: Integration with IK bandwidth
  - Default: use IK bandwidth
  - Allow manual bandwidth override
  - Report both conventional and robust SE

**Expected Output**: Sharp RDD estimator with IK bandwidth, conventional + robust SE, 10+ adversarial tests

**Time Estimate**: 6-7 hours

---

### Phase 3.4: CCT Bandwidth & Robust Inference (5-6 hours)

**Objective**: Implement CCT coverage error-optimal bandwidth and robust bias-corrected CIs

#### Tasks:

- [ ] **3.4.1**: Implement CCT bandwidth selector (`src/estimators/rdd/cct_bandwidth.jl`)
  - Two bandwidths: main (h) and bias estimation (b)
  - Coverage error-optimal for bias-corrected CIs
  - More conservative than IK (often h_CCT < h_IK)

- [ ] **3.4.2**: Integrate CCT bandwidth with Sharp RDD
  - Use h_CCT for main regression
  - Use b for bias estimation (typically b = 1.5h or 2h)
  - Bias-corrected point estimate
  - Robust variance accounting for bias uncertainty

- [ ] **3.4.3**: Known-answer tests
  - Linear DGP → CCT and IK should agree (zero bias)
  - Quadratic DGP → CCT provides valid CIs, conventional SE may under-cover

- [ ] **3.4.4**: Compare IK vs CCT bandwidth
  - Same data → typically h_CCT ≤ h_IK
  - Document trade-off: IK (smaller MSE) vs CCT (correct coverage)
  - Recommend CCT for inference, IK for exploration

**Expected Output**: CCT bandwidth working, integrated with Sharp RDD, comparison documented

**Time Estimate**: 5-6 hours

---

### Phase 3.5: McCrary Density Test (4-5 hours)

**Objective**: Implement manipulation testing via McCrary density test

#### Tasks:

- [ ] **3.5.1**: Implement McCrary test (`src/estimators/rdd/mccrary_test.jl`)
  - Local linear density estimation on each side
  - θ = log(f(c⁺)) - log(f(c⁻))
  - Standard error via binning method
  - p-value from normal approximation

- [ ] **3.5.2**: Visual density diagnostic
  - Histogram of running variable
  - Smooth density estimate overlaid
  - Vertical line at cutoff
  - Flag if discontinuity visible

- [ ] **3.5.3**: Known-answer tests
  - Uniform density → θ ≈ 0, p > 0.05
  - Bunching at cutoff (double density above) → θ > 0, p < 0.05
  - Avoidance of cutoff (half density above) → θ < 0, p < 0.05

- [ ] **3.5.4**: Adversarial tests (5+)
  - Very sparse data (n < 20)
  - Extremely lumpy density (many modes)
  - All X on one side (no comparison)
  - Discrete X (only integer values)
  - Heavy tails in X distribution

- [ ] **3.5.5**: Integration with RDD estimators
  - Automatic McCrary test when running RDD
  - Warning if p < 0.05: "Potential manipulation detected (p=X.XX). RDD may be invalid."
  - Option to suppress warning if known safe

**Expected Output**: McCrary test implemented, integrated with RDD, 5+ adversarial tests

**Time Estimate**: 4-5 hours

---

### Phase 3.6: Sensitivity Analysis Suite (4-5 hours)

**Objective**: Bandwidth sensitivity, donut-hole, and placebo cutoff tests

#### Tasks:

- [ ] **3.6.1**: Bandwidth sensitivity analysis
  - Compute RDD at h ∈ {0.5h*, h*, 2h*} where h* = optimal
  - Plot estimate vs bandwidth (h from 0.1h* to 5h*)
  - Report standard deviation of estimates across bandwidths
  - Flag if SD > 0.5 × estimate (highly sensitive)

- [ ] **3.6.2**: Donut-hole tests
  - Drop observations with |X - c| < δ for δ ∈ {0.01×range(X), 0.05×range(X)}
  - Re-estimate RDD
  - Compare to baseline (no donut)
  - Large change → manipulation at exact cutoff

- [ ] **3.6.3**: Placebo cutoff tests
  - Run RDD at c_placebo ∈ {c - 0.5h*, c + 0.5h*}
  - Should find no effect (p > 0.05)
  - Significant effect → violation of continuity assumption

- [ ] **3.6.4**: Known-answer tests for sensitivity
  - True discontinuity only at c → placebo tests fail to reject
  - Discontinuity at multiple points → placebo tests reject
  - Manipulation only at c → donut-hole changes estimate

- [ ] **3.6.5**: Sensitivity report generation
  - Automated report with all 3 tests
  - Traffic light system: GREEN (all pass), YELLOW (1 concern), RED (2+ concerns)
  - Interpretive guidance

**Expected Output**: Comprehensive sensitivity analysis suite, automated reporting

**Time Estimate**: 4-5 hours

---

### Phase 3.7: Monte Carlo Validation (5-6 hours)

**Objective**: Validate statistical properties with known ground truth

#### Tasks:

- [ ] **3.7.1**: Create DGP with sharp discontinuity
  - Running variable: X ~ Uniform(c - 1, c + 1)
  - Outcome: Y = X + τ·1{X ≥ c} + ε, ε ~ N(0, σ²)
  - True effect: τ = 2.0 (known)
  - No manipulation

- [ ] **3.7.2**: Validate bias < 0.05
  - Run 10,000 simulations
  - Verify mean(estimates) - τ < 0.05
  - Test for BOTH IK and CCT bandwidth

- [ ] **3.7.3**: Validate coverage 94-96%
  - Multi-alpha coverage: α ∈ {0.01, 0.05, 0.10} (Phase 1 gap filled)
  - Conventional SE: expect UNDER-coverage (< 94%)
  - Robust SE (CCT): expect CORRECT coverage (94-96%)
  - Document conventional SE failure

- [ ] **3.7.4**: Validate SE accuracy < 10%
  - Compare mean(robust SE) to empirical SD(estimates)
  - Should agree within 10%

- [ ] **3.7.5**: Test with nonlinear DGP
  - Quadratic outcome: Y = X² + τ·1{X ≥ c} + ε
  - Local linear has bias (approximation error)
  - Robust SE should still cover (accounts for bias)
  - Document bias vs coverage trade-off

- [ ] **3.7.6**: Test with manipulation
  - DGP with bunching: density doubles just above cutoff
  - McCrary test should detect (p < 0.05)
  - RDD estimate biased if manipulation ignored
  - Donut-hole test should reduce bias

**Expected Output**: Monte Carlo validation showing unbiasedness, correct coverage, SE accuracy

**Time Estimate**: 5-6 hours

---

### Phase 3.8: Cross-Language & R Validation (4-5 hours)

**Objective**: Validate against Python/R implementations (R's `rdrobust` as gold standard)

#### Tasks:

- [ ] **3.8.1**: Python implementation
  - Use custom implementation (no comprehensive Python RDD library)
  - Implement same methods (Sharp RDD, IK, CCT, McCrary)
  - Serve as cross-check, not gold standard

- [ ] **3.8.2**: Python-Julia cross-validation
  - Agreement to rtol < 1e-10 for estimates
  - SE agreement within 5%
  - Bandwidth agreement within 5%

- [ ] **3.8.3**: R `rdrobust` validation (GOLD STANDARD)
  - Install `rdrobust` package (Calonico et al.)
  - Run `rdrobust()` on same data
  - Compare:
    - Point estimates (should match to rtol < 1e-8)
    - Conventional SE (should match)
    - Robust SE (should match)
    - Bandwidth h (should match CCT)
  - R is reference implementation, Julia/Python must match

- [ ] **3.8.4**: McCrary test validation
  - R package `rdd` has `DCdensity()` for McCrary test
  - Compare θ and p-value to Julia implementation
  - Should match to rtol < 1e-6

- [ ] **3.8.5**: Document discrepancies
  - If Julia/Python differ from R, investigate
  - Common reasons: different kernel, bandwidth formula, bias correction
  - Resolve before claiming validation passed

**Expected Output**: Cross-language validation passing, R `rdrobust` agreement confirmed

**Time Estimate**: 4-5 hours

---

### Phase 3.9: Adversarial Testing & Edge Cases (3-4 hours)

**Objective**: Comprehensive adversarial tests for RDD-specific failure modes

#### Tasks:

- [ ] **3.9.1**: Consolidated adversarial test suite
  - Collect all adversarial tests from sub-phases
  - Ensure 20+ total tests across RDD, IK, CCT, McCrary

- [ ] **3.9.2**: Additional RDD-specific edge cases
  - Running variable with ties (multiple units at exact cutoff)
  - Treatment assigned on wrong side of cutoff (mis-specified c)
  - Multiple discontinuities in data (only one is causal)
  - Very wide bandwidth (h > range(X), global regression)
  - Very narrow bandwidth (h → 0, no data in window)

- [ ] **3.9.3**: Numerical stability tests
  - Very large sample (n = 100,000, ensure speed)
  - Very small sample (n = 50, ensure no crash)
  - Extreme outcome values (Y ∈ [1e-10, 1e10])
  - Near-perfect collinearity in local regression

- [ ] **3.9.4**: Falsification tests
  - Covariate balance: pre-treatment covariates should NOT have discontinuity
  - Run RDD on baseline covariates (age, gender, etc.)
  - Should find no effect (p > 0.05)
  - Discontinuity in covariate → RDD invalid (sorting on observables)

**Expected Output**: 20+ total adversarial tests passing, covariate balance checks

**Time Estimate**: 3-4 hours

---

### Phase 3.10: Documentation & Benchmarking (2-3 hours)

**Objective**: Complete documentation and performance baselines

#### Tasks:

- [ ] **3.10.1**: Write comprehensive docstrings (8 sections each)
  - Mathematical foundation (local polynomial regression, continuity assumption)
  - Bandwidth selection (IK vs CCT, MSE vs coverage error optimal)
  - Variance estimation (conventional vs robust bias-corrected)
  - Usage examples
  - Requirements (no manipulation, continuity, sufficient sample)
  - Benefits (credible identification, local causal effect)
  - Limitations (only LATE at cutoff, not ATE; requires no manipulation)
  - References (IK 2012, CCT 2014, McCrary 2008, Lee 2008)

- [ ] **3.10.2**: Document bandwidth method choices
  - **WHY IK** for exploration (smaller MSE, faster)
  - **WHY CCT** for inference (correct coverage, robust CIs)
  - When to use each
  - Trade-offs documented

- [ ] **3.10.3**: Document SE choices
  - **Conventional SE**: Fast, but under-covers with bias
  - **Robust SE**: Valid inference, correct coverage
  - Always report robust SE for final results

- [ ] **3.10.4**: Establish performance benchmarks
  - Execution time for n ∈ {100, 500, 1000, 5000, 10000}
  - IK bandwidth computation time
  - CCT bandwidth computation time
  - McCrary test computation time
  - Compare Julia vs Python vs R speed
  - Document in `benchmark/README.md`

- [ ] **3.10.5**: Create usage examples
  - Simple sharp RDD example
  - Bandwidth comparison (IK vs CCT)
  - Sensitivity analysis example
  - McCrary test interpretation

**Expected Output**: All estimators fully documented, benchmarks recorded

**Time Estimate**: 2-3 hours

---

## Methodological Concerns to Address

From `METHODOLOGICAL_CONCERNS.md`:

### MEDIUM-8: McCrary Density Test Missing ⚠️
**Issue**: RDD assumes units can't manipulate running variable precisely

**Mitigation** (Phase 3.5):
- Implement McCrary test (2008) for density discontinuity
- Automatic execution with RDD estimation
- Visual density plot around cutoff
- Warn if p < 0.05: "Potential manipulation detected"
- Document when manipulation invalidates RDD

**References**:
- McCrary, J. (2008). Manipulation of the running variable in the regression discontinuity design: A density test. *Journal of Econometrics*, 142(2), 698-714.

---

### MEDIUM-9: Bandwidth Sensitivity Analysis Missing ⚠️
**Issue**: Optimal bandwidth has uncertainty, results may be sensitive

**Mitigation** (Phase 3.6):
- Report results at h, 0.5h, 2h
- Bandwidth sensitivity plot (estimate vs h)
- Flag if highly sensitive (SD > 0.5 × estimate)
- Donut-hole tests (drop |X - c| < δ)
- Placebo cutoff tests (c ± offset)

**References**:
- Imbens, G., & Kalyanaraman, K. (2012). Optimal bandwidth choice for the regression discontinuity estimator. *Review of Economic Studies*, 79(3), 933-959.

---

## Validation Requirements

From `PHASE_COMPLETION_STANDARDS.md` and `ROADMAP.md`:

### Six-Layer Validation Architecture (MANDATORY)

1. **Layer 1: Known-Answer Tests** (2+ per estimator, 4+ total)
   - Linear DGP with discontinuity (exact recovery)
   - Quadratic DGP (bias with local linear, robust SE still valid)
   - Uniform density (McCrary θ ≈ 0)
   - Bunching at cutoff (McCrary θ > 0, p < 0.05)

2. **Layer 2: Adversarial Tests** (10+ per estimator, 20+ total)
   - IK bandwidth: No variation, sparse data, outliers, boundaries, bunching
   - Sharp RDD: n=1, one-sided data, zero bandwidth, extreme outcomes, perfect separation
   - McCrary test: Sparse data, discrete X, heavy tails
   - Edge cases: Running variable at cutoff, mis-specified cutoff, ties

3. **Layer 3: Monte Carlo Validation** (10,000 sims)
   - Bias < 0.05 (IK and CCT bandwidth)
   - Coverage 94-96% for α = 0.05 (robust SE)
   - Multi-alpha coverage (α ∈ {0.01, 0.05, 0.10})
   - Conventional SE under-coverage documented
   - SE accuracy < 10% (robust SE)
   - Nonlinear DGP: bias with local linear, coverage still valid

4. **Layer 4: Python-Julia Cross-Validation**
   - Estimate agreement < 0.001
   - SE agreement < 5%
   - Bandwidth agreement < 5%

5. **Layer 5: R Triangulation (GOLD STANDARD)**
   - R `rdrobust` package (Calonico et al.)
   - Agreement to rtol < 1e-8 for estimates
   - SE agreement < 1% (robust SE)
   - Bandwidth agreement < 1% (CCT)
   - McCrary test agreement vs `DCdensity()` from `rdd` package

6. **Layer 6: Golden Reference**
   - Lee (2008) close elections dataset OR
   - Simulated dataset with known discontinuity
   - Replicate published results

### Additional Requirements

- **Test Coverage**: >80% line coverage
- **Documentation**: 8-section docstrings for all estimators
- **Benchmarks**: Performance baselines for n ∈ {100, 500, 1000, 5000, 10000}
- **Error Handling**: Explicit error messages with solutions
- **Covariate Balance Tests**: Falsification using pre-treatment variables

---

## Risk Assessment & Mitigation

### High-Risk Areas

1. **CCT Robust Variance Formula** (HIGH)
   - **Risk**: Complex bias correction, easy to implement incorrectly
   - **Mitigation**: Validate against R `rdrobust`, cross-check with CCT (2014) paper formulas

2. **McCrary Test Implementation** (MEDIUM)
   - **Risk**: Density estimation tricky, binning choices affect results
   - **Mitigation**: Validate against R `rdd::DCdensity`, compare p-values

3. **Bandwidth Selection Edge Cases** (MEDIUM)
   - **Risk**: IK/CCT formulas may fail with sparse data or extreme curvature
   - **Mitigation**: Adversarial tests for sparse data, fallback to rule-of-thumb h

4. **R Package Availability** (LOW)
   - **Risk**: `rdrobust` may not be installed
   - **Mitigation**: Graceful degradation (skip R validation if unavailable), document installation

---

## Timeline & Milestones

**Total Estimate**: 28-32 hours over 5-7 days

| Phase | Tasks | Hours | Cumulative | Milestone |
|-------|-------|-------|------------|-----------|
| 3.1 | Foundation & Problem | 4-5 | 4-5 | RDD infrastructure ready |
| 3.2 | IK Bandwidth | 5-6 | 9-11 | IK bandwidth working |
| 3.3 | Sharp RDD (Local Linear) | 6-7 | 15-18 | Sharp RDD estimator complete |
| 3.4 | CCT Bandwidth & Robust CI | 5-6 | 20-24 | Robust inference working |
| 3.5 | McCrary Test | 4-5 | 24-29 | Manipulation testing operational |
| 3.6 | Sensitivity Analysis | 4-5 | 28-34 | Bandwidth/donut/placebo tests done |
| 3.7 | Monte Carlo | 5-6 | 33-40 | Statistical validation complete |
| 3.8 | Cross-Language & R | 4-5 | 37-45 | R `rdrobust` agreement confirmed |
| 3.9 | Adversarial Tests | 3-4 | 40-49 | 20+ adversarial tests passing |
| 3.10 | Documentation & Benchmarks | 2-3 | 42-52 | Phase 3 COMPLETE |

**Buffer**: +3-5 hours for unexpected issues

**Adjusted Estimate**: 28-32 hours (reasonable for Sharp RDD focus)

---

## Success Criteria

Phase 3 is **COMPLETE** when ALL of the following are true:

### Implementation
- [ ] Sharp RDD with local linear regression implemented
- [ ] IK bandwidth selection (MSE-optimal)
- [ ] CCT bandwidth selection (coverage error-optimal)
- [ ] Conventional SE and robust bias-corrected SE (CCT 2014)
- [ ] Triangular kernel function
- [ ] McCrary density test
- [ ] Bandwidth sensitivity analysis
- [ ] Donut-hole tests
- [ ] Placebo cutoff tests
- [ ] All estimators follow SciML pattern

### Validation
- [ ] All 6 validation layers passing
- [ ] 4+ known-answer tests passing
- [ ] 20+ adversarial tests passing (10+ per major component)
- [ ] Monte Carlo: bias < 0.05, coverage 94-96% (robust SE), SE accuracy < 10%
- [ ] Multi-alpha coverage tests passing (α ∈ {0.01, 0.05, 0.10})
- [ ] Conventional SE under-coverage documented (< 94%)
- [ ] Python-Julia cross-validation passing
- [ ] R `rdrobust` triangulation passing (rtol < 1e-8)
- [ ] McCrary test validation vs R `rdd::DCdensity`
- [ ] Covariate balance falsification tests (no discontinuity in pre-treatment vars)

### Documentation
- [ ] All estimators have 8-section docstrings
- [ ] Bandwidth method choices justified (IK vs CCT, MSE vs coverage)
- [ ] SE method choices justified (conventional vs robust)
- [ ] McCrary test interpretation documented
- [ ] Sensitivity analysis interpretation documented
- [ ] ROADMAP.md updated with Phase 3 completion

### Quality
- [ ] Test coverage >80%
- [ ] All tests deterministic (seeded)
- [ ] Benchmarks recorded (n ∈ {100, 500, 1000, 5000, 10000})
- [ ] No warnings in test output
- [ ] Performance: IK bandwidth < 100ms for n=1000

### Methodological
- [ ] McCrary test automatic with RDD (warns if p < 0.05)
- [ ] Bandwidth sensitivity reported at h, 0.5h, 2h
- [ ] Robust SE recommended (conventional SE documented as under-covering)
- [ ] Covariate balance checked (falsification)

---

## Scope Management

### Phase 3 Scope (PRIMARY)

**IN SCOPE**:
- Sharp RDD with local linear regression ✅
- IK bandwidth (MSE-optimal) ✅
- CCT bandwidth (coverage error-optimal) ✅
- Conventional and robust SE ✅
- Triangular kernel ✅
- McCrary density test ✅
- Bandwidth sensitivity, donut-hole, placebo tests ✅
- R `rdrobust` validation ✅

**OUT OF SCOPE** (Move to Phase 3B if time permits):
- Fuzzy RDD (essentially IV, complex)
- Local quadratic regression (minor improvement over linear)
- Multiple running variables (geographic RDD, rare)
- Kink RDD (different design)
- Alternative kernels (Epanechnikov, Gaussian - not essential)

### Decision Rule

**If running behind schedule** (>30 hours at Phase 3.7):
- Skip local quadratic (use local linear only)
- Simplify sensitivity analysis (drop donut-hole if needed)
- Defer Fuzzy RDD to Phase 3B

**If ahead of schedule** (<25 hours at Phase 3.7):
- Add local quadratic regression
- Add Epanechnikov kernel comparison
- Consider Fuzzy RDD implementation

---

## Lessons from Phase 1-2 to Apply

1. **Test-First Development**:
   - Write known-answer tests BEFORE implementation
   - Saves debugging time, provides confidence

2. **Adversarial Tests Prevent Production Failures**:
   - Edge cases (n=1, sparse data, extreme values) found issues in Phase 1
   - Invest 10+ adversarial tests per estimator upfront

3. **Ground Truth Validation Essential**:
   - Monte Carlo with known τ catches conceptual errors
   - Cross-language alone insufficient (shared errors)

4. **R as Gold Standard**:
   - R packages (`rdrobust`, `rdd`) are reference implementations
   - Julia/Python must match R, not vice versa

5. **Document Methodological Choices**:
   - Why CCT vs IK (coverage vs MSE)
   - Why robust SE vs conventional (coverage)
   - Reviewers will ask, have answers ready

6. **Proactive Concern Identification**:
   - Review METHODOLOGICAL_CONCERNS.md before starting
   - McCrary test (#8) and sensitivity (#9) flagged in advance

---

## References

**Key Papers**:
1. **Imbens, G., & Kalyanaraman, K. (2012)**. Optimal bandwidth choice for the regression discontinuity estimator. *Review of Economic Studies*, 79(3), 933-959.
   - IK MSE-optimal bandwidth formula

2. **Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014)**. Robust nonparametric confidence intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295-2326.
   - CCT coverage error-optimal bandwidth
   - Robust bias-corrected inference

3. **McCrary, J. (2008)**. Manipulation of the running variable in the regression discontinuity design: A density test. *Journal of Econometrics*, 142(2), 698-714.
   - Density discontinuity test for manipulation

4. **Lee, D. S. (2008)**. Randomized experiments from non-random selection in U.S. House elections. *Journal of Econometrics*, 142(2), 675-697.
   - Classic RDD application, potential golden reference dataset

5. **Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020)**. *A Practical Introduction to Regression Discontinuity Designs: Foundations*. Cambridge University Press.
   - Modern RDD textbook

**Software**:
- R package `rdrobust`: https://rdpackages.github.io/rdrobust/
  - Calonico, Cattaneo, Farrell, Titiunik (2017). Rdrobust: Software for Regression-discontinuity Designs. *Stata Journal*, 17(2), 372-404.
- R package `rdd`: https://cran.r-project.org/package=rdd
  - `DCdensity()` for McCrary test

**Textbooks**:
- Imbens & Lemieux (2008). Regression discontinuity designs: A guide to practice. *Journal of Econometrics*, 142(2), 615-635.
- Angrist & Pischke (2009). *Mostly Harmless Econometrics*. Chapter 6.

---

## Status Tracking

**Plan Created**: 2025-11-15
**Implementation Started**: TBD
**Phase 3.1 Complete**: TBD
**Phase 3.2 Complete**: TBD
**Phase 3.3 Complete**: TBD
**Phase 3.4 Complete**: TBD
**Phase 3.5 Complete**: TBD
**Phase 3.6 Complete**: TBD
**Phase 3.7 Complete**: TBD
**Phase 3.8 Complete**: TBD
**Phase 3.9 Complete**: TBD
**Phase 3.10 Complete**: TBD
**Phase 3 COMPLETE**: TBD

**Next Update**: After Phase 3.1 completion

---

## Decision Log

### 2025-11-15: Focus on Sharp RDD, Defer Fuzzy RDD
**Decision**: Sharp RDD primary focus, Fuzzy RDD out of scope (Phase 3B)

**Rationale**:
1. Sharp RDD more common (deterministic treatment assignment)
2. Fuzzy RDD essentially IV (already covered in Phase 4)
3. Time estimate: 28-32 hours for Sharp RDD (reasonable)
4. Fuzzy RDD adds 10-15 hours (brings total to 40-45 hours, too long)
5. Can implement Fuzzy RDD later if needed (uses Sharp RDD + IV infrastructure)

**Impact**: Phase 3 scope reduced to Sharp RDD, more manageable timeline

---

### 2025-11-15: IK and CCT Both Implemented
**Decision**: Implement both IK (2012) and CCT (2014) bandwidth selectors

**Rationale**:
1. IK: MSE-optimal, fast, widely used for exploration
2. CCT: Coverage error-optimal, robust CIs, recommended for inference
3. Trade-off documented: MSE (IK) vs coverage (CCT)
4. R `rdrobust` uses CCT by default (must match for validation)
5. Interview question: "Why CCT instead of IK?" → answer prepared

**Impact**: +5-6 hours for CCT implementation, but essential for rigor

---

### 2025-11-15: McCrary Test Mandatory
**Decision**: McCrary density test runs automatically with every RDD estimation

**Rationale**:
1. Manipulation testing is ASSUMPTION of RDD design
2. Flagged in METHODOLOGICAL_CONCERNS.md (#8) as MEDIUM priority
3. Easy to forget without automation
4. Warning if p < 0.05 alerts user to potential invalidity
5. Interview credibility: "Did you test for manipulation?" → YES

**Impact**: +4-5 hours for McCrary test, but prevents invalid RDD usage

---

### 2025-11-15: R `rdrobust` as Gold Standard
**Decision**: R package `rdrobust` is reference implementation, Julia/Python must match

**Rationale**:
1. `rdrobust` by Calonico, Cattaneo, Titiunik (authors of CCT 2014)
2. Widely used, peer-reviewed, battle-tested
3. Default in applied RDD research
4. Provides confidence that Julia implementation correct
5. Phase 1-2 lesson: External validation essential

**Impact**: +4-5 hours for R validation infrastructure, but establishes credibility

---

**Last Updated**: 2025-11-15
