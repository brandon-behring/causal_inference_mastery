# Known Bugs

**Last Updated**: 2025-12-27 (Session 147)
**Source**: `repo_review_codex.md` + verification tests

This document tracks known correctness and methodological bugs. Each bug has been verified with automated tests in `tests/validation/audit/test_codex_bugs.py`.

---

## FIXED (Sessions 106-110)

### ✅ BUG-8: SCM Optimization Silent Failure — **FIXED in Session 106**

**File**: `src/causal_inference/scm/weights.py`
**Fix**: Added `result.success` check after fallback optimizer. Now raises `ValueError` with diagnostic message.

### ✅ BUG-5: test_type_i_error.py Has Broken Imports — **FIXED in Session 106**

**File**: `tests/validation/monte_carlo/test_type_i_error.py` + bayesian module
**Fix**: Converted to relative imports in bayesian module. Both `pip install -e .` and direct import work.

### ✅ BUG-6: Stratified ATE Anti-Conservative SE — **FIXED in Session 106**

**File**: `src/causal_inference/rct/estimators_stratified.py`
**Fix**: When n₁=1 or n₀=1 in stratum, uses pooled variance from all strata (conservative estimate).

### ✅ BUG-7: ASCM Jackknife Not Real Jackknife — **FIXED in Session 107**

**File**: `src/causal_inference/scm/augmented_scm.py`
**Fix**: Replaced weight renormalization with `compute_scm_weights()` call in LOO loop. Now properly recomputes weights for each LOO configuration.

### ✅ BUG-1: Fuzzy RDD Kernel is No-Op — **FIXED in Session 108**

**File**: `src/causal_inference/rdd/fuzzy_rdd.py`
**Fix**: Implemented weighted 2SLS with kernel weights. Added `_compute_kernel_weights()` for triangular/rectangular/epanechnikov kernels. Added `_weighted_2sls()` with sandwich variance estimator. Kernel weights now properly applied in both first and second stage.

### ✅ BUG-2: CCT Bandwidth Mislabeled — **FIXED in Session 109**

**File**: `src/causal_inference/rdd/bandwidth.py`
**Fix**: Clarified that `cct_bandwidth()` is an approximation, not true CCT:
- Updated docstring with clear warning that function uses IK bandwidth with 1.5× scaling
- Added UserWarning at runtime recommending `rdrobust` for production use
- Updated API docs in `sharp_rdd.py`, `fuzzy_rdd.py`, and `__init__.py`
- Function kept for backward compatibility but limitation now transparent

### ✅ BUG-3: RKD SE Underestimation — **FIXED in Session 110**

**File**: `src/causal_inference/rkd/sharp_rkd.py`
**Fix**: Implemented full delta method for ratio SE:
- Now fits local polynomial for D (treatment) to get slope variances
- SE formula: `Var(τ) = [Var(Δβ_Y) + τ²·Var(Δδ_D)] / Δδ_D²`
- When D slopes are provided (known), variance = 0 (backward compatible)
- Monte Carlo validated: SE calibration within 30%, coverage ~95%

### ✅ BUG-4: AR Test Wrong With Controls — **FIXED in Session 110**

**File**: `src/causal_inference/iv/diagnostics.py`
**Fix**: Residualize instruments Z on controls X when present:
- `Z_perp = Z - X(X'X)⁻¹X'Z` (orthogonalize Z to X)
- Projection matrix uses `Z_perp` for AR test statistic
- Test validates: AR stat differs with/without correlated controls

### ✅ BUG-9: Event Study Allows Staggered Misuse — **FIXED in Session 110**

**File**: `src/causal_inference/did/event_study.py`
**Fix**: Detect staggered adoption from time-varying treatment data:
- If treatment varies within units, detect treatment start time per unit
- If start times differ across units, raise ValueError with helpful message
- Points users to `callaway_santanna()` or `sun_abraham()` for staggered designs
- Changed `units_treated` check from `.first()` to `.max()` for time-varying support

### ✅ BUG-10: Paired Variance Allowed With Replacement — **FIXED in Session 110**

**File**: `src/causal_inference/psm/psm_estimator.py`
**Fix**: Added `with_replacement` validation for paired variance:
- Raises ValueError when `variance_method='paired'` and `with_replacement=True`
- Error message explains the independence assumption violation
- Recommends `variance_method='abadie_imbens'` for matching with replacement

---

## LOW Severity (Documentation/Style) — ALL FIXED

### ✅ DOC-1: Import Path Ambiguity — **FIXED in Session 110**

**Files**: Various (4 src files, 33 test files)

**Issue**: Mixed `src.causal_inference.*` and `causal_inference.*` imports.

**Fix**:
- Standardized all imports to use `src.causal_inference.*` (codebase majority pattern)
- Used relative imports for within-package references (`from .types import ...`)
- Fixed 4 source files: bayesian/bayesian_dr.py, bunching/__init__.py, bunching/counterfactual.py, bunching/excess_mass.py
- Fixed 33 test files across test_bayesian, test_bunching, test_bounds, test_rkd, test_mte, test_scm, and validation directories

---

### ✅ DOC-2: Stale Docstrings — **Already Fixed (verified)**

**File**: `src/causal_inference/psm/psm_estimator.py`

**Status**: No stale docstrings found. Balance diagnostics are fully implemented and documented correctly.

---

## OUTSTANDING (Session 147+)

### ⚠️ BUG-11: Phillips-Perron Test Type I Error ~51%

**File**: `src/causal_inference/timeseries/stationarity.py`
**Priority**: HIGH
**Status**: KNOWN, xfailed tests

**Issue**: Phillips-Perron test has Type I error rate around 51% (expected ~5% at α=0.05). This means the test is essentially flipping a coin rather than providing valid inference.

**Evidence**: Monte Carlo test with 500 runs under H0 (random walk) shows:
- Type I error: 51% (should be ~5%)
- Power: 77% (should be >80%)
- PP/ADF agreement: 63.5% (should be >85%)

**Root Cause**: Likely issue with HAC variance estimation or critical values in PP implementation.

**xfailed Tests**:
- `tests/test_timeseries/test_stationarity_extended.py::test_pp_type1_error`
- `tests/test_timeseries/test_stationarity_extended.py::test_pp_power`
- `tests/test_timeseries/test_stationarity_extended.py::test_pp_vs_adf_agreement`

---

### ⚠️ BUG-12: Moving Block Bootstrap IRF Coverage ~42%

**File**: `src/causal_inference/timeseries/irf.py`
**Priority**: MEDIUM
**Status**: KNOWN, xfailed tests

**Issue**: MBB confidence bands for IRF have coverage around 42% for 90% CI target. Suggests block bootstrap not properly preserving temporal dependence in VAR residual context.

**Evidence**: Monte Carlo test with 100 runs shows:
- Coverage: 42% (should be ~90%)

**Root Cause**: Likely need for larger block lengths or different block bootstrap variant for VAR context.

**xfailed Tests**:
- `tests/test_timeseries/test_irf_extended.py::test_mbb_coverage`
- `tests/test_timeseries/test_irf_extended.py::test_joint_coverage_bonferroni`

---

## Verification

All correctness bugs have automated verification tests:

```bash
pytest tests/validation/audit/test_codex_bugs.py -v
```

---

## Bug Fix Tracking

| Bug | Priority | Session | Status |
|-----|----------|---------|--------|
| BUG-8 | HIGH | 106 | ✅ FIXED |
| BUG-5 | HIGH | 106 | ✅ FIXED |
| BUG-6 | HIGH | 106 | ✅ FIXED |
| BUG-7 | HIGH | 107 | ✅ FIXED |
| BUG-1 | HIGH | 108 | ✅ FIXED |
| BUG-2 | HIGH | 109 | ✅ FIXED |
| BUG-3 | MEDIUM | 110 | ✅ FIXED |
| BUG-4 | MEDIUM | 110 | ✅ FIXED |
| BUG-9 | MEDIUM | 110 | ✅ FIXED |
| BUG-10 | MEDIUM | 110 | ✅ FIXED |
| DOC-1 | LOW | 110 | ✅ FIXED |
| DOC-2 | LOW | — | ✅ Already Fixed |
| BUG-11 | HIGH | — | ⚠️ KNOWN (PP Type I error) |
| BUG-12 | MEDIUM | — | ⚠️ KNOWN (MBB coverage) |

**Outstanding bugs: 2 (1 HIGH, 1 MEDIUM)**

---

**Last Audit**: Session 83 (2025-12-19)
**Last Fix Session**: 110 (2025-12-24) + DOC fixes
**Bug Discovery**: Session 147 (2025-12-27) - Time series validation
