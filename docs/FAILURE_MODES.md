# Failure Mode Taxonomy

Comprehensive catalog of how causal inference methods can fail, with symptoms, causes, and fixes.

---

## Overview

Each method family has characteristic failure patterns. Understanding these helps:
1. Design better tests
2. Diagnose issues faster
3. Choose appropriate methods
4. Document limitations properly

---

## RCT Methods

### SimpleATE / NeymanATE

**Generally robust** - randomization handles most issues.

| Failure Mode | Symptom | Cause | Fix |
|--------------|---------|-------|-----|
| Extreme imbalance | Large SE, unstable estimates | Small n with unlucky randomization | Stratify or rerandomize |
| Compliance issues | Bias toward zero | Non-compliance | Use ITT or IV |
| Attrition | Bias | Differential dropout | Bounds analysis, IPW |

**Rarely fails** due to fundamental design.

---

### RegressionAdjustedATE

| Failure Mode | Symptom | Cause | Fix |
|--------------|---------|-------|-----|
| Overfitting | Variance inflation | Too many covariates for n | Reduce covariates, use LASSO |
| Multicollinearity | Unstable coefficients | Correlated covariates | Remove redundant variables |
| Model misspecification | Mild bias | Wrong functional form | Usually minor with randomization |

---

## Observational Methods

### IPW (Inverse Probability Weighting)

| Failure Mode | Symptom | Cause | Fix |
|--------------|---------|-------|-----|
| **Extreme weights** | Huge variance, unstable | PS near 0 or 1 | Trim/clip PS to [0.01, 0.99] |
| **Positivity violation** | No estimate possible | No overlap in PS | Restrict to common support |
| **PS model misspecification** | Bias | Wrong propensity model | Use flexible models, DR |
| **Perfect prediction** | PS = 0 or 1 | Separation in data | Check for deterministic assignment |

**Warning signs**:
- Max weight > 20
- Any PS < 0.01 or > 0.99
- Effective sample size much smaller than n

**Implementation**: `src/causal_inference/observational/ipw.py`

---

### DoublyRobust (DR)

| Failure Mode | Symptom | Cause | Fix |
|--------------|---------|-------|-----|
| **Both models wrong** | Bias | Neither PS nor outcome correct | Use more flexible models |
| **Extreme weights** | Variance explosion | Same as IPW | Trim weights |
| **Numerical instability** | NaN/Inf | Division by small PS | Stabilized weights |

**Advantage**: Consistent if *either* model is correct (not both).

**Implementation**: `src/causal_inference/observational/doubly_robust.py`

---

### PSM (Propensity Score Matching)

| Failure Mode | Symptom | Cause | Fix |
|--------------|---------|-------|-----|
| **Poor matches** | Bias, large distances | Insufficient overlap | Tighter caliper, exact matching |
| **Caliper too tight** | Many unmatched | Too restrictive | Loosen caliper |
| **Caliper too loose** | Bias from bad matches | Accepting poor matches | Tighten caliper |
| **Many-to-one matching** | Incorrect SE | Not accounting for replacement | Abadie-Imbens SE |

**See**: CONCERN-5 in `docs/METHODOLOGICAL_CONCERNS.md`

**Implementation**: `src/causal_inference/psm/`

---

## IV Methods

### 2SLS (Two-Stage Least Squares)

| Failure Mode | Symptom | Cause | Fix |
|--------------|---------|-------|-----|
| **Weak instruments** | Large bias, wrong SE | F < 10 | Use LIML or AR |
| **Invalid exclusion** | Bias | Instrument affects outcome directly | Find better instrument |
| **Heterogeneous effects** | LATE ≠ ATE | Effect varies by compliance | Report as LATE |

**Critical check**: First-stage F-statistic > 10

**Implementation**: `src/causal_inference/iv/two_stage_ls.py`

---

### LIML (Limited Information Maximum Likelihood)

| Failure Mode | Symptom | Cause | Fix |
|--------------|---------|-------|-----|
| **Many instruments** | Bias (less than 2SLS) | Over-identification | Reduce instruments |
| **Very weak instruments** | Still biased | F << 10 | Use Anderson-Rubin |

**Advantage**: More robust to weak instruments than 2SLS.

---

### Anderson-Rubin

| Failure Mode | Symptom | Cause | Fix |
|--------------|---------|-------|-----|
| **Wide CIs** | Low power | Cost of robustness | Accept or find stronger IV |
| **Empty CI** | No valid estimates | Data inconsistent with IV | Check assumptions |

**Advantage**: Valid inference regardless of instrument strength.

**Implementation**: `src/causal_inference/iv/weak_iv_robust.py`

---

## DiD Methods

### ClassicDiD / TWFE

| Failure Mode | Symptom | Cause | Fix |
|--------------|---------|-------|-----|
| **Parallel trends violation** | Bias | Differential pre-trends | Use pre-trends test, report |
| **Anticipation effects** | Bias | Units react before treatment | Exclude anticipation period |
| **Heterogeneous effects + staggered** | Bias (TWFE) | Negative weighting | Use CS or SA |

**CRITICAL**: TWFE biased with staggered adoption + heterogeneous effects.
**See**: CONCERN-11 in `docs/METHODOLOGICAL_CONCERNS.md`

**Implementation**: `src/causal_inference/did/`

---

### Callaway-Sant'Anna

| Failure Mode | Symptom | Cause | Fix |
|--------------|---------|-------|-----|
| **Few groups** | Large SE | Insufficient variation | Need more cohorts |
| **Never-treated missing** | Can't estimate | No valid control | Use not-yet-treated |
| **Parallel trends violation** | Bias | Same as classic DiD | Pre-trends test |

**Advantage**: Robust to heterogeneous effects in staggered designs.

---

### Sun-Abraham

| Failure Mode | Symptom | Cause | Fix |
|--------------|---------|-------|-----|
| **Multicollinearity** | Unstable estimates | Too many interaction terms | Aggregate event time |
| **Pre-trend in some cohorts** | Bias | Heterogeneous pre-trends | Investigate by cohort |

---

## RDD Methods

### SharpRDD

| Failure Mode | Symptom | Cause | Fix |
|--------------|---------|-------|-----|
| **Manipulation** | Bunching at cutoff | Units game the running variable | McCrary test, exclude |
| **Bandwidth too narrow** | Large variance | Too few observations | Widen bandwidth |
| **Bandwidth too wide** | Bias | Including non-local data | Use optimal bandwidth |
| **Wrong polynomial** | Bias or variance | Model misspecification | Use local linear |

**Diagnostic**: McCrary density test (should show no discontinuity)

**Implementation**: `src/causal_inference/rdd/`

---

### FuzzyRDD

Inherits all SharpRDD failures plus:

| Failure Mode | Symptom | Cause | Fix |
|--------------|---------|-------|-----|
| **Weak first stage** | Same as weak IV | Small discontinuity in treatment | Need larger jump |
| **Defiers** | Bias in LATE | Some cross in wrong direction | Monotonicity check |

---

## Common Cross-Method Issues

### Sample Size Issues

| n | Risk | Recommendation |
|---|------|----------------|
| < 50 | High variance, unstable | Increase sample if possible |
| 50-200 | Use small-sample corrections | HC3, t-distribution |
| 200-500 | Most methods OK | Standard approaches |
| > 500 | Generally fine | Default settings |

---

### Numerical Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Division by zero | NaN/Inf | Add small constant (1e-10) |
| Log of zero | -Inf | Clip or handle specially |
| Matrix singularity | LinAlgError | Check for collinearity |
| Overflow | Inf | Scale variables |

---

### Inference Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Wrong SE | Bad coverage | Check variance formula |
| Clustering ignored | SE too small | Use cluster-robust SE |
| Multiple testing | False discoveries | Adjust p-values |

---

## Method Selection to Avoid Failures

| If You're Worried About... | Avoid | Prefer |
|---------------------------|-------|--------|
| Weak instruments | 2SLS | LIML, Anderson-Rubin |
| Heterogeneous + staggered | TWFE | Callaway-Sant'Anna |
| Model misspecification | IPW alone, Regression alone | Doubly Robust |
| Extreme propensity scores | Unweighted IPW | Trimmed IPW, Matching |
| Small sample | HC0/HC1 | HC3, Bootstrap |

---

## See Also

- `docs/TROUBLESHOOTING.md` - Step-by-step debugging
- `docs/METHODOLOGICAL_CONCERNS.md` - Project-specific concerns
- `docs/METHOD_SELECTION.md` - Choosing the right method
- `/debug-validation` skill - Systematic debugging workflow
