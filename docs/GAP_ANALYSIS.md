# Gap Analysis: Missing Causal Inference Methods

**Audit Date**: 2025-12-19 (Session 83)
**Scope**: Identify methods not implemented that have high interview/research frequency

---

## Current Implementation Status

### Implemented Methods (14 Families)

| Family | Python | Julia | Estimators |
|--------|--------|-------|------------|
| RCT | ✅ | ✅ | SimpleATE, Stratified, IPW, Regression-Adjusted, Permutation |
| Observational | ✅ | ✅ | IPW, Doubly Robust, Outcome Regression |
| PSM | ✅ | ✅ | Nearest Neighbor, Caliper, Optimal |
| DiD | ✅ | ✅ | Classic 2x2, Event Study, Callaway-Sant'Anna, Sun-Abraham |
| IV | ✅ | ✅ | 2SLS, GMM, LIML, Fuller, Diagnostics |
| RDD | ✅ | ✅ | Sharp, Fuzzy, McCrary, Sensitivity |
| SCM | ✅ | ✅ | Basic, Augmented, Placebo Inference |
| CATE | ✅ | ✅ | S/T/X/R-Learners, DML |
| Sensitivity | ✅ | ✅ | Rosenbaum Bounds, E-values |
| RKD | ✅ | ✅ | Sharp, Fuzzy, Diagnostics |
| Bunching | ✅ | ✅ | Saez Excess Mass, Counterfactual |

---

## TIER 1: Critical Gaps (High Interview Frequency)

### 1. Heckman Selection Model

**Priority**: 🔴 CRITICAL
**Interview Frequency**: ~40% of applied micro interviews

**What it is**: Two-stage estimator for selection bias when outcome is only observed for a selected sample (e.g., wages observed only for employed).

**Key features needed**:
- Probit selection equation
- Inverse Mills ratio construction
- Outcome equation with selection correction
- Standard errors accounting for two-stage estimation
- Tests for selection (ρ significance)

**References**:
- Heckman (1979). Sample Selection Bias as a Specification Error
- Wooldridge (2010). Econometric Analysis of Cross Section and Panel Data, Ch. 19

**Estimated effort**: 2-3 sessions

---

### 2. Manski Bounds (Worst-Case Bounds)

**Priority**: 🔴 HIGH
**Interview Frequency**: ~25% of theory-focused interviews

**What it is**: Nonparametric bounds on treatment effects under minimal assumptions. Provides identification region when point identification fails.

**Key features needed**:
- Treatment effect bounds under no assumptions
- Monotone treatment response (MTR) bounds
- Monotone treatment selection (MTS) bounds
- Combined MTR+MTS bounds
- Instrumental variables bounds

**References**:
- Manski (1990). Nonparametric Bounds on Treatment Effects
- Manski (2003). Partial Identification of Probability Distributions

**Estimated effort**: 2 sessions

---

### 3. Lee Bounds (Attrition/Selection)

**Priority**: 🔴 HIGH
**Interview Frequency**: ~20% of RCT-focused interviews

**What it is**: Bounds on treatment effects when there's sample attrition or selection that may be affected by treatment.

**Key features needed**:
- Trimming procedure (trim always-takers or never-takers)
- Sharp bounds construction
- Confidence intervals via bootstrap
- Sensitivity to monotonicity assumption

**References**:
- Lee (2009). Training, Wages, and Sample Selection

**Estimated effort**: 1-2 sessions

---

### 4. Quantile Treatment Effects (QTE)

**Priority**: 🔴 HIGH
**Interview Frequency**: ~30% of distributional questions

**What it is**: Estimate treatment effects at different quantiles of the outcome distribution, not just the mean.

**Key features needed**:
- Unconditional QTE
- Conditional QTE (quantile regression)
- Distributional effects (entire distribution comparison)
- Standard errors (bootstrap, analytical)

**References**:
- Koenker & Bassett (1978). Regression Quantiles
- Firpo, Fortin, Lemieux (2009). Unconditional Quantile Regressions

**Estimated effort**: 2 sessions

---

## TIER 2: Medium Priority Gaps

### 5. Marginal Treatment Effects (MTE)

**Priority**: 🟡 MEDIUM
**Use case**: Understanding heterogeneity in IV settings, LATE decomposition

**What it is**: Treatment effects as a function of unobserved resistance to treatment. Generalizes LATE.

**Key features needed**:
- Local IV estimation along propensity score
- Policy-relevant treatment effects (PRTE)
- Average treatment effects from MTE

**References**:
- Heckman & Vytlacil (2005). Structural Equations, Treatment Effects

**Estimated effort**: 3 sessions

---

### 6. TMLE (Targeted Maximum Likelihood Estimation)

**Priority**: 🟡 MEDIUM
**Use case**: Doubly robust estimation with formal statistical guarantees

**What it is**: Fluctuation-based approach to doubly robust estimation. Has better finite-sample properties than standard DR.

**Note**: Partially covered by DML, but TMLE adds targeting step.

**References**:
- van der Laan & Rose (2011). Targeted Learning

**Estimated effort**: 2 sessions

---

### 7. Mediation Analysis

**Priority**: 🟡 MEDIUM
**Use case**: Decomposing direct and indirect effects

**Key features needed**:
- Natural direct effect (NDE)
- Natural indirect effect (NIE)
- Controlled direct effect (CDE)
- Sensitivity analysis for unmeasured confounding

**References**:
- Baron & Kenny (1986) - traditional
- Imai, Keele, Tingley (2010) - causal mediation

**Estimated effort**: 2 sessions

---

### 8. Control Function Approach

**Priority**: 🟡 MEDIUM
**Use case**: Nonlinear models with endogeneity

**What it is**: Alternative to IV for nonlinear models. Estimates residual from first stage and includes in second stage.

**References**:
- Rivers & Vuong (1988)
- Wooldridge (2015). Control Function Methods

**Estimated effort**: 1-2 sessions

---

### 9. Shift-Share / Bartik Instruments

**Priority**: 🟡 MEDIUM
**Use case**: Regional economics, labor economics

**What it is**: Instruments constructed from national/industry shocks interacted with local shares.

**Key features needed**:
- Shift-share construction
- Rotemberg weights
- Exposure robust standard errors
- Diagnostics (share exogeneity)

**References**:
- Goldsmith-Pinkham, Sorkin, Swift (2020). Bartik Instruments

**Estimated effort**: 2 sessions

---

## TIER 3: Lower Priority Gaps

### 10. Principal Stratification

**Use case**: Noncompliance beyond LATE, post-treatment confounding

**Estimated effort**: 3 sessions

---

### 11. Dynamic Treatment Regimes (DTR)

**Use case**: Sequential treatment decisions over time

**Estimated effort**: 4 sessions

---

### 12. Bayesian Causal Inference

**Use case**: Incorporating prior information, posterior distributions of effects

**Estimated effort**: 3 sessions

---

### 13. Generalized Random Forests (GRF)

**Use case**: Nonparametric CATE with confidence intervals

**Note**: Could wrap `econml` or `grf` R package

**Estimated effort**: 2 sessions

---

## Recommended Phase 12 Roadmap

Based on interview frequency and research importance:

```
Phase 12: Selection & Bounds (Sessions 84-89)
├── Session 84: Heckman Selection (Python)
├── Session 85: Heckman Selection (Julia + cross-language)
├── Session 86: Manski Bounds (Python)
├── Session 87: Lee Bounds (Python)
├── Session 88: Bounds (Julia + cross-language)
└── Session 89: QTE (both languages)

Phase 13: Advanced Methods (Sessions 90-95)
├── Session 90: MTE (Python)
├── Session 91: MTE (Julia)
├── Session 92: Mediation Analysis
├── Session 93: Control Function
├── Session 94: Shift-Share IV
└── Session 95: Cross-language parity for Phase 12-13
```

---

## Priority Matrix

| Method | Interview Freq | Research Use | Difficulty | Priority |
|--------|---------------|--------------|------------|----------|
| Heckman | 40% | Very High | Medium | 🔴 P1 |
| Manski Bounds | 25% | High | Medium | 🔴 P1 |
| Lee Bounds | 20% | High | Low | 🔴 P1 |
| QTE | 30% | Very High | Medium | 🔴 P1 |
| MTE | 15% | High | High | 🟡 P2 |
| TMLE | 10% | Medium | Medium | 🟡 P2 |
| Mediation | 20% | Medium | Medium | 🟡 P2 |
| Control Function | 10% | Medium | Low | 🟡 P2 |
| Shift-Share | 15% | Medium | Medium | 🟡 P2 |
| Principal Strat | 5% | Low | High | 🟢 P3 |
| DTR | 5% | Low | High | 🟢 P3 |
| Bayesian | 5% | Low | High | 🟢 P3 |
| GRF | 10% | Medium | Low | 🟢 P3 |

---

## Conclusion

**Highest impact additions for interview prep**:
1. Heckman Selection - Most frequently asked
2. QTE - Common distributional questions
3. Bounds (Manski + Lee) - Theory-heavy interviews

**Total estimated sessions for TIER 1**: 6-8 sessions

---

**Generated**: Session 83 Audit
