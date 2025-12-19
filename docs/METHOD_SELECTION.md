# Method Selection Decision Tree

A systematic guide for choosing the appropriate causal inference method.

---

## Quick Decision Flowchart

```
START: Do you want to estimate a causal effect?
│
├─ Step 1: Is treatment RANDOMIZED?
│   ├─ YES ──────────────────────────────────► RCT Methods
│   │                                           (SimpleATE, Neyman, Regression-adjusted)
│   └─ NO ──► Continue to Step 2
│
├─ Step 2: Do you have a valid INSTRUMENT?
│   ├─ YES ──────────────────────────────────► IV Methods
│   │   └─ Is instrument weak (F < 10)?        (2SLS, LIML, Anderson-Rubin)
│   │       ├─ YES ──► WeakIVRobust, LIML
│   │       └─ NO ───► 2SLS
│   └─ NO ──► Continue to Step 3
│
├─ Step 3: Is there a DISCONTINUITY in treatment assignment?
│   ├─ YES ──────────────────────────────────► RDD Methods
│   │   └─ Is assignment sharp or fuzzy?       (Sharp, Fuzzy, Local Linear)
│   │       ├─ SHARP ──► SharpRDD
│   │       └─ FUZZY ──► FuzzyRDD (combines RDD + IV)
│   └─ NO ──► Continue to Step 4
│
├─ Step 4: Do you have PRE/POST data with treatment/control groups?
│   ├─ YES ──────────────────────────────────► DiD Methods
│   │   └─ Is adoption staggered?              (Classic, TWFE, CS, SA)
│   │       ├─ NO (single treatment time) ──► ClassicDiD, TWFE
│   │       └─ YES (staggered) ──► Callaway-Sant'Anna, Sun-Abraham
│   └─ NO ──► Continue to Step 5
│
└─ Step 5: Can you assume SELECTION ON OBSERVABLES?
    ├─ YES ──────────────────────────────────► Observational Methods
    │   └─ What's your concern?                (IPW, DR, PSM, Regression)
    │       ├─ Model misspecification ──► DoublyRobust
    │       ├─ Need exact matches ──► PSM
    │       └─ General use ──► IPW or Regression
    └─ NO ──► Consider bounds analysis or find better data
```

---

## Detailed Method Selection

### 1. RCT Methods (Randomized Experiments)

**When to use**: Treatment was randomly assigned

| Method | Best For | Key Feature |
|--------|----------|-------------|
| `SimpleATE` | Basic experiments | Difference in means |
| `NeymanATE` | Conservative inference | Conservative variance (no covariance) |
| `RegressionAdjusted` | Improve precision | Control for pre-treatment covariates |
| `StratifiedATE` | Blocked designs | Account for stratification |

**Assumptions**:
- SUTVA (no interference, single treatment version)
- Random assignment

**Implementation**: `src/causal_inference/rct/`

---

### 2. IV Methods (Instrumental Variables)

**When to use**: You have an instrument that affects treatment but not outcome directly

| Method | Best For | Key Feature |
|--------|----------|-------------|
| `TwoStageLeastSquares` | Strong instruments (F > 10) | Standard IV estimator |
| `LIML` | Weak instruments | More robust to weak IV |
| `AndersonRubin` | Very weak instruments | Valid inference regardless of F |
| `WeakIVRobust` | Suspected weak IV | CLR/AR confidence sets |

**Assumptions**:
- Relevance: Instrument predicts treatment (testable: F > 10)
- Exclusion: Instrument affects outcome only through treatment (untestable)
- Exogeneity: Instrument uncorrelated with unobservables (untestable)
- Monotonicity: No defiers (for LATE interpretation)

**Diagnostics**:
- First-stage F-statistic
- Stock-Yogo weak IV test
- Sargan/Hansen over-identification test (if multiple instruments)

**Implementation**: `src/causal_inference/iv/`

---

### 3. RDD Methods (Regression Discontinuity)

**When to use**: Treatment assigned based on a cutoff in a running variable

| Method | Best For | Key Feature |
|--------|----------|-------------|
| `SharpRDD` | Deterministic cutoff | Local linear regression at cutoff |
| `FuzzyRDD` | Probabilistic cutoff | Combines RDD + IV |
| `LocalLinear` | Standard use | Linear fit in bandwidth |
| `LocalPolynomial` | Curvature | Higher-order polynomials |

**Assumptions**:
- Continuity: Potential outcomes continuous at cutoff
- No manipulation: Units cannot precisely manipulate running variable

**Diagnostics**:
- McCrary density test (manipulation check)
- Covariate balance at cutoff
- Bandwidth sensitivity analysis

**Implementation**: `src/causal_inference/rdd/`

---

### 4. DiD Methods (Difference-in-Differences)

**When to use**: You have panel data with treatment/control groups and pre/post periods

| Method | Best For | Key Feature |
|--------|----------|-------------|
| `ClassicDiD` | 2 periods, 2 groups | Simple before-after comparison |
| `TWFE` | Multiple periods/groups | Two-way fixed effects regression |
| `CallawayASantAnna` | Staggered adoption | Robust to heterogeneous effects |
| `SunAbraham` | Event studies | Interaction-weighted estimator |
| `EventStudy` | Dynamic effects | Effect by time relative to treatment |

**When NOT to use TWFE**:
- Staggered adoption + heterogeneous treatment effects → Use CS or SA
- See CONCERN-11 in `docs/METHODOLOGICAL_CONCERNS.md`

**Assumptions**:
- Parallel trends: Treatment/control would trend similarly absent treatment
- No anticipation: Units don't change behavior before treatment
- SUTVA: No spillovers

**Diagnostics**:
- Pre-trends test (event study with pre-periods)
- Placebo tests

**Implementation**: `src/causal_inference/did/`

---

### 5. Observational Methods

**When to use**: No randomization, instrument, discontinuity, or panel structure

| Method | Best For | Key Feature |
|--------|----------|-------------|
| `IPW` | Reweighting | Creates pseudo-population |
| `DoublyRobust` | Robustness | Consistent if either model correct |
| `OutcomeRegression` | Simple cases | Covariate adjustment |
| `PSM` | Exact matching | Match treated to controls |

**Assumptions**:
- Unconfoundedness: All confounders observed (untestable)
- Positivity/Overlap: 0 < P(T=1|X) < 1 for all X

**Diagnostics**:
- Covariate balance after weighting/matching
- Propensity score overlap plots
- Sensitivity analysis (Rosenbaum bounds)

**Implementation**: `src/causal_inference/observational/`, `src/causal_inference/psm/`

---

## Method Comparison Matrix

| Method Family | Estimand | Identification | Internal Validity | External Validity |
|---------------|----------|----------------|-------------------|-------------------|
| RCT | ATE | Randomization | High | Depends on sample |
| IV | LATE | Exclusion + Relevance | Medium-High | Compliers only |
| RDD | LATE at cutoff | Continuity | High near cutoff | Local to cutoff |
| DiD | ATT | Parallel trends | Medium | Treated units |
| Observational | ATE/ATT | Selection on observables | Low-Medium | Sample-dependent |

---

## Sample Size Guidelines

| Method | Minimum n | Recommended n | Notes |
|--------|-----------|---------------|-------|
| SimpleATE | 30/group | 100/group | Larger for small effects |
| IPW | 100 | 500+ | Need overlap |
| DiD | 50/group | 200/group | More for clustering |
| IV | 100 | 500+ | More for weak IV |
| RDD | 100 near cutoff | 500+ | Depends on bandwidth |

---

## Quick Reference: By Scenario

| Scenario | Recommended Method | Reason |
|----------|-------------------|--------|
| A/B test | SimpleATE or NeymanATE | Randomization ensures identification |
| Policy change at threshold | SharpRDD | Natural experiment at cutoff |
| Natural experiment with instrument | 2SLS | Exploit exogenous variation |
| New policy rollout over time | ClassicDiD | Compare pre/post, treatment/control |
| Staggered state policy adoption | Callaway-Sant'Anna | Handles heterogeneity correctly |
| Observational data, rich covariates | DoublyRobust | Best robustness properties |
| Need exact comparison units | PSM | Interpretable matches |
| Concerned about weak instrument | LIML or AndersonRubin | Robust to weak IV |

---

## See Also

- `docs/TROUBLESHOOTING.md` - When methods fail
- `docs/FAILURE_MODES.md` - Method-specific failure patterns
- `docs/METHODOLOGICAL_CONCERNS.md` - Known issues and solutions
- Query research-kb: `research_kb_get_concept "{method_name}"`
