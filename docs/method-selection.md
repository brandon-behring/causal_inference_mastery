# Method Selection Guide

Choosing the right causal inference method depends on your data structure, identification strategy, and assumptions. This guide provides a decision tree.

## Decision Tree

```
Is treatment randomly assigned?
├── YES → RCT estimators (simple, stratified, regression-adjusted)
└── NO → What identification strategy?
    ├── Selection on observables (conditional independence)
    │   ├── Many confounders → IPW, Doubly Robust, TMLE
    │   └── Few confounders, want matched comparison → Propensity Score Matching
    ├── Natural experiment
    │   ├── Before/after + treated/control → Difference-in-Differences
    │   │   ├── 2 periods, 2 groups → Classic DiD
    │   │   └── Staggered adoption → Callaway-Sant'Anna, Sun-Abraham
    │   ├── Running variable with cutoff → Regression Discontinuity
    │   │   ├── Sharp assignment → Sharp RDD
    │   │   └── Fuzzy assignment → Fuzzy RDD
    │   ├── Excluded instrument available → Instrumental Variables
    │   │   ├── Just-identified → 2SLS
    │   │   ├── Over-identified → LIML, GMM
    │   │   └── Weak instruments → Fuller, LIML
    │   └── Few treated units, donor pool → Synthetic Control
    ├── Heterogeneous effects needed
    │   ├── High-dimensional confounders → Causal Forest, DML
    │   ├── Meta-learners → S-learner, T-learner, X-learner, R-learner
    │   └── Quantile effects → QTE
    ├── Sensitivity analysis
    │   ├── How much confounding to explain away? → E-values
    │   └── Bounds on effect → Rosenbaum bounds, Manski bounds
    └── Advanced
        ├── Mediation analysis → Baron-Kenny, NDE/NIE
        ├── Time series causal effects → VAR, SVAR, Granger
        ├── Causal graph discovery → PC, FCI, GES
        └── Dynamic treatment regimes → DTR
```

## Method Comparison Table

| Method | Key Assumption | Data Needs | Strengths |
|--------|---------------|------------|-----------|
| **RCT** | Random assignment | Experimental data | Gold standard, unbiased |
| **IPW/DR** | Conditional independence | Observational + confounders | Flexible, doubly robust |
| **PSM** | Conditional independence | Observational + propensity | Intuitive, balance checks |
| **DiD** | Parallel trends | Panel or repeated cross-section | Widely applicable |
| **RDD** | Continuity at cutoff | Running variable | Strong internal validity |
| **IV** | Exclusion restriction | Instrument(s) | Handles endogeneity |
| **SCM** | Donor pool quality | Few treated, many donors | Data-driven counterfactual |
| **CATE** | Varies by learner | Large n, covariates | Individual-level effects |
| **E-values** | None (sensitivity) | Point estimate + CI | Quantifies robustness |

## When to Use What

### You have an experiment
→ Start with {doc}`user-guide/rct`. Use stratification or regression adjustment for efficiency.

### You observe treatment and potential confounders
→ {doc}`user-guide/observational` for IPW/DR, or {doc}`user-guide/matching` for PSM.

### You have before/after data with a policy change
→ {doc}`user-guide/did`. Check parallel trends. For staggered adoption, use Callaway-Sant'Anna.

### You have a score-based assignment rule
→ {doc}`user-guide/rdd`. Sharp if deterministic, fuzzy if probabilistic.

### You have an instrument
→ {doc}`user-guide/iv`. Always check first-stage F > 10 and report diagnostics.

### You want individual treatment effects
→ {doc}`user-guide/cate`. Causal forests for nonparametric, DML for semiparametric.

### You need to assess robustness
→ {doc}`user-guide/sensitivity`. E-values quantify how much confounding would be needed.
