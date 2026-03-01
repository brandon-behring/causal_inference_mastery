# Causal Inference Mastery

**Rigorous causal inference implementation in Python and Julia — 26 method families, 98K+ lines, 6-layer validation.**

A research-grade library covering the full spectrum of causal inference methods, from randomized experiments to causal discovery. Every estimator is validated against known answers, Monte Carlo simulations, cross-language parity (Python ↔ Julia), and R triangulation.

## At a Glance

::::{grid} 3
:gutter: 3

:::{grid-item-card} 26 Method Families
:link: method-selection
:link-type: doc

RCT, DiD, IV, RDD, SCM, CATE, time series, Bayesian, and more.
:::

:::{grid-item-card} 6-Layer Validation
:link: validation
:link-type: doc

Known-answer → Adversarial → Monte Carlo → Cross-language → R triangulation → Golden reference.
:::

:::{grid-item-card} 98K+ Lines
:link: architecture
:link-type: doc

54K Python + 44K Julia, 3,854 test functions, 90%+ coverage, 0 known bugs.
:::
::::

## Quick Example

```python
from causal_inference.did import DiDEstimator

# Classic 2x2 Difference-in-Differences
estimator = DiDEstimator()
result = estimator.fit(data, outcome="y", treatment="treated", time="post")
print(f"ATE: {result.ate:.2f}, SE: {result.se:.3f}, p: {result.p_value:.4f}")
# ATE: 2.00, SE: 0.354, p: 0.0001
```

## Method Families

| Category | Methods |
|----------|---------|
| **Experiments** | RCT (simple, stratified, regression-adjusted, IPW, permutation) |
| **Selection on Observables** | IPW, DR, TMLE, propensity score matching |
| **Natural Experiments** | DiD, RDD, RKD, synthetic control |
| **Endogeneity** | IV (2SLS, LIML, Fuller, GMM), control function, shift-share |
| **Heterogeneous Effects** | CATE (causal forests, DML, meta-learners), QTE, MTE |
| **Sensitivity & Bounds** | E-values, Rosenbaum bounds, Manski bounds, Lee bounds |
| **Advanced** | Mediation, time series (VAR, SVAR, IRF), panel, dynamic treatment regimes |
| **Specialized** | Bayesian, causal discovery (PC, FCI), bunching, principal stratification |

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting-started
method-selection
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user-guide/rct
user-guide/observational
user-guide/matching
user-guide/did
user-guide/iv
user-guide/rdd
user-guide/synthetic-control
user-guide/cate
user-guide/sensitivity
user-guide/time-series
user-guide/discovery
user-guide/advanced
```

```{toctree}
:maxdepth: 2
:caption: Reference

validation
architecture
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
api/rct
api/observational
api/psm
api/did
api/iv
api/rdd
api/rkd
api/scm
api/cate
api/sensitivity
api/bounds
api/qte
api/mte
api/mediation
api/control-function
api/shift-share
api/bunching
api/principal-strat
api/bayesian
api/timeseries
api/discovery
api/dtr
api/panel
api/utils
```

```{toctree}
:maxdepth: 1
:caption: Project

changelog
glossary
troubleshooting
```
