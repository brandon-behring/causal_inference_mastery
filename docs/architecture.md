# Architecture

## Package Structure

```
src/causal_inference/
├── rct/                  # Randomized experiments (5 estimators)
├── observational/        # IPW, DR, outcome regression
├── psm/                  # Propensity score matching
├── did/                  # DiD, Callaway-Sant'Anna, Sun-Abraham, event study
├── iv/                   # 2SLS, LIML, Fuller, GMM, diagnostics
├── rdd/                  # Sharp/fuzzy RDD, bandwidth selection
├── rkd/                  # Regression kink design
├── scm/                  # Synthetic control methods
├── cate/                 # Causal forests, DML, meta-learners
├── sensitivity/          # E-values, Rosenbaum bounds
├── bounds/               # Manski, Lee bounds
├── qte/                  # Quantile treatment effects
├── mte/                  # Marginal treatment effects
├── mediation/            # Baron-Kenny, NDE/NIE
├── control_function/     # Linear, nonlinear CF
├── shift_share/          # Bartik IV, Rotemberg diagnostics
├── bunching/             # Excess mass estimation
├── principal_stratification/  # Strata-specific effects
├── bayesian/             # Bayesian ATE, propensity
├── timeseries/           # VAR, SVAR, IRF, Granger, VECM
├── discovery/            # PC, FCI, GES, LiNGAM
├── dtr/                  # Dynamic treatment regimes
├── panel/                # Panel data methods
├── dynamic/              # Dynamic effects
├── selection/            # Heckman selection
├── utils/                # Variance estimators, helpers
├── data/                 # Example datasets
└── evaluation/           # Diagnostics utilities
```

## Design Principles

### 1. Test-First Development

Every estimator starts with known-answer tests before implementation.

### 2. Consistent Interface

All estimators follow the pattern:

```python
estimator = Estimator(**config)
result = estimator.fit(data, outcome="y", treatment="t", **kwargs)
# result.ate, result.se, result.p_value, result.ci_lower, result.ci_upper
```

### 3. No Silent Failures

Invalid inputs raise explicit `ValueError` or `TypeError` with diagnostic information. No NaN propagation.

### 4. Pure Functions Where Possible

Estimators are stateless — configuration at init, computation at `fit()`, results returned as frozen dataclasses.

### 5. Cross-Language Parity

Python and Julia implementations are validated to agree within $10^{-10}$, ensuring the math is correct independent of implementation language.

## Scale

| Metric | Value |
|--------|-------|
| Python LOC | 54,727 |
| Julia LOC | 43,699 |
| Total LOC | 98,426 |
| Method families | 26 |
| Python test functions | 3,854 |
| Julia assertions | 5,121 |
| Test coverage | 90%+ |
| Known bugs | 0 |
