# Changelog

## v0.1.0 (2026-02)

Initial release with 26 method families.

### Method Families

- **Experiments**: RCT (5 estimators: simple, stratified, regression, IPW, permutation)
- **Selection on Observables**: IPW, DR, outcome regression, propensity score matching
- **Difference-in-Differences**: Classic, TWFE, Callaway-Sant'Anna, Sun-Abraham, event study, wild bootstrap
- **Instrumental Variables**: 2SLS, LIML, Fuller, GMM, diagnostics
- **Regression Discontinuity**: Sharp, fuzzy, bandwidth selection, McCrary test
- **Regression Kink Design**: Sharp kink, fuzzy kink
- **Synthetic Control**: Basic, augmented, placebo inference
- **CATE**: Causal forests, DML, S/T/X/R-learners, neural CATE
- **Sensitivity**: E-values, Rosenbaum bounds
- **Bounds**: Manski, Lee bounds
- **QTE**: Quantile treatment effects
- **MTE**: Marginal treatment effects, LATE, policy parameters
- **Mediation**: Baron-Kenny, NDE/NIE, CDE
- **Control Function**: Linear, nonlinear (probit/logit)
- **Shift-Share**: Bartik IV, Rotemberg diagnostics
- **Bunching**: Excess mass estimation
- **Principal Stratification**: Strata-specific effects
- **Bayesian**: Bayesian ATE, propensity stratification
- **Time Series**: VAR, SVAR, IRF, Granger, VECM, cointegration
- **Causal Discovery**: PC, FCI, GES, LiNGAM
- **Dynamic Treatment Regimes**: DTR
- **Panel**: Fixed effects, random effects, dynamic models

### Validation

- 3,854 Python test functions
- 5,121 Julia assertions
- 6-layer validation: known-answer, adversarial, Monte Carlo, cross-language, R triangulation, golden reference
- 90%+ test coverage
- 0 known bugs
