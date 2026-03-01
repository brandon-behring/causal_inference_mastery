# Instrumental Variables

Handles endogeneity when treatment is correlated with unobserved confounders. An instrument $Z$ affects the outcome only through the treatment.

## Estimators

| Estimator | Use Case |
|-----------|----------|
| 2SLS | Standard IV, just-identified or over-identified |
| LIML | More robust to weak instruments than 2SLS |
| Fuller | Finite-sample bias correction (LIML variant) |
| GMM | Efficient with heteroskedasticity |

## Assumptions

1. **Relevance**: $\text{Cov}(Z, X) \neq 0$ (instrument predicts treatment)
2. **Exclusion restriction**: $Z$ affects $Y$ only through $X$
3. **Independence**: $Z$ is independent of unobserved confounders

## Example

```python
from causal_inference.iv import TwoStageLeastSquares

iv = TwoStageLeastSquares()
result = iv.fit(data, outcome="y", treatment="x", instruments=["z1", "z2"],
                covariates=["age", "education"])

print(f"LATE: {result.coefficient:.3f} (SE: {result.se:.3f})")
print(f"First-stage F: {result.first_stage_f:.1f}")
```

## Diagnostics

Always report these diagnostics:

| Diagnostic | Threshold | Purpose |
|-----------|-----------|---------|
| First-stage F-statistic | > 10 (Stock-Yogo) | Instrument strength |
| Sargan/Hansen J-test | p > 0.05 | Over-identification |
| Wu-Hausman test | — | Endogeneity of treatment |
| Anderson-Rubin test | — | Weak-instrument robust inference |

## Weak Instruments

When first-stage F < 10:
- **LIML** is less biased than 2SLS
- **Fuller** provides finite-sample correction
- **Anderson-Rubin** confidence sets are valid regardless of instrument strength

See {doc}`/api/iv` for full API reference.
