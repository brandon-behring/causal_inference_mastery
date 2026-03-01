# Randomized Controlled Trials

The gold standard for causal inference. When treatment is randomly assigned, the difference in means is an unbiased estimator of the Average Treatment Effect (ATE).

## Estimators

| Estimator | Use Case |
|-----------|----------|
| Simple difference-in-means | Basic RCT, equal allocation |
| Stratified | Pre-stratified randomization |
| Regression-adjusted | Covariate adjustment for efficiency |
| IPW | When treatment probability varies |
| Permutation | Exact inference, no distributional assumptions |

## Example

```python
from causal_inference.rct import SimpleATEEstimator

estimator = SimpleATEEstimator()
result = estimator.fit(data, outcome="y", treatment="treated")
print(f"ATE: {result.ate:.3f} ± {1.96 * result.se:.3f}")
```

## When to Use

- You have experimental data with random assignment
- Compliance is perfect (no non-compliance → LATE territory)
- Sample size is sufficient for target effect size

## Key Assumptions

1. **Random assignment** — Treatment is independent of potential outcomes
2. **SUTVA** — No interference between units; one version of treatment
3. **No attrition bias** — Missing outcomes are random

## Variance Estimators

The package provides multiple variance estimators for different settings:

- **Neyman** — Conservative, does not assume constant treatment effect
- **HC0-HC3** — Heteroskedasticity-consistent (Eicker-Huber-White family)
- **Cluster-robust** — For clustered randomization

See {doc}`/api/rct` for full API reference.
