# Observational Methods

When treatment is not randomized but you can observe all confounders, these methods recover causal effects under the conditional independence assumption.

## Methods

### Inverse Probability Weighting (IPW)

Reweights observations by the inverse of their treatment probability:

$$\hat{\tau}_{IPW} = \frac{1}{n}\sum_{i=1}^{n}\left[\frac{T_i Y_i}{\hat{e}(X_i)} - \frac{(1-T_i)Y_i}{1-\hat{e}(X_i)}\right]$$

where $\hat{e}(X_i)$ is the estimated propensity score.

### Doubly Robust (DR)

Combines outcome regression and IPW — consistent if *either* model is correct:

$$\hat{\tau}_{DR} = \frac{1}{n}\sum_{i=1}^{n}\left[\hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{T_i(Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1-T_i)(Y_i - \hat{\mu}_0(X_i))}{1-\hat{e}(X_i)}\right]$$

### Outcome Regression

Model the conditional expectation $E[Y|X, T]$ directly and compute the ATE as the average predicted difference.

## Key Assumption

**Conditional independence (unconfoundedness)**: $Y(0), Y(1) \perp T \mid X$

All confounders must be observed and included. Unobserved confounders violate this assumption — use sensitivity analysis ({doc}`sensitivity`) to assess robustness.

## Example

```python
from causal_inference.observational import DoublyRobustEstimator

dr = DoublyRobustEstimator()
result = dr.fit(data, outcome="y", treatment="treated", covariates=["x1", "x2", "x3"])
print(f"ATE: {result.ate:.3f}, 95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
```

See {doc}`/api/observational` for full API reference.
