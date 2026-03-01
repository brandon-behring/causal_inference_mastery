# Propensity Score Matching

Match treated and control units with similar propensity scores to create a pseudo-experimental comparison.

## How It Works

1. Estimate propensity score $\hat{e}(X) = P(T=1|X)$ via logistic regression
2. Match each treated unit to nearest control unit(s) by propensity score
3. Estimate ATE from matched sample
4. Check covariate balance

## Example

```python
from causal_inference.psm import PropensityScoreMatching

psm = PropensityScoreMatching(n_neighbors=1)
result = psm.fit(data, outcome="y", treatment="treated", covariates=["x1", "x2"])
print(f"ATT: {result.att:.3f} (SE: {result.se:.3f})")
print(f"Balance: {result.balance_summary}")
```

## Variance Estimation

Uses the Abadie-Imbens (2006) variance estimator, which accounts for the matching estimation step. This is more accurate than naive bootstrap for matched samples.

## Balance Diagnostics

After matching, verify covariate balance:
- Standardized mean differences (target: < 0.1)
- Variance ratios (target: 0.8–1.2)
- Visual: density plots of propensity scores before/after matching

## When to Prefer Matching vs. IPW

| Feature | Matching | IPW |
|---------|----------|-----|
| Interpretability | High (explicit comparisons) | Lower |
| Extreme propensity scores | Handles poorly | Unstable weights |
| Efficiency | Lower (discards data) | Higher (uses all data) |
| Balance checking | Visual, intuitive | Less intuitive |

See {doc}`/api/psm` for full API reference.
