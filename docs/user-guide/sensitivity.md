# Sensitivity Analysis

Quantify how robust your causal conclusions are to potential unobserved confounding.

## E-values

**Question**: How strong would an unmeasured confounder need to be to explain away the observed effect?

$$E\text{-value} = RR + \sqrt{RR \times (RR - 1)}$$

Higher E-values mean more robust findings. An E-value of 3.5 means a confounder would need a risk ratio of 3.5 with both treatment and outcome to reduce the effect to null.

```python
from causal_inference.sensitivity import compute_evalue

e = compute_evalue(point_estimate=2.5, ci_lower=1.3)
print(f"E-value (point): {e.point:.2f}")
print(f"E-value (CI): {e.ci_lower:.2f}")
```

## Rosenbaum Bounds

For matched studies: how much hidden bias (Γ) would be needed to alter the conclusion?

$$\Gamma = \frac{P(T=1|X,U=1) / P(T=0|X,U=1)}{P(T=1|X,U=0) / P(T=0|X,U=0)}$$

```python
from causal_inference.sensitivity import rosenbaum_bounds

bounds = rosenbaum_bounds(matched_data, outcome="y", gammas=[1.0, 1.5, 2.0, 2.5])
for g, p in zip(bounds.gammas, bounds.p_values):
    print(f"Γ={g:.1f}: p={p:.4f}")
```

A study is "robust to Γ=2.0" if the conclusion holds even when units could differ by a factor of 2 in treatment probability due to unobserved confounders.

See {doc}`/api/sensitivity` for full API reference.
