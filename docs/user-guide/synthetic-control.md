# Synthetic Control Method

Constructs a weighted combination of donor units to approximate the counterfactual for a single treated unit.

## How It Works

1. Select donor pool (untreated units)
2. Find weights $W^*$ that minimize pre-treatment fit: $\min_W \|Y_1^{pre} - Y_0^{pre} W\|$
3. Counterfactual: $\hat{Y}_1^{post}(0) = Y_0^{post} W^*$
4. Treatment effect: $\hat{\tau}_t = Y_{1t} - \hat{Y}_{1t}(0)$

## Example

```python
from causal_inference.scm import SyntheticControl

sc = SyntheticControl()
result = sc.fit(panel_data, treated_unit="California",
                time_var="year", outcome="smoking_rate",
                treatment_year=1989)

print(f"Average effect: {result.average_effect:.2f}")
print(f"Donor weights: {result.weights}")
```

## Inference

### Placebo Tests

Iteratively apply SCM to each donor unit and compare effect sizes. The treated unit's effect should be an outlier.

### Permutation p-value

$$p = \frac{\text{rank of treated unit's RMSPE ratio}}{J + 1}$$

See {doc}`/api/scm` for full API reference.
