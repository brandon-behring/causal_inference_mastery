# Difference-in-Differences

Identifies causal effects by comparing changes over time between treated and control groups.

## Classic 2×2 DiD

$$\hat{\tau}_{DiD} = (\bar{Y}_{T,post} - \bar{Y}_{T,pre}) - (\bar{Y}_{C,post} - \bar{Y}_{C,pre})$$

### Key Assumption

**Parallel trends**: In the absence of treatment, treated and control groups would have followed the same time trend.

## Estimators

| Estimator | Use Case |
|-----------|----------|
| Classic DiD | 2 periods, 2 groups |
| TWFE (Two-Way Fixed Effects) | Multiple periods, simple staggered |
| Callaway-Sant'Anna | Staggered adoption, heterogeneous effects |
| Sun-Abraham | Interaction-weighted, heterogeneous timing |
| Event Study | Dynamic treatment effects, pre-trend testing |

## Example

```python
from causal_inference.did import DiDEstimator

# Classic 2x2
estimator = DiDEstimator()
result = estimator.fit(data, outcome="y", treatment="treated", time="post")
print(f"ATE: {result.ate:.2f} (SE: {result.se:.3f})")
```

### Staggered Adoption

```python
from causal_inference.did import CallawaySantAnna

cs = CallawaySantAnna()
result = cs.fit(data, outcome="y", group="first_treated", time="year", id="unit_id")
print(f"Overall ATT: {result.att:.3f}")
```

## When TWFE Fails

TWFE can produce biased estimates under staggered adoption with heterogeneous treatment effects. The problem: already-treated units serve as "controls" for newly-treated units, and the implicit negative weighting can reverse the sign of the estimate.

**Solution**: Use Callaway-Sant'Anna or Sun-Abraham estimators, which properly handle staggered timing.

## Wild Bootstrap Inference

For clustered data with few clusters, use wild cluster bootstrap for correct inference:

```python
from causal_inference.did import WildBootstrapDiD

wb = WildBootstrapDiD(n_bootstrap=999)
result = wb.fit(data, outcome="y", treatment="treated", time="post", cluster="state")
```

See {doc}`/api/did` for full API reference.
