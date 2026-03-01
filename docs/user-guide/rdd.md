# Regression Discontinuity

Exploits a known cutoff in a running variable that determines treatment assignment. Strong internal validity — often considered the closest quasi-experimental design to an RCT.

## Sharp vs. Fuzzy RDD

**Sharp RDD**: Treatment is deterministic at the cutoff.
$$T_i = \mathbf{1}[X_i \geq c]$$

**Fuzzy RDD**: Treatment probability jumps at the cutoff but is not deterministic. Estimated via local IV.

## Example

```python
from causal_inference.rdd import SharpRDD

rdd = SharpRDD(bandwidth="optimal")
result = rdd.fit(data, outcome="y", running_var="score", cutoff=50)
print(f"LATE at cutoff: {result.late:.3f} (bandwidth: {result.bandwidth:.1f})")
```

## Key Components

### Bandwidth Selection

Optimal bandwidth balances bias (narrow) vs. variance (wide):
- **IK (Imbens-Kalyanaraman)**: Mean squared error optimal
- **CCT (Calonico-Cattaneo-Titiunik)**: Robust bias-corrected

### McCrary Density Test

Tests for manipulation of the running variable at the cutoff. A significant discontinuity in the density suggests sorting.

### Placebo Tests

- **Outcome at placebo cutoffs**: No effect expected at non-cutoff values
- **Pre-treatment covariates**: Should be smooth through the cutoff

See {doc}`/api/rdd` for full API reference.
