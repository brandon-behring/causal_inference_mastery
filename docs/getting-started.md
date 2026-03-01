# Getting Started

## Installation

```bash
pip install -e .
```

### Optional Extras

```bash
# Bayesian methods (PyMC, ArviZ)
pip install -e ".[bayesian]"

# R triangulation (rpy2)
pip install -e ".[r-triangulation]"

# Development tools
pip install -e ".[dev]"
```

## Quick Example: Difference-in-Differences

```python
import numpy as np
import pandas as pd
from causal_inference.did import DiDEstimator

# Generate data
np.random.seed(42)
n = 200
data = pd.DataFrame({
    "y": np.random.normal(0, 1, n),
    "treated": np.random.binomial(1, 0.5, n),
    "post": np.random.binomial(1, 0.5, n),
})
data["y"] += 2.0 * data["treated"] * data["post"]  # True ATE = 2.0

# Estimate
estimator = DiDEstimator()
result = estimator.fit(data, outcome="y", treatment="treated", time="post")
print(f"ATE: {result.ate:.2f} (SE: {result.se:.3f}, p: {result.p_value:.4f})")
```

## Quick Example: Instrumental Variables

```python
from causal_inference.iv import TwoStageLeastSquares

iv = TwoStageLeastSquares()
result = iv.fit(data, outcome="y", treatment="x", instruments=["z1", "z2"])
print(f"LATE: {result.coefficient:.3f}")
print(f"First-stage F: {result.first_stage_f:.1f}")
```

## Choosing a Method

Use the {doc}`method-selection` guide to find the right estimator for your research question. Key decision factors:

1. **Is treatment randomized?** → RCT estimators
2. **Can you assume selection on observables?** → IPW, DR, matching
3. **Is there a natural experiment?** → DiD, RDD, IV, SCM
4. **Do you need heterogeneous effects?** → CATE methods (causal forest, DML)
5. **Sensitivity analysis required?** → E-values, Rosenbaum bounds

## Next Steps

- {doc}`method-selection` — Decision tree for choosing methods
- {doc}`validation` — How every estimator is validated
- {doc}`user-guide/did` — DiD tutorial (most common method)
- {doc}`api/index` — Full API reference
