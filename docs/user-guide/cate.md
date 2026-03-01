# Conditional Average Treatment Effects

Estimate heterogeneous treatment effects — how does the treatment effect vary across subpopulations?

## Methods

### Meta-Learners

| Learner | Approach | Best When |
|---------|----------|-----------|
| **S-learner** | Single model: $E[Y|X,T]$ | Simple, large n |
| **T-learner** | Separate models per group | Strong heterogeneity |
| **X-learner** | Cross-fitting with propensity | Unequal group sizes |
| **R-learner** | Robinson transformation | Doubly robust, efficient |

### Causal Forest (Generalized Random Forest)

Non-parametric CATE estimation. Splits on covariates that maximize treatment effect heterogeneity.

```python
from causal_inference.cate import CausalForest

cf = CausalForest(n_trees=2000)
result = cf.fit(data, outcome="y", treatment="treated",
                covariates=["x1", "x2", "x3"])

# Individual treatment effects
ite = result.predict(new_data)
print(f"CATE range: [{ite.min():.2f}, {ite.max():.2f}]")
```

### Double Machine Learning (DML)

Semiparametric estimator using cross-fitting to avoid regularization bias:

1. Estimate nuisance functions (propensity, outcome) via ML
2. Construct orthogonal score
3. Estimate CATE from residualized data

```python
from causal_inference.cate import DoubleML

dml = DoubleML(model_y="random_forest", model_t="logistic")
result = dml.fit(data, outcome="y", treatment="treated",
                 covariates=["x1", "x2", "x3"])
```

See {doc}`/api/cate` for full API reference.
