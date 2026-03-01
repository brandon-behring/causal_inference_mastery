# Troubleshooting

Common issues and solutions when using causal-inference-mastery.

## Installation

### PyMC/ArviZ not found

```bash
# Bayesian methods require the [bayesian] extra
pip install -e ".[bayesian]"
```

### rpy2 errors

R triangulation tests require R installed and the `rpy2` package:

```bash
# Install R first, then:
pip install -e ".[r-triangulation]"
```

## Runtime Issues

### "Singular matrix" error in IV estimation

**Cause**: Perfect collinearity among instruments or covariates.

**Solution**: Check for redundant variables. Reduce instrument set. Verify first-stage F > 10.

### DiD: "Parallel trends assumption may be violated"

**Cause**: Pre-trend test detected significant divergence before treatment.

**Solution**:
1. Plot raw trends visually
2. Consider Callaway-Sant'Anna (more robust to heterogeneity)
3. Add time-varying controls
4. Report sensitivity to parallel trends violations

### Propensity scores near 0 or 1

**Cause**: Extreme propensity scores create unstable IPW weights.

**Solution**:
1. Trim propensity scores (e.g., restrict to [0.1, 0.9])
2. Use doubly robust estimator (more stable)
3. Check overlap assumption — may need different identification strategy

### Monte Carlo tests are slow

```bash
# Skip slow tests during development
pytest -m "not slow and not monte_carlo and not mcmc"

# Run only fast tests
pytest -m "not slow" -x
```

### Cross-language tests fail

**Cause**: Julia environment not configured or package versions differ.

**Solution**:
1. Ensure Julia is installed and in PATH
2. Run `julia -e 'using Pkg; Pkg.instantiate()'` in the `julia/` directory
3. Check that Python and Julia use the same test data

## Common Mistakes

### Using TWFE with staggered adoption

TWFE produces biased estimates when treatment timing varies and effects are heterogeneous. Use Callaway-Sant'Anna or Sun-Abraham instead.

### Ignoring weak instruments

Always report first-stage F-statistic. If F < 10, use LIML or Fuller instead of 2SLS, and report Anderson-Rubin confidence sets.

### Not checking overlap

Before using IPW or matching, verify that treated and control groups have overlapping covariate distributions. Non-overlap invalidates the estimates.

### Interpreting Granger causality as true causality

Granger causality tests predictive precedence, not structural causation. It can be confounded by omitted variables or common causes.
