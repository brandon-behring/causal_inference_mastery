# Troubleshooting Guide

Solutions for common issues in causal inference validation and implementation.

---

## Monte Carlo Validation Issues

### "Bias exceeds threshold"

**Symptom**: Monte Carlo test fails with `bias > 0.05` (RCT) or `bias > 0.10` (observational)

**Diagnosis**:
1. Check DGP parameters:
   ```python
   # Is true effect correctly specified?
   true_ate = 2.0  # Verify this matches what estimator should recover
   ```
2. Check sample size: Small n increases variance, may appear as bias
3. Check for numerical issues: Very large/small values can cause precision loss

**Solutions**:
| Cause | Fix |
|-------|-----|
| Wrong true effect in test | Verify DGP `true_ate` matches assertion |
| Sample too small | Increase n (try 500 → 1000) |
| Estimator bug | Compare against known implementation |
| Confounding in DGP | Ensure DGP satisfies method assumptions |

**Debug command**:
```bash
pytest tests/validation/monte_carlo/test_X.py -v --tb=long -x
```

---

### "Coverage outside 93-97% range"

**Symptom**: CI coverage significantly different from nominal 95%

**Diagnosis**:
1. **Under-coverage (< 93%)**: SEs too small or CI formula wrong
2. **Over-coverage (> 97%)**: SEs too large (conservative)

**Solutions**:
| Coverage | Likely Cause | Fix |
|----------|--------------|-----|
| < 90% | Wrong SE formula | Check variance estimator |
| 90-93% | Small sample, need t not z | Use t-distribution critical values |
| 93-97% | Good | Expected range |
| 97-99% | Conservative SE | May be acceptable |
| > 99% | SE way too large | Check for bugs |

**Check variance estimator**:
```python
# Are you using appropriate estimator for sample size?
# n < 250: HC3 recommended
# n >= 250: HC1 or HC0 acceptable
```

---

### "SE accuracy outside 0.9-1.1"

**Symptom**: Ratio of mean(SE) / std(estimates) is off

**Diagnosis**:
- Ratio < 0.9: SEs underestimate true variability
- Ratio > 1.1: SEs overestimate true variability

**Solutions**:
| Ratio | Issue | Fix |
|-------|-------|-----|
| < 0.8 | SE formula wrong | Review analytical SE derivation |
| 0.8-0.9 | Mild underestimation | Consider bootstrap |
| 0.9-1.1 | Good | Expected range |
| 1.1-1.2 | Conservative (OK) | Usually acceptable |
| > 1.2 | Severe overestimation | Check for double-counting |

---

### "Test timeout"

**Symptom**: Monte Carlo test takes too long

**Solutions**:
```bash
# Run with fewer simulations for debugging
pytest tests/validation/monte_carlo/test_X.py -v -x

# Or modify test temporarily
N_SIMULATIONS = 500  # Instead of 5000
```

---

## Cross-Language Parity Issues

### "Python-Julia mismatch > 1e-10"

**Symptom**: Same algorithm produces different results

**Diagnosis**:
1. Check random seeds are handled consistently
2. Check numerical precision (Float64 vs Float32)
3. Check algorithm implementation matches exactly

**Common causes**:
| Issue | Python | Julia | Fix |
|-------|--------|-------|-----|
| Random seed | `np.random.seed(42)` | `Random.seed!(42)` | Set both explicitly |
| Matrix inversion | `np.linalg.inv` | `inv()` | Use same decomposition |
| Default tolerance | `1e-8` | `1e-10` | Align tolerances |

**Debug approach**:
```python
# Print intermediate values in both languages
print(f"X'X = {X.T @ X}")
print(f"X'y = {X.T @ y}")
print(f"beta = {beta}")
```

---

### "Julia test fails but Python passes"

**Diagnosis**:
1. Different DGP random state
2. Different default parameters
3. Julia-specific numerical behavior

**Solutions**:
1. Use shared fixtures in `tests/fixtures/` or `julia/test/fixtures/`
2. Explicitly set all parameters (don't rely on defaults)
3. Check Julia package versions match expected

---

## Adversarial Test Issues

### "Edge case not handled"

**Symptom**: Test with n=1 or extreme values crashes

**Solutions**:
```python
# Add input validation
if len(outcome) < 2:
    raise ValueError(f"Need at least 2 observations, got {len(outcome)}")

if np.any(np.isnan(outcome)):
    raise ValueError("outcome contains NaN values")
```

---

### "NaN propagation"

**Symptom**: NaN appears in results unexpectedly

**Common sources**:
| Operation | Cause | Fix |
|-----------|-------|-----|
| `log(0)` | Zero propensity score | Clip to `[1e-6, 1-1e-6]` |
| `1/0` | Zero variance | Check for constant columns |
| `sqrt(-x)` | Negative variance estimate | Use absolute value or check |

**Debug**:
```python
# Add NaN checks
assert not np.any(np.isnan(result)), f"NaN in result: {result}"
```

---

## Method-Specific Issues

### PropensityScoreError

**Symptom**: "Propensity scores outside valid range"

**Causes**:
- Perfect prediction (separation)
- Insufficient overlap

**Solutions**:
```python
# Clip extreme propensity scores
ps = np.clip(ps, 0.01, 0.99)

# Or use trimming
mask = (ps > 0.1) & (ps < 0.9)
```

---

### WeakInstrumentError

**Symptom**: First-stage F < 10

**Solutions**:
1. Use LIML instead of 2SLS
2. Use Anderson-Rubin confidence sets
3. Find stronger instrument
4. Check `src/causal_inference/iv/weak_iv_robust.py` for robust methods

---

### ParallelTrendsViolation

**Symptom**: Pre-trends test shows divergence

**Solutions**:
1. Verify data: Are pre-periods truly pre-treatment?
2. Consider triple-differences
3. Use bounds (Rambachan & Roth)
4. Document as limitation

---

### ConvergenceError

**Symptom**: Optimization didn't converge

**Solutions**:
```python
# Increase max iterations
result = optimize(func, x0, maxiter=10000)

# Try different starting values
x0 = np.random.randn(p) * 0.1

# Scale variables
X = (X - X.mean()) / X.std()
```

---

## Environment Issues

### "Julia package not found"

**Fix**:
```bash
cd julia
julia --project -e "using Pkg; Pkg.instantiate()"
```

---

### "Python import error"

**Fix**:
```bash
pip install -e ".[dev]"
```

---

### "research-kb MCP not available"

**Diagnosis**:
```bash
# Check if server is configured
cat .mcp.json

# Check if research-kb is running
ls -la /tmp/research_kb_daemon.sock
```

**Fix**: Start the research-kb daemon or check MCP configuration.

---

## Quick Diagnostic Commands

```bash
# Run specific test with full output
pytest tests/path/to/test.py::test_name -v --tb=long

# Run with coverage to find untested code
pytest tests/ --cov=src/causal_inference --cov-report=term-missing

# Check Julia module loads
julia --project -e "using CausalEstimators; println(\"OK\")"

# Run Python type checking
mypy src/causal_inference/

# Check for linting issues
ruff src/ tests/
```

---

## See Also

- `docs/FAILURE_MODES.md` - Comprehensive failure taxonomy
- `docs/METHODOLOGICAL_CONCERNS.md` - Known methodological issues
- `/debug-validation` skill - Systematic debugging workflow
