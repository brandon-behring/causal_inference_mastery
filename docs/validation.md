# 6-Layer Validation Architecture

Every estimator in causal-inference-mastery is validated through a rigorous 6-layer testing architecture.

## Layer 1: Known-Answer Validation

Hand-calculated results for simple cases. The estimator must reproduce exact values.

```python
# Example: DiD 2x2 with known ATE = 2.0
result = estimator.fit(known_data)
assert abs(result.ate - 2.0) < 1e-10
```

**Coverage**: All 26 method families.

## Layer 2: Adversarial Testing

Edge cases and boundary conditions:

- Single observation (n=1)
- All treated / all control
- NaN and missing values
- Perfect collinearity
- Zero variance
- Extreme outliers

**Purpose**: Ensure graceful failure with informative error messages, not silent wrong answers.

## Layer 3: Monte Carlo Validation

Statistical validation via repeated simulation (500–5000 runs):

- **Bias**: Mean estimate should converge to true parameter
- **Coverage**: 95% CI should contain true parameter ~95% of the time
- **Type I error**: Rejection rate under null should be ~5% at α=0.05
- **Power**: Rejection rate under alternative should increase with n

```python
@pytest.mark.monte_carlo
def test_did_coverage():
    """95% CI should contain true ATE ~95% of the time."""
    hits = sum(run_simulation() for _ in range(1000))
    assert 0.92 < hits / 1000 < 0.98  # Allow Monte Carlo noise
```

## Layer 4: Cross-Language Validation

Python and Julia implementations must agree to 10 decimal places:

$$|\hat{\theta}_{Python} - \hat{\theta}_{Julia}| < 10^{-10}$$

**Coverage**: 25/25 method families validated (as of Session 184).

## Layer 5: R Triangulation

Compare against established R packages (e.g., `fixest`, `rdrobust`, `AER`, `grf`):

```python
@pytest.mark.cross_language
def test_iv_matches_r():
    """2SLS should match R's AER::ivreg to 6 decimals."""
    python_result = TwoStageLeastSquares().fit(data)
    r_result = run_r_ivreg(data)
    assert abs(python_result.coefficient - r_result) < 1e-6
```

**Coverage**: 25/25 method families validated.

## Layer 6: Golden Reference

Frozen JSON test results that never change. Any change in output triggers a test failure.

```python
def test_golden_reference():
    result = estimator.fit(canonical_data)
    expected = load_golden("iv_2sls_canonical.json")
    assert result.to_dict() == expected
```

**Coverage**: 11 frozen golden tests.

## Validation Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Test functions | — | 3,854 |
| Julia assertions | — | 5,121 |
| Coverage | ≥ 90% | 90%+ |
| Known bugs | 0 | 0 |
| Cross-language parity | 25/25 | 25/25 |
| R triangulation | 25/25 | 25/25 |

## Test Markers

```bash
# Run only fast tests
pytest -m "not slow and not monte_carlo"

# Run Monte Carlo validation
pytest -m monte_carlo

# Run cross-language tests (requires Julia)
pytest -m cross_language

# Run R triangulation (requires rpy2)
pytest -m "not mcmc" --ignore=tests/validation/r_triangulation
```
