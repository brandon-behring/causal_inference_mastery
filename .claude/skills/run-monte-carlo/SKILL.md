# Run Monte Carlo

Execute Monte Carlo validation with detailed analysis.

## Description
Run Monte Carlo - MC Validation Execution (project)

## Usage
/run-monte-carlo [METHOD] [--quick]

Example: `/run-monte-carlo IV` or `/run-monte-carlo DiD --quick`

## Prompt

You are running Monte Carlo validation for **{METHOD}**.

### Step 1: Locate Tests
Find Monte Carlo tests:
```bash
# Python
ls tests/validation/monte_carlo/test_*{method}*.py

# Julia
ls julia/test/*/test_*montecarlo*.jl
```

### Step 2: Run Validation

**Standard run** (5000 simulations):
```bash
pytest -m monte_carlo tests/validation/monte_carlo/ -k {method} -v --tb=short
```

**Quick run** (if --quick flag, use 500 simulations for debugging):
```bash
pytest -m monte_carlo tests/validation/monte_carlo/ -k {method} -v --tb=short -x
```

**Julia**:
```bash
cd julia && julia --project test/runtests.jl {method}
```

### Step 3: Analyze Results

For each test, extract and report:
1. **Bias**: Mean estimate - true value
   - Target: < 0.05 (RCT), < 0.10 (observational)
2. **Coverage**: Proportion of CIs containing true value
   - Target: 93-97% (nominal 95%)
3. **SE Accuracy**: Mean SE / SD of estimates
   - Target: 0.90-1.10 (within 10%)
4. **RMSE**: Root mean squared error

### Step 4: Diagnose Failures

If bias > threshold:
- Check DGP parameters (effect size, sample size)
- Verify estimator formula
- Check for numerical issues

If coverage outside range:
- Check SE formula
- Verify CI construction (z vs t)
- Check for finite-sample corrections needed

If SE accuracy off:
- Compare analytical vs bootstrap SE
- Check variance estimator choice (HC0/HC1/HC2/HC3)

### Step 5: Compare Languages

If both Python and Julia tests exist:
```bash
# Run both and compare
pytest tests/validation/monte_carlo/ -k {method} -v
julia --project julia/test/runtests.jl {method}
```

Verify results are within tolerance (differences < 0.02 for bias, < 2% for coverage).

## Output Format
```
=== {METHOD} Monte Carlo Results ===

## Python Results
| Estimator | Bias | Coverage | SE Accuracy | RMSE | Status |
|-----------|------|----------|-------------|------|--------|
| ... | ... | ... | ... | ... | PASS/FAIL |

## Julia Results
| Estimator | Bias | Coverage | SE Accuracy | RMSE | Status |
|-----------|------|----------|-------------|------|--------|
| ... | ... | ... | ... | ... | PASS/FAIL |

## Cross-Language Comparison
- Bias difference: X.XXX (threshold: 0.02)
- Coverage difference: X.X% (threshold: 2%)

## Diagnosis (if failures)
- Issue: ...
- Likely cause: ...
- Suggested fix: ...
```
