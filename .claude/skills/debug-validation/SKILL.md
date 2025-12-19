# Debug Validation

Debug failing validation tests with systematic diagnosis.

## Description
Debug Validation - Validation Debugging Workflow (project)

## Usage
/debug-validation [TEST_PATH_OR_ERROR]

Example: `/debug-validation tests/validation/monte_carlo/test_iv_monte_carlo.py::test_2sls_coverage`

## Prompt

You are debugging a validation failure in this causal inference repository.

### Step 1: Identify the Failure

1. If test path provided, run it:
   ```bash
   pytest {test_path} -v --tb=long
   ```

2. Capture the error output
3. Identify failure type:
   - **Bias failure**: Mean estimate != true value
   - **Coverage failure**: CI coverage outside 93-97%
   - **SE accuracy failure**: SE ratio outside 0.9-1.1
   - **Parity failure**: Python != Julia
   - **Assertion error**: Specific assertion failed

### Step 2: Read TROUBLESHOOTING.md

Check `docs/TROUBLESHOOTING.md` for known issues matching this failure pattern.

### Step 3: Systematic Diagnosis

**For Bias Failures**:
1. Check DGP parameters:
   - Is true effect correctly specified?
   - Is sample size sufficient?
   - Are there numerical precision issues?
2. Check estimator:
   - Read implementation in `src/causal_inference/`
   - Verify formula matches literature
   - Check for off-by-one errors
3. Check assumptions:
   - Is DGP satisfying method assumptions?
   - Any violations (confounding, weak instruments)?

**For Coverage Failures**:
1. Check SE estimation:
   - Which variance estimator? (HC0/HC1/HC2/HC3)
   - Is it appropriate for sample size?
2. Check CI construction:
   - Normal vs t-distribution?
   - Correct critical value?
3. Check for edge cases:
   - Extreme propensity scores?
   - Perfect prediction?

**For Parity Failures**:
1. Compare implementations:
   - Same algorithm?
   - Same numerical precision?
   - Same random seed handling?
2. Check tolerance:
   - Is rtol=1e-10 appropriate?
   - Are there expected differences?

### Step 4: Isolate the Issue

1. Create minimal reproducing case
2. Add diagnostic prints
3. Run single iteration to inspect values
4. Compare intermediate values between languages if parity issue

### Step 5: Check METHODOLOGICAL_CONCERNS.md

Look for documented concerns related to this failure:
- Is this a known limitation?
- Is there a recommended workaround?
- Should test be marked xfail?

### Step 6: Propose Fix

Based on diagnosis:
1. **Code fix**: Describe the change needed
2. **Test fix**: Adjust thresholds or mark xfail
3. **Documentation**: Update concerns if new issue found

## Output Format
```
=== Validation Debug Report ===

## Failure Summary
- Test: {test_path}
- Type: {bias/coverage/SE/parity}
- Error: {error_message}

## Diagnosis
1. Root cause: ...
2. Evidence: ...
3. Related concern: CONCERN-X (if applicable)

## Intermediate Values
{relevant debug output}

## Proposed Fix
- Type: {code/test/doc}
- Location: {file:line}
- Change: {description}

## Verification
Run after fix:
```bash
{command to verify fix}
```
```
