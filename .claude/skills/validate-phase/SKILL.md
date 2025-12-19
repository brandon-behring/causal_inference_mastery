# Validate Phase

Run the 6-layer validation checklist for a causal inference method.

## Description
Validate Phase - Run 6-Layer Validation Checklist (project)

## Usage
/validate-phase [METHOD]

Example: `/validate-phase IV` or `/validate-phase DiD`

## Prompt

You are validating the **{METHOD}** implementation in this causal inference repository.

Run the complete 6-layer validation checklist:

### Layer 1: Known-Answer Tests
1. Find known-answer tests: `tests/test_{method}/` or `julia/test/{method}/`
2. Run them: `pytest tests/test_{method}/ -v` or `julia --project test/{method}/runtests.jl`
3. Verify: 100% pass required

### Layer 2: Adversarial Tests
1. Find adversarial tests: `tests/validation/adversarial/test_{method}_adversarial.py`
2. Run them: `pytest tests/validation/adversarial/test_{method}_* -v`
3. Check edge cases: n=1, NaN inputs, perfect separation, extreme values

### Layer 3: Monte Carlo Validation
1. Find MC tests: `tests/validation/monte_carlo/test_{method}_*.py`
2. Run them: `pytest -m monte_carlo tests/validation/monte_carlo/ -k {method} -v`
3. Verify targets:
   - Bias < 0.05 (RCT) or < 0.10 (observational)
   - Coverage: 93-97%
   - SE accuracy: < 10-15%

### Layer 4: Cross-Language Parity
1. Find parity tests: `tests/validation/cross_language/test_python_julia_{method}.py`
2. Run them: `pytest tests/validation/cross_language/ -k {method} -v`
3. Verify: Python-Julia agreement to rtol < 1e-10

### Layer 5: Golden Reference (if applicable)
1. Check `tests/golden_results/` for frozen reference values
2. Verify no regression from known-good values

### Layer 6: Methodological Audit
1. Read `docs/METHODOLOGICAL_CONCERNS.md`
2. Find concerns related to {METHOD}
3. Verify each concern is addressed in implementation

## Output Format
Report results as:
```
=== {METHOD} Validation Report ===

Layer 1 (Known-Answer):    [PASS/FAIL] X/Y tests
Layer 2 (Adversarial):     [PASS/FAIL] X/Y tests
Layer 3 (Monte Carlo):     [PASS/FAIL] Bias: X.XX, Coverage: XX%
Layer 4 (Cross-Language):  [PASS/FAIL] X/Y parity tests
Layer 5 (Golden Ref):      [PASS/FAIL/N/A]
Layer 6 (Methodology):     [PASS/FAIL] X concerns addressed

Overall: [VALIDATED / NEEDS WORK]
```
