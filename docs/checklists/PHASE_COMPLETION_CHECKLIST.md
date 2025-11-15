# Phase Completion Checklist

**Purpose**: Quick-reference checklist for marking a phase complete
**Full Standards**: See `docs/standards/PHASE_COMPLETION_STANDARDS.md` for detailed requirements
**Created**: 2024-11-14

---

## Before Starting Phase

- [ ] **Read METHODOLOGICAL_CONCERNS.md** for phase-specific concerns
- [ ] **Review ROADMAP.md** validation requirements for this phase
- [ ] **Check test templates** in `templates/testing/` for boilerplate

---

## Implementation Checklist

### Code Quality

- [ ] **SciML Design Pattern** applied:
  - [ ] Problem type (immutable struct with validation)
  - [ ] Estimator type (zero-field struct)
  - [ ] Solution type (immutable results)
  - [ ] `solve(problem, estimator)` interface

- [ ] **File Organization** follows Phase 1 structure:
  - [ ] `src/estimators/[method]/[estimator].jl`
  - [ ] `test/estimators/[method]/test_[estimator].jl`
  - [ ] `test/validation/test_[validation_type].jl`
  - [ ] `validation/[language]_scripts/validate_[method].[ext]`

- [ ] **Naming Conventions** consistent:
  - [ ] Estimators: `[Method]ATE`, `[Method]ATT` (e.g., SimpleATE, IPWATE)
  - [ ] Problems: `[Method]Problem` (e.g., RCTProblem, DiDProblem)
  - [ ] Solutions: `[Method]Solution`
  - [ ] Test files: `test_[estimator_name].jl` (lowercase, underscores)

- [ ] **Code formatted** with JuliaFormatter (SciML style)

---

## Documentation Checklist

### Docstrings (8 Required Sections)

For EVERY estimator:

- [ ] **One-line summary** (< 80 chars)
- [ ] **Mathematical Foundation** section with LaTeX equations
- [ ] **Variance Estimation** section:
  - [ ] Which estimator used (HC3, Neyman, Horvitz-Thompson, etc.)
  - [ ] Why that estimator chosen (3+ reasons)
  - [ ] Assumptions listed
  - [ ] Alternatives considered and rejected
  - [ ] 2+ academic citations
- [ ] **Usage** section with runnable examples
- [ ] **Requirements** section listing assumptions
- [ ] **Benefits** section explaining when to use
- [ ] **Limitations** section explaining when NOT to use
- [ ] **References** section with 3+ academic citations

### Error Handling

- [ ] **All validation errors** use explicit format:
  ```julia
  throw(ArgumentError(
      "CRITICAL ERROR: [What went wrong]\n" *
      "Function: [function_name]\n" *
      "[Explanation]\n" *
      "[What to do]"
  ))
  ```

- [ ] **Required error checks** implemented:
  - [ ] No NaN/Inf in inputs
  - [ ] Treatment variation exists
  - [ ] Sufficient sample size
  - [ ] Required fields populated
  - [ ] Valid parameter ranges

---

## Validation Checklist (Six-Layer Architecture)

### Layer 1: Known-Answer Tests

- [ ] **At least 2 tests per estimator**
- [ ] **Analytically verifiable** results (zero effect, constant effect, etc.)
- [ ] **Deterministic** (use `Random.seed!()`)
- [ ] Tests **fail when estimator wrong** (negative test)

**Template**: `templates/testing/known_answer_test_template.jl`

### Layer 2: Adversarial Tests

- [ ] **Minimum 10 tests per estimator**
- [ ] **Edge cases** covered (n=1, all treated/control)
- [ ] **Numerical stability** tested (NaN, Inf, extreme values)
- [ ] **Invalid inputs** handled (mismatched lengths, empty arrays)
- [ ] **Boundary conditions** tested (zero variance, perfect collinearity)
- [ ] **Estimator-specific** pathologies tested

**Template**: `templates/testing/adversarial_test_template.jl`

### Layer 3: Monte Carlo Ground Truth

- [ ] **DGP with KNOWN treatment effect** (τ = 2.0 or specified)
- [ ] **N=10,000 simulations** for reliable estimates
- [ ] **Bias < 0.05** verified
- [ ] **Coverage 94-96%** for α=0.05
- [ ] **SE accuracy < 10%** (mean SE within 10% of empirical SD)
- [ ] **Multi-alpha coverage** tested (α ∈ {0.01, 0.05, 0.10})
- [ ] **Heteroskedasticity** tested (σ²₁ ≠ σ²₀)
- [ ] **Large sample** consistency (n=10,000, bias ≈ 0)

**Template**: `templates/testing/monte_carlo_validation_template.jl`

### Layer 4: Python-Julia Cross-Validation

- [ ] **Python implementation** exists (or reference library)
- [ ] **Estimate agreement**: |Julia - Python| < 0.001
- [ ] **SE agreement**: |Julia_SE - Python_SE| / Python_SE < 0.05
- [ ] **CI overlap** verified

**Reference**: `test/validation/test_python_parity.jl`

### Layer 5: R Triangulation

- [ ] **R implementation** created (independent from Python/Julia)
- [ ] **Estimate agreement**: |Julia - R| < 0.001
- [ ] **SE agreement**: |Julia_SE - R_SE| / R_SE < 0.05
- [ ] **Graceful degradation** when R unavailable (warning, not failure)

**Reference**: `validation/r_scripts/validate_rct.R`

### Layer 6: Golden Reference Tests

- [ ] **Published dataset** identified (LaLonde, Lee 2008, etc.)
- [ ] **Authoritative result** from paper/textbook
- [ ] **Result reproduced** within reasonable tolerance
- [ ] **Discrepancies** understood and documented

**Note**: Dataset-specific, implement per phase

---

## Test Coverage Checklist

- [ ] **Overall coverage >80%** (run coverage report)
- [ ] **All tests passing** (no skips, no warnings)
- [ ] **Tests deterministic** (seeded random, no flaky tests)
- [ ] **Test speed reasonable** (<1s per unit test, <10s per validation test)
- [ ] **Descriptive test names** (not `test1`, `test2`)
- [ ] **Independent tests** (no shared state between tests)

**Command**:
```bash
julia --project=. -e 'using Pkg; Pkg.test(coverage=true)'
```

---

## Performance Checklist

- [ ] **Benchmarks established** for all estimators
- [ ] **Execution time** documented for typical problem sizes
- [ ] **Memory allocation** patterns understood
- [ ] **Scaling behavior** documented (how time grows with n)
- [ ] **No performance regressions** vs previous phases

**Benchmark file**: `benchmark/benchmark_[method].jl`
**Documentation**: `benchmark/README.md`

**Command**:
```bash
julia --project=. benchmark/run_benchmarks.jl
```

---

## Phase-Specific Methodological Checklist

### Phase 2: Propensity Score Methods
- [ ] Bootstrap SE methods documented (pairs, block, wild)
- [ ] Abadie-Imbens correction for matching WITH replacement
- [ ] Balance diagnostics (SMD < 0.1 on ALL covariates)
- [ ] Love plots before/after matching

### Phase 3: Difference-in-Differences
- [ ] Goodman-Bacon decomposition implemented
- [ ] TWFE bias demonstrated with staggered adoption
- [ ] Modern methods (Callaway-Sant'Anna, Sun-Abraham, Borusyak)
- [ ] Parallel trends testing

### Phase 4: Instrumental Variables
- [ ] First-stage F-statistic computed
- [ ] Stock-Yogo critical values used
- [ ] Anderson-Rubin CIs for weak instruments
- [ ] LATE vs ATE distinction explained

### Phase 5: Regression Discontinuity
- [ ] McCrary density test implemented
- [ ] Bandwidth sensitivity analysis (0.5×, 1×, 2× optimal)
- [ ] Manipulation detection verified
- [ ] Optimal bandwidth selection (IK, CCT)

### Phase 8: CATE & Advanced Methods
- [ ] Honesty verification for causal forests
- [ ] Cross-fitting verification for Double ML (K ∈ {2, 5, 10})
- [ ] Heterogeneity simulation (known τ(x))
- [ ] Meta-learner comparison (S, T, X, R)

---

## Final Checks

### Documentation Updates

- [ ] **ROADMAP.md** updated:
  - [ ] Phase marked COMPLETE with actual deliverables
  - [ ] Metrics updated
  - [ ] Decision log entry added

- [ ] **METHODOLOGICAL_CONCERNS.md** updated:
  - [ ] Phase audit summary added
  - [ ] Any new concerns documented

- [ ] **README.md** includes usage examples for new estimators

### Manual Review

- [ ] **Code review** of all estimator implementations
- [ ] **Comparison** with authoritative references (textbooks, papers)
- [ ] **Cross-validation** with other software (R, Python, Stata)
- [ ] **Interview prep**: Can you explain all methodological choices?

### Governance Tests

- [ ] Run governance check (if available):
  ```bash
  julia --project=. test/governance/check_phase_standards.jl --phase=[N]
  ```
- [ ] Address ALL failures
- [ ] Get peer review (if available)

---

## Sign-Off

**Phase Number**: _____
**Phase Name**: _____________________________
**Completion Date**: _____-_____-_____
**Completed By**: _____________________________

**Audit Grade** (self-assessment):
- Validation Architecture: ____ / 6 layers complete
- Adversarial Tests: ____ tests (minimum 10 per estimator)
- Monte Carlo: Bias ____, Coverage ____, SE accuracy ____
- Test Coverage: ____% (target: >80%)
- Documentation: All estimators have 8 sections? [ ] Yes / [ ] No

**Overall Grade**: _____ (A+/A/B/C/F)

**Notes**:
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________

**Ready for Next Phase?**: [ ] Yes / [ ] No (if No, explain above)

---

## Quick Commands

```bash
# Run all tests with coverage
julia --project=. -e 'using Pkg; Pkg.test(coverage=true)'

# Run benchmarks
julia --project=. benchmark/run_benchmarks.jl

# Format code
julia --project=. scripts/format_code.jl  # if available

# Check governance
julia --project=. test/governance/check_phase_standards.jl --phase=[N]
```

---

**Last Updated**: 2024-11-14
**Next Review**: After Phase 2 completion
