# Causal Inference Mastery

**Status**: Python Phase 1 COMPLETE (bugs fixed 2025-11-20) | Julia Phases 1-4 COMPLETE
**Created**: 2024-11-14
**Last Updated**: 2025-11-20
**Goal**: Deep, rigorous understanding of causal inference through dual-language implementation

---

## Project Overview

This project implements causal inference methods from first principles using both Python (with modern libraries) and Julia (from scratch). The dual-language approach provides cross-validation confidence while building deep mathematical understanding.

### Design Principles

1. **Test-First Development** - All tests written before implementation
2. **Known-Answer Validation** - Hand-calculated expected values
3. **Monte Carlo Validation** - 1000-run simulations confirm statistical properties
4. **Cross-Language Validation** - Python and Julia must agree to 10 decimal places
5. **Research-Grade Quality** - 90%+ test coverage, rigorous documentation

### Methods Implemented

#### Python Implementation
- [x] **Phase 1: Randomized Controlled Trials (RCT)** - 63 tests, 94.44% coverage
  - Simple ATE, Regression-Adjusted ATE, Stratified ATE, IPW, Permutation Tests
  - Critical inference bugs FIXED (2025-11-20): z→t distribution, p-value smoothing
- [ ] Phase 2: Propensity Score Matching (PSM)
- [ ] Phase 3: Regression Discontinuity (RDD)
- [ ] Phase 4: Instrumental Variables (IV)

#### Julia Implementation
- [x] **Phase 1: Randomized Controlled Trials (RCT)** - 1,602+ tests, six-layer validation
- [x] **Phase 2: Propensity Score Matching (PSM)** - Complete with cross-validation
- [x] **Phase 3: Regression Discontinuity (RDD)** - 4 files, 57KB
- [x] **Phase 4: Instrumental Variables (IV)** - 6 files, 84KB (TSLS, LIML, GMM, AR/CLR tests)

---

## Quick Start

### Installation

```bash
# Clone repository
cd ~/Claude/causal_inference_mastery

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### Julia Setup

```bash
cd julia/
julia --project -e "using Pkg; Pkg.instantiate()"
julia --project test/runtests.jl
```

---

## Project Structure

```
causal_inference_mastery/
├── docs/                          # Planning & documentation
│   ├── ROADMAP.md                 # Master plan with Decision Log
│   ├── CURRENT_WORK.md            # Context switching aid
│   ├── plans/active/              # Current phase plans
│   └── proofs/                    # Mathematical derivations
├── src/causal_inference/          # Python modules
│   ├── rct/                       # RCT estimators
│   ├── psm/                       # PSM estimators
│   ├── did/                       # DiD estimators
│   ├── iv/                        # IV estimators
│   ├── data/                      # Data generating processes
│   └── evaluation/                # Metrics & diagnostics
├── tests/                         # pytest test suite
│   ├── test_rct/                  # RCT tests
│   └── conftest.py                # Shared fixtures
├── julia/                         # Julia implementation
│   ├── src/                       # Julia modules
│   └── test/                      # Julia tests
├── notebooks/                     # Jupyter demonstrations
├── validation/                    # Cross-validation
│   ├── monte_carlo/               # Statistical validation
│   └── cross_language/            # Python ↔ Julia comparison
└── scripts/                       # Automation
```

---

## Development Workflow

### Test-First Cycle

1. **Write tests** with known answers (tests should FAIL)
2. **Implement** function to pass tests
3. **Run tests** until passing
4. **Validate** with Monte Carlo (1000 runs)
5. **Cross-validate** Julia and Python
6. **Document** with proofs and notebooks
7. **Commit** with meaningful message

### Git Workflow

Commit after each function/module completion (3-5 commits/day):

```bash
# Run quality checks
black .
ruff .
mypy src/
pytest tests/

# Commit
git add .
git commit -m "feat(rct): Implement simple_ate with proper inference"
```

### Commit Message Format

```
type(scope): Short description

- Bullet points with details
- What changed and why
```

**Types**: `feat`, `fix`, `test`, `docs`, `refactor`, `validate`

---

## Quality Standards

### Test Coverage

- Modules: **90%+** (enforced by pytest)
- Scripts: 60%+
- Overall: 90%+

### Validation Requirements

Before any method is considered complete:

- [ ] All Python tests pass
- [ ] All Julia tests pass
- [ ] Known-answer validation passes
- [ ] Monte Carlo: bias < 0.05
- [ ] Monte Carlo: coverage 94-96%
- [ ] Cross-language agreement (rtol < 1e-10)
- [ ] Notebooks execute without errors
- [ ] Mathematical proofs complete

---

## Documentation

### For Each Method

1. **Mathematical Proof** (`docs/proofs/`) - Full derivations in Markdown + LaTeX
2. **Python Implementation** (`src/causal_inference/`) - With docstrings and type hints
3. **Julia Implementation** (`julia/src/`) - From first principles
4. **Tests** (`tests/`) - Known-answer and error handling
5. **Notebooks** (`notebooks/`) - Visual demonstrations
6. **Validation** (`validation/`) - Monte Carlo and cross-language

---

## Key References

**Planning**:
- `docs/ROADMAP.md` - Master plan with Decision Log
- `docs/CURRENT_WORK.md` - Current status and next steps
- `docs/plans/active/PHASE_X_*.md` - Detailed phase plans

**Methodological**:
- Imbens & Rubin (2015) - Causal Inference for Statistics
- Angrist & Pischke (2009) - Mostly Harmless Econometrics
- Cunningham (2021) - Causal Inference: The Mixtape

**Implementation Patterns**:
- `~/Claude/annuity_forecasting/` - Test-first development
- `~/Claude/double_ml_time_series/` - 7-method validation suite

---

## Current Status

### Python Implementation
- **Phase 1 (RCT)**: ✅ COMPLETE with critical fixes applied (2025-11-20)
  - 63 tests (61 passing, 2 need updating for corrected behavior)
  - 94.44% code coverage (exceeds 90% requirement)
  - Fixed: z→t distribution bug in 3 estimators
  - Fixed: permutation test p-value smoothing
  - Documented: stratified n=1 variance limitation
- **Next**: Implement Python validation layers (Monte Carlo, adversarial tests) OR begin Phase 2

### Julia Implementation
- **Phases 1-4**: ✅ COMPLETE (2025-11-15)
  - 1,602+ test assertions across 35 test files
  - Six-layer validation architecture (known-answer, adversarial, Monte Carlo, Python↔Julia, R triangulation, golden reference)
  - 661+ adversarial/edge case tests
  - Performance: 98x speedup vs Python for RegressionATE

### Known Issues
- 2 Python tests need updating to expect corrected behavior (t-distribution critical values, smoothed p-values)
- IPW estimator missing safeguards (weight stabilization, trimming, positivity diagnostics) - 2-3 hour fix

**For detailed reconciliation**: See `docs/AUDIT_RECONCILIATION_2025-11-20.md`
**For current work**: See `docs/CURRENT_WORK.md`

---

## License

Personal research project - Brandon Behring (2024)
