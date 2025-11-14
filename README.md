# Causal Inference Mastery

**Status**: Phase 1 - RCT Foundation (IN PROGRESS)
**Created**: 2024-11-14
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

- [ ] Phase 1: Randomized Controlled Trials (RCT)
- [ ] Phase 2: Propensity Score Matching (PSM)
- [ ] Phase 3: Difference-in-Differences (DiD)
- [ ] Phase 4: Instrumental Variables (IV)
- [ ] Phase 5: Regression Discontinuity (RDD)
- [ ] Phase 6: Sensitivity Analysis
- [ ] Phase 7: Matching Methods
- [ ] Phase 8: CATE & Advanced Methods

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

**Phase 1 Progress**: Infrastructure setup complete, beginning test development

See `docs/CURRENT_WORK.md` for detailed status.

---

## License

Personal research project - Brandon Behring (2024)
