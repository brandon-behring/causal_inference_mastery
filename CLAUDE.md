# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

---

## Shared Foundation (Hub Reference)

**This project uses shared patterns from lever_of_archimedes**:

See: `~/Claude/lever_of_archimedes/patterns/` for:
- `testing.md` - 6-layer validation architecture
- `sessions.md` - Session workflow (CURRENT_WORK.md, ROADMAP.md)
- `git.md` - Commit format and workflow
- `style/python_style.yaml` - Black 100-char, strict mypy
- `style/julia_style.yaml` - SciML formatting, 92-char lines

**Core principles** (from hub):
1. NEVER fail silently - explicit errors always
2. Simplicity over complexity - 20-50 line functions
3. Immutability by default
4. Fail fast with diagnostics

---

## Project Overview

Dual-language causal inference implementation for deep methodological understanding through rigorous, research-grade implementations.

- **Python**: Modern libraries (pyfixest, linearmodels, econml, dowhy)
- **Julia**: From-scratch implementations for mathematical rigor
- **Goal**: Cross-language validation to 10 decimal places

### Current Status (Verified Session 83)
- **Python**: Phases 1-11 COMPLETE (21,760 lines across 14 method families)
- **Julia**: Phases 1-11 COMPLETE (22,840 lines with cross-language parity)
- **Tests**: 7,178+ (1,778 Python functions, 5,400 Julia assertions)
- **Pass Rate**: 99.4% (Python), 99.6% (Julia)
- **Known Bugs**: 6 HIGH-severity documented in `docs/KNOWN_BUGS.md`

---

## Quick Reference Commands

```bash
# Python tests (full suite with coverage)
pytest tests/ --cov=src/causal_inference --cov-report=term-missing --cov-fail-under=90

# Python tests (fast, skip slow)
pytest tests/ -m "not slow"

# Python tests (single module)
pytest tests/test_rct/ -v

# Julia tests
cd julia && julia --project test/runtests.jl

# Julia tests (specific file)
julia --project -e "using Pkg; Pkg.test()" -- test/did/test_classic_did.jl

# Code quality
black src/ tests/
ruff src/ tests/
mypy src/

# Pre-commit hooks
pre-commit run --all-files
```

---

## Project Structure

```
causal_inference_mastery/
├── src/causal_inference/           # Python modules (21,760 lines)
│   ├── rct/                        # 5 RCT estimators (1,332 lines)
│   ├── observational/              # IPW, DR, outcome regression (1,485 lines)
│   ├── psm/                        # Propensity score matching (1,871 lines)
│   ├── did/                        # Difference-in-differences (3,077 lines)
│   ├── iv/                         # Instrumental variables (3,132 lines)
│   ├── rdd/                        # Regression discontinuity (2,356 lines)
│   ├── scm/                        # Synthetic control (2,169 lines)
│   ├── cate/                       # Treatment effect heterogeneity (1,542 lines)
│   ├── sensitivity/                # Sensitivity analysis (916 lines)
│   ├── rkd/                        # Regression kink design (2,086 lines)
│   └── bunching/                   # Bunching estimators (933 lines)
├── julia/src/                      # Julia modules (22,840 lines)
│   ├── did/, iv/, rdd/             # Core method implementations
│   ├── scm/, cate/, sensitivity/   # Advanced methods
│   ├── rkd/, bunching/             # Additional estimators
│   └── CausalEstimators.jl         # Main module
├── tests/                          # Python test suite (1,778 test functions)
│   ├── test_rct/, test_psm/        # Method-specific tests
│   └── validation/                 # 4-layer validation
│       ├── monte_carlo/            # Statistical simulations
│       ├── adversarial/            # Edge case tests
│       ├── cross_language/         # Python ↔ Julia parity
│       └── audit/                  # Bug exposure tests
└── docs/                           # Documentation (60+ files)
    ├── KNOWN_BUGS.md               # Tracked correctness issues
    ├── GAP_ANALYSIS.md             # Missing methods roadmap
    ├── plans/active/               # Current session plans
    └── plans/implemented/          # Completed sessions
```

---

## Validation Architecture (6 Layers)

| Layer | Purpose | Implementation | Status |
|-------|---------|----------------|--------|
| 1 | Known-Answer | Hand-calculated expected values | ✅ VERIFIED |
| 2 | Adversarial | Edge cases, boundary conditions | ⚠️ 7 xfails |
| 3 | Monte Carlo | 5,000-25,000 run simulations | ✅ VERIFIED |
| 4 | Cross-Language | Python ↔ Julia parity | ⏸️ Conditional |
| 5 | R Triangulation | External reference | ❌ NOT IMPLEMENTED |
| 6 | Golden Reference | 111KB JSON frozen results | ⏸️ EXISTS, UNUSED |

**Legend**: ✅ = Passing, ⚠️ = Partial, ⏸️ = Conditional/Unused, ❌ = Not implemented

### Monte Carlo Validation Standards

| Method Type | Bias Target | Coverage Target | SE Accuracy |
|-------------|-------------|-----------------|-------------|
| RCT (unconfounded) | < 0.05 | 93-97% | < 10% |
| Observational (confounded) | < 0.10 | 93-97.5% | < 15% |
| PSM | < 0.30 (expected) | ≥ 95% | < 150% (conservative) |

---

## Quality Standards

- **Test Coverage**: 90%+ (enforced by pytest)
- **Test-First Development**: MANDATORY - tests before implementation
- **Pre-commit Hooks**: Black, Ruff, Mypy, large commit warnings
- **Function Length**: 20-50 lines target

---

## Documentation Hierarchy

| Document | Purpose |
|----------|---------|
| `CURRENT_WORK.md` | 30-second context resume (session tracking) |
| `docs/ROADMAP.md` | Master plan, phase tracking |
| `docs/KNOWN_BUGS.md` | **6 HIGH-severity bugs tracked** |
| `docs/GAP_ANALYSIS.md` | Missing methods roadmap (Phase 12+) |
| `docs/AUDIT_RESULTS.md` | Session 83 comprehensive audit |
| `docs/METRICS_VERIFIED.md` | Verified line/test counts |
| `docs/METHODOLOGICAL_CONCERNS.md` | 13 tracked concerns (CRITICAL → MEDIUM) |
| `docs/METHOD_SELECTION.md` | Decision tree for method selection |
| `docs/TROUBLESHOOTING.md` | Debug guide for validation issues |
| `docs/GLOSSARY.md` | Terminology reference |
| `docs/FAILURE_MODES.md` | Method failure taxonomy |
| `docs/plans/active/` | In-progress phase plans |
| `docs/plans/implemented/` | Completed phase plans |

---

## Context Budget Awareness

| Document | Size | When to Load |
|----------|------|--------------|
| `CLAUDE.md` | ~25KB | Always (auto-loaded) |
| `CURRENT_WORK.md` | ~8KB | Session start |
| `docs/ROADMAP.md` | ~35KB | Phase planning |
| `docs/METHODOLOGICAL_CONCERNS.md` | ~16KB | Before implementing method |
| `docs/METHOD_SELECTION.md` | ~8KB | Choosing methods |
| `docs/TROUBLESHOOTING.md` | ~5KB | When debugging |
| `docs/FAILURE_MODES.md` | ~8KB | When tests fail |

**Total context for full load**: ~105KB (~26K tokens)
**Recommended working set**: CLAUDE.md + CURRENT_WORK.md (~33KB)

---

## MCP Tools Available

This project connects to `research-kb` for causal inference literature access:

| Tool | Purpose | Example |
|------|---------|---------|
| `research_kb_search` | Search papers/textbooks | `"weak instrument detection"` |
| `research_kb_get_concept` | Get method definition | `"difference-in-differences"` |
| `research_kb_graph_neighborhood` | Explore concept relationships | `"What does IV require?"` |
| `research_kb_list_sources` | Browse bibliography | Find Angrist papers |

**Configuration**: `.mcp.json`

---

## Custom Skills

| Skill | Purpose |
|-------|---------|
| `/validate-phase [METHOD]` | Run 6-layer validation checklist |
| `/check-method [METHOD]` | Audit methodological concerns |
| `/run-monte-carlo [METHOD]` | Execute Monte Carlo with analysis |
| `/compare-estimators [FAMILY]` | Generate estimator comparison |
| `/debug-validation [TEST]` | Systematic debugging workflow |
| `/session-init` | Initialize session + RAG health check |

**Location**: `.claude/skills/`

---

## Methodological Concerns (Critical)

**ALWAYS check `docs/METHODOLOGICAL_CONCERNS.md` before implementing**:

| ID | Phase | Issue | Priority |
|----|-------|-------|----------|
| CONCERN-5 | PSM | Bootstrap SE invalid for matching with replacement | HIGH |
| CONCERN-11 | DiD | TWFE bias with staggered adoption | CRITICAL |
| CONCERN-12 | DiD | Pre-trends testing limitations | CRITICAL |
| CONCERN-13 | DiD | Cluster SE with few clusters | HIGH |
| CONCERN-16 | IV | Weak instrument detection | CRITICAL |
| CONCERN-22 | RDD | McCrary density test validity | HIGH |

---

## Code Style

### Python
- **Formatter**: Black with 100-character lines
- **Type hints**: Required everywhere (mypy strict mode)
- **Docstrings**: NumPy style with Parameters, Returns, Raises, Examples

### Julia
- **Formatter**: SciML style, 92-character lines
- **Documentation**: Full docstrings with examples
- **Immutability**: Functions ending in `!` mutate, otherwise return new

---

## Session Workflow

This project follows the hub session pattern (see `patterns/sessions.md`):

1. **Start**: Check `CURRENT_WORK.md` for context
2. **Plan**: For tasks >1hr, create `docs/plans/active/SESSION_N_*.md`
3. **Implement**: Test-first, commit frequently
4. **Document**: Update session file with results
5. **Complete**: Move plan to `implemented/`, update `CURRENT_WORK.md`

### Git Commit Format
```
type(scope): Short description

- Detail 1
- Detail 2

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

Types: `feat`, `fix`, `test`, `docs`, `refactor`, `validate`

---

## Key Implementation Patterns

### Error Handling
```python
# ALWAYS fail explicitly, NEVER fail silently
if propensity_scores.min() < 1e-6:
    raise ValueError(
        f"Propensity scores too close to 0 (min={propensity_scores.min():.2e}). "
        f"Positivity violation detected. Consider trimming or checking covariates."
    )
```

### Input Validation
```python
def simple_ate(outcome: np.ndarray, treatment: np.ndarray) -> ATEResult:
    """Validate inputs immediately, fail fast with diagnostics."""
    if len(outcome) != len(treatment):
        raise ValueError(
            f"Length mismatch: outcome ({len(outcome)}) != treatment ({len(treatment)})"
        )
    if not np.isin(treatment, [0, 1]).all():
        raise ValueError("Treatment must be binary (0 or 1)")
```

### Monte Carlo Testing Pattern
```python
@pytest.mark.monte_carlo
def test_simple_ate_unbiased():
    """Monte Carlo validation: 5000 runs, bias < 0.05, coverage 93-97%."""
    results = []
    for _ in range(5000):
        y, t = generate_rct_dgp(n=200, true_ate=2.0)
        result = simple_ate(y, t)
        results.append(result)

    bias = np.mean([r.ate for r in results]) - 2.0
    coverage = np.mean([r.ci_lower < 2.0 < r.ci_upper for r in results])

    assert abs(bias) < 0.05, f"Bias {bias:.4f} exceeds threshold"
    assert 0.93 < coverage < 0.97, f"Coverage {coverage:.2%} outside range"
```

---

## Cross-Language Validation (Python ↔ Julia)

```python
# tests/validation/cross_language/test_rct_parity.py
def test_simple_ate_matches_julia():
    """Python and Julia must agree to 10 decimal places."""
    from julia import Julia
    jl = Julia(compiled_modules=False)

    # Same data in both languages
    y, t = load_test_fixture("rct_simple")

    py_result = simple_ate(y, t)
    jl_result = jl.eval("simple_ate(y, t)")

    assert np.isclose(py_result.ate, jl_result.ate, rtol=1e-10)
    assert np.isclose(py_result.se, jl_result.se, rtol=1e-10)
```

---

## Environment Setup

```bash
# Python
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pre-commit install

# Julia
cd julia
julia --project -e "using Pkg; Pkg.instantiate()"
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `CURRENT_WORK.md` | 30-second context resume |
| `docs/ROADMAP.md` | Master plan (761 lines) |
| `docs/METHODOLOGICAL_CONCERNS.md` | 13 concerns tracked |
| `pyproject.toml` | Python config (Black, Ruff, Mypy, pytest) |
| `julia/Project.toml` | Julia dependencies |
| `tests/conftest.py` | Shared pytest fixtures |
| `tests/golden_results/` | Reference JSON (111KB) |

---

## Known Issues & Limitations

1. **PSM Limited Overlap**: xfail test documents coverage drops to 31% with extreme propensity separation
2. **TWFE Bias**: Documented in CONCERN-11, use Callaway-Sant'Anna or Sun-Abraham instead
3. **Cross-Language Tests**: Infrastructure exists but deferred pending full Python parity

---

## Session History (Recent)

| Session | Focus | Status |
|---------|-------|--------|
| 83 | **Comprehensive Audit** - Bugs, metrics, docs | ✅ Complete |
| 63-82 | RKD, Bunching, Context Engineering | ✅ Complete |
| 62 | CATE Monte Carlo validation | ✅ Complete |
| 55-61 | IV stages, VCov, McCrary fixes | ✅ Complete |
| 37 | Test suite stabilization | ✅ Complete |
| 22 | Project audit & documentation cleanup | ✅ Complete |

**Current**: Session 83 - Comprehensive repository audit
**Next**: Phase 12 - Selection & Bounds (Heckman, Manski, Lee, QTE)
