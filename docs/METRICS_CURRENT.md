# Repository Metrics

**Generated**: 2025-12-31 22:46
**Session**: 164
**Generator**: `scripts/update_metrics.py`

---

## Test Counts

| Metric | Count |
|--------|-------|
| Python test functions (`def test_*`) | 3,854 |
| Julia test assertions (`@test`) | 5,121 |
| Total test assertions | 8,975 |
| Pytest collected (actual runnable) | 3,869 |
| Collection errors | 0 |

---

## Source Line Counts

| Language | Directory | Lines (non-empty) |
|----------|-----------|-------------------|
| Python | `src/` | 54,728 |
| Julia | `julia/src/` | 43,699 |
| **Total** | | **98,427** |

---

## Method Families (26)

- `bayesian`
- `bounds`
- `bunching`
- `cate`
- `control_function`
- `did`
- `discovery`
- `dtr`
- `dynamic`
- `iv`
- `mediation`
- `mte`
- `observational`
- `panel`
- `principal_stratification`
- `psm`
- `qte`
- `rct`
- `rdd`
- `rkd`
- `scm`
- `selection`
- `sensitivity`
- `shift_share`
- `timeseries`
- `utils`

---

## Methodology

- **Python tests**: `grep -rE "^\s*def test_" tests/`
- **Julia assertions**: `grep -rE "@test\s" julia/test/`
- **Source lines**: Non-empty lines in `.py` and `.jl` files
- **Method families**: Directories in `src/causal_inference/` with implementation files

---

## Usage

Regenerate this file:
```bash
python scripts/update_metrics.py --output
```

Check metrics without writing:
```bash
python scripts/update_metrics.py
```
