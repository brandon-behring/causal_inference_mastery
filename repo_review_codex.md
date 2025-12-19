# Repo Review (Codex)

## Scope
- Reviewed core Python source, tests/validation, docs, and repo metadata.
- Did not execute tests or benchmarks; findings are static review only.

## High-Risk Findings (Correctness / Misleading API)
- Fuzzy RDD ignores kernel weighting entirely; `kernel` only affects bandwidth selection. This makes `kernel='triangular'` a no-op and effectively uses a rectangular kernel. See `src/causal_inference/rdd/fuzzy_rdd.py`.
- `cct_bandwidth()` is labeled as Calonico-Cattaneo-Titiunik but is implemented as `1.5 * IK`. This is not CCT and will mislead users who expect CCT properties. See `src/causal_inference/rdd/bandwidth.py` and usage in `src/causal_inference/rdd/sharp_rdd.py`.
- RKD standard errors are simplified and likely understate uncertainty. Sharp RKD ignores uncertainty in the treatment kink (treated as known), and Fuzzy RKD omits covariance between reduced-form and first-stage kinks. This is explicitly noted in tests but not in the API. See `src/causal_inference/rkd/sharp_rkd.py`, `src/causal_inference/rkd/fuzzy_rkd.py`, and `tests/validation/cross_language/test_python_julia_rkd.py`.
- Anderson-Rubin test/CI uses a projection onto Z only, even when controls X are present. AR with controls should use Z residualized on X (or equivalently test Z in Y - beta D ~ Z + X). Current implementation can be wrong when X is present. See `src/causal_inference/iv/diagnostics.py`.

## Medium-Risk Methodological Issues
- IPW uses normalized (Hajek) weights but documentation claims “maintaining unbiasedness.” With estimated propensities, Hajek is only asymptotically unbiased. Finite-sample bias is expected. See `src/causal_inference/rct/estimators_ipw.py`.
- CATE meta-learners (S/T/X/R, DML) use ad-hoc ATE SEs based on residual variance or asymptotic normality. This is not valid for many ML base learners and can understate uncertainty. See `src/causal_inference/cate/meta_learners.py` and `src/causal_inference/cate/dml.py`.
- SCM inference is simplified. Placebo distribution is unfiltered by pre-treatment fit, and bootstrap resamples only pre-treatment periods. Both can be badly calibrated. See `src/causal_inference/scm/basic_scm.py` and `src/causal_inference/scm/augmented_scm.py`.
- PSM balance diagnostics for M>1 only use the first matched control, which is not representative. This can materially misstate balance. See `src/causal_inference/psm/psm_estimator.py` and `src/causal_inference/psm/balance.py`.
- RKD and RDD inference relies on t-distribution with simple df formulas; robust bias correction (RBC) and variance formulas are approximations. This can miscalibrate coverage in small samples. See `src/causal_inference/rdd/sharp_rdd.py`, `src/causal_inference/rdd/bandwidth.py`, and `src/causal_inference/rkd/*.py`.
- E-value conversions treat OR/HR as RR without caution about non-rare outcomes. This can be misleading for common outcomes. See `src/causal_inference/sensitivity/e_value.py`.

## Unstated / Underemphasized Assumptions
- PSM matching uses raw propensity distance and greedy order; results are order-dependent and sensitive to caliper choice. Logit propensity distance is not offered. See `src/causal_inference/psm/matching.py`.
- RDD/RKD assume no manipulation of the running variable and smooth potential outcomes; diagnostics exist but do not gate estimation. See `src/causal_inference/rdd/rdd_diagnostics.py` and `src/causal_inference/rkd/diagnostics.py`.
- DiD pre-trends tests are implemented but do not adjust for multiple comparisons in event study leads/lags. See `src/causal_inference/did/event_study.py`.
- IV diagnostics use Cragg-Donald, but there is no Kleibergen-Paap statistic for heteroskedastic or clustered settings. See `src/causal_inference/iv/diagnostics.py`.
- Bunching uses bin-centered estimates; kink alignment with bin centers is assumed, which can bias h0 and excess mass. See `src/causal_inference/bunching/excess_mass.py` and `src/causal_inference/bunching/counterfactual.py`.

## Test and Validation Gaps
- Cross-language tests are skipped unless `juliacall` is installed, but `juliacall` is not in dependencies. Claims of 100% parity are not enforceable by default. See `tests/validation/cross_language/julia_interface.py` and all `tests/validation/cross_language/*`.
- No Python RKD Monte Carlo validation suite exists despite TODOs in docs. See absence in `tests/validation/monte_carlo/` and note in `CURRENT_WORK.md`.
- Known skips/xfails remain (single-unit variance in RCTs, PSM limited overlap, AR over-identified test), but docs claim “all tests passing, no skips.” See `tests/validation/adversarial/test_simple_ate_adversarial.py`, `tests/validation/monte_carlo/test_monte_carlo_psm.py`, `tests/test_iv/test_diagnostics.py`, and `docs/standards/PHASE_COMPLETION_STANDARDS.md`.

## Documentation Drift / Repo Hygiene
- README says Python phases 3–5 are “planned,” but ROADMAP claims phases 1–11 complete. See `README.md` vs `docs/ROADMAP.md`.
- `docs/KNOWN_LIMITATIONS.md` still lists McCrary xfails and missing Python Fuller, but both appear resolved in code/tests. See `docs/KNOWN_LIMITATIONS.md`, `src/causal_inference/rdd/mccrary.py`, and `src/causal_inference/iv/fuller.py`.
- PSM docstring says balance diagnostics “not yet implemented,” but they are. See `src/causal_inference/psm/psm_estimator.py`.
- Julia `Manifest.toml` is ignored and not committed; Python deps are unpinned. This makes reproducibility and parity fragile. See `.gitignore` and `pyproject.toml`.
- Stray backup file: `julia/test/rdd/test_sensitivity.jl.backup` should be removed or ignored.

## Improvement Options (Pros / Cons)

### 1) RDD bandwidth + inference accuracy
Option A: Implement full CCT bandwidth + RBC per Cattaneo et al.
- Pros: Methodologically correct, improves CI calibration, matches expectations.
- Cons: Non-trivial implementation, more complex tuning, heavier testing burden.

Option B: Rename `cct_bandwidth` to `cct_approx` and update docs/tests to reflect approximation.
- Pros: Minimal code change, honest API, avoids false confidence.
- Cons: Users still lack a true CCT option.

Option C: Delegate bandwidth selection to a dedicated RDD library (e.g., `rdd` in R via `rpy2` or a Python port).
- Pros: Leverages validated implementations.
- Cons: Adds heavy dependencies and complicates reproducibility.

### 2) Fuzzy RDD kernel weighting
Option A: Implement weighted 2SLS with kernel weights (e.g., WLS in both stages).
- Pros: Aligns with local polynomial RDD theory; kernel parameter becomes meaningful.
- Cons: Requires weighted IV implementation and more testing.

Option B: Remove kernel option and clearly state “rectangular window only.”
- Pros: Clear API, no false precision.
- Cons: Less flexible and may reduce estimator quality.

### 3) RKD standard errors
Option A: Full delta method including variance of treatment slope and covariance terms.
- Pros: More accurate SEs, closer to Julia implementation.
- Cons: Requires careful covariance estimation and validation.

Option B: Bootstrap (clustered or local) for SEs.
- Pros: Avoids analytic derivations; handles complex correlation.
- Cons: Slower, requires careful bootstrap design near kink.

### 4) IV diagnostics robustness
Option A: Implement Kleibergen-Paap rk Wald F and related weak-IV tests.
- Pros: Valid under heteroskedasticity and clustering.
- Cons: Extra math/implementation complexity.

Option B: Delegate diagnostics to `linearmodels` and wrap its results.
- Pros: Leverages a mature implementation.
- Cons: External dependency and potential API drift.

### 5) CATE / ML inference validity
Option A: Use orthogonalized influence functions + cross-fitting for ATE SEs across learners.
- Pros: Statistically principled, consistent with modern practice.
- Cons: More code complexity, longer compute times.

Option B: Bootstrap for ATE CIs (optionally stratified by treatment).
- Pros: Simple and model-agnostic.
- Cons: Expensive and sometimes unstable for small samples.

### 6) PSM balance for M>1
Option A: Compute balance using all matched controls with weights (1/M each).
- Pros: Correct representation of the matched sample.
- Cons: Slightly more implementation complexity.

Option B: Limit balance diagnostics to M=1 and warn for M>1.
- Pros: Transparent, minimal change.
- Cons: Users lose diagnostics for many-to-one matching.

### 7) Cross-language parity enforcement
Option A: Add `juliacall` as an optional extra and wire a CI job to run cross-language tests.
- Pros: Parity claims become verifiable.
- Cons: Heavier CI, longer runtimes.

Option B: Split parity tests into a separate suite with explicit instructions and badges.
- Pros: Honest and user-friendly without blocking standard test runs.
- Cons: Still optional and easy to skip.

### 8) Documentation and reproducibility
Option A: Single source of truth for status (e.g., auto-generate README badges from tests/docs).
- Pros: Reduces drift.
- Cons: Requires scripting and maintenance.

Option B: Pin environments (pip-compile/poetry + Julia Manifest).
- Pros: Reproducible results and parity.
- Cons: More maintenance overhead when updating dependencies.

---

## Suggested Next Steps (If You Want Me To Implement)
1) Fix Fuzzy RDD kernel weighting or rename kernel option (small, high impact).
2) Update docs to remove inconsistencies and reflect actual test status.
3) Add Python RKD Monte Carlo validation suite (align with CURRENT_WORK).
4) Add a minimal KP weak-IV diagnostic or document limitation explicitly.

---

## Iteration 2 Addendum (Deeper Audit)

### Re-validated Findings (No Change)
- Fuzzy RDD kernel is unused beyond bandwidth selection; current behavior is rectangular-only despite `kernel` option. See `src/causal_inference/rdd/fuzzy_rdd.py`.
- `cct_bandwidth()` is an approximation (`1.5 * IK`) labeled as CCT. See `src/causal_inference/rdd/bandwidth.py`.
- Sharp RKD SE ignores variance of treatment slope; Fuzzy RKD SE omits covariance between first-stage and reduced-form kinks. See `src/causal_inference/rkd/*.py`.
- Anderson–Rubin test uses projection on Z only, even with controls X. See `src/causal_inference/iv/diagnostics.py`.
- Cross-language parity tests are optional and skip by default unless `juliacall` is installed. See `tests/validation/cross_language/*`.

### New High/Medium-Risk Findings
- **Broken test module**: `tests/validation/monte_carlo/test_type_i_error.py` imports non-existent modules (e.g., `causal_inference.rct.simple_ate`, `classic_did`, `two_stage_ls`, `sharp_rdd`) and will fail if executed. This suggests test drift and likely undermines claimed pass rates.
- **Stratified ATE variance underestimation**: For n1=1 or n0=1, variance is set to 0 and described as “conservative,” but this makes SEs too small (anti-conservative). See `src/causal_inference/rct/estimators_stratified.py`.
- **ASCM jackknife SE is not a true jackknife**: It renormalizes existing weights instead of recomputing SCM weights after leaving out each donor. This can materially understate uncertainty. See `src/causal_inference/scm/augmented_scm.py`.
- **SCM weight optimization failures are masked**: `compute_scm_weights` proceeds with weights even if both optimizers fail, without warning or error. This risks silent invalid results. See `src/causal_inference/scm/weights.py`.
- **Event study assumes a single common treatment time**: `event_study()` accepts a scalar `treatment_time` but does not validate that treated units share that timing. Misuse with staggered adoption yields biased TWFE-style dynamics. See `src/causal_inference/did/event_study.py`.
- **Paired-variance option ignores replacement**: `variance_method='paired'` is allowed even when `with_replacement=True`, but the paired variance formula assumes no replacement. See `src/causal_inference/psm/psm_estimator.py`.
- **Import-path ambiguity**: Mixed `src.causal_inference` and `causal_inference` imports plus `src/__init__.py` create packaging ambiguity. Editable installs may pass while normal installs break. See `src/__init__.py` and widespread imports.

### Additional Methodology Caveats
- Sharp RDD uses `n_eff` = count of positive weights for df; this can overstate df when weights are highly uneven. See `src/causal_inference/rdd/sharp_rdd.py`.
- CCT “robust” SE uses an ad-hoc 0.5 scaling for bias variance; this is not documented as an approximation and likely deviates from CCT formulas. See `src/causal_inference/rdd/sharp_rdd.py`.
- RDD/RKD covariate balance tests use multiple p-values without multiplicity adjustment. See `src/causal_inference/rdd/rdd_diagnostics.py` and `src/causal_inference/rkd/diagnostics.py`.
- CATE learners (S/T/X/R/DML) compute ATE SEs with heuristic formulas that are not valid for many ML base learners. See `src/causal_inference/cate/meta_learners.py` and `src/causal_inference/cate/dml.py`.
- DML cross-fitting uses unstratified folds; if treatment is imbalanced, some folds may lack treated or control units. No guardrails are present. See `src/causal_inference/cate/dml.py`.
- E-value and Rosenbaum bounds rely on approximations (RR/OR/HR conversions; normal approximation for Wilcoxon). Small-sample behavior may be off. See `src/causal_inference/sensitivity/e_value.py` and `src/causal_inference/sensitivity/rosenbaum.py`.

### Additional Improvement Options (Iteration 2)
Option A: Fix stale `test_type_i_error.py` or remove it from default suite.
- Pros: Restores test credibility and aligns suite with actual API.
- Cons: Requires rewriting DGPs and expected interfaces.

Option B: Recompute SCM weights inside ASCM jackknife (LOO) and add a warning if optimizers fail.
- Pros: Correct SEs and more transparent diagnostics.
- Cons: Slower runtime for ASCM inference.

Option C: Enforce assumptions via guards.
- Pros: Prevents silent misuse (event study with staggered adoption, paired variance with replacement).
- Cons: Stricter API may break existing user code.

Option D: Normalize import paths (`causal_inference` only) and remove `src/__init__.py`.
- Pros: Standard packaging, avoids editable-only import success.
- Cons: Requires refactor and test updates.
