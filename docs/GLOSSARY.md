# Glossary

Causal inference terminology reference.

---

## Core Concepts (48 Terms)

**Primary Source**: `research-kb/fixtures/concepts/seed_concepts.yaml`

The research-kb knowledge base contains 48 well-defined causal inference concepts including:
- **Methods**: IV, DiD, RDD, PSM, IPW, DR, DML, Causal Forests, etc.
- **Assumptions**: SUTVA, Unconfoundedness, Parallel Trends, Positivity, etc.
- **Problems**: Endogeneity, Confounding, Selection Bias, etc.
- **Estimands**: ATE, ATT, LATE, CATE, ITT, etc.
- **Theorems**: Frisch-Waugh-Lovell, CLT, Delta Method, etc.

**To query a concept**:
```
# Via MCP tool
research_kb_get_concept "difference-in-differences"

# Via CLI
research-kb concepts "instrumental variables"
```

---

## Implementation-Specific Terms

Terms specific to this codebase's implementation choices.

### Variance Estimators

| Term | Definition | When to Use |
|------|------------|-------------|
| **HC0** | Heteroskedasticity-consistent (White) | Large n (> 500), default |
| **HC1** | HC0 with degrees of freedom correction | Medium n (250-500) |
| **HC2** | Leverage-adjusted | General use |
| **HC3** | More aggressive leverage adjustment | Small n (< 250), recommended |
| **EHW** | Eicker-Huber-White | Synonym for HC0/robust |
| **Neyman variance** | Conservative RCT variance (no covariance term) | RCT, conservative inference |
| **Cluster-robust** | Accounts for within-cluster correlation | Clustered data, DiD |

**Code reference**: `src/causal_inference/utils/variance.py`

---

### Test Terminology

| Term | Definition |
|------|------------|
| **Known-answer test** | Test with hand-calculated expected values |
| **Adversarial test** | Edge cases, boundary conditions (n=1, NaN, etc.) |
| **Monte Carlo test** | Statistical simulation (5000+ runs) |
| **Parity test** | Python-Julia agreement verification |
| **Golden reference** | Frozen results for regression detection |

---

### Monte Carlo Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Bias** | mean(estimates) - true_value | < 0.05 (RCT), < 0.10 (obs) |
| **Coverage** | proportion(CI contains true_value) | 93-97% |
| **SE Accuracy** | mean(SE) / std(estimates) | 0.90-1.10 |
| **RMSE** | sqrt(mean((estimate - true)²)) | Depends on context |

---

### DGP Parameters

| Term | Definition |
|------|------------|
| **DGP** | Data Generating Process - simulation specification |
| **true_ate** | True average treatment effect in DGP |
| **n** | Sample size |
| **n_simulations** | Number of Monte Carlo repetitions |

---

### Method-Specific Terms

#### IV-Specific

| Term | Definition |
|------|------------|
| **First stage** | Regression of treatment on instrument |
| **Second stage** | Regression of outcome on predicted treatment |
| **F-statistic** | Test for instrument relevance (want F > 10) |
| **Stock-Yogo** | Critical values for weak instrument test |
| **Compliers** | Units whose treatment changes with instrument |

#### RDD-Specific

| Term | Definition |
|------|------------|
| **Running variable** | Continuous variable determining treatment |
| **Cutoff** | Threshold where treatment assignment changes |
| **Bandwidth** | Window around cutoff for estimation |
| **McCrary test** | Density test for manipulation |

#### DiD-Specific

| Term | Definition |
|------|------------|
| **Pre-period** | Time before treatment |
| **Post-period** | Time after treatment |
| **Event time** | Time relative to treatment (t - treatment_time) |
| **Staggered adoption** | Different units treated at different times |
| **Never-treated** | Control units that never receive treatment |

#### PSM-Specific

| Term | Definition |
|------|------------|
| **Propensity score** | P(T=1|X), probability of treatment |
| **Caliper** | Maximum distance for acceptable match |
| **Common support** | Region where both treatment groups exist |
| **ATT** | Average Treatment effect on the Treated |

---

### Error Types

| Error | Meaning |
|-------|---------|
| `PropensityScoreError` | Propensity scores outside (0,1) or extreme |
| `WeakInstrumentError` | First-stage F-statistic < threshold |
| `PositivityError` | No overlap in propensity scores |
| `ConvergenceError` | Optimization failed to converge |
| `ParityError` | Python-Julia results don't match |

---

### Code Conventions

| Convention | Meaning |
|------------|---------|
| `_result` suffix | Named tuple with estimate + SE + CI |
| `_dgp` suffix | Data generating function |
| `test_*_montecarlo` | Monte Carlo validation test |
| `test_*_adversarial` | Edge case test |
| `test_*_parity` | Cross-language test |

---

## Acronym Reference

| Acronym | Full Form |
|---------|-----------|
| ATE | Average Treatment Effect |
| ATT | Average Treatment effect on the Treated |
| ATU | Average Treatment effect on the Untreated |
| CATE | Conditional Average Treatment Effect |
| CI | Confidence Interval |
| CLT | Central Limit Theorem |
| DiD | Difference-in-Differences |
| DML | Double Machine Learning |
| DR | Doubly Robust |
| DGP | Data Generating Process |
| HTE | Heterogeneous Treatment Effects |
| IPW | Inverse Probability Weighting |
| ITT | Intent-to-Treat |
| IV | Instrumental Variables |
| LATE | Local Average Treatment Effect |
| LIML | Limited Information Maximum Likelihood |
| MC | Monte Carlo |
| MTE | Marginal Treatment Effect |
| OLS | Ordinary Least Squares |
| PSM | Propensity Score Matching |
| RCT | Randomized Controlled Trial |
| RDD | Regression Discontinuity Design |
| RMSE | Root Mean Squared Error |
| SE | Standard Error |
| SUTVA | Stable Unit Treatment Value Assumption |
| TWFE | Two-Way Fixed Effects |
| 2SLS | Two-Stage Least Squares |

---

## See Also

- Query research-kb for detailed concept definitions
- `docs/METHOD_SELECTION.md` - When to use each method
- `docs/METHODOLOGICAL_CONCERNS.md` - Known issues
