# Compare Estimators

Generate comparison table between causal inference estimators.

## Description
Compare Estimators - Estimator Comparison Tables (project)

## Usage
/compare-estimators [FAMILY]

Example: `/compare-estimators IV` or `/compare-estimators observational`

## Prompt

You are generating a comparison of estimators in the **{FAMILY}** family.

### Step 1: Identify Estimators

**By family**:
- **RCT**: SimpleATE, NeymanATE, RegressionAdjustedATE, StratifiedATE
- **Observational**: IPW, DoublyRobust, OutcomeRegression, PSM
- **DiD**: ClassicDiD, TWFE, CallawayASantAnna, SunAbraham, EventStudy
- **IV**: 2SLS, LIML, AndersonRubin, WeakIVRobust
- **RDD**: SharpRDD, FuzzyRDD, LocalLinear, LocalPolynomial

### Step 2: Generate Properties Table

For each estimator, document:

| Property | Description |
|----------|-------------|
| **Estimand** | What it estimates (ATE, ATT, LATE, etc.) |
| **Assumptions** | Required identifying assumptions |
| **Variance** | Variance estimator used |
| **Robustness** | What misspecification it's robust to |
| **Limitations** | Known failure modes |
| **Sample Size** | Minimum recommended n |
| **Complexity** | Computational cost |

### Step 3: Read Implementation Details

For each estimator:
1. Find source: `src/causal_inference/{family}/{estimator}.py`
2. Find Julia: `julia/src/{family}/{estimator}.jl`
3. Check docstrings for parameters
4. Note any warnings or caveats

### Step 4: Query Literature

Use research-kb to get authoritative definitions:
```
research_kb_get_concept "{estimator_name}"
research_kb_search "{estimator} vs {other_estimator}"
```

### Step 5: Generate Recommendation Matrix

Create a decision matrix:

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Large n, homogeneous effects | ... | ... |
| Small n, need robustness | ... | ... |
| Weak instruments suspected | ... | ... |
| Heterogeneous effects | ... | ... |

## Output Format
```
=== {FAMILY} Estimator Comparison ===

## Available Estimators
| Estimator | Python | Julia | Estimand |
|-----------|--------|-------|----------|
| ... | path | path | ATE/ATT/LATE |

## Properties Comparison
| Property | Est1 | Est2 | Est3 |
|----------|------|------|------|
| Assumptions | ... | ... | ... |
| Variance | ... | ... | ... |
| Robustness | ... | ... | ... |
| Limitations | ... | ... | ... |
| Min n | ... | ... | ... |

## When to Use Each
| Scenario | Recommended | Why |
|----------|-------------|-----|
| ... | ... | ... |

## References
- Primary sources from research-kb
- Implementation notes from code
```
