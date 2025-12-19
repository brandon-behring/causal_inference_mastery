# Check Method

Audit methodological concerns for a causal inference method.

## Description
Check Method - Methodological Audit (project)

## Usage
/check-method [METHOD]

Example: `/check-method IV` or `/check-method PSM`

## Prompt

You are auditing the methodological correctness of **{METHOD}** implementation.

### Step 1: Read Methodological Concerns
1. Read `docs/METHODOLOGICAL_CONCERNS.md`
2. Find all concerns related to {METHOD}
3. List each concern with its status

### Step 2: Check Implementation
For each concern found:
1. Locate the implementation file
2. Verify the concern is addressed in code
3. Check for appropriate error handling
4. Verify documentation mentions the concern

### Step 3: Cross-Reference with Literature
Query research-kb for primary sources:
- Use `research_kb_get_concept "{method_name}"` to get definition
- Use `research_kb_search "{method} assumptions"` for requirements
- Verify implementation matches literature specifications

### Step 4: Check Assumptions
For {METHOD}, verify these are validated:

**If IV**: Relevance (F > 10), Exclusion restriction (documented), Monotonicity (if LATE)
**If DiD**: Parallel trends (pre-trends test), No anticipation, SUTVA
**If RDD**: Continuity at cutoff, No manipulation (McCrary test), Bandwidth selection
**If PSM**: Overlap/positivity, Unconfoundedness (documented), Balance checks
**If IPW/DR**: Positivity, Model specification, Extreme weight handling

### Step 5: Variance Estimation
Verify appropriate variance estimator:
- HC3 for small samples (n < 250)
- Cluster-robust if clustered data
- Bootstrap if asymptotic invalid
- Check `docs/METHODOLOGICAL_CONCERNS.md` for method-specific guidance

## Output Format
```
=== {METHOD} Methodological Audit ===

## Concerns Found
| ID | Issue | Status | Implementation |
|----|-------|--------|----------------|
| CONCERN-X | ... | ADDRESSED/OPEN | file:line |

## Assumption Checks
- [x/] Assumption 1: Status
- [x/] Assumption 2: Status

## Variance Estimation
- Estimator used: ...
- Appropriate: YES/NO
- Justification: ...

## Literature Alignment
- Primary source: ...
- Implementation matches: YES/NO

## Recommendations
1. ...
```
