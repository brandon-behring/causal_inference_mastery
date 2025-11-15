# Phase Completion Standards

**Version**: 1.0
**Status**: Mandatory for all phases
**Created**: 2025-11-14
**Based on**: Phase 1 RCT Foundation audit (Grade: A+, 98/100)

## Purpose

This document codifies the methodological standards, validation architecture, and quality requirements that made Phase 1 exceptional. **All future phases (2-8) MUST meet these standards before being marked complete.**

These are not aspirational goals—they are **mandatory requirements** derived from Phase 1's successful execution.

---

## 1. Six-Layer Validation Architecture (MANDATORY)

Every estimator MUST pass all six validation layers:

### Layer 1: Known-Answer Tests
**Requirement**: At least 2 tests per estimator with analytically verifiable results

**Examples**:
- Zero treatment effect → estimate ≈ 0
- All units same value → variance = 0
- Perfect correlation → R² = 1

**Implementation**:
```julia
@testset "Known-Answer: Zero treatment effect" begin
    outcomes = ones(100)
    treatment = vcat(fill(true, 50), fill(false, 50))
    problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))
    solution = solve(problem, SimpleATE())
    @test abs(solution.estimate) < 1e-10
end
```

### Layer 2: Adversarial Tests
**Requirement**: Minimum 10 adversarial tests per estimator covering:
- **Edge cases**: n=1, n=2, all treated, all control
- **Numerical stability**: NaN, Inf, -Inf in outcomes
- **Degenerate inputs**: Perfect collinearity, zero variance, extreme outliers
- **Boundary conditions**: Propensity scores near 0/1, empty strata

**Purpose**: Prevent production bugs that normal tests miss

**Examples from Phase 1** (49 tests total):
- `test_simple_ate_extreme_outliers` - outcomes with ±1e10 values
- `test_ipwate_propensity_near_zero` - propensity = 0.001
- `test_regression_ate_perfect_collinearity` - X with duplicate columns
- `test_stratified_ate_one_unit_stratum` - stratum with n=1

**Standard**:
```julia
@testset "Adversarial: Edge Cases" begin
    # n=1
    @test_throws ArgumentError solve(problem_n1, estimator)

    # All treated
    @test_throws ArgumentError solve(problem_all_treated, estimator)

    # NaN in outcomes
    @test_throws ArgumentError solve(problem_with_nan, estimator)
end
```

### Layer 3: Monte Carlo Ground Truth Validation
**Requirement**: Data-generating process (DGP) with KNOWN treatment effect

**Standards**:
- Generate data with τ = 2.0 (known ground truth)
- N=10,000 simulations
- Check bias: |mean(estimates) - τ| < 0.05
- Check coverage: 94% ≤ coverage ≤ 96% for α=0.05
- Check SE accuracy: |mean(SE) - SD(estimates)| / SD(estimates) < 0.10

**Implementation** (from `test/validation/test_monte_carlo_ground_truth.jl:142-225`):
```julia
function dgp_constant_ate(;
    n::Int = 100,
    τ::Float64 = 2.0,
    σ_Y0::Float64 = 1.0,
    σ_Y1::Float64 = 1.0,
    seed::Union{Int,Nothing} = nothing,
)
    if !isnothing(seed); Random.seed!(seed); end

    Y0 = randn(n) .* σ_Y0
    Y1 = Y0 .+ τ .+ randn(n) .* (σ_Y1 - σ_Y0)
    treatment = vcat(fill(true, n÷2), fill(false, n - n÷2))
    outcomes = ifelse.(treatment, Y1, Y0)

    return outcomes, treatment, τ
end

@testset "Monte Carlo: SimpleATE" begin
    bias, coverage, se_accuracy = validate_monte_carlo(SimpleATE(), 10_000)
    @test abs(bias) < 0.05
    @test 0.94 <= coverage <= 0.96
    @test se_accuracy < 0.10
end
```

### Layer 4: PyCall Cross-Validation
**Requirement**: Compare Julia implementation against Python reference (if available)

**Standards**:
- Estimate agreement: |Julia - Python| < 0.001
- SE agreement: |Julia_SE - Python_SE| / Python_SE < 0.05
- CI agreement: overlapping confidence intervals

**Implementation** (from `test/validation/test_python_parity.jl`):
```julia
@testset "Python Parity: SimpleATE" begin
    julia_solution = solve(problem, SimpleATE())
    python_result = py"compute_simple_ate"(outcomes, treatment)

    @test abs(julia_solution.estimate - python_result["estimate"]) < 0.001
    @test abs(julia_solution.se - python_result["se"]) / python_result["se"] < 0.05
end
```

### Layer 5: R Triangulation
**Requirement**: Independent R implementation for reference validation

**Standards**:
- Estimate agreement: |Julia - R| < 0.001
- SE agreement: |Julia_SE - R_SE| / R_SE < 0.05
- Graceful degradation if R unavailable (warning, not failure)

**Implementation** (from `validation/r_scripts/validate_rct.R` + `test/validation/test_r_validation.jl`):
```julia
@testset "R Validation: SimpleATE" begin
    if !r_available()
        @warn "R not available, skipping R validation"
        return
    end

    r_result = run_r_validation("simple_ate", outcomes, treatment)
    julia_solution = solve(problem, SimpleATE())

    @test abs(julia_solution.estimate - r_result.estimate) < 0.001
    @test abs(julia_solution.se - r_result.se) / r_result.se < 0.05
end
```

### Layer 6: Golden Reference Tests
**Requirement**: Published results from authoritative papers

**Examples**:
- LaLonde (1986) dataset → NSW training program ATE
- Imbens & Rubin (2015) textbook examples
- Regression Discontinuity: Lee (2008) House elections data

**Standard**:
```julia
@testset "Golden Reference: LaLonde NSW" begin
    outcomes, treatment, covariates = load_lalonde_nsw()
    problem = RCTProblem(outcomes, treatment, covariates, nothing, (alpha=0.05,))
    solution = solve(problem, RegressionATE())

    # LaLonde (1986) reported ATE ≈ 1794, SE ≈ 633
    @test abs(solution.estimate - 1794) < 100
    @test abs(solution.se - 633) < 50
end
```

---

## 2. Variance Estimation Standards

### Requirement: Explicit Justification for Variance Estimator Choice

Every estimator MUST document:
1. **Which variance estimator** is used (HC3, Neyman, Horvitz-Thompson, etc.)
2. **Why that specific estimator** was chosen
3. **What assumptions** it makes
4. **What alternatives** exist and why they were rejected
5. **Academic references** supporting the choice

### Documentation Template

```julia
\"\"\"
# Variance Estimation

**[Estimator Name] Variance Estimator** ([Citation]):

We use [estimator name] specifically because:
- **[Reason 1]**: [Explanation]
- **[Reason 2]**: [Explanation]
- **[Reason 3]**: [Explanation]

Mathematical formula:
```math
Var(\\hat{\\tau}) = [formula]
```

**Alternative estimators**:
- [Alternative 1]: [Why not used]
- [Alternative 2]: [Why not used]

**References**:
- [Author et al. (Year). Title. *Journal*.]
\"\"\"
```

### Examples from Phase 1

**SimpleATE - Neyman Variance** (`src/estimators/rct/simple_ate.jl:24-39`):
```julia
# Variance Estimation

**Neyman Conservative Variance** (Neyman 1923):

We use separate variances for treated and control groups (heteroskedastic variance):

Var(τ̂) = s²₁/n₁ + s²₀/n₀

**Why NOT pooled variance (classical t-test)**:
- Pooled variance assumes σ²₁ = σ²₀ (homoskedasticity)
- This assumption often violated in causal inference
- Treatment can affect variance (e.g., more variable outcomes)
- Neyman variance is conservative: valid even if variances differ
```

**RegressionATE - HC3 Robust SE** (`src/estimators/rct/regression_ate.jl:25-50`):
```julia
# Variance Estimation

**HC3 Heteroskedasticity-Robust Standard Errors** (Long & Ervin 2000):

We use HC3 ("Jackknife") robust standard errors specifically because:
- **Best small-sample properties**: HC3 has lowest bias in small samples (n < 250)
- **Conservative**: More conservative than HC0, HC1, HC2 (protects Type I error)
- **Leverage adjustment**: Down-weights high-leverage observations appropriately
- **Recommended**: Long & Ervin (2000) recommend HC3 for small samples

**Alternative estimators**:
- HC0: No finite-sample correction (biased downward in small samples)
- HC1: df correction only (still biased in small samples)
- HC2: Less conservative than HC3 (higher Type I error)
```

### Mandatory Elements

All variance documentation MUST include:
- [ ] Mathematical formula in docstring
- [ ] Justification for choice (3+ reasons)
- [ ] Assumptions listed explicitly
- [ ] Alternatives considered and rejected
- [ ] At least 2 academic references
- [ ] Small-sample behavior discussed (if n < 250)

---

## 3. Code Organization Standards

### SciML Design Pattern (MANDATORY)

All estimation problems MUST follow the Problem-Estimator-Solution pattern:

```julia
# 1. Problem Type (immutable struct with validation)
struct RCTProblem <: AbstractCausalProblem
    outcomes::Vector{Float64}
    treatment::Vector{Bool}
    covariates::Union{Matrix{Float64}, Nothing}
    strata::Union{Vector{Int}, Nothing}
    parameters::NamedTuple

    # Inner constructor with validation
    function RCTProblem(outcomes, treatment, covariates, strata, parameters)
        validate_inputs(outcomes, treatment, covariates, strata)
        new(outcomes, treatment, covariates, strata, parameters)
    end
end

# 2. Estimator Type (zero-field struct)
struct SimpleATE <: AbstractRCTEstimator end

# 3. Solution Type (immutable results)
struct RCTSolution <: AbstractCausalSolution
    estimate::Float64
    se::Float64
    ci_lower::Float64
    ci_upper::Float64
    n_treated::Int
    n_control::Int
    retcode::Symbol
    original_problem::RCTProblem
end

# 4. Solver Interface
function solve(problem::RCTProblem, estimator::SimpleATE)::RCTSolution
    # Implementation
    return RCTSolution(...)
end
```

### File Organization (from Phase 1)

```
src/estimators/[method]/
├── simple_ate.jl          # Difference-in-means
├── regression_ate.jl      # ANCOVA
├── ipw_ate.jl            # Inverse probability weighting
├── stratified_ate.jl     # Blocked randomization
└── permutation_test.jl   # Randomization inference

test/estimators/[method]/
├── test_simple_ate.jl           # Unit tests (80%+ coverage)
├── test_regression_ate.jl
├── test_ipw_ate.jl
├── test_stratified_ate.jl
└── test_permutation_test.jl

test/validation/
├── test_monte_carlo_ground_truth.jl  # Layer 3
├── test_python_parity.jl             # Layer 4
├── test_r_validation.jl              # Layer 5
└── test_adversarial.jl               # Layer 2

validation/
├── r_scripts/
│   └── validate_[method].R    # R reference implementations
└── python_scripts/
    └── validate_[method].py   # Python reference implementations

benchmark/
└── benchmark_[method].jl      # Performance baselines
```

### Naming Conventions

- **Estimators**: `[Method]ATE`, `[Method]ATT`, `[Method]RD` (e.g., `SimpleATE`, `RegressionATE`)
- **Problems**: `[Method]Problem` (e.g., `RCTProblem`, `RDProblem`)
- **Solutions**: `[Method]Solution` (e.g., `RCTSolution`, `RDSolution`)
- **Test files**: `test_[estimator_name].jl` (lowercase with underscores)
- **Validation files**: `test_[validation_type].jl` (e.g., `test_monte_carlo_ground_truth.jl`)

---

## 4. Documentation Standards

### Docstring Requirements (MANDATORY)

Every estimator MUST have:

1. **One-line summary** (< 80 chars)
2. **Mathematical Foundation section** with LaTeX equations
3. **Variance Estimation section** (see section 2 above)
4. **Usage section** with runnable examples
5. **Requirements section** listing assumptions
6. **Benefits section** explaining when to use
7. **Limitations section** explaining when NOT to use
8. **References section** with 3+ academic citations

### Template

```julia
\"\"\"
    EstimatorName <: AbstractEstimatorType

[One-line summary]

# Mathematical Foundation

[Explanation with LaTeX math blocks]

# Variance Estimation

[See section 2 template above]

# Usage

```julia
[Runnable example code]
```

# Requirements

- [Assumption 1]
- [Assumption 2]

# Benefits

- [Advantage 1]
- [Advantage 2]

# Limitations

- [When not to use]
- [What can go wrong]

# References

- [Citation 1]
- [Citation 2]
- [Citation 3]
\"\"\"
struct EstimatorName <: AbstractEstimatorType end
```

### Example from Phase 1

See `src/estimators/rct/regression_ate.jl:5-80` for complete example with all required sections.

---

## 5. Test Coverage Standards

### Minimum Requirements

- **Unit tests**: 80%+ line coverage per estimator file
- **Integration tests**: All estimators work together correctly
- **Validation tests**: All six layers passing
- **Benchmark tests**: Performance baselines established

### Test Quality Standards

Tests MUST include:
- **Descriptive names**: `test_simple_ate_zero_treatment_effect` not `test1`
- **Clear failure messages**: Custom messages explaining what went wrong
- **Independence**: Each test runs in isolation (no shared state)
- **Determinism**: Use `Random.seed!()` for reproducible random tests
- **Speed**: Unit tests < 1s each, validation tests < 10s each

### Coverage Verification

```bash
# Run with coverage
julia --project=. -e 'using Pkg; Pkg.test(coverage=true)'

# Generate coverage report
julia --project=. -e 'using Coverage; coverage = process_folder(); covered_lines, total_lines = get_summary(coverage); println("Coverage: ", round(100 * covered_lines / total_lines, digits=2), "%")'
```

**Requirement**: Every phase MUST have >80% total test coverage before completion.

---

## 6. Performance Standards

### Benchmark Requirements

Every phase MUST establish performance baselines:

1. **Execution time** for typical problem sizes
2. **Memory allocation** patterns
3. **Scaling behavior** (how time grows with n)
4. **Regression detection** (performance doesn't degrade)

### Benchmark Template

```julia
using BenchmarkTools

@benchmark begin
    problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))
    solution = solve(problem, SimpleATE())
end setup=(
    outcomes = randn(1000);
    treatment = rand(Bool, 1000)
)
```

### Standards from Phase 1

- **SimpleATE**: ~10μs for n=1000
- **RegressionATE**: ~100μs for n=1000, p=10
- **IPWATE**: ~50μs for n=1000
- **StratifiedATE**: ~30μs for n=1000, S=5
- **PermutationTest**: ~20ms for n=1000, B=10000

**Requirement**: Document baseline performance in `benchmark/README.md`

---

## 7. Error Handling Standards

### Explicit Error Messages (MANDATORY)

All validation errors MUST follow this format:

```julia
throw(ArgumentError(
    "CRITICAL ERROR: [What went wrong]\n" *
    "Function: [function_name]\n" *
    "[Explanation of why it's a problem]\n" *
    "[What user should do to fix it]"
))
```

### Example from Phase 1 (`src/estimators/rct/ipw_ate.jl:169-176`):

```julia
if isnothing(covariates)
    throw(ArgumentError(
        "CRITICAL ERROR: Covariates field is nothing.\n" *
        "Function: solve(IPWATE)\n" *
        "IPWATE requires propensity scores in first column of covariates.\n" *
        "For simple RCT with p=0.5, use: covariates = hcat(fill(0.5, n))"
    ))
end
```

### Required Error Checks

Every estimator MUST validate:
- [ ] No NaN/Inf in outcomes
- [ ] Treatment variation exists (not all treated/control)
- [ ] Sufficient sample size (method-dependent)
- [ ] Required fields populated (covariates, strata, etc.)
- [ ] Valid parameter ranges (alpha ∈ (0,1), propensity ∈ (0,1), etc.)

---

## 8. Phase Completion Checklist

A phase is **NOT complete** until ALL of these are true:

### Implementation
- [ ] All estimators follow SciML Problem-Estimator-Solution pattern
- [ ] All estimators have comprehensive docstrings (8 required sections)
- [ ] All variance estimators documented with justification
- [ ] All error messages use explicit format
- [ ] Code formatted with JuliaFormatter (SciML style)

### Testing
- [ ] All six validation layers passing for every estimator
- [ ] At least 10 adversarial tests per estimator
- [ ] Monte Carlo validation shows bias < 0.05, coverage 94-96%
- [ ] Python parity tests passing (if Python implementation exists)
- [ ] R triangulation tests passing (or gracefully skipped)
- [ ] Test coverage >80% overall
- [ ] All tests deterministic (seeded random)

### Documentation
- [ ] ROADMAP.md updated with phase completion status
- [ ] METHODOLOGICAL_CONCERNS.md updated with any new concerns
- [ ] README.md includes usage examples
- [ ] All estimators have runnable examples in docstrings

### Performance
- [ ] Benchmarks established and documented
- [ ] Performance baselines recorded in `benchmark/README.md`
- [ ] No performance regressions vs previous phases

### Review
- [ ] Manual review of all estimator implementations
- [ ] Comparison with authoritative references (textbooks, papers)
- [ ] Cross-validation with other software (R, Python, Stata)

---

## 9. Methodological Best Practices

### From Phase 1 Audit

**Crown Jewels** (maintain in all phases):
- Six-layer validation prevents both false positives and false negatives
- Adversarial tests catch production bugs normal tests miss
- Monte Carlo validation proves statistical properties
- Cross-language validation prevents implementation bugs

**Lessons Learned**:
1. **Circular validation is useless**: Never validate code against itself
2. **Known-answer tests are gold**: Analytically verifiable results are best
3. **Document variance choices**: Future you will forget why you chose HC3
4. **Fail explicitly**: Silent failures create debugging nightmares
5. **Test edge cases**: Production users will find them

### Anti-Patterns to Avoid

**DON'T**:
- ❌ Validate estimator against itself (circular trap)
- ❌ Assume pooled variance (homoskedasticity assumption)
- ❌ Use HC0/HC1 without justification (HC3 better for small samples)
- ❌ Skip adversarial tests ("it won't happen in practice")
- ❌ Write vague error messages ("invalid input")
- ❌ Ignore warnings in tests
- ❌ Use random data without setting seed
- ❌ Skip documentation ("code is self-explanatory")

**DO**:
- ✅ Validate against independent implementations (R, Python)
- ✅ Use heteroskedastic variance by default (Neyman, HC3)
- ✅ Document why you chose variance estimator
- ✅ Test edge cases that "won't happen"
- ✅ Write explicit error messages with solutions
- ✅ Treat warnings as errors
- ✅ Set seed for reproducibility
- ✅ Document even when "obvious"

---

## 10. Academic Rigor Standards

### Literature Review Requirements

Before implementing any estimator:
1. **Read 3+ authoritative sources** (textbooks, survey papers)
2. **Compare variance estimators** across sources
3. **Document methodological debates** (e.g., TWFE bias)
4. **Cite modern methods** (post-2015 preferred)

### Citation Standards

Use academic citation format in all docstrings:
```julia
# References

- Neyman, J. (1923). On the Application of Probability Theory to Agricultural
  Experiments. *Statistical Science*, 5(4), 465-472.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference in Statistics, Social,
  and Biomedical Sciences*. Cambridge University Press.
```

### Methodological Concerns Documentation

When implementing estimators with known issues:
1. **Document in METHODOLOGICAL_CONCERNS.md** with severity level
2. **Add warning in estimator docstring**
3. **Implement diagnostic tests** if available
4. **Suggest robust alternatives**

Example: TWFE bias in DiD (Phase 3)
- Severity: CRITICAL
- Mitigation: Implement Callaway-Sant'Anna (2020) as alternative
- Warning in TWFE docstring about staggered adoption + heterogeneous TE

---

## 11. Continuous Improvement

### Version Control Standards

- **Commit messages**: Follow conventional commits (feat:, fix:, test:, docs:)
- **Branch strategy**: Feature branches for new estimators
- **Code review**: All estimators reviewed before merge
- **Documentation**: Update ROADMAP.md with every phase completion

### Audit Process

After each phase completion:
1. **Run full test suite** with coverage report
2. **Review all docstrings** for completeness
3. **Check METHODOLOGICAL_CONCERNS.md** for new issues
4. **Update standards** based on lessons learned
5. **Document decisions** in ROADMAP Decision Log

---

## Appendix A: Phase 1 Report Card (Grade: A+, 98/100)

### Strengths
- ✅ **Six-layer validation architecture**: Comprehensive, multi-faceted testing
- ✅ **49 adversarial tests**: Catches edge cases normal tests miss
- ✅ **Monte Carlo validation**: Proves statistical properties with known ground truth
- ✅ **HC3 documentation**: Clear justification for variance estimator choice
- ✅ **Explicit error messages**: Users know exactly what went wrong
- ✅ **SciML pattern**: Clean, composable architecture
- ✅ **219 tests passing**: High coverage, deterministic, fast

### Minor Gaps (-2 points)
- ⚠️ Coverage not tested at multiple alpha levels (0.01, 0.05, 0.10)
- ⚠️ No power analysis for sample size planning
- ⚠️ Type I error rate not empirically verified

### Recommendations for Future Phases
1. Add multi-alpha coverage tests (Phases 2-8)
2. Implement power calculators (Phase 2: IV, Phase 3: DiD)
3. Add Type I error verification to Monte Carlo tests

---

## Appendix B: Required Reading

Before implementing any phase:

**Textbooks**:
- Imbens & Rubin (2015). *Causal Inference in Statistics, Social, and Biomedical Sciences*
- Angrist & Pischke (2009). *Mostly Harmless Econometrics*
- Morgan & Winship (2015). *Counterfactuals and Causal Inference*

**Modern Methods**:
- Callaway & Sant'Anna (2020). Difference-in-Differences with multiple time periods. *Journal of Econometrics*
- Sun & Abraham (2021). Estimating dynamic treatment effects. *Journal of Econometrics*
- Borusyak et al. (2024). Revisiting Event-Study Designs. *Review of Economic Studies*

**Variance Estimation**:
- Long & Ervin (2000). Using Heteroscedasticity Consistent Standard Errors. *The American Statistician*
- MacKinnon & White (1985). Heteroskedasticity-consistent covariance matrix estimators. *Journal of Econometrics*

---

## Enforcement

These standards are **MANDATORY**, not aspirational.

**Before marking a phase complete**:
1. Run `julia --project=. test/governance/check_phase_standards.jl --phase=[N]`
2. Address ALL failures
3. Get peer review
4. Update ROADMAP.md with completion status

**Violations**: Phases marked "complete" without meeting standards will be REOPENED.

---

**Last Updated**: 2025-11-14
**Next Review**: After Phase 2 completion
**Maintained by**: Project Lead
