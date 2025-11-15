"""
Universal solve() interface for causal estimation.

Following SciML pattern: `solution = solve(problem, estimator)`

This file defines the main dispatch interface. Actual implementations are in
estimator-specific files (e.g., `estimators/rct/simple_ate.jl`).

# Examples

```julia
# Create problem
problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))

# Solve with different estimators
solution1 = solve(problem, SimpleATE())
solution2 = solve(problem, StratifiedATE())  # Requires strata in problem
solution3 = solve(problem, RegressionATE())  # Requires covariates in problem
```

# Design

Multiple dispatch allows:
- Same interface across all methods
- Easy to add new estimators
- Type-stable (compiler knows exact types)
- Extensible (users can add methods without modifying package)
"""

"""
    solve(problem::AbstractCausalProblem, estimator::AbstractCausalEstimator)

Universal interface for causal estimation.

Dispatches to appropriate solver based on problem and estimator types.

# Arguments
- `problem`: Problem specification (e.g., `RCTProblem`)
- `estimator`: Estimator choice (e.g., `SimpleATE()`)

# Returns
- `AbstractCausalSolution`: Solution with results and metadata

# Throws
- `ArgumentError`: If problem-estimator combination is invalid
- `EstimationError`: If estimation fails

# Examples

```julia
problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))
solution = solve(problem, SimpleATE())

# Access results
solution.estimate      # Point estimate
solution.se            # Standard error
solution.ci_lower      # Lower CI
solution.ci_upper      # Upper CI
solution.retcode       # :Success
```
"""
function solve end

# Fallback for unsupported combinations
function solve(problem::AbstractCausalProblem, estimator::AbstractCausalEstimator)
    throw(
        ArgumentError(
            "CRITICAL ERROR: Unsupported problem-estimator combination.\n" *
            "Problem type: $(typeof(problem))\n" *
            "Estimator type: $(typeof(estimator))\n" *
            "This combination is not implemented. Check documentation for supported methods.",
        ),
    )
end

# Specific implementations are defined in estimator files:
# - estimators/rct/simple_ate.jl: solve(::RCTProblem, ::SimpleATE)
# - estimators/rct/stratified_ate.jl: solve(::RCTProblem, ::StratifiedATE)
# - etc.
