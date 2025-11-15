"""
Plotting functions for RCT estimators.

These functions require Plots.jl to be loaded. They are loaded via Requires.jl when user runs `using Plots`.

# Note
The actual plot() implementations are in ../ext/CausalEstimatorsPlots.jl (package extension).
This file defines the interface and documentation.
"""

"""
    plot(solution::RCTSolution, plot_type::TreatmentEffectPlot=TreatmentEffectPlot(); config=PlotConfig())

Plot treatment effect estimate with confidence interval.

# Arguments
- `solution::RCTSolution`: Solution from RCT estimator
- `plot_type::TreatmentEffectPlot`: Plot specification (default: TreatmentEffectPlot())
- `config::PlotConfig`: Plot configuration (optional)

# Returns
- Plots.jl plot object

# Example
```julia
using CausalEstimators
using Plots

# Solve problem
problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))
solution = solve(problem, SimpleATE())

# Plot treatment effect
plot(solution)  # Default: treatment effect with CI

# Customize
plot(solution, config=PlotConfig(title="Estimated ATE", color_treated=:steelblue))
```

# Visual
```
       Treatment Effect Estimate
       |
    15 +------------------------+
       |           ●             |  ← Point estimate
    10 +           |             |
       |           |             |
     5 +       [---+---]         |  ← 95% CI
       |                         |
     0 +-------------------------+
       Control  Treatment  Effect
```
"""
function plot end  # Interface only - implementation in package extension

"""
    plot(problem::RCTProblem, solution::RCTSolution, plot_type::DistributionPlot; config=PlotConfig())

Plot distribution of outcomes by treatment group.

# Arguments
- `problem::RCTProblem`: Original problem with data
- `solution::RCTSolution`: Solution from estimator
- `plot_type::DistributionPlot`: Plot specification
- `config::PlotConfig`: Plot configuration (optional)

# Returns
- Plots.jl plot object

# Example
```julia
using CausalEstimators
using Plots

problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))
solution = solve(problem, SimpleATE())

# Plot outcome distributions
plot(problem, solution, DistributionPlot())
```

# Visual
```
    Outcome Distribution by Treatment Group
    |
  8 +  ██                       ██         |
    |  ██                       ██         |
  6 +  ██                       ██         |
    |  ██  ██                   ██  ██     |
  4 +  ██  ██                   ██  ██     |
    |  ██  ██  ██               ██  ██  ██ |
  2 +  ██  ██  ██               ██  ██  ██ |
    |  ██  ██  ██               ██  ██  ██ |
  0 +-------------------------------------|
      Control (red)          Treated (blue)
```
"""
# plot() for DistributionPlot defined in package extension

# Note: Actual implementations will be in ../ext/CausalEstimatorsPlots.jl
# when user loads Plots.jl
