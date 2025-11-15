"""
Plotting functions for PSM estimators.

These functions require Plots.jl to be loaded. They are loaded via Requires.jl when user runs `using Plots`.
"""

"""
    plot(solution::PSMSolution, plot_type::BalancePlot=BalancePlot(); config=PlotConfig())

Plot covariate balance before and after matching.

Shows standardized mean differences (SMD) for all covariates.

# Arguments
- `solution::PSMSolution`: Solution from PSM estimator
- `plot_type::BalancePlot`: Plot specification (default: BalancePlot())
- `config::PlotConfig`: Plot configuration (optional)

# Returns
- Plots.jl plot object

# Example
```julia
using CausalEstimators
using Plots

problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
solution = solve(problem, NearestNeighborPSM(M=1))

# Plot balance
plot(solution)  # Default: balance plot

# Customize
plot(solution, config=PlotConfig(title="Covariate Balance After Matching"))
```

# Visual
```
    Standardized Mean Difference (SMD)
        Before (red) vs After (blue)
    |
 X1 |---●-------- ●                    |  ← Improved
 X2 |----------●-- ●                   |  ← Improved
 X3 |-------●----- ●                   |  ← Improved
    |            |
    +-0.5  0.0  0.1  0.5---------------|
             Good balance threshold (|SMD| < 0.1)
```
"""
function plot end  # Interface only

"""
    plot(solution::PSMSolution, plot_type::PropensityPlot; config=PlotConfig())

Plot propensity score distribution by treatment group.

Shows overlap between treated and control propensity distributions.

# Arguments
- `solution::PSMSolution`: Solution from PSM estimator
- `plot_type::PropensityPlot`: Plot specification
- `config::PlotConfig`: Plot configuration (optional)

# Returns
- Plots.jl plot object

# Example
```julia
using CausalEstimators
using Plots

solution = solve(problem, NearestNeighborPSM(M=1))

# Plot propensity overlap
plot(solution, PropensityPlot())
```

# Visual
```
    Propensity Score Distribution
    |
  8 +  ██        ██                     |
    |  ██  ██    ██  ██                 |
  6 +  ██  ██    ██  ██                 |
    |  ██  ██  ████  ██                 |
  4 +  ██  ██████████████               |  ← Overlap region
    |  ██  ██████████████  ██           |     (common support)
  2 +  ██████████████████████           |
    |  ██████████████████████  ██       |
  0 +-------------------------------------|
    0.0   0.2   0.4   0.6   0.8   1.0
         Control (red)  Treated (blue)
```
"""
# plot() for PropensityPlot defined in package extension

"""
    plot(solution::PSMSolution, plot_type::MatchingPlot; config=PlotConfig())

Plot matched vs unmatched units in covariate space.

# Arguments
- `solution::PSMSolution`: Solution from PSM estimator
- `plot_type::MatchingPlot`: Plot specification
- `config::PlotConfig`: Plot configuration (optional)

# Returns
- Plots.jl plot object

# Example
```julia
using CausalEstimators
using Plots

solution = solve(problem, NearestNeighborPSM(M=1, caliper=0.25))

# Plot matching results
plot(solution, MatchingPlot())
```

# Visual
```
    Matched vs Unmatched Units
    |
 10 +  ●                    ●           |
    |     ●  ●          ●      ●        |  ● Matched treated
  8 +  ●  ×  ●          ●  ×   ●        |  × Matched control
    |     ×     ●           ×      ●    |  ○ Unmatched treated
  6 +        ×  ○        ×           ●  |  + Unmatched control
    |  ×        +  ×           ×        |
  4 +                 +                 |
    |           +                       |
  2 +-----------------------------------|
    0     2     4     6     8    10
              Covariate Value
```
"""
# plot() for MatchingPlot defined in package extension

# Note: Actual implementations will be in ../ext/CausalEstimatorsPlots.jl
