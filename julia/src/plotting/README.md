# Plotting Infrastructure

**Status**: ✅ Foundation Complete (Phase 2.5)
**Implementation**: Via package extension (loaded when user loads Plots.jl)

---

## Overview

CausalEstimators.jl provides plotting capabilities via opt-in package extensions (Julia 1.9+). Plots are **not** loaded by default to avoid dependency bloat. Users who want plotting capabilities simply load Plots.jl:

```julia
using CausalEstimators
using Plots  # This triggers the plotting extension

# Now plot() methods are available
solution = solve(problem, SimpleATE())
plot(solution)  # Works!
```

---

## Architecture

### Package Extension Pattern

**Why package extensions?**
- **Zero overhead**: Users without Plots.jl don't pay compilation cost
- **Optional dependency**: Plotting is opt-in, not required
- **Clean separation**: Plotting logic separate from estimation logic
- **SciML standard**: DifferentialEquations.jl, Optimization.jl use same pattern

**How it works**:
1. User loads `CausalEstimators` → No plotting functions available
2. User loads `Plots` → Extension `CausalEstimatorsPlots` activates
3. `plot()` methods become available for solutions

**Directory structure**:
```
src/plotting/
├── README.md               # This file
├── types.jl                # Plot specification types (always loaded)
├── rct_plots.jl            # RCT plot interface (docs only)
└── psm_plots.jl            # PSM plot interface (docs only)

ext/
└── CausalEstimatorsPlots/  # Package extension (loaded when Plots.jl loaded)
    ├── CausalEstimatorsPlots.jl  # Extension module
    ├── rct_plots_impl.jl   # RCT plot implementations
    └── psm_plots_impl.jl   # PSM plot implementations
```

---

## Available Plots

### RCT Plots

#### 1. Treatment Effect Plot (Default)

```julia
plot(solution::RCTSolution)
```

Shows point estimate with error bar for 95% CI.

**Use cases**:
- Quick visualization of treatment effect
- Presentation of results
- Sensitivity analysis comparison

**Example**:
```julia
solution = solve(problem, SimpleATE())
plot(solution, config=PlotConfig(title="Estimated ATE"))
```

#### 2. Distribution Plot

```julia
plot(problem::RCTProblem, solution::RCTSolution, DistributionPlot())
```

Shows outcome distributions by treatment group.

**Use cases**:
- Check distributional assumptions
- Identify outliers
- Visualize treatment effect in context

**Example**:
```julia
plot(problem, solution, DistributionPlot())
```

### PSM Plots

#### 1. Balance Plot (Default)

```julia
plot(solution::PSMSolution)
```

Shows standardized mean differences before and after matching.

**Use cases**:
- Verify balance improvement
- Identify problematic covariates
- Report balance diagnostics

**Example**:
```julia
solution = solve(problem, NearestNeighborPSM(M=1))
plot(solution)  # Shows SMD before/after
```

#### 2. Propensity Plot

```julia
plot(solution::PSMSolution, PropensityPlot())
```

Shows propensity score overlap between groups.

**Use cases**:
- Check common support assumption
- Identify propensity extremes
- Visualize positivity violation

**Example**:
```julia
plot(solution, PropensityPlot())
```

#### 3. Matching Plot

```julia
plot(solution::PSMSolution, MatchingPlot())
```

Shows matched vs unmatched units in covariate space.

**Use cases**:
- Visualize matching quality
- Identify unmatched regions
- Report matching coverage

**Example**:
```julia
plot(solution, MatchingPlot())
```

---

## Configuration

All plots accept optional `PlotConfig`:

```julia
config = PlotConfig(
    color_treated = :steelblue,    # Color for treated group
    color_control = :coral,        # Color for control group
    color_matched = :green,        # Color for matched units
    color_unmatched = :gray,       # Color for unmatched units
    alpha = 0.7,                   # Transparency
    markersize = 4,                # Point size
    linewidth = 2,                 # Line width
    grid = true,                   # Show grid
    legend = :topright,            # Legend position
    title = "My Plot",             # Plot title
    xlabel = "X-axis",             # X-axis label
    ylabel = "Y-axis"              # Y-axis label
)

plot(solution, config=config)
```

---

## Implementation Status

**Phase 2.5** (Current):
- ✅ Plot type specifications (`types.jl`)
- ✅ Interface definitions (`rct_plots.jl`, `psm_plots.jl`)
- ✅ Documentation in place
- ⏸️ Extension implementation pending (Phase 3.5)

**Phase 3.5** (Planned):
- Create `ext/CausalEstimatorsPlots/` directory
- Implement plot() methods for all plot types
- Add tests for plotting functions
- Add plotting examples to user guide

---

## Design Decisions

### 1. Package Extensions vs Requires.jl

**Chosen**: Package extensions (Julia 1.9+)

**Rationale**:
- Newer, cleaner approach
- Better precompilation
- Official Julia feature (not third-party)
- Future-proof

### 2. Plots.jl vs Makie.jl

**Chosen**: Plots.jl

**Rationale**:
- More mature ecosystem
- Simpler API for basic plots
- Better ecosystem compatibility
- Can add Makie backend later if needed

### 3. Plot Types as Structs

**Chosen**: Explicit plot type structs (TreatmentEffectPlot, BalancePlot, etc.)

**Rationale**:
- Type-safe dispatch
- Self-documenting (clear what plots are available)
- Extensible (users can add custom plot types)
- Follows Julia conventions

### 4. Default Plots

**Chosen**:
- `plot(solution::RCTSolution)` → TreatmentEffectPlot
- `plot(solution::PSMSolution)` → BalancePlot

**Rationale**:
- Most common use case
- Sensible defaults for each estimator type
- Users can specify other plot types explicitly

---

## Extension Implementation Guide

When implementing `ext/CausalEstimatorsPlots/CausalEstimatorsPlots.jl`:

```julia
module CausalEstimatorsPlots

using CausalEstimators
using Plots

# Import plot interface
import CausalEstimators: plot

# Include implementations
include("rct_plots_impl.jl")
include("psm_plots_impl.jl")

end
```

**Example implementation** (`rct_plots_impl.jl`):

```julia
# Treatment effect plot
function plot(solution::RCTSolution, ::TreatmentEffectPlot=TreatmentEffectPlot(); config=PlotConfig())
    # Extract data
    estimate = solution.estimate
    ci_lower = solution.ci_lower
    ci_upper = solution.ci_upper

    # Create plot
    p = Plots.plot(
        [1], [estimate],
        yerror = [(estimate - ci_lower, ci_upper - estimate)],
        marker = :circle,
        markersize = config.markersize,
        linewidth = config.linewidth,
        color = config.color_treated,
        legend = false,
        grid = config.grid,
        title = config.title == "" ? "Treatment Effect Estimate" : config.title,
        ylabel = "Effect Size",
        xticks = ([1], ["ATE"])
    )

    # Add zero reference line
    Plots.hline!([0], linestyle=:dash, color=:gray, label="")

    return p
end
```

---

## Testing Strategy

**Unit tests** (`test/plotting/test_rct_plots.jl`):
```julia
@testset "RCT Plotting" begin
    # Create test data
    problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))
    solution = solve(problem, SimpleATE())

    @testset "TreatmentEffectPlot" begin
        p = plot(solution)
        @test p isa Plots.Plot
        # Test that plot contains expected elements
    end

    @testset "DistributionPlot" begin
        p = plot(problem, solution, DistributionPlot())
        @test p isa Plots.Plot
    end
end
```

**Visual regression tests** (optional):
- Save reference images
- Compare new plots to reference
- Flag differences for manual review

---

## Future Extensions

1. **Interactive plots** (PlotlyJS backend):
   - Hover tooltips showing unit details
   - Zooming and panning
   - Export to HTML

2. **Diagnostic plots**:
   - Residual plots (RegressionATE)
   - QQ plots (normality check)
   - Influence plots (outlier detection)

3. **Comparison plots**:
   - Multiple estimators side-by-side
   - Sensitivity analysis curves
   - Subgroup analysis forest plots

4. **Publication-ready**:
   - LaTeX integration (PGFPlotsX)
   - Journal-specific formatting
   - Vector graphics export

---

## Summary

The plotting infrastructure provides:

✅ **Type-safe specifications**: PlotConfig, plot type structs
✅ **Interface definitions**: Documented plot() signatures
✅ **Opt-in loading**: Via package extensions (Julia 1.9+)
✅ **Extensible design**: Easy to add new plot types
✅ **Clear documentation**: Examples and use cases
✅ **Ready for implementation**: Extension skeleton prepared

**Status**: Foundation complete, implementation pending Phase 3.5

**Next steps**:
1. Create `ext/CausalEstimatorsPlots/` directory
2. Implement plot() methods
3. Add tests
4. Update user guide with plotting examples
