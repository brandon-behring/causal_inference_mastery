"""
Plot specification types for causal inference visualizations.

Following SciML pattern: plotting is opt-in via Requires.jl (loaded when user loads Plots.jl).
"""

"""
    PlotConfig

Configuration for plots (colors, sizes, labels, etc.).

# Fields
- `color_treated::Symbol`: Color for treated group (default: :blue)
- `color_control::Symbol`: Color for control group (default: :red)
- `alpha::Float64`: Transparency (default: 0.7)
- `markersize::Int`: Size of points (default: 4)
- `linewidth::Int`: Width of lines (default: 2)
- `grid::Bool`: Show grid (default: true)
- `legend::Symbol`: Legend position (default: :topright)
- `title::String`: Plot title (default: "")

# Example
```julia
config = PlotConfig(
    color_treated=:steelblue,
    color_control=:coral,
    title="Treatment Effect Estimate"
)
```
"""
Base.@kwdef struct PlotConfig
    color_treated::Symbol = :blue
    color_control::Symbol = :red
    color_matched::Symbol = :green
    color_unmatched::Symbol = :gray
    alpha::Float64 = 0.7
    markersize::Int = 4
    linewidth::Int = 2
    grid::Bool = true
    legend::Symbol = :topright
    title::String = ""
    xlabel::String = ""
    ylabel::String = ""
end

"""
    PlotType

Abstract type for plot specifications.
"""
abstract type PlotType end

"""
    TreatmentEffectPlot <: PlotType

Plot treatment effect estimate with confidence interval.

Shows point estimate as dot with error bar showing 95% CI.
"""
struct TreatmentEffectPlot <: PlotType end

"""
    DistributionPlot <: PlotType

Plot distribution of outcomes by treatment group.

Shows histograms or density plots for treated vs control.
"""
struct DistributionPlot <: PlotType end

"""
    BalancePlot <: PlotType

Plot covariate balance (PSM).

Shows standardized mean differences before and after matching.
"""
struct BalancePlot <: PlotType end

"""
    PropensityPlot <: PlotType

Plot propensity score distribution (PSM).

Shows overlap between treated and control propensity distributions.
"""
struct PropensityPlot <: PlotType end

"""
    MatchingPlot <: PlotType

Plot matched vs unmatched units (PSM).

Shows which units were successfully matched.
"""
struct MatchingPlot <: PlotType end
