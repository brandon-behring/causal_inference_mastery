#=
Dynamic Double Machine Learning Module

Sessions 159+: Dynamic treatment effects via g-estimation following
Lewis & Syrgkanis (2021).

This module provides:
- Dynamic DML for time series and panel data
- Sequential g-estimation algorithm
- Time-series aware cross-fitting strategies
- HAC-robust inference

# Example: Single Time Series
```julia
using CausalEstimators
using Random
Random.seed!(42)

# Simulate data
Y, D, X, true_effects = simulate_dynamic_dgp(n_obs=500, n_lags=3)

# Estimate dynamic treatment effects
result = dynamic_dml(Y, D, X; max_lag=2, n_folds=5)
println(summary(result))
```

# Example: Panel Data
```julia
using CausalEstimators
using Random
Random.seed!(42)

n_units, n_periods = 50, 20
n_obs = n_units * n_periods
unit_id = repeat(1:n_units, inner=n_periods)

# Your panel data
Y, D, X = ...  # Load data

result = dynamic_dml_panel(Y, D, X, unit_id; max_lag=2, n_folds=5)
println(summary(result))
```

# Reference
Lewis, G., & Syrgkanis, V. (2021). Double/Debiased Machine Learning for
Dynamic Treatment Effects via g-Estimation. arXiv:2002.07285.
=#
module Dynamic

# Types
include("types.jl")
using .DynamicTypes

# Cross-fitting strategies
include("cross_fitting.jl")
using .CrossFitting

# HAC inference
include("hac_inference.jl")
using .HACInference

# G-estimation algorithm
include("g_estimation.jl")
using .GEstimation

# Main estimator
include("dynamic_dml.jl")
using .DynamicDMLEstimator

# Re-export types
export DynamicDMLResult, TimeSeriesPanelData
export result_summary, dml_is_significant, get_lagged_data

# Re-export cross-fitting
export BlockedTimeSeriesSplit, RollingOriginSplit
export PanelStratifiedSplit, ProgressiveBlockSplit
export split_indices, n_splits, get_cross_validator

# Re-export HAC inference
export newey_west_variance, influence_function_se
export optimal_bandwidth, confidence_interval

# Re-export g-estimation
export sequential_g_estimation, aggregate_fold_estimates
export compute_cumulative_effect, compute_cumulative_influence

# Re-export main estimator
export dynamic_dml, dynamic_dml_panel, simulate_dynamic_dgp

end  # module Dynamic
