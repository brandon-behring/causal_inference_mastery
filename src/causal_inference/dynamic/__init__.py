"""Dynamic Double Machine Learning for Time Series Treatment Effects.

This module implements Dynamic DML following Lewis & Syrgkanis (2021) for
estimating treatment effects in time series and panel data settings.

Main Functions
--------------
dynamic_dml
    Main estimator for dynamic treatment effects with cross-fitting.
dynamic_dml_panel
    Convenience wrapper for panel data (unit-stratified cross-fitting).
simulate_dynamic_dgp
    Simulate data for validation and testing.

Cross-Fitting Strategies
------------------------
BlockedTimeSeriesSplit
    K-fold with contiguous time blocks.
RollingOriginSplit
    Expanding window (walk-forward validation).
PanelStratifiedSplit
    Split by unit for panel data.
ProgressiveBlockSplit
    Progressive training on past blocks.

Types
-----
DynamicDMLResult
    Result container with effects, standard errors, and diagnostics.
TimeSeriesPanelData
    Data structure for panel/time series.

References
----------
Lewis, G., & Syrgkanis, V. (2021). Double/Debiased Machine Learning for
Dynamic Treatment Effects via g-Estimation. arXiv:2002.07285.

Examples
--------
>>> import numpy as np
>>> from causal_inference.dynamic import dynamic_dml, simulate_dynamic_dgp
>>>
>>> # Simulate data
>>> Y, D, X, true_effects = simulate_dynamic_dgp(n_obs=500, seed=42)
>>>
>>> # Estimate dynamic treatment effects
>>> result = dynamic_dml(Y, D, X, max_lag=3)  # doctest: +SKIP
>>> print(result.summary())  # doctest: +SKIP
"""

from .cross_fitting import (
    BlockedTimeSeriesSplit,
    PanelStratifiedSplit,
    ProgressiveBlockSplit,
    RollingOriginSplit,
    get_cross_validator,
)
from .dynamic_dml import dynamic_dml, dynamic_dml_panel, simulate_dynamic_dgp
from .g_estimation import (
    aggregate_fold_estimates,
    compute_cumulative_effect,
    compute_cumulative_influence,
    sequential_g_estimation,
)
from .hac_inference import (
    clustered_hac_variance,
    confidence_interval,
    hac_ols_se,
    influence_function_se,
    newey_west_variance,
    optimal_bandwidth,
)
from .types import DynamicDMLResult, TimeSeriesPanelData, validate_dynamic_inputs

__all__ = [
    # Main estimators
    "dynamic_dml",
    "dynamic_dml_panel",
    "simulate_dynamic_dgp",
    # Result types
    "DynamicDMLResult",
    "TimeSeriesPanelData",
    # Cross-fitting
    "BlockedTimeSeriesSplit",
    "RollingOriginSplit",
    "PanelStratifiedSplit",
    "ProgressiveBlockSplit",
    "get_cross_validator",
    # G-estimation
    "sequential_g_estimation",
    "aggregate_fold_estimates",
    "compute_cumulative_effect",
    "compute_cumulative_influence",
    # HAC inference
    "newey_west_variance",
    "influence_function_se",
    "clustered_hac_variance",
    "hac_ols_se",
    "optimal_bandwidth",
    "confidence_interval",
    # Utilities
    "validate_dynamic_inputs",
]
