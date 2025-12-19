"""
Regression Kink Design (RKD) Module

Implements Sharp and Fuzzy RKD estimators with optimal bandwidth selection
and diagnostics for estimating causal effects at policy kinks.

Key Difference from RDD:
- RDD: Treatment effect = jump in LEVEL at cutoff
- RKD: Treatment effect = change in SLOPE at cutoff

Classes
-------
SharpRKD : Sharp regression kink design estimator
FuzzyRKD : Fuzzy regression kink design estimator (future)

Functions
---------
rkd_bandwidth : Optimal bandwidth for RKD (adapted from CCT)

Examples
--------
>>> from causal_inference.rkd import SharpRKD
>>> import numpy as np
>>>
>>> # Generate RKD data with kink at 0
>>> np.random.seed(42)
>>> X = np.random.uniform(-5, 5, 500)
>>> # Policy kinks: slope changes from 0.5 to 1.5 at X=0
>>> D = np.where(X < 0, 0.5 * X, 1.5 * X)
>>> Y = 2.0 * D + np.random.normal(0, 1, 500)  # True effect = 2.0
>>>
>>> # Fit Sharp RKD
>>> rkd = SharpRKD(cutoff=0.0)
>>> result = rkd.fit(Y, X, D)
>>> print(f"Kink effect: {result.estimate:.3f}")
Kink effect: 2.000

References
----------
Card, D., Lee, D. S., Pei, Z., & Weber, A. (2015). Inference on causal effects
    in a generalized regression kink design. Econometrica, 83(6), 2453-2483.
Nielsen, H. S., Sørensen, T., & Taber, C. (2010). Estimating the effect of
    student aid on college enrollment: Evidence from a government grant
    policy reform. American Economic Journal: Economic Policy, 2(2), 185-215.
Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric
    confidence intervals for regression-discontinuity designs.
    Econometrica, 82(6), 2295-2326.
"""

from .sharp_rkd import SharpRKD, SharpRKDResult
from .fuzzy_rkd import FuzzyRKD, FuzzyRKDResult
from .bandwidth import rkd_bandwidth, rkd_ik_bandwidth
from .diagnostics import (
    density_smoothness_test,
    covariate_smoothness_test,
    first_stage_test,
    rkd_diagnostics_summary,
    DensitySmoothnessResult,
    CovariateSmoothnessResult,
    FirstStageResult,
)

__all__ = [
    # Estimators
    "SharpRKD",
    "SharpRKDResult",
    "FuzzyRKD",
    "FuzzyRKDResult",
    # Bandwidth
    "rkd_bandwidth",
    "rkd_ik_bandwidth",
    # Diagnostics
    "density_smoothness_test",
    "covariate_smoothness_test",
    "first_stage_test",
    "rkd_diagnostics_summary",
    "DensitySmoothnessResult",
    "CovariateSmoothnessResult",
    "FirstStageResult",
]

__version__ = "0.2.0"
