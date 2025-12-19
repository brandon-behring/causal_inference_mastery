"""
Type definitions for bunching estimation.

Defines TypedDict result types for bunching analysis outputs.
"""

from typing import TypedDict, Optional
import numpy as np
from numpy.typing import NDArray


class CounterfactualResult(TypedDict):
    """Result from counterfactual density estimation.

    Attributes
    ----------
    bin_centers : NDArray[np.float64]
        Centers of histogram bins.
    actual_counts : NDArray[np.float64]
        Observed counts in each bin.
    counterfactual_counts : NDArray[np.float64]
        Estimated counterfactual counts (what would be observed without bunching).
    polynomial_coeffs : NDArray[np.float64]
        Coefficients of the fitted polynomial.
    polynomial_order : int
        Order of polynomial used.
    bunching_region : tuple[float, float]
        (lower, upper) bounds of excluded bunching region.
    r_squared : float
        R-squared of polynomial fit outside bunching region.
    n_bins : int
        Number of bins used.
    bin_width : float
        Width of each bin.
    """

    bin_centers: NDArray[np.float64]
    actual_counts: NDArray[np.float64]
    counterfactual_counts: NDArray[np.float64]
    polynomial_coeffs: NDArray[np.float64]
    polynomial_order: int
    bunching_region: tuple[float, float]
    r_squared: float
    n_bins: int
    bin_width: float


class BunchingResult(TypedDict):
    """Result from bunching estimation.

    Attributes
    ----------
    excess_mass : float
        Excess mass at the kink (b = B/h0, normalized by counterfactual height).
    excess_mass_se : float
        Standard error of excess mass (from bootstrap).
    excess_mass_count : float
        Raw excess count (B = actual - counterfactual in bunching region).
    elasticity : float
        Estimated elasticity (if kink parameters provided).
    elasticity_se : float
        Standard error of elasticity.
    kink_point : float
        Location of the kink.
    bunching_region : tuple[float, float]
        (lower, upper) bounds of bunching region.
    counterfactual : CounterfactualResult
        Full counterfactual estimation results.
    t1_rate : Optional[float]
        Tax/marginal rate below kink (for elasticity calculation).
    t2_rate : Optional[float]
        Tax/marginal rate above kink (for elasticity calculation).
    n_obs : int
        Total number of observations.
    n_bootstrap : int
        Number of bootstrap iterations used.
    convergence : bool
        Whether iterative integration converged.
    message : str
        Descriptive message about estimation.
    """

    excess_mass: float
    excess_mass_se: float
    excess_mass_count: float
    elasticity: float
    elasticity_se: float
    kink_point: float
    bunching_region: tuple[float, float]
    counterfactual: CounterfactualResult
    t1_rate: Optional[float]
    t2_rate: Optional[float]
    n_obs: int
    n_bootstrap: int
    convergence: bool
    message: str
