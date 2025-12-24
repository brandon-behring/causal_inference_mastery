"""
Bunching Estimation Module.

Implements methods for estimating behavioral responses from bunching at kinks
in budget constraints (e.g., tax kinks, subsidy thresholds).

Key Methods:
- `bunching_estimator()`: Main estimation function (Saez 2010, Chetty et al. 2011)
- `estimate_counterfactual()`: Polynomial counterfactual density
- `compute_excess_mass()`: Bunching mass calculation
- `compute_elasticity()`: Behavioral elasticity from bunching

References:
- Saez, E. (2010). "Do Taxpayers Bunch at Kink Points?" American Economic
    Journal: Economic Policy.
- Chetty, R., Friedman, J. N., Olsen, T., & Pistaferri, L. (2011). "Adjustment
    Costs, Firm Responses, and Micro vs. Macro Labor Supply Elasticities."
    Quarterly Journal of Economics.
- Kleven, H. J. (2016). "Bunching." Annual Review of Economics.
"""

from .types import (
    BunchingResult,
    CounterfactualResult,
)
from .counterfactual import (
    estimate_counterfactual,
    polynomial_counterfactual,
)
from .excess_mass import (
    bunching_estimator,
    compute_excess_mass,
    compute_elasticity,
    bootstrap_bunching_se,
)

__all__ = [
    # Result types
    "BunchingResult",
    "CounterfactualResult",
    # Counterfactual estimation
    "estimate_counterfactual",
    "polynomial_counterfactual",
    # Main estimator
    "bunching_estimator",
    "compute_excess_mass",
    "compute_elasticity",
    "bootstrap_bunching_se",
]
