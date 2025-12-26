"""R Triangulation tests for external validation against R packages.

This module provides Layer 5 validation by comparing our Python implementations
against established R packages:
- PStrata (Li & Li 2023) for Principal Stratification
- DTRreg (Wallace et al. 2017) for Dynamic Treatment Regimes

Tests skip gracefully when R/rpy2 is unavailable.
"""

from .r_interface import (
    check_r_available,
    check_pstrata_installed,
    check_dtrreg_installed,
)

__all__ = [
    "check_r_available",
    "check_pstrata_installed",
    "check_dtrreg_installed",
]
