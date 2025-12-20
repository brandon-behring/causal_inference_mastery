"""Causal inference methods: RCT, PSM, DiD, IV, RDD, Selection, Bounds, QTE."""

# Bounds module (Sessions 86-87)
from .bounds import (
    # Manski bounds
    manski_worst_case,
    manski_mtr,
    manski_mts,
    manski_mtr_mts,
    manski_iv,
    compare_bounds,
    # Lee bounds
    lee_bounds,
    lee_bounds_tightened,
    check_monotonicity,
    # Types
    ManskiBoundsResult,
    ManskiIVBoundsResult,
    LeeBoundsResult,
)

# QTE module (Session 88)
from .qte import (
    # Unconditional QTE
    unconditional_qte,
    unconditional_qte_band,
    # Conditional QTE
    conditional_qte,
    conditional_qte_band,
    # RIF-OLS
    rif_qte,
    rif_qte_band,
    # Types
    QTEResult,
    QTEBandResult,
)
