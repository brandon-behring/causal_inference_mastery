"""Panel data methods for causal inference.

This module provides:
- Double Machine Learning with Correlated Random Effects (DML-CRE) for average
  treatment effects
- Panel Quantile Treatment Effects via RIF regression

Key Features
------------
- Stratified cross-fitting by unit (preserves panel structure)
- Mundlak projection: includes time-means X̄ᵢ as covariates
- Clustered standard errors at unit level
- Supports balanced and unbalanced panels

Main Functions
--------------
DML-CRE (Average Treatment Effects):
- dml_cre : DML-CRE with binary treatment
- dml_cre_continuous : DML-CRE with continuous treatment

Panel QTE (Quantile Treatment Effects):
- panel_rif_qte : RIF-OLS for single quantile
- panel_rif_qte_band : RIF-OLS across multiple quantiles
- panel_unconditional_qte : Simple quantile difference (baseline)

Data Structures
---------------
- PanelData : Panel data container with validation
- DMLCREResult : Result dataclass for DML-CRE
- PanelQTEResult : Result dataclass for single quantile
- PanelQTEBandResult : Result dataclass for multiple quantiles

References
----------
- Mundlak, Y. (1978). "On the pooling of time series and cross section data."
  Econometrica 46(1): 69-85.
- Chernozhukov et al. (2018). "Double/debiased machine learning for treatment
  and structural parameters."
- Firpo, S., Fortin, N., & Lemieux, T. (2009). "Unconditional Quantile Regressions."
  Econometrica 77(3): 953-973.
- Wooldridge, J. M. (2010). "Econometric Analysis of Cross Section and Panel Data."
"""

from .types import PanelData, DMLCREResult, PanelQTEResult, PanelQTEBandResult
from .dml_cre import dml_cre
from .dml_cre_continuous import dml_cre_continuous
from .panel_qte import panel_rif_qte, panel_rif_qte_band, panel_unconditional_qte

__all__ = [
    # Data types
    "PanelData",
    "DMLCREResult",
    "PanelQTEResult",
    "PanelQTEBandResult",
    # DML-CRE functions
    "dml_cre",
    "dml_cre_continuous",
    # Panel QTE functions
    "panel_rif_qte",
    "panel_rif_qte_band",
    "panel_unconditional_qte",
]
