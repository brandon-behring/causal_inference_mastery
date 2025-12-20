"""
Quantile Treatment Effects (QTE) estimation module.

This module provides methods for estimating treatment effects at different
quantiles of the outcome distribution, answering questions like:
"Does the treatment help those at the bottom/top of the distribution differently?"

Three estimation approaches are available:

1. **Unconditional QTE** (simple quantile differences)
   - `unconditional_qte()`: Single quantile estimate with bootstrap CI
   - `unconditional_qte_band()`: Estimates across multiple quantiles

2. **Conditional QTE** (quantile regression with covariates)
   - `conditional_qte()`: Single quantile via statsmodels QuantReg
   - `conditional_qte_band()`: Estimates across multiple quantiles

3. **RIF-OLS** (Firpo et al. 2009 unconditional with covariates)
   - `rif_qte()`: Unconditional effect via Recentered Influence Functions
   - `rif_qte_band()`: RIF estimates across multiple quantiles

Key Distinction:
- Unconditional QTE: Effect on marginal distribution Q_tau(Y)
- Conditional QTE: Effect on conditional distribution Q_tau(Y|X)
- RIF-OLS: Unconditional effect Q_tau(Y) while controlling for X

References
----------
- Koenker, R., & Bassett Jr, G. (1978). Regression Quantiles. Econometrica.
- Firpo, S. (2007). Efficient Semiparametric Estimation of Quantile Treatment Effects.
- Firpo, S., Fortin, N., & Lemieux, T. (2009). Unconditional Quantile Regressions.

Examples
--------
>>> import numpy as np
>>> from causal_inference.qte import unconditional_qte, unconditional_qte_band

>>> # Simple RCT data
>>> np.random.seed(42)
>>> n = 200
>>> treatment = np.random.binomial(1, 0.5, n)
>>> outcome = np.random.normal(0, 1, n) + 2.0 * treatment

>>> # Estimate median treatment effect
>>> result = unconditional_qte(outcome, treatment, quantile=0.5)
>>> print(f"Median QTE: {result['tau_q']:.3f} (SE: {result['se']:.3f})")

>>> # Estimate QTE across multiple quantiles
>>> band = unconditional_qte_band(outcome, treatment)
>>> for q, qte in zip(band['quantiles'], band['qte_estimates']):
...     print(f"QTE at {q:.2f}: {qte:.3f}")
"""

from .conditional import conditional_qte, conditional_qte_band
from .rif import rif_qte, rif_qte_band
from .types import QTEBandResult, QTEResult
from .unconditional import unconditional_qte, unconditional_qte_band

__all__ = [
    # Types
    "QTEResult",
    "QTEBandResult",
    # Unconditional QTE
    "unconditional_qte",
    "unconditional_qte_band",
    # Conditional QTE
    "conditional_qte",
    "conditional_qte_band",
    # RIF-OLS
    "rif_qte",
    "rif_qte_band",
]
