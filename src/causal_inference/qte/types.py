"""
Type definitions for Quantile Treatment Effects (QTE) estimation.

This module defines TypedDict result types for QTE estimators, ensuring
consistent return types across unconditional, conditional, and RIF-based methods.

References
----------
- Koenker, R., & Bassett Jr, G. (1978). Regression Quantiles. Econometrica.
- Firpo, S. (2007). Efficient Semiparametric Estimation of Quantile Treatment Effects.
- Firpo, S., Fortin, N., & Lemieux, T. (2009). Unconditional Quantile Regressions.
"""

from typing import Literal, Optional, Tuple, TypedDict, Union

import numpy as np


class QTEResult(TypedDict):
    """Result from QTE estimation at a specific quantile.

    Attributes
    ----------
    tau_q : float
        Quantile treatment effect estimate at quantile tau.
        Represents Q_tau(Y|T=1) - Q_tau(Y|T=0) for unconditional QTE,
        or the treatment coefficient from quantile regression for conditional QTE.

    se : float
        Standard error of tau_q estimate.
        Typically computed via bootstrap for unconditional QTE,
        or asymptotically for conditional QTE.

    ci_lower : float
        Lower bound of (1-alpha)% confidence interval.

    ci_upper : float
        Upper bound of (1-alpha)% confidence interval.

    quantile : float
        The quantile tau in (0, 1) at which the effect is estimated.
        Common values: 0.25 (Q1), 0.5 (median), 0.75 (Q3).

    method : str
        Estimation method used. One of:
        - "unconditional": Simple quantile difference
        - "conditional": Quantile regression with covariates
        - "rif": Recentered Influence Function OLS

    n_treated : int
        Number of treated observations.

    n_control : int
        Number of control observations.

    n_total : int
        Total number of observations.

    outcome_support : Tuple[float, float]
        (min, max) of outcome variable, for interpretation and bounds checking.

    inference : str
        Type of standard errors used. One of:
        - "bootstrap": Resampling-based inference
        - "asymptotic": Analytic standard errors (quantile regression)

    pvalue : Optional[float]
        P-value for H0: tau_q = 0. Only computed for conditional QTE.

    Notes
    -----
    For unconditional QTE, the interpretation is straightforward:
    tau_q = Q_tau(Y_1) - Q_tau(Y_0), the difference in quantiles between
    treated and control potential outcome distributions.

    For conditional QTE via quantile regression, the interpretation is:
    tau_q is the effect of treatment on the tau-th conditional quantile
    of Y given covariates X.

    For RIF-based QTE, the interpretation is the marginal effect on the
    unconditional quantile, even when conditioning on covariates.
    """

    tau_q: float
    se: float
    ci_lower: float
    ci_upper: float
    quantile: float
    method: str
    n_treated: int
    n_control: int
    n_total: int
    outcome_support: Tuple[float, float]
    inference: str
    pvalue: Optional[float]


class QTEBandResult(TypedDict):
    """Result from QTE estimation across multiple quantiles.

    This provides a complete picture of treatment effect heterogeneity
    across the outcome distribution.

    Attributes
    ----------
    quantiles : np.ndarray
        Array of quantile values tau in (0, 1), shape (n_quantiles,).

    qte_estimates : np.ndarray
        QTE estimates at each quantile, shape (n_quantiles,).

    se_estimates : np.ndarray
        Standard errors at each quantile, shape (n_quantiles,).

    ci_lower : np.ndarray
        Lower bounds of pointwise confidence intervals, shape (n_quantiles,).

    ci_upper : np.ndarray
        Upper bounds of pointwise confidence intervals, shape (n_quantiles,).

    joint_ci_lower : Optional[np.ndarray]
        Lower bounds of joint (uniform) confidence band, shape (n_quantiles,).
        Only computed if joint=True. Uses bootstrap critical values.

    joint_ci_upper : Optional[np.ndarray]
        Upper bounds of joint confidence band, shape (n_quantiles,).
        Only computed if joint=True.

    method : str
        Estimation method: "unconditional", "conditional", or "rif".

    n_bootstrap : int
        Number of bootstrap replications used for inference.

    n_treated : int
        Number of treated observations.

    n_control : int
        Number of control observations.

    n_total : int
        Total number of observations.

    alpha : float
        Significance level used for confidence intervals.

    Notes
    -----
    Joint confidence bands provide simultaneous coverage across all quantiles,
    protecting against multiple testing. Use these when making statements about
    the entire QTE function rather than individual points.

    The relationship: joint_ci is wider than pointwise ci because it accounts
    for the probability of *any* quantile falling outside the band.
    """

    quantiles: np.ndarray
    qte_estimates: np.ndarray
    se_estimates: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    joint_ci_lower: Optional[np.ndarray]
    joint_ci_upper: Optional[np.ndarray]
    method: str
    n_bootstrap: int
    n_treated: int
    n_control: int
    n_total: int
    alpha: float


# Type alias for quantile parameter validation
QuantileValue = float  # Must be in (0, 1)

# Inference method types
InferenceMethod = Literal["bootstrap", "asymptotic"]

# QTE estimation method types
QTEMethod = Literal["unconditional", "conditional", "rif"]
