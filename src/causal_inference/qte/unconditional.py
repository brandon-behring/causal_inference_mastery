"""
Unconditional Quantile Treatment Effects estimation.

This module implements simple quantile-difference estimators with bootstrap inference,
suitable for RCTs or quasi-experimental settings with unconfounded treatment.

The unconditional QTE at quantile tau is:
    QTE(tau) = Q_tau(Y | T=1) - Q_tau(Y | T=0)

References
----------
- Koenker, R., & Bassett Jr, G. (1978). Regression Quantiles. Econometrica.
- Firpo, S. (2007). Efficient Semiparametric Estimation of Quantile Treatment Effects.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import stats

from .types import QTEBandResult, QTEResult


def unconditional_qte(
    outcome: Union[np.ndarray, List[float]],
    treatment: Union[np.ndarray, List[int]],
    quantile: float = 0.5,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> QTEResult:
    """
    Estimate unconditional QTE: Q_tau(Y|T=1) - Q_tau(Y|T=0).

    Simple difference-in-quantiles with bootstrap inference.
    Valid for RCTs or conditional on propensity score weights.

    Parameters
    ----------
    outcome : np.ndarray or list
        Outcome variable Y of shape (n,). Must be numeric.
    treatment : np.ndarray or list
        Binary treatment indicator T of shape (n,). Must be {0, 1}.
    quantile : float, default=0.5
        The quantile tau in (0, 1) at which to estimate the effect.
        Common values: 0.1 (10th percentile), 0.25 (Q1), 0.5 (median),
        0.75 (Q3), 0.9 (90th percentile).
    n_bootstrap : int, default=1000
        Number of bootstrap replications for standard error estimation.
    alpha : float, default=0.05
        Significance level for confidence interval. Default gives 95% CI.
    random_state : int, optional
        Random seed for reproducibility of bootstrap.

    Returns
    -------
    QTEResult
        Dictionary containing:
        - tau_q: QTE estimate
        - se: Bootstrap standard error
        - ci_lower, ci_upper: Confidence interval bounds
        - quantile: The quantile tau
        - method: "unconditional"
        - n_treated, n_control, n_total: Sample sizes
        - outcome_support: (min, max) of Y
        - inference: "bootstrap"

    Raises
    ------
    ValueError
        If inputs are invalid (empty, mismatched lengths, NaN, inf,
        non-binary treatment, no variation, invalid quantile/alpha).

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> # Generate RCT data with treatment effect of 2.0
    >>> n = 200
    >>> treatment = np.random.binomial(1, 0.5, n)
    >>> outcome = np.random.normal(0, 1, n) + 2.0 * treatment
    >>> result = unconditional_qte(outcome, treatment, quantile=0.5)
    >>> print(f"Median treatment effect: {result['tau_q']:.3f}")

    Notes
    -----
    This estimator assumes:
    1. Random or as-good-as-random treatment assignment (ignorability)
    2. Stable unit treatment value assumption (SUTVA)
    3. Sufficient sample size in each treatment arm (min 10 per arm recommended)

    For observational data with confounding, consider using propensity score
    weights or the conditional QTE with appropriate covariates.
    """
    # ========================================================================
    # INPUT VALIDATION - Fail Fast with Diagnostic Info
    # ========================================================================

    # Convert to numpy arrays
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    n = len(outcome)

    # 1. Check for empty arrays
    if n == 0 or len(treatment) == 0:
        raise ValueError(
            f"CRITICAL ERROR: Empty input arrays.\n"
            f"Function: unconditional_qte\n"
            f"Expected: Non-empty arrays\n"
            f"Got: len(outcome)={n}, len(treatment)={len(treatment)}"
        )

    # 2. Check for length mismatch
    if len(outcome) != len(treatment):
        raise ValueError(
            f"CRITICAL ERROR: Arrays have different lengths.\n"
            f"Function: unconditional_qte\n"
            f"Expected: Same length arrays\n"
            f"Got: len(outcome)={len(outcome)}, len(treatment)={len(treatment)}"
        )

    # 3. Check for NaN values
    if np.any(np.isnan(outcome)) or np.any(np.isnan(treatment)):
        raise ValueError(
            f"CRITICAL ERROR: NaN values detected.\n"
            f"Function: unconditional_qte\n"
            f"Got: {np.sum(np.isnan(outcome))} NaN in outcome, "
            f"{np.sum(np.isnan(treatment))} NaN in treatment"
        )

    # 4. Check for infinite values
    if np.any(np.isinf(outcome)) or np.any(np.isinf(treatment)):
        raise ValueError(
            f"CRITICAL ERROR: Infinite values detected.\n"
            f"Function: unconditional_qte\n"
            f"Got: {np.sum(np.isinf(outcome))} inf in outcome, "
            f"{np.sum(np.isinf(treatment))} inf in treatment"
        )

    # 5. Check for binary treatment
    unique_treatment = np.unique(treatment)
    if not np.all(np.isin(unique_treatment, [0, 1])):
        raise ValueError(
            f"CRITICAL ERROR: Treatment must be binary (0 or 1).\n"
            f"Function: unconditional_qte\n"
            f"Expected: Treatment values in {{0, 1}}\n"
            f"Got: Unique treatment values = {unique_treatment}"
        )

    # 6. Check for treatment variation
    if len(unique_treatment) < 2:
        raise ValueError(
            f"CRITICAL ERROR: No treatment variation.\n"
            f"Function: unconditional_qte\n"
            f"Got: All units have treatment={unique_treatment[0]}"
        )

    # 7. Validate quantile
    if quantile <= 0 or quantile >= 1:
        raise ValueError(
            f"CRITICAL ERROR: Invalid quantile value.\n"
            f"Function: unconditional_qte\n"
            f"Expected: quantile in (0, 1)\n"
            f"Got: quantile={quantile}"
        )

    # 8. Validate alpha
    if alpha <= 0 or alpha >= 1:
        raise ValueError(
            f"CRITICAL ERROR: Invalid alpha value.\n"
            f"Function: unconditional_qte\n"
            f"Expected: alpha in (0, 1)\n"
            f"Got: alpha={alpha}"
        )

    # 9. Check minimum sample sizes
    n_treated = int(np.sum(treatment == 1))
    n_control = int(np.sum(treatment == 0))

    if n_treated < 2:
        raise ValueError(
            f"CRITICAL ERROR: Insufficient treated observations.\n"
            f"Function: unconditional_qte\n"
            f"Expected: At least 2 treated observations\n"
            f"Got: n_treated={n_treated}"
        )

    if n_control < 2:
        raise ValueError(
            f"CRITICAL ERROR: Insufficient control observations.\n"
            f"Function: unconditional_qte\n"
            f"Expected: At least 2 control observations\n"
            f"Got: n_control={n_control}"
        )

    # ========================================================================
    # QTE ESTIMATION
    # ========================================================================

    # Split outcomes by treatment status
    y_treated = outcome[treatment == 1]
    y_control = outcome[treatment == 0]

    # Point estimate: difference in quantiles
    q_treated = np.quantile(y_treated, quantile)
    q_control = np.quantile(y_control, quantile)
    tau_q = q_treated - q_control

    # ========================================================================
    # BOOTSTRAP INFERENCE
    # ========================================================================

    rng = np.random.default_rng(random_state)
    bootstrap_estimates = _bootstrap_qte(
        y_treated=y_treated,
        y_control=y_control,
        quantile=quantile,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )

    # Standard error from bootstrap distribution
    se = np.std(bootstrap_estimates, ddof=1)

    # Percentile-based confidence interval
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

    # ========================================================================
    # RETURN RESULT
    # ========================================================================

    return QTEResult(
        tau_q=float(tau_q),
        se=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        quantile=quantile,
        method="unconditional",
        n_treated=n_treated,
        n_control=n_control,
        n_total=n,
        outcome_support=(float(outcome.min()), float(outcome.max())),
        inference="bootstrap",
        pvalue=None,  # Not computed for unconditional
    )


def unconditional_qte_band(
    outcome: Union[np.ndarray, List[float]],
    treatment: Union[np.ndarray, List[int]],
    quantiles: Optional[List[float]] = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    joint: bool = False,
    random_state: Optional[int] = None,
) -> QTEBandResult:
    """
    Estimate QTE across multiple quantiles with optional joint confidence band.

    Provides a complete picture of treatment effect heterogeneity across the
    outcome distribution. Useful for detecting whether treatment benefits
    some parts of the distribution more than others.

    Parameters
    ----------
    outcome : np.ndarray or list
        Outcome variable Y of shape (n,).
    treatment : np.ndarray or list
        Binary treatment indicator T of shape (n,).
    quantiles : list of float, optional
        Quantiles at which to estimate QTE. Default is [0.1, 0.25, 0.5, 0.75, 0.9].
        All values must be in (0, 1).
    n_bootstrap : int, default=1000
        Number of bootstrap replications.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    joint : bool, default=False
        If True, compute joint (uniform) confidence band that provides
        simultaneous coverage across all quantiles.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    QTEBandResult
        Dictionary containing arrays of estimates across quantiles.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> treatment = np.random.binomial(1, 0.5, n)
    >>> # Heterogeneous treatment effect: larger at upper quantiles
    >>> outcome = np.random.normal(0, 1, n) + (1 + np.random.normal(0, 0.5, n)) * treatment
    >>> result = unconditional_qte_band(outcome, treatment)
    >>> print(f"Median QTE: {result['qte_estimates'][2]:.3f}")

    Notes
    -----
    Joint confidence bands use bootstrap critical values based on the
    supremum of the t-statistic process across quantiles. These are wider
    than pointwise intervals but provide valid uniform inference.
    """
    # Default quantiles
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    # Convert inputs
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    quantiles_arr = np.asarray(quantiles)

    # Validate quantiles
    if np.any(quantiles_arr <= 0) or np.any(quantiles_arr >= 1):
        raise ValueError(
            f"CRITICAL ERROR: All quantiles must be in (0, 1).\n"
            f"Function: unconditional_qte_band\n"
            f"Got: quantiles={quantiles}"
        )

    n = len(outcome)
    n_quantiles = len(quantiles)

    # Basic validation (defer detailed validation to unconditional_qte)
    if n == 0:
        raise ValueError("CRITICAL ERROR: Empty input arrays.")

    # Split by treatment
    y_treated = outcome[treatment == 1]
    y_control = outcome[treatment == 0]
    n_treated = len(y_treated)
    n_control = len(y_control)

    if n_treated < 2 or n_control < 2:
        raise ValueError(
            f"CRITICAL ERROR: Insufficient observations.\n"
            f"Got: n_treated={n_treated}, n_control={n_control}"
        )

    # ========================================================================
    # POINT ESTIMATES
    # ========================================================================

    qte_estimates = np.zeros(n_quantiles)
    for i, q in enumerate(quantiles):
        qte_estimates[i] = np.quantile(y_treated, q) - np.quantile(y_control, q)

    # ========================================================================
    # BOOTSTRAP INFERENCE
    # ========================================================================

    rng = np.random.default_rng(random_state)

    # Bootstrap matrix: (n_bootstrap, n_quantiles)
    bootstrap_matrix = np.zeros((n_bootstrap, n_quantiles))

    for b in range(n_bootstrap):
        # Resample within each treatment group
        idx_t = rng.choice(n_treated, n_treated, replace=True)
        idx_c = rng.choice(n_control, n_control, replace=True)

        y_t_boot = y_treated[idx_t]
        y_c_boot = y_control[idx_c]

        for i, q in enumerate(quantiles):
            bootstrap_matrix[b, i] = np.quantile(y_t_boot, q) - np.quantile(y_c_boot, q)

    # Pointwise standard errors and CIs
    se_estimates = np.std(bootstrap_matrix, axis=0, ddof=1)
    ci_lower = np.percentile(bootstrap_matrix, 100 * alpha / 2, axis=0)
    ci_upper = np.percentile(bootstrap_matrix, 100 * (1 - alpha / 2), axis=0)

    # ========================================================================
    # JOINT CONFIDENCE BAND (optional)
    # ========================================================================

    joint_ci_lower: Optional[np.ndarray] = None
    joint_ci_upper: Optional[np.ndarray] = None

    if joint:
        # Compute supremum of t-statistics across quantiles
        # t_b = max_q |QTE_b(q) - QTE(q)| / SE(q)
        t_stats = np.abs(bootstrap_matrix - qte_estimates) / np.maximum(se_estimates, 1e-10)
        sup_t_stats = np.max(t_stats, axis=1)

        # Critical value: (1-alpha) quantile of supremum distribution
        critical_value = np.percentile(sup_t_stats, 100 * (1 - alpha))

        # Joint band: point estimate +/- critical_value * SE
        joint_ci_lower = qte_estimates - critical_value * se_estimates
        joint_ci_upper = qte_estimates + critical_value * se_estimates

    # ========================================================================
    # RETURN RESULT
    # ========================================================================

    return QTEBandResult(
        quantiles=quantiles_arr,
        qte_estimates=qte_estimates,
        se_estimates=se_estimates,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        joint_ci_lower=joint_ci_lower,
        joint_ci_upper=joint_ci_upper,
        method="unconditional",
        n_bootstrap=n_bootstrap,
        n_treated=n_treated,
        n_control=n_control,
        n_total=n,
        alpha=alpha,
    )


def _bootstrap_qte(
    y_treated: np.ndarray,
    y_control: np.ndarray,
    quantile: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute bootstrap distribution of QTE estimate.

    Resamples separately within treated and control groups to preserve
    group sample sizes (stratified bootstrap).

    Parameters
    ----------
    y_treated : np.ndarray
        Outcomes for treated units.
    y_control : np.ndarray
        Outcomes for control units.
    quantile : float
        Quantile at which to compute QTE.
    n_bootstrap : int
        Number of bootstrap replications.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Bootstrap QTE estimates, shape (n_bootstrap,).
    """
    n_t = len(y_treated)
    n_c = len(y_control)

    bootstrap_estimates = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Stratified resampling: resample within each group
        idx_t = rng.choice(n_t, n_t, replace=True)
        idx_c = rng.choice(n_c, n_c, replace=True)

        q_t = np.quantile(y_treated[idx_t], quantile)
        q_c = np.quantile(y_control[idx_c], quantile)

        bootstrap_estimates[b] = q_t - q_c

    return bootstrap_estimates
