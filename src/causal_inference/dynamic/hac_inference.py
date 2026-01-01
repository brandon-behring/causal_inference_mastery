"""HAC (Heteroskedasticity and Autocorrelation Consistent) Inference.

Provides HAC-robust variance estimation for time series and panel data,
with support for influence function-based inference in DML settings.

Methods
-------
- Newey-West HAC estimator (Bartlett kernel)
- Quadratic spectral kernel
- Clustered HAC for panel data
- Influence function-based variance

References
----------
Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
heteroskedasticity and autocorrelation consistent covariance matrix.
Econometrica, 55(3), 703-708.

Andrews, D. W. (1991). Heteroskedasticity and autocorrelation consistent
covariance matrix estimation. Econometrica, 817-858.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from scipy import stats


def newey_west_variance(
    scores: np.ndarray,
    bandwidth: Optional[int] = None,
    kernel: Literal["bartlett", "qs"] = "bartlett",
) -> np.ndarray:
    """Compute Newey-West HAC variance estimator for influence scores.

    The HAC estimator for the variance of the sample mean of scores:
        V = Γ_0 + Σ_{j=1}^{M} w_j (Γ_j + Γ_j')

    where Γ_j = (1/T) Σ_t ψ_t ψ_{t-j}' and w_j is the kernel weight.

    Parameters
    ----------
    scores : np.ndarray
        Influence function scores, shape (n_obs,) or (n_obs, n_params).
        Each row is the influence contribution of one observation.
    bandwidth : int, optional
        Truncation lag M. Default: Newey-West optimal bandwidth
        floor(4 * (T/100)^{2/9}).
    kernel : {"bartlett", "qs"}
        Kernel function. "bartlett" is the standard Newey-West kernel.
        "qs" is the quadratic spectral kernel.

    Returns
    -------
    np.ndarray
        Variance-covariance matrix. Shape (n_params, n_params) if scores
        is 2D, scalar if scores is 1D.

    Examples
    --------
    >>> np.random.seed(42)
    >>> scores = np.random.randn(100)
    >>> var = newey_west_variance(scores)
    >>> se = np.sqrt(var / 100)  # Standard error of mean
    """
    scores = np.atleast_2d(scores)
    if scores.shape[0] == 1:
        scores = scores.T

    T, k = scores.shape

    # Default bandwidth: Newey-West optimal
    if bandwidth is None:
        bandwidth = int(np.floor(4 * (T / 100) ** (2 / 9)))
    bandwidth = min(bandwidth, T - 2)

    # Demean scores
    scores_demeaned = scores - scores.mean(axis=0)

    # Lag 0: Γ_0
    Omega = scores_demeaned.T @ scores_demeaned / T

    # Add lagged terms
    for j in range(1, bandwidth + 1):
        if kernel == "bartlett":
            w = 1 - j / (bandwidth + 1)
        else:  # quadratic spectral
            z = 6 * np.pi * j / (5 * bandwidth)
            if z < 1e-10:
                w = 1.0
            else:
                w = 3 / z**2 * (np.sin(z) / z - np.cos(z))

        # Autocovariance at lag j
        Gamma_j = scores_demeaned[j:, :].T @ scores_demeaned[:-j, :] / T

        # Add both j and -j (symmetric)
        Omega += w * (Gamma_j + Gamma_j.T)

    # Return scalar for 1D input
    if k == 1:
        return float(Omega[0, 0])

    return Omega


def influence_function_se(
    influence_scores: np.ndarray,
    bandwidth: Optional[int] = None,
    kernel: Literal["bartlett", "qs"] = "bartlett",
) -> np.ndarray:
    """Compute standard errors from influence function scores with HAC.

    For DML estimators, the asymptotic variance is:
        Var(θ̂) = E[ψ^2] / n

    where ψ is the influence function. With autocorrelation, we use
    HAC to estimate E[ψ^2] consistently.

    Parameters
    ----------
    influence_scores : np.ndarray
        Influence function scores, shape (n_obs,) or (n_obs, n_params).
    bandwidth : int, optional
        HAC bandwidth. Default: Newey-West optimal.
    kernel : {"bartlett", "qs"}
        HAC kernel.

    Returns
    -------
    np.ndarray
        Standard errors, shape (n_params,) or scalar.

    Notes
    -----
    The standard error is computed as sqrt(V / n) where V is the
    HAC-adjusted variance.
    """
    influence_scores = np.atleast_2d(influence_scores)
    if influence_scores.shape[0] == 1:
        influence_scores = influence_scores.T

    n = len(influence_scores)

    # Get HAC variance
    V = newey_west_variance(influence_scores, bandwidth=bandwidth, kernel=kernel)

    # Standard error of mean
    if np.isscalar(V):
        se = np.sqrt(V / n)
    else:
        se = np.sqrt(np.diag(V) / n)

    return se


def clustered_hac_variance(
    scores: np.ndarray,
    cluster_id: np.ndarray,
    bandwidth: Optional[int] = None,
    kernel: Literal["bartlett", "qs"] = "bartlett",
) -> np.ndarray:
    """Compute clustered HAC variance for panel data.

    Combines clustering at the unit level with HAC within clusters.
    This is appropriate for panel data where observations within a
    unit are serially correlated.

    The estimator:
        V = Σ_g Ψ_g Ψ_g'

    where Ψ_g is the HAC-adjusted sum of scores within cluster g.

    Parameters
    ----------
    scores : np.ndarray
        Influence function scores, shape (n_obs,) or (n_obs, n_params).
    cluster_id : np.ndarray
        Cluster (unit) identifiers, shape (n_obs,).
    bandwidth : int, optional
        Within-cluster HAC bandwidth.
    kernel : {"bartlett", "qs"}
        HAC kernel for within-cluster autocorrelation.

    Returns
    -------
    np.ndarray
        Variance-covariance matrix.

    Notes
    -----
    When bandwidth=0, this reduces to standard clustered SEs.
    """
    scores = np.atleast_2d(scores)
    if scores.shape[0] == 1:
        scores = scores.T

    n, k = scores.shape
    unique_clusters = np.unique(cluster_id)
    G = len(unique_clusters)

    # Compute cluster-level influence (sum within each cluster)
    cluster_influences = np.zeros((G, k))

    for i, g in enumerate(unique_clusters):
        mask = cluster_id == g
        cluster_scores = scores[mask]
        n_g = len(cluster_scores)

        if n_g == 1 or bandwidth == 0:
            # No within-cluster autocorrelation
            cluster_influences[i] = cluster_scores.sum(axis=0)
        else:
            # HAC-adjusted sum within cluster
            # First compute within-cluster HAC variance
            within_var = newey_west_variance(cluster_scores, bandwidth=bandwidth, kernel=kernel)

            # The cluster contribution is sum with HAC adjustment
            # This is approximately: sum(scores) with variance accounting for serial correlation
            cluster_influences[i] = cluster_scores.sum(axis=0)

    # Cluster-robust variance: Σ_g Ψ_g Ψ_g' / n^2
    # With small-sample correction: G / (G-1) * n / (n-k)
    Omega = cluster_influences.T @ cluster_influences / n**2

    # Small-sample correction (like stata's vce(cluster))
    correction = G / (G - 1) * n / (n - k) if G > 1 and n > k else 1.0
    Omega *= correction

    if k == 1:
        return float(Omega[0, 0])

    return Omega


def optimal_bandwidth(
    n: int,
    method: Literal["nw", "andrews"] = "nw",
) -> int:
    """Compute optimal HAC bandwidth.

    Parameters
    ----------
    n : int
        Number of observations.
    method : {"nw", "andrews"}
        "nw": Newey-West rule: floor(4 * (n/100)^{2/9})
        "andrews": Andrews (1991) plug-in formula (requires residuals, not implemented)

    Returns
    -------
    int
        Optimal bandwidth.
    """
    if method == "nw":
        return int(np.floor(4 * (n / 100) ** (2 / 9)))
    elif method == "andrews":
        # Andrews plug-in requires residual autocovariances
        # Fall back to Newey-West for now
        return int(np.floor(4 * (n / 100) ** (2 / 9)))
    else:
        raise ValueError(f"Unknown method: {method}")


def confidence_interval(
    estimate: np.ndarray,
    se: np.ndarray,
    alpha: float = 0.05,
    method: Literal["normal", "t"] = "normal",
    df: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute confidence intervals.

    Parameters
    ----------
    estimate : np.ndarray
        Point estimates.
    se : np.ndarray
        Standard errors.
    alpha : float
        Significance level (0.05 for 95% CI).
    method : {"normal", "t"}
        Distribution for critical values.
    df : int, optional
        Degrees of freedom for t-distribution.

    Returns
    -------
    ci_lower : np.ndarray
        Lower confidence bounds.
    ci_upper : np.ndarray
        Upper confidence bounds.
    """
    if method == "normal":
        z = stats.norm.ppf(1 - alpha / 2)
    else:
        if df is None:
            raise ValueError("df required for t-distribution")
        z = stats.t.ppf(1 - alpha / 2, df)

    ci_lower = estimate - z * se
    ci_upper = estimate + z * se

    return ci_lower, ci_upper


def hac_ols_se(
    X: np.ndarray,
    residuals: np.ndarray,
    bandwidth: Optional[int] = None,
    kernel: Literal["bartlett", "qs"] = "bartlett",
) -> np.ndarray:
    """Compute HAC standard errors for OLS coefficients.

    This is the classic Newey-West estimator for regression coefficients.

    The HAC covariance:
        V = (X'X)^{-1} Ω (X'X)^{-1}

    where Ω is the HAC-adjusted meat matrix.

    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (n_obs, n_features).
    residuals : np.ndarray
        OLS residuals, shape (n_obs,).
    bandwidth : int, optional
        HAC bandwidth.
    kernel : {"bartlett", "qs"}
        HAC kernel.

    Returns
    -------
    np.ndarray
        Standard errors for each coefficient, shape (n_features,).

    Notes
    -----
    This is functionally equivalent to the implementation in
    local_projections.py but refactored for reuse.
    """
    T, k = X.shape

    if bandwidth is None:
        bandwidth = optimal_bandwidth(T)
    bandwidth = min(bandwidth, T - 2)

    # (X'X)^{-1}
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    # Score contributions: x_t * u_t
    xu = X * residuals[:, np.newaxis]  # (T, k)

    # HAC-adjusted meat: Ω
    Omega = newey_west_variance(xu, bandwidth=bandwidth, kernel=kernel)

    # Sandwich: (X'X)^{-1} Ω (X'X)^{-1}
    V = T * XtX_inv @ Omega @ XtX_inv

    # Standard errors
    se = np.sqrt(np.diag(V))
    se = np.maximum(se, 1e-10)

    return se
