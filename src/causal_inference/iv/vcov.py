"""
Variance-Covariance Matrix Computation for 2SLS Estimators.

This module provides variance-covariance estimators for two-stage least squares (2SLS)
instrumental variables regression. Three types of standard errors are supported:

1. Standard (homoskedastic): Assumes constant error variance
2. Robust (heteroskedasticity-robust): White/HC0 sandwich estimator
3. Clustered (cluster-robust): Accounts for within-cluster correlation

Key References:
    - White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator
      and a Direct Test for Heteroskedasticity." Econometrica, 48(4), 817-838.
    - Cameron, A.C., and D.L. Miller (2015). "A Practitioner's Guide to Cluster-Robust
      Inference." Journal of Human Resources, 50(2), 317-372.
    - Wooldridge, J.M. (2010). "Econometric Analysis of Cross Section and Panel Data",
      2nd ed., Chapter 8.

Mathematical Framework:
    - Standard: V = σ² (X'P_Z X)⁻¹
    - Robust:   V = (X'P_Z X)⁻¹ (X'P_Z Ω P_Z X) (X'P_Z X)⁻¹, where Ω = diag(e²)
    - Clustered: V = (X'P_Z X)⁻¹ (Σ_g X_g'P_Z e_g e_g'P_Z X_g) (X'P_Z X)⁻¹

    where:
    - X: Design matrix [D, controls]
    - P_Z: Projection matrix onto instruments Z
    - e: Second-stage residuals
    - g: Cluster index
"""

import numpy as np


def compute_standard_vcov(XPX_inv: np.ndarray, sigma2: float) -> np.ndarray:
    """
    Compute standard homoskedastic variance-covariance matrix for 2SLS.

    Formula: V = σ² (X'P_Z X)⁻¹

    Assumes constant error variance (homoskedasticity). This is the classical
    formula but is invalid if heteroskedasticity is present.

    Parameters
    ----------
    XPX_inv : np.ndarray
        Inverse of (X'P_Z X) matrix, where:
        - X: Design matrix [D, controls, constant]
        - P_Z: Projection matrix onto instruments
    sigma2 : float
        Residual variance σ² = e'e / (n - k)

    Returns
    -------
    np.ndarray
        Variance-covariance matrix of shape (k, k)

    Notes
    -----
    This estimator is consistent only under homoskedasticity.
    Use robust or clustered SEs if heteroskedasticity is suspected.

    Examples
    --------
    >>> XPX_inv = np.linalg.inv(X.T @ P_Z @ X)
    >>> sigma2 = np.sum(residuals ** 2) / (n - k)
    >>> vcov = compute_standard_vcov(XPX_inv, sigma2)
    >>> se = np.sqrt(np.diag(vcov))
    """
    return sigma2 * XPX_inv


def compute_robust_vcov(
    XPX_inv: np.ndarray,
    DX: np.ndarray,
    P_Z: np.ndarray,
    residuals: np.ndarray,
) -> np.ndarray:
    """
    Compute heteroskedasticity-robust variance-covariance matrix (White/HC0).

    Formula: V = (X'P_Z X)⁻¹ (X'P_Z Ω P_Z X) (X'P_Z X)⁻¹
    where Ω = diag(e²)

    This is the White (1980) sandwich estimator, also known as HC0.
    It is consistent even with heteroskedasticity but may undercover
    in finite samples (use HC1, HC2, or HC3 for finite-sample corrections).

    Parameters
    ----------
    XPX_inv : np.ndarray
        Inverse of (X'P_Z X) matrix
    DX : np.ndarray, shape (n, k)
        Design matrix [D, X, constant] including all regressors
    P_Z : np.ndarray, shape (n, n)
        Projection matrix onto instruments: P_Z = Z(Z'Z)⁻¹Z'
    residuals : np.ndarray, shape (n,)
        Second-stage residuals: e = Y - X'β̂

    Returns
    -------
    np.ndarray
        Robust variance-covariance matrix of shape (k, k)

    Notes
    -----
    The sandwich formula has three parts:
    1. Bread: (X'P_Z X)⁻¹
    2. Meat: X'P_Z Ω P_Z X, where Ω = diag(e²)
    3. Bread: (X'P_Z X)⁻¹

    Examples
    --------
    >>> # After 2SLS estimation
    >>> vcov_robust = compute_robust_vcov(XPX_inv, DX, P_Z, residuals)
    >>> se_robust = np.sqrt(np.diag(vcov_robust))
    """
    Omega = np.diag(residuals**2)
    meat = DX.T @ P_Z @ Omega @ P_Z @ DX
    return XPX_inv @ meat @ XPX_inv


def compute_clustered_vcov(
    XPX_inv: np.ndarray,
    DX: np.ndarray,
    P_Z: np.ndarray,
    residuals: np.ndarray,
    clusters: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Compute cluster-robust variance-covariance matrix.

    Formula: V = (X'P_Z X)⁻¹ (Σ_g X_g'P_Z e_g e_g'P_Z X_g) (X'P_Z X)⁻¹
    with finite-sample correction: (G / (G - 1)) * ((n - 1) / (n - k))

    Accounts for arbitrary correlation within clusters while assuming
    independence across clusters. This is the default choice when
    observations are grouped (e.g., students within schools, firms within
    industries, repeated observations on individuals).

    Parameters
    ----------
    XPX_inv : np.ndarray
        Inverse of (X'P_Z X) matrix
    DX : np.ndarray, shape (n, k)
        Design matrix [D, X, constant]
    P_Z : np.ndarray, shape (n, n)
        Projection matrix onto instruments
    residuals : np.ndarray, shape (n,)
        Second-stage residuals
    clusters : np.ndarray, shape (n,)
        Cluster identifiers (e.g., school IDs, firm IDs)
    n : int
        Number of observations

    Returns
    -------
    np.ndarray
        Cluster-robust variance-covariance matrix of shape (k, k)

    Warnings
    --------
    UserWarning
        If number of clusters < 20 (clustered SEs unreliable)

    Notes
    -----
    Cluster-robust inference requires:
    1. Many clusters (G ≥ 50 recommended, G ≥ 20 minimum)
    2. Balanced cluster sizes (unbalanced OK, but avoid one huge cluster)
    3. Independence across clusters

    With few clusters (G < 20), t-tests and F-tests become unreliable.
    Consider:
    - Wild cluster bootstrap (Cameron et al. 2008)
    - Robust SEs instead
    - Aggregating to cluster level

    The finite-sample correction:
    - (G / (G - 1)): Corrects for estimating cluster means
    - ((n - 1) / (n - k)): Corrects for estimating k parameters

    Examples
    --------
    >>> # After 2SLS estimation with clustered data
    >>> vcov_cluster = compute_clustered_vcov(XPX_inv, DX, P_Z, residuals, clusters, n)
    >>> se_cluster = np.sqrt(np.diag(vcov_cluster))
    """
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    k = DX.shape[1]

    # Warn if too few clusters
    if G < 20:
        import warnings

        warnings.warn(
            f"Only {G} clusters. Clustered standard errors may be unreliable with <20 clusters. "
            f"Consider using robust SEs instead or wild cluster bootstrap.",
            UserWarning,
        )

    # Compute cluster-robust meat
    meat = np.zeros((k, k))

    for g in unique_clusters:
        cluster_mask = clusters == g
        DX_g = DX[cluster_mask]
        e_g = residuals[cluster_mask]
        PZ_DX_g = P_Z[cluster_mask, :] @ DX  # P_Z @ DX for cluster g
        meat += PZ_DX_g.T @ np.outer(e_g, e_g) @ PZ_DX_g

    # Apply finite-sample correction
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    return correction * XPX_inv @ meat @ XPX_inv
