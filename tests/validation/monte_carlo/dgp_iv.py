"""
Data Generating Processes (DGPs) for IV Monte Carlo validation.

This module provides DGPs for validating Instrumental Variables estimators:
- 2SLS (Two-Stage Least Squares)
- LIML (Limited Information Maximum Likelihood)
- Fuller (bias-corrected k-class estimators)
- GMM (Generalized Method of Moments)

All DGPs have known true effects and calibrated first-stage strength for validation.

Key References:
    - Stock & Yogo (2005). "Testing for Weak Instruments in Linear IV Regression"
    - Staiger & Stock (1997). "Instrumental Variables Regression with Weak Instruments"
    - Anderson & Rubin (1949). "Estimation of the Parameters of a Single Equation"
    - Fuller (1977). "Some Properties of a Modification of the Limited Information Estimator"

Instrument Strength Classification (Stock-Yogo):
    - Strong: F > 16.38 (10% maximal bias)
    - Moderate: F > 8.96 (15% maximal bias)
    - Weak: F < 10 (rule of thumb)
    - Very Weak: F < 5
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class IVData:
    """Container for IV simulation data with known ground truth.

    Attributes
    ----------
    Y : np.ndarray
        Outcome variable (n,)
    D : np.ndarray
        Endogenous treatment variable (n,)
    Z : np.ndarray
        Instrumental variables (n, q)
    X : np.ndarray | None
        Exogenous controls (n, k) or None
    true_beta : float
        True causal effect of D on Y
    first_stage_pi : float | np.ndarray
        First-stage coefficient(s) π from D = πZ + ν
    endogeneity_rho : float
        Correlation between first-stage and second-stage errors
    n : int
        Sample size
    n_instruments : int
        Number of instruments (q)
    n_endogenous : int
        Number of endogenous variables (p)
    expected_f_stat : float
        Expected first-stage F-statistic under the DGP
    """

    Y: np.ndarray
    D: np.ndarray
    Z: np.ndarray
    X: Optional[np.ndarray]
    true_beta: float
    first_stage_pi: float
    endogeneity_rho: float
    n: int
    n_instruments: int
    n_endogenous: int
    expected_f_stat: float


# =============================================================================
# Strong Instrument DGP
# =============================================================================


def dgp_iv_strong(
    n: int = 1000,
    true_beta: float = 0.5,
    endogeneity_rho: float = 0.5,
    random_state: Optional[int] = None,
) -> IVData:
    """
    IV DGP with strong instruments (F >> 20).

    Model:
        First stage:  D = π*Z + ν
        Second stage: Y = β*D + ε
        Cov(ν, ε) = ρ (endogeneity)

    Parameters
    ----------
    n : int, default=1000
        Sample size
    true_beta : float, default=0.5
        True causal effect
    endogeneity_rho : float, default=0.5
        Correlation between errors (creates endogeneity)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    IVData
        Container with outcomes, treatment, instruments, and ground truth

    Notes
    -----
    With π=0.8 and n=1000, expected F ≈ 640.
    All estimators (2SLS, LIML, Fuller) should be nearly unbiased.
    """
    rng = np.random.RandomState(random_state)

    # Instrument (exogenous by construction)
    Z = rng.normal(0, 1, n)

    # Correlated errors (creates endogeneity)
    # Generate from bivariate normal with correlation rho
    cov_matrix = [[1.0, endogeneity_rho], [endogeneity_rho, 1.0]]
    errors = rng.multivariate_normal([0, 0], cov_matrix, n)
    nu = errors[:, 0]  # First-stage error
    epsilon = errors[:, 1]  # Second-stage error

    # First stage: D = π*Z + ν
    pi = 0.8  # Strong: F ≈ n * π² / Var(ν) ≈ 640 for n=1000
    D = pi * Z + nu

    # Second stage: Y = β*D + ε
    Y = true_beta * D + epsilon

    # Expected F-statistic: F ≈ n * π² / σ²_ν (for just-identified case)
    expected_f = n * (pi**2)

    return IVData(
        Y=Y,
        D=D,
        Z=Z.reshape(-1, 1),
        X=None,
        true_beta=true_beta,
        first_stage_pi=pi,
        endogeneity_rho=endogeneity_rho,
        n=n,
        n_instruments=1,
        n_endogenous=1,
        expected_f_stat=expected_f,
    )


# =============================================================================
# Moderate Instrument DGP
# =============================================================================


def dgp_iv_moderate(
    n: int = 1000,
    true_beta: float = 0.5,
    endogeneity_rho: float = 0.5,
    random_state: Optional[int] = None,
) -> IVData:
    """
    IV DGP with moderate instrument strength (F ≈ 15).

    At this strength, 2SLS begins to show some bias.
    Fuller-1 starts to outperform 2SLS.

    Parameters
    ----------
    n : int, default=1000
        Sample size
    true_beta : float, default=0.5
        True causal effect
    endogeneity_rho : float, default=0.5
        Correlation between errors
    random_state : int, optional
        Random seed

    Returns
    -------
    IVData
        Container with simulation data

    Notes
    -----
    With π=0.12 and n=1000, expected F ≈ 14.
    This is below Stock-Yogo 10% bias threshold (16.38).
    """
    rng = np.random.RandomState(random_state)

    Z = rng.normal(0, 1, n)

    cov_matrix = [[1.0, endogeneity_rho], [endogeneity_rho, 1.0]]
    errors = rng.multivariate_normal([0, 0], cov_matrix, n)
    nu = errors[:, 0]
    epsilon = errors[:, 1]

    # Moderate: F ≈ n * π² ≈ 14 for n=1000
    pi = 0.12
    D = pi * Z + nu
    Y = true_beta * D + epsilon

    expected_f = n * (pi**2)

    return IVData(
        Y=Y,
        D=D,
        Z=Z.reshape(-1, 1),
        X=None,
        true_beta=true_beta,
        first_stage_pi=pi,
        endogeneity_rho=endogeneity_rho,
        n=n,
        n_instruments=1,
        n_endogenous=1,
        expected_f_stat=expected_f,
    )


# =============================================================================
# Weak Instrument DGP
# =============================================================================


def dgp_iv_weak(
    n: int = 1000,
    true_beta: float = 0.5,
    endogeneity_rho: float = 0.5,
    random_state: Optional[int] = None,
) -> IVData:
    """
    IV DGP with weak instruments (F ≈ 8).

    At this strength:
    - 2SLS is substantially biased toward OLS
    - 2SLS confidence intervals have severe undercoverage
    - LIML and Fuller are less biased but have higher variance
    - Anderson-Rubin CIs remain valid

    Parameters
    ----------
    n : int, default=1000
        Sample size
    true_beta : float, default=0.5
        True causal effect
    endogeneity_rho : float, default=0.5
        Correlation between errors
    random_state : int, optional
        Random seed

    Returns
    -------
    IVData
        Container with simulation data

    Notes
    -----
    With π=0.09 and n=1000, expected F ≈ 8.
    This is a canonical "weak IV" scenario per Staiger & Stock (1997).
    """
    rng = np.random.RandomState(random_state)

    Z = rng.normal(0, 1, n)

    cov_matrix = [[1.0, endogeneity_rho], [endogeneity_rho, 1.0]]
    errors = rng.multivariate_normal([0, 0], cov_matrix, n)
    nu = errors[:, 0]
    epsilon = errors[:, 1]

    # Weak: F ≈ n * π² ≈ 8 for n=1000
    pi = 0.09
    D = pi * Z + nu
    Y = true_beta * D + epsilon

    expected_f = n * (pi**2)

    return IVData(
        Y=Y,
        D=D,
        Z=Z.reshape(-1, 1),
        X=None,
        true_beta=true_beta,
        first_stage_pi=pi,
        endogeneity_rho=endogeneity_rho,
        n=n,
        n_instruments=1,
        n_endogenous=1,
        expected_f_stat=expected_f,
    )


# =============================================================================
# Very Weak Instrument DGP
# =============================================================================


def dgp_iv_very_weak(
    n: int = 1000,
    true_beta: float = 0.5,
    endogeneity_rho: float = 0.5,
    random_state: Optional[int] = None,
) -> IVData:
    """
    IV DGP with very weak instruments (F ≈ 3).

    At this strength:
    - 2SLS is severely biased (essentially OLS)
    - LIML/Fuller may be unstable
    - Only Anderson-Rubin CIs are reliable
    - Point estimates should not be trusted

    Parameters
    ----------
    n : int, default=1000
        Sample size
    true_beta : float, default=0.5
        True causal effect
    endogeneity_rho : float, default=0.5
        Correlation between errors
    random_state : int, optional
        Random seed

    Returns
    -------
    IVData
        Container with simulation data

    Notes
    -----
    With π=0.05 and n=1000, expected F ≈ 2.5.
    This demonstrates the breakdown of standard IV inference.
    """
    rng = np.random.RandomState(random_state)

    Z = rng.normal(0, 1, n)

    cov_matrix = [[1.0, endogeneity_rho], [endogeneity_rho, 1.0]]
    errors = rng.multivariate_normal([0, 0], cov_matrix, n)
    nu = errors[:, 0]
    epsilon = errors[:, 1]

    # Very weak: F ≈ n * π² ≈ 2.5 for n=1000
    pi = 0.05
    D = pi * Z + nu
    Y = true_beta * D + epsilon

    expected_f = n * (pi**2)

    return IVData(
        Y=Y,
        D=D,
        Z=Z.reshape(-1, 1),
        X=None,
        true_beta=true_beta,
        first_stage_pi=pi,
        endogeneity_rho=endogeneity_rho,
        n=n,
        n_instruments=1,
        n_endogenous=1,
        expected_f_stat=expected_f,
    )


# =============================================================================
# Over-Identified DGP (Multiple Instruments)
# =============================================================================


def dgp_iv_over_identified(
    n: int = 1000,
    true_beta: float = 0.5,
    endogeneity_rho: float = 0.5,
    n_instruments: int = 3,
    random_state: Optional[int] = None,
) -> IVData:
    """
    IV DGP with multiple instruments (q > p, over-identified).

    Useful for:
    - GMM efficiency gains (two-step GMM < one-step variance)
    - Hansen J-test for overidentifying restrictions
    - Testing LIML with many instruments

    Parameters
    ----------
    n : int, default=1000
        Sample size
    true_beta : float, default=0.5
        True causal effect
    endogeneity_rho : float, default=0.5
        Correlation between errors
    n_instruments : int, default=3
        Number of instruments
    random_state : int, optional
        Random seed

    Returns
    -------
    IVData
        Container with simulation data

    Notes
    -----
    All instruments are valid (satisfy exclusion restriction).
    For testing Hansen J-test power, use dgp_iv_invalid_instruments().
    """
    rng = np.random.RandomState(random_state)

    # Multiple instruments (all valid)
    Z = rng.normal(0, 1, (n, n_instruments))

    cov_matrix = [[1.0, endogeneity_rho], [endogeneity_rho, 1.0]]
    errors = rng.multivariate_normal([0, 0], cov_matrix, n)
    nu = errors[:, 0]
    epsilon = errors[:, 1]

    # First stage: D = π₁Z₁ + π₂Z₂ + ... + ν
    # Use decreasing coefficients: strongest instrument first
    pi_coeffs = np.array([0.5 / (i + 1) for i in range(n_instruments)])
    D = Z @ pi_coeffs + nu

    # Second stage: Y = β*D + ε
    Y = true_beta * D + epsilon

    # Expected F for combined instruments (approximate)
    total_pi_sq = np.sum(pi_coeffs**2)
    expected_f = n * total_pi_sq

    return IVData(
        Y=Y,
        D=D,
        Z=Z,
        X=None,
        true_beta=true_beta,
        first_stage_pi=pi_coeffs[0],  # Report strongest
        endogeneity_rho=endogeneity_rho,
        n=n,
        n_instruments=n_instruments,
        n_endogenous=1,
        expected_f_stat=expected_f,
    )


# =============================================================================
# IV with Exogenous Controls
# =============================================================================


def dgp_iv_with_controls(
    n: int = 1000,
    true_beta: float = 0.5,
    n_controls: int = 2,
    endogeneity_rho: float = 0.5,
    random_state: Optional[int] = None,
) -> IVData:
    """
    IV DGP with exogenous control variables.

    Model:
        First stage:  D = π*Z + γ₁X₁ + γ₂X₂ + ν
        Second stage: Y = β*D + δ₁X₁ + δ₂X₂ + ε

    Parameters
    ----------
    n : int, default=1000
        Sample size
    true_beta : float, default=0.5
        True causal effect of D on Y
    n_controls : int, default=2
        Number of exogenous controls
    endogeneity_rho : float, default=0.5
        Correlation between errors
    random_state : int, optional
        Random seed

    Returns
    -------
    IVData
        Container with simulation data

    Notes
    -----
    Controls affect both D and Y (confounders properly controlled).
    The instrument Z remains valid (uncorrelated with ε).
    """
    rng = np.random.RandomState(random_state)

    # Instrument
    Z = rng.normal(0, 1, n)

    # Exogenous controls
    X = rng.normal(0, 1, (n, n_controls))

    # Errors with endogeneity
    cov_matrix = [[1.0, endogeneity_rho], [endogeneity_rho, 1.0]]
    errors = rng.multivariate_normal([0, 0], cov_matrix, n)
    nu = errors[:, 0]
    epsilon = errors[:, 1]

    # Control effects (on D and Y)
    gamma = rng.uniform(0.3, 0.7, n_controls)  # Effect on D
    delta = rng.uniform(0.2, 0.5, n_controls)  # Effect on Y

    # First stage: D = π*Z + γ'X + ν
    pi = 0.5
    D = pi * Z + X @ gamma + nu

    # Second stage: Y = β*D + δ'X + ε
    Y = true_beta * D + X @ delta + epsilon

    # Expected F (partial, controlling for X)
    expected_f = n * (pi**2) * 0.8  # Approximate

    return IVData(
        Y=Y,
        D=D,
        Z=Z.reshape(-1, 1),
        X=X,
        true_beta=true_beta,
        first_stage_pi=pi,
        endogeneity_rho=endogeneity_rho,
        n=n,
        n_instruments=1,
        n_endogenous=1,
        expected_f_stat=expected_f,
    )


# =============================================================================
# Heteroskedastic Errors DGP
# =============================================================================


def dgp_iv_heteroskedastic(
    n: int = 1000,
    true_beta: float = 0.5,
    endogeneity_rho: float = 0.5,
    random_state: Optional[int] = None,
) -> IVData:
    """
    IV DGP with heteroskedastic errors.

    Error variance depends on treatment level:
        Var(ε|D) = 1 + 0.5*|D|

    This tests:
    - Robust (HC0) standard errors
    - Difference between standard and robust inference

    Parameters
    ----------
    n : int, default=1000
        Sample size
    true_beta : float, default=0.5
        True causal effect
    endogeneity_rho : float, default=0.5
        Correlation between errors
    random_state : int, optional
        Random seed

    Returns
    -------
    IVData
        Container with simulation data

    Notes
    -----
    Standard (homoskedastic) SEs will be incorrect.
    Robust SEs should provide correct coverage.
    """
    rng = np.random.RandomState(random_state)

    Z = rng.normal(0, 1, n)

    # Base errors with endogeneity
    cov_matrix = [[1.0, endogeneity_rho], [endogeneity_rho, 1.0]]
    errors = rng.multivariate_normal([0, 0], cov_matrix, n)
    nu = errors[:, 0]
    epsilon_base = errors[:, 1]

    # First stage
    pi = 0.5
    D = pi * Z + nu

    # Heteroskedastic scaling: variance increases with |D|
    scale = np.sqrt(1 + 0.5 * np.abs(D))
    epsilon = epsilon_base * scale

    # Second stage
    Y = true_beta * D + epsilon

    expected_f = n * (pi**2)

    return IVData(
        Y=Y,
        D=D,
        Z=Z.reshape(-1, 1),
        X=None,
        true_beta=true_beta,
        first_stage_pi=pi,
        endogeneity_rho=endogeneity_rho,
        n=n,
        n_instruments=1,
        n_endogenous=1,
        expected_f_stat=expected_f,
    )


# =============================================================================
# Invalid Instruments DGP (for Hansen J-test power)
# =============================================================================


def dgp_iv_invalid_instruments(
    n: int = 1000,
    true_beta: float = 0.5,
    endogeneity_rho: float = 0.5,
    violation_strength: float = 0.3,
    random_state: Optional[int] = None,
) -> IVData:
    """
    IV DGP with one invalid instrument (exclusion restriction violated).

    For testing Hansen J-test power to detect invalid instruments.

    Model:
        D = π₁Z₁ + π₂Z₂ + ν
        Y = β*D + λZ₂ + ε  (Z₂ directly affects Y - INVALID)

    Parameters
    ----------
    n : int, default=1000
        Sample size
    true_beta : float, default=0.5
        True causal effect
    endogeneity_rho : float, default=0.5
        Correlation between errors
    violation_strength : float, default=0.3
        Direct effect of invalid instrument on Y
    random_state : int, optional
        Random seed

    Returns
    -------
    IVData
        Container with simulation data

    Notes
    -----
    Hansen J-test should reject H₀ (valid instruments) with high probability.
    The true_beta in the returned data is still 0.5, but 2SLS estimates
    will be biased due to the invalid instrument.
    """
    rng = np.random.RandomState(random_state)

    # Two instruments: Z₁ (valid) and Z₂ (invalid - directly affects Y)
    Z = rng.normal(0, 1, (n, 2))
    Z1, Z2 = Z[:, 0], Z[:, 1]

    cov_matrix = [[1.0, endogeneity_rho], [endogeneity_rho, 1.0]]
    errors = rng.multivariate_normal([0, 0], cov_matrix, n)
    nu = errors[:, 0]
    epsilon = errors[:, 1]

    # First stage: both instruments affect D
    pi1, pi2 = 0.4, 0.3
    D = pi1 * Z1 + pi2 * Z2 + nu

    # Second stage: Z₂ directly affects Y (VIOLATION)
    Y = true_beta * D + violation_strength * Z2 + epsilon

    expected_f = n * (pi1**2 + pi2**2)

    return IVData(
        Y=Y,
        D=D,
        Z=Z,
        X=None,
        true_beta=true_beta,  # True structural effect
        first_stage_pi=pi1,
        endogeneity_rho=endogeneity_rho,
        n=n,
        n_instruments=2,
        n_endogenous=1,
        expected_f_stat=expected_f,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def compute_ols_probability_limit(endogeneity_rho: float, sigma_d: float = 1.0) -> float:
    """
    Compute the probability limit of OLS estimator under endogeneity.

    Under the DGP:
        D = π*Z + ν, Var(ν) = 1
        Y = β*D + ε, Var(ε) = 1
        Cov(ν, ε) = ρ

    plim(β_OLS) = β + Cov(D, ε) / Var(D) = β + ρ / Var(D)

    Parameters
    ----------
    endogeneity_rho : float
        Correlation between first-stage and second-stage errors
    sigma_d : float, default=1.0
        Standard deviation of D (approximately)

    Returns
    -------
    float
        The bias direction (positive when ρ > 0)

    Notes
    -----
    This helps verify that weak IV bias is "toward OLS" as expected.
    """
    return endogeneity_rho / (sigma_d**2)
