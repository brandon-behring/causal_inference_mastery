"""Test fixtures for CATE estimation tests.

Provides data generating processes (DGPs) for testing meta-learners:
- Constant treatment effect (homogeneous)
- Linear heterogeneous effect (CATE varies with X)
- Nonlinear heterogeneous effect
"""

import pytest
import numpy as np
from typing import Literal


def generate_cate_dgp(
    n: int = 500,
    p: int = 2,
    effect_type: Literal["constant", "linear", "nonlinear"] = "constant",
    true_ate: float = 2.0,
    noise_sd: float = 1.0,
    treatment_prob: float = 0.5,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate data for CATE testing.

    Data Generating Process:
    - X ~ N(0, I_p)
    - T ~ Bernoulli(treatment_prob)
    - Y = baseline(X) + τ(X) * T + ε, where ε ~ N(0, noise_sd²)

    Treatment effect τ(X) depends on effect_type:
    - "constant": τ(x) = true_ate
    - "linear": τ(x) = true_ate + x₁ (heterogeneity in first covariate)
    - "nonlinear": τ(x) = true_ate * (1 + 0.5 * sin(x₁))

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of covariates.
    effect_type : {"constant", "linear", "nonlinear"}
        Type of treatment effect heterogeneity.
    true_ate : float
        True average treatment effect (baseline).
    noise_sd : float
        Standard deviation of outcome noise.
    treatment_prob : float
        Probability of treatment assignment.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (outcomes, treatment, covariates, true_cate) where:
        - outcomes: shape (n,)
        - treatment: shape (n,)
        - covariates: shape (n, p)
        - true_cate: shape (n,) - true individual treatment effects
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate covariates
    X = np.random.randn(n, p)

    # Generate treatment
    T = np.random.binomial(1, treatment_prob, n).astype(float)

    # Baseline outcome (no treatment)
    # Y₀ = 1 + 0.5 * X₁ + 0.3 * X₂ + ε
    baseline = 1 + 0.5 * X[:, 0]
    if p > 1:
        baseline += 0.3 * X[:, 1]

    # Treatment effect
    if effect_type == "constant":
        true_cate = np.full(n, true_ate)
    elif effect_type == "linear":
        # τ(x) = true_ate + x₁
        true_cate = true_ate + X[:, 0]
    elif effect_type == "nonlinear":
        # τ(x) = true_ate * (1 + 0.5 * sin(x₁))
        true_cate = true_ate * (1 + 0.5 * np.sin(X[:, 0]))
    else:
        raise ValueError(f"Unknown effect_type: {effect_type}")

    # Generate outcomes
    noise = np.random.randn(n) * noise_sd
    Y = baseline + true_cate * T + noise

    return Y, T, X, true_cate


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture
def constant_effect_data():
    """Data with constant treatment effect (τ = 2.0)."""
    return generate_cate_dgp(
        n=500,
        p=2,
        effect_type="constant",
        true_ate=2.0,
        noise_sd=1.0,
        seed=42,
    )


@pytest.fixture
def linear_heterogeneous_data():
    """Data with linear heterogeneous treatment effect (τ(x) = 2 + x₁)."""
    return generate_cate_dgp(
        n=500,
        p=2,
        effect_type="linear",
        true_ate=2.0,
        noise_sd=1.0,
        seed=42,
    )


@pytest.fixture
def nonlinear_heterogeneous_data():
    """Data with nonlinear heterogeneous treatment effect."""
    return generate_cate_dgp(
        n=500,
        p=2,
        effect_type="nonlinear",
        true_ate=2.0,
        noise_sd=1.0,
        seed=42,
    )


@pytest.fixture
def high_dimensional_data():
    """Data with many covariates (p=15)."""
    return generate_cate_dgp(
        n=500,
        p=15,
        effect_type="constant",
        true_ate=2.0,
        noise_sd=1.0,
        seed=42,
    )


@pytest.fixture
def small_sample_data():
    """Small sample data (n=50)."""
    return generate_cate_dgp(
        n=50,
        p=2,
        effect_type="constant",
        true_ate=2.0,
        noise_sd=1.0,
        seed=42,
    )


@pytest.fixture
def single_covariate_data():
    """Data with single covariate (p=1)."""
    return generate_cate_dgp(
        n=200,
        p=1,
        effect_type="constant",
        true_ate=2.0,
        noise_sd=1.0,
        seed=42,
    )
