"""
Data Generating Processes (DGPs) for Monte Carlo validation.

All DGPs have known true ATE = 2.0 for validation purposes.
"""

import numpy as np
from typing import Tuple


def dgp_simple_rct(n: int = 100, true_ate: float = 2.0, random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple RCT with homoskedastic errors.

    DGP:
        Y(1) ~ N(2, 1)
        Y(0) ~ N(0, 1)
        T ~ Bernoulli(0.5)
        Y = T*Y(1) + (1-T)*Y(0)

    True ATE = 2.0
    """
    rng = np.random.RandomState(random_state)

    n1 = n // 2
    n0 = n - n1
    treatment = np.array([1] * n1 + [0] * n0)
    rng.shuffle(treatment)

    y1 = rng.normal(true_ate, 1.0, n)
    y0 = rng.normal(0.0, 1.0, n)

    outcomes = treatment * y1 + (1 - treatment) * y0

    return outcomes, treatment


def dgp_heteroskedastic_rct(n: int = 200, true_ate: float = 2.0, random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    RCT with heteroskedastic errors (different variances by group).

    DGP:
        Y(1) ~ N(2, 4)  # Higher variance in treated
        Y(0) ~ N(0, 1)
        T ~ Bernoulli(0.5)

    True ATE = 2.0
    Tests Neyman variance (robust to heteroskedasticity)
    """
    rng = np.random.RandomState(random_state)

    n1 = n // 2
    n0 = n - n1
    treatment = np.array([1] * n1 + [0] * n0)
    rng.shuffle(treatment)

    y1 = rng.normal(true_ate, 2.0, n)  # σ=2
    y0 = rng.normal(0.0, 1.0, n)       # σ=1

    outcomes = treatment * y1 + (1 - treatment) * y0

    return outcomes, treatment


def dgp_small_sample_rct(n: int = 20, true_ate: float = 2.0, random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Small sample RCT to test t-distribution inference.

    DGP:
        Y(1) ~ N(2, 1)
        Y(0) ~ N(0, 1)
        T ~ Bernoulli(0.5)
        n = 20

    True ATE = 2.0
    Tests t-distribution vs z-distribution (critical for small samples)
    """
    return dgp_simple_rct(n=n, true_ate=true_ate, random_state=random_state)


def dgp_stratified_rct(n_per_stratum: int = 40, n_strata: int = 3, true_ate: float = 2.0,
                       random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified RCT with different baseline levels.

    DGP:
        3 strata with baselines [0, 5, 10]
        Within each stratum:
            Y(1) ~ N(baseline + 2, 1)
            Y(0) ~ N(baseline, 1)
            T ~ Bernoulli(0.5)

    True ATE = 2.0 in all strata (average ATE = 2.0)
    """
    rng = np.random.RandomState(random_state)

    baselines = [i * 5.0 for i in range(n_strata)]

    outcomes = []
    treatment = []
    strata = []

    for s, baseline in enumerate(baselines):
        n1 = n_per_stratum // 2
        n0 = n_per_stratum - n1

        t = np.array([1] * n1 + [0] * n0)
        rng.shuffle(t)

        y = np.where(t == 1,
                    rng.normal(baseline + true_ate, 1.0, n_per_stratum),
                    rng.normal(baseline, 1.0, n_per_stratum))

        outcomes.extend(y)
        treatment.extend(t)
        strata.extend([s] * n_per_stratum)

    return np.array(outcomes), np.array(treatment), np.array(strata)


def dgp_regression_rct(n: int = 100, true_ate: float = 2.0, covariate_effect: float = 3.0,
                      random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RCT with covariate for regression adjustment.

    DGP:
        X ~ N(0, 1)
        T ~ Bernoulli(0.5)
        Y = 2*T + 3*X + ε
        ε ~ N(0, 1)

    True ATE = 2.0
    Covariate strongly predicts outcome (variance reduction)
    """
    rng = np.random.RandomState(random_state)

    X = rng.normal(0, 1, n)
    n1 = n // 2
    n0 = n - n1
    treatment = np.array([1] * n1 + [0] * n0)
    rng.shuffle(treatment)

    outcomes = true_ate * treatment + covariate_effect * X + rng.normal(0, 1, n)

    return outcomes, treatment, X


def dgp_ipw_rct(n: int = 100, true_ate: float = 2.0, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RCT with non-constant propensity scores.

    DGP:
        X ~ N(0, 1)
        propensity(X) = 1/(1 + exp(-0.5*X))  # Logistic
        T ~ Bernoulli(propensity(X))
        Y = 2*T + X + ε
        ε ~ N(0, 1)

    True ATE = 2.0
    Propensity varies with covariate
    """
    rng = np.random.RandomState(random_state)

    X = rng.normal(0, 1, n)

    # Propensity depends on X
    propensity = 1 / (1 + np.exp(-0.5 * X))
    treatment = (rng.uniform(0, 1, n) < propensity).astype(float)

    # Outcomes (ATE = 2.0)
    outcomes = true_ate * treatment + X + rng.normal(0, 1, n)

    return outcomes, treatment, propensity
