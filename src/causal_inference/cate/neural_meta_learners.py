"""Neural network versions of CATE meta-learners.

This module implements neural network versions of S/T/X/R-learners using sklearn
MLPRegressor and MLPClassifier. These provide more flexible function approximation
than linear models while maintaining the same algorithmic structure.

Methods
-------
- neural_s_learner: Single neural network with treatment as feature
- neural_t_learner: Separate neural networks for treated/control groups
- neural_x_learner: Cross-learner with neural networks and propensity weighting
- neural_r_learner: Robinson transformation with neural networks

References
----------
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects
  using machine learning." PNAS 116(10): 4156-4165.
- Nie & Wager (2021). "Quasi-oracle estimation of heterogeneous treatment effects."
  Biometrika 108(2): 299-319.
"""

from typing import Tuple

import numpy as np
from scipy import stats
from sklearn.neural_network import MLPClassifier, MLPRegressor

from .base import CATEResult, validate_cate_inputs


def _get_mlp_regressor(
    hidden_layers: Tuple[int, ...] = (100, 50),
    max_iter: int = 200,
    random_state: int = 42,
) -> MLPRegressor:
    """Create configured MLPRegressor with early stopping.

    Parameters
    ----------
    hidden_layers : tuple of int
        Hidden layer sizes.
    max_iter : int
        Maximum training iterations.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    MLPRegressor
        Configured regressor with early stopping.
    """
    return MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
        n_iter_no_change=10,
    )


def _get_mlp_classifier(
    hidden_layers: Tuple[int, ...] = (100, 50),
    max_iter: int = 200,
    random_state: int = 42,
) -> MLPClassifier:
    """Create configured MLPClassifier with early stopping.

    Parameters
    ----------
    hidden_layers : tuple of int
        Hidden layer sizes.
    max_iter : int
        Maximum training iterations.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    MLPClassifier
        Configured classifier with early stopping.
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
        n_iter_no_change=10,
    )


def neural_s_learner(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    hidden_layers: Tuple[int, ...] = (100, 50),
    max_iter: int = 200,
    alpha: float = 0.05,
) -> CATEResult:
    """Neural S-Learner for CATE estimation.

    Trains a single neural network on (X, T) to predict Y, then computes
    CATE as the difference in predictions with T=1 vs T=0.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y, shape (n,).
    treatment : np.ndarray
        Binary treatment indicator T, shape (n,).
    covariates : np.ndarray
        Covariate matrix X, shape (n, p) or (n,).
    hidden_layers : tuple of int, default=(100, 50)
        Hidden layer sizes for the neural network.
    max_iter : int, default=200
        Maximum training iterations.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    CATEResult
        Dictionary with cate, ate, ate_se, ci_lower, ci_upper, method.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> X = np.random.randn(n, 2)
    >>> T = np.random.binomial(1, 0.5, n)
    >>> Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)
    >>> result = neural_s_learner(Y, T, X)
    >>> abs(result["ate"] - 2.0) < 0.5
    True
    """
    # Validate inputs
    validate_cate_inputs(outcomes, treatment, covariates)

    # Ensure 2D covariates
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n = len(outcomes)

    # Augment features with treatment
    X_augmented = np.column_stack([covariates, treatment])

    # Fit single neural network
    model = _get_mlp_regressor(hidden_layers, max_iter)
    model.fit(X_augmented, outcomes)

    # Create counterfactual feature matrices
    X_treated = np.column_stack([covariates, np.ones(n)])
    X_control = np.column_stack([covariates, np.zeros(n)])

    # Predict potential outcomes
    mu_1 = model.predict(X_treated)
    mu_0 = model.predict(X_control)

    # Compute CATE
    cate = mu_1 - mu_0
    ate = float(np.mean(cate))

    # Compute SE from residuals (conservative approach)
    predictions = model.predict(X_augmented)
    residuals = outcomes - predictions
    residual_var = np.var(residuals, ddof=1)

    n_treated = int(np.sum(treatment))
    n_control = n - n_treated
    ate_se = float(np.sqrt(residual_var * (1 / n_treated + 1 / n_control)))

    # Confidence interval
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z * ate_se
    ci_upper = ate + z * ate_se

    return CATEResult(
        cate=cate,
        ate=ate,
        ate_se=ate_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        method="neural_s_learner",
    )


def neural_t_learner(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    hidden_layers: Tuple[int, ...] = (100, 50),
    max_iter: int = 200,
    alpha: float = 0.05,
) -> CATEResult:
    """Neural T-Learner for CATE estimation.

    Trains separate neural networks for treated and control groups,
    then computes CATE as the difference in predictions.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y, shape (n,).
    treatment : np.ndarray
        Binary treatment indicator T, shape (n,).
    covariates : np.ndarray
        Covariate matrix X, shape (n, p) or (n,).
    hidden_layers : tuple of int, default=(100, 50)
        Hidden layer sizes for each neural network.
    max_iter : int, default=200
        Maximum training iterations.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    CATEResult
        Dictionary with cate, ate, ate_se, ci_lower, ci_upper, method.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> X = np.random.randn(n, 2)
    >>> T = np.random.binomial(1, 0.5, n)
    >>> Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)
    >>> result = neural_t_learner(Y, T, X)
    >>> abs(result["ate"] - 2.0) < 0.5
    True
    """
    # Validate inputs
    validate_cate_inputs(outcomes, treatment, covariates)

    # Ensure 2D covariates
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n = len(outcomes)

    # Split by treatment
    treated_mask = treatment == 1
    control_mask = treatment == 0

    X_treated = covariates[treated_mask]
    X_control = covariates[control_mask]
    Y_treated = outcomes[treated_mask]
    Y_control = outcomes[control_mask]

    # Fit separate neural networks
    model_1 = _get_mlp_regressor(hidden_layers, max_iter, random_state=42)
    model_0 = _get_mlp_regressor(hidden_layers, max_iter, random_state=43)

    model_1.fit(X_treated, Y_treated)
    model_0.fit(X_control, Y_control)

    # Predict for all units
    mu_1 = model_1.predict(covariates)
    mu_0 = model_0.predict(covariates)

    # Compute CATE
    cate = mu_1 - mu_0
    ate = float(np.mean(cate))

    # Compute SE from pooled residual variance
    residuals_1 = Y_treated - model_1.predict(X_treated)
    residuals_0 = Y_control - model_0.predict(X_control)

    var_1 = np.var(residuals_1, ddof=1) if len(residuals_1) > 1 else 0.0
    var_0 = np.var(residuals_0, ddof=1) if len(residuals_0) > 1 else 0.0

    n_treated = len(Y_treated)
    n_control = len(Y_control)
    ate_se = float(np.sqrt(var_1 / n_treated + var_0 / n_control))

    # Confidence interval
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z * ate_se
    ci_upper = ate + z * ate_se

    return CATEResult(
        cate=cate,
        ate=ate,
        ate_se=ate_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        method="neural_t_learner",
    )


def neural_x_learner(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    hidden_layers: Tuple[int, ...] = (100, 50),
    max_iter: int = 200,
    alpha: float = 0.05,
) -> CATEResult:
    """Neural X-Learner for CATE estimation.

    Four-stage cross-learner:
    1. Fit T-learner models (mu_1, mu_0)
    2. Compute imputed treatment effects
    3. Fit CATE models on imputed effects
    4. Combine with propensity weighting

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y, shape (n,).
    treatment : np.ndarray
        Binary treatment indicator T, shape (n,).
    covariates : np.ndarray
        Covariate matrix X, shape (n, p) or (n,).
    hidden_layers : tuple of int, default=(100, 50)
        Hidden layer sizes for neural networks.
    max_iter : int, default=200
        Maximum training iterations.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    CATEResult
        Dictionary with cate, ate, ate_se, ci_lower, ci_upper, method.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> X = np.random.randn(n, 2)
    >>> T = np.random.binomial(1, 0.5, n)
    >>> Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)
    >>> result = neural_x_learner(Y, T, X)
    >>> abs(result["ate"] - 2.0) < 0.5
    True
    """
    # Validate inputs
    validate_cate_inputs(outcomes, treatment, covariates)

    # Ensure 2D covariates
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n = len(outcomes)

    # Split by treatment
    treated_mask = treatment == 1
    control_mask = treatment == 0

    X_treated = covariates[treated_mask]
    X_control = covariates[control_mask]
    Y_treated = outcomes[treated_mask]
    Y_control = outcomes[control_mask]

    # Stage 1: Fit T-learner models
    model_1 = _get_mlp_regressor(hidden_layers, max_iter, random_state=42)
    model_0 = _get_mlp_regressor(hidden_layers, max_iter, random_state=43)

    model_1.fit(X_treated, Y_treated)
    model_0.fit(X_control, Y_control)

    # Stage 2: Compute imputed treatment effects
    # For treated: D_1 = Y(1) - mu_0(X)
    # For control: D_0 = mu_1(X) - Y(0)
    D_1 = Y_treated - model_0.predict(X_treated)
    D_0 = model_1.predict(X_control) - Y_control

    # Stage 3: Fit CATE models on imputed effects
    tau_model_1 = _get_mlp_regressor(hidden_layers, max_iter, random_state=44)
    tau_model_0 = _get_mlp_regressor(hidden_layers, max_iter, random_state=45)

    tau_model_1.fit(X_treated, D_1)
    tau_model_0.fit(X_control, D_0)

    # Stage 4: Fit propensity model
    prop_model = _get_mlp_classifier(hidden_layers, max_iter, random_state=46)
    prop_model.fit(covariates, treatment.astype(int))
    propensity = prop_model.predict_proba(covariates)[:, 1]
    propensity = np.clip(propensity, 0.01, 0.99)

    # Combine with propensity weighting
    tau_1_pred = tau_model_1.predict(covariates)
    tau_0_pred = tau_model_0.predict(covariates)

    # X-learner combination: weight by propensity
    cate = propensity * tau_0_pred + (1 - propensity) * tau_1_pred
    ate = float(np.mean(cate))

    # Compute SE from imputed effect variance
    var_D1 = np.var(D_1, ddof=1) if len(D_1) > 1 else 0.0
    var_D0 = np.var(D_0, ddof=1) if len(D_0) > 1 else 0.0

    n_treated = len(Y_treated)
    n_control = len(Y_control)
    ate_se = float(np.sqrt(var_D1 / n_treated + var_D0 / n_control))

    # Confidence interval
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z * ate_se
    ci_upper = ate + z * ate_se

    return CATEResult(
        cate=cate,
        ate=ate,
        ate_se=ate_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        method="neural_x_learner",
    )


def neural_r_learner(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    hidden_layers: Tuple[int, ...] = (100, 50),
    max_iter: int = 200,
    alpha: float = 0.05,
) -> CATEResult:
    """Neural R-Learner for CATE estimation.

    Robinson transformation with neural networks:
    1. Estimate propensity e(X) and outcome m(X) with neural networks
    2. Residualize: Y_tilde = Y - m(X), T_tilde = T - e(X)
    3. Estimate CATE via weighted regression on transformed features

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y, shape (n,).
    treatment : np.ndarray
        Binary treatment indicator T, shape (n,).
    covariates : np.ndarray
        Covariate matrix X, shape (n, p) or (n,).
    hidden_layers : tuple of int, default=(100, 50)
        Hidden layer sizes for neural networks.
    max_iter : int, default=200
        Maximum training iterations.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    CATEResult
        Dictionary with cate, ate, ate_se, ci_lower, ci_upper, method.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> X = np.random.randn(n, 2)
    >>> T = np.random.binomial(1, 0.5, n)
    >>> Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)
    >>> result = neural_r_learner(Y, T, X)
    >>> abs(result["ate"] - 2.0) < 0.5
    True
    """
    # Validate inputs
    validate_cate_inputs(outcomes, treatment, covariates)

    # Ensure 2D covariates
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n = len(outcomes)

    # Stage 1: Estimate propensity e(X)
    prop_model = _get_mlp_classifier(hidden_layers, max_iter, random_state=42)
    prop_model.fit(covariates, treatment.astype(int))
    e_hat = prop_model.predict_proba(covariates)[:, 1]
    e_hat = np.clip(e_hat, 0.01, 0.99)

    # Stage 1: Estimate outcome m(X)
    outcome_model = _get_mlp_regressor(hidden_layers, max_iter, random_state=43)
    outcome_model.fit(covariates, outcomes)
    m_hat = outcome_model.predict(covariates)

    # Stage 2: Residualize
    Y_tilde = outcomes - m_hat
    T_tilde = treatment - e_hat

    # Stage 3: Estimate CATE via transformed features
    # tau(X) = argmin sum_i (Y_tilde_i - tau(X_i) * T_tilde_i)^2
    # Transform: X * T_tilde gives features, T_tilde gives intercept contribution
    X_transformed = covariates * T_tilde[:, np.newaxis]
    X_with_intercept = np.column_stack([X_transformed, T_tilde])

    cate_model = _get_mlp_regressor(hidden_layers, max_iter, random_state=44)
    cate_model.fit(X_with_intercept, Y_tilde)

    # Predict CATE
    X_pred = np.column_stack([covariates, np.ones(n)])
    cate = cate_model.predict(X_pred)
    ate = float(np.mean(cate))

    # Compute SE using pseudo-outcomes
    # Pseudo-outcome: psi = Y_tilde / T_tilde + tau(X)
    # Only use observations with |T_tilde| > threshold
    valid_mask = np.abs(T_tilde) > 0.1
    if np.sum(valid_mask) > 10:
        pseudo_outcomes = Y_tilde[valid_mask] / T_tilde[valid_mask]
        ate_se = float(np.std(pseudo_outcomes, ddof=1) / np.sqrt(np.sum(valid_mask)))
    else:
        # Fallback: use residual variance
        residuals = Y_tilde - cate * T_tilde
        ate_se = float(np.std(residuals, ddof=1) / np.sqrt(n))

    # Confidence interval
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z * ate_se
    ci_upper = ate + z * ate_se

    return CATEResult(
        cate=cate,
        ate=ate,
        ate_se=ate_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        method="neural_r_learner",
    )
