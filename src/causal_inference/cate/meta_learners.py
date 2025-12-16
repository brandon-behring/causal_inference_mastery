"""Meta-learners for CATE estimation.

Implements S-Learner and T-Learner approaches for estimating heterogeneous
treatment effects (Conditional Average Treatment Effects, CATE).

Meta-learners are generic frameworks that use any supervised learning algorithm
as a base learner to estimate treatment effect heterogeneity.

Algorithm Overview
------------------
**S-Learner** (Single model):
1. Fit μ(X, T) on combined data: augment X with T as additional feature
2. Predict: μ̂(xᵢ, 1) and μ̂(xᵢ, 0) for each unit
3. CATE: τ̂(xᵢ) = μ̂(xᵢ, 1) - μ̂(xᵢ, 0)

**T-Learner** (Two models):
1. Fit μ₀(X) on control group: X[T=0] → Y[T=0]
2. Fit μ₁(X) on treated group: X[T=1] → Y[T=1]
3. CATE: τ̂(xᵢ) = μ̂₁(xᵢ) - μ̂₀(xᵢ)

References
----------
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects
  using machine learning." PNAS 116(10): 4156-4165.
"""

import numpy as np
from typing import Literal
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from .base import CATEResult, validate_cate_inputs


def _get_model(model_type: str, **kwargs):
    """Get sklearn model instance by name.

    Parameters
    ----------
    model_type : str
        One of "linear", "ridge", "random_forest".
    **kwargs
        Additional arguments passed to model constructor.

    Returns
    -------
    sklearn estimator
        Instantiated model.

    Raises
    ------
    ValueError
        If model_type is not recognized.
    """
    if model_type == "linear":
        return LinearRegression(**kwargs)
    elif model_type == "ridge":
        return Ridge(**kwargs)
    elif model_type == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=42, **kwargs)
    else:
        raise ValueError(
            f"CRITICAL ERROR: Unknown model type.\n"
            f"Function: _get_model\n"
            f"Got: model_type = '{model_type}'\n"
            f"Valid options: 'linear', 'ridge', 'random_forest'"
        )


def s_learner(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    model: Literal["linear", "ridge", "random_forest"] = "linear",
    alpha: float = 0.05,
) -> CATEResult:
    """Estimate CATE using S-Learner (Single model approach).

    The S-Learner fits a single model μ(X, T) that includes treatment as a feature,
    then estimates CATE by comparing predictions under T=1 vs T=0.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y of shape (n,).
    treatment : np.ndarray
        Binary treatment indicator of shape (n,). Values must be 0 or 1.
    covariates : np.ndarray
        Covariate matrix X of shape (n, p). Can be (n,) for single covariate.
    model : {"linear", "ridge", "random_forest"}, default="linear"
        Base learner for outcome modeling:
        - "linear": OLS regression
        - "ridge": L2-regularized regression
        - "random_forest": Random forest regressor
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    CATEResult
        Dictionary with keys:
        - cate: Individual treatment effects τ(xᵢ) of shape (n,)
        - ate: Average treatment effect (mean of CATE)
        - ate_se: Standard error of ATE
        - ci_lower: Lower bound of (1-α)% CI
        - ci_upper: Upper bound of (1-α)% CI
        - method: "s_learner"

    Raises
    ------
    ValueError
        If inputs are invalid or model type is unknown.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 200
    >>> X = np.random.randn(n, 2)
    >>> T = np.random.binomial(1, 0.5, n)
    >>> Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)  # True ATE = 2
    >>> result = s_learner(Y, T, X)
    >>> print(f"ATE: {result['ate']:.2f}")
    ATE: 2.00

    Notes
    -----
    **Known Limitation**: S-learner is biased toward 0 when the treatment effect
    is small relative to the main effects of X. This is because the model may
    "regularize away" the treatment effect if it's not a strong predictor.

    The S-learner works best when:
    - Treatment effects are large relative to outcome variance
    - The base learner can capture treatment-covariate interactions

    See Also
    --------
    t_learner : Two-model approach that avoids regularization bias.
    """
    # Validate inputs
    outcomes, treatment, covariates = validate_cate_inputs(
        outcomes, treatment, covariates
    )

    n = len(outcomes)

    # Build augmented feature matrix [X | T]
    X_augmented = np.column_stack([covariates, treatment])

    # Fit single model on combined data
    learner = _get_model(model)
    learner.fit(X_augmented, outcomes)

    # Create counterfactual feature matrices
    X_treated = np.column_stack([covariates, np.ones(n)])
    X_control = np.column_stack([covariates, np.zeros(n)])

    # Predict potential outcomes
    mu_1 = learner.predict(X_treated)
    mu_0 = learner.predict(X_control)

    # CATE estimates
    cate = mu_1 - mu_0

    # Compute ATE
    ate = np.mean(cate)

    # SE estimation: Use residual-based approach
    # For ATE, we use the standard error of the treatment coefficient
    # Residuals from the model
    y_pred = learner.predict(X_augmented)
    residuals = outcomes - y_pred

    # SE of ATE using influence function approach
    # For S-learner: SE ≈ sqrt(Var(residuals) * (1/n1 + 1/n0))
    n1 = np.sum(treatment == 1)
    n0 = np.sum(treatment == 0)
    residual_var = np.var(residuals, ddof=1)
    ate_se = np.sqrt(residual_var * (1/n1 + 1/n0))

    # Confidence interval using normal approximation
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z_crit * ate_se
    ci_upper = ate + z_crit * ate_se

    return CATEResult(
        cate=cate,
        ate=float(ate),
        ate_se=float(ate_se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        method="s_learner",
    )


def t_learner(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    model: Literal["linear", "ridge", "random_forest"] = "linear",
    alpha: float = 0.05,
) -> CATEResult:
    """Estimate CATE using T-Learner (Two-model approach).

    The T-Learner fits separate models for treated and control groups,
    then estimates CATE as the difference in predictions.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y of shape (n,).
    treatment : np.ndarray
        Binary treatment indicator of shape (n,). Values must be 0 or 1.
    covariates : np.ndarray
        Covariate matrix X of shape (n, p). Can be (n,) for single covariate.
    model : {"linear", "ridge", "random_forest"}, default="linear"
        Base learner for outcome modeling:
        - "linear": OLS regression
        - "ridge": L2-regularized regression
        - "random_forest": Random forest regressor
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    CATEResult
        Dictionary with keys:
        - cate: Individual treatment effects τ(xᵢ) of shape (n,)
        - ate: Average treatment effect (mean of CATE)
        - ate_se: Standard error of ATE
        - ci_lower: Lower bound of (1-α)% CI
        - ci_upper: Upper bound of (1-α)% CI
        - method: "t_learner"

    Raises
    ------
    ValueError
        If inputs are invalid or model type is unknown.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 200
    >>> X = np.random.randn(n, 2)
    >>> T = np.random.binomial(1, 0.5, n)
    >>> Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)  # True ATE = 2
    >>> result = t_learner(Y, T, X)
    >>> print(f"ATE: {result['ate']:.2f}")
    ATE: 2.01

    Notes
    -----
    **Known Limitation**: T-learner can overfit when treatment and control groups
    have different covariate distributions (lack of overlap). Each model only
    sees part of the covariate space, which can lead to poor extrapolation.

    The T-learner works best when:
    - There is good covariate overlap between treatment groups
    - Sample sizes are large enough to fit two separate models
    - Treatment effect heterogeneity is substantial

    **Advantages over S-learner**:
    - No regularization bias toward zero
    - Can capture different functional forms for μ₁(x) and μ₀(x)

    See Also
    --------
    s_learner : Single-model approach.
    """
    # Validate inputs
    outcomes, treatment, covariates = validate_cate_inputs(
        outcomes, treatment, covariates
    )

    n = len(outcomes)

    # Split data by treatment status
    treated_mask = treatment == 1
    control_mask = treatment == 0

    X_treated = covariates[treated_mask]
    Y_treated = outcomes[treated_mask]

    X_control = covariates[control_mask]
    Y_control = outcomes[control_mask]

    # Fit separate models
    model_1 = _get_model(model)
    model_0 = _get_model(model)

    model_1.fit(X_treated, Y_treated)
    model_0.fit(X_control, Y_control)

    # Predict potential outcomes for all units
    mu_1 = model_1.predict(covariates)
    mu_0 = model_0.predict(covariates)

    # CATE estimates
    cate = mu_1 - mu_0

    # Compute ATE
    ate = np.mean(cate)

    # SE estimation using residuals from each model
    # Residuals for treated and control models
    residuals_1 = Y_treated - model_1.predict(X_treated)
    residuals_0 = Y_control - model_0.predict(X_control)

    n1 = len(Y_treated)
    n0 = len(Y_control)

    # Pooled variance approach for T-learner
    var_1 = np.var(residuals_1, ddof=1)
    var_0 = np.var(residuals_0, ddof=1)
    ate_se = np.sqrt(var_1/n1 + var_0/n0)

    # Confidence interval using normal approximation
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z_crit * ate_se
    ci_upper = ate + z_crit * ate_se

    return CATEResult(
        cate=cate,
        ate=float(ate),
        ate_se=float(ate_se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        method="t_learner",
    )
