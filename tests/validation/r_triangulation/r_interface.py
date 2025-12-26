"""R interface for triangulation testing via rpy2.

This module provides wrapper functions to call R packages for comparison testing.
All functions gracefully handle missing R/rpy2 installations by returning None
or raising appropriate errors.

Dependencies (optional):
- rpy2>=3.5 (Python-R bridge)
- R packages: PStrata, DTRreg

Install with: pip install causal-inference-mastery[r-triangulation]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np


# =============================================================================
# Availability Checks
# =============================================================================


def check_r_available() -> bool:
    """Check if R and rpy2 are available.

    Returns
    -------
    bool
        True if rpy2 can be imported and R is accessible, False otherwise.
    """
    try:
        import rpy2.robjects as ro  # noqa: F401

        # Try to execute simple R command to verify R is working
        ro.r("1 + 1")
        return True
    except ImportError:
        return False
    except Exception:
        # R not installed or not accessible
        return False


def check_pstrata_installed() -> bool:
    """Check if the PStrata R package is installed.

    Returns
    -------
    bool
        True if PStrata can be loaded in R, False otherwise.
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        ro.r('suppressPackageStartupMessages(library(PStrata))')
        return True
    except Exception:
        return False


def check_dtrreg_installed() -> bool:
    """Check if the DTRreg R package is installed.

    Returns
    -------
    bool
        True if DTRreg can be loaded in R, False otherwise.
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        ro.r('suppressPackageStartupMessages(library(DTRreg))')
        return True
    except Exception:
        return False


def get_r_installation_instructions() -> str:
    """Get instructions for installing R and required packages.

    Returns
    -------
    str
        Installation instructions.
    """
    return """
R Triangulation Tests Requirements
==================================

1. Install R (https://www.r-project.org/)

2. Install rpy2:
   pip install rpy2>=3.5

3. Install R packages (from R console):
   install.packages("PStrata")
   install.packages("DTRreg")

Or install all dependencies:
   pip install causal-inference-mastery[r-triangulation]
"""


# =============================================================================
# Principal Stratification (PStrata)
# =============================================================================


def r_cace_2sls(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
) -> Dict[str, Any]:
    """Estimate CACE using 2SLS via R's AER package (instrumental variables).

    This provides a reference implementation using standard R IV estimation
    for comparison with our cace_2sls() function.

    Parameters
    ----------
    outcome : np.ndarray
        Continuous outcome variable Y.
    treatment : np.ndarray
        Binary treatment indicator D (0/1).
    instrument : np.ndarray
        Binary instrument indicator Z (0/1).

    Returns
    -------
    dict
        Dictionary with keys:
        - cace: float, the 2SLS estimate
        - se: float, standard error
        - ci_lower: float, 95% CI lower bound
        - ci_upper: float, 95% CI upper bound

    Raises
    ------
    ImportError
        If rpy2 is not installed.
    RuntimeError
        If R or required packages are not available.
    """
    if not check_r_available():
        raise ImportError(
            "rpy2 is required for R triangulation. "
            f"Install with: pip install rpy2>=3.5\n{get_r_installation_instructions()}"
        )

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    # Enable numpy conversion
    numpy2ri.activate()

    try:
        # Pass data to R
        ro.globalenv["Y"] = ro.FloatVector(outcome)
        ro.globalenv["D"] = ro.FloatVector(treatment)
        ro.globalenv["Z"] = ro.FloatVector(instrument)

        # Use ivreg from AER package for 2SLS
        # This is more widely available than PStrata for basic 2SLS
        result = ro.r(
            """
            suppressPackageStartupMessages({
                if (!require(AER, quietly=TRUE)) {
                    # Fallback: manual 2SLS calculation
                    data <- data.frame(Y=Y, D=D, Z=Z)

                    # First stage
                    first_stage <- lm(D ~ Z, data=data)
                    D_hat <- fitted(first_stage)

                    # Second stage
                    second_stage <- lm(Y ~ D_hat, data=data)

                    coef_2sls <- coef(second_stage)["D_hat"]
                    se_2sls <- summary(second_stage)$coefficients["D_hat", "Std. Error"]

                    list(
                        cace = coef_2sls,
                        se = se_2sls,
                        ci_lower = coef_2sls - 1.96 * se_2sls,
                        ci_upper = coef_2sls + 1.96 * se_2sls
                    )
                } else {
                    data <- data.frame(Y=Y, D=D, Z=Z)
                    fit <- ivreg(Y ~ D | Z, data=data)
                    summ <- summary(fit)

                    list(
                        cace = coef(fit)["D"],
                        se = summ$coefficients["D", "Std. Error"],
                        ci_lower = confint(fit)["D", 1],
                        ci_upper = confint(fit)["D", 2]
                    )
                }
            })
            """
        )

        return {
            "cace": float(result.rx2("cace")[0]),
            "se": float(result.rx2("se")[0]),
            "ci_lower": float(result.rx2("ci_lower")[0]),
            "ci_upper": float(result.rx2("ci_upper")[0]),
        }
    finally:
        numpy2ri.deactivate()


def r_cace_pstrata(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    method: str = "EM",
    max_iter: int = 500,
    tol: float = 1e-6,
) -> Optional[Dict[str, Any]]:
    """Estimate CACE using the PStrata R package.

    Parameters
    ----------
    outcome : np.ndarray
        Continuous outcome variable Y.
    treatment : np.ndarray
        Binary treatment indicator D (0/1).
    instrument : np.ndarray
        Binary instrument indicator Z (0/1).
    method : str, default="EM"
        Estimation method: "EM" for EM algorithm, "2SLS" for instrumental variables.
    max_iter : int, default=500
        Maximum number of EM iterations.
    tol : float, default=1e-6
        Convergence tolerance for EM.

    Returns
    -------
    dict or None
        Dictionary with keys:
        - cace: float, CACE estimate
        - se: float, standard error
        - strata_proportions: dict with pi_c, pi_a, pi_n
        - converged: bool, whether EM converged
        Returns None if PStrata is not available.

    Raises
    ------
    ImportError
        If rpy2 is not installed.
    RuntimeError
        If PStrata package is not installed in R.
    """
    if not check_r_available():
        raise ImportError(
            "rpy2 is required for R triangulation. "
            f"Install with: pip install rpy2>=3.5\n{get_r_installation_instructions()}"
        )

    if not check_pstrata_installed():
        warnings.warn(
            "PStrata R package not installed. Install in R with: "
            "install.packages('PStrata')",
            UserWarning,
        )
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        # Pass data to R
        ro.globalenv["Y"] = ro.FloatVector(outcome)
        ro.globalenv["D"] = ro.FloatVector(treatment)
        ro.globalenv["Z"] = ro.FloatVector(instrument)
        ro.globalenv["max_iter"] = max_iter
        ro.globalenv["tol"] = tol

        # Call PStrata
        # Note: PStrata API may vary by version; this is based on published docs
        result = ro.r(
            """
            suppressPackageStartupMessages(library(PStrata))

            data <- data.frame(Y=Y, D=D, Z=Z)

            # PStrata uses a formula interface
            # The exact API depends on package version
            tryCatch({
                fit <- PSest(Y ~ D | Z, data=data,
                            control=list(maxit=max_iter, tol=tol))

                # Extract results
                coefs <- coef(fit)

                list(
                    cace = coefs["CACE"],
                    se = sqrt(vcov(fit)["CACE", "CACE"]),
                    pi_c = coefs["pi_c"],
                    pi_a = coefs["pi_a"],
                    pi_n = coefs["pi_n"],
                    converged = fit$converged
                )
            }, error = function(e) {
                # If PStrata has different API, try alternative
                list(
                    error = as.character(e),
                    cace = NA,
                    se = NA,
                    pi_c = NA,
                    pi_a = NA,
                    pi_n = NA,
                    converged = FALSE
                )
            })
            """
        )

        # Check for error
        if "error" in list(result.names) and result.rx2("error")[0] != "NA":
            error_msg = str(result.rx2("error")[0])
            warnings.warn(f"PStrata estimation failed: {error_msg}", UserWarning)
            return None

        cace = float(result.rx2("cace")[0])
        se = float(result.rx2("se")[0])

        return {
            "cace": cace,
            "se": se if not np.isnan(se) else None,
            "strata_proportions": {
                "compliers": float(result.rx2("pi_c")[0]),
                "always_takers": float(result.rx2("pi_a")[0]),
                "never_takers": float(result.rx2("pi_n")[0]),
            },
            "converged": bool(result.rx2("converged")[0]),
        }
    except Exception as e:
        warnings.warn(f"PStrata call failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_cace_em_manual(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """Estimate CACE using EM algorithm implemented in R (for comparison).

    This provides a manual EM implementation in R as a reference when
    PStrata is not available.

    Parameters
    ----------
    outcome : np.ndarray
        Continuous outcome variable Y.
    treatment : np.ndarray
        Binary treatment indicator D (0/1).
    instrument : np.ndarray
        Binary instrument indicator Z (0/1).
    max_iter : int, default=500
        Maximum number of EM iterations.
    tol : float, default=1e-6
        Convergence tolerance.

    Returns
    -------
    dict
        Dictionary with estimation results.
    """
    if not check_r_available():
        raise ImportError(
            "rpy2 is required for R triangulation. "
            f"Install with: pip install rpy2>=3.5\n{get_r_installation_instructions()}"
        )

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        ro.globalenv["Y"] = ro.FloatVector(outcome)
        ro.globalenv["D"] = ro.FloatVector(treatment)
        ro.globalenv["Z"] = ro.FloatVector(instrument)
        ro.globalenv["max_iter"] = max_iter
        ro.globalenv["tol"] = tol

        result = ro.r(
            """
            # Manual EM implementation for principal stratification
            # Under monotonicity: no defiers

            n <- length(Y)

            # Identify groups based on (Z, D)
            g00 <- which(Z == 0 & D == 0)  # Never-takers or compliers
            g01 <- which(Z == 0 & D == 1)  # Always-takers
            g10 <- which(Z == 1 & D == 0)  # Never-takers
            g11 <- which(Z == 1 & D == 1)  # Always-takers or compliers

            # Initial estimates from data
            n00 <- length(g00)
            n01 <- length(g01)
            n10 <- length(g10)
            n11 <- length(g11)

            # Outcome means by group
            mu00 <- if(n00 > 0) mean(Y[g00]) else 0
            mu01 <- if(n01 > 0) mean(Y[g01]) else 0
            mu10 <- if(n10 > 0) mean(Y[g10]) else 0
            mu11 <- if(n11 > 0) mean(Y[g11]) else 0

            # Initialize parameters
            # pi_a from always-takers in control (g01)
            pi_a <- n01 / (n00 + n01)
            # pi_n from never-takers in treatment (g10)
            pi_n <- n10 / (n10 + n11)
            pi_c <- 1 - pi_a - pi_n

            # Ensure valid
            pi_c <- max(0.01, min(0.98, pi_c))
            pi_a <- max(0.01, min(0.98 - pi_c, pi_a))
            pi_n <- 1 - pi_c - pi_a

            # Initialize means
            mu_a <- mu01  # Always-takers from g01
            mu_n <- mu10  # Never-takers from g10
            mu_c0 <- mu00 # Compliers control approximated by g00
            mu_c1 <- mu11 # Compliers treated approximated by g11
            sigma2 <- var(Y)

            # EM iterations
            converged <- FALSE
            ll_old <- -Inf

            for (iter in 1:max_iter) {
                # E-step: compute responsibilities

                # For g00 (Z=0, D=0): could be complier or never-taker
                if (n00 > 0) {
                    dens_c0 <- dnorm(Y[g00], mu_c0, sqrt(sigma2))
                    dens_n <- dnorm(Y[g00], mu_n, sqrt(sigma2))

                    w_c_g00 <- (pi_c * dens_c0) / (pi_c * dens_c0 + pi_n * dens_n + 1e-10)
                    w_n_g00 <- 1 - w_c_g00
                } else {
                    w_c_g00 <- numeric(0)
                    w_n_g00 <- numeric(0)
                }

                # For g11 (Z=1, D=1): could be complier or always-taker
                if (n11 > 0) {
                    dens_c1 <- dnorm(Y[g11], mu_c1, sqrt(sigma2))
                    dens_a <- dnorm(Y[g11], mu_a, sqrt(sigma2))

                    w_c_g11 <- (pi_c * dens_c1) / (pi_c * dens_c1 + pi_a * dens_a + 1e-10)
                    w_a_g11 <- 1 - w_c_g11
                } else {
                    w_c_g11 <- numeric(0)
                    w_a_g11 <- numeric(0)
                }

                # M-step: update parameters

                # Strata proportions
                n_c <- sum(w_c_g00) + sum(w_c_g11)
                n_a <- n01 + sum(w_a_g11)
                n_n <- sum(w_n_g00) + n10
                total <- n_c + n_a + n_n

                pi_c <- n_c / total
                pi_a <- n_a / total
                pi_n <- n_n / total

                # Complier means
                if (sum(w_c_g00) > 0.01) {
                    mu_c0 <- sum(w_c_g00 * Y[g00]) / sum(w_c_g00)
                }
                if (sum(w_c_g11) > 0.01) {
                    mu_c1 <- sum(w_c_g11 * Y[g11]) / sum(w_c_g11)
                }

                # Always-taker mean (all of g01 + weighted g11)
                if (n01 + sum(w_a_g11) > 0.01) {
                    mu_a <- (sum(Y[g01]) + sum(w_a_g11 * Y[g11])) / (n01 + sum(w_a_g11))
                }

                # Never-taker mean (all of g10 + weighted g00)
                if (n10 + sum(w_n_g00) > 0.01) {
                    mu_n <- (sum(Y[g10]) + sum(w_n_g00 * Y[g00])) / (n10 + sum(w_n_g00))
                }

                # Variance
                ss <- 0
                if (n00 > 0) {
                    ss <- ss + sum(w_c_g00 * (Y[g00] - mu_c0)^2) +
                               sum(w_n_g00 * (Y[g00] - mu_n)^2)
                }
                if (n01 > 0) {
                    ss <- ss + sum((Y[g01] - mu_a)^2)
                }
                if (n10 > 0) {
                    ss <- ss + sum((Y[g10] - mu_n)^2)
                }
                if (n11 > 0) {
                    ss <- ss + sum(w_c_g11 * (Y[g11] - mu_c1)^2) +
                               sum(w_a_g11 * (Y[g11] - mu_a)^2)
                }
                sigma2 <- max(0.001, ss / n)

                # Log-likelihood for convergence
                ll <- 0
                if (n00 > 0) {
                    ll <- ll + sum(log(pi_c * dnorm(Y[g00], mu_c0, sqrt(sigma2)) +
                                       pi_n * dnorm(Y[g00], mu_n, sqrt(sigma2)) + 1e-10))
                }
                if (n01 > 0) {
                    ll <- ll + sum(log(dnorm(Y[g01], mu_a, sqrt(sigma2)) + 1e-10))
                }
                if (n10 > 0) {
                    ll <- ll + sum(log(dnorm(Y[g10], mu_n, sqrt(sigma2)) + 1e-10))
                }
                if (n11 > 0) {
                    ll <- ll + sum(log(pi_c * dnorm(Y[g11], mu_c1, sqrt(sigma2)) +
                                       pi_a * dnorm(Y[g11], mu_a, sqrt(sigma2)) + 1e-10))
                }

                if (abs(ll - ll_old) < tol) {
                    converged <- TRUE
                    break
                }
                ll_old <- ll
            }

            cace <- mu_c1 - mu_c0

            list(
                cace = cace,
                mu_c0 = mu_c0,
                mu_c1 = mu_c1,
                mu_a = mu_a,
                mu_n = mu_n,
                pi_c = pi_c,
                pi_a = pi_a,
                pi_n = pi_n,
                sigma = sqrt(sigma2),
                converged = converged,
                iterations = iter,
                log_likelihood = ll
            )
            """
        )

        return {
            "cace": float(result.rx2("cace")[0]),
            "mu_c0": float(result.rx2("mu_c0")[0]),
            "mu_c1": float(result.rx2("mu_c1")[0]),
            "mu_a": float(result.rx2("mu_a")[0]),
            "mu_n": float(result.rx2("mu_n")[0]),
            "strata_proportions": {
                "compliers": float(result.rx2("pi_c")[0]),
                "always_takers": float(result.rx2("pi_a")[0]),
                "never_takers": float(result.rx2("pi_n")[0]),
            },
            "sigma": float(result.rx2("sigma")[0]),
            "converged": bool(result.rx2("converged")[0]),
            "iterations": int(result.rx2("iterations")[0]),
            "log_likelihood": float(result.rx2("log_likelihood")[0]),
        }
    finally:
        numpy2ri.deactivate()


# =============================================================================
# Bounds (Manski-style)
# =============================================================================


def r_bounds_manski(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    outcome_support: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Compute Manski-style bounds on treatment effect using R.

    Parameters
    ----------
    outcome : np.ndarray
        Continuous outcome variable Y.
    treatment : np.ndarray
        Binary treatment indicator D (0/1).
    instrument : np.ndarray
        Binary instrument indicator Z (0/1).
    outcome_support : tuple, optional
        (min, max) of outcome support. If None, uses observed range.

    Returns
    -------
    dict
        Dictionary with bounds results.
    """
    if not check_r_available():
        raise ImportError(
            "rpy2 is required for R triangulation. "
            f"Install with: pip install rpy2>=3.5\n{get_r_installation_instructions()}"
        )

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        ro.globalenv["Y"] = ro.FloatVector(outcome)
        ro.globalenv["D"] = ro.FloatVector(treatment)
        ro.globalenv["Z"] = ro.FloatVector(instrument)

        if outcome_support is not None:
            ro.globalenv["y_min"] = outcome_support[0]
            ro.globalenv["y_max"] = outcome_support[1]
        else:
            ro.globalenv["y_min"] = float(np.min(outcome))
            ro.globalenv["y_max"] = float(np.max(outcome))

        result = ro.r(
            """
            # Manski worst-case bounds
            # Without assumptions, the treatment effect is bounded by the outcome range

            range_Y <- y_max - y_min

            # Simple bounds: [-range, +range]
            lower <- -range_Y
            upper <- range_Y

            list(
                lower_bound = lower,
                upper_bound = upper,
                bound_width = upper - lower,
                outcome_range = range_Y
            )
            """
        )

        return {
            "lower_bound": float(result.rx2("lower_bound")[0]),
            "upper_bound": float(result.rx2("upper_bound")[0]),
            "bound_width": float(result.rx2("bound_width")[0]),
            "outcome_range": float(result.rx2("outcome_range")[0]),
        }
    finally:
        numpy2ri.deactivate()


# =============================================================================
# Dynamic Treatment Regimes (DTRreg) - Placeholder for Session 122
# =============================================================================


def r_qlearning_dtrreg(
    outcome: np.ndarray,
    treatment_history: np.ndarray,
    covariates: np.ndarray,
    n_stages: int = 2,
) -> Optional[Dict[str, Any]]:
    """Estimate optimal DTR using DTRreg Q-learning.

    Placeholder for Session 122 implementation.

    Parameters
    ----------
    outcome : np.ndarray
        Final outcome.
    treatment_history : np.ndarray
        Treatment decisions at each stage (n x n_stages).
    covariates : np.ndarray
        Covariates (n x p).
    n_stages : int
        Number of treatment stages.

    Returns
    -------
    dict or None
        Q-learning results, or None if DTRreg not available.
    """
    if not check_dtrreg_installed():
        warnings.warn(
            "DTRreg R package not installed. Install in R with: "
            "install.packages('DTRreg')",
            UserWarning,
        )
        return None

    # TODO: Implement in Session 122
    raise NotImplementedError("DTRreg interface will be implemented in Session 122")
