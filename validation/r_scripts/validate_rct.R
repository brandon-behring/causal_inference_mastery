#!/usr/bin/env Rscript

# ==============================================================================
# R Reference Implementations for RCT Estimators
# ==============================================================================
#
# Implements all 5 RCT estimators using base R and standard packages.
# Outputs results to CSV for Julia cross-validation.
#
# Purpose: Triangulation validation (Python, Julia, R)
# Why: If Python and Julia share a conceptual bug, R catches it
#
# Estimators:
# 1. SimpleATE - Two-sample t-test (Welch's)
# 2. StratifiedATE - Stratified randomization analysis
# 3. RegressionATE - ANCOVA with HC3 robust SE
# 4. PermutationTest - Exact permutation test (coin package)
# 5. IPWATE - Inverse probability weighting
#
# References:
# - Imbens & Rubin (2015). Causal Inference for Statistics, Social, and Biomedical Sciences
# - Rosenbaum (2002). Observational Studies
# ==============================================================================

# Required packages
if (!require("sandwich", quietly = TRUE)) install.packages("sandwich")
if (!require("coin", quietly = TRUE)) install.packages("coin")

library(sandwich)  # For HC3 robust standard errors
library(coin)      # For exact permutation tests


# ==============================================================================
# 1. SimpleATE: Two-Sample T-Test (Welch's)
# ==============================================================================

#' Simple Average Treatment Effect Estimator
#'
#' Uses Welch's t-test (unequal variances) to estimate ATE in RCTs.
#'
#' @param outcomes Numeric vector of outcomes
#' @param treatment Logical or 0/1 vector of treatment assignment
#' @param alpha Significance level for confidence interval (default 0.05)
#'
#' @return List with estimate, se, ci_lower, ci_upper
simple_ate_r <- function(outcomes, treatment, alpha = 0.05) {
  # Ensure treatment is logical
  treatment <- as.logical(treatment)

  # Welch's two-sample t-test (allows unequal variances)
  test_result <- t.test(
    outcomes[treatment],
    outcomes[!treatment],
    var.equal = FALSE,  # Neyman heteroskedasticity-robust
    conf.level = 1 - alpha
  )

  # Extract results
  estimate <- as.numeric(test_result$estimate[1] - test_result$estimate[2])
  se <- as.numeric(test_result$stderr)
  ci_lower <- as.numeric(test_result$conf.int[1])
  ci_upper <- as.numeric(test_result$conf.int[2])

  return(list(
    estimate = estimate,
    se = se,
    ci_lower = ci_lower,
    ci_upper = ci_upper
  ))
}


# ==============================================================================
# 2. StratifiedATE: Stratified Randomization
# ==============================================================================

#' Stratified Average Treatment Effect Estimator
#'
#' Estimates ATE accounting for stratified randomization.
#' Uses precision-weighted average across strata.
#'
#' @param outcomes Numeric vector of outcomes
#' @param treatment Logical or 0/1 vector of treatment assignment
#' @param strata Integer vector of stratum assignments
#' @param alpha Significance level (default 0.05)
#'
#' @return List with estimate, se, ci_lower, ci_upper
stratified_ate_r <- function(outcomes, treatment, strata, alpha = 0.05) {
  treatment <- as.logical(treatment)
  strata <- as.integer(strata)

  unique_strata <- sort(unique(strata))
  n_strata <- length(unique_strata)

  # Storage for stratum-specific estimates
  stratum_ates <- numeric(n_strata)
  stratum_vars <- numeric(n_strata)
  stratum_weights <- numeric(n_strata)

  # Compute ATE and variance for each stratum
  for (i in seq_along(unique_strata)) {
    s <- unique_strata[i]
    mask <- strata == s

    y_s <- outcomes[mask]
    t_s <- treatment[mask]

    # Stratum-specific means
    y1_s <- y_s[t_s]
    y0_s <- y_s[!t_s]

    n1_s <- length(y1_s)
    n0_s <- length(y0_s)

    # Stratum ATE
    stratum_ates[i] <- mean(y1_s) - mean(y0_s)

    # Neyman variance for this stratum
    var1_s <- var(y1_s) / n1_s
    var0_s <- var(y0_s) / n0_s
    stratum_vars[i] <- var1_s + var0_s

    # Weight by stratum size
    stratum_weights[i] <- sum(mask)
  }

  # Normalize weights
  stratum_weights <- stratum_weights / sum(stratum_weights)

  # Weighted average ATE
  estimate <- sum(stratum_weights * stratum_ates)

  # Variance of weighted average
  variance <- sum(stratum_weights^2 * stratum_vars)
  se <- sqrt(variance)

  # Confidence interval (normal approximation)
  z_crit <- qnorm(1 - alpha / 2)
  ci_lower <- estimate - z_crit * se
  ci_upper <- estimate + z_crit * se

  return(list(
    estimate = estimate,
    se = se,
    ci_lower = ci_lower,
    ci_upper = ci_upper
  ))
}


# ==============================================================================
# 3. RegressionATE: ANCOVA with HC3 Robust SE
# ==============================================================================

#' Regression-Adjusted ATE Estimator
#'
#' Uses OLS regression with covariates and HC3 robust standard errors.
#' HC3 recommended for n < 250 (Long & Ervin 2000).
#'
#' @param outcomes Numeric vector of outcomes
#' @param treatment Logical or 0/1 vector of treatment assignment
#' @param covariates Matrix of covariates (n x p)
#' @param alpha Significance level (default 0.05)
#'
#' @return List with estimate, se, ci_lower, ci_upper
#'
#' @references
#' Long, J. S., & Ervin, L. H. (2000). Using heteroscedasticity consistent
#' standard errors in the linear regression model. The American Statistician, 54(3), 217-224.
regression_ate_r <- function(outcomes, treatment, covariates, alpha = 0.05) {
  treatment <- as.numeric(as.logical(treatment))

  # Prepare data frame
  if (is.vector(covariates)) {
    covariates <- matrix(covariates, ncol = 1)
  }
  colnames(covariates) <- paste0("X", seq_len(ncol(covariates)))

  data <- data.frame(Y = outcomes, T = treatment, covariates)

  # Fit OLS model: Y ~ T + X
  formula_str <- paste("Y ~ T +", paste(colnames(covariates), collapse = " + "))
  model <- lm(as.formula(formula_str), data = data)

  # Extract treatment coefficient
  estimate <- coef(model)["T"]

  # HC3 robust standard errors (recommended for small samples)
  vcov_hc3 <- vcovHC(model, type = "HC3")
  se <- sqrt(vcov_hc3["T", "T"])

  # Confidence interval
  t_crit <- qt(1 - alpha / 2, df = model$df.residual)
  ci_lower <- estimate - t_crit * se
  ci_upper <- estimate + t_crit * se

  return(list(
    estimate = as.numeric(estimate),
    se = as.numeric(se),
    ci_lower = as.numeric(ci_lower),
    ci_upper = as.numeric(ci_upper)
  ))
}


# ==============================================================================
# 4. PermutationTest: Exact Permutation Test
# ==============================================================================

#' Permutation Test for Treatment Effect
#'
#' Fisher exact test via Monte Carlo permutations.
#' Uses coin package for robust implementation.
#'
#' @param outcomes Numeric vector of outcomes
#' @param treatment Logical or 0/1 vector of treatment assignment
#' @param n_permutations Number of permutations (default 1000)
#' @param random_seed Random seed for reproducibility
#' @param alpha Significance level (default 0.05)
#'
#' @return List with observed_statistic, p_value, permutation_distribution
#'
#' @references
#' Fisher, R. A. (1935). The Design of Experiments.
#' Rosenbaum, P. R. (2002). Observational Studies (Chapter 2).
permutation_test_r <- function(outcomes, treatment, n_permutations = 1000,
                                random_seed = NULL, alpha = 0.05) {
  if (!is.null(random_seed)) {
    set.seed(random_seed)
  }

  treatment <- as.logical(treatment)

  # Observed test statistic (difference in means)
  observed_stat <- mean(outcomes[treatment]) - mean(outcomes[!treatment])

  # Prepare data for coin package
  data <- data.frame(
    outcome = outcomes,
    group = factor(treatment, levels = c(FALSE, TRUE))
  )

  # Independence test via Monte Carlo permutations
  perm_test <- independence_test(
    outcome ~ group,
    data = data,
    distribution = approximate(nresample = n_permutations),
    alternative = "two.sided"
  )

  # Extract p-value
  p_value <- pvalue(perm_test)

  # Generate permutation distribution manually for compatibility
  # (coin package doesn't expose raw permutation statistics easily)
  permutation_distribution <- numeric(n_permutations)
  n <- length(outcomes)
  n1 <- sum(treatment)

  for (i in seq_len(n_permutations)) {
    # Random permutation of treatment labels
    perm_treatment <- sample(treatment)
    permutation_distribution[i] <- mean(outcomes[perm_treatment]) -
                                    mean(outcomes[!perm_treatment])
  }

  return(list(
    observed_statistic = observed_stat,
    p_value = as.numeric(p_value),
    permutation_distribution = permutation_distribution
  ))
}


# ==============================================================================
# 5. IPWATE: Inverse Probability Weighting
# ==============================================================================

#' Inverse Probability Weighted ATE Estimator
#'
#' Uses known or estimated propensity scores to weight observations.
#' Horvitz-Thompson estimator with robust variance.
#'
#' @param outcomes Numeric vector of outcomes
#' @param treatment Logical or 0/1 vector of treatment assignment
#' @param propensity Numeric vector of propensity scores (P(T=1|X))
#' @param alpha Significance level (default 0.05)
#'
#' @return List with estimate, se, ci_lower, ci_upper
#'
#' @references
#' Horvitz, D. G., & Thompson, D. J. (1952). A generalization of sampling without
#' replacement from a finite universe. Journal of the American Statistical Association, 47(260), 663-685.
ipw_ate_r <- function(outcomes, treatment, propensity, alpha = 0.05) {
  treatment <- as.logical(treatment)

  # IPW weights
  weights <- ifelse(treatment, 1 / propensity, 1 / (1 - propensity))

  # Weighted means
  weighted_outcomes_1 <- outcomes * treatment * weights
  weighted_outcomes_0 <- outcomes * (!treatment) * weights

  sum_weights_1 <- sum(weights[treatment])
  sum_weights_0 <- sum(weights[!treatment])

  mu1_hat <- sum(weighted_outcomes_1) / sum_weights_1
  mu0_hat <- sum(weighted_outcomes_0) / sum_weights_0

  estimate <- mu1_hat - mu0_hat

  # Horvitz-Thompson robust variance
  # Var(μ̂) = Σ(w_i² * (Y_i - μ̂)²) / (Σw_i)²
  var1 <- sum(weights[treatment]^2 * (outcomes[treatment] - mu1_hat)^2) / sum_weights_1^2
  var0 <- sum(weights[!treatment]^2 * (outcomes[!treatment] - mu0_hat)^2) / sum_weights_0^2

  variance <- var1 + var0
  se <- sqrt(variance)

  # Confidence interval (normal approximation)
  z_crit <- qnorm(1 - alpha / 2)
  ci_lower <- estimate - z_crit * se
  ci_upper <- estimate + z_crit * se

  return(list(
    estimate = estimate,
    se = se,
    ci_lower = ci_lower,
    ci_upper = ci_upper
  ))
}


# ==============================================================================
# Validation Test Cases
# ==============================================================================

#' Run validation test cases and write results to CSV
#'
#' @param output_file Path to output CSV file
run_validation <- function(output_file = "r_validation_results.csv") {
  results <- data.frame(
    estimator = character(),
    test_case = character(),
    estimate = numeric(),
    se = numeric(),
    ci_lower = numeric(),
    ci_upper = numeric(),
    stringsAsFactors = FALSE
  )

  set.seed(42)  # For reproducibility

  # -------------------------------------------------------------------------
  # Test Case 1: SimpleATE with balanced design
  # -------------------------------------------------------------------------
  outcomes_1 <- c(10, 12, 11, 4, 5, 3)
  treatment_1 <- c(TRUE, TRUE, TRUE, FALSE, FALSE, FALSE)

  result_simple_1 <- simple_ate_r(outcomes_1, treatment_1)
  results <- rbind(results, data.frame(
    estimator = "SimpleATE",
    test_case = "balanced_design",
    estimate = result_simple_1$estimate,
    se = result_simple_1$se,
    ci_lower = result_simple_1$ci_lower,
    ci_upper = result_simple_1$ci_upper
  ))

  # -------------------------------------------------------------------------
  # Test Case 2: StratifiedATE
  # -------------------------------------------------------------------------
  n <- 100
  strata_2 <- rep(1:5, each = 20)
  treatment_2 <- sample(rep(c(TRUE, FALSE), each = 50))
  outcomes_2 <- rnorm(n) + 2 * treatment_2 + strata_2 * 0.5

  result_strat_2 <- stratified_ate_r(outcomes_2, treatment_2, strata_2)
  results <- rbind(results, data.frame(
    estimator = "StratifiedATE",
    test_case = "five_strata",
    estimate = result_strat_2$estimate,
    se = result_strat_2$se,
    ci_lower = result_strat_2$ci_lower,
    ci_upper = result_strat_2$ci_upper
  ))

  # -------------------------------------------------------------------------
  # Test Case 3: RegressionATE
  # -------------------------------------------------------------------------
  n <- 100
  treatment_3 <- sample(rep(c(TRUE, FALSE), each = 50))
  covariate_3 <- rnorm(n)
  outcomes_3 <- 2 * treatment_3 + 0.5 * covariate_3 + rnorm(n)
  covariates_3 <- matrix(covariate_3, ncol = 1)

  result_reg_3 <- regression_ate_r(outcomes_3, treatment_3, covariates_3)
  results <- rbind(results, data.frame(
    estimator = "RegressionATE",
    test_case = "single_covariate",
    estimate = result_reg_3$estimate,
    se = result_reg_3$se,
    ci_lower = result_reg_3$ci_lower,
    ci_upper = result_reg_3$ci_upper
  ))

  # -------------------------------------------------------------------------
  # Test Case 4: PermutationTest
  # -------------------------------------------------------------------------
  outcomes_4 <- c(10, 12, 11, 4, 5, 3)
  treatment_4 <- c(TRUE, TRUE, TRUE, FALSE, FALSE, FALSE)

  result_perm_4 <- permutation_test_r(outcomes_4, treatment_4,
                                       n_permutations = 1000, random_seed = 42)
  results <- rbind(results, data.frame(
    estimator = "PermutationTest",
    test_case = "small_sample",
    estimate = result_perm_4$observed_statistic,
    se = NA,  # Permutation test doesn't provide SE in traditional sense
    ci_lower = NA,
    ci_upper = NA
  ))

  # -------------------------------------------------------------------------
  # Test Case 5: IPWATE
  # -------------------------------------------------------------------------
  n <- 100
  propensity_5 <- runif(n, 0.3, 0.7)
  treatment_5 <- as.logical(rbinom(n, 1, propensity_5))
  outcomes_5 <- rnorm(n) + 2 * treatment_5

  result_ipw_5 <- ipw_ate_r(outcomes_5, treatment_5, propensity_5)
  results <- rbind(results, data.frame(
    estimator = "IPWATE",
    test_case = "varying_propensity",
    estimate = result_ipw_5$estimate,
    se = result_ipw_5$se,
    ci_lower = result_ipw_5$ci_lower,
    ci_upper = result_ipw_5$ci_upper
  ))

  # Write results to CSV
  write.csv(results, output_file, row.names = FALSE)
  cat("Validation results written to:", output_file, "\n")

  return(results)
}


# ==============================================================================
# Main Execution
# ==============================================================================

if (!interactive()) {
  # Get output file from command line args or use default
  args <- commandArgs(trailingOnly = TRUE)
  output_file <- if (length(args) > 0) args[1] else "r_validation_results.csv"

  # Run validation
  results <- run_validation(output_file)

  # Print summary
  cat("\n")
  cat("=" , rep("=", 78), "\n", sep = "")
  cat("R Validation Results Summary\n")
  cat("=" , rep("=", 78), "\n", sep = "")
  print(results)
}
