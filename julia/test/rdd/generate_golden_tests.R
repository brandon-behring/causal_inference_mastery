#!/usr/bin/env Rscript
# Golden Test Generation for Phase 3 RDD
#
# Generates 15 reference test cases using R rdrobust (gold standard)
# These tests must pass BEFORE Julia implementation is considered correct
#
# Requirements: R packages rdrobust, rddensity
# Usage: Rscript generate_golden_tests.R

library(rdrobust)
library(rddensity)

set.seed(42)  # Reproducibility

# Helper: Generate RDD data
generate_rdd_data <- function(n, tau, dgp_type = "linear", sd_noise = 1.0,
                              p_covariates = 0, cutoff = 0.0) {
  # Running variable
  x <- runif(n, -2, 2)

  # Treatment assignment (sharp RDD)
  treatment <- as.numeric(x >= cutoff)

  # Outcome generation based on DGP type
  if (dgp_type == "linear") {
    # y = α + β*x + τ*D + ε
    y <- 2 + 3*x + tau*treatment + rnorm(n, 0, sd_noise)
  } else if (dgp_type == "quadratic") {
    # y = α + β₁*x + β₂*x² + τ*D + ε
    y <- 2 + 3*x + 0.5*x^2 + tau*treatment + rnorm(n, 0, sd_noise)
  } else if (dgp_type == "disc_slope") {
    # Discontinuous slope: different β left/right of cutoff
    beta_left <- 2.0
    beta_right <- 4.0
    y <- 2 + ifelse(x < cutoff, beta_left*x, beta_right*x) + tau*treatment + rnorm(n, 0, sd_noise)
  }

  # Covariates (if requested)
  covariates <- NULL
  if (p_covariates > 0) {
    covariates <- matrix(rnorm(n * p_covariates), nrow = n, ncol = p_covariates)
  }

  list(
    x = x,
    y = y,
    treatment = treatment,
    covariates = covariates,
    cutoff = cutoff,
    true_ate = tau,
    dgp_type = dgp_type,
    n = n
  )
}

# Test case specifications
test_cases <- list(
  # 1. Baseline: Medium sample, linear DGP, no covariates, moderate effect
  list(name = "baseline_linear_n1000",
       n = 1000, tau = 5.0, dgp_type = "linear", p_covariates = 0),

  # 2. Small sample: n=200
  list(name = "small_sample_n200",
       n = 200, tau = 3.0, dgp_type = "linear", p_covariates = 0),

  # 3. Large sample: n=5000
  list(name = "large_sample_n5000",
       n = 5000, tau = 4.0, dgp_type = "linear", p_covariates = 0),

  # 4. Zero effect (H0: τ = 0)
  list(name = "null_effect_tau0",
       n = 1000, tau = 0.0, dgp_type = "linear", p_covariates = 0),

  # 5. Negative effect
  list(name = "negative_effect_tau_neg3",
       n = 1000, tau = -3.0, dgp_type = "linear", p_covariates = 0),

  # 6. Quadratic DGP (functional form test)
  list(name = "quadratic_dgp_n1000",
       n = 1000, tau = 5.0, dgp_type = "quadratic", p_covariates = 0),

  # 7. Discontinuous slope (different trends left/right)
  list(name = "disc_slope_n1000",
       n = 1000, tau = 2.0, dgp_type = "disc_slope", p_covariates = 0),

  # 8. With covariates: 3 baseline variables
  list(name = "with_covariates_p3",
       n = 1000, tau = 5.0, dgp_type = "linear", p_covariates = 3),

  # 9. With many covariates: 10 baseline variables
  list(name = "with_covariates_p10",
       n = 1000, tau = 5.0, dgp_type = "linear", p_covariates = 10),

  # 10. High noise (low signal-to-noise)
  list(name = "high_noise_sd5",
       n = 1000, tau = 5.0, dgp_type = "linear", p_covariates = 0, sd_noise = 5.0),

  # 11. Low noise (high signal-to-noise)
  list(name = "low_noise_sd0p5",
       n = 1000, tau = 5.0, dgp_type = "linear", p_covariates = 0, sd_noise = 0.5),

  # 12. Non-zero cutoff: c = 0.5
  list(name = "nonzero_cutoff_c0p5",
       n = 1000, tau = 5.0, dgp_type = "linear", p_covariates = 0, cutoff = 0.5),

  # 13. Negative cutoff: c = -0.5
  list(name = "negative_cutoff_cneg0p5",
       n = 1000, tau = 5.0, dgp_type = "linear", p_covariates = 0, cutoff = -0.5),

  # 14. Large effect (τ = 10)
  list(name = "large_effect_tau10",
       n = 1000, tau = 10.0, dgp_type = "linear", p_covariates = 0),

  # 15. Small effect (τ = 0.5, power test)
  list(name = "small_effect_tau0p5",
       n = 2000, tau = 0.5, dgp_type = "linear", p_covariates = 0)
)

# Default sd_noise if not specified
for (i in seq_along(test_cases)) {
  if (is.null(test_cases[[i]]$sd_noise)) {
    test_cases[[i]]$sd_noise <- 1.0
  }
  if (is.null(test_cases[[i]]$cutoff)) {
    test_cases[[i]]$cutoff <- 0.0
  }
}

# Generate golden test outputs
cat("Generating 15 golden test cases from R rdrobust...\n")

for (i in seq_along(test_cases)) {
  tc <- test_cases[[i]]
  cat(sprintf("Test %2d: %s\n", i, tc$name))

  # Generate data
  data <- generate_rdd_data(
    n = tc$n,
    tau = tc$tau,
    dgp_type = tc$dgp_type,
    sd_noise = tc$sd_noise,
    p_covariates = tc$p_covariates,
    cutoff = tc$cutoff
  )

  # Run rdrobust (CCT default bandwidth + inference)
  # bwselect="msecomb2" is CCT coverage-error-optimal (default)
  rd_result <- rdrobust(
    y = data$y,
    x = data$x,
    c = data$cutoff,
    kernel = "triangular",  # Default kernel
    bwselect = "msecomb2",  # CCT bandwidth (default)
    all = TRUE              # Include IK bandwidth too
  )

  # Run McCrary density test
  density_result <- rddensity(X = data$x, c = data$cutoff)

  # Store results
  golden_output <- list(
    # Test metadata
    test_name = tc$name,
    test_number = i,
    generated_date = Sys.time(),

    # Data
    x = data$x,
    y = data$y,
    treatment = data$treatment,
    covariates = data$covariates,
    cutoff = data$cutoff,
    true_ate = data$true_ate,
    dgp_type = data$dgp_type,
    n = data$n,

    # rdrobust outputs (CCT default)
    estimate_cct = rd_result$Estimate[1],       # Conventional estimate
    se_cct = rd_result$se[1],                   # Conventional SE
    ci_lower_cct = rd_result$ci[1,1],           # Lower CI
    ci_upper_cct = rd_result$ci[1,2],           # Upper CI
    estimate_bc = rd_result$Estimate[2],        # Bias-corrected estimate
    se_bc = rd_result$se[2],                    # Robust SE
    ci_lower_bc = rd_result$ci[2,1],            # Bias-corrected lower CI
    ci_upper_bc = rd_result$ci[2,2],            # Bias-corrected upper CI
    pvalue = rd_result$pv[3],                   # Robust p-value

    # Bandwidth info
    bandwidth_cct = rd_result$bws[1,1],         # CCT main bandwidth
    bandwidth_cct_bias = rd_result$bws[1,2],    # CCT bias-correction bandwidth
    bandwidth_ik = rd_result$bws[2,1],          # IK bandwidth

    # Sample info
    n_left = rd_result$N[1],                    # Obs left of cutoff
    n_right = rd_result$N[2],                   # Obs right of cutoff
    n_eff_left = rd_result$N_h[1],              # Effective N left (within bandwidth)
    n_eff_right = rd_result$N_h[2],             # Effective N right

    # McCrary density test
    mccrary_pvalue = summary(density_result)$pv[1],
    mccrary_discontinuity = density_result$hat$diff,
    mccrary_se = density_result$hat$se_diff,

    # Full rdrobust object (for detailed validation if needed)
    rdrobust_full = rd_result,
    rddensity_full = density_result
  )

  # Save to RDS file
  output_file <- file.path("golden", sprintf("case_%02d_%s.rds", i, tc$name))
  saveRDS(golden_output, output_file)

  cat(sprintf("  Saved: %s\n", output_file))
  cat(sprintf("  ATE: %.4f (true: %.4f), CI: [%.4f, %.4f], p=%.4f\n",
              golden_output$estimate_bc, golden_output$true_ate,
              golden_output$ci_lower_bc, golden_output$ci_upper_bc,
              golden_output$pvalue))
  cat(sprintf("  Bandwidth (CCT): h=%.4f, b=%.4f (IK: h=%.4f)\n",
              golden_output$bandwidth_cct, golden_output$bandwidth_cct_bias,
              golden_output$bandwidth_ik))
  cat(sprintf("  McCrary p-value: %.4f (discontinuity: %.4f)\n\n",
              golden_output$mccrary_pvalue, golden_output$mccrary_discontinuity))
}

cat("✓ Golden test generation complete!\n")
cat("  Generated 15 test cases in test/rdd/golden/\n")
cat("  Julia implementation must match these results (rtol < 1e-8)\n")
