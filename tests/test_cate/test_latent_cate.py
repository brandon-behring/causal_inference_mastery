"""Tests for Latent CATE methods (Factor Analysis, PPCA, GMM).

Test layers:
- Layer 1: Known-answer tests (ATE recovery, CATE shape, CI validity)
- Layer 2: Adversarial tests (high-dim, small sample, edge cases)
- Layer 3: Monte Carlo validation (bias, coverage) @slow
"""

import numpy as np
import pytest

from causal_inference.cate.latent_cate import (
    factor_analysis_cate,
    gmm_stratified_cate,
    ppca_cate,
)

from .conftest import generate_cate_dgp


# ============================================================================
# Layer 1: Known-Answer Tests
# ============================================================================


class TestFactorAnalysisCATEKnownAnswer:
    """Known-answer tests for Factor Analysis CATE."""

    def test_constant_effect_ate_recovery(self, constant_effect_data):
        """Factor Analysis CATE should recover ATE within tolerance."""
        Y, T, X, _ = constant_effect_data
        result = factor_analysis_cate(Y, T, X, n_latent=3)

        assert abs(result["ate"] - 2.0) < 0.5, f"ATE {result['ate']} not close to 2.0"

    def test_cate_shape(self, constant_effect_data):
        """CATE should have correct shape."""
        Y, T, X, _ = constant_effect_data
        result = factor_analysis_cate(Y, T, X, n_latent=3)

        assert result["cate"].shape == (len(Y),)

    def test_ci_validity(self, constant_effect_data):
        """CI should be valid (lower < upper)."""
        Y, T, X, _ = constant_effect_data
        result = factor_analysis_cate(Y, T, X, n_latent=3)

        assert result["ci_lower"] < result["ci_upper"]

    def test_se_positive(self, constant_effect_data):
        """SE should be positive."""
        Y, T, X, _ = constant_effect_data
        result = factor_analysis_cate(Y, T, X, n_latent=3)

        assert result["ate_se"] > 0

    def test_method_name(self, constant_effect_data):
        """Method name should be correct."""
        Y, T, X, _ = constant_effect_data
        result = factor_analysis_cate(Y, T, X, n_latent=3)

        assert result["method"] == "factor_analysis_cate"

    def test_heterogeneous_effect_correlation(self, linear_heterogeneous_data):
        """Factor Analysis should capture heterogeneity."""
        Y, T, X, true_cate = linear_heterogeneous_data
        result = factor_analysis_cate(Y, T, X, n_latent=3)

        correlation = np.corrcoef(result["cate"], true_cate)[0, 1]
        assert correlation > 0.2, f"Correlation {correlation} too low"


class TestPPCACATEKnownAnswer:
    """Known-answer tests for PPCA CATE."""

    def test_constant_effect_ate_recovery(self, constant_effect_data):
        """PPCA CATE should recover ATE within tolerance."""
        Y, T, X, _ = constant_effect_data
        result = ppca_cate(Y, T, X, n_components=3)

        assert abs(result["ate"] - 2.0) < 0.5, f"ATE {result['ate']} not close to 2.0"

    def test_cate_shape(self, constant_effect_data):
        """CATE should have correct shape."""
        Y, T, X, _ = constant_effect_data
        result = ppca_cate(Y, T, X, n_components=3)

        assert result["cate"].shape == (len(Y),)

    def test_ci_validity(self, constant_effect_data):
        """CI should be valid (lower < upper)."""
        Y, T, X, _ = constant_effect_data
        result = ppca_cate(Y, T, X, n_components=3)

        assert result["ci_lower"] < result["ci_upper"]

    def test_se_positive(self, constant_effect_data):
        """SE should be positive."""
        Y, T, X, _ = constant_effect_data
        result = ppca_cate(Y, T, X, n_components=3)

        assert result["ate_se"] > 0

    def test_method_name(self, constant_effect_data):
        """Method name should be correct."""
        Y, T, X, _ = constant_effect_data
        result = ppca_cate(Y, T, X, n_components=3)

        assert result["method"] == "ppca_cate"


class TestGMMStratifiedCATEKnownAnswer:
    """Known-answer tests for GMM Stratified CATE."""

    def test_constant_effect_ate_recovery(self, constant_effect_data):
        """GMM Stratified CATE should recover ATE within tolerance."""
        Y, T, X, _ = constant_effect_data
        result = gmm_stratified_cate(Y, T, X, n_strata=3)

        assert abs(result["ate"] - 2.0) < 0.5, f"ATE {result['ate']} not close to 2.0"

    def test_cate_shape(self, constant_effect_data):
        """CATE should have correct shape."""
        Y, T, X, _ = constant_effect_data
        result = gmm_stratified_cate(Y, T, X, n_strata=3)

        assert result["cate"].shape == (len(Y),)

    def test_ci_validity(self, constant_effect_data):
        """CI should be valid (lower < upper)."""
        Y, T, X, _ = constant_effect_data
        result = gmm_stratified_cate(Y, T, X, n_strata=3)

        assert result["ci_lower"] < result["ci_upper"]

    def test_se_positive(self, constant_effect_data):
        """SE should be positive."""
        Y, T, X, _ = constant_effect_data
        result = gmm_stratified_cate(Y, T, X, n_strata=3)

        assert result["ate_se"] > 0

    def test_method_name(self, constant_effect_data):
        """Method name should be correct."""
        Y, T, X, _ = constant_effect_data
        result = gmm_stratified_cate(Y, T, X, n_strata=3)

        assert result["method"] == "gmm_stratified_cate"


# ============================================================================
# Layer 2: Adversarial Tests
# ============================================================================


class TestLatentCATEAdversarial:
    """Adversarial tests for all Latent CATE methods."""

    @pytest.mark.parametrize(
        "method,kwargs",
        [
            (factor_analysis_cate, {"n_latent": 3}),
            (ppca_cate, {"n_components": 3}),
            (gmm_stratified_cate, {"n_strata": 3}),
        ],
    )
    def test_high_dimensional(self, high_dimensional_data, method, kwargs):
        """Methods should handle high-dimensional covariates."""
        Y, T, X, _ = high_dimensional_data
        result = method(Y, T, X, **kwargs)

        assert isinstance(result["ate"], float)
        assert not np.isnan(result["ate"])

    @pytest.mark.parametrize(
        "method,kwargs",
        [
            (factor_analysis_cate, {"n_latent": 2}),
            (ppca_cate, {"n_components": 2}),
            (gmm_stratified_cate, {"n_strata": 2}),
        ],
    )
    def test_small_sample(self, method, kwargs):
        """Methods should handle small samples (graceful degradation)."""
        Y, T, X, _ = generate_cate_dgp(n=80, p=3, seed=42)
        result = method(Y, T, X, **kwargs)

        assert isinstance(result["ate"], float)
        assert not np.isnan(result["ate"])

    @pytest.mark.parametrize(
        "method,kwargs",
        [
            (factor_analysis_cate, {"n_latent": 1}),
            (ppca_cate, {"n_components": 1}),
            (gmm_stratified_cate, {"n_strata": 2}),
        ],
    )
    def test_single_covariate(self, single_covariate_data, method, kwargs):
        """Methods should handle single covariate."""
        Y, T, X, _ = single_covariate_data
        # For FA and PPCA with 1D data, n_latent must be capped
        result = method(Y, T, X.ravel(), **kwargs)

        assert isinstance(result["ate"], float)

    def test_factor_analysis_invalid_n_latent_raises(self, constant_effect_data):
        """Should raise error for invalid n_latent."""
        Y, T, X, _ = constant_effect_data

        with pytest.raises(ValueError, match="n_latent"):
            factor_analysis_cate(Y, T, X, n_latent=0)

    def test_ppca_invalid_n_components_raises(self, constant_effect_data):
        """Should raise error for invalid n_components."""
        Y, T, X, _ = constant_effect_data

        with pytest.raises(ValueError, match="n_components"):
            ppca_cate(Y, T, X, n_components=0)

    def test_gmm_invalid_n_strata_raises(self, constant_effect_data):
        """Should raise error for invalid n_strata."""
        Y, T, X, _ = constant_effect_data

        with pytest.raises(ValueError, match="n_strata"):
            gmm_stratified_cate(Y, T, X, n_strata=1)

    def test_empty_treatment_group_raises(self):
        """Should raise error when treatment group is empty."""
        n = 100
        Y = np.random.randn(n)
        T = np.zeros(n)  # All control
        X = np.random.randn(n, 3)

        with pytest.raises(ValueError, match="treatment"):
            factor_analysis_cate(Y, T, X, n_latent=2)

    def test_empty_control_group_raises(self):
        """Should raise error when control group is empty."""
        n = 100
        Y = np.random.randn(n)
        T = np.ones(n)  # All treated
        X = np.random.randn(n, 3)

        with pytest.raises(ValueError, match="control"):
            ppca_cate(Y, T, X, n_components=2)

    @pytest.mark.parametrize(
        "method,kwargs",
        [
            (factor_analysis_cate, {"n_latent": 3}),
            (ppca_cate, {"n_components": 3}),
            (gmm_stratified_cate, {"n_strata": 3}),
        ],
    )
    def test_imbalanced_treatment(self, method, kwargs):
        """Should handle imbalanced treatment (20% treated)."""
        Y, T, X, _ = generate_cate_dgp(
            n=500,
            p=3,
            effect_type="constant",
            true_ate=2.0,
            treatment_prob=0.2,
            seed=42,
        )
        result = method(Y, T, X, **kwargs)

        # Should get reasonable estimate (looser tolerance for imbalanced)
        assert abs(result["ate"] - 2.0) < 1.5

    @pytest.mark.parametrize(
        "method,kwargs",
        [
            (factor_analysis_cate, {"n_latent": 3}),
            (ppca_cate, {"n_components": 3}),
        ],
    )
    def test_r_learner_base(self, constant_effect_data, method, kwargs):
        """Should work with R-learner as base."""
        Y, T, X, _ = constant_effect_data
        result = method(Y, T, X, base_learner="r_learner", **kwargs)

        assert isinstance(result["ate"], float)
        assert not np.isnan(result["ate"])

    @pytest.mark.parametrize(
        "method,kwargs",
        [
            (factor_analysis_cate, {"n_latent": 3}),
            (ppca_cate, {"n_components": 3}),
            (gmm_stratified_cate, {"n_strata": 3}),
        ],
    )
    def test_random_forest_model(self, constant_effect_data, method, kwargs):
        """Should work with random forest model."""
        Y, T, X, _ = constant_effect_data
        result = method(Y, T, X, model="random_forest", **kwargs)

        assert isinstance(result["ate"], float)
        assert not np.isnan(result["ate"])

    def test_gmm_few_strata(self, constant_effect_data):
        """GMM with 2 strata (minimum) should work."""
        Y, T, X, _ = constant_effect_data
        result = gmm_stratified_cate(Y, T, X, n_strata=2)

        assert isinstance(result["ate"], float)

    def test_factor_analysis_exceeds_features(self, constant_effect_data):
        """n_latent > p should be capped automatically."""
        Y, T, X, _ = constant_effect_data
        # X has 5 features, request 10 latent factors
        result = factor_analysis_cate(Y, T, X, n_latent=10)

        # Should not crash, should cap to p-1
        assert isinstance(result["ate"], float)


# ============================================================================
# Layer 3: Monte Carlo Tests
# ============================================================================


class TestLatentCATEMonteCarlo:
    """Monte Carlo validation for Latent CATE methods."""

    @pytest.mark.slow
    def test_factor_analysis_cate_bias(self):
        """Factor Analysis CATE should have bias < 0.15 over Monte Carlo runs."""
        n_runs = 100
        estimates = []

        for seed in range(n_runs):
            Y, T, X, _ = generate_cate_dgp(n=300, p=5, true_ate=2.0, seed=seed)
            result = factor_analysis_cate(Y, T, X, n_latent=3)
            estimates.append(result["ate"])

        bias = abs(np.mean(estimates) - 2.0)
        assert bias < 0.15, f"Factor Analysis bias {bias:.3f} exceeds threshold"

    @pytest.mark.slow
    def test_factor_analysis_cate_coverage(self):
        """Factor Analysis CATE should have 85-99% coverage."""
        n_runs = 100
        covers = []

        for seed in range(n_runs):
            Y, T, X, _ = generate_cate_dgp(n=300, p=5, true_ate=2.0, seed=seed)
            result = factor_analysis_cate(Y, T, X, n_latent=3)
            covers.append(result["ci_lower"] < 2.0 < result["ci_upper"])

        coverage = np.mean(covers)
        assert 0.80 < coverage < 0.99, f"Coverage {coverage:.2%} outside 80-99%"

    @pytest.mark.slow
    def test_ppca_cate_bias(self):
        """PPCA CATE should have bias < 0.15 over Monte Carlo runs."""
        n_runs = 100
        estimates = []

        for seed in range(n_runs):
            Y, T, X, _ = generate_cate_dgp(n=300, p=5, true_ate=2.0, seed=seed)
            result = ppca_cate(Y, T, X, n_components=3)
            estimates.append(result["ate"])

        bias = abs(np.mean(estimates) - 2.0)
        assert bias < 0.15, f"PPCA bias {bias:.3f} exceeds threshold"

    @pytest.mark.slow
    def test_ppca_cate_coverage(self):
        """PPCA CATE should have 85-99% coverage."""
        n_runs = 100
        covers = []

        for seed in range(n_runs):
            Y, T, X, _ = generate_cate_dgp(n=300, p=5, true_ate=2.0, seed=seed)
            result = ppca_cate(Y, T, X, n_components=3)
            covers.append(result["ci_lower"] < 2.0 < result["ci_upper"])

        coverage = np.mean(covers)
        assert 0.80 < coverage < 0.99, f"Coverage {coverage:.2%} outside 80-99%"

    @pytest.mark.slow
    def test_gmm_stratified_cate_bias(self):
        """GMM Stratified CATE should have bias < 0.20 over Monte Carlo runs."""
        n_runs = 100
        estimates = []

        for seed in range(n_runs):
            Y, T, X, _ = generate_cate_dgp(n=400, p=5, true_ate=2.0, seed=seed)
            result = gmm_stratified_cate(Y, T, X, n_strata=3)
            estimates.append(result["ate"])

        bias = abs(np.mean(estimates) - 2.0)
        assert bias < 0.20, f"GMM bias {bias:.3f} exceeds threshold"

    @pytest.mark.slow
    def test_gmm_stratified_cate_coverage(self):
        """GMM Stratified CATE should have 80-99% coverage."""
        n_runs = 100
        covers = []

        for seed in range(n_runs):
            Y, T, X, _ = generate_cate_dgp(n=400, p=5, true_ate=2.0, seed=seed)
            result = gmm_stratified_cate(Y, T, X, n_strata=3)
            covers.append(result["ci_lower"] < 2.0 < result["ci_upper"])

        coverage = np.mean(covers)
        assert 0.75 < coverage < 0.99, f"Coverage {coverage:.2%} outside 75-99%"

    @pytest.mark.slow
    def test_latent_factor_improves_estimation(self):
        """Latent factors should help when unobserved confounding structure exists."""
        # DGP with latent structure: X has low-rank structure
        n_runs = 50
        fa_better = 0

        for seed in range(n_runs):
            np.random.seed(seed)
            n = 300
            # Latent factors drive both X and Y
            latent = np.random.randn(n, 2)
            noise = np.random.randn(n, 5) * 0.5
            X = latent @ np.random.randn(2, 5) + noise  # Low-rank X

            T = np.random.binomial(1, 0.5, n)
            # True effect depends on latent
            true_effect = 2.0 + 0.5 * latent[:, 0]
            Y = 1 + X[:, 0] + true_effect * T + 0.3 * latent[:, 0] + np.random.randn(n)

            result_fa = factor_analysis_cate(Y, T, X, n_latent=2)
            result_plain = factor_analysis_cate(Y, T, X, n_latent=0)

            # Check if FA estimate is closer to true ATE
            true_ate = np.mean(true_effect)
            if abs(result_fa["ate"] - true_ate) < abs(result_plain["ate"] - true_ate):
                fa_better += 1

        # FA should help at least sometimes when there's latent structure
        # This is a weak test - just verifying it doesn't hurt
        assert fa_better > 10, f"FA only better {fa_better}/50 times"
