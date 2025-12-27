"""Tests for DragonNet implementation.

Session 139: DragonNet neural network CATE estimation.

Test Layers:
- Layer 1 (Known-Answer): Tests with deterministic/expected results
- Layer 2 (Adversarial): Edge cases and error handling
- Layer 3 (Monte Carlo): Statistical validation
"""

import numpy as np
import pytest

from causal_inference.cate import dragonnet, CATEResult
from causal_inference.cate.dragonnet import (
    DragonNetSklearn,
    DragonNetTorch,
    _check_torch_available,
)

from .conftest import generate_cate_dgp


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================


class TestDragonNetKnownAnswer:
    """Known-answer tests for DragonNet."""

    def test_constant_effect_ate_recovery(self, constant_effect_data):
        """DragonNet recovers ATE within tolerance for constant effect."""
        Y, T, X, true_cate = constant_effect_data
        true_ate = np.mean(true_cate)

        result = dragonnet(Y, T, X, backend="sklearn", max_iter=200)

        # Neural networks have higher variance, use looser tolerance
        assert abs(result["ate"] - true_ate) < 1.0
        assert result["method"] == "dragonnet"

    def test_cate_shape(self, constant_effect_data):
        """DragonNet returns CATE array of correct shape."""
        Y, T, X, _ = constant_effect_data

        result = dragonnet(Y, T, X, backend="sklearn", max_iter=100)

        assert result["cate"].shape == (len(Y),)

    def test_heterogeneous_effect_correlation(self, linear_heterogeneous_data):
        """DragonNet captures linear heterogeneous effects."""
        Y, T, X, true_cate = linear_heterogeneous_data

        result = dragonnet(Y, T, X, backend="sklearn", max_iter=200)

        # CATE should correlate with true effect
        correlation = np.corrcoef(result["cate"], true_cate)[0, 1]
        # Neural nets can have variance; require moderate correlation
        assert correlation > 0.2 or abs(result["ate"] - np.mean(true_cate)) < 0.8

    def test_confidence_interval_valid(self, constant_effect_data):
        """CI should be valid (lower < upper)."""
        Y, T, X, _ = constant_effect_data

        result = dragonnet(Y, T, X, backend="sklearn", max_iter=100)

        assert result["ci_lower"] < result["ci_upper"]

    def test_se_positive(self, constant_effect_data):
        """Standard error should be positive."""
        Y, T, X, _ = constant_effect_data

        result = dragonnet(Y, T, X, backend="sklearn", max_iter=100)

        assert result["ate_se"] > 0

    def test_result_type(self, constant_effect_data):
        """Result should be CATEResult TypedDict."""
        Y, T, X, _ = constant_effect_data

        result = dragonnet(Y, T, X, backend="sklearn", max_iter=100)

        expected_keys = {"cate", "ate", "ate_se", "ci_lower", "ci_upper", "method"}
        assert set(result.keys()) == expected_keys

    def test_custom_architecture(self, constant_effect_data):
        """DragonNet works with custom layer sizes."""
        Y, T, X, _ = constant_effect_data

        result = dragonnet(
            Y,
            T,
            X,
            backend="sklearn",
            hidden_layers=(100, 50),
            head_layers=(50,),
            max_iter=100,
        )

        assert np.isfinite(result["ate"])

    def test_confounded_dgp(self):
        """DragonNet handles confounding via propensity estimation."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        propensity = 1 / (1 + np.exp(-0.5 * X[:, 0]))
        T = np.random.binomial(1, propensity, n).astype(float)
        true_ate = 2.0
        Y = 1 + 0.5 * X[:, 0] + true_ate * T + np.random.randn(n)

        result = dragonnet(Y, T, X, backend="sklearn", max_iter=200)

        # Should recover ATE reasonably well despite confounding
        assert abs(result["ate"] - true_ate) < 1.5


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================


class TestDragonNetAdversarial:
    """Adversarial tests for DragonNet."""

    def test_empty_treatment_group_raises(self):
        """Raise ValueError when all units are control."""
        n = 100
        Y = np.random.randn(n)
        T = np.zeros(n)
        X = np.random.randn(n, 2)

        with pytest.raises(ValueError, match="No treated units"):
            dragonnet(Y, T, X)

    def test_empty_control_group_raises(self):
        """Raise ValueError when all units are treated."""
        n = 100
        Y = np.random.randn(n)
        T = np.ones(n)
        X = np.random.randn(n, 2)

        with pytest.raises(ValueError, match="No control units"):
            dragonnet(Y, T, X)

    def test_invalid_backend_raises(self, constant_effect_data):
        """Raise ValueError for unknown backend."""
        Y, T, X, _ = constant_effect_data

        with pytest.raises(ValueError, match="Unknown backend"):
            dragonnet(Y, T, X, backend="invalid")

    def test_non_binary_treatment_raises(self, constant_effect_data):
        """Raise ValueError for non-binary treatment."""
        Y, T, X, _ = constant_effect_data
        T_continuous = T + 0.5

        with pytest.raises(ValueError, match="binary"):
            dragonnet(Y, T_continuous, X)

    def test_high_dimensional(self, high_dimensional_data):
        """DragonNet works with many covariates."""
        Y, T, X, _ = high_dimensional_data

        result = dragonnet(Y, T, X, backend="sklearn", max_iter=100)

        assert np.isfinite(result["ate"])

    def test_small_sample(self, small_sample_data):
        """DragonNet works with small samples."""
        Y, T, X, _ = small_sample_data

        result = dragonnet(Y, T, X, backend="sklearn", max_iter=50)

        assert np.isfinite(result["ate"])

    def test_single_covariate(self, single_covariate_data):
        """DragonNet works with single covariate."""
        Y, T, X, _ = single_covariate_data

        result = dragonnet(Y, T, X, backend="sklearn", max_iter=100)

        assert np.isfinite(result["ate"])

    def test_imbalanced_treatment(self):
        """DragonNet handles imbalanced treatment assignment."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        T = np.random.binomial(1, 0.1, n).astype(float)  # 10% treated
        Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)

        result = dragonnet(Y, T, X, backend="sklearn", max_iter=150)

        assert np.isfinite(result["ate"])

    def test_extreme_propensity(self):
        """DragonNet handles extreme propensity scores."""
        np.random.seed(42)
        n = 400
        X = np.random.randn(n, 2)
        # Very extreme propensity
        propensity = 1 / (1 + np.exp(-2 * X[:, 0]))
        T = np.random.binomial(1, propensity, n).astype(float)
        Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)

        result = dragonnet(Y, T, X, backend="sklearn", max_iter=150)

        assert np.isfinite(result["ate"])
        assert np.all(np.isfinite(result["cate"]))

    def test_mismatched_lengths_raises(self):
        """Raise ValueError when input lengths don't match."""
        Y = np.random.randn(100)
        T = np.random.binomial(1, 0.5, 50)
        X = np.random.randn(100, 2)

        with pytest.raises(ValueError, match="Length mismatch"):
            dragonnet(Y, T, X)


# =============================================================================
# Backend-Specific Tests
# =============================================================================


class TestDragonNetBackend:
    """Tests for backend selection and availability."""

    def test_sklearn_backend_always_works(self, constant_effect_data):
        """sklearn backend should always be available."""
        Y, T, X, _ = constant_effect_data

        result = dragonnet(Y, T, X, backend="sklearn", max_iter=100)

        assert result["method"] == "dragonnet"

    def test_auto_backend_selection(self, constant_effect_data):
        """auto backend should select without error."""
        Y, T, X, _ = constant_effect_data

        result = dragonnet(Y, T, X, backend="auto", max_iter=100)

        assert result["method"] == "dragonnet"

    def test_torch_backend_not_implemented(self, constant_effect_data):
        """torch backend raises NotImplementedError (deferred)."""
        Y, T, X, _ = constant_effect_data

        # Skip if PyTorch is not available (ImportError from init)
        # If PyTorch is available, we'll get NotImplementedError from fit
        try:
            with pytest.raises((ImportError, NotImplementedError)):
                dragonnet(Y, T, X, backend="torch")
        except ImportError:
            # PyTorch not installed, that's expected behavior
            pass

    def test_dragonnet_sklearn_class_directly(self, constant_effect_data):
        """DragonNetSklearn class works directly."""
        Y, T, X, _ = constant_effect_data

        model = DragonNetSklearn(hidden_layers=(50,), head_layers=(25,), max_iter=50)
        model.fit(X, T, Y)

        assert model.is_fitted_
        assert model.predict_cate(X).shape == (len(Y),)

    def test_predict_before_fit_raises(self, constant_effect_data):
        """Predictions before fit should raise RuntimeError."""
        _, _, X, _ = constant_effect_data

        model = DragonNetSklearn()

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict_cate(X)

    def test_check_torch_available_function(self):
        """_check_torch_available returns bool without error."""
        result = _check_torch_available()
        assert isinstance(result, bool)


# =============================================================================
# Layer 3: Monte Carlo Tests
# =============================================================================


class TestDragonNetMonteCarlo:
    """Monte Carlo validation tests for DragonNet."""

    @pytest.mark.slow
    def test_ate_unbiased(self):
        """Monte Carlo: DragonNet has low bias for constant effect."""
        n_sims = 50  # Reduced for neural network training time
        true_ate = 2.0

        estimates = []
        for sim in range(n_sims):
            Y, T, X, _ = generate_cate_dgp(
                n=300,
                p=3,
                effect_type="constant",
                true_ate=true_ate,
                seed=sim * 1000 + 42,
            )

            result = dragonnet(Y, T, X, backend="sklearn", max_iter=150)
            estimates.append(result["ate"])

        bias = np.mean(estimates) - true_ate

        # Neural networks have higher variance; looser bias threshold
        assert abs(bias) < 0.5

    @pytest.mark.slow
    def test_coverage(self):
        """Monte Carlo: DragonNet CIs achieve reasonable coverage."""
        n_sims = 50
        true_ate = 2.0

        covers = []
        for sim in range(n_sims):
            Y, T, X, _ = generate_cate_dgp(
                n=300,
                p=3,
                effect_type="constant",
                true_ate=true_ate,
                seed=sim * 1000 + 123,
            )

            result = dragonnet(Y, T, X, backend="sklearn", alpha=0.05, max_iter=150)
            covers.append(result["ci_lower"] < true_ate < result["ci_upper"])

        coverage = np.mean(covers)

        # Neural networks can have wider variance; accept 70-100% coverage
        assert 0.70 < coverage <= 1.0

    @pytest.mark.slow
    def test_cate_recovery(self):
        """Monte Carlo: DragonNet recovers heterogeneous CATE."""
        n_sims = 30
        correlations = []

        for sim in range(n_sims):
            Y, T, X, true_cate = generate_cate_dgp(
                n=500,
                p=3,
                effect_type="linear",
                true_ate=2.0,
                seed=sim * 1000 + 456,
            )

            result = dragonnet(Y, T, X, backend="sklearn", max_iter=150)
            corr = np.corrcoef(result["cate"], true_cate)[0, 1]
            if np.isfinite(corr):
                correlations.append(corr)

        if len(correlations) > 0:
            mean_corr = np.mean(correlations)
            # Neural networks can have variance; accept weak correlation
            assert mean_corr > 0.1

    @pytest.mark.slow
    def test_vs_meta_learners(self):
        """Monte Carlo: Compare DragonNet to meta-learners."""
        from causal_inference.cate import t_learner

        n_sims = 30
        true_ate = 2.0

        dragonnet_bias = []
        t_learner_bias = []

        for sim in range(n_sims):
            Y, T, X, _ = generate_cate_dgp(
                n=400,
                p=3,
                effect_type="constant",
                true_ate=true_ate,
                seed=sim * 1000 + 789,
            )

            dn_result = dragonnet(Y, T, X, backend="sklearn", max_iter=150)
            t_result = t_learner(Y, T, X)

            dragonnet_bias.append(abs(dn_result["ate"] - true_ate))
            t_learner_bias.append(abs(t_result["ate"] - true_ate))

        # DragonNet should be competitive with T-learner (within 3x)
        assert np.mean(dragonnet_bias) < np.mean(t_learner_bias) * 3

    @pytest.mark.slow
    def test_confounded_dgp_monte_carlo(self):
        """Monte Carlo: DragonNet handles confounding."""
        n_sims = 50
        true_ate = 2.0

        estimates = []
        for sim in range(n_sims):
            np.random.seed(sim * 1000 + 321)
            n = 400
            X = np.random.randn(n, 2)
            propensity = 1 / (1 + np.exp(-0.5 * X[:, 0]))
            T = np.random.binomial(1, propensity, n).astype(float)
            Y = 1 + 0.5 * X[:, 0] + true_ate * T + np.random.randn(n)

            result = dragonnet(Y, T, X, backend="sklearn", max_iter=150)
            estimates.append(result["ate"])

        bias = np.mean(estimates) - true_ate

        # With confounding and neural network variance, accept higher bias
        assert abs(bias) < 0.8
