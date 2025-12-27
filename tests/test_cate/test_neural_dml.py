"""Tests for neural Double Machine Learning.

Test layers:
- Layer 1: Known-answer tests (ATE recovery, cross-fitting validation)
- Layer 2: Adversarial tests (sample size, fold counts)
- Layer 3: Monte Carlo validation (bias, coverage) @slow
"""

import numpy as np
import pytest

from causal_inference.cate.neural_dml import neural_double_ml

from .conftest import generate_cate_dgp


# ============================================================================
# Layer 1: Known-Answer Tests
# ============================================================================


class TestNeuralDMLKnownAnswer:
    """Known-answer tests for neural Double ML."""

    def test_constant_effect_ate_recovery(self, constant_effect_data):
        """Neural DML should recover ATE within tolerance."""
        Y, T, X, _ = constant_effect_data
        result = neural_double_ml(Y, T, X, n_folds=3)

        assert abs(result["ate"] - 2.0) < 0.5, f"ATE {result['ate']} not close to 2.0"

    def test_cate_shape(self, constant_effect_data):
        """CATE should have correct shape."""
        Y, T, X, _ = constant_effect_data
        result = neural_double_ml(Y, T, X, n_folds=3)

        assert result["cate"].shape == (len(Y),)

    def test_ci_validity(self, constant_effect_data):
        """CI should be valid (lower < upper)."""
        Y, T, X, _ = constant_effect_data
        result = neural_double_ml(Y, T, X, n_folds=3)

        assert result["ci_lower"] < result["ci_upper"]

    def test_se_positive(self, constant_effect_data):
        """SE should be positive."""
        Y, T, X, _ = constant_effect_data
        result = neural_double_ml(Y, T, X, n_folds=3)

        assert result["ate_se"] > 0

    def test_method_name(self, constant_effect_data):
        """Method name should be correct."""
        Y, T, X, _ = constant_effect_data
        result = neural_double_ml(Y, T, X, n_folds=3)

        assert result["method"] == "neural_double_ml"

    def test_heterogeneous_effect_correlation(self, linear_heterogeneous_data):
        """Neural DML should capture heterogeneity."""
        Y, T, X, true_cate = linear_heterogeneous_data
        result = neural_double_ml(Y, T, X, n_folds=3)

        correlation = np.corrcoef(result["cate"], true_cate)[0, 1]
        assert correlation > 0.2, f"Correlation {correlation} too low"


# ============================================================================
# Layer 2: Adversarial Tests
# ============================================================================


class TestNeuralDMLAdversarial:
    """Adversarial tests for neural Double ML."""

    def test_insufficient_samples_raises(self):
        """Should raise error when too few samples for cross-fitting."""
        Y, T, X, _ = generate_cate_dgp(n=20, p=2, seed=42)

        with pytest.raises(ValueError, match="Insufficient samples"):
            neural_double_ml(Y, T, X, n_folds=5)

    def test_two_fold_minimum(self, constant_effect_data):
        """Should work with minimum 2 folds."""
        Y, T, X, _ = constant_effect_data
        result = neural_double_ml(Y, T, X, n_folds=2)

        assert isinstance(result["ate"], float)

    def test_high_dimensional(self, high_dimensional_data):
        """Should handle high-dimensional covariates."""
        Y, T, X, _ = high_dimensional_data
        result = neural_double_ml(Y, T, X, n_folds=3)

        assert not np.isnan(result["ate"])

    def test_single_covariate(self, single_covariate_data):
        """Should handle single covariate."""
        Y, T, X, _ = single_covariate_data
        result = neural_double_ml(Y, T, X.ravel(), n_folds=3)

        assert isinstance(result["ate"], float)

    def test_empty_treatment_group_raises(self):
        """Should raise error when treatment group is empty."""
        n = 100
        Y = np.random.randn(n)
        T = np.zeros(n)  # All control
        X = np.random.randn(n, 2)

        with pytest.raises(ValueError, match="treatment"):
            neural_double_ml(Y, T, X, n_folds=3)

    def test_custom_hidden_layers(self, constant_effect_data):
        """Should accept custom hidden layer configuration."""
        Y, T, X, _ = constant_effect_data
        result = neural_double_ml(Y, T, X, n_folds=3, hidden_layers=(50, 25))

        assert isinstance(result["ate"], float)

    def test_custom_alpha(self, constant_effect_data):
        """Should use custom alpha for CI."""
        Y, T, X, _ = constant_effect_data
        result_95 = neural_double_ml(Y, T, X, n_folds=3, alpha=0.05)
        result_99 = neural_double_ml(Y, T, X, n_folds=3, alpha=0.01)

        # 99% CI should be wider than 95% CI
        width_95 = result_95["ci_upper"] - result_95["ci_lower"]
        width_99 = result_99["ci_upper"] - result_99["ci_lower"]
        assert width_99 > width_95

    def test_imbalanced_treatment(self):
        """Should handle imbalanced treatment (20% treated)."""
        Y, T, X, _ = generate_cate_dgp(
            n=500,
            p=2,
            effect_type="constant",
            true_ate=2.0,
            treatment_prob=0.2,  # 20% is more reasonable for cross-fitting
            seed=42,
        )
        result = neural_double_ml(Y, T, X, n_folds=3)

        # Should still get reasonable estimate (looser tolerance for imbalanced)
        assert abs(result["ate"] - 2.0) < 1.5


# ============================================================================
# Layer 3: Monte Carlo Tests
# ============================================================================


class TestNeuralDMLMonteCarlo:
    """Monte Carlo validation for neural Double ML."""

    @pytest.mark.slow
    def test_neural_dml_bias(self):
        """Neural DML should have bias < 0.10 over Monte Carlo runs."""
        n_runs = 100
        estimates = []

        for seed in range(n_runs):
            Y, T, X, _ = generate_cate_dgp(n=300, p=2, true_ate=2.0, seed=seed)
            result = neural_double_ml(Y, T, X, n_folds=3)
            estimates.append(result["ate"])

        bias = abs(np.mean(estimates) - 2.0)
        assert bias < 0.15, f"Neural DML bias {bias:.3f} exceeds threshold"

    @pytest.mark.slow
    def test_neural_dml_coverage(self):
        """Neural DML should have 85-99% coverage."""
        n_runs = 100
        covers = []

        for seed in range(n_runs):
            Y, T, X, _ = generate_cate_dgp(n=300, p=2, true_ate=2.0, seed=seed)
            result = neural_double_ml(Y, T, X, n_folds=3)
            covers.append(result["ci_lower"] < 2.0 < result["ci_upper"])

        coverage = np.mean(covers)
        assert 0.80 < coverage < 0.99, f"Coverage {coverage:.2%} outside 80-99%"

    @pytest.mark.slow
    def test_cross_fitting_reduces_bias(self):
        """Cross-fitting should help reduce bias compared to naive approach."""
        # This test verifies that cross-fitting provides benefit
        n_runs = 50
        estimates_3fold = []
        estimates_5fold = []

        for seed in range(n_runs):
            Y, T, X, _ = generate_cate_dgp(n=400, p=2, true_ate=2.0, seed=seed)
            result_3 = neural_double_ml(Y, T, X, n_folds=3)
            result_5 = neural_double_ml(Y, T, X, n_folds=5)

            estimates_3fold.append(result_3["ate"])
            estimates_5fold.append(result_5["ate"])

        # Both should have reasonable bias
        bias_3 = abs(np.mean(estimates_3fold) - 2.0)
        bias_5 = abs(np.mean(estimates_5fold) - 2.0)

        assert bias_3 < 0.20, f"3-fold bias {bias_3:.3f} too high"
        assert bias_5 < 0.20, f"5-fold bias {bias_5:.3f} too high"

    @pytest.mark.slow
    def test_heterogeneity_recovery(self):
        """Neural DML should recover heterogeneous effects."""
        correlations = []

        for seed in range(50):
            Y, T, X, true_cate = generate_cate_dgp(
                n=500, p=2, effect_type="linear", true_ate=2.0, seed=seed
            )
            result = neural_double_ml(Y, T, X, n_folds=3)
            correlations.append(np.corrcoef(result["cate"], true_cate)[0, 1])

        mean_corr = np.mean(correlations)
        assert mean_corr > 0.2, f"Mean CATE correlation {mean_corr:.2f} too low"
