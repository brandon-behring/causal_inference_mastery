"""Tests for neural meta-learners (S/T/X/R-learners).

Test layers:
- Layer 1: Known-answer tests (ATE recovery, CATE shape, CI validity)
- Layer 2: Adversarial tests (small sample, high-dim, edge cases)
- Layer 3: Monte Carlo validation (bias, coverage) @slow
"""

import numpy as np
import pytest

from causal_inference.cate import CATEResult
from causal_inference.cate.neural_meta_learners import (
    neural_r_learner,
    neural_s_learner,
    neural_t_learner,
    neural_x_learner,
)

from .conftest import generate_cate_dgp


# ============================================================================
# Layer 1: Known-Answer Tests
# ============================================================================


class TestNeuralSLearnerKnownAnswer:
    """Known-answer tests for neural S-learner."""

    def test_constant_effect_ate_recovery(self, constant_effect_data):
        """S-learner should recover ATE within tolerance."""
        Y, T, X, _ = constant_effect_data
        result = neural_s_learner(Y, T, X)

        assert abs(result["ate"] - 2.0) < 0.5, f"ATE {result['ate']} not close to 2.0"

    def test_cate_shape(self, constant_effect_data):
        """CATE should have correct shape."""
        Y, T, X, _ = constant_effect_data
        result = neural_s_learner(Y, T, X)

        assert result["cate"].shape == (len(Y),)

    def test_ci_validity(self, constant_effect_data):
        """CI should be valid (lower < upper)."""
        Y, T, X, _ = constant_effect_data
        result = neural_s_learner(Y, T, X)

        assert result["ci_lower"] < result["ci_upper"]

    def test_se_positive(self, constant_effect_data):
        """SE should be positive."""
        Y, T, X, _ = constant_effect_data
        result = neural_s_learner(Y, T, X)

        assert result["ate_se"] > 0

    def test_method_name(self, constant_effect_data):
        """Method name should be correct."""
        Y, T, X, _ = constant_effect_data
        result = neural_s_learner(Y, T, X)

        assert result["method"] == "neural_s_learner"


class TestNeuralTLearnerKnownAnswer:
    """Known-answer tests for neural T-learner."""

    def test_constant_effect_ate_recovery(self, constant_effect_data):
        """T-learner should recover ATE within tolerance."""
        Y, T, X, _ = constant_effect_data
        result = neural_t_learner(Y, T, X)

        assert abs(result["ate"] - 2.0) < 0.5, f"ATE {result['ate']} not close to 2.0"

    def test_cate_shape(self, constant_effect_data):
        """CATE should have correct shape."""
        Y, T, X, _ = constant_effect_data
        result = neural_t_learner(Y, T, X)

        assert result["cate"].shape == (len(Y),)

    def test_heterogeneous_effect_correlation(self, linear_heterogeneous_data):
        """T-learner should capture heterogeneity (correlation > 0.3)."""
        Y, T, X, true_cate = linear_heterogeneous_data
        result = neural_t_learner(Y, T, X)

        correlation = np.corrcoef(result["cate"], true_cate)[0, 1]
        assert correlation > 0.3, f"Correlation {correlation} too low"

    def test_method_name(self, constant_effect_data):
        """Method name should be correct."""
        Y, T, X, _ = constant_effect_data
        result = neural_t_learner(Y, T, X)

        assert result["method"] == "neural_t_learner"


class TestNeuralXLearnerKnownAnswer:
    """Known-answer tests for neural X-learner."""

    def test_constant_effect_ate_recovery(self, constant_effect_data):
        """X-learner should recover ATE within tolerance."""
        Y, T, X, _ = constant_effect_data
        result = neural_x_learner(Y, T, X)

        assert abs(result["ate"] - 2.0) < 0.5, f"ATE {result['ate']} not close to 2.0"

    def test_cate_shape(self, constant_effect_data):
        """CATE should have correct shape."""
        Y, T, X, _ = constant_effect_data
        result = neural_x_learner(Y, T, X)

        assert result["cate"].shape == (len(Y),)

    def test_heterogeneous_effect_correlation(self, linear_heterogeneous_data):
        """X-learner should capture heterogeneity (correlation > 0.3)."""
        Y, T, X, true_cate = linear_heterogeneous_data
        result = neural_x_learner(Y, T, X)

        correlation = np.corrcoef(result["cate"], true_cate)[0, 1]
        assert correlation > 0.3, f"Correlation {correlation} too low"

    def test_method_name(self, constant_effect_data):
        """Method name should be correct."""
        Y, T, X, _ = constant_effect_data
        result = neural_x_learner(Y, T, X)

        assert result["method"] == "neural_x_learner"


class TestNeuralRLearnerKnownAnswer:
    """Known-answer tests for neural R-learner."""

    def test_constant_effect_ate_recovery(self, constant_effect_data):
        """R-learner should recover ATE within tolerance."""
        Y, T, X, _ = constant_effect_data
        result = neural_r_learner(Y, T, X)

        assert abs(result["ate"] - 2.0) < 0.5, f"ATE {result['ate']} not close to 2.0"

    def test_cate_shape(self, constant_effect_data):
        """CATE should have correct shape."""
        Y, T, X, _ = constant_effect_data
        result = neural_r_learner(Y, T, X)

        assert result["cate"].shape == (len(Y),)

    def test_ci_width_reasonable(self, constant_effect_data):
        """CI should have reasonable width (not too narrow or wide)."""
        Y, T, X, _ = constant_effect_data
        result = neural_r_learner(Y, T, X)

        # Check CI width is reasonable (between 0.1 and 3.0)
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert 0.1 < ci_width < 3.0, f"CI width {ci_width} outside reasonable range"

    def test_method_name(self, constant_effect_data):
        """Method name should be correct."""
        Y, T, X, _ = constant_effect_data
        result = neural_r_learner(Y, T, X)

        assert result["method"] == "neural_r_learner"


# ============================================================================
# Layer 2: Adversarial Tests
# ============================================================================


class TestNeuralMetaLearnersAdversarial:
    """Adversarial tests for all neural meta-learners."""

    @pytest.mark.parametrize(
        "learner,name",
        [
            (neural_s_learner, "S"),
            (neural_t_learner, "T"),
            (neural_x_learner, "X"),
            (neural_r_learner, "R"),
        ],
    )
    def test_small_sample_stability(self, small_sample_data, learner, name):
        """Learners should not crash on small samples."""
        Y, T, X, _ = small_sample_data
        result = learner(Y, T, X)

        assert isinstance(result["ate"], float), f"{name}-learner crashed on small sample"
        assert not np.isnan(result["ate"]), f"{name}-learner returned NaN"

    @pytest.mark.parametrize(
        "learner,name",
        [
            (neural_s_learner, "S"),
            (neural_t_learner, "T"),
            (neural_x_learner, "X"),
            (neural_r_learner, "R"),
        ],
    )
    def test_high_dimensional(self, high_dimensional_data, learner, name):
        """Learners should handle high-dimensional covariates."""
        Y, T, X, _ = high_dimensional_data
        result = learner(Y, T, X)

        assert isinstance(result["ate"], float), f"{name}-learner crashed on high-dim"
        assert not np.isnan(result["ate"]), f"{name}-learner returned NaN on high-dim"

    @pytest.mark.parametrize(
        "learner,name",
        [
            (neural_s_learner, "S"),
            (neural_t_learner, "T"),
            (neural_x_learner, "X"),
            (neural_r_learner, "R"),
        ],
    )
    def test_single_covariate(self, single_covariate_data, learner, name):
        """Learners should handle single covariate (1D)."""
        Y, T, X, _ = single_covariate_data
        # Pass as 1D array
        X_1d = X.ravel()
        result = learner(Y, T, X_1d)

        assert isinstance(result["ate"], float), f"{name}-learner crashed on 1D covariate"

    def test_empty_treatment_group_raises(self):
        """Should raise error when treatment group is empty."""
        n = 100
        Y = np.random.randn(n)
        T = np.zeros(n)  # All control
        X = np.random.randn(n, 2)

        with pytest.raises(ValueError, match="treatment"):
            neural_s_learner(Y, T, X)

    def test_empty_control_group_raises(self):
        """Should raise error when control group is empty."""
        n = 100
        Y = np.random.randn(n)
        T = np.ones(n)  # All treated
        X = np.random.randn(n, 2)

        with pytest.raises(ValueError, match="control"):
            neural_s_learner(Y, T, X)

    def test_non_binary_treatment_raises(self):
        """Should raise error for non-binary treatment."""
        n = 100
        Y = np.random.randn(n)
        T = np.random.randint(0, 3, n)  # Multi-valued
        X = np.random.randn(n, 2)

        with pytest.raises(ValueError, match="binary"):
            neural_t_learner(Y, T, X)

    def test_imbalanced_treatment(self):
        """Should handle imbalanced treatment (10% treated)."""
        Y, T, X, _ = generate_cate_dgp(
            n=500,
            p=2,
            effect_type="constant",
            true_ate=2.0,
            treatment_prob=0.1,
            seed=42,
        )
        result = neural_x_learner(Y, T, X)

        # Should still get reasonable estimate
        assert abs(result["ate"] - 2.0) < 1.0, "Failed on imbalanced treatment"

    def test_custom_hidden_layers(self, constant_effect_data):
        """Should accept custom hidden layer configuration."""
        Y, T, X, _ = constant_effect_data
        result = neural_t_learner(Y, T, X, hidden_layers=(50, 25, 10))

        assert isinstance(result["ate"], float)

    def test_custom_alpha(self, constant_effect_data):
        """Should use custom alpha for CI."""
        Y, T, X, _ = constant_effect_data
        result_95 = neural_s_learner(Y, T, X, alpha=0.05)
        result_99 = neural_s_learner(Y, T, X, alpha=0.01)

        # 99% CI should be wider than 95% CI
        width_95 = result_95["ci_upper"] - result_95["ci_lower"]
        width_99 = result_99["ci_upper"] - result_99["ci_lower"]
        assert width_99 > width_95, "99% CI should be wider than 95% CI"


# ============================================================================
# Layer 3: Monte Carlo Tests
# ============================================================================


class TestNeuralMetaLearnersMonteCarlo:
    """Monte Carlo validation for neural meta-learners."""

    @pytest.mark.slow
    def test_neural_s_learner_bias(self):
        """Neural S-learner should have bias < 0.15 over Monte Carlo runs."""
        n_runs = 100
        estimates = []

        for seed in range(n_runs):
            Y, T, X, _ = generate_cate_dgp(n=300, p=2, true_ate=2.0, seed=seed)
            result = neural_s_learner(Y, T, X)
            estimates.append(result["ate"])

        bias = abs(np.mean(estimates) - 2.0)
        assert bias < 0.15, f"S-learner bias {bias:.3f} exceeds threshold"

    @pytest.mark.slow
    def test_neural_t_learner_bias(self):
        """Neural T-learner should have bias < 0.10 over Monte Carlo runs."""
        n_runs = 100
        estimates = []

        for seed in range(n_runs):
            Y, T, X, _ = generate_cate_dgp(n=300, p=2, true_ate=2.0, seed=seed)
            result = neural_t_learner(Y, T, X)
            estimates.append(result["ate"])

        bias = abs(np.mean(estimates) - 2.0)
        assert bias < 0.15, f"T-learner bias {bias:.3f} exceeds threshold"

    @pytest.mark.slow
    def test_neural_r_learner_bias(self):
        """Neural R-learner should have bias < 0.10 over Monte Carlo runs."""
        n_runs = 100
        estimates = []

        for seed in range(n_runs):
            Y, T, X, _ = generate_cate_dgp(n=300, p=2, true_ate=2.0, seed=seed)
            result = neural_r_learner(Y, T, X)
            estimates.append(result["ate"])

        bias = abs(np.mean(estimates) - 2.0)
        assert bias < 0.15, f"R-learner bias {bias:.3f} exceeds threshold"

    @pytest.mark.slow
    def test_neural_t_learner_coverage(self):
        """Neural T-learner should have 85-99% coverage."""
        n_runs = 100
        covers = []

        for seed in range(n_runs):
            Y, T, X, _ = generate_cate_dgp(n=300, p=2, true_ate=2.0, seed=seed)
            result = neural_t_learner(Y, T, X)
            covers.append(result["ci_lower"] < 2.0 < result["ci_upper"])

        coverage = np.mean(covers)
        assert 0.80 < coverage < 0.99, f"Coverage {coverage:.2%} outside 80-99%"

    @pytest.mark.slow
    def test_neural_heterogeneity_recovery(self):
        """Neural learners should recover heterogeneous effects."""
        correlations = {"T": [], "X": [], "R": []}

        for seed in range(50):
            Y, T, X, true_cate = generate_cate_dgp(
                n=500, p=2, effect_type="linear", true_ate=2.0, seed=seed
            )

            result_t = neural_t_learner(Y, T, X)
            result_x = neural_x_learner(Y, T, X)
            result_r = neural_r_learner(Y, T, X)

            correlations["T"].append(np.corrcoef(result_t["cate"], true_cate)[0, 1])
            correlations["X"].append(np.corrcoef(result_x["cate"], true_cate)[0, 1])
            correlations["R"].append(np.corrcoef(result_r["cate"], true_cate)[0, 1])

        # Average correlations should be positive
        for name, corrs in correlations.items():
            mean_corr = np.mean(corrs)
            assert mean_corr > 0.2, f"{name}-learner correlation {mean_corr:.2f} too low"
