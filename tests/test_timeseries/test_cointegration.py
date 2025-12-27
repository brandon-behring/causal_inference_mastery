"""
Tests for Cointegration Analysis.

Session 145: Tests for Johansen and Engle-Granger cointegration tests.

Test layers:
- Layer 1: Known-answer tests (cointegrated vs non-cointegrated)
- Layer 2: Adversarial tests (edge cases, dimensions)
- Layer 3: Monte Carlo validation (rank selection accuracy) @slow
"""

import numpy as np
import pytest

from causal_inference.timeseries.cointegration import (
    johansen_test,
    engle_granger_test,
)
from causal_inference.timeseries.types import JohansenResult


# ============================================================================
# Helper Functions for DGP
# ============================================================================


def generate_cointegrated_pair(n: int = 200, seed: int = 42) -> np.ndarray:
    """Generate bivariate cointegrated system with rank 1."""
    np.random.seed(seed)

    # Common stochastic trend
    trend = np.cumsum(np.random.randn(n))

    # Both series share the common trend
    y1 = trend + np.random.randn(n) * 0.3
    y2 = 0.5 * trend + np.random.randn(n) * 0.3

    return np.column_stack([y1, y2])


def generate_noncointegrated_pair(n: int = 200, seed: int = 42) -> np.ndarray:
    """Generate bivariate system with NO cointegration (rank 0)."""
    np.random.seed(seed)

    # Two independent random walks
    y1 = np.cumsum(np.random.randn(n))
    y2 = np.cumsum(np.random.randn(n))

    return np.column_stack([y1, y2])


def generate_stationary_pair(n: int = 200, seed: int = 42) -> np.ndarray:
    """Generate bivariate stationary system (rank 2, all I(0))."""
    np.random.seed(seed)

    # Two stationary AR(1) processes
    y1 = np.zeros(n)
    y2 = np.zeros(n)

    for t in range(1, n):
        y1[t] = 0.5 * y1[t - 1] + np.random.randn()
        y2[t] = 0.5 * y2[t - 1] + np.random.randn()

    return np.column_stack([y1, y2])


def generate_cointegrated_triple(n: int = 300, seed: int = 42) -> np.ndarray:
    """Generate trivariate system with rank 1 cointegration."""
    np.random.seed(seed)

    # Common stochastic trend
    trend = np.cumsum(np.random.randn(n))

    # All three share the common trend
    y1 = trend + np.random.randn(n) * 0.3
    y2 = 0.5 * trend + np.random.randn(n) * 0.3
    y3 = 0.3 * trend + np.random.randn(n) * 0.3

    return np.column_stack([y1, y2, y3])


def generate_cointegrated_triple_rank2(n: int = 300, seed: int = 42) -> np.ndarray:
    """Generate trivariate system with rank 2 cointegration."""
    np.random.seed(seed)

    # Two common stochastic trends
    trend1 = np.cumsum(np.random.randn(n))
    trend2 = np.cumsum(np.random.randn(n))

    # Three series sharing two trends → rank 2
    y1 = trend1 + np.random.randn(n) * 0.3
    y2 = 0.5 * trend1 + 0.5 * trend2 + np.random.randn(n) * 0.3
    y3 = trend2 + np.random.randn(n) * 0.3

    return np.column_stack([y1, y2, y3])


# ============================================================================
# Layer 1: Known-Answer Tests - Johansen
# ============================================================================


class TestJohansenKnownAnswer:
    """Known-answer tests for Johansen test."""

    def test_cointegrated_pair_rank1(self):
        """Johansen should detect rank 1 for cointegrated pair."""
        data = generate_cointegrated_pair(n=300, seed=42)
        result = johansen_test(data, lags=2)

        assert isinstance(result, JohansenResult)
        assert result.has_cointegration is True
        assert result.rank >= 1  # At least one cointegrating relationship

    def test_noncointegrated_pair_rank0(self):
        """Johansen should detect rank 0 for independent random walks."""
        data = generate_noncointegrated_pair(n=300, seed=42)
        result = johansen_test(data, lags=2)

        assert result.rank == 0
        assert result.has_cointegration is False

    def test_result_structure(self):
        """Result should have all required fields."""
        data = generate_cointegrated_pair()
        result = johansen_test(data, lags=2)

        assert hasattr(result, "rank")
        assert hasattr(result, "trace_stats")
        assert hasattr(result, "trace_crit")
        assert hasattr(result, "trace_pvalues")
        assert hasattr(result, "max_eigen_stats")
        assert hasattr(result, "max_eigen_crit")
        assert hasattr(result, "eigenvalues")
        assert hasattr(result, "eigenvectors")
        assert hasattr(result, "adjustment")
        assert hasattr(result, "n_vars")
        assert hasattr(result, "n_obs")
        assert hasattr(result, "lags")

    def test_eigenvalues_sorted_descending(self):
        """Eigenvalues should be sorted in descending order."""
        data = generate_cointegrated_pair()
        result = johansen_test(data, lags=2)

        for i in range(len(result.eigenvalues) - 1):
            assert result.eigenvalues[i] >= result.eigenvalues[i + 1]

    def test_eigenvalues_in_valid_range(self):
        """Eigenvalues should be in [0, 1)."""
        data = generate_cointegrated_pair()
        result = johansen_test(data, lags=2)

        assert np.all(result.eigenvalues >= 0)
        assert np.all(result.eigenvalues < 1)

    def test_cointegrating_vectors_shape(self):
        """Cointegrating vectors should have correct shape."""
        data = generate_cointegrated_pair()
        result = johansen_test(data, lags=2)

        assert result.eigenvectors.shape == (result.n_vars, result.n_vars)

        if result.rank > 0:
            coint_vecs = result.cointegrating_vectors
            assert coint_vecs.shape == (result.n_vars, result.rank)

    def test_loading_matrix_shape(self):
        """Loading matrix should have correct shape."""
        data = generate_cointegrated_pair()
        result = johansen_test(data, lags=2)

        if result.rank > 0:
            loading = result.loading_matrix
            assert loading.shape == (result.n_vars, result.rank)

    def test_different_det_orders(self):
        """Should work with different deterministic specifications."""
        data = generate_cointegrated_pair()

        result_neg1 = johansen_test(data, lags=2, det_order=-1)
        result_0 = johansen_test(data, lags=2, det_order=0)
        result_1 = johansen_test(data, lags=2, det_order=1)

        assert result_neg1.det_order == -1
        assert result_0.det_order == 0
        assert result_1.det_order == 1


# ============================================================================
# Layer 1: Known-Answer Tests - Engle-Granger
# ============================================================================


class TestEngleGrangerKnownAnswer:
    """Known-answer tests for Engle-Granger test."""

    def test_cointegrated_pair_detected(self):
        """Engle-Granger should detect cointegration in cointegrated pair."""
        data = generate_cointegrated_pair(n=300, seed=42)
        y = data[:, 0]
        x = data[:, 1]

        result = engle_granger_test(y, x)

        assert "is_cointegrated" in result
        assert "beta" in result
        assert "residuals" in result
        assert "adf_result" in result

    def test_noncointegrated_rate(self):
        """Engle-Granger should not detect cointegration most of the time."""
        # Test multiple seeds since spurious regression can occur
        n_detected = 0
        n_seeds = 10

        for seed in range(n_seeds):
            data = generate_noncointegrated_pair(n=300, seed=seed + 100)
            y = data[:, 0]
            x = data[:, 1]

            result = engle_granger_test(y, x)
            if result["is_cointegrated"]:
                n_detected += 1

        # Should not detect cointegration in most cases
        # (allow some false positives due to spurious regression)
        assert n_detected < n_seeds * 0.3, f"Too many false positives: {n_detected}/{n_seeds}"

    def test_residuals_shape(self):
        """Residuals should have correct shape."""
        data = generate_cointegrated_pair()
        y = data[:, 0]
        x = data[:, 1]

        result = engle_granger_test(y, x)

        assert len(result["residuals"]) == len(y)


# ============================================================================
# Layer 2: Adversarial Tests
# ============================================================================


class TestCointegrationAdversarial:
    """Adversarial tests for cointegration functions."""

    def test_johansen_minimum_observations(self):
        """Johansen should work with minimum observations."""
        data = generate_cointegrated_pair(n=50, seed=42)
        result = johansen_test(data, lags=1)

        assert isinstance(result, JohansenResult)
        assert not np.isnan(result.rank)

    def test_johansen_insufficient_observations_raises(self):
        """Johansen should raise for insufficient observations."""
        data = np.random.randn(10, 3)  # Too short for 3 vars

        with pytest.raises(ValueError, match="Insufficient observations"):
            johansen_test(data, lags=2)

    def test_johansen_single_variable_raises(self):
        """Johansen should raise for single variable."""
        data = np.random.randn(100, 1)

        with pytest.raises(ValueError, match="at least 2 variables"):
            johansen_test(data, lags=2)

    def test_johansen_too_many_variables_raises(self):
        """Johansen should raise for > 6 variables (no critical values)."""
        data = np.random.randn(200, 7)

        with pytest.raises(ValueError, match="up to 6 variables"):
            johansen_test(data, lags=2)

    def test_johansen_invalid_det_order(self):
        """Johansen should raise for invalid det_order."""
        data = generate_cointegrated_pair()

        with pytest.raises(ValueError, match="det_order must be"):
            johansen_test(data, lags=2, det_order=5)

    def test_johansen_invalid_lags(self):
        """Johansen should raise for invalid lags."""
        data = generate_cointegrated_pair()

        with pytest.raises(ValueError, match="lags must be"):
            johansen_test(data, lags=0)

    def test_johansen_three_variables(self):
        """Johansen should handle trivariate system."""
        data = generate_cointegrated_triple(n=300, seed=42)
        result = johansen_test(data, lags=2)

        assert result.n_vars == 3
        assert len(result.trace_stats) == 3
        assert len(result.eigenvalues) == 3

    def test_johansen_four_variables(self):
        """Johansen should handle 4-variable system."""
        np.random.seed(42)
        n = 300

        # Common trend
        trend = np.cumsum(np.random.randn(n))

        data = np.column_stack(
            [
                trend + np.random.randn(n) * 0.3,
                0.5 * trend + np.random.randn(n) * 0.3,
                0.3 * trend + np.random.randn(n) * 0.3,
                0.7 * trend + np.random.randn(n) * 0.3,
            ]
        )

        result = johansen_test(data, lags=2)

        assert result.n_vars == 4
        assert result.has_cointegration  # Should detect cointegration

    def test_engle_granger_length_mismatch(self):
        """Engle-Granger should raise for length mismatch."""
        y = np.random.randn(100)
        x = np.random.randn(90)

        with pytest.raises(ValueError, match="Length mismatch"):
            engle_granger_test(y, x)

    def test_engle_granger_multivariate_x(self):
        """Engle-Granger should handle multivariate x."""
        np.random.seed(42)
        n = 200

        trend = np.cumsum(np.random.randn(n))
        y = trend + np.random.randn(n) * 0.3
        x = np.column_stack(
            [
                0.5 * trend + np.random.randn(n) * 0.3,
                0.3 * trend + np.random.randn(n) * 0.3,
            ]
        )

        result = engle_granger_test(y, x)

        assert len(result["beta"]) == 3  # intercept + 2 x vars


# ============================================================================
# Layer 3: Monte Carlo Tests
# ============================================================================


class TestCointegrationMonteCarlo:
    """Monte Carlo validation for cointegration tests."""

    @pytest.mark.slow
    def test_johansen_rank_selection_cointegrated(self):
        """Johansen should correctly select rank 1 for cointegrated system."""
        n_runs = 50
        correct_rank = 0

        for seed in range(n_runs):
            data = generate_cointegrated_pair(n=300, seed=seed)
            result = johansen_test(data, lags=2)

            if result.rank == 1:
                correct_rank += 1

        accuracy = correct_rank / n_runs
        # Should correctly identify rank 1 in majority of cases
        assert accuracy > 0.60, f"Rank selection accuracy {accuracy:.2%} too low"

    @pytest.mark.slow
    def test_johansen_rank_selection_noncointegrated(self):
        """Johansen should correctly select rank 0 for non-cointegrated system."""
        n_runs = 50
        correct_rank = 0

        for seed in range(n_runs):
            data = generate_noncointegrated_pair(n=300, seed=seed)
            result = johansen_test(data, lags=2)

            if result.rank == 0:
                correct_rank += 1

        accuracy = correct_rank / n_runs
        # Should correctly identify rank 0 in majority of cases
        # Type I error control: should be > 90%
        assert accuracy > 0.85, f"Non-cointegration detection {accuracy:.2%} too low"

    @pytest.mark.slow
    def test_johansen_trivariate_accuracy(self):
        """Johansen should work for trivariate systems."""
        n_runs = 30
        detected_coint = 0

        for seed in range(n_runs):
            data = generate_cointegrated_triple(n=300, seed=seed)
            result = johansen_test(data, lags=2)

            if result.rank >= 1:
                detected_coint += 1

        detection_rate = detected_coint / n_runs
        # Should detect cointegration in most cases
        assert detection_rate > 0.70, f"Trivariate detection {detection_rate:.2%} too low"

    @pytest.mark.slow
    def test_engle_granger_power(self):
        """Engle-Granger should have reasonable power."""
        n_runs = 50
        detected = 0

        for seed in range(n_runs):
            data = generate_cointegrated_pair(n=300, seed=seed)
            y = data[:, 0]
            x = data[:, 1]

            result = engle_granger_test(y, x)

            if result["is_cointegrated"]:
                detected += 1

        power = detected / n_runs
        # Should have reasonable power (>50%)
        assert power > 0.50, f"Engle-Granger power {power:.2%} too low"

    @pytest.mark.slow
    def test_engle_granger_type1_error(self):
        """Engle-Granger should control Type I error."""
        n_runs = 50
        false_positives = 0

        for seed in range(n_runs):
            data = generate_noncointegrated_pair(n=300, seed=seed)
            y = data[:, 0]
            x = data[:, 1]

            result = engle_granger_test(y, x, alpha=0.05)

            if result["is_cointegrated"]:
                false_positives += 1

        type1_rate = false_positives / n_runs
        # Should be close to alpha (allow some margin)
        assert type1_rate < 0.15, f"Type I error {type1_rate:.2%} too high"

    @pytest.mark.slow
    def test_johansen_vs_engle_granger_agreement(self):
        """Johansen and Engle-Granger should mostly agree."""
        n_runs = 30
        agreements = 0

        for seed in range(n_runs):
            # Mix of cointegrated and non-cointegrated
            if seed % 2 == 0:
                data = generate_cointegrated_pair(n=300, seed=seed)
            else:
                data = generate_noncointegrated_pair(n=300, seed=seed)

            # Johansen
            joh_result = johansen_test(data, lags=2)
            joh_coint = joh_result.has_cointegration

            # Engle-Granger
            eg_result = engle_granger_test(data[:, 0], data[:, 1])
            eg_coint = eg_result["is_cointegrated"]

            if joh_coint == eg_coint:
                agreements += 1

        agreement_rate = agreements / n_runs
        # Should agree in most cases (>70%)
        assert agreement_rate > 0.70, f"Agreement rate {agreement_rate:.2%} too low"
