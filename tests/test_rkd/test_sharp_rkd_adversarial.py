"""
Adversarial Tests for Sharp Regression Kink Design (RKD)

Tests extreme cases, boundary conditions, and failure modes to ensure
robust behavior under adversarial inputs.
"""

import numpy as np
import pytest

from src.causal_inference.rkd import SharpRKD


# =============================================================================
# Extreme Values
# =============================================================================


class TestRKDExtremeValues:
    """Test behavior with extreme input values."""

    def test_very_large_values(self):
        """Should handle very large outcome values."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-5, 5, n)
        d = np.where(x < 0, 0.5 * x, 1.5 * x)
        y = 1e6 * d + rng.normal(0, 1e5, n)  # Large scale

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        assert result.retcode in ("success", "warning")
        assert np.isfinite(result.estimate)
        # Should recover ~1e6
        assert 0.5e6 < result.estimate < 1.5e6

    def test_very_small_values(self):
        """Should handle very small outcome values."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-5, 5, n)
        d = np.where(x < 0, 0.5 * x, 1.5 * x)
        y = 1e-6 * d + rng.normal(0, 1e-7, n)  # Small scale

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        assert result.retcode in ("success", "warning")
        assert np.isfinite(result.estimate)
        # Should recover ~1e-6
        assert 0.5e-6 < result.estimate < 1.5e-6

    def test_extreme_cutoff_values(self):
        """Should handle extreme cutoff locations."""
        rng = np.random.default_rng(42)

        # Large positive cutoff
        cutoff = 1000.0
        x = rng.uniform(cutoff - 5, cutoff + 5, 500)
        d = np.where(x < cutoff, 0.5 * (x - cutoff), 1.5 * (x - cutoff))
        y = 2.0 * d + rng.normal(0, 1, 500)

        rkd = SharpRKD(cutoff=cutoff, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        assert result.retcode in ("success", "warning")
        assert np.isfinite(result.estimate)

    def test_negative_treatment_slopes(self):
        """Should handle negative slopes in treatment."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-5, 5, n)
        # Negative slopes that become more negative
        d = np.where(x < 0, -0.5 * x, -1.5 * x)
        y = 2.0 * d + rng.normal(0, 0.5, n)

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d, slope_d_left=-0.5, slope_d_right=-1.5)

        assert result.retcode in ("success", "warning")
        assert np.isfinite(result.estimate)


# =============================================================================
# Boundary Conditions
# =============================================================================


class TestRKDBoundaryConditions:
    """Test behavior at boundaries and edge cases."""

    def test_all_data_left_of_cutoff(self):
        """Should handle case where all data is left of cutoff."""
        rng = np.random.default_rng(42)
        x = rng.uniform(-10, -1, 100)  # All left of 0
        d = 0.5 * x
        y = 2.0 * d + rng.normal(0, 1, 100)

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        # Should error or return NaN - no data on right
        assert result.retcode == "error" or np.isnan(result.estimate)

    def test_all_data_right_of_cutoff(self):
        """Should handle case where all data is right of cutoff."""
        rng = np.random.default_rng(42)
        x = rng.uniform(1, 10, 100)  # All right of 0
        d = 1.5 * x
        y = 2.0 * d + rng.normal(0, 1, 100)

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        # Should error or return NaN - no data on left
        assert result.retcode == "error" or np.isnan(result.estimate)

    def test_very_narrow_bandwidth(self):
        """Should handle very narrow bandwidth."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.uniform(-5, 5, n)
        d = np.where(x < 0, 0.5 * x, 1.5 * x)
        y = 2.0 * d + rng.normal(0, 0.5, n)

        rkd = SharpRKD(cutoff=0.0, bandwidth=0.1)  # Very narrow
        result = rkd.fit(y, x, d)

        # May error due to insufficient data or succeed with high variance
        if result.retcode != "error":
            assert np.isfinite(result.estimate)

    def test_very_wide_bandwidth(self):
        """Should handle very wide bandwidth."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-5, 5, n)
        d = np.where(x < 0, 0.5 * x, 1.5 * x)
        y = 2.0 * d + rng.normal(0, 0.5, n)

        rkd = SharpRKD(cutoff=0.0, bandwidth=100.0)  # Very wide
        result = rkd.fit(y, x, d)

        assert result.retcode in ("success", "warning")
        assert np.isfinite(result.estimate)

    def test_single_observation_per_side(self):
        """Should handle minimal observations on each side."""
        rng = np.random.default_rng(42)
        # Very sparse data - need at least 10 total
        x = np.array([-3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 5])
        d = np.where(x < 0, 0.5 * x, 1.5 * x)
        y = 2.0 * d + rng.normal(0, 0.1, len(x))

        rkd = SharpRKD(cutoff=0.0, bandwidth=3.0, polynomial_order=1)
        result = rkd.fit(y, x, d)

        # Should complete (may error or warn due to small sample)
        assert result.retcode in ("success", "warning", "error")


# =============================================================================
# Data Quality Issues
# =============================================================================


class TestRKDDataQuality:
    """Test behavior with data quality issues."""

    def test_duplicate_x_values(self):
        """Should handle duplicate running variable values."""
        rng = np.random.default_rng(42)
        x = np.array([-2, -2, -1, -1, 1, 1, 2, 2] * 50)  # Duplicates
        d = np.where(x < 0, 0.5 * x, 1.5 * x)
        y = 2.0 * d + rng.normal(0, 0.5, len(x))

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        assert result.retcode in ("success", "warning")
        assert np.isfinite(result.estimate)

    def test_heavily_imbalanced_sides(self):
        """Should handle imbalanced data across cutoff."""
        rng = np.random.default_rng(42)
        # 90% left, 10% right
        x_left = rng.uniform(-5, 0, 450)
        x_right = rng.uniform(0, 5, 50)
        x = np.concatenate([x_left, x_right])
        d = np.where(x < 0, 0.5 * x, 1.5 * x)
        y = 2.0 * d + rng.normal(0, 0.5, len(x))

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        assert result.retcode in ("success", "warning")
        assert np.isfinite(result.estimate)
        # Sample sizes should reflect imbalance
        assert result.n_left > result.n_right

    def test_heteroskedastic_errors(self):
        """Should be robust to heteroskedastic errors."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-5, 5, n)
        d = np.where(x < 0, 0.5 * x, 1.5 * x)
        # Variance increases with |x|
        noise_scale = 0.1 + 0.5 * np.abs(x)
        y = 2.0 * d + rng.normal(0, noise_scale)

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        assert result.retcode in ("success", "warning")
        assert np.isfinite(result.estimate)
        # Should still be reasonably close to true effect
        assert abs(result.estimate - 2.0) < 1.0

    def test_outliers_in_outcome(self):
        """Should be somewhat robust to outliers."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-5, 5, n)
        d = np.where(x < 0, 0.5 * x, 1.5 * x)
        y = 2.0 * d + rng.normal(0, 0.5, n)

        # Add outliers
        outlier_idx = rng.choice(n, size=10, replace=False)
        y[outlier_idx] += rng.choice([-10, 10], size=10)

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        assert result.retcode in ("success", "warning")
        assert np.isfinite(result.estimate)


# =============================================================================
# Model Misspecification
# =============================================================================


class TestRKDMisspecification:
    """Test behavior under model misspecification."""

    def test_nonlinear_relationship(self):
        """Should handle nonlinear outcome-treatment relationship."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-5, 5, n)
        d = np.where(x < 0, 0.5 * x, 1.5 * x)
        # Quadratic relationship
        y = 2.0 * d + 0.1 * d**2 + rng.normal(0, 0.5, n)

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        assert result.retcode in ("success", "warning")
        assert np.isfinite(result.estimate)
        # Local linear should still estimate local effect reasonably

    def test_curved_regression_function(self):
        """Should handle curvature in the regression function."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-5, 5, n)
        d = np.where(x < 0, 0.5 * x, 1.5 * x)
        # Add curvature
        y = 2.0 * d + 0.05 * x**2 + rng.normal(0, 0.5, n)

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0, polynomial_order=2)
        result = rkd.fit(y, x, d)

        assert result.retcode in ("success", "warning")
        assert np.isfinite(result.estimate)

    def test_smooth_kink_instead_of_sharp(self):
        """Should handle smooth (rounded) kinks."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-5, 5, n)

        # Smooth kink using tanh
        smoothness = 2.0
        slope_blend = 0.5 + 0.5 * np.tanh(smoothness * x)
        d = slope_blend * 1.5 * x + (1 - slope_blend) * 0.5 * x

        y = 2.0 * d + rng.normal(0, 0.5, n)

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        # Should still produce finite result
        assert result.retcode in ("success", "warning")
        assert np.isfinite(result.estimate)


# =============================================================================
# Numerical Stability
# =============================================================================


class TestRKDNumericalStability:
    """Test numerical stability."""

    def test_nearly_singular_design(self):
        """Should handle near-singular design matrices."""
        rng = np.random.default_rng(42)
        # Nearly collinear data
        x_base = rng.uniform(-5, 5, 250)
        x = np.concatenate([x_base, x_base + 1e-10])  # Near-duplicate
        d = np.where(x < 0, 0.5 * x, 1.5 * x)
        y = 2.0 * d + rng.normal(0, 0.5, len(x))

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        # Should complete without crashing
        assert result.retcode in ("success", "warning", "error")

    def test_very_small_kink(self):
        """Should handle very small kink magnitude."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-5, 5, n)
        # Very small kink: 0.5 vs 0.51
        d = np.where(x < 0, 0.50 * x, 0.51 * x)
        y = 2.0 * d + rng.normal(0, 0.5, n)

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d, slope_d_left=0.50, slope_d_right=0.51)

        # Should complete (may have large SE due to small kink)
        if result.retcode != "error":
            assert np.isfinite(result.estimate)
            # SE should be large due to small kink denominator
            assert result.se > 1.0

    def test_zero_noise(self):
        """Should handle deterministic relationship (zero noise)."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-5, 5, n)
        d = np.where(x < 0, 0.5 * x, 1.5 * x)
        y = 2.0 * d  # No noise

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        assert result.retcode in ("success", "warning")
        # Should recover exact effect
        assert abs(result.estimate - 2.0) < 0.1


# =============================================================================
# Input Validation
# =============================================================================


class TestRKDInputValidation:
    """Test input validation."""

    def test_empty_arrays(self):
        """Should reject empty arrays."""
        rkd = SharpRKD(cutoff=0.0, bandwidth=1.0)

        with pytest.raises(ValueError):
            rkd.fit(np.array([]), np.array([]), np.array([]))

    def test_wrong_dimensions(self):
        """Should reject 2D arrays."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, (100, 2))  # 2D
        x = rng.uniform(-1, 1, (100, 2))
        d = x

        rkd = SharpRKD(cutoff=0.0, bandwidth=1.0)

        # Should either flatten or raise error
        try:
            result = rkd.fit(y, x, d)
            # If it doesn't raise, check it did something reasonable
            assert np.isfinite(result.estimate) or result.retcode == "error"
        except (ValueError, IndexError):
            pass  # Expected

    def test_invalid_bandwidth_string(self):
        """Should reject invalid bandwidth string."""
        with pytest.raises(ValueError, match="Unknown bandwidth"):
            rkd = SharpRKD(cutoff=0.0, bandwidth="invalid")
            rng = np.random.default_rng(42)
            y = rng.normal(0, 1, 100)
            x = rng.uniform(-1, 1, 100)
            d = x
            rkd.fit(y, x, d)

    def test_invalid_kernel(self):
        """Should reject invalid kernel."""
        rng = np.random.default_rng(42)
        n = 100
        x = rng.uniform(-5, 5, n)
        d = np.where(x < 0, 0.5 * x, 1.5 * x)
        y = 2.0 * d + rng.normal(0, 1, n)

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0, kernel="invalid")

        with pytest.raises(ValueError, match="Unknown kernel"):
            rkd.fit(y, x, d)
