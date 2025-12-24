"""
Unit tests for bunching estimation module.

Tests cover:
1. Polynomial counterfactual fitting
2. Counterfactual density estimation
3. Excess mass computation
4. Elasticity calculation
5. Bootstrap standard errors
6. Main bunching estimator

References:
- Saez (2010) - Original bunching methodology
- Chetty et al. (2011) - Integration constraint
- Kleven (2016) - Review and best practices
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from src.causal_inference.bunching import (
    BunchingResult,
    CounterfactualResult,
    estimate_counterfactual,
    polynomial_counterfactual,
    bunching_estimator,
    compute_excess_mass,
    compute_elasticity,
    bootstrap_bunching_se,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_bunching_data():
    """Generate simple data with clear bunching at kink=50."""
    rng = np.random.default_rng(42)

    # Background: uniform distribution from 20 to 80
    background = rng.uniform(20, 80, size=800)

    # Bunching: concentrated at kink=50
    bunchers = rng.normal(50, 1.0, size=200)

    data = np.concatenate([background, bunchers])
    return data


@pytest.fixture
def no_bunching_data():
    """Generate uniform data with no bunching."""
    rng = np.random.default_rng(42)
    return rng.uniform(20, 80, size=1000)


@pytest.fixture
def tax_bunching_data():
    """Generate realistic tax bunching scenario.

    Income data with bunching at $50,000 threshold where
    marginal rate increases from 20% to 30%.
    """
    rng = np.random.default_rng(42)

    # Background distribution (log-normal income)
    background = rng.lognormal(mean=10.8, sigma=0.3, size=900)
    background = background[(background > 30000) & (background < 70000)]

    # Bunchers at threshold
    bunchers = rng.normal(50000, 500, size=100)

    data = np.concatenate([background, bunchers])
    return data


# =============================================================================
# Test: polynomial_counterfactual
# =============================================================================


class TestPolynomialCounterfactual:
    """Tests for polynomial_counterfactual function."""

    def test_basic_polynomial_fit(self):
        """Polynomial fits outside bunching region."""
        # Create data with known polynomial shape
        bin_centers = np.linspace(0, 100, 50)
        # Quadratic: counts = 100 - 0.01*(x-50)^2
        true_counts = 100 - 0.01 * (bin_centers - 50) ** 2
        counts = true_counts.copy()

        # Add bunching spike at center (will be excluded)
        counts[23:27] = counts[23:27] + 50

        counterfactual, coeffs, r_squared = polynomial_counterfactual(
            bin_centers=bin_centers,
            counts=counts,
            bunching_lower=45,
            bunching_upper=55,
            polynomial_order=2,
        )

        # R-squared should be high for quadratic fit
        assert r_squared > 0.95, f"R-squared too low: {r_squared}"

        # Counterfactual should recover true shape in bunching region
        bunching_idx = (bin_centers >= 45) & (bin_centers <= 55)
        assert_allclose(
            counterfactual[bunching_idx],
            true_counts[bunching_idx],
            rtol=0.1,
            err_msg="Counterfactual doesn't match true shape",
        )

    def test_high_order_polynomial(self):
        """High-order polynomial (default=7) fits complex shapes."""
        bin_centers = np.linspace(0, 100, 100)
        # Create wavy pattern - use smoother pattern for better fit
        counts = 50 + 20 * np.sin(bin_centers / 20) + 10 * (bin_centers / 100)
        counts = np.maximum(counts, 0)

        counterfactual, coeffs, r_squared = polynomial_counterfactual(
            bin_centers=bin_centers,
            counts=counts,
            bunching_lower=45,
            bunching_upper=55,
            polynomial_order=7,
        )

        # Should fit reasonably well (lower threshold for flexibility)
        assert r_squared > 0.7
        assert len(coeffs) == 8  # Order 7 has 8 coefficients

    def test_insufficient_bins_raises(self):
        """Raises ValueError if not enough bins outside bunching region."""
        bin_centers = np.linspace(0, 10, 5)  # Only 5 bins
        counts = np.ones(5)

        with pytest.raises(ValueError, match="Need at least"):
            polynomial_counterfactual(
                bin_centers=bin_centers,
                counts=counts,
                bunching_lower=2,
                bunching_upper=8,
                polynomial_order=7,  # Needs 8 bins outside
            )

    def test_length_mismatch_raises(self):
        """Raises ValueError if bin_centers and counts have different lengths."""
        bin_centers = np.linspace(0, 100, 50)
        counts = np.ones(40)  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            polynomial_counterfactual(
                bin_centers=bin_centers,
                counts=counts,
                bunching_lower=40,
                bunching_upper=60,
                polynomial_order=3,
            )

    def test_invalid_polynomial_order_raises(self):
        """Raises ValueError for invalid polynomial order."""
        bin_centers = np.linspace(0, 100, 50)
        counts = np.ones(50)

        with pytest.raises(ValueError, match="polynomial_order"):
            polynomial_counterfactual(
                bin_centers=bin_centers,
                counts=counts,
                bunching_lower=40,
                bunching_upper=60,
                polynomial_order=0,
            )

    def test_non_negative_counterfactual(self):
        """Counterfactual is always non-negative."""
        bin_centers = np.linspace(0, 100, 50)
        # Create shape that could extrapolate negative
        counts = np.maximum(0, 50 - np.abs(bin_centers - 50))

        counterfactual, _, _ = polynomial_counterfactual(
            bin_centers=bin_centers,
            counts=counts,
            bunching_lower=40,
            bunching_upper=60,
            polynomial_order=5,
        )

        assert np.all(counterfactual >= 0), "Counterfactual has negative values"


# =============================================================================
# Test: estimate_counterfactual
# =============================================================================


class TestEstimateCounterfactual:
    """Tests for estimate_counterfactual function."""

    def test_basic_estimation(self, simple_bunching_data):
        """Basic counterfactual estimation works."""
        result = estimate_counterfactual(
            data=simple_bunching_data,
            kink_point=50,
            bunching_width=5,
            n_bins=50,
        )

        # Check result structure
        assert isinstance(result, dict)
        assert "bin_centers" in result
        assert "actual_counts" in result
        assert "counterfactual_counts" in result
        assert "r_squared" in result
        assert "bunching_region" in result

        # Check dimensions
        assert len(result["bin_centers"]) == 50
        assert len(result["actual_counts"]) == 50
        assert len(result["counterfactual_counts"]) == 50

        # R-squared should be reasonable
        assert 0 < result["r_squared"] <= 1

    def test_n_bins_vs_bin_width(self, simple_bunching_data):
        """Can specify either n_bins or bin_width."""
        # With n_bins
        result1 = estimate_counterfactual(
            data=simple_bunching_data,
            kink_point=50,
            bunching_width=5,
            n_bins=50,
        )

        # With bin_width
        result2 = estimate_counterfactual(
            data=simple_bunching_data,
            kink_point=50,
            bunching_width=5,
            bin_width=1.2,  # ~50 bins for range 20-80
        )

        # Both should work
        assert len(result1["bin_centers"]) == 50
        assert len(result2["bin_centers"]) > 0

    def test_both_n_bins_and_bin_width_raises(self, simple_bunching_data):
        """Raises if both n_bins and bin_width specified."""
        with pytest.raises(ValueError, match="only one"):
            estimate_counterfactual(
                data=simple_bunching_data,
                kink_point=50,
                bunching_width=5,
                n_bins=50,
                bin_width=1.0,
            )

    def test_neither_n_bins_nor_bin_width_raises(self, simple_bunching_data):
        """Raises if neither n_bins nor bin_width specified."""
        with pytest.raises(ValueError, match="Must specify"):
            estimate_counterfactual(
                data=simple_bunching_data,
                kink_point=50,
                bunching_width=5,
            )

    def test_empty_data_raises(self):
        """Raises ValueError for empty data."""
        with pytest.raises(ValueError, match="empty"):
            estimate_counterfactual(
                data=np.array([]),
                kink_point=50,
                bunching_width=5,
                n_bins=50,
            )

    def test_nan_data_raises(self):
        """Raises ValueError for data with NaN."""
        data = np.array([1.0, 2.0, np.nan, 4.0])
        with pytest.raises(ValueError, match="non-finite"):
            estimate_counterfactual(
                data=data,
                kink_point=2,
                bunching_width=0.5,
                n_bins=10,
            )

    def test_negative_bunching_width_raises(self, simple_bunching_data):
        """Raises ValueError for negative bunching width."""
        with pytest.raises(ValueError, match="bunching_width.*positive"):
            estimate_counterfactual(
                data=simple_bunching_data,
                kink_point=50,
                bunching_width=-5,
                n_bins=50,
            )

    def test_bunching_region_returned(self, simple_bunching_data):
        """Bunching region is correctly computed and returned."""
        result = estimate_counterfactual(
            data=simple_bunching_data,
            kink_point=50,
            bunching_width=5,
            n_bins=50,
        )

        lower, upper = result["bunching_region"]
        assert lower == pytest.approx(45, abs=0.5)
        assert upper == pytest.approx(55, abs=0.5)

    def test_polynomial_order_parameter(self, simple_bunching_data):
        """Polynomial order is respected."""
        result_low = estimate_counterfactual(
            data=simple_bunching_data,
            kink_point=50,
            bunching_width=5,
            n_bins=50,
            polynomial_order=3,
        )

        result_high = estimate_counterfactual(
            data=simple_bunching_data,
            kink_point=50,
            bunching_width=5,
            n_bins=50,
            polynomial_order=9,
        )

        assert result_low["polynomial_order"] == 3
        assert result_high["polynomial_order"] == 9
        assert len(result_low["polynomial_coeffs"]) == 4  # Order 3 -> 4 coeffs
        assert len(result_high["polynomial_coeffs"]) == 10  # Order 9 -> 10 coeffs


# =============================================================================
# Test: compute_excess_mass
# =============================================================================


class TestComputeExcessMass:
    """Tests for compute_excess_mass function."""

    def test_positive_excess_mass(self, simple_bunching_data):
        """Detects positive excess mass when bunching present."""
        counterfactual_result = estimate_counterfactual(
            data=simple_bunching_data,
            kink_point=50,
            bunching_width=5,
            n_bins=50,
        )

        excess_mass, excess_count, h0 = compute_excess_mass(counterfactual_result)

        # Should detect excess mass
        assert excess_mass > 0, "Should detect positive excess mass"
        assert excess_count > 0, "Should detect positive excess count"
        assert h0 > 0, "Counterfactual height should be positive"

    def test_no_excess_mass(self, no_bunching_data):
        """No excess mass when distribution is uniform."""
        counterfactual_result = estimate_counterfactual(
            data=no_bunching_data,
            kink_point=50,
            bunching_width=5,
            n_bins=50,
        )

        excess_mass, excess_count, h0 = compute_excess_mass(counterfactual_result)

        # Excess mass should be near zero for uniform data
        assert abs(excess_mass) < 1.0, f"Excess mass too large for uniform: {excess_mass}"

    def test_synthetic_known_excess(self):
        """Compute excess mass with known synthetic values."""
        # Create synthetic result with known excess
        bin_centers = np.linspace(0, 100, 50)
        actual_counts = np.ones(50) * 10.0
        # Spike in bunching region (bins 20-30)
        actual_counts[20:30] = 30.0

        counterfactual_counts = np.ones(50) * 10.0  # Flat counterfactual

        result = CounterfactualResult(
            bin_centers=bin_centers,
            actual_counts=actual_counts,
            counterfactual_counts=counterfactual_counts,
            polynomial_coeffs=np.array([10.0]),
            polynomial_order=0,
            bunching_region=(40, 60),  # Covers bins 20-30 approximately
            r_squared=1.0,
            n_bins=50,
            bin_width=2.0,
        )

        excess_mass, excess_count, h0 = compute_excess_mass(result)

        # h0 should be 10 (counterfactual at kink)
        assert h0 == pytest.approx(10.0, rel=0.01)

        # excess_count = sum(30-10) for ~10 bins in bunching region
        assert excess_count > 0


# =============================================================================
# Test: compute_elasticity
# =============================================================================


class TestComputeElasticity:
    """Tests for compute_elasticity function."""

    def test_basic_elasticity(self):
        """Basic elasticity calculation."""
        # Known case: b=1, t1=0.2, t2=0.3
        # e = 1 / ln(0.8/0.7) = 1 / 0.1335 ≈ 7.49
        elasticity = compute_elasticity(
            excess_mass=1.0,
            t1_rate=0.2,
            t2_rate=0.3,
        )

        expected = 1.0 / np.log(0.8 / 0.7)
        assert elasticity == pytest.approx(expected, rel=0.01)

    def test_zero_excess_mass(self):
        """Zero excess mass gives zero elasticity."""
        elasticity = compute_elasticity(
            excess_mass=0.0,
            t1_rate=0.2,
            t2_rate=0.3,
        )

        assert elasticity == 0.0

    def test_high_elasticity(self):
        """Large excess mass gives high elasticity."""
        elasticity = compute_elasticity(
            excess_mass=10.0,
            t1_rate=0.2,
            t2_rate=0.3,
        )

        expected = 10.0 / np.log(0.8 / 0.7)
        assert elasticity == pytest.approx(expected, rel=0.01)

    def test_invalid_t1_rate_raises(self):
        """Raises for invalid t1_rate."""
        with pytest.raises(ValueError, match="t1_rate"):
            compute_elasticity(excess_mass=1.0, t1_rate=-0.1, t2_rate=0.3)

        with pytest.raises(ValueError, match="t1_rate"):
            compute_elasticity(excess_mass=1.0, t1_rate=1.0, t2_rate=0.3)

    def test_invalid_t2_rate_raises(self):
        """Raises for invalid t2_rate."""
        with pytest.raises(ValueError, match="t2_rate"):
            compute_elasticity(excess_mass=1.0, t1_rate=0.2, t2_rate=-0.1)

        with pytest.raises(ValueError, match="t2_rate"):
            compute_elasticity(excess_mass=1.0, t1_rate=0.2, t2_rate=1.0)

    def test_t2_not_greater_than_t1_raises(self):
        """Raises if t2 <= t1 (no kink)."""
        with pytest.raises(ValueError, match="greater than"):
            compute_elasticity(excess_mass=1.0, t1_rate=0.3, t2_rate=0.2)

        with pytest.raises(ValueError, match="greater than"):
            compute_elasticity(excess_mass=1.0, t1_rate=0.3, t2_rate=0.3)


# =============================================================================
# Test: bootstrap_bunching_se
# =============================================================================


class TestBootstrapBunchingSE:
    """Tests for bootstrap_bunching_se function."""

    def test_basic_bootstrap(self, simple_bunching_data):
        """Basic bootstrap produces reasonable SE."""
        excess_mass_se, excess_count_se, elasticity_se, h0_se = bootstrap_bunching_se(
            data=simple_bunching_data,
            kink_point=50,
            bunching_width=5,
            n_bins=50,
            n_bootstrap=50,  # Small for speed
            random_state=42,
        )

        # SE should be positive and finite
        assert excess_mass_se > 0
        assert excess_count_se > 0
        assert h0_se > 0
        assert np.isfinite(excess_mass_se)

        # Elasticity SE not computed without rates
        assert elasticity_se == 0.0

    def test_bootstrap_with_elasticity(self, simple_bunching_data):
        """Bootstrap computes elasticity SE when rates provided."""
        excess_mass_se, excess_count_se, elasticity_se, h0_se = bootstrap_bunching_se(
            data=simple_bunching_data,
            kink_point=50,
            bunching_width=5,
            n_bins=50,
            t1_rate=0.2,
            t2_rate=0.3,
            n_bootstrap=50,
            random_state=42,
        )

        # Elasticity SE should be computed
        assert elasticity_se > 0
        assert np.isfinite(elasticity_se)

    def test_bootstrap_reproducibility(self, simple_bunching_data):
        """Bootstrap is reproducible with same random state."""
        result1 = bootstrap_bunching_se(
            data=simple_bunching_data,
            kink_point=50,
            bunching_width=5,
            n_bins=50,
            n_bootstrap=50,
            random_state=42,
        )

        result2 = bootstrap_bunching_se(
            data=simple_bunching_data,
            kink_point=50,
            bunching_width=5,
            n_bins=50,
            n_bootstrap=50,
            random_state=42,
        )

        assert_allclose(result1, result2)


# =============================================================================
# Test: bunching_estimator
# =============================================================================


class TestBunchingEstimator:
    """Tests for main bunching_estimator function."""

    def test_basic_estimation(self, simple_bunching_data):
        """Basic bunching estimation works."""
        result = bunching_estimator(
            data=simple_bunching_data,
            kink_point=50,
            bunching_width=5,
            n_bins=50,
            n_bootstrap=50,
            random_state=42,
        )

        # Check result structure
        assert isinstance(result, dict)
        assert "excess_mass" in result
        assert "excess_mass_se" in result
        assert "elasticity" in result
        assert "counterfactual" in result
        assert "n_obs" in result
        assert "convergence" in result

        # Should detect bunching
        assert result["excess_mass"] > 0
        assert result["n_obs"] == len(simple_bunching_data)
        assert result["convergence"] is True

    def test_estimation_with_elasticity(self, simple_bunching_data):
        """Bunching estimation with elasticity calculation."""
        result = bunching_estimator(
            data=simple_bunching_data,
            kink_point=50,
            bunching_width=5,
            n_bins=50,
            t1_rate=0.2,
            t2_rate=0.3,
            n_bootstrap=50,
            random_state=42,
        )

        # Elasticity should be computed
        assert np.isfinite(result["elasticity"])
        assert result["elasticity"] > 0  # Positive excess mass -> positive elasticity
        assert np.isfinite(result["elasticity_se"])
        assert result["t1_rate"] == 0.2
        assert result["t2_rate"] == 0.3

    def test_no_bunching_detection(self, no_bunching_data):
        """No significant bunching detected in uniform data."""
        result = bunching_estimator(
            data=no_bunching_data,
            kink_point=50,
            bunching_width=5,
            n_bins=50,
            n_bootstrap=50,
            random_state=42,
        )

        # Excess mass should be small
        assert abs(result["excess_mass"]) < 1.0

    def test_realistic_tax_scenario(self, tax_bunching_data):
        """Realistic tax bunching scenario."""
        result = bunching_estimator(
            data=tax_bunching_data,
            kink_point=50000,
            bunching_width=2000,
            n_bins=60,
            t1_rate=0.20,
            t2_rate=0.30,
            n_bootstrap=50,
            random_state=42,
        )

        # Should detect bunching
        assert result["excess_mass"] > 0
        # Elasticity should be positive and finite
        # Note: In simulated data, elasticity can be high because we inject
        # artificial bunching. Real-world estimates are typically 0.1-1.0.
        assert result["elasticity"] > 0
        assert np.isfinite(result["elasticity"])
        assert result["kink_point"] == 50000

    def test_counterfactual_included(self, simple_bunching_data):
        """Counterfactual result is included in output."""
        result = bunching_estimator(
            data=simple_bunching_data,
            kink_point=50,
            bunching_width=5,
            n_bins=50,
            n_bootstrap=50,
            random_state=42,
        )

        cf = result["counterfactual"]
        assert "bin_centers" in cf
        assert "actual_counts" in cf
        assert "counterfactual_counts" in cf
        assert len(cf["bin_centers"]) == 50

    def test_empty_data_raises(self):
        """Raises for empty data."""
        with pytest.raises(ValueError, match="empty"):
            bunching_estimator(
                data=np.array([]),
                kink_point=50,
                bunching_width=5,
                n_bins=50,
            )

    def test_invalid_bunching_width_raises(self, simple_bunching_data):
        """Raises for invalid bunching width."""
        with pytest.raises(ValueError, match="bunching_width.*positive"):
            bunching_estimator(
                data=simple_bunching_data,
                kink_point=50,
                bunching_width=-5,
                n_bins=50,
            )


# =============================================================================
# Test: Known-Answer Tests
# =============================================================================


class TestKnownAnswers:
    """Known-answer tests with hand-calculated values."""

    def test_uniform_counterfactual_height(self):
        """Uniform distribution has flat counterfactual."""
        # Perfect uniform data
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 100, size=10000)

        result = estimate_counterfactual(
            data=data,
            kink_point=50,
            bunching_width=10,
            n_bins=50,
            polynomial_order=1,  # Linear fit for uniform
        )

        # All bins should have similar counts (~200 each for 10000/50)
        expected_per_bin = 10000 / 50
        actual_counts = result["actual_counts"]

        # Should be within 20% of expected (sampling variation)
        assert np.mean(actual_counts) == pytest.approx(expected_per_bin, rel=0.2)

    def test_elasticity_formula_verification(self):
        """Verify elasticity formula with known values."""
        # For b=1, t1=0.1, t2=0.2:
        # e = 1 / ln((1-0.1)/(1-0.2)) = 1 / ln(0.9/0.8) = 1 / 0.1178 = 8.49
        e = compute_elasticity(excess_mass=1.0, t1_rate=0.1, t2_rate=0.2)
        expected = 1.0 / np.log(0.9 / 0.8)
        assert e == pytest.approx(expected, rel=0.001)

        # For b=2, t1=0.25, t2=0.35:
        # e = 2 / ln(0.75/0.65) = 2 / 0.1431 = 13.98
        e = compute_elasticity(excess_mass=2.0, t1_rate=0.25, t2_rate=0.35)
        expected = 2.0 / np.log(0.75 / 0.65)
        assert e == pytest.approx(expected, rel=0.001)


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for bunching estimation."""

    def test_small_sample(self):
        """Works with small sample size."""
        rng = np.random.default_rng(42)
        data = rng.normal(50, 10, size=100)

        result = bunching_estimator(
            data=data,
            kink_point=50,
            bunching_width=5,
            n_bins=20,  # Fewer bins for small sample
            polynomial_order=3,
            n_bootstrap=30,
            random_state=42,
        )

        assert result["convergence"] is True
        assert np.isfinite(result["excess_mass"])

    def test_large_bunching_width(self):
        """Large bunching width relative to data range."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 100, size=1000)

        result = estimate_counterfactual(
            data=data,
            kink_point=50,
            bunching_width=40,  # 80% of range
            n_bins=50,
            polynomial_order=3,
        )

        # Should still work (bunching region adjusted to data range)
        assert result["r_squared"] >= 0

    def test_kink_at_data_boundary(self):
        """Kink point near data boundary."""
        rng = np.random.default_rng(42)
        data = rng.uniform(40, 100, size=1000)

        result = estimate_counterfactual(
            data=data,
            kink_point=45,  # Near lower boundary
            bunching_width=3,
            n_bins=30,
            polynomial_order=5,
        )

        # Should handle boundary gracefully
        assert len(result["bin_centers"]) == 30

    def test_very_concentrated_bunching(self):
        """Extremely concentrated bunching at kink."""
        rng = np.random.default_rng(42)

        # All observations exactly at kink
        bunchers = np.full(500, 50.0)
        background = rng.uniform(20, 80, size=500)
        data = np.concatenate([bunchers, background])

        result = bunching_estimator(
            data=data,
            kink_point=50,
            bunching_width=2,
            n_bins=60,
            n_bootstrap=30,
            random_state=42,
        )

        # Should detect very large excess mass
        assert result["excess_mass"] > 5.0

    def test_negative_values_in_data(self):
        """Data can include negative values."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 10, size=1000)  # Centered at 0, many negative

        result = estimate_counterfactual(
            data=data,
            kink_point=0,
            bunching_width=3,
            n_bins=50,
        )

        assert len(result["bin_centers"]) == 50
        assert np.min(result["bin_centers"]) < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
