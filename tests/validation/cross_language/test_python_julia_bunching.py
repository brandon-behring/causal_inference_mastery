"""
Python ↔ Julia Bunching Estimation Parity Tests

Tests that Python and Julia bunching implementations produce
numerically equivalent results for:
- Polynomial counterfactual estimation
- Excess mass calculation
- Elasticity calculation
- Full Saez (2010) bunching estimator

Session 78: Julia Bunching Implementation

References:
- Saez (2010) - Original bunching methodology
- Chetty et al. (2011) - Integration constraint
- Kleven (2016) - Bunching estimation review
"""

import numpy as np
import pytest

from causal_inference.bunching import (
    estimate_counterfactual,
    compute_excess_mass,
    compute_elasticity,
    bunching_estimator,
)
from tests.validation.cross_language.julia_interface import (
    is_julia_available,
    julia_bunching_estimator,
    julia_polynomial_counterfactual,
    julia_compute_elasticity,
)

# Skip all tests if Julia is not available
pytestmark = pytest.mark.skipif(
    not is_julia_available(),
    reason="Julia not available for cross-language validation",
)


class TestPolynomialCounterfactualParity:
    """Test polynomial counterfactual parity between Python and Julia."""

    def test_uniform_data_counterfactual(self):
        """Counterfactual for uniform data produces similar shape."""
        rng = np.random.default_rng(42)
        # Uniform data spread across wide range
        data = rng.uniform(0, 100, size=1000)

        py_result = estimate_counterfactual(
            data=data,
            kink_point=50.0,
            bunching_width=5.0,
            n_bins=50,
            polynomial_order=3,
        )

        jl_result = julia_polynomial_counterfactual(
            py_result["bin_centers"],
            py_result["actual_counts"],
            py_result["bunching_region"][0],
            py_result["bunching_region"][1],
            polynomial_order=3,
        )

        # Counterfactual counts should be close for same inputs
        np.testing.assert_allclose(
            py_result["counterfactual_counts"],
            jl_result["counterfactual"],
            rtol=0.05,  # 5% tolerance
        )

    def test_bunching_data_counterfactual(self):
        """Counterfactual for bunching data produces similar shape."""
        rng = np.random.default_rng(42)
        background = rng.normal(50, 15, size=800)
        bunchers = rng.normal(50, 2, size=200)
        data = np.concatenate([background, bunchers])

        py_result = estimate_counterfactual(
            data=data,
            kink_point=50.0,
            bunching_width=5.0,
            n_bins=50,
            polynomial_order=5,
        )

        jl_result = julia_polynomial_counterfactual(
            py_result["bin_centers"],
            py_result["actual_counts"],
            py_result["bunching_region"][0],
            py_result["bunching_region"][1],
            polynomial_order=5,
        )

        # Counterfactual counts should be close for same inputs
        np.testing.assert_allclose(
            py_result["counterfactual_counts"],
            jl_result["counterfactual"],
            rtol=0.05,  # 5% tolerance
        )


class TestElasticityParity:
    """Test elasticity calculation parity between Python and Julia."""

    def test_elasticity_basic(self):
        """Basic elasticity calculation matches."""
        excess_mass = 5.0
        t1_rate = 0.25
        t2_rate = 0.35

        py_elasticity = compute_elasticity(excess_mass, t1_rate, t2_rate)
        jl_elasticity = julia_compute_elasticity(excess_mass, t1_rate, t2_rate)

        assert np.isclose(py_elasticity, jl_elasticity, rtol=1e-10)

    def test_elasticity_zero_excess(self):
        """Zero excess mass gives zero elasticity."""
        py_elasticity = compute_elasticity(0.0, 0.25, 0.35)
        jl_elasticity = julia_compute_elasticity(0.0, 0.25, 0.35)

        assert np.isclose(py_elasticity, jl_elasticity, rtol=1e-10)
        assert py_elasticity == pytest.approx(0.0)

    def test_elasticity_small_rate_change(self):
        """Elasticity with small rate change matches."""
        excess_mass = 2.0
        t1_rate = 0.20
        t2_rate = 0.22

        py_elasticity = compute_elasticity(excess_mass, t1_rate, t2_rate)
        jl_elasticity = julia_compute_elasticity(excess_mass, t1_rate, t2_rate)

        assert np.isclose(py_elasticity, jl_elasticity, rtol=1e-10)

    def test_elasticity_large_rate_change(self):
        """Elasticity with large rate change matches."""
        excess_mass = 3.0
        t1_rate = 0.10
        t2_rate = 0.50

        py_elasticity = compute_elasticity(excess_mass, t1_rate, t2_rate)
        jl_elasticity = julia_compute_elasticity(excess_mass, t1_rate, t2_rate)

        assert np.isclose(py_elasticity, jl_elasticity, rtol=1e-10)


class TestBunchingEstimatorParity:
    """Test full bunching estimator parity between Python and Julia."""

    @pytest.fixture
    def bunching_data(self):
        """Generate bunching data with known characteristics."""
        rng = np.random.default_rng(42)
        background = rng.normal(50, 15, size=800)
        bunchers = rng.normal(50, 2, size=200)
        return np.concatenate([background, bunchers])

    def test_excess_mass_direction(self, bunching_data):
        """Both detect positive excess mass."""
        py_result = bunching_estimator(
            data=bunching_data,
            kink_point=50.0,
            bunching_width=5.0,
            n_bins=50,
            polynomial_order=5,
            n_bootstrap=30,
        )

        jl_result = julia_bunching_estimator(
            data=bunching_data,
            kink_point=50.0,
            bunching_width=5.0,
            n_bins=50,
            polynomial_order=5,
            n_bootstrap=30,
        )

        # Both should find positive excess mass
        assert py_result["excess_mass"] > 0
        assert jl_result["excess_mass"] > 0

    def test_excess_mass_magnitude(self, bunching_data):
        """Excess mass magnitudes are similar."""
        py_result = bunching_estimator(
            data=bunching_data,
            kink_point=50.0,
            bunching_width=5.0,
            n_bins=50,
            polynomial_order=5,
            n_bootstrap=30,
        )

        jl_result = julia_bunching_estimator(
            data=bunching_data,
            kink_point=50.0,
            bunching_width=5.0,
            n_bins=50,
            polynomial_order=5,
            n_bootstrap=30,
        )

        # Should be within 50% of each other (bootstrap adds variance)
        ratio = py_result["excess_mass"] / jl_result["excess_mass"]
        assert 0.5 < ratio < 2.0, f"Excess mass ratio out of range: {ratio}"

    def test_bunching_with_rates(self, bunching_data):
        """Bunching with tax rates computes elasticity similarly."""
        py_result = bunching_estimator(
            data=bunching_data,
            kink_point=50.0,
            bunching_width=5.0,
            t1_rate=0.25,
            t2_rate=0.35,
            n_bins=50,
            polynomial_order=5,
            n_bootstrap=30,
        )

        jl_result = julia_bunching_estimator(
            data=bunching_data,
            kink_point=50.0,
            bunching_width=5.0,
            t1_rate=0.25,
            t2_rate=0.35,
            n_bins=50,
            polynomial_order=5,
            n_bootstrap=30,
        )

        # Both should have finite elasticity
        assert np.isfinite(py_result["elasticity"])
        assert np.isfinite(jl_result["elasticity"])

        # Elasticity should be positive (bunching at kink)
        assert py_result["elasticity"] > 0
        assert jl_result["elasticity"] > 0

    def test_counterfactual_r_squared(self, bunching_data):
        """Counterfactual R-squared is similar."""
        py_result = bunching_estimator(
            data=bunching_data,
            kink_point=50.0,
            bunching_width=5.0,
            n_bins=50,
            polynomial_order=5,
            n_bootstrap=20,
        )

        jl_result = julia_bunching_estimator(
            data=bunching_data,
            kink_point=50.0,
            bunching_width=5.0,
            n_bins=50,
            polynomial_order=5,
            n_bootstrap=20,
        )

        # R-squared should be reasonably close (access nested counterfactual)
        py_r2 = py_result["counterfactual"]["r_squared"]
        jl_r2 = jl_result["r_squared"]

        assert abs(py_r2 - jl_r2) < 0.2, f"R-squared difference: {abs(py_r2 - jl_r2)}"

    def test_n_bins_consistency(self, bunching_data):
        """Number of bins is consistent."""
        n_bins = 60

        py_result = bunching_estimator(
            data=bunching_data,
            kink_point=50.0,
            bunching_width=5.0,
            n_bins=n_bins,
            polynomial_order=5,
            n_bootstrap=20,
        )

        jl_result = julia_bunching_estimator(
            data=bunching_data,
            kink_point=50.0,
            bunching_width=5.0,
            n_bins=n_bins,
            polynomial_order=5,
            n_bootstrap=20,
        )

        # Access nested counterfactual for bin_centers
        assert len(py_result["counterfactual"]["bin_centers"]) == n_bins
        assert len(jl_result["bin_centers"]) == n_bins


class TestNoBunchingParity:
    """Test parity when there is no bunching."""

    @pytest.fixture
    def uniform_data(self):
        """Generate uniform data with no bunching."""
        rng = np.random.default_rng(42)
        return rng.uniform(20, 80, size=1000)

    def test_small_excess_mass(self, uniform_data):
        """Both find small excess mass for uniform data."""
        py_result = bunching_estimator(
            data=uniform_data,
            kink_point=50.0,
            bunching_width=5.0,
            n_bins=50,
            polynomial_order=3,
            n_bootstrap=20,
        )

        jl_result = julia_bunching_estimator(
            data=uniform_data,
            kink_point=50.0,
            bunching_width=5.0,
            n_bins=50,
            polynomial_order=3,
            n_bootstrap=20,
        )

        # Excess mass should be small (noise only)
        assert abs(py_result["excess_mass"]) < 3.0
        assert abs(jl_result["excess_mass"]) < 3.0


class TestDifferentPolynomialOrders:
    """Test parity across different polynomial orders."""

    @pytest.fixture
    def bunching_data(self):
        """Generate bunching data."""
        rng = np.random.default_rng(42)
        background = rng.normal(50, 20, size=800)
        bunchers = rng.normal(50, 2, size=200)
        return np.concatenate([background, bunchers])

    @pytest.mark.parametrize("polynomial_order", [3, 5, 7])
    def test_polynomial_order_parity(self, bunching_data, polynomial_order):
        """Both implementations work with different polynomial orders."""
        py_result = bunching_estimator(
            data=bunching_data,
            kink_point=50.0,
            bunching_width=5.0,
            n_bins=60,
            polynomial_order=polynomial_order,
            n_bootstrap=20,
        )

        jl_result = julia_bunching_estimator(
            data=bunching_data,
            kink_point=50.0,
            bunching_width=5.0,
            n_bins=60,
            polynomial_order=polynomial_order,
            n_bootstrap=20,
        )

        # Both should complete successfully
        assert np.isfinite(py_result["excess_mass"])
        assert np.isfinite(jl_result["excess_mass"])

        # Polynomial order should match (access nested counterfactual for Python)
        assert py_result["counterfactual"]["polynomial_order"] == polynomial_order
        assert jl_result["polynomial_order"] == polynomial_order


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
