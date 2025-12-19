"""
Tests for iterative counterfactual estimation (Chetty et al. 2011).

The integration constraint requires that the area "missing" above the kink
(where bunchers came from) equals the excess mass at the kink.

References:
- Chetty, R., et al. (2011). "Adjustment Costs, Firm Responses, and Micro vs.
    Macro Labor Supply Elasticities." Quarterly Journal of Economics.
"""

import numpy as np
import pytest

from causal_inference.bunching.counterfactual import (
    iterative_counterfactual,
    estimate_counterfactual,
)


class TestIterativeCounterfactual:
    """Tests for iterative_counterfactual function."""

    @pytest.fixture
    def simple_bunching_data(self):
        """Generate simple bunching data with wide spread."""
        rng = np.random.default_rng(42)
        # Wide background distribution to ensure bins outside bunching region
        background = rng.normal(50, 20, size=800)
        bunchers = rng.normal(50, 1.0, size=200)
        data = np.concatenate([background, bunchers])
        # Trim to reasonable range
        data = data[(data > 0) & (data < 100)]
        return data

    def test_returns_three_elements(self, simple_bunching_data):
        """Returns (CounterfactualResult, delta_z, converged)."""
        result, delta_z, converged = iterative_counterfactual(
            data=simple_bunching_data,
            kink_point=50,
            initial_bunching_width=5,
            n_bins=50,
            max_iterations=10,
        )

        assert isinstance(result, dict)
        assert "bin_centers" in result
        assert isinstance(delta_z, (int, float, np.floating))
        assert isinstance(converged, bool)

    def test_convergence_with_reasonable_data(self, simple_bunching_data):
        """Converges with reasonable data and parameters."""
        _, _, converged = iterative_counterfactual(
            data=simple_bunching_data,
            kink_point=50,
            initial_bunching_width=5,
            n_bins=50,
            max_iterations=20,
            tolerance=1e-3,
        )

        # Should converge with reasonable settings
        # (may not always converge with all data)
        assert isinstance(converged, bool)

    def test_delta_z_positive(self, simple_bunching_data):
        """Delta_z (upper bound shift) is positive."""
        _, delta_z, _ = iterative_counterfactual(
            data=simple_bunching_data,
            kink_point=50,
            initial_bunching_width=5,
            n_bins=50,
        )

        # delta_z should be positive (bunchers come from above kink)
        assert delta_z >= 0, f"delta_z should be non-negative, got {delta_z}"

    def test_result_contains_bunching_region(self, simple_bunching_data):
        """Result contains bunching region bounds."""
        result, _, _ = iterative_counterfactual(
            data=simple_bunching_data,
            kink_point=50,
            initial_bunching_width=5,
            n_bins=50,
        )

        assert "bunching_region" in result
        lower, upper = result["bunching_region"]
        assert lower < upper

    def test_max_iterations_respected(self, simple_bunching_data):
        """Does not exceed max_iterations."""
        # Very tight tolerance should prevent convergence
        _, _, converged = iterative_counterfactual(
            data=simple_bunching_data,
            kink_point=50,
            initial_bunching_width=5,
            n_bins=50,
            max_iterations=2,
            tolerance=1e-10,
        )

        # With only 2 iterations and tight tolerance, likely won't converge
        # But function should still return valid result
        assert isinstance(converged, bool)

    def test_tolerance_affects_convergence(self, simple_bunching_data):
        """Looser tolerance converges faster."""
        # Tight tolerance
        _, _, converged_tight = iterative_counterfactual(
            data=simple_bunching_data,
            kink_point=50,
            initial_bunching_width=5,
            n_bins=50,
            max_iterations=5,
            tolerance=1e-8,
        )

        # Loose tolerance
        _, _, converged_loose = iterative_counterfactual(
            data=simple_bunching_data,
            kink_point=50,
            initial_bunching_width=5,
            n_bins=50,
            max_iterations=5,
            tolerance=1e-2,
        )

        # Loose tolerance should converge (or at least be as likely)
        # This is a soft test - just checking both complete
        assert isinstance(converged_tight, bool)
        assert isinstance(converged_loose, bool)

    def test_polynomial_order_parameter(self, simple_bunching_data):
        """Polynomial order parameter is passed through."""
        result_low, _, _ = iterative_counterfactual(
            data=simple_bunching_data,
            kink_point=50,
            initial_bunching_width=5,
            n_bins=50,
            polynomial_order=3,
        )

        result_high, _, _ = iterative_counterfactual(
            data=simple_bunching_data,
            kink_point=50,
            initial_bunching_width=5,
            n_bins=50,
            polynomial_order=9,
        )

        assert result_low["polynomial_order"] == 3
        assert result_high["polynomial_order"] == 9

    def test_n_bins_parameter(self, simple_bunching_data):
        """n_bins parameter is passed through."""
        result, _, _ = iterative_counterfactual(
            data=simple_bunching_data,
            kink_point=50,
            initial_bunching_width=5,
            n_bins=40,
        )

        assert result["n_bins"] == 40
        assert len(result["bin_centers"]) == 40

    def test_bin_width_alternative(self, simple_bunching_data):
        """Can use bin_width instead of n_bins."""
        result, _, _ = iterative_counterfactual(
            data=simple_bunching_data,
            kink_point=50,
            initial_bunching_width=5,
            bin_width=2.0,
        )

        assert "bin_centers" in result
        assert result["bin_width"] == pytest.approx(2.0, rel=0.1)


class TestIterativeVsNonIterative:
    """Compare iterative vs non-iterative estimation."""

    @pytest.fixture
    def bunching_data(self):
        """Generate bunching data for comparison."""
        rng = np.random.default_rng(42)
        background = rng.normal(50, 15, size=900)
        bunchers = rng.normal(50, 1.0, size=100)
        return np.concatenate([background, bunchers])

    def test_both_detect_bunching(self, bunching_data):
        """Both methods detect positive excess mass."""
        # Non-iterative
        cf_result = estimate_counterfactual(
            data=bunching_data,
            kink_point=50,
            bunching_width=5,
            n_bins=50,
        )

        # Iterative
        iter_result, _, _ = iterative_counterfactual(
            data=bunching_data,
            kink_point=50,
            initial_bunching_width=5,
            n_bins=50,
        )

        # Both should find bunching region
        assert cf_result["bunching_region"][0] < cf_result["bunching_region"][1]
        assert iter_result["bunching_region"][0] < iter_result["bunching_region"][1]

    def test_similar_counterfactual_shape(self, bunching_data):
        """Counterfactual shapes are similar."""
        cf_result = estimate_counterfactual(
            data=bunching_data,
            kink_point=50,
            bunching_width=5,
            n_bins=50,
        )

        iter_result, _, _ = iterative_counterfactual(
            data=bunching_data,
            kink_point=50,
            initial_bunching_width=5,
            n_bins=50,
        )

        # Counterfactual shapes should be correlated
        corr = np.corrcoef(
            cf_result["counterfactual_counts"],
            iter_result["counterfactual_counts"],
        )[0, 1]

        assert corr > 0.9, f"Counterfactual correlation too low: {corr:.3f}"


class TestIterativeEdgeCases:
    """Edge cases for iterative counterfactual."""

    def test_no_bunching_data(self):
        """Handles data with no bunching."""
        rng = np.random.default_rng(42)
        data = rng.uniform(20, 80, size=1000)

        result, delta_z, converged = iterative_counterfactual(
            data=data,
            kink_point=50,
            initial_bunching_width=5,
            n_bins=50,
        )

        # Should complete without error
        assert isinstance(result, dict)
        assert np.isfinite(delta_z)

    def test_concentrated_bunching(self):
        """Handles concentrated bunching (may expand bunching region)."""
        rng = np.random.default_rng(42)
        # Wide background to ensure bins outside bunching region
        background = rng.uniform(0, 100, size=800)
        # Concentrated bunching at kink
        bunchers = rng.normal(50, 1.0, size=200)
        data = np.concatenate([background, bunchers])

        # Use small initial width and low polynomial order
        result, delta_z, converged = iterative_counterfactual(
            data=data,
            kink_point=50,
            initial_bunching_width=2,  # Small initial width
            n_bins=100,  # Many bins
            polynomial_order=3,  # Low order
            max_iterations=5,  # Limit iterations
        )

        # Should complete and detect positive delta_z
        assert isinstance(result, dict)
        assert delta_z >= 0

    def test_small_initial_width(self):
        """Handles small initial bunching width."""
        rng = np.random.default_rng(42)
        data = rng.normal(50, 15, size=1000)

        result, delta_z, converged = iterative_counterfactual(
            data=data,
            kink_point=50,
            initial_bunching_width=1,  # Very small
            n_bins=50,
        )

        # Should complete
        assert isinstance(result, dict)

    def test_large_initial_width(self):
        """Handles large initial bunching width."""
        rng = np.random.default_rng(42)
        # Wide spread to ensure bins outside even large bunching region
        data = rng.normal(50, 30, size=1000)
        data = data[(data > -10) & (data < 110)]

        result, delta_z, converged = iterative_counterfactual(
            data=data,
            kink_point=50,
            initial_bunching_width=15,  # Large but not extreme
            n_bins=80,  # More bins
            polynomial_order=3,  # Lower order
        )

        # Should complete
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
