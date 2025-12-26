"""Triangulation tests: Python Principal Stratification vs R implementations.

This module provides Layer 5 validation by comparing our Python implementations
against R packages (PStrata, AER) and manual R implementations.

Tests skip gracefully when R/rpy2 is unavailable.

Tolerance levels (established in plan):
- 2SLS CACE: rtol=0.05 (deterministic, should match closely)
- EM CACE: rtol=0.10 (stochastic, local optima possible)
- Strata proportions: rtol=0.15 (EM local optima)
- Bounds: rtol=0.10

Run with: pytest tests/validation/r_triangulation/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.validation.r_triangulation.r_interface import (
    check_pstrata_installed,
    check_r_available,
    r_bounds_manski,
    r_cace_2sls,
    r_cace_em_manual,
    r_cace_pstrata,
)

# Lazy import to avoid errors when principal_stratification not fully implemented
try:
    from src.causal_inference.principal_stratification import (
        cace_2sls,
        cace_em,
        ps_bounds_monotonicity,
        ps_bounds_no_assumption,
    )

    PS_AVAILABLE = True
except ImportError:
    PS_AVAILABLE = False


# =============================================================================
# Skip conditions
# =============================================================================

# Skip all tests in this module if R/rpy2 not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R/rpy2 not available for triangulation tests",
)

requires_ps = pytest.mark.skipif(
    not PS_AVAILABLE,
    reason="principal_stratification module not available",
)

requires_pstrata = pytest.mark.skipif(
    not check_pstrata_installed(),
    reason="PStrata R package not installed",
)


# =============================================================================
# Test Data Generator
# =============================================================================


def generate_ps_dgp(
    n: int = 1000,
    pi_c: float = 0.60,
    pi_a: float = 0.20,
    pi_n: float = 0.20,
    true_cace: float = 2.0,
    baseline: float = 1.0,
    noise_sd: float = 1.0,
    seed: int = 42,
) -> dict:
    """Generate data from principal stratification DGP.

    Parameters
    ----------
    n : int
        Sample size.
    pi_c, pi_a, pi_n : float
        True proportions of compliers, always-takers, never-takers.
    true_cace : float
        True complier average causal effect.
    baseline : float
        Baseline outcome level.
    noise_sd : float
        Outcome noise standard deviation.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with Y, D, Z arrays and true parameters.
    """
    np.random.seed(seed)

    # Normalize proportions
    total = pi_c + pi_a + pi_n
    pi_c, pi_a, pi_n = pi_c / total, pi_a / total, pi_n / total

    # Random instrument assignment
    Z = np.random.binomial(1, 0.5, n).astype(float)

    # Assign strata
    strata = np.random.choice([0, 1, 2], size=n, p=[pi_c, pi_a, pi_n])

    # Treatment based on strata and instrument
    D = np.zeros(n)
    D[strata == 0] = Z[strata == 0]  # Compliers follow Z
    D[strata == 1] = 1.0  # Always-takers always treated
    D[strata == 2] = 0.0  # Never-takers never treated

    # Outcome: effect only for treated compliers
    # Y = baseline + CACE * D * (stratum == complier) + noise
    # Simplified: Y = baseline + CACE * D + noise (assuming exclusion restriction)
    Y = baseline + true_cace * D + noise_sd * np.random.randn(n)

    return {
        "Y": Y,
        "D": D,
        "Z": Z,
        "true_cace": true_cace,
        "true_pi_c": pi_c,
        "true_pi_a": pi_a,
        "true_pi_n": pi_n,
        "strata": strata,
    }


# =============================================================================
# Layer 5: CACE 2SLS Triangulation
# =============================================================================


@requires_ps
class TestCACE2SLSVsR:
    """Compare Python cace_2sls() to R 2SLS implementations."""

    def test_2sls_point_estimate_parity(self):
        """Python 2SLS CACE should match R 2SLS within rtol=0.05."""
        data = generate_ps_dgp(n=1000, seed=42)

        # Python result
        py_result = cace_2sls(data["Y"], data["D"], data["Z"])

        # R result (using AER::ivreg or manual 2SLS)
        r_result = r_cace_2sls(data["Y"], data["D"], data["Z"])

        assert np.isclose(py_result["cace"], r_result["cace"], rtol=0.05), (
            f"Python CACE={py_result['cace']:.4f} vs R CACE={r_result['cace']:.4f}"
        )

    def test_2sls_se_parity(self):
        """Python 2SLS SE should match R 2SLS SE within rtol=0.10."""
        data = generate_ps_dgp(n=1000, seed=123)

        py_result = cace_2sls(data["Y"], data["D"], data["Z"])
        r_result = r_cace_2sls(data["Y"], data["D"], data["Z"])

        # SE comparison (slightly looser tolerance)
        assert np.isclose(py_result["se"], r_result["se"], rtol=0.10), (
            f"Python SE={py_result['se']:.4f} vs R SE={r_result['se']:.4f}"
        )

    def test_2sls_ci_parity(self):
        """Python 2SLS CI should match R 2SLS CI reasonably."""
        data = generate_ps_dgp(n=1000, seed=456)

        py_result = cace_2sls(data["Y"], data["D"], data["Z"])
        r_result = r_cace_2sls(data["Y"], data["D"], data["Z"])

        # CI bounds comparison
        assert np.isclose(py_result["ci_lower"], r_result["ci_lower"], rtol=0.10), (
            f"CI lower: Python={py_result['ci_lower']:.4f} vs R={r_result['ci_lower']:.4f}"
        )
        assert np.isclose(py_result["ci_upper"], r_result["ci_upper"], rtol=0.10), (
            f"CI upper: Python={py_result['ci_upper']:.4f} vs R={r_result['ci_upper']:.4f}"
        )

    def test_2sls_multiple_seeds(self):
        """2SLS parity should hold across multiple random seeds."""
        for seed in [1, 42, 123, 456, 789]:
            data = generate_ps_dgp(n=500, seed=seed)

            py_result = cace_2sls(data["Y"], data["D"], data["Z"])
            r_result = r_cace_2sls(data["Y"], data["D"], data["Z"])

            assert np.isclose(py_result["cace"], r_result["cace"], rtol=0.05), (
                f"Seed {seed}: Python CACE={py_result['cace']:.4f} vs R={r_result['cace']:.4f}"
            )


# =============================================================================
# Layer 5: CACE EM Triangulation
# =============================================================================


@requires_ps
class TestCACEEMVsR:
    """Compare Python cace_em() to R EM implementations."""

    def test_em_cace_vs_manual_r(self):
        """Python EM CACE should match manual R EM within rtol=0.10."""
        data = generate_ps_dgp(n=1000, pi_c=0.6, pi_a=0.2, pi_n=0.2, seed=42)

        # Python EM
        py_result = cace_em(data["Y"], data["D"], data["Z"])

        # Manual R EM
        r_result = r_cace_em_manual(data["Y"], data["D"], data["Z"])

        # CACE should be similar (allow for local optima)
        assert np.isclose(py_result["cace"], r_result["cace"], rtol=0.10), (
            f"Python CACE={py_result['cace']:.4f} vs R CACE={r_result['cace']:.4f}"
        )

    def test_em_strata_proportions_vs_r(self):
        """EM strata proportions should match R within rtol=0.15."""
        data = generate_ps_dgp(n=1000, pi_c=0.5, pi_a=0.3, pi_n=0.2, seed=42)

        py_result = cace_em(data["Y"], data["D"], data["Z"])
        r_result = r_cace_em_manual(data["Y"], data["D"], data["Z"])

        # Compare strata proportions
        py_props = py_result["strata_proportions"]
        r_props = r_result["strata_proportions"]

        assert np.isclose(py_props["compliers"], r_props["compliers"], rtol=0.15), (
            f"pi_c: Python={py_props['compliers']:.3f} vs R={r_props['compliers']:.3f}"
        )
        assert np.isclose(py_props["always_takers"], r_props["always_takers"], rtol=0.15), (
            f"pi_a: Python={py_props['always_takers']:.3f} vs R={r_props['always_takers']:.3f}"
        )
        assert np.isclose(py_props["never_takers"], r_props["never_takers"], rtol=0.15), (
            f"pi_n: Python={py_props['never_takers']:.3f} vs R={r_props['never_takers']:.3f}"
        )

    def test_em_convergence_agreement(self):
        """Both Python and R EM should converge on same data."""
        data = generate_ps_dgp(n=1000, seed=42)

        py_result = cace_em(data["Y"], data["D"], data["Z"])
        r_result = r_cace_em_manual(data["Y"], data["D"], data["Z"])

        # Both should converge
        assert py_result["converged"], "Python EM did not converge"
        assert r_result["converged"], "R EM did not converge"

    @requires_pstrata
    def test_em_cace_vs_pstrata(self):
        """Python EM should match PStrata package within rtol=0.10."""
        data = generate_ps_dgp(n=1000, seed=42)

        py_result = cace_em(data["Y"], data["D"], data["Z"])
        r_result = r_cace_pstrata(data["Y"], data["D"], data["Z"], method="EM")

        if r_result is None:
            pytest.skip("PStrata call failed")

        assert np.isclose(py_result["cace"], r_result["cace"], rtol=0.10), (
            f"Python CACE={py_result['cace']:.4f} vs PStrata={r_result['cace']:.4f}"
        )

    @requires_pstrata
    def test_strata_vs_pstrata(self):
        """Strata proportions should match PStrata within rtol=0.15."""
        data = generate_ps_dgp(n=1000, seed=42)

        py_result = cace_em(data["Y"], data["D"], data["Z"])
        r_result = r_cace_pstrata(data["Y"], data["D"], data["Z"])

        if r_result is None:
            pytest.skip("PStrata call failed")

        py_props = py_result["strata_proportions"]
        r_props = r_result["strata_proportions"]

        for stratum in ["compliers", "always_takers", "never_takers"]:
            assert np.isclose(py_props[stratum], r_props[stratum], rtol=0.15), (
                f"{stratum}: Python={py_props[stratum]:.3f} vs PStrata={r_props[stratum]:.3f}"
            )


# =============================================================================
# Layer 5: Bounds Triangulation
# =============================================================================


@requires_ps
class TestBoundsVsR:
    """Compare Python bounds functions to R implementations."""

    def test_manski_bounds_parity(self):
        """Manski-style bounds should match R within rtol=0.10."""
        data = generate_ps_dgp(n=1000, seed=42)

        # Python no-assumption bounds
        py_result = ps_bounds_no_assumption(data["Y"], data["D"], data["Z"])

        # R Manski bounds
        r_result = r_bounds_manski(data["Y"], data["D"], data["Z"])

        # Bounds should match
        assert np.isclose(py_result["lower_bound"], r_result["lower_bound"], rtol=0.10), (
            f"Lower: Python={py_result['lower_bound']:.4f} vs R={r_result['lower_bound']:.4f}"
        )
        assert np.isclose(py_result["upper_bound"], r_result["upper_bound"], rtol=0.10), (
            f"Upper: Python={py_result['upper_bound']:.4f} vs R={r_result['upper_bound']:.4f}"
        )

    def test_bounds_width_parity(self):
        """Bound width should match R."""
        data = generate_ps_dgp(n=1000, seed=42)

        py_result = ps_bounds_no_assumption(data["Y"], data["D"], data["Z"])
        r_result = r_bounds_manski(data["Y"], data["D"], data["Z"])

        assert np.isclose(py_result["bound_width"], r_result["bound_width"], rtol=0.10), (
            f"Width: Python={py_result['bound_width']:.4f} vs R={r_result['bound_width']:.4f}"
        )

    def test_custom_support_parity(self):
        """Bounds with custom support should match R."""
        data = generate_ps_dgp(n=1000, seed=42)
        support = (-5.0, 10.0)

        py_result = ps_bounds_no_assumption(
            data["Y"], data["D"], data["Z"], outcome_support=support
        )
        r_result = r_bounds_manski(data["Y"], data["D"], data["Z"], outcome_support=support)

        assert np.isclose(py_result["lower_bound"], r_result["lower_bound"], rtol=0.10)
        assert np.isclose(py_result["upper_bound"], r_result["upper_bound"], rtol=0.10)


# =============================================================================
# Stress Tests
# =============================================================================


@requires_ps
class TestTriangulationStress:
    """Stress tests for triangulation robustness."""

    def test_small_sample(self):
        """Triangulation should work with small samples (n=100)."""
        data = generate_ps_dgp(n=100, seed=42)

        py_result = cace_2sls(data["Y"], data["D"], data["Z"])
        r_result = r_cace_2sls(data["Y"], data["D"], data["Z"])

        # Larger tolerance for small samples
        assert np.isclose(py_result["cace"], r_result["cace"], rtol=0.15), (
            f"Small sample: Python={py_result['cace']:.4f} vs R={r_result['cace']:.4f}"
        )

    def test_large_sample(self):
        """Triangulation should be precise with large samples (n=5000)."""
        data = generate_ps_dgp(n=5000, seed=42)

        py_result = cace_2sls(data["Y"], data["D"], data["Z"])
        r_result = r_cace_2sls(data["Y"], data["D"], data["Z"])

        # Tight tolerance for large samples
        assert np.isclose(py_result["cace"], r_result["cace"], rtol=0.02), (
            f"Large sample: Python={py_result['cace']:.4f} vs R={r_result['cace']:.4f}"
        )

    def test_extreme_strata(self):
        """Triangulation with extreme strata proportions."""
        # Mostly compliers
        data = generate_ps_dgp(n=1000, pi_c=0.9, pi_a=0.05, pi_n=0.05, seed=42)

        py_result = cace_2sls(data["Y"], data["D"], data["Z"])
        r_result = r_cace_2sls(data["Y"], data["D"], data["Z"])

        assert np.isclose(py_result["cace"], r_result["cace"], rtol=0.05)

    def test_high_noise(self):
        """Triangulation with high outcome noise."""
        data = generate_ps_dgp(n=1000, noise_sd=5.0, seed=42)

        py_result = cace_2sls(data["Y"], data["D"], data["Z"])
        r_result = r_cace_2sls(data["Y"], data["D"], data["Z"])

        # SE should match even with high noise
        assert np.isclose(py_result["se"], r_result["se"], rtol=0.15)


# =============================================================================
# Monte Carlo Triangulation
# =============================================================================


@requires_ps
class TestMonteCarloTriangulation:
    """Monte Carlo tests to verify systematic agreement."""

    @pytest.mark.slow
    def test_2sls_systematic_agreement(self):
        """2SLS should systematically agree across 50 simulations."""
        diffs = []

        for seed in range(50):
            data = generate_ps_dgp(n=500, seed=seed)

            py_result = cace_2sls(data["Y"], data["D"], data["Z"])
            r_result = r_cace_2sls(data["Y"], data["D"], data["Z"])

            diff = py_result["cace"] - r_result["cace"]
            diffs.append(diff)

        diffs = np.array(diffs)

        # Mean difference should be near zero (no systematic bias)
        assert abs(np.mean(diffs)) < 0.05, f"Mean diff={np.mean(diffs):.4f}, should be ~0"

        # Max absolute difference should be small
        assert np.max(np.abs(diffs)) < 0.2, f"Max diff={np.max(np.abs(diffs)):.4f}"

    @pytest.mark.slow
    def test_em_systematic_agreement(self):
        """EM should systematically agree across 30 simulations."""
        cace_diffs = []

        for seed in range(30):
            data = generate_ps_dgp(n=500, seed=seed)

            py_result = cace_em(data["Y"], data["D"], data["Z"])
            r_result = r_cace_em_manual(data["Y"], data["D"], data["Z"])

            diff = py_result["cace"] - r_result["cace"]
            cace_diffs.append(diff)

        cace_diffs = np.array(cace_diffs)

        # Mean difference should be near zero
        assert abs(np.mean(cace_diffs)) < 0.1, f"Mean CACE diff={np.mean(cace_diffs):.4f}"

        # Most differences should be small
        pct_small = np.mean(np.abs(cace_diffs) < 0.2)
        assert pct_small > 0.8, f"Only {pct_small:.0%} diffs < 0.2"
