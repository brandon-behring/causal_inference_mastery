"""Golden Reference Regression Tests.

This module tests that current implementations produce results matching
the frozen golden reference (tests/golden_results/python_golden_results.json).

Purpose:
- Catch unintended regressions in estimator behavior
- Validate that refactoring preserves numerical results
- Ensure reproducibility across versions

Created: Session 166 (Independent Audit 2026-01-01)
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.causal_inference.rct.estimators import simple_ate
from src.causal_inference.rct.estimators_stratified import stratified_ate
from src.causal_inference.rct.estimators_regression import regression_adjusted_ate
from src.causal_inference.rct.estimators_ipw import ipw_ate


# Load golden results at module level
GOLDEN_PATH = Path(__file__).parent.parent / "golden_results" / "python_golden_results.json"


@pytest.fixture(scope="module")
def golden_results():
    """Load golden reference results."""
    if not GOLDEN_PATH.exists():
        pytest.skip(f"Golden results file not found: {GOLDEN_PATH}")
    with open(GOLDEN_PATH) as f:
        return json.load(f)


class TestGoldenReferenceRCT:
    """Test RCT estimators against golden reference."""

    def test_balanced_rct_simple_ate(self, golden_results):
        """Test simple_ate matches golden reference for balanced RCT."""
        data = golden_results["balanced_rct"]["data"]
        expected = golden_results["balanced_rct"]["simple_ate"]

        treatment = np.array(data["treatment"])
        outcomes = np.array(data["outcomes"])

        result = simple_ate(outcomes, treatment)

        # Use rtol=1e-10 for numerical precision (not exact equality due to float representation)
        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-10), (
            f"ATE mismatch: got {result['estimate']}, expected {expected['estimate']}"
        )
        assert np.isclose(result["se"], expected["se"], rtol=1e-10), (
            f"SE mismatch: got {result['se']}, expected {expected['se']}"
        )

    def test_stratified_rct_stratified_ate(self, golden_results):
        """Test stratified_ate matches golden reference."""
        data = golden_results["stratified_rct"]["data"]
        expected = golden_results["stratified_rct"]["stratified_ate"]

        treatment = np.array(data["treatment"])
        outcomes = np.array(data["outcomes"])
        strata = np.array(data["strata"])

        result = stratified_ate(outcomes, treatment, strata)

        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-10), (
            f"ATE mismatch: got {result['estimate']}, expected {expected['estimate']}"
        )
        assert np.isclose(result["se"], expected["se"], rtol=1e-10), (
            f"SE mismatch: got {result['se']}, expected {expected['se']}"
        )

    def test_regression_rct_regression_adjusted_ate(self, golden_results):
        """Test regression_adjusted_ate matches golden reference."""
        data = golden_results["regression_rct"]["data"]
        expected = golden_results["regression_rct"]["regression_adjusted_ate"]

        treatment = np.array(data["treatment"])
        outcomes = np.array(data["outcomes"])
        covariate = np.array(data["covariate"])

        result = regression_adjusted_ate(outcomes, treatment, covariate)

        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-10), (
            f"ATE mismatch: got {result["estimate"]}, expected {expected['ate']}"
        )
        assert np.isclose(result["se"], expected["se"], rtol=1e-10), (
            f"SE mismatch: got {result["se"]}, expected {expected['se']}"
        )

    def test_ipw_varying_ipw_ate(self, golden_results):
        """Test ipw_ate matches golden reference with varying propensity."""
        data = golden_results["ipw_varying"]["data"]
        expected = golden_results["ipw_varying"]["ipw_ate"]

        treatment = np.array(data["treatment"])
        outcomes = np.array(data["outcomes"])
        propensity = np.array(data["propensity"])

        result = ipw_ate(outcomes, treatment, propensity)

        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-10), (
            f"ATE mismatch: got {result["estimate"]}, expected {expected['ate']}"
        )
        assert np.isclose(result["se"], expected["se"], rtol=1e-10), (
            f"SE mismatch: got {result["se"]}, expected {expected['se']}"
        )

    def test_large_sample_simple_ate(self, golden_results):
        """Test simple_ate matches golden reference for large sample."""
        data = golden_results["large_sample"]["data"]
        expected = golden_results["large_sample"]["simple_ate"]

        treatment = np.array(data["treatment"])
        outcomes = np.array(data["outcomes"])

        result = simple_ate(outcomes, treatment)

        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-10), (
            f"ATE mismatch: got {result["estimate"]}, expected {expected['ate']}"
        )
        assert np.isclose(result["se"], expected["se"], rtol=1e-10), (
            f"SE mismatch: got {result["se"]}, expected {expected['se']}"
        )

    def test_large_sample_regression_adjusted_ate(self, golden_results):
        """Test regression_adjusted_ate matches golden reference for large sample."""
        data = golden_results["large_sample"]["data"]
        expected = golden_results["large_sample"]["regression_adjusted_ate"]

        treatment = np.array(data["treatment"])
        outcomes = np.array(data["outcomes"])
        covariate = np.array(data["covariate"])

        result = regression_adjusted_ate(outcomes, treatment, covariate)

        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-10), (
            f"ATE mismatch: got {result["estimate"]}, expected {expected['ate']}"
        )
        assert np.isclose(result["se"], expected["se"], rtol=1e-10), (
            f"SE mismatch: got {result["se"]}, expected {expected['se']}"
        )

    def test_large_sample_ipw_ate(self, golden_results):
        """Test ipw_ate matches golden reference for large sample."""
        data = golden_results["large_sample"]["data"]
        expected = golden_results["large_sample"]["ipw_ate"]

        treatment = np.array(data["treatment"])
        outcomes = np.array(data["outcomes"])
        propensity = np.array(data["propensity"])

        result = ipw_ate(outcomes, treatment, propensity)

        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-10), (
            f"ATE mismatch: got {result["estimate"]}, expected {expected['ate']}"
        )
        assert np.isclose(result["se"], expected["se"], rtol=1e-10), (
            f"SE mismatch: got {result["se"]}, expected {expected['se']}"
        )


class TestGoldenReferenceMetadata:
    """Validate golden reference file structure."""

    def test_golden_file_exists(self):
        """Verify golden results file exists."""
        assert GOLDEN_PATH.exists(), f"Golden results file missing: {GOLDEN_PATH}"

    def test_golden_file_has_expected_cases(self, golden_results):
        """Verify expected test cases are present."""
        expected_cases = [
            "balanced_rct",
            "stratified_rct",
            "regression_rct",
            "permutation_small",
            "ipw_varying",
            "large_sample",
        ]
        for case in expected_cases:
            assert case in golden_results, f"Missing test case: {case}"

    def test_golden_file_has_descriptions(self, golden_results):
        """Verify all test cases have descriptions."""
        for case_name, case_data in golden_results.items():
            assert "description" in case_data, f"Missing description for {case_name}"
            assert len(case_data["description"]) > 0, f"Empty description for {case_name}"

    def test_golden_file_has_data(self, golden_results):
        """Verify all test cases have input data."""
        for case_name, case_data in golden_results.items():
            assert "data" in case_data, f"Missing data for {case_name}"
            # Not all cases have treatment/outcomes (IV uses y, d, z)
            assert len(case_data["data"]) > 0, f"Empty data for {case_name}"


# =============================================================================
# PSM Golden Reference Tests (Session 180)
# =============================================================================


class TestGoldenReferencePSM:
    """Test PSM estimators against golden reference."""

    def test_psm_observable_confounding(self, golden_results):
        """Test psm_ate matches golden reference for observable confounding case."""
        from src.causal_inference.psm import psm_ate

        if "psm_observable_confounding" not in golden_results:
            pytest.skip("PSM golden reference not available")

        data = golden_results["psm_observable_confounding"]["data"]
        expected = golden_results["psm_observable_confounding"]["psm_ate"]

        treatment = np.array(data["treatment"])
        outcomes = np.array(data["outcomes"])
        covariates = np.array(data["covariates"])

        result = psm_ate(outcomes, treatment, covariates)

        # PSM has some stochastic component, use rtol=1e-6 for near-exact match
        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-6), (
            f"PSM estimate mismatch: got {result['estimate']}, expected {expected['estimate']}"
        )


# =============================================================================
# IV Golden Reference Tests (Session 180)
# =============================================================================


class TestGoldenReferenceIV:
    """Test IV estimators against golden reference."""

    def test_iv_tsls(self, golden_results):
        """Test 2SLS matches golden reference."""
        from src.causal_inference.iv import TwoStageLeastSquares

        if "iv_single_instrument" not in golden_results:
            pytest.skip("IV golden reference not available")

        data = golden_results["iv_single_instrument"]["data"]
        expected = golden_results["iv_single_instrument"]["tsls"]

        y = np.array(data["y"])
        d = np.array(data["d"]).reshape(-1, 1)
        z = np.array(data["z"]).reshape(-1, 1)

        tsls = TwoStageLeastSquares(inference="robust")
        tsls.fit(y, d, z)

        assert np.isclose(tsls.coef_[0], expected["coefficient"], rtol=1e-10), (
            f"2SLS coef mismatch: got {tsls.coef_[0]}, expected {expected['coefficient']}"
        )

    def test_iv_liml(self, golden_results):
        """Test LIML matches golden reference."""
        from src.causal_inference.iv import LIML

        if "iv_single_instrument" not in golden_results:
            pytest.skip("IV golden reference not available")

        data = golden_results["iv_single_instrument"]["data"]
        expected = golden_results["iv_single_instrument"]["liml"]

        y = np.array(data["y"])
        d = np.array(data["d"]).reshape(-1, 1)
        z = np.array(data["z"]).reshape(-1, 1)

        liml = LIML(inference="robust")
        liml.fit(y, d, z)

        assert np.isclose(liml.coef_[0], expected["coefficient"], rtol=1e-10), (
            f"LIML coef mismatch: got {liml.coef_[0]}, expected {expected['coefficient']}"
        )


# =============================================================================
# DiD Golden Reference Tests (Session 180)
# =============================================================================


class TestGoldenReferenceDiD:
    """Test DiD estimators against golden reference."""

    def test_did_classic_2x2(self, golden_results):
        """Test did_2x2 matches golden reference."""
        from src.causal_inference.did import did_2x2

        if "did_classic_2x2" not in golden_results:
            pytest.skip("DiD golden reference not available")

        data = golden_results["did_classic_2x2"]["data"]
        expected = golden_results["did_classic_2x2"]["did_2x2"]

        outcome = np.array(data["outcome"])
        treatment = np.array(data["treatment"])
        post = np.array(data["post"])
        unit_id = np.array(data["unit_id"])

        result = did_2x2(
            outcomes=outcome,
            treatment=treatment,
            post=post,
            unit_id=unit_id,
        )

        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-10), (
            f"DiD estimate mismatch: got {result['estimate']}, expected {expected['estimate']}"
        )


# =============================================================================
# RDD Golden Reference Tests (Session 180)
# =============================================================================


class TestGoldenReferenceRDD:
    """Test RDD estimators against golden reference."""

    def test_rdd_sharp(self, golden_results):
        """Test SharpRDD matches golden reference."""
        from src.causal_inference.rdd import SharpRDD

        if "rdd_sharp" not in golden_results:
            pytest.skip("RDD sharp golden reference not available")

        data = golden_results["rdd_sharp"]["data"]
        expected = golden_results["rdd_sharp"]["sharp_rdd"]

        outcome = np.array(data["outcome"])
        running = np.array(data["running"])

        rdd = SharpRDD(cutoff=0.0, bandwidth=expected["bandwidth"], inference="robust")
        rdd.fit(outcome, running)

        assert np.isclose(rdd.coef_, expected["estimate"], rtol=1e-10), (
            f"Sharp RDD estimate mismatch: got {rdd.coef_}, expected {expected['estimate']}"
        )

    def test_rdd_fuzzy(self, golden_results):
        """Test FuzzyRDD matches golden reference."""
        from src.causal_inference.rdd import FuzzyRDD

        if "rdd_fuzzy" not in golden_results:
            pytest.skip("RDD fuzzy golden reference not available")

        data = golden_results["rdd_fuzzy"]["data"]
        expected = golden_results["rdd_fuzzy"]["fuzzy_rdd"]

        outcome = np.array(data["outcome"])
        running = np.array(data["running"])
        treatment = np.array(data["treatment"])

        fuzzy = FuzzyRDD(cutoff=0.0, bandwidth=expected["bandwidth"], inference="robust")
        fuzzy.fit(outcome, running, treatment)

        assert np.isclose(fuzzy.coef_, expected["estimate"], rtol=1e-10), (
            f"Fuzzy RDD estimate mismatch: got {fuzzy.coef_}, expected {expected['estimate']}"
        )
