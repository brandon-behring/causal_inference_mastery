"""Capture golden results from Python library implementations.

These golden results will be used to validate Julia from-scratch implementations.
We run all 5 estimators on carefully designed test cases and save the outputs.

Following Brandon's principle: NEVER FAIL SILENTLY. All errors explicit.
"""

import numpy as np
import json
from pathlib import Path

from src.causal_inference.rct.estimators import simple_ate
from src.causal_inference.rct.estimators_stratified import stratified_ate
from src.causal_inference.rct.estimators_regression import regression_adjusted_ate
from src.causal_inference.rct.estimators_permutation import permutation_test
from src.causal_inference.rct.estimators_ipw import ipw_ate

# Additional family imports for Session 180 golden reference expansion
from src.causal_inference.psm import psm_ate
from src.causal_inference.iv import TwoStageLeastSquares, LIML
from src.causal_inference.did import did_2x2, event_study
from src.causal_inference.rdd import SharpRDD, FuzzyRDD


def capture_golden_results():
    """
    Capture golden results from all 5 estimators on reference datasets.

    Returns
    -------
    dict
        Dictionary mapping test case names to estimator results.
    """
    golden_results = {}

    # ============================================================================
    # Test Case 1: Balanced RCT with moderate effect
    # ============================================================================

    np.random.seed(42)
    n = 100
    treatment_balanced = np.array([1, 0] * (n // 2))
    outcomes_balanced = 5 * treatment_balanced + np.random.normal(10, 2, n)

    golden_results["balanced_rct"] = {
        "description": "Balanced RCT, n=100, true ATE=5, noise sd=2",
        "data": {
            "treatment": treatment_balanced.tolist(),
            "outcomes": outcomes_balanced.tolist(),
        },
        "simple_ate": simple_ate(outcomes_balanced, treatment_balanced),
    }

    # ============================================================================
    # Test Case 2: Stratified RCT (2 strata, different baselines)
    # ============================================================================

    strata = np.array([1]*50 + [2]*50)
    treatment_stratified = np.array([1, 0] * 25 + [1, 0] * 25)

    # Stratum 1: High baseline (Y ~ 100)
    # Stratum 2: Low baseline (Y ~ 10)
    # Both have ATE = 5
    outcomes_stratified = np.zeros(100)
    outcomes_stratified[:50] = 100 + 5 * treatment_stratified[:50] + np.random.normal(0, 2, 50)
    outcomes_stratified[50:] = 10 + 5 * treatment_stratified[50:] + np.random.normal(0, 2, 50)

    golden_results["stratified_rct"] = {
        "description": "Stratified RCT, 2 strata with different baselines, true ATE=5",
        "data": {
            "treatment": treatment_stratified.tolist(),
            "outcomes": outcomes_stratified.tolist(),
            "strata": strata.tolist(),
        },
        "simple_ate": simple_ate(outcomes_stratified, treatment_stratified),
        "stratified_ate": stratified_ate(outcomes_stratified, treatment_stratified, strata),
    }

    # ============================================================================
    # Test Case 3: RCT with covariate (for regression adjustment)
    # ============================================================================

    X_single = np.random.normal(5, 2, 100)
    treatment_reg = np.array([1, 0] * 50)
    # Y = 3*T + 2*X + noise (ATE = 3)
    outcomes_reg = 3 * treatment_reg + 2 * X_single + np.random.normal(0, 1, 100)

    golden_results["regression_rct"] = {
        "description": "RCT with covariate, Y = 3*T + 2*X + noise, true ATE=3",
        "data": {
            "treatment": treatment_reg.tolist(),
            "outcomes": outcomes_reg.tolist(),
            "covariate": X_single.tolist(),
        },
        "simple_ate": simple_ate(outcomes_reg, treatment_reg),
        "regression_adjusted_ate": regression_adjusted_ate(outcomes_reg, treatment_reg, X_single),
    }

    # ============================================================================
    # Test Case 4: Small sample for permutation test (exact)
    # ============================================================================

    treatment_small = np.array([1, 1, 1, 0, 0, 0])
    outcomes_small = np.array([10.0, 12.0, 11.0, 4.0, 5.0, 3.0])

    golden_results["permutation_small"] = {
        "description": "Small sample (n=6) for exact permutation test, strong effect",
        "data": {
            "treatment": treatment_small.tolist(),
            "outcomes": outcomes_small.tolist(),
        },
        "simple_ate": simple_ate(outcomes_small, treatment_small),
        "permutation_test_exact": permutation_test(
            outcomes_small, treatment_small, n_permutations=None, random_seed=42
        ),
        "permutation_test_monte_carlo": permutation_test(
            outcomes_small, treatment_small, n_permutations=1000, random_seed=42
        ),
    }

    # ============================================================================
    # Test Case 5: IPW with varying propensity
    # ============================================================================

    treatment_ipw = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    outcomes_ipw = np.array([10.0, 12.0, 9.0, 11.0, 4.0, 5.0, 3.0, 6.0])
    propensity_ipw = np.array([0.7, 0.6, 0.8, 0.5, 0.6, 0.7, 0.5, 0.8])

    golden_results["ipw_varying"] = {
        "description": "IPW with varying propensity scores, n=8",
        "data": {
            "treatment": treatment_ipw.tolist(),
            "outcomes": outcomes_ipw.tolist(),
            "propensity": propensity_ipw.tolist(),
        },
        "simple_ate": simple_ate(outcomes_ipw, treatment_ipw),
        "ipw_ate": ipw_ate(outcomes_ipw, treatment_ipw, propensity_ipw),
    }

    # ============================================================================
    # Test Case 6: Large sample for Monte Carlo validation
    # ============================================================================

    np.random.seed(123)
    n_large = 500
    treatment_large = np.array([1, 0] * (n_large // 2))
    X_large = np.random.normal(0, 3, n_large)
    outcomes_large = 4 * treatment_large + 1.5 * X_large + np.random.normal(0, 2, n_large)
    propensity_large = np.full(n_large, 0.5)

    golden_results["large_sample"] = {
        "description": "Large sample (n=500) with covariate, true ATE=4",
        "data": {
            "treatment": treatment_large.tolist(),
            "outcomes": outcomes_large.tolist(),
            "covariate": X_large.tolist(),
            "propensity": propensity_large.tolist(),
        },
        "simple_ate": simple_ate(outcomes_large, treatment_large),
        "regression_adjusted_ate": regression_adjusted_ate(outcomes_large, treatment_large, X_large),
        "ipw_ate": ipw_ate(outcomes_large, treatment_large, propensity_large),
        "permutation_test_monte_carlo": permutation_test(
            outcomes_large, treatment_large, n_permutations=1000, random_seed=123
        ),
    }

    # ============================================================================
    # PSM Test Cases (Session 180)
    # ============================================================================
    golden_results.update(capture_psm_golden())

    # ============================================================================
    # IV Test Cases (Session 180)
    # ============================================================================
    golden_results.update(capture_iv_golden())

    # ============================================================================
    # DiD Test Cases (Session 180)
    # ============================================================================
    golden_results.update(capture_did_golden())

    # ============================================================================
    # RDD Test Cases (Session 180)
    # ============================================================================
    golden_results.update(capture_rdd_golden())

    return golden_results


def capture_psm_golden() -> dict:
    """
    Capture PSM golden reference results.

    Test cases:
    - Simple matching with strong treatment effect
    - Matching with balance check
    """
    results = {}
    np.random.seed(42)

    # Test Case: PSM with observable confounding
    n = 200
    X = np.random.randn(n, 2)  # 2 covariates
    propensity = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    treatment = (np.random.rand(n) < propensity).astype(int)

    # Outcome depends on treatment and covariates
    # True ATE = 3.0
    outcome = 3.0 * treatment + 2.0 * X[:, 0] + 1.5 * X[:, 1] + np.random.randn(n) * 0.5

    # Run PSM
    psm_result = psm_ate(outcome, treatment, X)

    results["psm_observable_confounding"] = {
        "description": "PSM with observable confounding, true ATE=3.0, n=200",
        "data": {
            "treatment": treatment.tolist(),
            "outcomes": outcome.tolist(),
            "covariates": X.tolist(),
        },
        "psm_ate": psm_result,
    }

    return results


def capture_iv_golden() -> dict:
    """
    Capture IV golden reference results.

    Test cases:
    - 2SLS with single instrument
    - LIML comparison
    """
    results = {}
    np.random.seed(123)

    # Test Case: IV with single instrument
    n = 300
    z = np.random.randn(n)  # Instrument
    u = np.random.randn(n) * 0.5  # Confounder

    # Endogenous treatment: correlated with u
    d = 0.8 * z + 0.5 * u + np.random.randn(n) * 0.3

    # Outcome: true effect = 2.0
    y = 2.0 * d + u + np.random.randn(n) * 0.5

    # Reshape for estimators (D = treatment, Z = instrument)
    D = d.reshape(-1, 1)
    Z = z.reshape(-1, 1)

    # Run 2SLS
    tsls = TwoStageLeastSquares(inference="robust")
    tsls.fit(y, D, Z)

    # Run LIML
    liml = LIML(inference="robust")
    liml.fit(y, D, Z)

    results["iv_single_instrument"] = {
        "description": "IV with single instrument, true effect=2.0, n=300",
        "data": {
            "y": y.tolist(),
            "d": d.tolist(),
            "z": z.tolist(),
        },
        "tsls": {
            "coefficient": float(tsls.coef_[0]),
            "se": float(tsls.se_[0]) if hasattr(tsls, "se_") else None,
        },
        "liml": {
            "coefficient": float(liml.coef_[0]),
            "se": float(liml.se_[0]) if hasattr(liml, "se_") else None,
        },
    }

    return results


def capture_did_golden() -> dict:
    """
    Capture DiD golden reference results.

    Test cases:
    - Classic 2x2 DiD
    """
    results = {}
    np.random.seed(456)

    # Test Case: Classic 2x2 DiD
    n_units = 50  # 25 treated, 25 control
    n_periods = 2  # pre and post

    # Panel structure
    n = n_units * n_periods
    unit_id = np.repeat(np.arange(n_units), n_periods)
    treatment = np.repeat(np.array([1] * 25 + [0] * 25), n_periods)
    post = np.tile([0, 1], n_units)

    # Parallel trends with treatment effect = 5.0
    baseline = 10 + 2 * treatment + np.random.randn(n) * 0.5
    outcome = baseline + 3 * post + 5.0 * (treatment * post) + np.random.randn(n) * 0.5

    did_result = did_2x2(
        outcomes=outcome,
        treatment=treatment,
        post=post,
        unit_id=unit_id,
    )

    results["did_classic_2x2"] = {
        "description": "Classic 2x2 DiD, true ATT=5.0, n_units=50",
        "data": {
            "outcome": outcome.tolist(),
            "treatment": treatment.tolist(),
            "post": post.tolist(),
            "unit_id": unit_id.tolist(),
        },
        "did_2x2": did_result,
    }

    return results


def capture_rdd_golden() -> dict:
    """
    Capture RDD golden reference results.

    Test cases:
    - Sharp RDD at cutoff 0
    - Fuzzy RDD
    """
    results = {}
    np.random.seed(789)

    # Test Case: Sharp RDD
    n = 200
    running = np.random.uniform(-1, 1, n)

    # Continuous outcome with jump at cutoff
    # True treatment effect = 4.0
    treatment = (running >= 0).astype(int)
    outcome = 2.0 * running + 4.0 * treatment + np.random.randn(n) * 0.5

    # Fit Sharp RDD
    rdd = SharpRDD(cutoff=0.0, bandwidth=0.5, inference="robust")
    rdd.fit(outcome, running)

    results["rdd_sharp"] = {
        "description": "Sharp RDD at cutoff=0, true effect=4.0, n=200",
        "data": {
            "outcome": outcome.tolist(),
            "running": running.tolist(),
            "treatment": treatment.tolist(),
        },
        "sharp_rdd": {
            "estimate": float(rdd.coef_),
            "se": float(rdd.se_) if hasattr(rdd, "se_") else None,
            "bandwidth": 0.5,
        },
    }

    # Test Case: Fuzzy RDD
    np.random.seed(321)
    n = 200
    running = np.random.uniform(-1, 1, n)

    # Fuzzy treatment: probability jumps at cutoff
    prob = 0.2 + 0.6 * (running >= 0).astype(float)
    treatment = (np.random.rand(n) < prob).astype(int)

    # True LATE = 3.5
    outcome = 1.5 * running + 3.5 * treatment + np.random.randn(n) * 0.5

    fuzzy = FuzzyRDD(cutoff=0.0, bandwidth=0.5, inference="robust")
    fuzzy.fit(outcome, running, treatment)

    results["rdd_fuzzy"] = {
        "description": "Fuzzy RDD at cutoff=0, true LATE=3.5, n=200",
        "data": {
            "outcome": outcome.tolist(),
            "running": running.tolist(),
            "treatment": treatment.tolist(),
        },
        "fuzzy_rdd": {
            "estimate": float(fuzzy.coef_),
            "se": float(fuzzy.se_) if hasattr(fuzzy, "se_") else None,
            "bandwidth": 0.5,
        },
    }

    return results


def convert_numpy_to_python(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Parameters
    ----------
    obj : any
        Object to convert (dict, list, numpy array, numpy scalar, etc.)

    Returns
    -------
    any
        Converted object with native Python types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj


def save_golden_results(results: dict, output_path: Path):
    """
    Save golden results to JSON file.

    Parameters
    ----------
    results : dict
        Golden results from capture_golden_results()
    output_path : Path
        Path to save JSON file

    Raises
    ------
    IOError
        If file cannot be written (following NEVER FAIL SILENTLY principle)
    """
    try:
        # Convert numpy types to Python types for JSON serialization
        results_serializable = convert_numpy_to_python(results)

        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"✅ Golden results saved to: {output_path}")
        print(f"   File size: {output_path.stat().st_size:,} bytes")
    except Exception as e:
        raise IOError(
            f"CRITICAL ERROR: Failed to save golden results.\n"
            f"Function: save_golden_results\n"
            f"Path: {output_path}\n"
            f"Error: {str(e)}"
        )


def main():
    """Main entry point."""
    print("=" * 80)
    print("CAPTURING GOLDEN RESULTS FROM PYTHON LIBRARY IMPLEMENTATIONS")
    print("=" * 80)

    # Capture results
    print("\n📊 Running all estimators on reference datasets...")
    golden_results = capture_golden_results()

    # Print summary
    print(f"\n✅ Captured {len(golden_results)} test cases:")
    for test_name, test_data in golden_results.items():
        print(f"   - {test_name}: {test_data['description']}")

    # Save to file
    output_dir = Path("tests/golden_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "python_golden_results.json"

    save_golden_results(golden_results, output_path)

    print("\n" + "=" * 80)
    print("GOLDEN RESULTS CAPTURE COMPLETE")
    print("=" * 80)
    print(f"\nNext step: Implement Julia estimators and validate against these results.")
    print(f"Validation tolerance: rtol < 1e-10 (near machine precision)")


if __name__ == "__main__":
    main()
