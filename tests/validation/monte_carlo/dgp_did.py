"""
Data Generating Processes (DGPs) for DiD Monte Carlo validation.

This module provides DGPs for validating Difference-in-Differences estimators:
- Classic 2×2 DiD
- Staggered adoption with homogeneous/heterogeneous effects
- Event study designs

All DGPs have known true effects for validation purposes.

References:
    - Bertrand, Duflo, Mullainathan (2004). "How much should we trust differences-in-differences estimates?"
    - Goodman-Bacon (2021). "Difference-in-Differences with Variation in Treatment Timing"
    - Callaway & Sant'Anna (2021). "Difference-in-Differences with Multiple Time Periods"
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class DiDData:
    """Container for DiD simulation data with known ground truth."""

    outcomes: np.ndarray
    treatment: np.ndarray  # Unit-level treatment indicator
    post: np.ndarray  # Post-period indicator
    unit_id: np.ndarray
    time: np.ndarray
    true_att: float
    n_treated: int
    n_control: int
    n_periods: int


@dataclass
class StaggeredData:
    """Container for staggered DiD simulation data."""

    outcomes: np.ndarray
    treatment: np.ndarray  # Time-varying treatment indicator (D_it)
    time: np.ndarray
    unit_id: np.ndarray
    treatment_time: np.ndarray  # Treatment time per unit (np.inf for never-treated)
    true_att: float
    cohort_effects: Dict[int, float]  # τ_g per cohort
    n_units: int
    n_periods: int


# =============================================================================
# Classic 2×2 DiD DGPs
# =============================================================================


def dgp_did_2x2_simple(
    n_treated: int = 50,
    n_control: int = 50,
    n_pre: int = 1,
    n_post: int = 1,
    true_att: float = 2.0,
    sigma: float = 1.0,
    unit_fe_sigma: float = 1.0,
    time_fe_sigma: float = 0.5,
    random_state: Optional[int] = None,
) -> DiDData:
    """
    Simple 2×2 DiD DGP with known treatment effect.

    DGP:
        Y_it = α_i + λ_t + τ·D_i·Post_t + ε_it

        where:
        - α_i ~ N(0, unit_fe_sigma²) : unit fixed effects
        - λ_t ~ N(0, time_fe_sigma²) : time fixed effects
        - τ = true_att : treatment effect on the treated
        - D_i = 1 for treated units, 0 for control
        - Post_t = 1 for post-treatment periods, 0 for pre
        - ε_it ~ N(0, σ²) : idiosyncratic errors

    Parameters
    ----------
    n_treated : int, default=50
        Number of treated units
    n_control : int, default=50
        Number of control units
    n_pre : int, default=1
        Number of pre-treatment periods
    n_post : int, default=1
        Number of post-treatment periods
    true_att : float, default=2.0
        True average treatment effect on treated
    sigma : float, default=1.0
        Standard deviation of idiosyncratic errors
    unit_fe_sigma : float, default=1.0
        Standard deviation of unit fixed effects
    time_fe_sigma : float, default=0.5
        Standard deviation of time fixed effects
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    DiDData
        Container with outcomes, treatment, post indicators, and ground truth
    """
    rng = np.random.RandomState(random_state)

    n_units = n_treated + n_control
    n_periods = n_pre + n_post

    # Unit and time indices
    unit_id = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)

    # Treatment indicator (unit-level)
    treatment_unit = np.array([1] * n_treated + [0] * n_control)
    treatment = np.repeat(treatment_unit, n_periods)

    # Post indicator (time-level)
    post_time = np.array([0] * n_pre + [1] * n_post)
    post = np.tile(post_time, n_units)

    # Fixed effects
    unit_fe = rng.normal(0, unit_fe_sigma, n_units)
    time_fe = rng.normal(0, time_fe_sigma, n_periods)

    # Expand fixed effects to observation level
    unit_fe_obs = unit_fe[unit_id]
    time_fe_obs = time_fe[time]

    # Idiosyncratic errors
    epsilon = rng.normal(0, sigma, n_units * n_periods)

    # Outcome: Y = α_i + λ_t + τ·D·Post + ε
    outcomes = unit_fe_obs + time_fe_obs + true_att * treatment * post + epsilon

    return DiDData(
        outcomes=outcomes,
        treatment=treatment,
        post=post,
        unit_id=unit_id,
        time=time,
        true_att=true_att,
        n_treated=n_treated,
        n_control=n_control,
        n_periods=n_periods,
    )


def dgp_did_2x2_heteroskedastic(
    n_treated: int = 50,
    n_control: int = 50,
    n_pre: int = 1,
    n_post: int = 1,
    true_att: float = 2.0,
    sigma_treated: float = 2.0,
    sigma_control: float = 1.0,
    random_state: Optional[int] = None,
) -> DiDData:
    """
    2×2 DiD DGP with heteroskedastic errors.

    Same as dgp_did_2x2_simple but with different error variances
    for treated vs control groups. Tests robustness of standard errors.

    DGP:
        Y_it = α_i + λ_t + τ·D_i·Post_t + ε_it
        ε_it ~ N(0, σ_treated²) if D_i = 1
        ε_it ~ N(0, σ_control²) if D_i = 0

    True ATT = true_att
    """
    rng = np.random.RandomState(random_state)

    n_units = n_treated + n_control
    n_periods = n_pre + n_post

    # Unit and time indices
    unit_id = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)

    # Treatment indicator
    treatment_unit = np.array([1] * n_treated + [0] * n_control)
    treatment = np.repeat(treatment_unit, n_periods)

    # Post indicator
    post_time = np.array([0] * n_pre + [1] * n_post)
    post = np.tile(post_time, n_units)

    # Fixed effects
    unit_fe = rng.normal(0, 1.0, n_units)
    time_fe = rng.normal(0, 0.5, n_periods)

    unit_fe_obs = unit_fe[unit_id]
    time_fe_obs = time_fe[time]

    # Heteroskedastic errors
    sigma = np.where(treatment == 1, sigma_treated, sigma_control)
    epsilon = rng.normal(0, sigma)

    # Outcome
    outcomes = unit_fe_obs + time_fe_obs + true_att * treatment * post + epsilon

    return DiDData(
        outcomes=outcomes,
        treatment=treatment,
        post=post,
        unit_id=unit_id,
        time=time,
        true_att=true_att,
        n_treated=n_treated,
        n_control=n_control,
        n_periods=n_periods,
    )


def dgp_did_2x2_serial_correlation(
    n_treated: int = 50,
    n_control: int = 50,
    n_pre: int = 5,
    n_post: int = 5,
    true_att: float = 2.0,
    rho: float = 0.5,
    sigma: float = 1.0,
    random_state: Optional[int] = None,
) -> DiDData:
    """
    2×2 DiD DGP with AR(1) serial correlation within units.

    Critical test for cluster-robust SE validity (Bertrand et al. 2004).

    DGP:
        Y_it = α_i + λ_t + τ·D_i·Post_t + u_it
        u_it = ρ·u_{i,t-1} + ε_it
        ε_it ~ N(0, σ²)

    Serial correlation within units requires cluster-robust SEs.
    Naive SEs will be too small, leading to over-rejection.

    True ATT = true_att
    """
    rng = np.random.RandomState(random_state)

    n_units = n_treated + n_control
    n_periods = n_pre + n_post

    # Unit and time indices
    unit_id = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)

    # Treatment indicator
    treatment_unit = np.array([1] * n_treated + [0] * n_control)
    treatment = np.repeat(treatment_unit, n_periods)

    # Post indicator
    post_time = np.array([0] * n_pre + [1] * n_post)
    post = np.tile(post_time, n_units)

    # Fixed effects
    unit_fe = rng.normal(0, 1.0, n_units)
    time_fe = rng.normal(0, 0.5, n_periods)

    # AR(1) errors within each unit
    errors = np.zeros(n_units * n_periods)
    for i in range(n_units):
        start = i * n_periods
        end = (i + 1) * n_periods
        # Generate AR(1) process
        innovations = rng.normal(0, sigma, n_periods)
        u = np.zeros(n_periods)
        u[0] = innovations[0]
        for t in range(1, n_periods):
            u[t] = rho * u[t - 1] + innovations[t]
        errors[start:end] = u

    # Outcome
    unit_fe_obs = unit_fe[unit_id]
    time_fe_obs = time_fe[time]
    outcomes = unit_fe_obs + time_fe_obs + true_att * treatment * post + errors

    return DiDData(
        outcomes=outcomes,
        treatment=treatment,
        post=post,
        unit_id=unit_id,
        time=time,
        true_att=true_att,
        n_treated=n_treated,
        n_control=n_control,
        n_periods=n_periods,
    )


# =============================================================================
# Staggered DiD DGPs
# =============================================================================


def dgp_staggered_homogeneous(
    n_units: int = 150,
    n_periods: int = 10,
    cohorts: tuple = (5, 7),
    true_effect: float = 2.0,
    never_treated_frac: float = 0.2,
    sigma: float = 1.0,
    random_state: Optional[int] = None,
) -> StaggeredData:
    """
    Staggered DiD DGP with homogeneous treatment effects.

    When effects are homogeneous across cohorts, TWFE should be unbiased.
    Used to verify both TWFE and modern estimators work correctly.

    DGP:
        Y_it = α_i + λ_t + τ·D_it + ε_it

        where D_it = 1{t >= g_i} for units in cohort g
        τ is constant across all cohorts (homogeneous)

    Parameters
    ----------
    n_units : int, default=150
        Total number of units
    n_periods : int, default=10
        Number of time periods (0 to n_periods-1)
    cohorts : tuple, default=(5, 7)
        Treatment timing for each cohort
    true_effect : float, default=2.0
        True ATT (same for all cohorts)
    never_treated_frac : float, default=0.2
        Fraction of units that are never treated
    sigma : float, default=1.0
        Error standard deviation
    random_state : int, optional
        Random seed

    Returns
    -------
    StaggeredData
        Container with staggered DiD data and ground truth
    """
    rng = np.random.RandomState(random_state)

    # Assign units to cohorts (including never-treated)
    n_never = int(n_units * never_treated_frac)
    n_treated_total = n_units - n_never
    n_per_cohort = n_treated_total // len(cohorts)

    # Build treatment_time array
    treatment_time = []
    for g in cohorts:
        treatment_time.extend([g] * n_per_cohort)
    # Add remaining to last cohort
    remainder = n_treated_total - len(treatment_time)
    if remainder > 0:
        treatment_time.extend([cohorts[-1]] * remainder)
    # Add never-treated
    treatment_time.extend([np.inf] * n_never)
    treatment_time = np.array(treatment_time)

    # Shuffle unit assignment
    rng.shuffle(treatment_time)

    # Build panel data
    unit_id = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)

    # Treatment indicator: D_it = 1{t >= g_i}
    treatment_time_expanded = treatment_time[unit_id]
    treatment = (time >= treatment_time_expanded).astype(float)

    # Fixed effects
    unit_fe = rng.normal(0, 1.0, n_units)
    time_fe = rng.normal(0, 0.5, n_periods)

    unit_fe_obs = unit_fe[unit_id]
    time_fe_obs = time_fe[time]

    # Errors
    epsilon = rng.normal(0, sigma, n_units * n_periods)

    # Outcome: Y = α_i + λ_t + τ·D_it + ε
    outcomes = unit_fe_obs + time_fe_obs + true_effect * treatment + epsilon

    # Cohort effects (all same)
    cohort_effects = {g: true_effect for g in cohorts}

    return StaggeredData(
        outcomes=outcomes,
        treatment=treatment,
        time=time,
        unit_id=unit_id,
        treatment_time=treatment_time,
        true_att=true_effect,
        cohort_effects=cohort_effects,
        n_units=n_units,
        n_periods=n_periods,
    )


def dgp_staggered_heterogeneous(
    n_units: int = 150,
    n_periods: int = 10,
    cohort_effects: Optional[Dict[int, float]] = None,
    never_treated_frac: float = 0.2,
    sigma: float = 1.0,
    random_state: Optional[int] = None,
) -> StaggeredData:
    """
    Staggered DiD DGP with heterogeneous treatment effects across cohorts.

    CRITICAL: When effects vary by cohort, TWFE is BIASED.
    Callaway-Sant'Anna and Sun-Abraham remain unbiased.

    DGP:
        Y_it = α_i + λ_t + τ_g·D_it + ε_it

        where τ_g varies by cohort g.

    True ATT = weighted average of τ_g (weighted by cohort size × exposure)

    Parameters
    ----------
    n_units : int, default=150
        Total number of units
    n_periods : int, default=10
        Number of time periods
    cohort_effects : dict, optional
        Mapping {cohort: effect}. Default: {5: 1.0, 7: 5.0}
    never_treated_frac : float, default=0.2
        Fraction never-treated
    sigma : float, default=1.0
        Error standard deviation
    random_state : int, optional
        Random seed

    Returns
    -------
    StaggeredData
        Container with staggered DiD data and ground truth

    Notes
    -----
    Default cohort_effects={5: 1.0, 7: 5.0} creates strong heterogeneity
    (4.0 difference). TWFE bias will be substantial in this case.

    True ATT is the simple average of cohort effects weighted by
    post-treatment exposure: ATT = Σ_g (n_g × (T-g)) × τ_g / Σ_g (n_g × (T-g))

    For equal cohort sizes and cohorts {5,7} in T=10 periods:
    - Cohort 5: exposed for 5 periods (5,6,7,8,9)
    - Cohort 7: exposed for 3 periods (7,8,9)
    - ATT ≈ (5×1.0 + 3×5.0) / (5+3) = 20/8 = 2.5

    But for ATT(g,t) interpretation: simple average is (1.0 + 5.0)/2 = 3.0
    """
    rng = np.random.RandomState(random_state)

    if cohort_effects is None:
        cohort_effects = {5: 1.0, 7: 5.0}

    cohorts = list(cohort_effects.keys())

    # Assign units to cohorts
    n_never = int(n_units * never_treated_frac)
    n_treated_total = n_units - n_never
    n_per_cohort = n_treated_total // len(cohorts)

    treatment_time = []
    for g in cohorts:
        treatment_time.extend([g] * n_per_cohort)
    remainder = n_treated_total - len(treatment_time)
    if remainder > 0:
        treatment_time.extend([cohorts[-1]] * remainder)
    treatment_time.extend([np.inf] * n_never)
    treatment_time = np.array(treatment_time)
    rng.shuffle(treatment_time)

    # Build panel
    unit_id = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)

    # Treatment indicator
    treatment_time_expanded = treatment_time[unit_id]
    treatment = (time >= treatment_time_expanded).astype(float)

    # Fixed effects
    unit_fe = rng.normal(0, 1.0, n_units)
    time_fe = rng.normal(0, 0.5, n_periods)
    unit_fe_obs = unit_fe[unit_id]
    time_fe_obs = time_fe[time]

    # Heterogeneous treatment effects
    effect_obs = np.zeros(len(unit_id))
    for i, g in enumerate(treatment_time):
        if np.isfinite(g):
            mask = (unit_id == i) & (time >= g)
            effect_obs[mask] = cohort_effects[int(g)]

    # Errors
    epsilon = rng.normal(0, sigma, n_units * n_periods)

    # Outcome
    outcomes = unit_fe_obs + time_fe_obs + effect_obs * treatment + epsilon

    # Compute true ATT (simple average of cohort effects)
    # This is the CS/SA target parameter
    true_att = np.mean(list(cohort_effects.values()))

    return StaggeredData(
        outcomes=outcomes,
        treatment=treatment,
        time=time,
        unit_id=unit_id,
        treatment_time=treatment_time,
        true_att=true_att,
        cohort_effects=cohort_effects,
        n_units=n_units,
        n_periods=n_periods,
    )


def dgp_staggered_dynamic_effects(
    n_units: int = 150,
    n_periods: int = 10,
    cohorts: tuple = (3, 5, 7),
    effect_base: float = 1.0,
    effect_growth: float = 0.5,
    never_treated_frac: float = 0.2,
    sigma: float = 1.0,
    random_state: Optional[int] = None,
) -> StaggeredData:
    """
    Staggered DiD DGP with dynamic treatment effects.

    Effects grow over time since treatment (event time).
    Tests event study estimation of time-varying effects.

    DGP:
        Y_it = α_i + λ_t + τ(e)·D_it + ε_it

        where e = t - g_i is event time (time since treatment)
        τ(e) = effect_base + effect_growth × e for e >= 0

    Parameters
    ----------
    n_units : int, default=150
        Total units
    n_periods : int, default=10
        Total periods
    cohorts : tuple, default=(3, 5, 7)
        Treatment times
    effect_base : float, default=1.0
        Immediate effect at event time 0
    effect_growth : float, default=0.5
        Effect increase per period post-treatment
    never_treated_frac : float, default=0.2
        Fraction never-treated
    sigma : float, default=1.0
        Error SD
    random_state : int, optional
        Random seed

    Returns
    -------
    StaggeredData
        Container with dynamic effects data
    """
    rng = np.random.RandomState(random_state)

    # Assign cohorts
    n_never = int(n_units * never_treated_frac)
    n_treated_total = n_units - n_never
    n_per_cohort = n_treated_total // len(cohorts)

    treatment_time = []
    for g in cohorts:
        treatment_time.extend([g] * n_per_cohort)
    remainder = n_treated_total - len(treatment_time)
    if remainder > 0:
        treatment_time.extend([cohorts[-1]] * remainder)
    treatment_time.extend([np.inf] * n_never)
    treatment_time = np.array(treatment_time)
    rng.shuffle(treatment_time)

    # Build panel
    unit_id = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)

    treatment_time_expanded = treatment_time[unit_id]
    treatment = (time >= treatment_time_expanded).astype(float)

    # Fixed effects
    unit_fe = rng.normal(0, 1.0, n_units)
    time_fe = rng.normal(0, 0.5, n_periods)
    unit_fe_obs = unit_fe[unit_id]
    time_fe_obs = time_fe[time]

    # Dynamic effects based on event time
    event_time = time - treatment_time_expanded
    effect_obs = np.where(
        (treatment == 1) & np.isfinite(treatment_time_expanded),
        effect_base + effect_growth * event_time,
        0.0,
    )

    # Errors
    epsilon = rng.normal(0, sigma, n_units * n_periods)

    # Outcome
    outcomes = unit_fe_obs + time_fe_obs + effect_obs + epsilon

    # Cohort effects (evaluated at event time 0)
    cohort_effects = {g: effect_base for g in cohorts}

    # True ATT (average over all treated observations)
    # This depends on the distribution of event times
    true_att = effect_base + effect_growth * 2  # Approximate average event time

    return StaggeredData(
        outcomes=outcomes,
        treatment=treatment,
        time=time,
        unit_id=unit_id,
        treatment_time=treatment_time,
        true_att=true_att,
        cohort_effects=cohort_effects,
        n_units=n_units,
        n_periods=n_periods,
    )


# =============================================================================
# Event Study DGPs
# =============================================================================


def dgp_event_study_null_pretrends(
    n_treated: int = 100,
    n_control: int = 100,
    n_pre: int = 5,
    n_post: int = 5,
    treatment_time: int = 5,
    true_effect: float = 2.0,
    sigma: float = 1.0,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Event study DGP with true null pre-trends.

    All pre-treatment effects are exactly zero (parallel trends hold).
    Post-treatment effect is constant.

    DGP:
        Y_it = α_i + λ_t + Σ_k β_k·D_i·1{t - g = k} + ε_it

        where:
        - β_k = 0 for k < 0 (true parallel trends)
        - β_k = true_effect for k >= 0 (constant post-treatment)

    Returns
    -------
    dict
        - outcomes, treatment, time, unit_id: arrays
        - true_pretrend_effects: dict {k: 0 for k in pre-periods}
        - true_post_effects: dict {k: true_effect for k in post-periods}
        - event_times: list of event time values
    """
    rng = np.random.RandomState(random_state)

    n_units = n_treated + n_control
    n_periods = n_pre + n_post

    # Build panel
    unit_id = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)

    # Treatment indicator (unit-level)
    treatment_unit = np.array([1] * n_treated + [0] * n_control)
    treatment = np.repeat(treatment_unit, n_periods)

    # Event time: k = t - treatment_time
    event_time = time - treatment_time

    # Fixed effects
    unit_fe = rng.normal(0, 1.0, n_units)
    time_fe = rng.normal(0, 0.5, n_periods)
    unit_fe_obs = unit_fe[unit_id]
    time_fe_obs = time_fe[time]

    # Treatment effect: 0 for k < 0, true_effect for k >= 0
    effect = np.where((treatment == 1) & (event_time >= 0), true_effect, 0.0)

    # Errors
    epsilon = rng.normal(0, sigma, n_units * n_periods)

    # Outcome
    outcomes = unit_fe_obs + time_fe_obs + effect + epsilon

    # Event time range
    min_event = -n_pre
    max_event = n_post - 1
    event_times = list(range(min_event, max_event + 1))

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "time": time,
        "unit_id": unit_id,
        "event_time": event_time,
        "true_pretrend_effects": {k: 0.0 for k in range(min_event, 0)},
        "true_post_effects": {k: true_effect for k in range(0, max_event + 1)},
        "event_times": event_times,
        "treatment_time": treatment_time,
        "n_treated": n_treated,
        "n_control": n_control,
    }


def dgp_event_study_violated_pretrends(
    n_treated: int = 100,
    n_control: int = 100,
    n_pre: int = 5,
    n_post: int = 5,
    treatment_time: int = 5,
    true_effect: float = 2.0,
    pretrend_slope: float = 0.3,
    sigma: float = 1.0,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Event study DGP with violated pre-trends (anticipation effects).

    Pre-treatment effects increase linearly toward treatment.
    Tests detection of parallel trends violations.

    DGP:
        β_k = pretrend_slope × k for k < 0 (linear pre-trend)
        β_k = true_effect for k >= 0

    The slope creates anticipation effects approaching treatment.
    """
    rng = np.random.RandomState(random_state)

    n_units = n_treated + n_control
    n_periods = n_pre + n_post

    unit_id = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)

    treatment_unit = np.array([1] * n_treated + [0] * n_control)
    treatment = np.repeat(treatment_unit, n_periods)

    event_time = time - treatment_time

    unit_fe = rng.normal(0, 1.0, n_units)
    time_fe = rng.normal(0, 0.5, n_periods)
    unit_fe_obs = unit_fe[unit_id]
    time_fe_obs = time_fe[time]

    # Treatment effect with pre-trends
    effect = np.zeros(len(unit_id))
    treated_mask = treatment == 1

    # Pre-treatment: linear trend
    pre_mask = treated_mask & (event_time < 0)
    effect[pre_mask] = pretrend_slope * event_time[pre_mask]

    # Post-treatment: constant effect
    post_mask = treated_mask & (event_time >= 0)
    effect[post_mask] = true_effect

    epsilon = rng.normal(0, sigma, n_units * n_periods)
    outcomes = unit_fe_obs + time_fe_obs + effect + epsilon

    min_event = -n_pre
    max_event = n_post - 1
    event_times = list(range(min_event, max_event + 1))

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "time": time,
        "unit_id": unit_id,
        "event_time": event_time,
        "true_pretrend_effects": {k: pretrend_slope * k for k in range(min_event, 0)},
        "true_post_effects": {k: true_effect for k in range(0, max_event + 1)},
        "event_times": event_times,
        "treatment_time": treatment_time,
        "n_treated": n_treated,
        "n_control": n_control,
        "pretrend_slope": pretrend_slope,
    }


def dgp_event_study_dynamic(
    n_treated: int = 100,
    n_control: int = 100,
    n_pre: int = 5,
    n_post: int = 5,
    treatment_time: int = 5,
    effect_path: Optional[Dict[int, float]] = None,
    sigma: float = 1.0,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Event study DGP with dynamic (time-varying) treatment effects.

    Effects can vary by event time, testing whether event study
    correctly recovers the dynamic treatment effect path.

    DGP:
        β_k given by effect_path dictionary for each event time k

    Default effect_path: {0: 1.0, 1: 2.0, 2: 2.5, 3: 3.0, 4: 3.0}
    (effect grows then stabilizes)
    """
    rng = np.random.RandomState(random_state)

    if effect_path is None:
        effect_path = {0: 1.0, 1: 2.0, 2: 2.5, 3: 3.0, 4: 3.0}

    n_units = n_treated + n_control
    n_periods = n_pre + n_post

    unit_id = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)

    treatment_unit = np.array([1] * n_treated + [0] * n_control)
    treatment = np.repeat(treatment_unit, n_periods)

    event_time = time - treatment_time

    unit_fe = rng.normal(0, 1.0, n_units)
    time_fe = rng.normal(0, 0.5, n_periods)
    unit_fe_obs = unit_fe[unit_id]
    time_fe_obs = time_fe[time]

    # Dynamic treatment effects
    effect = np.zeros(len(unit_id))
    for k, tau_k in effect_path.items():
        mask = (treatment == 1) & (event_time == k)
        effect[mask] = tau_k

    epsilon = rng.normal(0, sigma, n_units * n_periods)
    outcomes = unit_fe_obs + time_fe_obs + effect + epsilon

    min_event = -n_pre
    max_event = n_post - 1
    event_times = list(range(min_event, max_event + 1))

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "time": time,
        "unit_id": unit_id,
        "event_time": event_time,
        "true_pretrend_effects": {k: 0.0 for k in range(min_event, 0)},
        "true_post_effects": effect_path,
        "event_times": event_times,
        "treatment_time": treatment_time,
        "n_treated": n_treated,
        "n_control": n_control,
    }
