# Import Random for bootstrap resampling
using Random

"""
Staggered Difference-in-Differences Implementation.

This module provides infrastructure for staggered DiD designs where treatment timing
varies across units (cohorts treated at different times).

⚠️ **WARNING**: The TWFE estimator in this module is BIASED with heterogeneous treatment effects.
Use Callaway-Sant'Anna or Sun-Abraham estimators instead (see callaway_santanna.jl and sun_abraham.jl).

# Mathematical Framework

## Staggered DiD Design
Treatment timing varies across units:
- Group g₁ treated at time t₁
- Group g₂ treated at time t₂
- Group g₃ never treated (control)

Creates multiple treatment "cohorts" based on first treatment time.

## TWFE Estimator (Biased!)
```math
Y_{it} = α_i + λ_t + τ·D_{it} + ε_{it}
```

where:
- α_i = unit fixed effects
- λ_t = time fixed effects
- D_{it} = 1 if unit i is treated at time t
- τ = average treatment effect (BIASED if effects are heterogeneous!)

## Why TWFE is Biased
With staggered adoption + heterogeneous effects:
- TWFE uses already-treated units as "controls" for newly-treated units
- Creates negative weights on some treatment effects
- Estimate can be opposite sign of true effect!

See: Goodman-Bacon (2021), de Chaisemartin & D'Haultfœuille (2020)

# Key References
- Goodman-Bacon, Andrew. 2021. "Difference-in-Differences with Variation in Treatment Timing."
  *Journal of Econometrics* 225(2): 254-277.
- Callaway, Brantly, and Pedro H.C. Sant'Anna. 2021. "Difference-in-Differences with
  Multiple Time Periods." *Journal of Econometrics* 225(2): 200-230.
- Sun, Liyang, and Sarah Abraham. 2021. "Estimating Dynamic Treatment Effects in Event
  Studies with Heterogeneous Treatment Effects." *Journal of Econometrics* 225(2): 175-199.
"""

using Statistics
using LinearAlgebra

# =============================================================================
# Staggered DiD Problem Type
# =============================================================================

"""
    StaggeredDiDProblem{T<:Real, P<:NamedTuple} <: AbstractDiDProblem{T,P}

Staggered Difference-in-Differences problem specification.

In staggered designs, treatment timing varies across units, creating multiple "cohorts"
based on when units first receive treatment.

# Fields
- `outcomes::Vector{T}`: Outcome variable (n_obs = n_units × n_periods)
- `treatment::Vector{Bool}`: Binary treatment indicator (0/1) for each observation
- `time::Vector{Int}`: Time period for each observation
- `unit_id::Vector{Int}`: Unit identifier for each observation
- `treatment_time::Vector{T}`: Treatment time for each unit (Inf for never-treated)
- `parameters::P`: Analysis parameters (alpha, etc.)

# Treatment Time Convention
- `treatment_time[i] = t`: Unit i first treated at time t
- `treatment_time[i] = Inf`: Unit i never treated (pure control)

# Validation
Constructor validates:
- All arrays have matching lengths
- treatment_time has one entry per unit
- Treatment times are within observed time range
- Variation exists in treatment timing (≥2 cohorts OR never-treated units)
- No units switch from treated back to untreated

# Examples
```julia
using CausalEstimators

# 3 cohorts: treated at t=3, t=5, and never-treated
n_units = 30  # 10 per cohort
n_periods = 10

# Create panel data
outcomes = Float64[]
treatment = Bool[]
time_vec = Int[]
unit_vec = Int[]

for unit in 1:n_units
    # Assign cohort
    if unit <=10
        treat_time = 3.0  # Early cohort
    elseif unit <= 20
        treat_time = 5.0  # Late cohort
    else
        treat_time = Inf  # Never treated
    end

    for t in 1:n_periods
        push!(outcomes, randn() + (t ≥ treat_time ? 2.0 : 0.0))  # Effect = 2.0
        push!(treatment, t ≥ treat_time && !isinf(treat_time))
        push!(time_vec, t)
        push!(unit_vec, unit)
    end
end

# Treatment time per unit
treatment_time = vcat(fill(3.0, 10), fill(5.0, 10), fill(Inf, 10))

problem = StaggeredDiDProblem(
    outcomes,
    treatment,
    time_vec,
    unit_vec,
    treatment_time,
    (alpha=0.05,)
)

# Identify cohorts
cohorts = unique(treatment_time[isfinite.(treatment_time)])  # [3.0, 5.0]
```

# Theory
Staggered DiD requires:
1. **Multiple treatment times**: Different groups treated at different times
2. **Never-treated OR late-treated controls**: Pure comparison group
3. **Parallel trends**: Within each (group, time) pair

Modern methods (Callaway-Sant'Anna, Sun-Abraham) address TWFE bias by:
- Constructing clean 2×2 comparisons (CS)
- Avoiding already-treated units as controls (SA)
"""
struct StaggeredDiDProblem{T<:Real,P<:NamedTuple} <: AbstractDiDProblem{T,P}
    outcomes::AbstractVector{T}
    treatment::AbstractVector{Bool}
    time::AbstractVector{Int}
    unit_id::AbstractVector{Int}
    treatment_time::AbstractVector{T}  # Treatment time per unit (Inf for never-treated)
    parameters::P

    function StaggeredDiDProblem(
        outcomes::AbstractVector{T},
        treatment::AbstractVector{Bool},
        time::AbstractVector{Int},
        unit_id::AbstractVector{Int},
        treatment_time::AbstractVector{T},
        parameters::P
    ) where {T<:Real,P<:NamedTuple}
        n_obs = length(outcomes)

        # Validate array lengths
        if length(treatment) != n_obs
            throw(ArgumentError("treatment must have same length as outcomes (got $(length(treatment)), expected $n_obs)"))
        end
        if length(time) != n_obs
            throw(ArgumentError("time must have same length as outcomes (got $(length(time)), expected $n_obs)"))
        end
        if length(unit_id) != n_obs
            throw(ArgumentError("unit_id must have same length as outcomes (got $(length(unit_id)), expected $n_obs)"))
        end

        # Check treatment_time length matches number of unique units
        unique_units = unique(unit_id)
        n_units = length(unique_units)

        if length(treatment_time) != n_units
            throw(ArgumentError(
                "treatment_time must have one entry per unit. " *
                "Got $(length(treatment_time)) entries but $n_units unique units"
            ))
        end

        # Check treatment times are within observed time range
        time_min = minimum(time)
        time_max = maximum(time)
        finite_treatment_times = treatment_time[isfinite.(treatment_time)]

        if length(finite_treatment_times) > 0
            if minimum(finite_treatment_times) < time_min
                throw(ArgumentError(
                    "treatment_time contains values ($(minimum(finite_treatment_times))) " *
                    "before first time period ($time_min)"
                ))
            end
            if maximum(finite_treatment_times) > time_max
                throw(ArgumentError(
                    "treatment_time contains values ($(maximum(finite_treatment_times))) " *
                    "after last time period ($time_max)"
                ))
            end
        end

        # Check for variation in treatment timing (required for staggered design)
        cohorts = unique(finite_treatment_times)
        never_treated = any(isinf.(treatment_time))

        if length(cohorts) < 2 && !never_treated
            throw(ArgumentError(
                "Staggered design requires variation in treatment timing. " *
                "Found only one cohort and no never-treated units. " *
                "For single treatment time, use DiDProblem (Classic 2×2 DiD) instead."
            ))
        end

        # Validate treatment consistency (units don't switch back from treated to untreated)
        for uid in unique_units
            unit_mask = unit_id .== uid
            unit_times = time[unit_mask]
            unit_treatment = treatment[unit_mask]

            # Sort by time
            sorted_idx = sortperm(unit_times)
            sorted_treatment = unit_treatment[sorted_idx]

            # Check for switches from treated back to untreated
            if any(sorted_treatment)  # Unit is ever treated
                first_treated_idx = findfirst(sorted_treatment)
                if any(.!sorted_treatment[first_treated_idx:end])
                    throw(ArgumentError(
                        "Unit $uid switches from treated back to untreated. " *
                        "Staggered DiD requires permanent treatment (once treated, always treated)."
                    ))
                end
            end
        end

        new{T,P}(outcomes, treatment, time, unit_id, treatment_time, parameters)
    end
end

# =============================================================================
# Helper Functions for Cohort Identification
# =============================================================================

"""
    identify_cohorts(treatment_time::Vector{<:Real})

Identify unique treatment cohorts (excluding never-treated).

# Arguments
- `treatment_time::Vector`: Treatment time per unit (Inf for never-treated)

# Returns
- `cohorts::Vector`: Sorted array of unique treatment times (finite only)

# Examples
```julia
treatment_time = [3.0, 5.0, 3.0, Inf, 5.0, Inf]
cohorts = identify_cohorts(treatment_time)  # [3.0, 5.0]
```
"""
function identify_cohorts(treatment_time::Vector{T}) where {T<:Real}
    finite_times = treatment_time[isfinite.(treatment_time)]
    return sort(unique(finite_times))
end

"""
    get_cohort_sizes(treatment_time::Vector{<:Real})

Count number of units in each cohort.

# Arguments
- `treatment_time::Vector`: Treatment time per unit (Inf for never-treated)

# Returns
- `sizes::Dict{Float64, Int}`: Mapping from cohort → count
  - Includes Inf key for never-treated count

# Examples
```julia
treatment_time = [3.0, 5.0, 3.0, Inf, 5.0, Inf]
sizes = get_cohort_sizes(treatment_time)
# Dict(3.0 => 2, 5.0 => 2, Inf => 2)
```
"""
function get_cohort_sizes(treatment_time::Vector{T}) where {T<:Real}
    sizes = Dict{T, Int}()
    for t in unique(treatment_time)
        sizes[t] = count(==(t), treatment_time)
    end
    return sizes
end

"""
    validate_cohort_variation(treatment_time::Vector{<:Real})

Check if there is sufficient variation in treatment timing for staggered design.

# Arguments
- `treatment_time::Vector`: Treatment time per unit (Inf for never-treated)

# Returns
- `true` if valid (≥2 cohorts OR ≥1 cohort + never-treated)

# Throws
- `ArgumentError`: If only one cohort and no never-treated units

# Examples
```julia
# Valid: 2 cohorts
validate_cohort_variation([3.0, 3.0, 5.0, 5.0])  # true

# Valid: 1 cohort + never-treated
validate_cohort_variation([3.0, 3.0, Inf, Inf])  # true

# Invalid: Only 1 cohort, no never-treated
validate_cohort_variation([3.0, 3.0, 3.0, 3.0])  # throws ArgumentError
```
"""
function validate_cohort_variation(treatment_time::Vector{T}) where {T<:Real}
    cohorts = identify_cohorts(treatment_time)
    never_treated = any(isinf.(treatment_time))

    if length(cohorts) < 2 && !never_treated
        throw(ArgumentError(
            "Insufficient variation in treatment timing. " *
            "Staggered design requires ≥2 cohorts OR ≥1 cohort + never-treated units. " *
            "Found $(length(cohorts)) cohort(s) and $(never_treated ? "some" : "no") never-treated units."
        ))
    end

    return true
end

# =============================================================================
# TWFE Estimator (BIASED - Use for comparison only)
# =============================================================================

"""
    StaggeredTWFE <: AbstractDiDEstimator

Two-Way Fixed Effects estimator for staggered DiD.

⚠️ **WARNING**: This estimator is BIASED when treatment effects are heterogeneous across
groups or time periods. Use Callaway-Sant'Anna or Sun-Abraham estimators instead.

# Fields
- `cluster_se::Bool`: Use cluster-robust standard errors (default: true, clusters by unit_id)

# Mathematical Framework

TWFE regression:
```math
Y_{it} = α_i + λ_t + τ·D_{it} + ε_{it}
```

where:
- α_i = unit fixed effects (absorbs time-invariant unit characteristics)
- λ_t = time fixed effects (absorbs common time trends)
- D_{it} = 1 if unit i is treated at time t, 0 otherwise
- τ = average treatment effect (BIASED!)

# Why Biased?

With staggered adoption and heterogeneous treatment effects:
1. **Forbidden comparisons**: TWFE compares newly-treated units to already-treated units
2. **Negative weights**: Some treatment effects get negative weights in τ̂
3. **Sign reversal**: τ̂ can have opposite sign of true ATT

Example (from Goodman-Bacon 2021):
- Early cohort: τ_early = +5
- Late cohort: τ_late = +10
- TWFE estimates: τ̂ = -2 (WRONG SIGN!)

# When TWFE is Okay

TWFE is unbiased when:
- Treatment effects are homogeneous (same τ for all groups/times)
- Only one treatment time (classic 2×2 DiD)
- All units treated at once (event study)

# Alternatives (Unbiased)

- **Callaway-Sant'Anna**: Constructs clean 2×2 comparisons, aggregates
- **Sun-Abraham**: Interaction-weighted estimator, excludes already-treated controls

# Examples
```julia
using CausalEstimators

# Create staggered problem (3 cohorts: t=3, t=5, never)
problem = StaggeredDiDProblem(...)

# TWFE (for comparison - expect bias!)
twfe_est = StaggeredTWFE(cluster_se=true)
solution = solve(problem, twfe_est)

println("TWFE estimate: \$(solution.estimate)")
println("WARNING: This may be biased if effects are heterogeneous!")

# Use modern methods instead
cs_est = CallawaySantAnna()
cs_solution = solve(problem, cs_est)
println("Callaway-Sant'Anna (unbiased): \$(cs_solution.estimate)")
```

# References
- Goodman-Bacon, Andrew. 2021. "Difference-in-Differences with Variation in Treatment Timing."
  *Journal of Econometrics* 225(2): 254-277.
- de Chaisemartin, Clément, and Xavier D'Haultfœuille. 2020. "Two-Way Fixed Effects
  Estimators with Heterogeneous Treatment Effects." *American Economic Review* 110(9): 2964-2996.
"""
Base.@kwdef struct StaggeredTWFE <: AbstractDiDEstimator
    cluster_se::Bool = true
end

"""
    solve(problem::StaggeredDiDProblem, estimator::StaggeredTWFE)

Estimate average treatment effect using Two-Way Fixed Effects (TWFE).

⚠️ **WARNING**: Displays bias warning when heterogeneous effects are likely.

# Algorithm
1. Demean outcomes by unit and time (removes fixed effects)
2. Demean treatment indicator by unit and time
3. Run OLS: Ŷ_demeaned = τ̂·D_demeaned + ε
4. Compute cluster-robust SEs (by unit_id)
5. Construct confidence intervals and p-values

# Returns
- `DiDSolution`: Standard DiD solution with:
  - `estimate`: TWFE τ̂ (POTENTIALLY BIASED!)
  - `se`: Cluster-robust standard error
  - `ci_lower`, `ci_upper`: 95% confidence interval
  - `p_value`: Two-sided p-value
  - `retcode`: :Success with bias warning, or :Warning if suspicious

# Theory

TWFE demeaning (within transformation):
```math
Ỹ_{it} = Y_{it} - Ȳ_i - Ȳ_t + Ȳ
D̃_{it} = D_{it} - D̄_i - D̄_t + D̄
```

where Ȳ_i = unit mean, Ȳ_t = time mean, Ȳ = grand mean.

Then OLS: Ỹ = τ̂·D̃ + ε

This is equivalent to:
```math
Y_{it} = α_i + λ_t + τ·D_{it} + ε_{it}
```

# Bias Mechanism

With heterogeneous effects τ_{it}:
- TWFE weights: w_{it} ∝ Var(D̃_{it})
- Negative weights possible when comparing:
  - Newly-treated to already-treated (forbidden comparison!)
  - Early-treated to late-treated (reverses roles)
- Result: τ̂_TWFE can be far from true ATT

See Goodman-Bacon decomposition for details.

# Examples
```julia
# Estimate TWFE (for comparison with modern methods)
solution = solve(problem, StaggeredTWFE())

# Check if bias warning displayed
if solution.retcode == :Warning
    println("Bias likely! Use Callaway-Sant'Anna or Sun-Abraham instead.")
end
```
"""
function solve(problem::StaggeredDiDProblem{T,P}, estimator::StaggeredTWFE) where {T<:Real,P<:NamedTuple}
    # Extract data
    Y = problem.outcomes
    D = convert(Vector{T}, problem.treatment)  # Convert Bool to T
    time = problem.time
    unit_id = problem.unit_id
    alpha = get(problem.parameters, :alpha, 0.05)

    n = length(Y)
    unique_units = unique(unit_id)
    unique_times = unique(time)
    n_units = length(unique_units)
    n_periods = length(unique_times)

    # =========================================================================
    # Step 1: TWFE Demeaning (Within Transformation)
    # =========================================================================

    # Compute unit means
    Y_unit_means = zeros(T, n)
    D_unit_means = zeros(T, n)
    for (i, uid) in enumerate(unique_units)
        mask = unit_id .== uid
        Y_unit_means[mask] .= mean(Y[mask])
        D_unit_means[mask] .= mean(D[mask])
    end

    # Compute time means
    Y_time_means = zeros(T, n)
    D_time_means = zeros(T, n)
    for t in unique_times
        mask = time .== t
        Y_time_means[mask] .= mean(Y[mask])
        D_time_means[mask] .= mean(D[mask])
    end

    # Grand means
    Y_grand_mean = mean(Y)
    D_grand_mean = mean(D)

    # Demean: Ỹ = Y - Ȳ_unit - Ȳ_time + Ȳ_grand
    Y_demeaned = Y .- Y_unit_means .- Y_time_means .+ Y_grand_mean
    D_demeaned = D .- D_unit_means .- D_time_means .+ D_grand_mean

    # =========================================================================
    # Step 2: OLS Regression (Demeaned Variables)
    # =========================================================================

    # OLS: Ỹ = τ̂·D̃ + ε
    # τ̂ = Cov(D̃, Ỹ) / Var(D̃)

    # Check for variation in demeaned treatment
    D_demeaned_var = var(D_demeaned)
    if D_demeaned_var < 1e-10
        # No variation after demeaning - cannot estimate
        return DiDSolution(
            estimate=T(NaN),
            se=T(NaN),
            ci_lower=T(NaN),
            ci_upper=T(NaN),
            p_value=T(NaN),
            t_stat=T(NaN),
            df=0,
            n_obs=n,
            n_treated=sum(problem.treatment),
            n_control=n - sum(problem.treatment),
            parallel_trends_test=nothing,
            retcode=:Failure
        )
    end

    # OLS coefficient
    tau_hat = cov(D_demeaned, Y_demeaned) / D_demeaned_var

    # Residuals
    residuals = Y_demeaned .- tau_hat .* D_demeaned

    # =========================================================================
    # Step 3: Cluster-Robust Standard Errors
    # =========================================================================

    if estimator.cluster_se
        # Cluster by unit_id (standard for panel data)
        # Sandwich estimator: V = (D̃'D̃)^{-1} * [Σ_c u_c'u_c] * (D̃'D̃)^{-1}

        # Meat of sandwich (cluster-robust)
        # Score for cluster c: s_c = Σ_i∈c (D̃_i × ε_i)
        # V = (D̃'D̃)^{-2} × Σ_c s_c²
        meat = zero(T)
        for uid in unique_units
            mask = unit_id .== uid
            D_c = D_demeaned[mask]
            u_c = residuals[mask]

            # Cluster score: element-wise multiply, then sum
            cluster_score = sum(D_c .* u_c)
            # Sum of squared cluster scores
            meat += cluster_score^2
        end

        # Note: This is 1D version of sandwich (scalar τ)

        # Bread of sandwich
        DtD = sum(D_demeaned .^ 2)  # D̃'D̃

        # Degrees of freedom: n_clusters - 1 (Bertrand et al. 2004)
        df = n_units - 1

        # Finite-sample adjustment (matching statsmodels)
        # adjustment = [G/(G-1)] * [(N-1)/(N-k)]
        # For TWFE: k = n_units + n_periods (number of fixed effects)
        k_fe = n_units + n_periods
        if (n_units - 1) > 0 && (n - k_fe) > 0
            cluster_adj = n_units / (n_units - 1)
            hc1_adj = (n - 1) / (n - k_fe)
            adjustment = cluster_adj * hc1_adj
        else
            adjustment = T(1.0)
        end

        # Variance (meat is already sum of squared scores, don't square again)
        V_tau = adjustment * meat / (DtD^2)
        se_tau = sqrt(V_tau)

    else
        # Heteroskedasticity-robust (HC1)
        # V = (D̃'D̃)^{-1} * Σ ε_i^2 D̃_i^2 * (D̃'D̃)^{-1}

        meat = sum((residuals .^ 2) .* (D_demeaned .^ 2))
        DtD = sum(D_demeaned .^ 2)

        # HC1 adjustment: n/(n-k)
        k_fe = n_units + n_periods
        adjustment = n / (n - k_fe)

        V_tau = adjustment * meat / (DtD^2)
        se_tau = sqrt(V_tau)

        # Degrees of freedom for non-clustered: n - k
        df = n - k_fe
    end

    # =========================================================================
    # Step 4: Inference
    # =========================================================================

    # T-statistic
    t_stat = tau_hat / se_tau

    # P-value (two-sided, t-distribution)
    # Using normal approximation for large df
    if df > 30
        # Normal approximation
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    else
        # T-distribution (using approximation)
        p_value = 2 * (1 - _t_cdf(abs(t_stat), df))
    end

    # Confidence interval (t-distribution)
    if df > 30
        t_crit = _normal_quantile(1 - alpha/2)
    else
        t_crit = _t_quantile(1 - alpha/2, df)
    end

    ci_lower = tau_hat - t_crit * se_tau
    ci_upper = tau_hat + t_crit * se_tau

    # =========================================================================
    # Step 5: Bias Warning
    # =========================================================================

    # Check if heterogeneous effects are likely
    cohorts = identify_cohorts(problem.treatment_time)
    n_cohorts = length(cohorts)

    # Display bias warning
    println()
    println("=" ^ 70)
    println("⚠️  TWFE BIAS WARNING")
    println("=" ^ 70)
    println("You are using Two-Way Fixed Effects (TWFE) with staggered adoption.")
    println()
    println("TWFE is BIASED when treatment effects are heterogeneous across:")
    println("  • Treatment cohorts (early vs late adopters)")
    println("  • Time periods (immediate vs delayed effects)")
    println()
    println("Your data has:")
    println("  • $n_cohorts treatment cohort(s): $cohorts")
    println("  • $n_periods time periods")
    println("  • $n_units total units")
    println()
    println("⚠️  RECOMMENDATION: Use modern DiD methods instead:")
    println("  • Callaway-Sant'Anna (2021) - Robust to heterogeneous effects")
    println("  • Sun-Abraham (2021) - Interaction-weighted estimator")
    println()
    println("TWFE estimate: $(round(tau_hat, digits=4)) (POTENTIALLY BIASED!)")
    println("=" ^ 70)
    println()

    # Set retcode based on concern level
    retcode = :Success  # Still succeeded, but with warning

    # Construct solution
    return DiDSolution(
        estimate=tau_hat,
        se=se_tau,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        t_stat=t_stat,
        df=df,
        n_obs=n,
        n_treated=sum(problem.treatment),
        n_control=n - sum(problem.treatment),
        parallel_trends_test=nothing,
        retcode=retcode
    )
end

# =============================================================================
# Statistical Utilities (Reused from classic_did.jl)
# =============================================================================

"""Normal CDF approximation (Abramowitz & Stegun)."""
function _normal_cdf(x::T) where {T<:Real}
    if x < zero(T)
        return one(T) - _normal_cdf(-x)
    end

    t = one(T) / (one(T) + T(0.2316419) * x)
    d = T(0.3989423) * exp(-x^2 / 2)

    prob = d * t * (T(0.31938153) +
                    t * (T(-0.356563782) +
                    t * (T(1.781477937) +
                    t * (T(-1.821255978) +
                    t * T(1.330274429)))))

    return one(T) - prob
end

"""Normal quantile approximation (Beasley-Springer-Moro)."""
function _normal_quantile(p::T) where {T<:Real}
    if p <= zero(T) || p >= one(T)
        throw(ArgumentError("p must be in (0, 1)"))
    end

    # For p close to 0.5, use direct approximation
    if abs(p - T(0.5)) < T(0.42)
        r = (p - T(0.5))^2
        return (p - T(0.5)) * (T(-0.295408) + r * T(4.54858)) /
               (one(T) + r * (T(-0.304568) + r * T(0.000319)))
    end

    # For tails, use log transformation
    if p < T(0.5)
        r = sqrt(-log(p))
    else
        r = sqrt(-log(one(T) - p))
    end

    num = T(2.32121) + r * (T(0.319381) + r * T(0.0008))
    den = one(T) + r * (T(0.099348) + r * T(0.002888))
    z = (num / den) - r

    return p < T(0.5) ? -z : z
end

"""T-distribution CDF approximation (Cornish-Fisher expansion)."""
function _t_cdf(t::T, df::Int) where {T<:Real}
    # For large df (>30), use normal approximation
    if df > 30
        return _normal_cdf(t)
    end

    # For small df, use approximation based on beta distribution
    # This is a simplified version - production code should use Distributions.jl

    x = df / (df + t^2)
    if df == 1
        # Cauchy distribution
        return T(0.5) + atan(t) / T(π)
    elseif df == 2
        # Exact formula for df=2
        return T(0.5) + t / (2 * sqrt(2 + t^2))
    else
        # General approximation (less accurate, but avoids dependency)
        # Convert to normal using Wilson-Hilferty transformation
        z = (t / sqrt(df)) * (1 - 1/(4*df)) + t^3 / (96 * df^2)
        return _normal_cdf(z)
    end
end

"""T-distribution quantile approximation."""
function _t_quantile(p::T, df::Int) where {T<:Real}
    # For large df (>30), use normal quantile
    if df > 30
        return _normal_quantile(p)
    end

    # For small df, use approximation
    # This is simplified - production code should use Distributions.jl

    z = _normal_quantile(p)

    # Cornish-Fisher expansion for t-distribution
    g1 = (z^3 + z) / (4 * df)
    g2 = (5*z^5 + 16*z^3 + 3*z) / (96 * df^2)

    return z + g1 + g2
end

# ============================================================================
# Callaway-Sant'Anna (2021) Estimator
# ============================================================================

"""
    CallawaySantAnna

Callaway-Sant'Anna (2021) group-time ATT estimator for staggered DiD designs.

Unlike TWFE, this estimator is unbiased even when treatment effects vary across
cohorts or over time. It computes clean 2×2 DiD comparisons for each cohort-time
cell and aggregates them with non-negative weights.

# Two-step procedure:
1. Estimate ATT(g,t) for each cohort g and time t ≥ g
   ATT(g,t) = E[Y_t - Y_{g-1} | G=g] - E[Y_t - Y_{g-1} | C]
   where C is the control group (never-treated or not-yet-treated)

2. Aggregate ATT(g,t) to summary estimand with three schemes:
   - Simple: Average over all (g,t) with weights = group size
   - Dynamic: Average by event time (k = t-g)
   - Group: Average by cohort g

# Fields
- `aggregation::Symbol`: Aggregation scheme (:simple, :dynamic, or :group)
- `control_group::Symbol`: Control group (:nevertreated or :notyettreated)
- `alpha::Float64`: Significance level for CIs (default: 0.05)
- `n_bootstrap::Int`: Number of bootstrap samples (default: 250)
- `random_seed::Union{Int,Nothing}`: Random seed for reproducibility

# References
Callaway, Brantly, and Pedro H.C. Sant'Anna. 2021. "Difference-in-Differences with
Multiple Time Periods." Journal of Econometrics 225(2): 200-230.
https://doi.org/10.1016/j.jeconom.2020.12.001
"""
Base.@kwdef struct CallawaySantAnna <: AbstractDiDEstimator
    aggregation::Symbol = :simple
    control_group::Symbol = :nevertreated
    alpha::Float64 = 0.05
    n_bootstrap::Int = 250
    random_seed::Union{Int,Nothing} = nothing

    function CallawaySantAnna(aggregation, control_group, alpha, n_bootstrap, random_seed)
        # Validate aggregation
        if aggregation ∉ [:simple, :dynamic, :group]
            throw(ArgumentError(
                "aggregation must be :simple, :dynamic, or :group. Got: $aggregation"
            ))
        end

        # Validate control_group
        if control_group ∉ [:nevertreated, :notyettreated]
            throw(ArgumentError(
                "control_group must be :nevertreated or :notyettreated. Got: $control_group"
            ))
        end

        # Validate alpha
        if !(0 < alpha < 1)
            throw(ArgumentError(
                "alpha must be between 0 and 1. Got: $alpha"
            ))
        end

        # Validate n_bootstrap
        if n_bootstrap < 50
            throw(ArgumentError(
                "n_bootstrap must be >= 50 for reliable inference. Got: $n_bootstrap"
            ))
        end

        new(aggregation, control_group, alpha, n_bootstrap, random_seed)
    end
end

"""
    _compute_att_gt(problem, control_group)

Compute ATT(g,t) for each cohort g and time t.

ATT(g,t) = E[Y_t - Y_{g-1} | G=g] - E[Y_t - Y_{g-1} | C]

where:
- g: Treatment cohort (time when cohort is first treated)
- t: Current time period (t ≥ g for post-treatment)
- Y_t: Outcome at time t
- Y_{g-1}: Outcome at time g-1 (period before treatment)
- C: Control group (never-treated or not-yet-treated at time t)

# Returns
Vector of NamedTuples with fields:
- cohort: Treatment cohort g
- time: Time period t
- event_time: Event time (t - g)
- att: ATT(g,t) estimate
- weight: Number of treated units in cohort g
- n_treated: Number of treated observations
- n_control: Number of control observations used
"""
function _compute_att_gt(
    problem::StaggeredDiDProblem{T,P},
    control_group::Symbol
) where {T<:Real,P<:NamedTuple}

    Y = problem.outcomes
    time = problem.time
    unit_id = problem.unit_id
    treatment_time = problem.treatment_time

    # Get cohorts and periods
    cohorts = identify_cohorts(treatment_time)
    periods = sort(unique(time))

    # Get unique units (sorted for alignment with treatment_time)
    unique_units = sort(unique(unit_id))

    att_gt_results = NamedTuple{
        (:cohort, :time, :event_time, :att, :weight, :n_treated, :n_control),
        Tuple{Int, Int, Int, T, Int, Int, Int}
    }[]

    # Iterate over cohorts and time periods
    for g in cohorts
        # Find units in cohort g
        cohort_mask = treatment_time .== g
        cohort_units = unique_units[cohort_mask]

        for t in periods
            # Only compute ATT for post-treatment periods (t >= g)
            if t < g
                continue
            end

            # Get pre-treatment period (g-1)
            pre_period = Int(g - 1)

            # Skip if pre-period not in data
            if pre_period ∉ periods
                continue
            end

            # Compute ATT(g,t) using double difference
            # Treated group: units in cohort g
            treated_t = _get_outcome_for_units(Y, unit_id, time, cohort_units, t)
            treated_pre = _get_outcome_for_units(Y, unit_id, time, cohort_units, pre_period)
            treated_diff = treated_t .- treated_pre

            # Control group: depends on control_group parameter
            if control_group == :nevertreated
                # Never-treated units (treatment_time = Inf)
                never_treated_mask = isinf.(treatment_time)
                control_units = unique_units[never_treated_mask]
            else  # :notyettreated
                # Not-yet-treated at time t (includes never-treated and future cohorts)
                not_yet_mask = treatment_time .> t
                control_units = unique_units[not_yet_mask]
            end

            # Skip if no control units available
            if isempty(control_units)
                continue
            end

            control_t = _get_outcome_for_units(Y, unit_id, time, control_units, t)
            control_pre = _get_outcome_for_units(Y, unit_id, time, control_units, pre_period)
            control_diff = control_t .- control_pre

            # ATT(g,t) = double difference
            att_gt = mean(treated_diff) - mean(control_diff)

            push!(att_gt_results, (
                cohort = Int(g),
                time = Int(t),
                event_time = Int(t - g),
                att = att_gt,
                weight = length(cohort_units),  # Number of units in cohort
                n_treated = length(treated_diff),
                n_control = length(control_diff)
            ))
        end
    end

    return att_gt_results
end

"""
    _get_outcome_for_units(outcomes, unit_id, time, units, time_period)

Get outcomes for specific units at a specific time period.

# Arguments
- `outcomes`: Outcome vector
- `unit_id`: Unit identifier vector
- `time`: Time period vector
- `units`: Array of unit IDs to select
- `time_period`: Time period to select

# Returns
Array of outcomes for those units at that time
"""
function _get_outcome_for_units(
    outcomes::AbstractVector{T},
    unit_id::AbstractVector{Int},
    time::AbstractVector{Int},
    units::AbstractVector{Int},
    time_period::Int
) where {T<:Real}

    mask = (in.(unit_id, Ref(Set(units)))) .& (time .== time_period)
    return outcomes[mask]
end

"""
    _aggregate_simple(att_gt_results)

Simple aggregation: weighted average over all ATT(g,t).

Weights are group sizes (number of units in each cohort).

# Returns
- `att`: Weighted average ATT
- `weights`: Array of weights used
"""
function _aggregate_simple(att_gt_results::Vector{<:NamedTuple})
    if isempty(att_gt_results)
        throw(ArgumentError("No ATT(g,t) estimates to aggregate"))
    end

    weights = Float64[r.weight for r in att_gt_results]
    atts = Float64[r.att for r in att_gt_results]

    # Weighted average
    att = sum(atts .* weights) / sum(weights)

    return att, weights
end

"""
    _aggregate_dynamic(att_gt_results)

Dynamic aggregation: Average ATT by event time (k = t - g).

# Returns
- `att_dynamic`: Dict mapping event_time → ATT
- `att_overall`: Overall ATT (average over all event times)
- `weights`: Array of weights used
"""
function _aggregate_dynamic(att_gt_results::Vector{<:NamedTuple})
    if isempty(att_gt_results)
        throw(ArgumentError("No ATT(g,t) estimates to aggregate"))
    end

    # Group by event time
    event_times = sort(unique([r.event_time for r in att_gt_results]))
    att_dynamic = Dict{Int, Float64}()

    all_atts = Float64[]
    all_weights = Float64[]

    for k in event_times
        # Filter results for this event time
        k_results = filter(r -> r.event_time == k, att_gt_results)

        weights = Float64[r.weight for r in k_results]
        atts = Float64[r.att for r in k_results]

        # Weighted average for this event time
        att_k = sum(atts .* weights) / sum(weights)
        att_dynamic[k] = att_k

        push!(all_atts, att_k)
        push!(all_weights, sum(weights))
    end

    # Overall ATT: average over event times weighted by total group size at each event time
    att_overall = sum(all_atts .* all_weights) / sum(all_weights)

    return att_dynamic, att_overall, all_weights
end

"""
    _aggregate_group(att_gt_results)

Group aggregation: Average ATT by cohort g.

# Returns
- `att_group`: Dict mapping cohort → ATT
- `att_overall`: Overall ATT (average over all cohorts)
- `weights`: Array of weights used
"""
function _aggregate_group(att_gt_results::Vector{<:NamedTuple})
    if isempty(att_gt_results)
        throw(ArgumentError("No ATT(g,t) estimates to aggregate"))
    end

    # Group by cohort
    cohorts = sort(unique([r.cohort for r in att_gt_results]))
    att_group = Dict{Int, Float64}()

    all_atts = Float64[]
    all_weights = Float64[]

    for g in cohorts
        # Filter results for this cohort
        g_results = filter(r -> r.cohort == g, att_gt_results)

        weights = Float64[r.weight for r in g_results]
        atts = Float64[r.att for r in g_results]

        # Weighted average for this cohort (over time periods)
        att_g = sum(atts .* weights) / sum(weights)
        att_group[g] = att_g

        push!(all_atts, att_g)
        push!(all_weights, g_results[1].weight)  # Cohort size (same across times)
    end

    # Overall ATT: average over cohorts weighted by cohort size
    att_overall = sum(all_atts .* all_weights) / sum(all_weights)

    return att_group, att_overall, all_weights
end

"""
    _bootstrap_resample(problem, rng)

Resample units with replacement for bootstrap.

# Arguments
- `problem`: Original StaggeredDiDProblem
- `rng`: Random number generator

# Returns
Resampled StaggeredDiDProblem (same number of units, resampled with replacement)
"""
function _bootstrap_resample(
    problem::StaggeredDiDProblem{T,P},
    rng
) where {T<:Real,P<:NamedTuple}

    # Get unique units
    unique_units = sort(unique(problem.unit_id))
    n_units = length(unique_units)

    # Resample units with replacement
    resampled_units = rand(rng, unique_units, n_units)

    # Build resampled data
    resampled_outcomes = T[]
    resampled_treatment = Bool[]
    resampled_time = Int[]
    resampled_unit_id = Int[]
    resampled_treatment_time = T[]

    new_unit_id = 0
    for unit in resampled_units
        # Get all observations for this unit
        unit_mask = problem.unit_id .== unit

        append!(resampled_outcomes, problem.outcomes[unit_mask])
        append!(resampled_treatment, problem.treatment[unit_mask])
        append!(resampled_time, problem.time[unit_mask])

        # Assign new unit ID (to handle duplicates from resampling)
        n_obs_for_unit = sum(unit_mask)
        append!(resampled_unit_id, fill(new_unit_id, n_obs_for_unit))

        # Get treatment time for this unit
        unit_idx = findfirst(==(unit), unique_units)
        push!(resampled_treatment_time, problem.treatment_time[unit_idx])

        new_unit_id += 1
    end

    # Create resampled problem (use positional arguments)
    return StaggeredDiDProblem(
        resampled_outcomes,
        resampled_treatment,
        resampled_time,
        resampled_unit_id,
        resampled_treatment_time,
        problem.parameters
    )
end

"""
    solve(problem::StaggeredDiDProblem, estimator::CallawaySantAnna)

Estimate causal effect using Callaway-Sant'Anna (2021) group-time ATT estimator.

This estimator is unbiased with heterogeneous treatment effects, unlike TWFE.

# Algorithm
1. Compute ATT(g,t) for all cohort-time cells using clean 2×2 DiD
2. Aggregate ATT(g,t) using specified scheme (simple/dynamic/group)
3. Compute bootstrap standard errors (resampling units)

# Returns
NamedTuple with:
- `att`: Overall ATT estimate
- `se`: Bootstrap standard error
- `t_stat`: t-statistic for H₀: ATT=0
- `p_value`: Two-sided p-value
- `ci_lower`, `ci_upper`: (1-α)×100% confidence interval
- `att_gt`: Vector of ATT(g,t) for each cohort × time
- `aggregated`: Aggregation-specific results (Dict or Float64)
- `control_group`: Control group used (:nevertreated or :notyettreated)
- `n_bootstrap`: Number of bootstrap samples
- `n_cohorts`: Number of treatment cohorts
- `n_obs`: Total observations
- `retcode`: :Success or :Failure

# Throws
- `ArgumentError`: If no never-treated units and control_group=:nevertreated,
                  or if bootstrap fails
"""
function solve(
    problem::StaggeredDiDProblem{T,P},
    estimator::CallawaySantAnna
) where {T<:Real,P<:NamedTuple}

    # Validate control group availability
    never_treated_mask = isinf.(problem.treatment_time)
    if estimator.control_group == :nevertreated && !any(never_treated_mask)
        throw(ArgumentError(
            "control_group=:nevertreated requires never-treated units, but none found. " *
            "Use control_group=:notyettreated or add never-treated units to data."
        ))
    end

    # Set random seed if specified
    if !isnothing(estimator.random_seed)
        rng = Random.MersenneTwister(estimator.random_seed)
    else
        rng = Random.GLOBAL_RNG
    end

    # Step 1: Compute ATT(g,t) for all cohort-time cells
    att_gt_results = _compute_att_gt(problem, estimator.control_group)

    if isempty(att_gt_results)
        throw(ArgumentError(
            "No ATT(g,t) estimates computed. Check that data has sufficient " *
            "pre-treatment periods and control units."
        ))
    end

    # Step 2: Aggregate ATT(g,t)
    if estimator.aggregation == :simple
        att, weights_used = _aggregate_simple(att_gt_results)
        aggregated = att
    elseif estimator.aggregation == :dynamic
        att_dynamic, att, weights_used = _aggregate_dynamic(att_gt_results)
        aggregated = att_dynamic
    elseif estimator.aggregation == :group
        att_group, att, weights_used = _aggregate_group(att_gt_results)
        aggregated = att_group
    end

    # Step 3: Bootstrap standard errors
    bootstrap_estimates = Float64[]
    bootstrap_errors = []  # Track errors for debugging
    for i in 1:estimator.n_bootstrap
        try
            # Resample units with replacement
            boot_problem = _bootstrap_resample(problem, rng)

            # Compute ATT(g,t) on bootstrap sample
            boot_att_gt = _compute_att_gt(boot_problem, estimator.control_group)

            # Skip if no estimates (rare edge case)
            if isempty(boot_att_gt)
                push!(bootstrap_errors, ("Sample $i", "No ATT(g,t) estimates"))
                continue
            end

            # Aggregate bootstrap ATT(g,t) the same way
            if estimator.aggregation == :simple
                boot_att, _ = _aggregate_simple(boot_att_gt)
            elseif estimator.aggregation == :dynamic
                _, boot_att, _ = _aggregate_dynamic(boot_att_gt)
            elseif estimator.aggregation == :group
                _, boot_att, _ = _aggregate_group(boot_att_gt)
            end

            push!(bootstrap_estimates, boot_att)
        catch e
            # Skip bootstrap samples that fail (e.g., no control units after resample)
            push!(bootstrap_errors, ("Sample $i", string(e)))
            continue
        end
    end

    # Debug: Show first 5 errors if bootstrap fails
    if length(bootstrap_estimates) == 0 && length(bootstrap_errors) > 0
        println("\n⚠️  Bootstrap Diagnostic: First 5 errors:")
        for (i, (sample, error)) in enumerate(bootstrap_errors[1:min(5, length(bootstrap_errors))])
            println("  $sample: $error")
        end
        println()
    end

    # Check bootstrap success rate
    success_rate = length(bootstrap_estimates) / estimator.n_bootstrap
    if success_rate < 0.8
        throw(ArgumentError(
            "Bootstrap failed: Only $(length(bootstrap_estimates))/$(estimator.n_bootstrap) " *
            "samples succeeded ($(round(success_rate*100, digits=1))%). " *
            "Data may be too small or imbalanced for bootstrap inference."
        ))
    end

    # Compute bootstrap SE
    se = std(bootstrap_estimates, corrected=true)

    # Inference
    t_stat = se > 0 ? att / se : T(Inf)
    df = length(bootstrap_estimates) - 1
    p_value = 2 * (1 - _t_cdf(abs(t_stat), df))

    # Confidence interval (percentile method)
    alpha = estimator.alpha
    ci_lower = quantile(bootstrap_estimates, alpha / 2)
    ci_upper = quantile(bootstrap_estimates, 1 - alpha / 2)

    # Count cohorts
    cohorts = identify_cohorts(problem.treatment_time)
    n_cohorts = length(cohorts)

    return (
        att = att,
        se = se,
        t_stat = t_stat,
        p_value = p_value,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        att_gt = att_gt_results,
        aggregated = aggregated,
        control_group = estimator.control_group,
        n_bootstrap = estimator.n_bootstrap,
        n_cohorts = n_cohorts,
        n_obs = length(problem.outcomes),
        retcode = :Success
    )
end

# ============================================================================
# Sun-Abraham (2021) Estimator
# ============================================================================

"""
    SunAbraham

Sun-Abraham (2021) interaction-weighted estimator for staggered DiD designs.

Unlike TWFE, this estimator uses cohort × event time interactions with proper
weighting to avoid bias from heterogeneous treatment effects.

# Regression model:
Y_it = α_i + λ_t + Σ_{g,l} β_{g,l}·D_it^{g,l} + ε_it

where:
- α_i: Unit fixed effects
- λ_t: Time fixed effects
- D_it^{g,l} = 1{G_i = g}·1{t - G_i = l} (cohort g × event time l interaction)
- β_{g,l}: Treatment effect for cohort g at event time l (relative to treatment)

Then aggregate:
ATT = Σ_{g,l} w_{g,l}·β_{g,l}

where w_{g,l} are the share of treated observations in cohort g at event time l:
w_{g,l} = N_{g,l} / Σ_{g',l'} N_{g',l'}

# Fields
- `alpha::Float64`: Significance level for CIs (default: 0.05)
- `cluster_se::Bool`: Use cluster-robust SEs at unit level (default: true)

# References
Sun, Liyang, and Sarah Abraham. 2021. "Estimating Dynamic Treatment Effects in Event
Studies with Heterogeneous Treatment Effects." Journal of Econometrics 225(2): 175-199.
https://doi.org/10.1016/j.jeconom.2020.12.001
"""
Base.@kwdef struct SunAbraham <: AbstractDiDEstimator
    alpha::Float64 = 0.05
    cluster_se::Bool = true

    function SunAbraham(alpha, cluster_se)
        # Validate alpha
        if !(0 < alpha < 1)
            throw(ArgumentError(
                "alpha must be between 0 and 1. Got: $alpha"
            ))
        end

        new(alpha, cluster_se)
    end
end

"""
    _create_interaction_dummies(problem)

Create cohort × event time interaction dummies.

For each cohort g and event time l (where l ≥ 0, post-treatment):
    D_it^{g,l} = 1{G_i = g}·1{t - G_i = l}

# Returns
- `interaction_map`: Dict mapping (cohort, event_time) → interaction dummy vector
- `interaction_keys`: Sorted list of (cohort, event_time) tuples
"""
function _create_interaction_dummies(problem::StaggeredDiDProblem{T,P}) where {T<:Real,P<:NamedTuple}
    n = length(problem.outcomes)
    unit_id = problem.unit_id
    time = problem.time
    treatment_time = problem.treatment_time

    # Get unique units (sorted for alignment)
    unique_units = sort(unique(unit_id))

    # Map unit_id to treatment_time
    unit_to_treatment_time = Dict{Int, T}()
    for (i, uid) in enumerate(unique_units)
        unit_to_treatment_time[uid] = treatment_time[i]
    end

    # Compute event time for each observation
    event_time = zeros(Int, n)
    for i in 1:n
        uid = unit_id[i]
        tt = unit_to_treatment_time[uid]
        if isfinite(tt)
            event_time[i] = time[i] - Int(tt)
        else
            event_time[i] = -999  # Sentinel for never-treated
        end
    end

    # Create interaction dummies for each cohort × event time
    # Only for post-treatment periods (event_time >= 0) and treated units
    cohorts = identify_cohorts(treatment_time)

    interaction_map = Dict{Tuple{Int, Int}, Vector{Float64}}()
    interaction_keys = Tuple{Int, Int}[]

    for g in cohorts
        # Find observations where unit is in cohort g and event_time >= 0
        for i in 1:n
            uid = unit_id[i]
            tt = unit_to_treatment_time[uid]

            if tt == g && event_time[i] >= 0
                l = event_time[i]
                key = (Int(g), l)

                # Create dummy if doesn't exist
                if !haskey(interaction_map, key)
                    interaction_map[key] = zeros(Float64, n)
                    push!(interaction_keys, key)
                end

                # Set dummy to 1 for this observation
                interaction_map[key][i] = 1.0
            end
        end
    end

    # Sort keys for consistent ordering
    sort!(interaction_keys)

    return interaction_map, interaction_keys
end

"""
    _fit_sun_abraham_regression(problem, interaction_map, interaction_keys, cluster_se)

Fit Sun-Abraham regression with unit + time FE + interactions.

Y_it = α_i + λ_t + Σ_{g,l} β_{g,l}·D_it^{g,l} + ε_it

# Returns
- `coefficients`: Dict mapping interaction key → coefficient
- `cov_matrix`: Covariance matrix of interaction coefficients
- `df_resid`: Residual degrees of freedom
"""
function _fit_sun_abraham_regression(
    problem::StaggeredDiDProblem{T,P},
    interaction_map::Dict{Tuple{Int, Int}, Vector{Float64}},
    interaction_keys::Vector{Tuple{Int, Int}},
    cluster_se::Bool
) where {T<:Real,P<:NamedTuple}

    n = length(problem.outcomes)
    Y = problem.outcomes
    unit_id = problem.unit_id
    time = problem.time

    # Get unique units and times
    unique_units = sort(unique(unit_id))
    unique_times = sort(unique(time))

    n_units = length(unique_units)
    n_times = length(unique_times)

    # Create unit fixed effects (drop first for identification)
    unit_dummies = zeros(Float64, n, n_units - 1)
    unit_map = Dict(uid => i for (i, uid) in enumerate(unique_units))
    for i in 1:n
        uid_idx = unit_map[unit_id[i]]
        if uid_idx > 1  # Drop first unit
            unit_dummies[i, uid_idx - 1] = 1.0
        end
    end

    # Create time fixed effects (drop first for identification)
    time_dummies = zeros(Float64, n, n_times - 1)
    time_map = Dict(t => i for (i, t) in enumerate(unique_times))
    for i in 1:n
        t_idx = time_map[time[i]]
        if t_idx > 1  # Drop first time
            time_dummies[i, t_idx - 1] = 1.0
        end
    end

    # Construct design matrix: [interactions, unit FE, time FE]
    n_interactions = length(interaction_keys)
    X = zeros(Float64, n, n_interactions + (n_units - 1) + (n_times - 1))

    # Add interaction dummies (first columns)
    for (col_idx, key) in enumerate(interaction_keys)
        X[:, col_idx] = interaction_map[key]
    end

    # Add unit fixed effects
    X[:, (n_interactions + 1):(n_interactions + n_units - 1)] = unit_dummies

    # Add time fixed effects
    X[:, (n_interactions + n_units):(n_interactions + n_units - 1 + n_times - 1)] = time_dummies

    # Fit OLS regression
    XtX = X' * X
    XtY = X' * Y

    # Solve normal equations
    beta_hat = XtX \ XtY

    # Compute residuals
    Y_hat = X * beta_hat
    residuals = Y .- Y_hat

    # Degrees of freedom
    k = size(X, 2)
    df_resid = n - k

    # Compute covariance matrix
    if cluster_se
        # Cluster-robust covariance matrix (clustered by unit)
        # Σ = (X'X)^{-1} × [Σ_c u_c'u_c] × (X'X)^{-1}
        XtX_inv = inv(XtX)

        # Compute meat matrix
        meat = zeros(Float64, k, k)
        for uid in unique_units
            unit_mask = unit_id .== uid
            X_c = X[unit_mask, :]
            u_c = residuals[unit_mask]

            meat += X_c' * (u_c * u_c') * X_c
        end

        # Adjustment factor
        n_clusters = n_units
        cluster_adj = n_clusters / (n_clusters - 1)
        hc1_adj = (n - 1) / (n - k)
        adjustment = cluster_adj * hc1_adj

        cov_beta = adjustment .* (XtX_inv * meat * XtX_inv)
    else
        # Homoskedastic covariance
        sigma2 = sum(residuals .^ 2) / df_resid
        cov_beta = sigma2 .* inv(XtX)
    end

    # Extract coefficients and covariance for interaction terms only
    coefficients = Dict{Tuple{Int, Int}, Float64}()
    for (col_idx, key) in enumerate(interaction_keys)
        coefficients[key] = beta_hat[col_idx]
    end

    # Extract covariance matrix for interaction terms only
    cov_matrix = cov_beta[1:n_interactions, 1:n_interactions]

    return coefficients, cov_matrix, df_resid
end

"""
    _extract_sun_abraham_cohort_effects(coefficients, cov_matrix, interaction_keys, alpha, df_resid)

Extract cohort × event time coefficients β_{g,l} with inference.

# Returns
Vector of NamedTuples with fields:
- cohort: Cohort g
- event_time: Event time l
- coef: β_{g,l}
- se: Standard error
- t_stat: t-statistic
- p_value: p-value
- ci_lower, ci_upper: Confidence interval
"""
function _extract_sun_abraham_cohort_effects(
    coefficients::Dict{Tuple{Int, Int}, Float64},
    cov_matrix::Matrix{Float64},
    interaction_keys::Vector{Tuple{Int, Int}},
    alpha::Float64,
    df_resid::Int
)
    cohort_effects = NamedTuple{
        (:cohort, :event_time, :coef, :se, :t_stat, :p_value, :ci_lower, :ci_upper),
        Tuple{Int, Int, Float64, Float64, Float64, Float64, Float64, Float64}
    }[]

    for (idx, (g, l)) in enumerate(interaction_keys)
        coef = coefficients[(g, l)]
        se = sqrt(cov_matrix[idx, idx])

        t_stat = se > 0 ? coef / se : Inf
        p_value = 2 * (1 - _t_cdf(abs(t_stat), df_resid))

        # Confidence interval
        t_crit = _t_quantile(1 - alpha / 2, df_resid)
        ci_lower = coef - t_crit * se
        ci_upper = coef + t_crit * se

        push!(cohort_effects, (
            cohort = g,
            event_time = l,
            coef = coef,
            se = se,
            t_stat = t_stat,
            p_value = p_value,
            ci_lower = ci_lower,
            ci_upper = ci_upper
        ))
    end

    return cohort_effects
end

"""
    _compute_sun_abraham_weights(problem, interaction_map, interaction_keys)

Compute sample share weights w_{g,l} for aggregation.

w_{g,l} = N_{g,l} / Σ_{g',l'} N_{g',l'}

where N_{g,l} is the number of treated observations with cohort g at event time l.

# Returns
Vector of NamedTuples with fields:
- cohort: Cohort g
- event_time: Event time l
- weight: w_{g,l}
- n_obs: N_{g,l}
"""
function _compute_sun_abraham_weights(
    problem::StaggeredDiDProblem{T,P},
    interaction_map::Dict{Tuple{Int, Int}, Vector{Float64}},
    interaction_keys::Vector{Tuple{Int, Int}}
) where {T<:Real,P<:NamedTuple}

    # Count treated observations for each cohort × event time
    n_obs_dict = Dict{Tuple{Int, Int}, Int}()
    total_treated = 0

    for (g, l) in interaction_keys
        n_obs = Int(sum(interaction_map[(g, l)]))
        n_obs_dict[(g, l)] = n_obs
        total_treated += n_obs
    end

    # Compute weights as proportions
    weights = NamedTuple{
        (:cohort, :event_time, :weight, :n_obs),
        Tuple{Int, Int, Float64, Int}
    }[]

    for (g, l) in interaction_keys
        n_obs = n_obs_dict[(g, l)]
        weight = total_treated > 0 ? n_obs / total_treated : 0.0

        push!(weights, (
            cohort = g,
            event_time = l,
            weight = weight,
            n_obs = n_obs
        ))
    end

    return weights
end

"""
    _aggregate_sun_abraham_att(cohort_effects, weights, cov_matrix)

Aggregate cohort effects using sample share weights.

ATT = Σ_{g,l} w_{g,l}·β_{g,l}

Standard error via delta method:
    Var(ATT) = w' Σ w
where Σ is the covariance matrix of β_{g,l}

# Returns
- `att`: Aggregated ATT
- `se`: Standard error
"""
function _aggregate_sun_abraham_att(
    cohort_effects::Vector{<:NamedTuple},
    weights::Vector{<:NamedTuple},
    cov_matrix::Matrix{Float64}
)
    # Compute weighted average
    att = 0.0
    for i in 1:length(cohort_effects)
        att += cohort_effects[i].coef * weights[i].weight
    end

    # Standard error via delta method
    # Weights vector
    w = [weights[i].weight for i in 1:length(weights)]

    # Var(ATT) = w' Σ w
    var_att = w' * cov_matrix * w

    se = sqrt(var_att)

    return att, se
end

"""
    solve(problem::StaggeredDiDProblem, estimator::SunAbraham)

Estimate causal effect using Sun-Abraham (2021) interaction-weighted estimator.

This estimator is unbiased with heterogeneous treatment effects and provides clean
event study estimates without TWFE bias.

# Algorithm
1. Create cohort × event time interaction dummies D_it^{g,l}
2. Regression: Y_it = α_i + λ_t + Σ_{g,l} β_{g,l}·D_it^{g,l} + ε_it
3. Extract β_{g,l} coefficients with cluster-robust SEs
4. Compute weights w_{g,l} based on sample composition
5. Aggregate: ATT = Σ w_{g,l}·β_{g,l} with delta method SE

# Returns
NamedTuple with:
- `att`: Weighted average treatment effect
- `se`: Standard error (delta method with cluster-robust SEs if cluster_se=true)
- `t_stat`: t-statistic for H₀: ATT=0
- `p_value`: Two-sided p-value
- `ci_lower`, `ci_upper`: (1-α)×100% confidence interval
- `cohort_effects`: Vector of β_{g,l} for each cohort × event time
- `weights`: Vector of w_{g,l} weights used
- `n_obs`: Total observations
- `n_treated`: Total treated observations
- `n_control`: Total control observations
- `n_cohorts`: Number of treatment cohorts
- `cluster_se_used`: Whether cluster SEs were used
- `retcode`: :Success or :Failure

# Throws
- `ArgumentError`: If no never-treated units (need clean control group),
                  or if fewer than 2 cohorts
"""
function solve(
    problem::StaggeredDiDProblem{T,P},
    estimator::SunAbraham
) where {T<:Real,P<:NamedTuple}

    # Validate: Need never-treated units as clean control group
    never_treated_mask = isinf.(problem.treatment_time)
    if !any(never_treated_mask)
        throw(ArgumentError(
            "Sun-Abraham estimator requires never-treated units as control group. " *
            "No never-treated units found in data. Consider using Callaway-Sant'Anna " *
            "with control_group=:notyettreated if no never-treated units available."
        ))
    end

    # Validate: Need at least 2 cohorts
    cohorts = identify_cohorts(problem.treatment_time)
    n_cohorts = length(cohorts)
    if n_cohorts < 2
        throw(ArgumentError(
            "Sun-Abraham estimator requires at least 2 cohorts. Found $n_cohorts. " *
            "For single treatment time, use EventStudy instead."
        ))
    end

    # Step 1: Create cohort × event time interactions
    interaction_map, interaction_keys = _create_interaction_dummies(problem)

    # Step 2: Run regression with unit + time FE + interactions
    coefficients, cov_matrix, df_resid = _fit_sun_abraham_regression(
        problem, interaction_map, interaction_keys, estimator.cluster_se
    )

    # Step 3: Extract cohort effects β_{g,l}
    cohort_effects = _extract_sun_abraham_cohort_effects(
        coefficients, cov_matrix, interaction_keys, estimator.alpha, df_resid
    )

    # Step 4: Compute weights w_{g,l} based on sample composition
    weights = _compute_sun_abraham_weights(problem, interaction_map, interaction_keys)

    # Step 5: Aggregate ATT = Σ w_{g,l}·β_{g,l}
    att, se_att = _aggregate_sun_abraham_att(cohort_effects, weights, cov_matrix)

    # Inference
    t_stat = se_att > 0 ? att / se_att : T(Inf)
    p_value = 2 * (1 - _t_cdf(abs(t_stat), df_resid))

    # Confidence interval using delta method
    t_crit = _t_quantile(1 - estimator.alpha / 2, df_resid)
    ci_lower = att - t_crit * se_att
    ci_upper = att + t_crit * se_att

    # Diagnostics
    n_obs = length(problem.outcomes)
    n_treated = sum(problem.treatment)
    n_control = n_obs - n_treated

    return (
        att = att,
        se = se_att,
        t_stat = t_stat,
        p_value = p_value,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        cohort_effects = cohort_effects,
        weights = weights,
        n_obs = n_obs,
        n_treated = n_treated,
        n_control = n_control,
        n_cohorts = n_cohorts,
        cluster_se_used = estimator.cluster_se,
        retcode = :Success
    )
end
