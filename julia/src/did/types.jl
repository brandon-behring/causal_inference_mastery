"""
Difference-in-Differences (DiD) type definitions.

Following SciML Problem-Estimator-Solution pattern for causal inference.
"""

# =============================================================================
# Abstract Types
# =============================================================================

"""
    AbstractDiDProblem{T,P}

Abstract type for DiD problem specifications.

Type parameters match AbstractCausalProblem:
- `T`: Outcome type (Float64, Float32, etc.)
- `P`: Parameter type (NamedTuple)
"""
abstract type AbstractDiDProblem{T,P} <: AbstractCausalProblem{T,P} end

"""
    AbstractDiDEstimator

Abstract type for DiD estimators.
"""
abstract type AbstractDiDEstimator <: AbstractCausalEstimator end

"""
    AbstractDiDSolution

Abstract type for DiD estimation results.
"""
abstract type AbstractDiDSolution <: AbstractCausalSolution end

# =============================================================================
# Problem Type: Classic 2×2 DiD
# =============================================================================

"""
    DiDProblem{T<:Real, P<:NamedTuple} <: AbstractDiDProblem{T,P}

Classic 2×2 Difference-in-Differences problem specification.

DiD compares changes over time between treated and control groups.
Requires panel data with pre-treatment and post-treatment periods.

# Fields
- `outcomes::Vector{T}`: Outcome variable (dependent variable)
- `treatment::Vector{Bool}`: Treatment indicator (1 if unit ever treated)
- `post::Vector{Bool}`: Post-treatment period indicator (1 if after treatment)
- `unit_id::Vector{Int}`: Unit identifiers (individuals, firms, states, etc.)
- `time::Union{Nothing, Vector{Int}}`: Time period identifiers (optional, for diagnostics)
- `parameters::P`: Analysis parameters (alpha, cluster_se, etc.)

# Classic 2×2 Setup
- Two groups: Treated vs Control (time-invariant treatment)
- Two periods: Pre vs Post
- Outcome observed for all units in both periods

# Mathematical Framework
```math
Y_{it} = α + β·Treatment_i + γ·Post_t + δ·(Treatment_i × Post_t) + ε_{it}
```

where δ is the DiD estimator (average treatment effect).

# Parallel Trends Assumption
DiD identifies causal effects if:
- Without treatment, treated and control groups would have followed parallel trends
- This is UNTESTABLE but can be assessed with:  - Pre-treatment trend tests (if ≥2 pre-periods)
  - Placebo tests
  - Event studies

# Constructor Validation
- Validates `length(outcomes) == length(treatment) == length(post) == length(unit_id)`
- Ensures treatment is time-invariant (constant within units)
- Checks for observations in all 2×2 cells (treated/control × pre/post)
- Validates `time` dimensions if provided

# Examples
```julia
using CausalEstimators

# Job training program: Treatment group receives training in 2010
# Outcome: Log earnings (observed 2008-2010)

unit_id = [1, 1, 2, 2, 3, 3, 4, 4]  # 4 units, 2 periods each
time = [2008, 2010, 2008, 2010, 2008, 2010, 2008, 2010]
treatment = [true, true, true, true, false, false, false, false]  # Units 1-2 treated
post = [false, true, false, true, false, true, false, true]  # 2010 is post
outcomes = [9.5, 10.2, 9.3, 10.1, 9.4, 9.6, 9.2, 9.5]  # Log earnings

problem = DiDProblem(
    outcomes,
    treatment,
    post,
    unit_id,
    time,
    (alpha=0.05, cluster_se=true)
)
```

# Theory
The DiD estimator identifies the Average Treatment Effect on the Treated (ATT):

```math
δ̂_{DiD} = (Ȳ_{treated,post} - Ȳ_{treated,pre}) - (Ȳ_{control,post} - Ȳ_{control,pre})
```

Under parallel trends, this recovers the causal effect of treatment.

# References
- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Bertrand, M., Duflo, E., & Mullainathan, S. (2004). "How much should we trust differences-in-differences estimates?" *Quarterly Journal of Economics*, 119(1), 249-275.
"""
struct DiDProblem{T<:Real,P<:NamedTuple} <: AbstractDiDProblem{T,P}
    outcomes::AbstractVector{T}
    treatment::AbstractVector{Bool}
    post::AbstractVector{Bool}
    unit_id::AbstractVector{Int}
    time::Union{Nothing,AbstractVector{Int}}
    parameters::P

    function DiDProblem(
        outcomes::AbstractVector{T},
        treatment::AbstractVector{Bool},
        post::AbstractVector{Bool},
        unit_id::AbstractVector{Int},
        time::Union{Nothing,AbstractVector{Int}},
        parameters::P
    ) where {T<:Real,P<:NamedTuple}
        n = length(outcomes)

        # Validate dimensions
        if length(treatment) != n
            throw(ArgumentError("treatment must have same length as outcomes (got $(length(treatment)), expected $n)"))
        end
        if length(post) != n
            throw(ArgumentError("post must have same length as outcomes (got $(length(post)), expected $n)"))
        end
        if length(unit_id) != n
            throw(ArgumentError("unit_id must have same length as outcomes (got $(length(unit_id)), expected $n)"))
        end
        if !isnothing(time) && length(time) != n
            throw(ArgumentError("time must have same length as outcomes (got $(length(time)), expected $n)"))
        end

        # Validate treatment is time-invariant (constant within units)
        for uid in unique(unit_id)
            unit_mask = unit_id .== uid
            unit_treatment = treatment[unit_mask]
            if !all(unit_treatment .== unit_treatment[1])
                throw(ArgumentError(
                    "Treatment must be time-invariant (constant within units). " *
                    "Unit $uid has varying treatment status across time. " *
                    "For time-varying treatment, use StaggeredDiDProblem."
                ))
            end
        end

        # NOTE: 2×2 cell validation moved to ClassicDiD estimator
        # EventStudy can handle edge cases (no pre-period, all treated, etc.)
        # ClassicDiD requires strict 2×2 cells and validates in solve()

        new{T,P}(outcomes, treatment, post, unit_id, time, parameters)
    end
end

# =============================================================================
# Solution Type
# =============================================================================

"""
    DiDSolution{T<:Real} <: AbstractDiDSolution

Solution from DiD estimator.

# Fields
- `estimate::T`: Point estimate of treatment effect (δ in DiD regression)
- `se::T`: Standard error (cluster-robust if cluster_se=true)
- `ci_lower::T`: Lower confidence interval bound
- `ci_upper::T`: Upper confidence interval bound
- `p_value::T`: P-value for H₀: δ = 0
- `t_stat::T`: T-statistic
- `df::Int`: Degrees of freedom (n - k for standard, n_clusters - k for cluster)
- `n_obs::Int`: Number of observations used
- `n_treated::Int`: Number of treated units
- `n_control::Int`: Number of control units
- `parallel_trends_test::Union{Nothing, NamedTuple}`: Pre-trends test results (if run)
- `retcode::Symbol`: Return code (:Success, :Warning, :Failure)

# Examples
```julia
# After solving DiD problem
solution = solve(problem, ClassicDiD())

println("Treatment effect: \$(solution.estimate) ± \$(solution.se)")
println("95% CI: [\$(solution.ci_lower), \$(solution.ci_upper)]")
println("P-value: \$(solution.p_value)")
println("N treated: \$(solution.n_treated), N control: \$(solution.n_control)")

if !isnothing(solution.parallel_trends_test)
    println("Parallel trends test p-value: \$(solution.parallel_trends_test.p_value)")
end
```

# Interpretation
- `estimate`: Average Treatment Effect on the Treated (ATT)
- `ci_lower`, `ci_upper`: Cluster-robust CI (if cluster_se=true)
- `parallel_trends_test.passes`: True if pre-trends test passes (p > α)

# References
- Bertrand, M., Duflo, E., & Mullainathan, S. (2004). "How much should we trust differences-in-differences estimates?" *Quarterly Journal of Economics*, 119(1), 249-275.
"""
struct DiDSolution{T<:Real} <: AbstractDiDSolution
    estimate::T
    se::T
    ci_lower::T
    ci_upper::T
    p_value::T
    t_stat::T
    df::Int
    n_obs::Int
    n_treated::Int
    n_control::Int
    parallel_trends_test::Union{Nothing,NamedTuple}
    retcode::Symbol

    function DiDSolution(;
        estimate::T,
        se::T,
        ci_lower::T,
        ci_upper::T,
        p_value::T,
        t_stat::T,
        df::Int,
        n_obs::Int,
        n_treated::Int,
        n_control::Int,
        parallel_trends_test::Union{Nothing,NamedTuple}=nothing,
        retcode::Symbol=:Success
    ) where {T<:Real}
        new{T}(estimate, se, ci_lower, ci_upper, p_value, t_stat, df,
               n_obs, n_treated, n_control, parallel_trends_test, retcode)
    end
end

# =============================================================================
# Estimator Types
# =============================================================================

"""
    ClassicDiD <: AbstractDiDEstimator

Classic 2×2 Difference-in-Differences estimator.

Estimates treatment effect using OLS regression with cluster-robust standard errors:

```math
Y_{it} = α + β·Treatment_i + γ·Post_t + δ·(Treatment_i × Post_t) + ε_{it}
```

where δ is the DiD estimator (ATT).

# Fields
- `cluster_se::Bool`: Use cluster-robust standard errors (default: true, clusters by unit_id)
- `test_parallel_trends::Bool`: Run pre-treatment trends test (default: false, requires ≥2 pre-periods)

# Cluster-Robust Standard Errors
Following Bertrand et al. (2004), cluster by unit_id to account for:
- Serial correlation within units
- Heteroskedasticity across units

Degrees of freedom: n_clusters - k (conservative, recommended)

# Examples
```julia
using CausalEstimators

# Default: cluster-robust SEs, no pre-trends test
solution = solve(problem, ClassicDiD())

# With pre-trends test (requires ≥2 pre-periods)
solution = solve(problem, ClassicDiD(test_parallel_trends=true))

# Standard SEs (not recommended for panel data)
solution = solve(problem, ClassicDiD(cluster_se=false))
```

# Interpretation
- `estimate`: Average effect of treatment on treated units
- Valid under parallel trends assumption (untestable without pre-periods)
- Cluster-robust SEs address serial correlation

# References
- Bertrand, M., Duflo, E., & Mullainathan, S. (2004). "How much should we trust differences-in-differences estimates?" *Quarterly Journal of Economics*, 119(1), 249-275.
"""
Base.@kwdef struct ClassicDiD <: AbstractDiDEstimator
    cluster_se::Bool = true
    test_parallel_trends::Bool = false
end

"""
    EventStudy <: AbstractDiDEstimator

Event study (dynamic DiD) estimator with leads and lags.

Estimates treatment effects over time using Two-Way Fixed Effects (TWFE):

```math
Y_{it} = α_i + λ_t + Σ_{k≠-1} β_k·1{t - T_i = k} + ε_{it}
```

where:
- α_i = unit fixed effects
- λ_t = time fixed effects
- k = event time (relative to treatment)
- k = -1 is omitted (normalization)

# Fields
- `n_leads::Int`: Number of pre-treatment periods to include (default: auto-detect)
- `n_lags::Int`: Number of post-treatment periods to include (default: auto-detect)
- `omit_period::Int`: Period to omit for normalization (default: -1)
- `cluster_se::Bool`: Use cluster-robust SEs (default: true)

# Pre-Trends Testing
Event study coefficients β_k for k < 0 test parallel trends:
- H₀: β_{-2} = β_{-3} = ... = 0 (pre-trends)
- Reject if joint F-test p < α

# Examples
```julia
using CausalEstimators

# Auto-detect leads/lags
solution = solve(problem, EventStudy())

# Manual specification
solution = solve(problem, EventStudy(n_leads=3, n_lags=5))

# Access results
println("Pre-treatment coefficients: \$(solution.coefficients_pre)")
println("Post-treatment coefficients: \$(solution.coefficients_post)")
println("Joint F-test p-value: \$(solution.pre_trends_test.p_value)")
```

# References
- Freyaldenhoven, S., Hansen, C., & Shapiro, J. M. (2019). "Pre-event trends in the panel event-study design." *American Economic Review*, 109(9), 3307-3338.
"""
Base.@kwdef struct EventStudy <: AbstractDiDEstimator
    n_leads::Union{Int,Nothing} = nothing
    n_lags::Union{Int,Nothing} = nothing
    omit_period::Int = -1
    cluster_se::Bool = true
end
