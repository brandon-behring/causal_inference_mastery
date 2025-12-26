"""
    Panel Data Types for DML-CRE

Type definitions for panel data and DML-CRE results.
"""

"""
    PanelData{T<:Real}

Panel data structure for DML-CRE estimation.

Long-format panel data with unit and time identifiers.

# Fields
- `outcomes::Vector{T}`: Outcome variable Y of length n_obs.
- `treatment::Vector{T}`: Treatment variable D of length n_obs. Can be binary or continuous.
- `covariates::Matrix{T}`: Time-varying covariates X of size (n_obs, p).
- `unit_id::Vector{Int}`: Unit identifiers of length n_obs.
- `time::Vector{Int}`: Time period identifiers of length n_obs.

# Properties (computed)
- `n_obs`: Total number of observations (unit-time pairs).
- `n_units`: Number of unique units.
- `n_periods`: Number of unique time periods.
- `n_covariates`: Number of covariates p.
- `is_balanced`: True if all units have same number of observations.

# Examples
```julia
using Random
Random.seed!(42)

# Create balanced panel: 10 units, 5 periods each
n_units, n_periods = 10, 5
n_obs = n_units * n_periods
unit_id = repeat(1:n_units, inner=n_periods)
time = repeat(1:n_periods, outer=n_units)
Y = randn(n_obs)
D = rand(n_obs) .< 0.5
X = randn(n_obs, 3)

panel = PanelData(Y, Float64.(D), X, unit_id, time)
println("Units: \$(panel.n_units), Periods: \$(panel.n_periods)")
```
"""
struct PanelData{T<:Real}
    outcomes::Vector{T}
    treatment::Vector{T}
    covariates::Matrix{T}
    unit_id::Vector{Int}
    time::Vector{Int}

    function PanelData(
        outcomes::AbstractVector{<:Real},
        treatment::AbstractVector{<:Real},
        covariates::AbstractMatrix{<:Real},
        unit_id::AbstractVector{<:Integer},
        time::AbstractVector{<:Integer},
    )
        T = promote_type(eltype(outcomes), eltype(treatment), eltype(covariates))

        # Convert to common types
        y = convert(Vector{T}, outcomes)
        d = convert(Vector{T}, treatment)
        X = convert(Matrix{T}, covariates)
        uid = convert(Vector{Int}, unit_id)
        t = convert(Vector{Int}, time)

        n = length(y)

        # Validation
        length(d) == n || error(
            "CRITICAL ERROR: Length mismatch.\n" *
            "Function: PanelData\n" *
            "outcomes: $n, treatment: $(length(d))"
        )
        length(uid) == n || error(
            "CRITICAL ERROR: Length mismatch.\n" *
            "Function: PanelData\n" *
            "outcomes: $n, unit_id: $(length(uid))"
        )
        length(t) == n || error(
            "CRITICAL ERROR: Length mismatch.\n" *
            "Function: PanelData\n" *
            "outcomes: $n, time: $(length(t))"
        )
        size(X, 1) == n || error(
            "CRITICAL ERROR: Covariate rows mismatch.\n" *
            "Function: PanelData\n" *
            "outcomes: $n, covariate rows: $(size(X, 1))"
        )

        # Check for NaN/Inf
        any(isnan, y) || any(isinf, y) && error(
            "CRITICAL ERROR: NaN or Inf in outcomes.\n" *
            "Function: PanelData"
        )
        any(isnan, d) || any(isinf, d) && error(
            "CRITICAL ERROR: NaN or Inf in treatment.\n" *
            "Function: PanelData"
        )
        any(isnan, X) || any(isinf, X) && error(
            "CRITICAL ERROR: NaN or Inf in covariates.\n" *
            "Function: PanelData"
        )

        # Check minimum units
        unique_units = unique(uid)
        n_units = length(unique_units)
        n_units >= 2 || error(
            "CRITICAL ERROR: Need at least 2 units.\n" *
            "Function: PanelData\n" *
            "n_units: $n_units"
        )

        # Check minimum observations per unit
        for unit in unique_units
            unit_obs = count(==(unit), uid)
            unit_obs >= 2 || error(
                "CRITICAL ERROR: Each unit needs at least 2 observations.\n" *
                "Function: PanelData\n" *
                "Unit $unit has only $unit_obs observation(s)"
            )
        end

        new{T}(y, d, X, uid, t)
    end
end

# Property accessors
n_obs(panel::PanelData) = length(panel.outcomes)
n_units(panel::PanelData) = length(unique(panel.unit_id))
n_periods(panel::PanelData) = length(unique(panel.time))
n_covariates(panel::PanelData) = size(panel.covariates, 2)

function is_balanced(panel::PanelData)
    unique_units = unique(panel.unit_id)
    obs_per_unit = [count(==(u), panel.unit_id) for u in unique_units]
    return length(unique(obs_per_unit)) == 1
end

"""
    get_unique_units(panel::PanelData)

Return sorted vector of unique unit identifiers.
"""
get_unique_units(panel::PanelData) = sort(unique(panel.unit_id))

"""
    get_unit_indices(panel::PanelData, unit::Int)

Return indices for a specific unit.
"""
get_unit_indices(panel::PanelData, unit::Int) = findall(==(unit), panel.unit_id)

"""
    compute_unit_means(panel::PanelData)

Compute time-averaged covariates X̄ᵢ for each observation.

For each observation (i, t), returns the unit-level mean
X̄ᵢ = (1/Tᵢ) Σₜ Xᵢₜ

# Returns
- `Matrix{Float64}`: Unit means of shape (n_obs, p), aligned with observations.

# Notes
This is the Mundlak (1978) projection: by including X̄ᵢ alongside
Xᵢₜ, we can model the correlation between covariates and
unobserved unit effects.
"""
function compute_unit_means(panel::PanelData{T}) where T
    n = n_obs(panel)
    p = n_covariates(panel)
    unit_means = zeros(T, n, p)

    for unit in get_unique_units(panel)
        unit_idx = get_unit_indices(panel, unit)
        unit_mean = vec(mean(panel.covariates[unit_idx, :], dims=1))
        for i in unit_idx
            unit_means[i, :] = unit_mean
        end
    end

    return unit_means
end

"""
    compute_treatment_mean(panel::PanelData)

Compute time-averaged treatment D̄ᵢ for each observation.

# Returns
- `Vector{Float64}`: Unit treatment means of length n_obs.
"""
function compute_treatment_mean(panel::PanelData{T}) where T
    n = n_obs(panel)
    treatment_means = zeros(T, n)

    for unit in get_unique_units(panel)
        unit_idx = get_unit_indices(panel, unit)
        treatment_means[unit_idx] .= mean(panel.treatment[unit_idx])
    end

    return treatment_means
end


"""
    DMLCREResult

Result from Panel DML-CRE estimation.

# Fields
- `ate::Float64`: Average treatment effect estimate.
- `ate_se::Float64`: Standard error of ATE (clustered by unit).
- `ci_lower::Float64`: Lower bound of confidence interval.
- `ci_upper::Float64`: Upper bound of confidence interval.
- `cate::Vector{Float64}`: Conditional average treatment effects τ(Xᵢₜ, X̄ᵢ).
- `method::Symbol`: `:dml_cre` or `:dml_cre_continuous`.
- `n_units::Int`: Number of units.
- `n_obs::Int`: Number of observations.
- `n_folds::Int`: Number of cross-fitting folds.
- `outcome_r2::Float64`: R-squared of outcome model.
- `treatment_r2::Float64`: R-squared of treatment model.
- `unit_effects::Vector{Float64}`: Estimated unit effects α̂ᵢ.
- `fold_estimates::Vector{Float64}`: Per-fold ATE estimates.
- `fold_ses::Vector{Float64}`: Per-fold standard errors.
"""
struct DMLCREResult
    ate::Float64
    ate_se::Float64
    ci_lower::Float64
    ci_upper::Float64
    cate::Vector{Float64}
    method::Symbol
    n_units::Int
    n_obs::Int
    n_folds::Int
    outcome_r2::Float64
    treatment_r2::Float64
    unit_effects::Vector{Float64}
    fold_estimates::Vector{Float64}
    fold_ses::Vector{Float64}
end
