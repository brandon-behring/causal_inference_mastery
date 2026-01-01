#=
Type definitions for Dynamic Double Machine Learning.

Defines DynamicDMLResult and TimeSeriesPanelData for dynamic treatment
effect estimation following Lewis & Syrgkanis (2021).

Reference:
Lewis, G., & Syrgkanis, V. (2021). Double/Debiased Machine Learning for
Dynamic Treatment Effects via g-Estimation. arXiv:2002.07285.
=#

module DynamicTypes

using Statistics
using Printf

export DynamicDMLResult, TimeSeriesPanelData
export result_summary, dml_is_significant, get_lagged_data

"""
    DynamicDMLResult

Result from Dynamic DML estimation with treatment effects at each lag.

# Fields
- `theta::Vector{Float64}`: Treatment effects at each lag h = 0, 1, ..., max_lag
- `theta_se::Vector{Float64}`: HAC-robust standard errors
- `ci_lower::Vector{Float64}`: Lower confidence bounds
- `ci_upper::Vector{Float64}`: Upper confidence bounds
- `cumulative_effect::Float64`: Discounted sum of effects
- `cumulative_effect_se::Float64`: SE of cumulative effect
- `influence_function::Matrix{Float64}`: Shape (n_obs, max_lag + 1)
- `nuisance_r2::Dict{Symbol,Vector{Float64}}`: R² for nuisance models
- `method::String`: Estimation method identifier
- `max_lag::Int`: Maximum treatment lag
- `n_folds::Int`: Number of cross-fitting folds
- `hac_bandwidth::Int`: HAC bandwidth used
- `hac_kernel::String`: HAC kernel ("bartlett" or "qs")
- `n_obs::Int`: Number of observations
- `alpha::Float64`: Significance level
- `discount_factor::Float64`: Discount factor for cumulative effect
"""
struct DynamicDMLResult
    theta::Vector{Float64}
    theta_se::Vector{Float64}
    ci_lower::Vector{Float64}
    ci_upper::Vector{Float64}
    cumulative_effect::Float64
    cumulative_effect_se::Float64
    influence_function::Matrix{Float64}
    nuisance_r2::Dict{Symbol,Vector{Float64}}
    method::String
    max_lag::Int
    n_folds::Int
    hac_bandwidth::Int
    hac_kernel::String
    n_obs::Int
    alpha::Float64
    discount_factor::Float64
end

"""
    result_summary(result::DynamicDMLResult) -> String

Return formatted summary of dynamic treatment effects.
"""
function result_summary(result::DynamicDMLResult)
    lines = [
        "Dynamic DML Results (n=$(result.n_obs))",
        "=" ^ 50,
        "Method: $(result.method)",
        "Max lag: $(result.max_lag), Folds: $(result.n_folds)",
        "HAC: $(result.hac_kernel) (bandwidth=$(result.hac_bandwidth))",
        "",
        "Treatment Effects by Lag:",
        "-" ^ 50,
        @sprintf("%-6s %-12s %-12s %-20s", "Lag", "Effect", "SE", "95% CI"),
        "-" ^ 50,
    ]

    for h in 0:result.max_lag
        effect = result.theta[h + 1]
        se = result.theta_se[h + 1]
        ci_lo = result.ci_lower[h + 1]
        ci_hi = result.ci_upper[h + 1]
        push!(lines, @sprintf("%-6d %-12.4f %-12.4f [%.4f, %.4f]", h, effect, se, ci_lo, ci_hi))
    end

    push!(lines, "-" ^ 50)
    push!(lines, @sprintf("Cumulative effect: %.4f (SE: %.4f)", result.cumulative_effect, result.cumulative_effect_se))
    push!(lines, "Discount factor: $(result.discount_factor)")

    return join(lines, "\n")
end

# Import Printf for formatting
using Printf

"""
    dml_is_significant(result::DynamicDMLResult, lag::Int; alpha=nothing) -> Bool

Check if effect at given lag is statistically significant.
"""
function dml_is_significant(result::DynamicDMLResult, lag::Int; alpha::Union{Float64,Nothing}=nothing)
    α = isnothing(alpha) ? result.alpha : alpha
    ci_lo = result.ci_lower[lag + 1]
    ci_hi = result.ci_upper[lag + 1]
    return ci_lo > 0 || ci_hi < 0
end


"""
    TimeSeriesPanelData

Panel or time series data structure for dynamic treatment effects.

# Fields
- `outcomes::Vector{Float64}`: Outcome variable Y
- `treatments::Matrix{Float64}`: Treatment(s), shape (n_obs, n_treatments)
- `states::Matrix{Float64}`: Covariates X, shape (n_obs, n_covariates)
- `unit_id::Union{Vector{Int},Nothing}`: Unit IDs for panel data
- `time_id::Vector{Int}`: Time period identifiers
- `data_type::Symbol`: :single_series or :panel
- `n_units::Int`: Number of units
- `n_periods::Int`: Number of time periods
- `n_treatments::Int`: Number of treatments
- `n_covariates::Int`: Number of covariates
"""
struct TimeSeriesPanelData
    outcomes::Vector{Float64}
    treatments::Matrix{Float64}
    states::Matrix{Float64}
    unit_id::Union{Vector{Int},Nothing}
    time_id::Vector{Int}
    data_type::Symbol
    n_units::Int
    n_periods::Int
    n_treatments::Int
    n_covariates::Int
end

"""
    TimeSeriesPanelData(outcomes, treatments, states; unit_id=nothing, time_id=nothing)

Construct TimeSeriesPanelData from arrays.
"""
function TimeSeriesPanelData(
    outcomes::AbstractVector{<:Real},
    treatments::AbstractVecOrMat{<:Real},
    states::AbstractMatrix{<:Real};
    unit_id::Union{AbstractVector{<:Integer},Nothing}=nothing,
    time_id::Union{AbstractVector{<:Integer},Nothing}=nothing
)
    n_obs = length(outcomes)

    # Validate lengths
    if size(treatments, 1) != n_obs
        error("CRITICAL ERROR: Length mismatch. outcomes has $n_obs observations, treatments has $(size(treatments, 1)).")
    end
    if size(states, 1) != n_obs
        error("CRITICAL ERROR: Length mismatch. outcomes has $n_obs observations, states has $(size(states, 1)).")
    end

    # Handle 1D treatments
    T_mat = treatments isa AbstractVector ? reshape(Float64.(treatments), :, 1) : Float64.(treatments)
    n_treatments = size(T_mat, 2)
    n_covariates = size(states, 2)

    # Determine data type
    if isnothing(unit_id)
        data_type = :single_series
        n_units = 1
        n_periods = n_obs
        time_id_out = isnothing(time_id) ? collect(0:n_obs-1) : Int.(time_id)
        unit_id_out = nothing
    else
        data_type = :panel
        unit_id_int = Int.(unit_id)
        unique_units = unique(unit_id_int)
        n_units = length(unique_units)
        n_periods = n_obs ÷ n_units

        if isnothing(time_id)
            time_id_out = zeros(Int, n_obs)
            for u in unique_units
                mask = unit_id_int .== u
                time_id_out[mask] .= collect(0:sum(mask)-1)
            end
        else
            time_id_out = Int.(time_id)
        end
        unit_id_out = unit_id_int
    end

    return TimeSeriesPanelData(
        Float64.(outcomes),
        T_mat,
        Float64.(states),
        unit_id_out,
        time_id_out,
        data_type,
        n_units,
        n_periods,
        n_treatments,
        n_covariates
    )
end

"""
    get_lagged_data(data::TimeSeriesPanelData, max_lag::Int)

Create lagged treatment matrix for dynamic estimation.

# Returns
- `Y::Vector{Float64}`: Outcomes trimmed to valid observations
- `T_lagged::Array{Float64,3}`: Shape (n_valid, max_lag + 1, n_treatments)
- `X::Matrix{Float64}`: States trimmed to valid observations
- `valid_mask::BitVector`: Boolean mask for valid observations
"""
function get_lagged_data(data::TimeSeriesPanelData, max_lag::Int)
    if data.data_type == :single_series
        n_valid = data.n_periods - max_lag
        valid_mask = falses(data.n_periods)
        valid_mask[(max_lag + 1):end] .= true

        Y = data.outcomes[valid_mask]
        X = data.states[valid_mask, :]

        # Build lagged treatment matrix
        T_lagged = zeros(n_valid, max_lag + 1, data.n_treatments)
        for h in 0:max_lag
            # Lag h: treatment at t - h
            T_lagged[:, h + 1, :] = data.treatments[(max_lag - h + 1):(data.n_periods - h), :]
        end

        return Y, T_lagged, X, valid_mask
    else
        # Panel data
        Y_list = Vector{Float64}[]
        T_list = Array{Float64,3}[]
        X_list = Matrix{Float64}[]
        valid_indices = Int[]

        unique_units = unique(data.unit_id)
        for u in unique_units
            mask = data.unit_id .== u
            unit_outcomes = data.outcomes[mask]
            unit_treatments = data.treatments[mask, :]
            unit_states = data.states[mask, :]
            n_periods_u = length(unit_outcomes)

            if n_periods_u <= max_lag
                continue  # Skip units with insufficient periods
            end

            n_valid_u = n_periods_u - max_lag
            push!(Y_list, unit_outcomes[(max_lag + 1):end])
            push!(X_list, unit_states[(max_lag + 1):end, :])

            # Build lagged treatments for this unit
            T_lagged_u = zeros(n_valid_u, max_lag + 1, data.n_treatments)
            for h in 0:max_lag
                T_lagged_u[:, h + 1, :] = unit_treatments[(max_lag - h + 1):(n_periods_u - h), :]
            end
            push!(T_list, T_lagged_u)

            # Track valid indices
            unit_indices = findall(mask)
            append!(valid_indices, unit_indices[(max_lag + 1):end])
        end

        Y = vcat(Y_list...)
        T_lagged = cat(T_list...; dims=1)
        X = vcat(X_list...)
        valid_mask = falses(length(data.outcomes))
        valid_mask[valid_indices] .= true

        return Y, T_lagged, X, valid_mask
    end
end

end  # module DynamicTypes
