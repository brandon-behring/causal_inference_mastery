"""
Propensity score matching algorithms.
"""

"""
    nearest_neighbor_match(propensity_treated, propensity_control, indices_treated, indices_control;
                          M=1, with_replacement=false, caliper=Inf)

Find nearest neighbor matches for treated units based on propensity score distance.

# Arguments
- `propensity_treated::Vector{Float64}`: Propensity scores for treated units
- `propensity_control::Vector{Float64}`: Propensity scores for control units
- `indices_treated::Vector{Int}`: Original indices of treated units
- `indices_control::Vector{Int}`: Original indices of control units
- `M::Int`: Number of matches per treated unit (default: 1)
- `with_replacement::Bool`: Allow reusing control units (default: false)
- `caliper::Float64`: Maximum propensity score distance (default: Inf, no restriction)

# Returns
- `matches::Vector{Vector{Int}}`: For each treated unit, vector of matched control indices
- `distances::Vector{Vector{Float64}}`: Propensity score distances for each match
- `n_matched::Int`: Number of successfully matched treated units

# Method
For each treated unit i:
1. Compute propensity score distance to all available control units: |e_i - e_j|
2. Find M nearest controls within caliper distance
3. If with_replacement=false, remove matched controls from pool

# Notes
- Greedy algorithm: matches in order of treated indices (consider randomizing order)
- Units outside caliper are dropped (may reduce sample size)
- With replacement: each control can match multiple treated units
- Without replacement: controls matched once, then removed from pool

# Example
```julia
# Match each treated unit to 1 nearest control
matches, distances, n_matched = nearest_neighbor_match(
    propensity[treatment],
    propensity[.!treatment],
    findall(treatment),
    findall(.!treatment),
    M=1,
    with_replacement=false,
    caliper=0.1
)
```

# References
- Abadie, A., & Imbens, G. W. (2006). Large sample properties of matching estimators for
  average treatment effects. *Econometrica*, 74(1), 235-267.
- Rubin, D. B. (1973). Matching to remove bias in observational studies. *Biometrics*, 159-183.
"""
function nearest_neighbor_match(
    propensity_treated::Vector{Float64},
    propensity_control::Vector{Float64},
    indices_treated::Vector{Int},
    indices_control::Vector{Int};
    M::Int = 1,
    with_replacement::Bool = false,
    caliper::Float64 = Inf,
)
    n_treated = length(propensity_treated)
    n_control = length(propensity_control)

    # ========================================================================
    # Validate Inputs
    # ========================================================================

    if n_treated == 0
        throw(
            ArgumentError(
                "CRITICAL ERROR: No treated units.\\n" *
                "Function: nearest_neighbor_match\\n" *
                "Cannot perform matching with zero treated units.\\n" *
                "Ensure treatment group is non-empty.",
            ),
        )
    end

    if n_control == 0
        throw(
            ArgumentError(
                "CRITICAL ERROR: No control units.\\n" *
                "Function: nearest_neighbor_match\\n" *
                "Cannot perform matching with zero control units.\\n" *
                "Ensure control group is non-empty.",
            ),
        )
    end

    if M < 1
        throw(
            ArgumentError(
                "CRITICAL ERROR: Invalid M value.\\n" *
                "Function: nearest_neighbor_match\\n" *
                "M must be ≥ 1, got M = $M\\n" *
                "M is the number of matches per treated unit.",
            ),
        )
    end

    if M > n_control && !with_replacement
        throw(
            ArgumentError(
                "CRITICAL ERROR: Insufficient controls for M:1 matching.\\n" *
                "Function: nearest_neighbor_match\\n" *
                "M = $M but only $n_control control units available.\\n" *
                "Either reduce M or use with_replacement=true.",
            ),
        )
    end

    if caliper <= 0 && caliper != Inf
        throw(
            ArgumentError(
                "CRITICAL ERROR: Invalid caliper.\\n" *
                "Function: nearest_neighbor_match\\n" *
                "caliper must be > 0 or Inf, got caliper = $caliper\\n" *
                "Typical values: 0.1 (strict), 0.25 (moderate), Inf (no restriction).",
            ),
        )
    end

    if length(indices_treated) != n_treated
        throw(
            ArgumentError(
                "CRITICAL ERROR: Mismatched indices.\\n" *
                "Function: nearest_neighbor_match\\n" *
                "propensity_treated has $n_treated elements, indices_treated has $(length(indices_treated))\\n" *
                "Must have same length.",
            ),
        )
    end

    if length(indices_control) != n_control
        throw(
            ArgumentError(
                "CRITICAL ERROR: Mismatched indices.\\n" *
                "Function: nearest_neighbor_match\\n" *
                "propensity_control has $n_control elements, indices_control has $(length(indices_control))\\n" *
                "Must have same length.",
            ),
        )
    end

    # ========================================================================
    # Perform Matching
    # ========================================================================

    matches = Vector{Vector{Int}}(undef, n_treated)
    distances = Vector{Vector{Float64}}(undef, n_treated)

    # Track available controls (for matching without replacement)
    available_controls = if with_replacement
        # With replacement: all controls always available
        collect(1:n_control)
    else
        # Without replacement: remove controls as they're matched
        collect(1:n_control)
    end

    n_matched = 0

    for i in 1:n_treated
        e_i = propensity_treated[i]

        # Compute distances to all available controls
        if isempty(available_controls)
            # No controls left (only possible without replacement)
            matches[i] = Int[]
            distances[i] = Float64[]
            continue
        end

        # Distance to all available controls
        dists = abs.(propensity_control[available_controls] .- e_i)

        # Apply caliper: only consider controls within distance
        within_caliper = dists .<= caliper
        valid_controls = available_controls[within_caliper]
        valid_dists = dists[within_caliper]

        if isempty(valid_controls)
            # No controls within caliper
            matches[i] = Int[]
            distances[i] = Float64[]
            continue
        end

        # Find M nearest controls
        n_available = length(valid_controls)
        n_matches = min(M, n_available)

        # Sort by distance and take M nearest
        sorted_idx = sortperm(valid_dists)
        matched_idx = sorted_idx[1:n_matches]

        # Store matched control indices (original indices, not available_controls indices)
        matches[i] = indices_control[valid_controls[matched_idx]]
        distances[i] = valid_dists[matched_idx]

        # If without replacement, remove matched controls
        if !with_replacement
            # Remove matched controls from available pool
            # Need to map matched_idx back to positions in available_controls
            matched_controls = valid_controls[matched_idx]
            available_controls = setdiff(available_controls, matched_controls)
        end

        n_matched += 1
    end

    return matches, distances, n_matched
end


"""
    compute_ate_from_matches(outcomes, treatment, matches)

Compute average treatment effect (ATE) from matched pairs.

# Arguments
- `outcomes::Vector{Float64}`: Observed outcomes for all units
- `treatment::Vector{Bool}`: Treatment indicator
- `matches::Vector{Vector{Int}}`: Matched control indices for each treated unit

# Returns
- `ate::Float64`: Average treatment effect estimate
- `treated_outcomes::Vector{Float64}`: Outcomes for matched treated units
- `control_outcomes::Vector{Vector{Float64}}`: Matched control outcomes for each treated

# Method
For M:1 matching (M controls per treated):
```
ATE = (1/N_matched) ∑ᵢ (Yᵢ - (1/M) ∑ⱼ Yⱼ(ᵢ))
```

where:
- N_matched = number of treated units with at least one match
- Yᵢ = outcome for treated unit i
- Yⱼ(ᵢ) = outcome for j-th matched control for unit i
- M = number of matches (may vary by unit if some dropped)

# Notes
- Only includes treated units with at least one match
- Control outcomes averaged if M > 1
- Imputation estimator (simple difference-in-means on matched sample)

# Example
```julia
ate, y_treated, y_control = compute_ate_from_matches(outcomes, treatment, matches)
```
"""
function compute_ate_from_matches(
    outcomes::Vector{Float64},
    treatment::Vector{Bool},
    matches::Vector{Vector{Int}},
)
    n_treated = sum(treatment)
    indices_treated = findall(treatment)

    # Validate inputs
    if length(matches) != n_treated
        throw(
            ArgumentError(
                "CRITICAL ERROR: Mismatched lengths.\\n" *
                "Function: compute_ate_from_matches\\n" *
                "matches has $(length(matches)) elements, but $n_treated treated units.\\n" *
                "Must have one match vector per treated unit.",
            ),
        )
    end

    # Collect treated and matched control outcomes
    treated_outcomes = Float64[]
    control_outcomes = Vector{Float64}[]

    for i in 1:n_treated
        matched_controls = matches[i]

        # Skip if no matches
        if isempty(matched_controls)
            continue
        end

        # Treated outcome
        y_i = outcomes[indices_treated[i]]
        push!(treated_outcomes, y_i)

        # Matched control outcomes
        y_controls = outcomes[matched_controls]
        push!(control_outcomes, y_controls)
    end

    # Compute ATE
    n_matched = length(treated_outcomes)

    if n_matched == 0
        throw(
            ArgumentError(
                "CRITICAL ERROR: No matched pairs.\\n" *
                "Function: compute_ate_from_matches\\n" *
                "All treated units failed to match.\\n" *
                "Check caliper settings or propensity score overlap.",
            ),
        )
    end

    # ATE = E[Y_i - mean(Y_j(i))]
    ate = sum(treated_outcomes[i] - mean(control_outcomes[i]) for i in 1:n_matched) /
          n_matched

    return ate, treated_outcomes, control_outcomes
end
