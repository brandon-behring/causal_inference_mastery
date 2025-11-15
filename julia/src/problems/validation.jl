"""
Input validation utilities for RCT problems.

Following Brandon's principles:
- NEVER FAIL SILENTLY
- Fail Fast (validate in constructor, not solve())
- Explicit error messages with diagnostic info
"""

"""
    validate_rct_inputs(outcomes, treatment, covariates, strata)

Validate inputs for RCTProblem construction.

Throws ArgumentError with diagnostic info if validation fails.

# Checks Performed
1. Non-empty arrays
2. Matching lengths
3. No NaN/Inf values
4. Treatment is binary
5. Treatment has variation (both groups present)
6. Covariate matrix dimensions match
7. Strata values are valid

# Throws
- `ArgumentError`: If any validation check fails (with diagnostic message)
"""
function validate_rct_inputs(
    outcomes::Vector{T},
    treatment::Vector{Bool},
    covariates::Union{Nothing,Matrix{T}},
    strata::Union{Nothing,Vector{Int}},
) where {T<:Real}
    n = length(outcomes)

    # 1. Check non-empty
    if n == 0
        throw(
            ArgumentError(
                "CRITICAL ERROR: Empty input arrays.\n" *
                "Function: RCTProblem constructor\n" *
                "Expected: Non-empty arrays\n" *
                "Got: length(outcomes) = 0",
            ),
        )
    end

    # 2. Check lengths match
    if length(treatment) != n
        throw(
            ArgumentError(
                "CRITICAL ERROR: Mismatched array lengths.\n" *
                "Function: RCTProblem constructor\n" *
                "Expected: Same length arrays\n" *
                "Got: length(outcomes)=$(n), length(treatment)=$(length(treatment))",
            ),
        )
    end

    # 3. Check for NaN in outcomes
    if any(isnan, outcomes)
        nan_indices = findall(isnan, outcomes)
        throw(
            ArgumentError(
                "CRITICAL ERROR: NaN values detected in outcomes.\n" *
                "Function: RCTProblem constructor\n" *
                "NaN indicates data quality issues that must be addressed.\n" *
                "Indices with NaN: $(nan_indices)\n" *
                "Count: $(length(nan_indices))",
            ),
        )
    end

    # 4. Check for Inf in outcomes
    if any(isinf, outcomes)
        inf_indices = findall(isinf, outcomes)
        throw(
            ArgumentError(
                "CRITICAL ERROR: Infinite values detected in outcomes.\n" *
                "Function: RCTProblem constructor\n" *
                "Indices with Inf: $(inf_indices)\n" *
                "Count: $(length(inf_indices))",
            ),
        )
    end

    # 5. Check treatment variation
    n_treated = sum(treatment)
    n_control = n - n_treated

    if n_treated == 0
        throw(
            ArgumentError(
                "CRITICAL ERROR: No treated units in data.\n" *
                "Function: RCTProblem constructor\n" *
                "Cannot estimate treatment effect without treated group.\n" *
                "Got: All units have treatment=false",
            ),
        )
    end

    if n_control == 0
        throw(
            ArgumentError(
                "CRITICAL ERROR: No control units in data.\n" *
                "Function: RCTProblem constructor\n" *
                "Cannot estimate treatment effect without control group.\n" *
                "Got: All units have treatment=true",
            ),
        )
    end

    # 6. Validate covariates if provided
    if !isnothing(covariates)
        n_rows, n_cols = size(covariates)

        if n_rows != n
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Covariate matrix has wrong number of rows.\n" *
                    "Function: RCTProblem constructor\n" *
                    "Expected: $(n) rows (same as outcomes)\n" *
                    "Got: $(n_rows) rows",
                ),
            )
        end

        if n_cols == 0
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Covariate matrix has zero columns.\n" *
                    "Function: RCTProblem constructor\n" *
                    "If no covariates, pass `nothing` instead of empty matrix.",
                ),
            )
        end

        # Check for NaN/Inf in covariates
        if any(isnan, covariates)
            throw(
                ArgumentError(
                    "CRITICAL ERROR: NaN values detected in covariate matrix.\n" *
                    "Function: RCTProblem constructor\n" *
                    "Count: $(sum(isnan, covariates))",
                ),
            )
        end

        if any(isinf, covariates)
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Infinite values detected in covariate matrix.\n" *
                    "Function: RCTProblem constructor\n" *
                    "Count: $(sum(isinf, covariates))",
                ),
            )
        end
    end

    # 7. Validate strata if provided
    if !isnothing(strata)
        if length(strata) != n
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Strata vector has wrong length.\n" *
                    "Function: RCTProblem constructor\n" *
                    "Expected: $(n) (same as outcomes)\n" *
                    "Got: $(length(strata))",
                ),
            )
        end

        # Check for negative or zero strata
        if any(<=(0), strata)
            invalid_strata = unique(strata[strata .<=0])
            throw(
                ArgumentError(
                    "CRITICAL ERROR: Invalid strata values.\n" *
                    "Function: RCTProblem constructor\n" *
                    "Strata must be positive integers (1, 2, 3, ...).\n" *
                    "Got invalid values: $(invalid_strata)",
                ),
            )
        end

        # Check that each stratum has both treatment and control
        # (This is checked more thoroughly in StratifiedATE solve(), but quick check here)
        unique_strata = unique(strata)
        for s in unique_strata
            in_stratum = strata .== s
            has_treated = any(treatment[in_stratum])
            has_control = any(.!treatment[in_stratum])

            if !has_treated || !has_control
                throw(
                    ArgumentError(
                        "CRITICAL ERROR: Stratum $(s) lacks treatment variation.\n" *
                        "Function: RCTProblem constructor\n" *
                        "Each stratum must have both treated and control units.\n" *
                        "Stratum $(s): treated=$(has_treated), control=$(has_control)",
                    ),
                )
            end
        end
    end

    return nothing
end
