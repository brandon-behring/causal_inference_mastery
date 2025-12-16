#=
Outcome Regression Models for Doubly Robust Estimation

Fits separate outcome models for treated (T=1) and control (T=0) units
to estimate E[Y|T, X] needed for AIPW estimation.

Functions:
- fit_outcome_models: Fit μ₀(X) and μ₁(X) using linear regression
- compute_r2: Compute coefficient of determination

References:
- Bang & Robins (2005). Doubly robust estimation.
- Kennedy (2016). Semiparametric theory and empirical processes.
=#

"""
    fit_outcome_models(outcomes, treatment, covariates)

Fit separate outcome models for treated and control units.

Fits two linear regression models:
- μ₁(X) = E[Y|T=1, X] using treated units
- μ₀(X) = E[Y|T=0, X] using control units

Then predicts on ALL covariates (not just own treatment group).

# Arguments
- `outcomes::Vector{T}`: Observed outcomes Y
- `treatment::Vector{Bool}`: Treatment indicators T ∈ {0,1}
- `covariates::Matrix{T}`: Covariate matrix X (n × p)

# Returns
NamedTuple with:
- `mu0_predictions::Vector{T}`: E[Y|T=0, X] for all X
- `mu1_predictions::Vector{T}`: E[Y|T=1, X] for all X
- `mu0_r2::T`: R² for control model (in-sample)
- `mu1_r2::T`: R² for treated model (in-sample)
- `mu0_coefficients::Vector{T}`: Control model coefficients (intercept + slopes)
- `mu1_coefficients::Vector{T}`: Treated model coefficients (intercept + slopes)

# Mathematical Model

Linear regression: Y = β₀ + β'X + ε

For DR estimation, we fit separate models allowing different β by treatment group.

# Example
```julia
X = randn(100, 2)
T = rand(100) .< 0.5
Y = 2.0 .* T .+ 0.5 .* X[:, 1] .+ randn(100)

result = fit_outcome_models(Y, T, X)
# result.mu1_predictions - result.mu0_predictions should be near 2.0
```

# Notes
- Uses GLM.jl's lm() for linear regression
- Predicts on all covariates (needed for DR estimation)
- Returns in-sample R² for model quality assessment
"""
function fit_outcome_models(
    outcomes::AbstractVector{T},
    treatment::AbstractVector{Bool},
    covariates::AbstractMatrix{T}
) where {T<:Real}
    n, p = size(covariates)

    # Separate data by treatment
    control_mask = .!treatment
    treated_mask = treatment

    n_control = sum(control_mask)
    n_treated = sum(treated_mask)

    if n_control < p + 1
        throw(ArgumentError(
            "Insufficient control units ($n_control) for $p covariates. " *
            "Need at least $(p + 1) control units."
        ))
    end

    if n_treated < p + 1
        throw(ArgumentError(
            "Insufficient treated units ($n_treated) for $p covariates. " *
            "Need at least $(p + 1) treated units."
        ))
    end

    # Extract data by treatment group
    Y_control = outcomes[control_mask]
    X_control = covariates[control_mask, :]

    Y_treated = outcomes[treated_mask]
    X_treated = covariates[treated_mask, :]

    # ==========================================================================
    # Fit Control Model: μ₀(X) = E[Y|T=0, X]
    # ==========================================================================

    # Build DataFrame for GLM
    df_control = DataFrame(X_control, :auto)
    df_control.Y = Y_control

    # Create formula: Y ~ X1 + X2 + ... + Xp
    feature_names = names(df_control)[1:end-1]
    formula_str = "Y ~ " * join(feature_names, " + ")
    formula = eval(Meta.parse("@formula($formula_str)"))

    # Fit linear regression
    mu0_model = try
        lm(formula, df_control)
    catch e
        msg = hasfield(typeof(e), :msg) ? e.msg : string(e)
        throw(ArgumentError(
            "Control outcome model failed to fit: $msg. " *
            "Check for collinearity or insufficient variation."
        ))
    end

    mu0_coefficients = coef(mu0_model)

    # ==========================================================================
    # Fit Treated Model: μ₁(X) = E[Y|T=1, X]
    # ==========================================================================

    df_treated = DataFrame(X_treated, :auto)
    df_treated.Y = Y_treated

    mu1_model = try
        lm(formula, df_treated)
    catch e
        msg = hasfield(typeof(e), :msg) ? e.msg : string(e)
        throw(ArgumentError(
            "Treated outcome model failed to fit: $msg. " *
            "Check for collinearity or insufficient variation."
        ))
    end

    mu1_coefficients = coef(mu1_model)

    # ==========================================================================
    # Predict on ALL Covariates
    # ==========================================================================

    # Build full DataFrame for prediction
    df_all = DataFrame(covariates, :auto)
    df_all.Y = outcomes  # Needed for predict (even though not used)

    # GLM.predict returns Vector{Union{Missing, T}} - convert to Vector{T}
    mu0_predictions = convert(Vector{T}, collect(skipmissing(predict(mu0_model, df_all))))
    mu1_predictions = convert(Vector{T}, collect(skipmissing(predict(mu1_model, df_all))))

    # ==========================================================================
    # Compute In-Sample R²
    # ==========================================================================

    mu0_fitted = convert(Vector{T}, collect(skipmissing(predict(mu0_model, df_control))))
    mu1_fitted = convert(Vector{T}, collect(skipmissing(predict(mu1_model, df_treated))))

    mu0_r2 = compute_r2(Y_control, mu0_fitted)
    mu1_r2 = compute_r2(Y_treated, mu1_fitted)

    return (
        mu0_predictions = convert(Vector{T}, mu0_predictions),
        mu1_predictions = convert(Vector{T}, mu1_predictions),
        mu0_r2 = T(mu0_r2),
        mu1_r2 = T(mu1_r2),
        mu0_coefficients = convert(Vector{T}, mu0_coefficients),
        mu1_coefficients = convert(Vector{T}, mu1_coefficients)
    )
end


"""
    compute_r2(y_true, y_pred)

Compute coefficient of determination (R²).

R² = 1 - SS_res / SS_tot
   = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²

# Arguments
- `y_true::Vector{T}`: True values
- `y_pred::Vector{T}`: Predicted values

# Returns
- `r2::T`: Coefficient of determination

# Notes
- R² can be negative if model is worse than mean prediction
- R² = 1.0 means perfect prediction
- R² = 0.0 means model explains no variance
"""
function compute_r2(
    y_true::AbstractVector{T},
    y_pred::AbstractVector{T}
) where {T<:Real}
    y_mean = mean(y_true)

    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- y_mean).^2)

    if ss_tot < 1e-10
        # If no variance in y, R² undefined, return 0
        return T(0)
    end

    r2 = 1 - ss_res / ss_tot
    return T(r2)
end
