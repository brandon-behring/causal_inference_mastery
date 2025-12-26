"""
    Panel DML-CRE (Mundlak Approach)

Double Machine Learning with Correlated Random Effects for panel data,
implementing the Mundlak (1978) approach to handle unobserved unit heterogeneity.

# Key Features
- Stratified cross-fitting by unit (preserves panel structure)
- Mundlak projection: includes time-means X̄ᵢ as covariates
- Clustered standard errors at unit level
- Supports balanced and unbalanced panels

# References
- Mundlak, Y. (1978). "On the pooling of time series and cross section data."
- Chernozhukov et al. (2018). "Double/debiased machine learning."
"""

using Statistics
using Random
using Distributions
using GLM

include("types.jl")

"""
    stratified_kfold_by_unit(unit_id, n_folds; random_state=42)

Create stratified K-fold splits by unit.

Each fold contains complete unit histories - no unit is split across train and test.

# Arguments
- `unit_id::Vector{Int}`: Unit identifiers for each observation.
- `n_folds::Int`: Number of folds.
- `random_state::Int`: Random seed for reproducibility.

# Returns
- Vector of (train_idx, test_idx) tuples.
"""
function stratified_kfold_by_unit(
    unit_id::Vector{Int},
    n_folds::Int;
    random_state::Int=42
)
    Random.seed!(random_state)

    unique_units = unique(unit_id)
    n_units_total = length(unique_units)

    n_folds <= n_units_total || error(
        "CRITICAL ERROR: n_folds > n_units.\n" *
        "Function: stratified_kfold_by_unit\n" *
        "n_folds: $n_folds, n_units: $n_units_total\n" *
        "Cannot have more folds than units."
    )

    # Shuffle units
    shuffled_units = shuffle(unique_units)

    # Assign units to folds
    fold_assignment = [(i - 1) % n_folds + 1 for i in 1:n_units_total]

    # Create fold splits
    folds = Vector{Tuple{Vector{Int}, Vector{Int}}}()

    for k in 1:n_folds
        test_units = shuffled_units[fold_assignment .== k]
        train_units = shuffled_units[fold_assignment .!= k]

        test_idx = findall(u -> u in test_units, unit_id)
        train_idx = findall(u -> u in train_units, unit_id)

        push!(folds, (train_idx, test_idx))
    end

    return folds
end

"""
    fit_ridge_regression(X, y; lambda=1.0)

Fit ridge regression and return prediction function.
"""
function fit_ridge_regression(X::Matrix{T}, y::Vector{T}; lambda::Real=1.0) where T
    n, p = size(X)
    XtX = X' * X
    Xty = X' * y

    # Ridge solution: (X'X + λI)⁻¹ X'y
    coef = (XtX + lambda * I(p)) \ Xty
    return coef
end

"""
    cross_fit_panel_nuisance_binary(panel, X_augmented, n_folds)

Cross-fit nuisance models for binary treatment with stratified folds.

Returns (m_hat, e_hat, fold_info).
"""
function cross_fit_panel_nuisance_binary(
    panel::PanelData{T},
    X_augmented::Matrix{T},
    n_folds::Int;
    random_state::Int=42
) where T
    n = n_obs(panel)
    m_hat = zeros(T, n)
    e_hat = zeros(T, n)

    fold_info = stratified_kfold_by_unit(panel.unit_id, n_folds; random_state=random_state)

    for (train_idx, test_idx) in fold_info
        X_train = X_augmented[train_idx, :]
        X_test = X_augmented[test_idx, :]
        Y_train = panel.outcomes[train_idx]
        D_train = panel.treatment[train_idx]

        # Add intercept
        X_train_int = hcat(ones(length(train_idx)), X_train)
        X_test_int = hcat(ones(length(test_idx)), X_test)

        # Fit outcome model: E[Y|X, X̄] using ridge
        coef_y = fit_ridge_regression(X_train_int, Y_train)
        m_hat[test_idx] = X_test_int * coef_y

        # Fit propensity model: P(D=1|X, X̄) using logistic regression
        # Simple logistic via GLM
        df_train = (y=D_train, X=X_train_int)
        # Use ridge-penalized logistic approximation
        coef_d = fit_ridge_regression(X_train_int, D_train; lambda=0.1)
        linear_pred = X_test_int * coef_d
        # Sigmoid transform
        e_hat[test_idx] = 1.0 ./ (1.0 .+ exp.(-linear_pred))
    end

    # Clip propensity to avoid extreme weights
    e_hat .= clamp.(e_hat, 0.01, 0.99)

    return m_hat, e_hat, fold_info
end

"""
    cross_fit_panel_nuisance_continuous(panel, X_augmented, n_folds)

Cross-fit nuisance models for continuous treatment with stratified folds.

Returns (m_hat, g_hat, fold_info).
"""
function cross_fit_panel_nuisance_continuous(
    panel::PanelData{T},
    X_augmented::Matrix{T},
    n_folds::Int;
    random_state::Int=42
) where T
    n = n_obs(panel)
    m_hat = zeros(T, n)
    g_hat = zeros(T, n)

    fold_info = stratified_kfold_by_unit(panel.unit_id, n_folds; random_state=random_state)

    for (train_idx, test_idx) in fold_info
        X_train = X_augmented[train_idx, :]
        X_test = X_augmented[test_idx, :]
        Y_train = panel.outcomes[train_idx]
        D_train = panel.treatment[train_idx]

        # Add intercept
        X_train_int = hcat(ones(length(train_idx)), X_train)
        X_test_int = hcat(ones(length(test_idx)), X_test)

        # Fit outcome model: E[Y|X, X̄] using ridge
        coef_y = fit_ridge_regression(X_train_int, Y_train)
        m_hat[test_idx] = X_test_int * coef_y

        # Fit treatment model: E[D|X, X̄] using ridge
        coef_d = fit_ridge_regression(X_train_int, D_train)
        g_hat[test_idx] = X_test_int * coef_d
    end

    return m_hat, g_hat, fold_info
end

"""
    clustered_influence_se(Y_tilde, D_tilde, theta, unit_id)

Compute clustered standard error via influence function.

For panel data, we cluster the influence function at the unit level
to account for within-unit correlation.
"""
function clustered_influence_se(
    Y_tilde::Vector{T},
    D_tilde::Vector{T},
    theta::Float64,
    unit_id::Vector{Int}
) where T
    n = length(Y_tilde)
    unique_units = unique(unit_id)
    n_units_total = length(unique_units)

    # Denominator: E[D_tilde²]
    D_tilde_sq_mean = mean(D_tilde .^ 2)

    if D_tilde_sq_mean < 1e-10
        return std(Y_tilde) / sqrt(n)
    end

    # Compute influence function for each observation
    psi = (Y_tilde .- theta .* D_tilde) .* D_tilde ./ D_tilde_sq_mean

    # Sum influence functions within each cluster (unit)
    cluster_psi = zeros(n_units_total)
    for (i, unit) in enumerate(unique_units)
        unit_idx = findall(==(unit), unit_id)
        cluster_psi[i] = sum(psi[unit_idx])
    end

    # Clustered variance: Var(θ̂) = (1/n²) * Σᵢ (Σₜ ψᵢₜ)²
    var_theta = sum(cluster_psi .^ 2) / (n^2)

    return sqrt(var_theta)
end

"""
    compute_unit_effects_from_residuals(panel, Y_tilde, D_tilde, theta)

Estimate unit fixed effects from residuals.

α̂ᵢ = (1/Tᵢ) Σₜ (Ỹᵢₜ - θ·D̃ᵢₜ)
"""
function compute_unit_effects_from_residuals(
    panel::PanelData{T},
    Y_tilde::Vector{T},
    D_tilde::Vector{T},
    theta::Float64
) where T
    unique_units = get_unique_units(panel)
    n_units_total = length(unique_units)
    unit_effects = zeros(n_units_total)

    for (i, unit) in enumerate(unique_units)
        unit_idx = get_unit_indices(panel, unit)
        residual = Y_tilde[unit_idx] .- theta .* D_tilde[unit_idx]
        unit_effects[i] = mean(residual)
    end

    return unit_effects
end

"""
    compute_fold_estimates(Y_tilde, D_tilde, unit_id, fold_info)

Compute per-fold ATE estimates for stability analysis.
"""
function compute_fold_estimates(
    Y_tilde::Vector{T},
    D_tilde::Vector{T},
    unit_id::Vector{Int},
    fold_info::Vector{Tuple{Vector{Int}, Vector{Int}}}
) where T
    n_folds = length(fold_info)
    fold_estimates = zeros(n_folds)
    fold_ses = zeros(n_folds)

    for (i, (_, test_idx)) in enumerate(fold_info)
        Y_fold = Y_tilde[test_idx]
        D_fold = D_tilde[test_idx]
        unit_fold = unit_id[test_idx]

        D_sq_sum = sum(D_fold .^ 2)
        if D_sq_sum > 1e-10
            fold_estimates[i] = sum(Y_fold .* D_fold) / D_sq_sum
            fold_ses[i] = clustered_influence_se(
                Y_fold, D_fold, fold_estimates[i], unit_fold
            )
        else
            fold_estimates[i] = NaN
            fold_ses[i] = NaN
        end
    end

    return fold_estimates, fold_ses
end

"""
    dml_cre(panel; n_folds=5, alpha=0.05)

Estimate treatment effects using Panel DML-CRE with binary treatment.

Implements Double Machine Learning with Correlated Random Effects
for panel data following Mundlak (1978).

# Arguments
- `panel::PanelData`: Panel data with outcomes, treatment, covariates, unit_id, time.
- `n_folds::Int`: Number of folds for cross-fitting. Must be <= n_units.
- `alpha::Float64`: Significance level for confidence intervals.

# Returns
- `DMLCREResult`: Results including ATE, SE, CATE, unit effects, and diagnostics.

# Examples
```julia
using Random
Random.seed!(42)

# Generate panel data
n_units, n_periods = 50, 10
n_obs_total = n_units * n_periods
unit_id = repeat(1:n_units, inner=n_periods)
time = repeat(1:n_periods, outer=n_units)
X = randn(n_obs_total, 3)
D = rand(n_obs_total) .< 0.5
Y = X[:, 1] .+ 2.0 .* D .+ randn(n_obs_total)

panel = PanelData(Y, Float64.(D), X, unit_id, time)
result = dml_cre(panel)
println("ATE: \$(round(result.ate, digits=3)) ± \$(round(result.ate_se, digits=3))")
```

# Notes
**Mundlak (1978) Approach**:

The key insight is that unobserved unit effects αᵢ may be correlated
with covariates. Mundlak's solution:

1. Assume E[αᵢ | Xᵢ] = γ·X̄ᵢ (linear projection)
2. Include X̄ᵢ = mean(Xᵢₜ over t) as additional covariates
3. This controls for time-invariant confounding through the projection

**Stratified Cross-Fitting**:

Unlike standard DML which splits by observation, Panel DML-CRE splits
by unit to preserve the panel structure. All observations for unit i
are in the same fold.
"""
function dml_cre(
    panel::PanelData{T};
    n_folds::Int=5,
    alpha::Float64=0.05
) where T
    # Validate binary treatment
    unique_treatments = unique(panel.treatment)
    sort!(unique_treatments)
    (unique_treatments ≈ [0.0, 1.0]) || error(
        "CRITICAL ERROR: Treatment must be binary (0, 1).\n" *
        "Function: dml_cre\n" *
        "Found unique values: $unique_treatments\n" *
        "Use dml_cre_continuous for continuous treatment."
    )

    # Validate n_folds
    n_folds >= 2 || error(
        "CRITICAL ERROR: n_folds must be >= 2.\n" *
        "Function: dml_cre\n" *
        "Got: n_folds = $n_folds"
    )

    n_units_total = n_units(panel)
    n_folds <= n_units_total || error(
        "CRITICAL ERROR: n_folds > n_units.\n" *
        "Function: dml_cre\n" *
        "n_folds: $n_folds, n_units: $n_units_total\n" *
        "Cannot have more folds than units."
    )

    n = n_obs(panel)

    # =========================================================================
    # Step 1: Compute unit means (Mundlak projection)
    # =========================================================================
    unit_means = compute_unit_means(panel)

    # Augment covariates with unit means: [Xᵢₜ, X̄ᵢ]
    X_augmented = hcat(panel.covariates, unit_means)

    # =========================================================================
    # Step 2: Cross-fit nuisance models
    # =========================================================================
    m_hat, e_hat, fold_info = cross_fit_panel_nuisance_binary(
        panel, X_augmented, n_folds
    )

    # Compute R-squared for diagnostics
    ss_total_y = sum((panel.outcomes .- mean(panel.outcomes)) .^ 2)
    ss_resid_y = sum((panel.outcomes .- m_hat) .^ 2)
    outcome_r2 = ss_total_y > 0 ? 1 - ss_resid_y / ss_total_y : 0.0

    # Pseudo R² for propensity (McFadden)
    null_ll = sum(panel.treatment .* log(0.5) .+ (1 .- panel.treatment) .* log(0.5))
    model_ll = sum(
        panel.treatment .* log.(e_hat .+ 1e-10) .+
        (1 .- panel.treatment) .* log.(1 .- e_hat .+ 1e-10)
    )
    treatment_r2 = null_ll != 0 ? 1 - model_ll / null_ll : 0.0

    # =========================================================================
    # Step 3: Compute residuals
    # =========================================================================
    Y_tilde = panel.outcomes .- m_hat
    D_tilde = panel.treatment .- e_hat

    # =========================================================================
    # Step 4: Estimate ATE
    # =========================================================================
    D_tilde_sq_sum = sum(D_tilde .^ 2)

    D_tilde_sq_sum >= 1e-10 || error(
        "CRITICAL ERROR: Treatment residuals too small.\n" *
        "Function: dml_cre\n" *
        "Sum of D_tilde² = $D_tilde_sq_sum\n" *
        "Propensity model may be too good (no residual variation)."
    )

    ate = sum(Y_tilde .* D_tilde) / D_tilde_sq_sum

    # =========================================================================
    # Step 5: Estimate CATE (heterogeneous effects)
    # =========================================================================
    # For CATE, use weighted regression on X_augmented
    X_transformed = X_augmented .* D_tilde
    X_with_intercept = hcat(X_transformed, D_tilde)

    coef_cate = fit_ridge_regression(X_with_intercept, Y_tilde; lambda=0.0)

    X_pred = hcat(X_augmented, ones(n))
    cate = X_pred * coef_cate

    # =========================================================================
    # Step 6: Compute SE via clustered influence function
    # =========================================================================
    ate_se = clustered_influence_se(Y_tilde, D_tilde, ate, panel.unit_id)

    # Confidence interval
    z_crit = quantile(Normal(), 1 - alpha / 2)
    ci_lower = ate - z_crit * ate_se
    ci_upper = ate + z_crit * ate_se

    # =========================================================================
    # Step 7: Compute unit effects
    # =========================================================================
    unit_effects = compute_unit_effects_from_residuals(panel, Y_tilde, D_tilde, ate)

    # =========================================================================
    # Step 8: Per-fold estimates
    # =========================================================================
    fold_estimates, fold_ses = compute_fold_estimates(
        Y_tilde, D_tilde, panel.unit_id, fold_info
    )

    return DMLCREResult(
        Float64(ate),
        Float64(ate_se),
        Float64(ci_lower),
        Float64(ci_upper),
        Float64.(cate),
        :dml_cre,
        n_units_total,
        n,
        n_folds,
        Float64(outcome_r2),
        Float64(treatment_r2),
        Float64.(unit_effects),
        Float64.(fold_estimates),
        Float64.(fold_ses)
    )
end

"""
    dml_cre_continuous(panel; n_folds=5, alpha=0.05)

Estimate treatment effects using Panel DML-CRE with continuous treatment.

Implements Double Machine Learning with Correlated Random Effects
for panel data following Mundlak (1978), adapted for continuous treatment.

# Arguments
- `panel::PanelData`: Panel data with outcomes, treatment, covariates, unit_id, time.
- `n_folds::Int`: Number of folds for cross-fitting. Must be <= n_units.
- `alpha::Float64`: Significance level for confidence intervals.

# Returns
- `DMLCREResult`: Results including ATE (marginal effect dE[Y]/dD), SE, CATE,
  unit effects, and diagnostics.

# Notes
**Key Difference from Binary Treatment**:

For continuous treatment, the propensity model is replaced by a
treatment regression model:
- Binary: e(X) = P(D=1|X) using classification
- Continuous: g(X) = E[D|X] using regression

**Interpretation**:

The ATE θ represents the marginal effect dE[Y]/dD, interpreted as:
"A one-unit increase in treatment D causes a θ-unit change in Y,
controlling for covariates and unobserved unit heterogeneity."
"""
function dml_cre_continuous(
    panel::PanelData{T};
    n_folds::Int=5,
    alpha::Float64=0.05
) where T
    # Validate n_folds
    n_folds >= 2 || error(
        "CRITICAL ERROR: n_folds must be >= 2.\n" *
        "Function: dml_cre_continuous\n" *
        "Got: n_folds = $n_folds"
    )

    n_units_total = n_units(panel)
    n_folds <= n_units_total || error(
        "CRITICAL ERROR: n_folds > n_units.\n" *
        "Function: dml_cre_continuous\n" *
        "n_folds: $n_folds, n_units: $n_units_total\n" *
        "Cannot have more folds than units."
    )

    # Check treatment variation
    treatment_std = std(panel.treatment)
    treatment_std >= 1e-6 || error(
        "CRITICAL ERROR: Treatment has no variation.\n" *
        "Function: dml_cre_continuous\n" *
        "std(D) = $treatment_std\n" *
        "Continuous treatment requires variation."
    )

    n = n_obs(panel)

    # =========================================================================
    # Step 1: Compute unit means (Mundlak projection)
    # =========================================================================
    unit_means = compute_unit_means(panel)

    # Augment covariates with unit means: [Xᵢₜ, X̄ᵢ]
    X_augmented = hcat(panel.covariates, unit_means)

    # =========================================================================
    # Step 2: Cross-fit nuisance models
    # =========================================================================
    m_hat, g_hat, fold_info = cross_fit_panel_nuisance_continuous(
        panel, X_augmented, n_folds
    )

    # Compute R-squared for diagnostics
    ss_total_y = sum((panel.outcomes .- mean(panel.outcomes)) .^ 2)
    ss_resid_y = sum((panel.outcomes .- m_hat) .^ 2)
    outcome_r2 = ss_total_y > 0 ? 1 - ss_resid_y / ss_total_y : 0.0

    ss_total_d = sum((panel.treatment .- mean(panel.treatment)) .^ 2)
    ss_resid_d = sum((panel.treatment .- g_hat) .^ 2)
    treatment_r2 = ss_total_d > 0 ? 1 - ss_resid_d / ss_total_d : 0.0

    # =========================================================================
    # Step 3: Compute residuals
    # =========================================================================
    Y_tilde = panel.outcomes .- m_hat
    D_tilde = panel.treatment .- g_hat

    # =========================================================================
    # Step 4: Estimate ATE (marginal effect)
    # =========================================================================
    D_tilde_sq_sum = sum(D_tilde .^ 2)

    D_tilde_sq_sum >= 1e-10 || error(
        "CRITICAL ERROR: Treatment residuals too small.\n" *
        "Function: dml_cre_continuous\n" *
        "Sum of D_tilde² = $D_tilde_sq_sum\n" *
        "Treatment model may be too good (no residual variation)."
    )

    ate = sum(Y_tilde .* D_tilde) / D_tilde_sq_sum

    # =========================================================================
    # Step 5: Estimate CATE (heterogeneous effects)
    # =========================================================================
    # For CATE, use weighted regression on X_augmented
    X_transformed = X_augmented .* D_tilde
    X_with_intercept = hcat(X_transformed, D_tilde)

    coef_cate = fit_ridge_regression(X_with_intercept, Y_tilde; lambda=0.0)

    X_pred = hcat(X_augmented, ones(n))
    cate = X_pred * coef_cate

    # =========================================================================
    # Step 6: Compute SE via clustered influence function
    # =========================================================================
    ate_se = clustered_influence_se(Y_tilde, D_tilde, ate, panel.unit_id)

    # Confidence interval
    z_crit = quantile(Normal(), 1 - alpha / 2)
    ci_lower = ate - z_crit * ate_se
    ci_upper = ate + z_crit * ate_se

    # =========================================================================
    # Step 7: Compute unit effects
    # =========================================================================
    unit_effects = compute_unit_effects_from_residuals(panel, Y_tilde, D_tilde, ate)

    # =========================================================================
    # Step 8: Per-fold estimates
    # =========================================================================
    fold_estimates, fold_ses = compute_fold_estimates(
        Y_tilde, D_tilde, panel.unit_id, fold_info
    )

    return DMLCREResult(
        Float64(ate),
        Float64(ate_se),
        Float64(ci_lower),
        Float64(ci_upper),
        Float64.(cate),
        :dml_cre_continuous,
        n_units_total,
        n,
        n_folds,
        Float64(outcome_r2),
        Float64(treatment_r2),
        Float64.(unit_effects),
        Float64.(fold_estimates),
        Float64.(fold_ses)
    )
end
