#=
Sequential G-Estimation for Dynamic Treatment Effects.

Implements the sequential g-estimation algorithm from Lewis & Syrgkanis (2021)
for estimating dynamic treatment effects with machine learning nuisance estimation.

Reference:
Lewis, G., & Syrgkanis, V. (2021). Double/Debiased Machine Learning for
Dynamic Treatment Effects via g-Estimation. arXiv:2002.07285.
=#

module GEstimation

using Statistics
using LinearAlgebra

export sequential_g_estimation, aggregate_fold_estimates
export compute_cumulative_effect, compute_cumulative_influence

"""
    get_nuisance_model(model_type::Symbol, task::Symbol)

Get a nuisance model for outcome or propensity estimation.
Returns a simple ridge regression function for now.
"""
function fit_ridge(X::Matrix{Float64}, y::Vector{Float64}; alpha::Float64=1.0)
    n, p = size(X)
    # Ridge: (X'X + αI)^{-1} X'y
    XtX = X' * X + alpha * I
    beta = XtX \ (X' * y)
    return beta
end

function predict_ridge(X::Matrix{Float64}, beta::Vector{Float64})
    return X * beta
end

"""
    sequential_g_estimation(Y, T_lagged, X, train_mask, test_mask; nuisance_model=:ridge)

Core sequential g-estimation algorithm.

Implements the peeling procedure from Lewis & Syrgkanis (2021):
For h = max_lag, max_lag-1, ..., 0:
    1. Adjust outcome: Ỹ_h = Y - Σ_{k>h} θ̂_k T_{t-k}
    2. Estimate nuisances: q(X) = E[Ỹ_h | X], p(X) = E[T_h | X]
    3. Solve moment: E[(Ỹ_h - θ_h T_h - q(X)) (T_h - p(X))] = 0
    4. Estimate θ_h = Cov(Ỹ_h - q(X), T_h - p(X)) / Var(T_h - p(X))

# Arguments
- `Y`: Outcomes, shape (n,)
- `T_lagged`: Lagged treatments, shape (n, max_lag + 1, n_treatments)
- `X`: Covariates, shape (n, p)
- `train_mask`: Boolean mask for training
- `test_mask`: Boolean mask for testing
- `nuisance_model`: Model type (:ridge, :rf, :gb)

# Returns
- `theta`: Treatment effects, shape (max_lag + 1, n_treatments)
- `influence_scores`: Influence for test obs, shape (n_test, max_lag + 1)
- `nuisance_r2`: R² values for nuisance models
"""
function sequential_g_estimation(
    Y::Vector{Float64},
    T_lagged::Array{Float64,3},
    X::Matrix{Float64},
    train_mask::BitVector,
    test_mask::BitVector;
    nuisance_model::Symbol=:ridge
)
    n, max_lag_plus_1, n_treatments = size(T_lagged)
    max_lag = max_lag_plus_1 - 1

    # Storage
    theta = zeros(max_lag_plus_1, n_treatments)
    n_test = sum(test_mask)
    influence_scores = zeros(n_test, max_lag_plus_1)
    nuisance_r2 = Dict{Symbol,Vector{Float64}}(
        :outcome_r2 => Float64[],
        :propensity_r2 => Float64[]
    )

    # Get train/test data
    Y_train = Y[train_mask]
    Y_test = Y[test_mask]
    X_train = X[train_mask, :]
    X_test = X[test_mask, :]
    T_train = T_lagged[train_mask, :, :]
    T_test = T_lagged[test_mask, :, :]

    # Adjusted outcome (will be modified as we peel)
    Y_adj_train = copy(Y_train)
    Y_adj_test = copy(Y_test)

    # Sequential peeling: from most distant lag to lag 0
    for h in max_lag:-1:0
        # Current treatment at lag h (use first treatment if multiple)
        T_h_train = T_train[:, h + 1, 1]
        T_h_test = T_test[:, h + 1, 1]

        # Step 1: Estimate outcome nuisance q(X) = E[Y_adj | X]
        beta_outcome = fit_ridge(X_train, Y_adj_train)
        q_train = predict_ridge(X_train, beta_outcome)
        q_test = predict_ridge(X_test, beta_outcome)

        # Outcome R²
        ss_res = sum((Y_adj_train .- q_train).^2)
        ss_tot = sum((Y_adj_train .- mean(Y_adj_train)).^2)
        r2_outcome = ss_tot > 0 ? 1 - ss_res / ss_tot : 0.0
        push!(nuisance_r2[:outcome_r2], r2_outcome)

        # Step 2: Estimate propensity nuisance p(X) = E[T | X]
        beta_propensity = fit_ridge(X_train, T_h_train)
        p_train = predict_ridge(X_train, beta_propensity)
        p_test = predict_ridge(X_test, beta_propensity)

        # Propensity R²
        ss_res_p = sum((T_h_train .- p_train).^2)
        ss_tot_p = sum((T_h_train .- mean(T_h_train)).^2)
        r2_propensity = ss_tot_p > 0 ? 1 - ss_res_p / ss_tot_p : 0.0
        push!(nuisance_r2[:propensity_r2], r2_propensity)

        # Step 3: Compute residuals
        Y_tilde_train = Y_adj_train .- q_train
        T_tilde_train = T_h_train .- p_train
        Y_tilde_test = Y_adj_test .- q_test
        T_tilde_test = T_h_test .- p_test

        # Step 4: Estimate theta_h via moment condition
        # theta = Cov(Y_tilde, T_tilde) / Var(T_tilde)
        cov_val = mean(Y_tilde_train .* T_tilde_train)
        var_t = mean(T_tilde_train.^2)

        theta_h = var_t > 1e-10 ? cov_val / var_t : 0.0
        theta[h + 1, 1] = theta_h

        # Step 5: Compute influence scores for test observations
        # ψ = (Y_tilde - θ T_tilde) * T_tilde / Var(T_tilde)
        if var_t > 1e-10
            psi = (Y_tilde_test .- theta_h .* T_tilde_test) .* T_tilde_test ./ var_t
        else
            psi = zeros(n_test)
        end
        influence_scores[:, h + 1] = psi

        # Step 6: Adjust outcomes for next iteration (peel off this lag's effect)
        Y_adj_train = Y_adj_train .- theta_h .* T_h_train
        Y_adj_test = Y_adj_test .- theta_h .* T_h_test
    end

    return theta, influence_scores, nuisance_r2
end

"""
    aggregate_fold_estimates(fold_thetas, fold_influences, fold_n_test)

Aggregate treatment effect estimates across cross-fitting folds.
"""
function aggregate_fold_estimates(
    fold_thetas::Vector{Matrix{Float64}},
    fold_influences::Vector{Matrix{Float64}},
    fold_n_test::Vector{Int}
)
    n_total = sum(fold_n_test)
    max_lag_plus_1 = size(fold_thetas[1], 1)
    n_treatments = size(fold_thetas[1], 2)

    # Weighted average of theta estimates
    theta = zeros(max_lag_plus_1, n_treatments)
    for (fold_theta, n_test) in zip(fold_thetas, fold_n_test)
        theta .+= fold_theta .* n_test
    end
    theta ./= n_total

    # Concatenate influence scores
    influence = vcat(fold_influences...)

    return theta, influence
end

"""
    compute_cumulative_effect(theta; discount_factor=0.99)

Compute discounted cumulative treatment effect: Θ = Σ_h δ^h θ_h
"""
function compute_cumulative_effect(theta::AbstractVecOrMat{<:Real}; discount_factor::Float64=0.99)
    θ = theta isa AbstractMatrix ? theta[:, 1] : theta
    max_lag_plus_1 = length(θ)
    weights = [discount_factor^h for h in 0:(max_lag_plus_1 - 1)]
    cumulative = sum(weights .* θ)
    return cumulative, weights
end

"""
    compute_cumulative_influence(influence, weights)

Compute influence function for cumulative effect.
"""
function compute_cumulative_influence(influence::Matrix{Float64}, weights::Vector{Float64})
    return influence * weights
end

end  # module GEstimation
