#=
Latent CATE Methods: Factor Analysis, PPCA, GMM Stratification

Implements CATE estimation with latent confounder adjustment,
inspired by CEVAE (Louizos et al. 2017) but using simpler
dimensionality reduction instead of variational inference.

Methods:
- factor_analysis_cate: CATE with Factor Analysis latent factor augmentation
- ppca_cate: CATE with Probabilistic PCA component augmentation
- gmm_stratified_cate: CATE with GMM-based stratification

Session 156: Latent CATE Julia Parity.

References:
- Louizos et al. (2017). "Causal Effect Inference with Deep Latent-Variable Models."
  NeurIPS.
=#

using LinearAlgebra
using Statistics
using Random
using Distributions


# =============================================================================
# Core Algorithms
# =============================================================================

"""
    _pca_transform(X; n_components=5) -> Matrix

Extract principal components from data using SVD.

# Arguments
- `X::Matrix{T}`: Data matrix (n × p)
- `n_components::Int`: Number of principal components (default: 5)

# Returns
- `scores::Matrix{T}`: Principal component scores (n × k)
"""
function _pca_transform(X::Matrix{T}; n_components::Int=5) where {T<:Real}
    n, p = size(X)

    # Cap at min(n, p) - 1 to avoid degenerate case
    k = min(n_components, min(n, p) - 1, p)
    if k < 1
        k = 1
    end

    # Center data
    means = vec(mean(X, dims=1))
    X_centered = X .- means'

    # SVD decomposition
    F = svd(X_centered)

    # Project onto first k components
    # Scores = U * S[:, 1:k] or equivalently X_centered * V[:, 1:k]
    scores = X_centered * F.V[:, 1:k]

    return scores
end


"""
    _factor_analysis(X; n_factors=5, max_iter=100, tol=1e-4) -> Matrix

Extract latent factors using Factor Analysis EM algorithm.

Model: X = L @ F + noise, where L is the loading matrix and F are factors.

# Arguments
- `X::Matrix{T}`: Data matrix (n × p)
- `n_factors::Int`: Number of latent factors (default: 5)
- `max_iter::Int`: Maximum EM iterations (default: 100)
- `tol::Float64`: Convergence tolerance (default: 1e-4)

# Returns
- `F::Matrix{T}`: Factor scores (n × k)
"""
function _factor_analysis(
    X::Matrix{T};
    n_factors::Int = 5,
    max_iter::Int = 100,
    tol::Float64 = 1e-4
) where {T<:Real}
    n, p = size(X)
    k = min(n_factors, p - 1, n - 1)
    if k < 1
        k = 1
    end

    # Center data
    means = vec(mean(X, dims=1))
    X_c = X .- means'

    # Initialize via PCA
    initial_scores = _pca_transform(X; n_components=k)
    L = (X_c' * initial_scores) / T(n)  # Loading matrix p × k

    # Initialize noise variance (diagonal)
    residuals = X_c .- initial_scores * L'
    Psi = vec(var(residuals, dims=1))
    Psi = max.(Psi, T(1e-6))  # Ensure positive

    F = zeros(T, n, k)  # Factor scores

    prev_ll = T(-Inf)

    for iter in 1:max_iter
        # E-step: Compute expected factor scores
        # E[F|X] = (L'Psi⁻¹L + I)⁻¹ L'Psi⁻¹X'
        Psi_inv = T(1.0) ./ Psi
        LtPsiInvL = L' * (Psi_inv .* L)  # k × k
        M = LtPsiInvL + I(k)  # k × k
        M_inv = inv(M)

        # Factor scores: n × k
        # F = X_c @ diag(Psi_inv) @ L @ M_inv'
        X_scaled = X_c .* Psi_inv'
        F = (X_scaled * L) * M_inv

        # M-step: Update loading matrix and noise variance
        # L_new = X' @ F @ (F'F + n*I*sigma²)⁻¹ ≈ X'F / n
        FtF = F' * F
        FtF_reg = FtF + T(1e-6) * I(k)  # Regularization
        L_new = (X_c' * F) / T(n)

        # Update noise variance
        reconstruction = F * L_new'
        residuals = X_c .- reconstruction
        Psi_new = vec(var(residuals, dims=1, corrected=false))
        Psi_new = max.(Psi_new, T(1e-6))

        # Check convergence via log-likelihood proxy (reconstruction error)
        ll = -T(0.5) * sum(residuals.^2)

        if abs(ll - prev_ll) < tol * abs(prev_ll + T(1e-10))
            L = L_new
            Psi = Psi_new
            break
        end

        prev_ll = ll
        L = L_new
        Psi = Psi_new
    end

    return F
end


"""
    _kmeans_init(X, K; max_iter=50) -> Vector{Int}

Simple K-means initialization for GMM.

# Arguments
- `X::Matrix{T}`: Data matrix (n × p)
- `K::Int`: Number of clusters
- `max_iter::Int`: Maximum iterations

# Returns
- `labels::Vector{Int}`: Cluster assignments (1 to K)
"""
function _kmeans_init(X::Matrix{T}, K::Int; max_iter::Int = 50) where {T<:Real}
    n, p = size(X)

    # Initialize centroids by picking K random points
    indices = randperm(n)[1:min(K, n)]
    centroids = X[indices, :]

    labels = zeros(Int, n)

    for iter in 1:max_iter
        # Assign points to nearest centroid
        old_labels = copy(labels)
        for i in 1:n
            min_dist = T(Inf)
            for k in 1:K
                dist = sum((X[i, :] .- centroids[k, :]).^2)
                if dist < min_dist
                    min_dist = dist
                    labels[i] = k
                end
            end
        end

        # Check convergence
        if labels == old_labels
            break
        end

        # Update centroids
        for k in 1:K
            mask = labels .== k
            if sum(mask) > 0
                centroids[k, :] = vec(mean(X[mask, :], dims=1))
            end
        end
    end

    return labels
end


"""
    _gmm_fit(X; n_components=3, max_iter=100, tol=1e-4) -> Vector{Int}

Fit Gaussian Mixture Model using EM algorithm.

# Arguments
- `X::Matrix{T}`: Data matrix (n × p)
- `n_components::Int`: Number of mixture components (default: 3)
- `max_iter::Int`: Maximum EM iterations (default: 100)
- `tol::Float64`: Convergence tolerance (default: 1e-4)

# Returns
- `labels::Vector{Int}`: Hard cluster assignments
"""
function _gmm_fit(
    X::Matrix{T};
    n_components::Int = 3,
    max_iter::Int = 100,
    tol::Float64 = 1e-4
) where {T<:Real}
    n, p = size(X)
    K = n_components

    # K-means initialization
    labels = _kmeans_init(X, K)

    # Initialize parameters from K-means
    weights = zeros(T, K)
    means = [zeros(T, p) for _ in 1:K]
    covs = [Matrix{T}(I(p)) for _ in 1:K]

    for k in 1:K
        mask = labels .== k
        n_k = sum(mask)
        weights[k] = T(n_k) / T(n)

        if n_k > 0
            means[k] = vec(mean(X[mask, :], dims=1))
            if n_k > p
                C = cov(X[mask, :])
                # Regularize to ensure positive definite
                covs[k] = C + T(1e-6) * I(p)
            else
                covs[k] = Matrix{T}(I(p))
            end
        end
    end

    # EM iterations
    responsibilities = zeros(T, n, K)
    prev_ll = T(-Inf)

    for iter in 1:max_iter
        # E-step: Compute responsibilities
        for k in 1:K
            try
                mvn = MvNormal(means[k], Symmetric(covs[k]))
                for i in 1:n
                    responsibilities[i, k] = weights[k] * pdf(mvn, X[i, :])
                end
            catch
                # Fallback for numerical issues
                for i in 1:n
                    diff = X[i, :] - means[k]
                    responsibilities[i, k] = weights[k] * exp(-T(0.5) * sum(diff.^2))
                end
            end
        end

        # Normalize responsibilities
        row_sums = sum(responsibilities, dims=2)
        row_sums = max.(row_sums, T(1e-10))
        responsibilities ./= row_sums

        # M-step: Update parameters
        N_k = vec(sum(responsibilities, dims=1))

        for k in 1:K
            N_k_safe = max(N_k[k], T(1e-10))

            # Update weight
            weights[k] = N_k_safe / T(n)

            # Update mean
            means[k] = vec(sum(responsibilities[:, k] .* X, dims=1)) / N_k_safe

            # Update covariance
            X_centered = X .- means[k]'
            weighted_cov = (responsibilities[:, k] .* X_centered)' * X_centered / N_k_safe
            covs[k] = Symmetric(weighted_cov) + T(1e-6) * I(p)
        end

        # Check convergence via log-likelihood
        ll = sum(log.(max.(row_sums, T(1e-300))))

        if abs(ll - prev_ll) < tol * abs(prev_ll + T(1e-10))
            break
        end
        prev_ll = ll
    end

    # Hard assignments
    labels = [argmax(responsibilities[i, :]) for i in 1:n]

    return labels
end


# =============================================================================
# Estimator Types
# =============================================================================

"""
    FactorAnalysisCATEEstimator <: AbstractCATEEstimator

CATE estimation with Factor Analysis latent factor augmentation.

Extracts latent factors from covariates using Factor Analysis,
augments the feature space, and applies a base meta-learner.

# Fields
- `n_latent::Int`: Number of latent factors to extract
- `base_learner::Symbol`: Meta-learner to use (:t_learner or :r_learner)

# Example
```julia
using CausalEstimators

problem = CATEProblem(Y, T, X, (alpha=0.05,))
estimator = FactorAnalysisCATEEstimator(n_latent=3, base_learner=:t_learner)
solution = solve(problem, estimator)
```

# References
- Louizos et al. (2017). "Causal Effect Inference with Deep Latent-Variable Models."
"""
struct FactorAnalysisCATEEstimator <: AbstractCATEEstimator
    n_latent::Int
    base_learner::Symbol

    function FactorAnalysisCATEEstimator(;
        n_latent::Int = 5,
        base_learner::Symbol = :t_learner
    )
        if n_latent < 1
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid FactorAnalysisCATEEstimator configuration.\n" *
                "Function: FactorAnalysisCATEEstimator\n" *
                "n_latent must be >= 1, got $n_latent"
            ))
        end
        if base_learner ∉ (:t_learner, :r_learner)
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid FactorAnalysisCATEEstimator configuration.\n" *
                "Function: FactorAnalysisCATEEstimator\n" *
                "base_learner must be :t_learner or :r_learner, got :$base_learner"
            ))
        end
        new(n_latent, base_learner)
    end
end


"""
    PPCACATEEstimator <: AbstractCATEEstimator

CATE estimation with Probabilistic PCA augmentation.

Extracts principal components from covariates using PCA,
augments the feature space, and applies a base meta-learner.

# Fields
- `n_components::Int`: Number of principal components to extract
- `base_learner::Symbol`: Meta-learner to use (:t_learner or :r_learner)

# Example
```julia
using CausalEstimators

problem = CATEProblem(Y, T, X, (alpha=0.05,))
estimator = PPCACATEEstimator(n_components=3, base_learner=:t_learner)
solution = solve(problem, estimator)
```
"""
struct PPCACATEEstimator <: AbstractCATEEstimator
    n_components::Int
    base_learner::Symbol

    function PPCACATEEstimator(;
        n_components::Int = 5,
        base_learner::Symbol = :t_learner
    )
        if n_components < 1
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid PPCACATEEstimator configuration.\n" *
                "Function: PPCACATEEstimator\n" *
                "n_components must be >= 1, got $n_components"
            ))
        end
        if base_learner ∉ (:t_learner, :r_learner)
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid PPCACATEEstimator configuration.\n" *
                "Function: PPCACATEEstimator\n" *
                "base_learner must be :t_learner or :r_learner, got :$base_learner"
            ))
        end
        new(n_components, base_learner)
    end
end


"""
    GMMStratifiedCATEEstimator <: AbstractCATEEstimator

CATE estimation with GMM-based stratification.

Identifies latent subgroups using Gaussian Mixture Model,
then estimates CATE within each stratum.

# Fields
- `n_strata::Int`: Number of GMM components (strata)
- `base_learner::Symbol`: Meta-learner to use within strata (:t_learner or :r_learner)

# Example
```julia
using CausalEstimators

problem = CATEProblem(Y, T, X, (alpha=0.05,))
estimator = GMMStratifiedCATEEstimator(n_strata=3, base_learner=:t_learner)
solution = solve(problem, estimator)
```
"""
struct GMMStratifiedCATEEstimator <: AbstractCATEEstimator
    n_strata::Int
    base_learner::Symbol

    function GMMStratifiedCATEEstimator(;
        n_strata::Int = 3,
        base_learner::Symbol = :t_learner
    )
        if n_strata < 2
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid GMMStratifiedCATEEstimator configuration.\n" *
                "Function: GMMStratifiedCATEEstimator\n" *
                "n_strata must be >= 2, got $n_strata"
            ))
        end
        if base_learner ∉ (:t_learner, :r_learner)
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid GMMStratifiedCATEEstimator configuration.\n" *
                "Function: GMMStratifiedCATEEstimator\n" *
                "base_learner must be :t_learner or :r_learner, got :$base_learner"
            ))
        end
        new(n_strata, base_learner)
    end
end


# =============================================================================
# Solve Methods
# =============================================================================

"""
    solve(problem::CATEProblem, estimator::FactorAnalysisCATEEstimator) -> CATESolution

Estimate CATE using Factor Analysis latent factor augmentation.

# Algorithm
1. Extract latent factors F from covariates X using FA
2. Augment covariates: X_aug = [X, F]
3. Apply base meta-learner (T-learner or R-learner) on augmented data

# Arguments
- `problem::CATEProblem`: CATE estimation problem
- `estimator::FactorAnalysisCATEEstimator`: FA CATE estimator

# Returns
- `CATESolution`: Results with individual CATE, ATE, SE, and CI
"""
function solve(
    problem::CATEProblem{T,P},
    estimator::FactorAnalysisCATEEstimator
)::CATESolution{T,P} where {T<:Real, P<:NamedTuple}
    (; outcomes, treatment, covariates, parameters) = problem
    alpha = get(parameters, :alpha, 0.05)
    n, p = size(covariates)

    # Cap n_latent at p-1
    n_latent = min(estimator.n_latent, p - 1)
    if n_latent < 1
        n_latent = 1
    end

    # Extract latent factors
    factors = _factor_analysis(covariates; n_factors=n_latent)

    # Augment covariates
    X_augmented = hcat(covariates, factors)

    # Create augmented problem
    augmented_problem = CATEProblem(outcomes, treatment, X_augmented, parameters)

    # Apply base learner
    if estimator.base_learner == :t_learner
        base_solution = solve(augmented_problem, TLearner())
    else  # :r_learner
        base_solution = solve(augmented_problem, RLearner())
    end

    # Return with updated method name
    return CATESolution{T,P}(
        base_solution.cate,
        base_solution.ate,
        base_solution.se,
        base_solution.ci_lower,
        base_solution.ci_upper,
        :factor_analysis_cate,
        :Success,
        problem
    )
end


"""
    solve(problem::CATEProblem, estimator::PPCACATEEstimator) -> CATESolution

Estimate CATE using Probabilistic PCA augmentation.

# Algorithm
1. Extract principal components from covariates X using PCA
2. Augment covariates: X_aug = [X, PC]
3. Apply base meta-learner (T-learner or R-learner) on augmented data

# Arguments
- `problem::CATEProblem`: CATE estimation problem
- `estimator::PPCACATEEstimator`: PPCA CATE estimator

# Returns
- `CATESolution`: Results with individual CATE, ATE, SE, and CI
"""
function solve(
    problem::CATEProblem{T,P},
    estimator::PPCACATEEstimator
)::CATESolution{T,P} where {T<:Real, P<:NamedTuple}
    (; outcomes, treatment, covariates, parameters) = problem
    alpha = get(parameters, :alpha, 0.05)
    n, p = size(covariates)

    # Cap n_components at p-1
    n_components = min(estimator.n_components, p - 1)
    if n_components < 1
        n_components = 1
    end

    # Extract principal components
    pc_scores = _pca_transform(covariates; n_components=n_components)

    # Augment covariates
    X_augmented = hcat(covariates, pc_scores)

    # Create augmented problem
    augmented_problem = CATEProblem(outcomes, treatment, X_augmented, parameters)

    # Apply base learner
    if estimator.base_learner == :t_learner
        base_solution = solve(augmented_problem, TLearner())
    else  # :r_learner
        base_solution = solve(augmented_problem, RLearner())
    end

    # Return with updated method name
    return CATESolution{T,P}(
        base_solution.cate,
        base_solution.ate,
        base_solution.se,
        base_solution.ci_lower,
        base_solution.ci_upper,
        :ppca_cate,
        :Success,
        problem
    )
end


"""
    solve(problem::CATEProblem, estimator::GMMStratifiedCATEEstimator) -> CATESolution

Estimate CATE using GMM-based stratification.

# Algorithm
1. Fit GMM to identify latent strata in covariate space
2. For each stratum with sufficient samples:
   - Estimate CATE within stratum using base meta-learner
3. Aggregate stratum-specific estimates

# Arguments
- `problem::CATEProblem`: CATE estimation problem
- `estimator::GMMStratifiedCATEEstimator`: GMM stratified CATE estimator

# Returns
- `CATESolution`: Results with individual CATE, ATE, SE, and CI
"""
function solve(
    problem::CATEProblem{T,P},
    estimator::GMMStratifiedCATEEstimator
)::CATESolution{T,P} where {T<:Real, P<:NamedTuple}
    (; outcomes, treatment, covariates, parameters) = problem
    alpha = get(parameters, :alpha, 0.05)
    n, p = size(covariates)

    # Fit GMM to identify strata
    strata = _gmm_fit(covariates; n_components=estimator.n_strata)

    # Initialize CATE vector
    cate = zeros(T, n)
    stratum_ates = T[]
    stratum_weights = Int[]

    # Estimate CATE within each stratum
    for s in 1:estimator.n_strata
        stratum_mask = strata .== s
        n_stratum = sum(stratum_mask)

        if n_stratum < 4
            continue  # Skip strata with too few samples
        end

        # Check treatment variation in stratum
        treated_mask = treatment .& stratum_mask
        control_mask = (.!treatment) .& stratum_mask

        n_treated = sum(treated_mask)
        n_control = sum(control_mask)

        if n_treated < 2 || n_control < 2
            # Fallback: simple difference of means for sparse strata
            if n_treated > 0 && n_control > 0
                y_treated = mean(outcomes[treated_mask])
                y_control = mean(outcomes[control_mask])
                stratum_effect = y_treated - y_control
                cate[stratum_mask] .= stratum_effect
                push!(stratum_ates, stratum_effect)
                push!(stratum_weights, n_stratum)
            end
            continue
        end

        # Get stratum data
        X_stratum = covariates[stratum_mask, :]
        Y_stratum = outcomes[stratum_mask]
        T_stratum = treatment[stratum_mask]

        # Create stratum problem
        try
            stratum_problem = CATEProblem(Y_stratum, T_stratum, X_stratum, parameters)

            # Apply base learner
            if estimator.base_learner == :t_learner
                stratum_solution = solve(stratum_problem, TLearner())
            else  # :r_learner
                stratum_solution = solve(stratum_problem, RLearner())
            end

            cate[stratum_mask] = stratum_solution.cate
            push!(stratum_ates, stratum_solution.ate)
            push!(stratum_weights, n_stratum)
        catch e
            # Fallback on error
            if n_treated > 0 && n_control > 0
                y_treated = mean(outcomes[treated_mask])
                y_control = mean(outcomes[control_mask])
                stratum_effect = y_treated - y_control
                cate[stratum_mask] .= stratum_effect
                push!(stratum_ates, stratum_effect)
                push!(stratum_weights, n_stratum)
            end
        end
    end

    # Compute weighted ATE
    if !isempty(stratum_ates)
        weights = T.(stratum_weights) / sum(stratum_weights)
        ate = T(sum(stratum_ates .* weights))
    else
        # Fallback: simple difference of means
        n_treated = sum(treatment)
        n_control = n - n_treated
        if n_treated > 0 && n_control > 0
            ate = T(mean(outcomes[treatment]) - mean(outcomes[.!treatment]))
        else
            ate = T(0.0)
        end
    end

    # Compute SE from CATE variance
    cate_var = var(cate, corrected=true)
    se = T(sqrt(max(cate_var, T(1e-10)) / n))

    # Ensure SE is reasonable
    if !isfinite(se) || se <= 0
        n_treated = sum(treatment)
        n_control = n - n_treated
        if n_treated > 1 && n_control > 1
            var_treated = var(outcomes[treatment], corrected=true)
            var_control = var(outcomes[.!treatment], corrected=true)
            se = T(sqrt(var_treated / n_treated + var_control / n_control))
        else
            se = T(std(outcomes) / sqrt(n))
        end
    end

    # Confidence interval
    z_crit = T(quantile(Normal(), 1 - alpha / 2))
    ci_lower = ate - z_crit * se
    ci_upper = ate + z_crit * se

    return CATESolution{T,P}(
        cate,
        ate,
        se,
        ci_lower,
        ci_upper,
        :gmm_stratified_cate,
        :Success,
        problem
    )
end
