#=
Synthetic Control Methods - Weight Optimization

Implements simplex-constrained optimization for synthetic control weights:
    minimize ||Y₁_pre - Y₀_pre' * W||²
    subject to: W >= 0, sum(W) = 1
=#

using LinearAlgebra
using Statistics

"""
    compute_scm_weights(treated_pre, control_pre; max_iter=1000, tol=1e-8)

Compute optimal synthetic control weights via projected gradient descent.

# Arguments
- `treated_pre::Vector{T}`: Pre-treatment outcomes for treated (n_pre,)
- `control_pre::Matrix{T}`: Pre-treatment outcomes for controls (n_control, n_pre)
- `max_iter::Int`: Maximum iterations
- `tol::Float64`: Convergence tolerance

# Returns
- `weights::Vector{T}`: Optimal weights (n_control,)
- `converged::Bool`: Whether optimization converged
"""
function compute_scm_weights(
    treated_pre::Vector{T},
    control_pre::Matrix{T};
    covariates_treated::Union{Nothing,Vector{T}} = nothing,
    covariates_control::Union{Nothing,Matrix{T}} = nothing,
    covariate_weight::Float64 = 1.0,
    max_iter::Int = 1000,
    tol::Float64 = 1e-8,
) where {T<:Real}
    n_control, n_pre = size(control_pre)

    # Objective function: ||Y₁ - Y₀' * W||²
    # Gradient: 2 * Y₀ * (Y₀' * W - Y₁)

    function objective(w::Vector{T})
        synthetic = control_pre' * w  # (n_pre,)
        obj = sum((treated_pre .- synthetic) .^ 2)

        # Add covariate term if provided
        if covariates_treated !== nothing && covariates_control !== nothing
            cov_synthetic = covariates_control' * w
            obj += covariate_weight * sum((covariates_treated .- cov_synthetic) .^ 2)
        end

        return obj
    end

    function gradient!(g::Vector{T}, w::Vector{T})
        synthetic = control_pre' * w
        residual = synthetic .- treated_pre
        g .= 2.0 .* (control_pre * residual)

        # Add covariate gradient
        if covariates_treated !== nothing && covariates_control !== nothing
            cov_synthetic = covariates_control' * w
            cov_residual = cov_synthetic .- covariates_treated
            g .+= 2.0 .* covariate_weight .* (covariates_control * cov_residual)
        end

        return g
    end

    # Initialize with uniform weights
    w = fill(one(T) / n_control, n_control)
    g = similar(w)

    # Projected gradient descent with adaptive step size
    step_size = 1.0
    prev_obj = objective(w)
    converged = false

    for iter in 1:max_iter
        # Compute gradient
        gradient!(g, w)

        # Line search with Armijo condition
        for _ in 1:20
            w_new = w .- step_size .* g
            _project_simplex!(w_new)

            new_obj = objective(w_new)

            if new_obj < prev_obj - 1e-4 * step_size * dot(g, w .- w_new)
                w .= w_new
                prev_obj = new_obj
                step_size *= 1.2  # Increase step size
                break
            else
                step_size *= 0.5  # Decrease step size
            end
        end

        # Check convergence
        if norm(g) < tol
            converged = true
            break
        end
    end

    # Final projection to ensure constraints
    _project_simplex!(w)

    return w, converged
end

"""
    _project_simplex!(w)

Project vector onto probability simplex (w >= 0, sum(w) = 1).

Uses algorithm from Duchi et al. (2008) "Efficient Projections onto the
l1-Ball for Learning in High Dimensions".
"""
function _project_simplex!(w::Vector{T}) where {T<:Real}
    n = length(w)

    # Sort in descending order
    u = sort(w, rev=true)

    # Find ρ
    cssv = cumsum(u)
    rho = findlast(i -> u[i] + (1 - cssv[i]) / i > 0, 1:n)

    if rho === nothing
        rho = 1
    end

    # Compute threshold
    theta = (cssv[rho] - 1) / rho

    # Project
    w .= max.(w .- theta, zero(T))

    return w
end

"""
    compute_pre_treatment_fit(treated_pre, control_pre, weights)

Compute pre-treatment fit statistics.

# Returns
- `rmse::T`: Root mean squared error
- `r_squared::T`: R-squared (1 = perfect fit)
"""
function compute_pre_treatment_fit(
    treated_pre::Vector{T},
    control_pre::Matrix{T},
    weights::Vector{T},
) where {T<:Real}
    synthetic = control_pre' * weights
    residuals = treated_pre .- synthetic

    ss_res = sum(residuals .^ 2)
    ss_tot = sum((treated_pre .- mean(treated_pre)) .^ 2)

    rmse = sqrt(ss_res / length(treated_pre))
    r_squared = ss_tot > 1e-10 ? one(T) - ss_res / ss_tot : zero(T)

    return rmse, r_squared
end
