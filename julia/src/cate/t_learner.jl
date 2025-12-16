#=
T-Learner (Two Models) Implementation

The T-Learner fits separate models for treated and control groups.

Algorithm:
1. Fit μ₀(X) on control group: X[T=0] → Y[T=0]
2. Fit μ₁(X) on treated group: X[T=1] → Y[T=1]
3. CATE(x) = μ̂₁(x) - μ̂₀(x)

Reference:
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects"
=#

"""
    solve(problem::CATEProblem, estimator::TLearner)::CATESolution

Estimate CATE using T-Learner (Two models approach).

# Algorithm
1. Split data by treatment status
2. Fit μ₀(X) on control group: X[T=0] → Y[T=0]
3. Fit μ₁(X) on treated group: X[T=1] → Y[T=1]
4. For all units: τ̂(xᵢ) = μ̂₁(xᵢ) - μ̂₀(xᵢ)

# Arguments
- `problem::CATEProblem`: CATE estimation problem
- `estimator::TLearner`: T-Learner estimator with model choice

# Returns
- `CATESolution`: Results including individual CATE, ATE, SE, and CI

# Example
```julia
using CausalEstimators
using Random

Random.seed!(42)
n = 200
X = randn(n, 3)
T = rand(n) .> 0.5
# Heterogeneous effect: τ(x) = 2 + x₁
tau_true = 2.0 .+ X[:, 1]
Y = 1.0 .+ X[:, 1] .+ tau_true .* T .+ randn(n)

problem = CATEProblem(Y, T, X, (alpha=0.05,))
solution = solve(problem, TLearner())

correlation = cor(solution.cate, tau_true)
println("CATE correlation with truth: \$correlation")
```
"""
function solve(
    problem::CATEProblem{T,P},
    estimator::TLearner
)::CATESolution{T,P} where {T<:Real, P<:NamedTuple}
    # Extract from problem
    (; outcomes, treatment, covariates, parameters) = problem
    alpha = get(parameters, :alpha, 0.05)

    n = length(outcomes)
    p = size(covariates, 2)

    # =========================================================================
    # Step 1: Split data by treatment
    # =========================================================================

    treated_idx = treatment
    control_idx = .!treatment

    Y_1 = outcomes[treated_idx]
    Y_0 = outcomes[control_idx]

    X_1 = covariates[treated_idx, :]
    X_0 = covariates[control_idx, :]

    n_1 = length(Y_1)
    n_0 = length(Y_0)

    # =========================================================================
    # Step 2: Fit separate models
    # =========================================================================

    # Add intercept to design matrices
    X_1_int = add_intercept(X_1)
    X_0_int = add_intercept(X_0)

    β_1, _, _ = fit_model(X_1_int, Y_1, estimator.model)
    β_0, _, _ = fit_model(X_0_int, Y_0, estimator.model)

    # =========================================================================
    # Step 3: Predict for all units
    # =========================================================================

    # Full design matrix with intercept
    X_all_int = add_intercept(covariates)

    μ_1 = predict_ols(X_all_int, β_1)  # μ̂₁(xᵢ) for all i
    μ_0 = predict_ols(X_all_int, β_0)  # μ̂₀(xᵢ) for all i

    # CATE for all units
    cate = μ_1 - μ_0

    # =========================================================================
    # Step 4: Compute ATE and SE
    # =========================================================================

    ate = mean(cate)
    se = compute_ate_se(cate, treatment, outcomes, covariates)

    # Confidence interval
    df = min(n_0, n_1) - (p + 1)
    if df < 1
        df = 1
    end
    t_crit = quantile(TDist(df), 1 - alpha / 2)
    ci_lower = ate - t_crit * se
    ci_upper = ate + t_crit * se

    return CATESolution{T,P}(
        cate,
        ate,
        se,
        ci_lower,
        ci_upper,
        :t_learner,
        :Success,
        problem
    )
end
