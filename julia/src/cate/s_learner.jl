#=
S-Learner (Single Model) Implementation

The S-Learner fits a single model μ(X, T) with treatment as a feature,
then estimates CATE by comparing predictions under T=1 vs T=0.

Algorithm:
1. Augment covariates: X_aug = [X, T]
2. Fit μ(X_aug) → Y
3. CATE(x) = μ̂([x, 1]) - μ̂([x, 0])

Reference:
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects"
=#

"""
    solve(problem::CATEProblem, estimator::SLearner)::CATESolution

Estimate CATE using S-Learner (Single model approach).

# Algorithm
1. Augment covariates with treatment: X_aug = [1, X, T]
2. Fit single model: μ̂(X_aug) → Y
3. For each unit, compute: τ̂(xᵢ) = μ̂([1, xᵢ, 1]) - μ̂([1, xᵢ, 0])

# Arguments
- `problem::CATEProblem`: CATE estimation problem
- `estimator::SLearner`: S-Learner estimator with model choice

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
Y = 1.0 .+ X[:, 1] .+ 2.0 .* T .+ randn(n)

problem = CATEProblem(Y, T, X, (alpha=0.05,))
solution = solve(problem, SLearner())

println("ATE: \$(solution.ate) ± \$(solution.se)")
```
"""
function solve(
    problem::CATEProblem{T,P},
    estimator::SLearner
)::CATESolution{T,P} where {T<:Real, P<:NamedTuple}
    # Extract from problem
    (; outcomes, treatment, covariates, parameters) = problem
    alpha = get(parameters, :alpha, 0.05)

    n = length(outcomes)
    p = size(covariates, 2)

    # =========================================================================
    # Step 1: Augment covariates with treatment
    # =========================================================================

    # X_aug = [intercept, covariates, treatment]
    X_aug = hcat(ones(T, n), covariates, T.(treatment))

    # =========================================================================
    # Step 2: Fit single model
    # =========================================================================

    β, predictions, residuals = fit_model(X_aug, outcomes, estimator.model)

    # =========================================================================
    # Step 3: Compute CATE for each unit
    # =========================================================================

    # For CATE, we need μ̂([1, x, 1]) - μ̂([1, x, 0])
    # Since model is linear: CATE = β[end] (coefficient on treatment)
    # But for flexibility, compute explicitly:

    # Design matrices for treated and control predictions
    X_treat = hcat(ones(T, n), covariates, ones(T, n))  # T=1
    X_ctrl = hcat(ones(T, n), covariates, zeros(T, n))   # T=0

    pred_treat = X_treat * β
    pred_ctrl = X_ctrl * β

    cate = pred_treat - pred_ctrl

    # =========================================================================
    # Step 4: Compute ATE and SE
    # =========================================================================

    ate = mean(cate)
    se = compute_ate_se(cate, treatment, outcomes, covariates)

    # Confidence interval (t-distribution for small samples)
    df = n - (p + 2)  # degrees of freedom
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
        :s_learner,
        :Success,
        problem
    )
end
