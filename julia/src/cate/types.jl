#=
CATE (Conditional Average Treatment Effect) Types

Defines the Problem-Estimator-Solution types for heterogeneous treatment effect estimation.

References:
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects"
- Nie & Wager (2021). "Quasi-oracle estimation of heterogeneous treatment effects"
- Chernozhukov et al. (2018). "Double/debiased machine learning"
=#

# =============================================================================
# Abstract Type Hierarchy
# =============================================================================

"""
    AbstractCATEProblem{T,P} <: AbstractCausalProblem{T,P}

Abstract type for Conditional Average Treatment Effect (CATE) estimation problems.

Type parameters:
- T: Numeric type for outcomes (Float64, Float32, etc.)
- P: Parameter type (NamedTuple)
"""
abstract type AbstractCATEProblem{T,P} <: AbstractCausalProblem{T,P} end

"""
    AbstractCATEEstimator <: AbstractCausalEstimator

Abstract type for CATE estimators (meta-learners).
"""
abstract type AbstractCATEEstimator <: AbstractCausalEstimator end

"""
    AbstractCATESolution <: AbstractCausalSolution

Abstract type for CATE estimation results.
"""
abstract type AbstractCATESolution <: AbstractCausalSolution end

# =============================================================================
# Problem Type
# =============================================================================

"""
    CATEProblem{T<:Real, P<:NamedTuple} <: AbstractCATEProblem{T,P}

Problem specification for CATE estimation.

# Fields
- `outcomes::Vector{T}`: Outcome variable Y of shape (n,)
- `treatment::Vector{Bool}`: Binary treatment indicator T ∈ {0,1}
- `covariates::Matrix{T}`: Covariate matrix X of shape (n, p)
- `parameters::P`: NamedTuple with estimation parameters

# Parameters (in NamedTuple)
- `alpha::Float64`: Significance level (default: 0.05)

# Constructor Validation
- Length consistency: outcomes, treatment, covariates must match
- No NaN/Inf values in outcomes or covariates
- Treatment must have variation (some treated, some control)
- At least 2 observations

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
```
"""
struct CATEProblem{T<:Real, P<:NamedTuple} <: AbstractCATEProblem{T,P}
    outcomes::Vector{T}
    treatment::Vector{Bool}
    covariates::Matrix{T}
    parameters::P

    function CATEProblem(
        outcomes::Vector{T},
        treatment::Vector{Bool},
        covariates::Matrix{T},
        parameters::P
    ) where {T<:Real, P<:NamedTuple}
        # Validate inputs
        validate_cate_inputs(outcomes, treatment, covariates)
        new{T,P}(outcomes, treatment, covariates, parameters)
    end
end

# Convenience constructor with automatic type conversion
function CATEProblem(
    outcomes::AbstractVector,
    treatment::AbstractVector,
    covariates::AbstractMatrix,
    parameters::NamedTuple = (alpha=0.05,)
)
    T = promote_type(eltype(outcomes), eltype(covariates))
    outcomes_vec = convert(Vector{T}, outcomes)
    covariates_mat = convert(Matrix{T}, covariates)
    treatment_bool = convert(Vector{Bool}, treatment)

    CATEProblem(outcomes_vec, treatment_bool, covariates_mat, parameters)
end

"""
    validate_cate_inputs(outcomes, treatment, covariates)

Validate inputs for CATE estimation. Throws ArgumentError on invalid inputs.
"""
function validate_cate_inputs(outcomes, treatment, covariates)
    n = length(outcomes)

    # Length consistency
    if length(treatment) != n
        throw(ArgumentError(
            "CRITICAL ERROR: Length mismatch.\n" *
            "Function: validate_cate_inputs\n" *
            "outcomes: $n, treatment: $(length(treatment))"
        ))
    end

    if size(covariates, 1) != n
        throw(ArgumentError(
            "CRITICAL ERROR: Covariate rows mismatch.\n" *
            "Function: validate_cate_inputs\n" *
            "outcomes: $n, covariate rows: $(size(covariates, 1))"
        ))
    end

    # Check for NaN/Inf
    if any(isnan.(outcomes)) || any(isinf.(outcomes))
        throw(ArgumentError(
            "CRITICAL ERROR: NaN or Inf in outcomes.\n" *
            "Function: validate_cate_inputs"
        ))
    end

    if any(isnan.(covariates)) || any(isinf.(covariates))
        throw(ArgumentError(
            "CRITICAL ERROR: NaN or Inf in covariates.\n" *
            "Function: validate_cate_inputs"
        ))
    end

    # Treatment variation
    n_treated = sum(treatment)
    n_control = n - n_treated

    if n_treated == 0 || n_control == 0
        throw(ArgumentError(
            "CRITICAL ERROR: No treatment variation.\n" *
            "Function: validate_cate_inputs\n" *
            "n_treated: $n_treated, n_control: $n_control"
        ))
    end

    # Minimum sample size
    if n < 4
        throw(ArgumentError(
            "CRITICAL ERROR: Insufficient sample size.\n" *
            "Function: validate_cate_inputs\n" *
            "n: $n (need at least 4)"
        ))
    end

    return nothing
end

# =============================================================================
# Solution Type
# =============================================================================

"""
    CATESolution{T<:Real, P<:NamedTuple} <: AbstractCATESolution

Results from CATE estimation.

# Fields
- `cate::Vector{T}`: Individual treatment effects τ̂(xᵢ) for each unit
- `ate::T`: Average treatment effect (mean of CATE)
- `se::T`: Standard error of ATE
- `ci_lower::T`: Lower bound of confidence interval
- `ci_upper::T`: Upper bound of confidence interval
- `method::Symbol`: Estimation method (:s_learner, :t_learner, etc.)
- `retcode::Symbol`: Return code (:Success, :Warning)
- `original_problem::CATEProblem{T,P}`: Original problem for reproducibility

# Example
```julia
solution = solve(problem, TLearner())
println("ATE: \$(solution.ate) ± \$(solution.se)")
println("CATE range: [\$(minimum(solution.cate)), \$(maximum(solution.cate))]")
```
"""
struct CATESolution{T<:Real, P<:NamedTuple} <: AbstractCATESolution
    cate::Vector{T}
    ate::T
    se::T
    ci_lower::T
    ci_upper::T
    method::Symbol
    retcode::Symbol
    original_problem::CATEProblem{T,P}
end

# =============================================================================
# Estimator Types
# =============================================================================

"""
    SLearner <: AbstractCATEEstimator

S-Learner (Single model) for CATE estimation.

Fits a single model μ(X, T) that includes treatment as a feature,
then estimates CATE by comparing predictions under T=1 vs T=0.

# Algorithm
1. Augment covariates: X_aug = [X, T]
2. Fit μ(X_aug) → Y
3. CATE(x) = μ̂([x, 1]) - μ̂([x, 0])

# Fields
- `model::Symbol`: Base learner (:ols, :ridge)

# Pros
- Simple, uses all data
- Implicit regularization toward homogeneous effects

# Cons
- Treatment effect may be shrunk toward zero
- Less flexible for heterogeneity

# References
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects"
"""
struct SLearner <: AbstractCATEEstimator
    model::Symbol

    function SLearner(; model::Symbol = :ols)
        if model ∉ (:ols, :ridge)
            throw(ArgumentError("SLearner model must be :ols or :ridge, got $model"))
        end
        new(model)
    end
end

"""
    TLearner <: AbstractCATEEstimator

T-Learner (Two models) for CATE estimation.

Fits separate models for treated and control groups.

# Algorithm
1. Fit μ₀(X) on control group: X[T=0] → Y[T=0]
2. Fit μ₁(X) on treated group: X[T=1] → Y[T=1]
3. CATE(x) = μ̂₁(x) - μ̂₀(x)

# Fields
- `model::Symbol`: Base learner (:ols, :ridge)

# Pros
- Flexible, can capture different patterns in each group
- Intuitive interpretation

# Cons
- May have high variance with small groups
- Extrapolation issues if covariate distributions differ

# References
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects"
"""
struct TLearner <: AbstractCATEEstimator
    model::Symbol

    function TLearner(; model::Symbol = :ols)
        if model ∉ (:ols, :ridge)
            throw(ArgumentError("TLearner model must be :ols or :ridge, got $model"))
        end
        new(model)
    end
end

"""
    XLearner <: AbstractCATEEstimator

X-Learner (Cross-learner) for CATE estimation.

Uses propensity-weighted combination of imputed treatment effects.
Particularly effective when treatment groups are imbalanced.

# Algorithm
1. Fit μ₀, μ₁ (T-learner step)
2. Compute imputed effects:
   D₁ = Y₁ - μ̂₀(X₁) for treated
   D₀ = μ̂₁(X₀) - Y₀ for control
3. Fit τ₁(X) → D₁, τ₀(X) → D₀
4. CATE(x) = ê(x)·τ₀(x) + (1-ê(x))·τ₁(x)

# Fields
- `model::Symbol`: Base learner (:ols, :ridge)

# Pros
- Handles imbalanced treatment groups well
- Uses propensity for adaptive weighting

# Cons
- More complex than S/T-learners
- Requires propensity estimation

# References
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects"
"""
struct XLearner <: AbstractCATEEstimator
    model::Symbol

    function XLearner(; model::Symbol = :ols)
        if model ∉ (:ols, :ridge)
            throw(ArgumentError("XLearner model must be :ols or :ridge, got $model"))
        end
        new(model)
    end
end

"""
    RLearner <: AbstractCATEEstimator

R-Learner (Robinson transformation) for CATE estimation.

Uses residualization to achieve double robustness.

# Algorithm
1. Fit ê(X) = P(T=1|X) [propensity]
2. Fit m̂(X) = E[Y|X] [outcome]
3. Compute residuals: Ỹ = Y - m̂(X), T̃ = T - ê(X)
4. θ̂ = Σ(Ỹ·T̃) / Σ(T̃²)

# Fields
- `model::Symbol`: Base learner (:ols, :ridge)

# Pros
- Doubly robust (consistent if either nuisance model correct)
- Orthogonal to nuisance estimation errors

# Cons
- Requires good nuisance model estimates
- May have higher variance than simpler methods

# References
- Nie & Wager (2021). "Quasi-oracle estimation of heterogeneous treatment effects"
- Robinson (1988). "Root-N-consistent semiparametric regression"
"""
struct RLearner <: AbstractCATEEstimator
    model::Symbol

    function RLearner(; model::Symbol = :ols)
        if model ∉ (:ols, :ridge)
            throw(ArgumentError("RLearner model must be :ols or :ridge, got $model"))
        end
        new(model)
    end
end

"""
    DoubleMachineLearning <: AbstractCATEEstimator

Double Machine Learning with K-fold cross-fitting.

Eliminates regularization bias by using out-of-sample predictions
for nuisance parameters.

# Algorithm
1. Split data into K folds
2. For each fold k:
   - Train ê, m̂ on OTHER folds
   - Predict on fold k (out-of-sample)
3. Use cross-fitted residuals for R-learner estimation

# Fields
- `n_folds::Int`: Number of cross-fitting folds (default: 5)
- `model::Symbol`: Base learner (:ols, :ridge)

# Pros
- Eliminates regularization bias
- Valid asymptotic inference with ML nuisance models

# Cons
- Computationally more expensive (K× fitting)
- Requires sufficient sample size for cross-fitting

# References
- Chernozhukov et al. (2018). "Double/debiased machine learning"
"""
struct DoubleMachineLearning <: AbstractCATEEstimator
    n_folds::Int
    model::Symbol

    function DoubleMachineLearning(; n_folds::Int = 5, model::Symbol = :ols)
        if n_folds < 2
            throw(ArgumentError("n_folds must be ≥ 2, got $n_folds"))
        end
        if model ∉ (:ols, :ridge)
            throw(ArgumentError("DoubleMachineLearning model must be :ols or :ridge, got $model"))
        end
        new(n_folds, model)
    end
end
