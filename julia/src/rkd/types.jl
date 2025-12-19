"""
Regression Kink Design (RKD) type definitions.

Following SciML Problem-Estimator-Solution pattern for causal inference.

Key Difference from RDD:
- RDD: Treatment effect = jump in LEVEL at cutoff
- RKD: Treatment effect = change in SLOPE at cutoff

References:
- Card, D., Lee, D. S., Pei, Z., & Weber, A. (2015). Inference on causal effects
    in a generalized regression kink design. Econometrica, 83(6), 2453-2483.
- Nielsen, H. S., Sorensen, T., & Taber, C. (2010). Estimating the effect of
    student aid on college enrollment. American Economic Journal: Economic Policy.
"""

# =============================================================================
# Abstract Types
# =============================================================================

"""
    AbstractRKDProblem{T,P}

Abstract type for RKD problem specifications.

Type parameters:
- `T`: Outcome type (Float64, Float32, etc.)
- `P`: Parameter type (NamedTuple)
"""
abstract type AbstractRKDProblem{T,P} <: AbstractCausalProblem{T,P} end

"""
    AbstractRKDEstimator

Abstract type for RKD estimators.
"""
abstract type AbstractRKDEstimator <: AbstractCausalEstimator end

"""
    AbstractRKDSolution

Abstract type for RKD estimation results.
"""
abstract type AbstractRKDSolution <: AbstractCausalSolution end

# =============================================================================
# Problem Type
# =============================================================================

"""
    RKDProblem{T<:Real, P<:NamedTuple} <: AbstractRKDProblem{T,P}

Regression Kink Design problem specification.

RKD exploits a kink (change in slope) in a policy function to estimate causal
effects. Unlike RDD which uses jumps in levels, RKD uses changes in the
derivative of the policy function.

# Fields
- `outcomes::Vector{T}`: Outcome variable Y
- `running_var::Vector{T}`: Running variable X
- `treatment::Vector{T}`: Treatment variable D (with kink at cutoff)
- `cutoff::T`: Kink point in the policy function
- `covariates::Union{Nothing, Matrix{T}}`: Baseline covariates (optional)
- `parameters::P`: Analysis parameters (alpha, kernel, etc.)

# Sharp vs Fuzzy RKD
- **Sharp RKD**: D is deterministic function of X with known kink
  - τ = Δslope(Y) / Δslope(D)
- **Fuzzy RKD**: E[D|X] has a kink at cutoff (stochastic)
  - Uses 2SLS approach

# Constructor Validation
- Validates array lengths match
- Ensures cutoff is within range of running_var
- Checks for observations on both sides of cutoff
- Validates finite values

# Examples
```julia
using CausalEstimators

# Sharp RKD: Policy kink at X=0
# D = 0.5*X for X < 0, D = 1.5*X for X >= 0 (kink = 1.0)
# Y = 2.0*D + noise (true effect = 2.0)
X = randn(500)
D = [x < 0 ? 0.5*x : 1.5*x for x in X]
Y = 2.0 .* D .+ randn(500)

problem = RKDProblem(Y, X, D, 0.0, nothing, (alpha=0.05,))
solution = solve(problem, SharpRKD())
```

# Theory
The Sharp RKD estimator identifies:
```math
τ_RKD = \\frac{\\lim_{x↓c} ∂E[Y|X=x]/∂x - \\lim_{x↑c} ∂E[Y|X=x]/∂x}{\\lim_{x↓c} ∂E[D|X=x]/∂x - \\lim_{x↑c} ∂E[D|X=x]/∂x}
```

Under smoothness assumptions, this identifies the causal effect of D on Y.
"""
struct RKDProblem{T<:Real,P<:NamedTuple} <: AbstractRKDProblem{T,P}
    outcomes::AbstractVector{T}
    running_var::AbstractVector{T}
    treatment::AbstractVector{T}
    cutoff::T
    covariates::Union{Nothing,AbstractMatrix{T}}
    parameters::P

    function RKDProblem(
        outcomes::AbstractVector{T},
        running_var::AbstractVector{T},
        treatment::AbstractVector{T},
        cutoff::T,
        covariates::Union{Nothing,AbstractMatrix{T}},
        parameters::P
    ) where {T<:Real,P<:NamedTuple}
        n = length(outcomes)

        # Validate dimensions
        if length(running_var) != n
            throw(ArgumentError(
                "running_var must have same length as outcomes " *
                "(got $(length(running_var)), expected $n)"
            ))
        end
        if length(treatment) != n
            throw(ArgumentError(
                "treatment must have same length as outcomes " *
                "(got $(length(treatment)), expected $n)"
            ))
        end

        # Validate covariates dimensions
        if !isnothing(covariates)
            if size(covariates, 1) != n
                throw(ArgumentError(
                    "covariates must have n rows (got $(size(covariates, 1)), expected $n)"
                ))
            end
        end

        # Check for non-finite values
        if any(!isfinite, outcomes)
            throw(ArgumentError("outcomes contains non-finite values (NaN or Inf)"))
        end
        if any(!isfinite, running_var)
            throw(ArgumentError("running_var contains non-finite values (NaN or Inf)"))
        end
        if any(!isfinite, treatment)
            throw(ArgumentError("treatment contains non-finite values (NaN or Inf)"))
        end

        # Validate cutoff is within data range
        min_x, max_x = extrema(running_var)
        if cutoff < min_x || cutoff > max_x
            throw(ArgumentError(
                "cutoff ($cutoff) must be within range of running_var [$min_x, $max_x]"
            ))
        end

        # Check for observations on both sides of cutoff
        n_below = sum(running_var .< cutoff)
        n_above = sum(running_var .>= cutoff)
        if n_below < 3
            throw(ArgumentError(
                "Insufficient observations below cutoff ($n_below < 3) - " *
                "need enough data for slope estimation"
            ))
        end
        if n_above < 3
            throw(ArgumentError(
                "Insufficient observations above cutoff ($n_above < 3) - " *
                "need enough data for slope estimation"
            ))
        end

        new{T,P}(outcomes, running_var, treatment, cutoff, covariates, parameters)
    end
end

# Convenience constructor with type conversion
function RKDProblem(
    outcomes::AbstractVector,
    running_var::AbstractVector,
    treatment::AbstractVector,
    cutoff::Real,
    covariates::Union{Nothing,AbstractMatrix}=nothing,
    parameters::NamedTuple=(alpha=0.05,)
)
    T = promote_type(eltype(outcomes), eltype(running_var), eltype(treatment), typeof(cutoff))
    T = T <: Real ? T : Float64

    outcomes_t = convert(Vector{T}, outcomes)
    running_var_t = convert(Vector{T}, running_var)
    treatment_t = convert(Vector{T}, treatment)
    cutoff_t = convert(T, cutoff)
    covariates_t = isnothing(covariates) ? nothing : convert(Matrix{T}, covariates)

    RKDProblem(outcomes_t, running_var_t, treatment_t, cutoff_t, covariates_t, parameters)
end

# =============================================================================
# Solution Type
# =============================================================================

"""
    RKDSolution{T<:Real} <: AbstractRKDSolution

Solution from RKD estimation.

# Fields
- `estimate::T`: Point estimate of kink effect
- `se::T`: Standard error (delta method)
- `ci_lower::T`: Lower confidence interval bound
- `ci_upper::T`: Upper confidence interval bound
- `t_stat::T`: T-statistic
- `p_value::T`: P-value for H₀: τ = 0
- `bandwidth::T`: Bandwidth used for estimation
- `kernel::Symbol`: Kernel function used
- `n_eff_left::Int`: Effective sample size left of cutoff
- `n_eff_right::Int`: Effective sample size right of cutoff
- `outcome_slope_left::T`: Estimated slope of Y left of kink
- `outcome_slope_right::T`: Estimated slope of Y right of kink
- `outcome_kink::T`: Change in Y slope (numerator)
- `treatment_slope_left::T`: Estimated slope of D left of kink
- `treatment_slope_right::T`: Estimated slope of D right of kink
- `treatment_kink::T`: Change in D slope (denominator)
- `polynomial_order::Int`: Polynomial order used
- `retcode::Symbol`: Return code (:Success, :Warning, :Failure)
- `message::String`: Descriptive message

# Examples
```julia
solution = solve(problem, SharpRKD())

println("Kink effect: \$(solution.estimate) ± \$(solution.se)")
println("95% CI: [\$(solution.ci_lower), \$(solution.ci_upper)]")
println("Outcome kink: \$(solution.outcome_kink)")
println("Treatment kink: \$(solution.treatment_kink)")
```
"""
struct RKDSolution{T<:Real} <: AbstractRKDSolution
    estimate::T
    se::T
    ci_lower::T
    ci_upper::T
    t_stat::T
    p_value::T
    bandwidth::T
    kernel::Symbol
    n_eff_left::Int
    n_eff_right::Int
    outcome_slope_left::T
    outcome_slope_right::T
    outcome_kink::T
    treatment_slope_left::T
    treatment_slope_right::T
    treatment_kink::T
    polynomial_order::Int
    retcode::Symbol
    message::String

    function RKDSolution(;
        estimate::T,
        se::T,
        ci_lower::T,
        ci_upper::T,
        t_stat::T,
        p_value::T,
        bandwidth::T,
        kernel::Symbol=:triangular,
        n_eff_left::Int,
        n_eff_right::Int,
        outcome_slope_left::T,
        outcome_slope_right::T,
        outcome_kink::T,
        treatment_slope_left::T,
        treatment_slope_right::T,
        treatment_kink::T,
        polynomial_order::Int=1,
        retcode::Symbol=:Success,
        message::String="Estimation completed successfully"
    ) where {T<:Real}
        new{T}(
            estimate, se, ci_lower, ci_upper, t_stat, p_value, bandwidth, kernel,
            n_eff_left, n_eff_right, outcome_slope_left, outcome_slope_right,
            outcome_kink, treatment_slope_left, treatment_slope_right, treatment_kink,
            polynomial_order, retcode, message
        )
    end
end

# Display method
function Base.show(io::IO, solution::RKDSolution{T}) where {T}
    println(io, "RKDSolution{$T}")
    println(io, "  Estimate: $(round(solution.estimate, digits=4))")
    println(io, "  Std. Error: $(round(solution.se, digits=4))")
    println(io, "  95% CI: [$(round(solution.ci_lower, digits=4)), $(round(solution.ci_upper, digits=4))]")
    println(io, "  p-value: $(round(solution.p_value, digits=4))")
    println(io, "  Bandwidth: $(round(solution.bandwidth, digits=4))")
    println(io, "  Kernel: $(solution.kernel)")
    println(io, "  N (left/right): $(solution.n_eff_left)/$(solution.n_eff_right)")
    println(io, "  Outcome kink (ΔY): $(round(solution.outcome_kink, digits=4))")
    println(io, "  Treatment kink (ΔD): $(round(solution.treatment_kink, digits=4))")
    println(io, "  Status: $(solution.retcode)")
end

# =============================================================================
# Kernel Types (reuse RDD kernels or define RKD-specific)
# =============================================================================

"""
    RKDKernel

Abstract type for RKD kernel functions.

RKD uses the same kernels as RDD:
- **Triangular** (default): K(u) = (1 - |u|) for |u| ≤ 1
- **Uniform**: K(u) = 0.5 for |u| ≤ 1
- **Epanechnikov**: K(u) = 0.75(1 - u²) for |u| ≤ 1
"""
abstract type RKDKernel end

struct TriangularRKDKernel <: RKDKernel end
struct UniformRKDKernel <: RKDKernel end
struct EpanechnikovRKDKernel <: RKDKernel end

"""
    rkd_kernel_function(kernel::RKDKernel, u::Real) -> Float64

Evaluate RKD kernel at normalized distance u.
"""
function rkd_kernel_function(::TriangularRKDKernel, u::Real)
    abs(u) <= 1.0 ? (1.0 - abs(u)) : 0.0
end

function rkd_kernel_function(::UniformRKDKernel, u::Real)
    abs(u) <= 1.0 ? 0.5 : 0.0
end

function rkd_kernel_function(::EpanechnikovRKDKernel, u::Real)
    abs(u) <= 1.0 ? 0.75 * (1.0 - u^2) : 0.0
end

# Convert symbol to kernel
function get_rkd_kernel(kernel::Symbol)
    if kernel == :triangular
        return TriangularRKDKernel()
    elseif kernel == :uniform || kernel == :rectangular
        return UniformRKDKernel()
    elseif kernel == :epanechnikov
        return EpanechnikovRKDKernel()
    else
        throw(ArgumentError("Unknown kernel: $kernel. Use :triangular, :uniform, or :epanechnikov"))
    end
end

# =============================================================================
# Estimator Types
# =============================================================================

"""
    SharpRKD <: AbstractRKDEstimator

Sharp Regression Kink Design estimator.

Estimates causal effect at kink using local polynomial regression.
The key insight is that RKD estimates changes in slopes, not levels.

# Fields
- `bandwidth::Union{Nothing, Float64}`: Bandwidth for estimation (nothing = auto)
- `kernel::Symbol`: Kernel function (:triangular, :uniform, :epanechnikov)
- `polynomial_order::Int`: Local polynomial order (1, 2, or 3)
- `alpha::Float64`: Significance level for CI

# Estimation Method
Fits local polynomial regressions on each side of the cutoff:
```math
Y = α_L + β_L(X - c) + ... + ε  for X < c
Y = α_R + β_R(X - c) + ... + ε  for X ≥ c
```

Kink effect: τ = (β_R^Y - β_L^Y) / (β_R^D - β_L^D)

Standard error via delta method.

# Bandwidth Selection
RKD requires wider bandwidths than RDD because estimating derivatives
requires more data. The optimal rate is n^{-1/9} vs RDD's n^{-1/5}.

# Examples
```julia
# Default: auto bandwidth, triangular kernel, linear polynomial
solution = solve(problem, SharpRKD())

# Custom: specified bandwidth, quadratic polynomial
solution = solve(problem, SharpRKD(bandwidth=2.0, polynomial_order=2))
```

# References
- Card et al. (2015) - Generalized RKD
- Nielsen et al. (2010) - Kink identification
"""
Base.@kwdef struct SharpRKD <: AbstractRKDEstimator
    bandwidth::Union{Nothing,Float64} = nothing
    kernel::Symbol = :triangular
    polynomial_order::Int = 1
    alpha::Float64 = 0.05

    function SharpRKD(bandwidth, kernel, polynomial_order, alpha)
        if !isnothing(bandwidth) && bandwidth <= 0
            throw(ArgumentError("bandwidth must be positive (got $bandwidth)"))
        end
        if !(kernel in [:triangular, :uniform, :rectangular, :epanechnikov])
            throw(ArgumentError(
                "kernel must be :triangular, :uniform, or :epanechnikov (got $kernel)"
            ))
        end
        if !(polynomial_order in [1, 2, 3])
            throw(ArgumentError("polynomial_order must be 1, 2, or 3 (got $polynomial_order)"))
        end
        if !(0 < alpha < 1)
            throw(ArgumentError("alpha must be in (0, 1) (got $alpha)"))
        end
        new(bandwidth, kernel, polynomial_order, alpha)
    end
end

"""
    FuzzyRKD <: AbstractRKDEstimator

Fuzzy Regression Kink Design estimator using 2SLS.

Fuzzy RKD occurs when E[D|X] has a kink at cutoff but D is stochastic
(not a deterministic function of X).

# Method
Uses Two-Stage Least Squares:
- First stage: Estimate kink in E[D|X]
- Reduced form: Estimate kink in E[Y|X]
- Effect: τ = reduced_form_kink / first_stage_kink

# Estimand
Local Average Treatment Effect (LATE) for compliers.

# Fields
- `bandwidth::Union{Nothing, Float64}`: Bandwidth (nothing = auto)
- `kernel::Symbol`: Kernel function
- `polynomial_order::Int`: Local polynomial order
- `alpha::Float64`: Significance level

# Example
```julia
# Fuzzy kink data: D has stochastic kink at cutoff
solution = solve(problem, FuzzyRKD())

println("LATE: \$(solution.estimate)")
println("First stage kink: \$(solution.first_stage_kink)")
println("First stage F: \$(solution.first_stage_f_stat)")
```
"""
Base.@kwdef struct FuzzyRKD <: AbstractRKDEstimator
    bandwidth::Union{Nothing,Float64} = nothing
    kernel::Symbol = :triangular
    polynomial_order::Int = 1
    alpha::Float64 = 0.05

    function FuzzyRKD(bandwidth, kernel, polynomial_order, alpha)
        if !isnothing(bandwidth) && bandwidth <= 0
            throw(ArgumentError("bandwidth must be positive"))
        end
        if !(kernel in [:triangular, :uniform, :rectangular, :epanechnikov])
            throw(ArgumentError("kernel must be :triangular, :uniform, or :epanechnikov"))
        end
        if !(polynomial_order in [1, 2, 3])
            throw(ArgumentError("polynomial_order must be 1, 2, or 3"))
        end
        if !(0 < alpha < 1)
            throw(ArgumentError("alpha must be in (0, 1)"))
        end
        new(bandwidth, kernel, polynomial_order, alpha)
    end
end

# =============================================================================
# Fuzzy RKD Solution Type
# =============================================================================

"""
    FuzzyRKDSolution{T<:Real} <: AbstractRKDSolution

Solution from Fuzzy RKD estimation (2SLS).

# Additional Fields (vs RKDSolution)
- `first_stage_slope_left::T`: First stage slope left of kink
- `first_stage_slope_right::T`: First stage slope right of kink
- `first_stage_kink::T`: Change in E[D|X] slope
- `reduced_form_slope_left::T`: Reduced form slope left
- `reduced_form_slope_right::T`: Reduced form slope right
- `reduced_form_kink::T`: Change in E[Y|X] slope
- `first_stage_f_stat::T`: F-statistic for first stage strength
- `weak_first_stage::Bool`: True if F < 10
"""
struct FuzzyRKDSolution{T<:Real} <: AbstractRKDSolution
    estimate::T
    se::T
    ci_lower::T
    ci_upper::T
    t_stat::T
    p_value::T
    bandwidth::T
    kernel::Symbol
    n_eff_left::Int
    n_eff_right::Int
    first_stage_slope_left::T
    first_stage_slope_right::T
    first_stage_kink::T
    reduced_form_slope_left::T
    reduced_form_slope_right::T
    reduced_form_kink::T
    first_stage_f_stat::T
    weak_first_stage::Bool
    polynomial_order::Int
    retcode::Symbol
    message::String

    function FuzzyRKDSolution(;
        estimate::T,
        se::T,
        ci_lower::T,
        ci_upper::T,
        t_stat::T,
        p_value::T,
        bandwidth::T,
        kernel::Symbol=:triangular,
        n_eff_left::Int,
        n_eff_right::Int,
        first_stage_slope_left::T,
        first_stage_slope_right::T,
        first_stage_kink::T,
        reduced_form_slope_left::T,
        reduced_form_slope_right::T,
        reduced_form_kink::T,
        first_stage_f_stat::T,
        weak_first_stage::Bool,
        polynomial_order::Int=1,
        retcode::Symbol=:Success,
        message::String="Estimation completed successfully"
    ) where {T<:Real}
        new{T}(
            estimate, se, ci_lower, ci_upper, t_stat, p_value, bandwidth, kernel,
            n_eff_left, n_eff_right, first_stage_slope_left, first_stage_slope_right,
            first_stage_kink, reduced_form_slope_left, reduced_form_slope_right,
            reduced_form_kink, first_stage_f_stat, weak_first_stage, polynomial_order,
            retcode, message
        )
    end
end

# Display method for FuzzyRKDSolution
function Base.show(io::IO, solution::FuzzyRKDSolution{T}) where {T}
    println(io, "FuzzyRKDSolution{$T}")
    println(io, "  LATE Estimate: $(round(solution.estimate, digits=4))")
    println(io, "  Std. Error: $(round(solution.se, digits=4))")
    println(io, "  95% CI: [$(round(solution.ci_lower, digits=4)), $(round(solution.ci_upper, digits=4))]")
    println(io, "  p-value: $(round(solution.p_value, digits=4))")
    println(io, "  Bandwidth: $(round(solution.bandwidth, digits=4))")
    println(io, "  First stage kink: $(round(solution.first_stage_kink, digits=4))")
    println(io, "  Reduced form kink: $(round(solution.reduced_form_kink, digits=4))")
    println(io, "  First stage F: $(round(solution.first_stage_f_stat, digits=2))")
    if solution.weak_first_stage
        println(io, "  ⚠️  Weak first stage (F < 10)")
    end
    println(io, "  Status: $(solution.retcode)")
end
