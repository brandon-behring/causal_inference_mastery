"""
Regression Discontinuity Design (RDD) type definitions.

Following SciML Problem-Estimator-Solution pattern for causal inference.
"""

# =============================================================================
# Abstract Types
# =============================================================================

"""
    AbstractRDDProblem{T,P}

Abstract type for RDD problem specifications.

Type parameters match AbstractCausalProblem:
- `T`: Outcome type (Float64, Float32, etc.)
- `P`: Parameter type (NamedTuple)
"""
abstract type AbstractRDDProblem{T,P} <: AbstractCausalProblem{T,P} end

"""
    AbstractRDDEstimator

Abstract type for RDD estimators.
"""
abstract type AbstractRDDEstimator <: AbstractCausalEstimator end

"""
    AbstractRDDSolution

Abstract type for RDD estimation results.
"""
abstract type AbstractRDDSolution <: AbstractCausalSolution end

# =============================================================================
# Problem Type
# =============================================================================

"""
    RDDProblem{T<:Real, P<:NamedTuple} <: AbstractRDDProblem{T,P}

Regression Discontinuity Design problem specification.

RDD exploits a discontinuous assignment rule to estimate causal effects.
Treatment is assigned based on whether a running variable (forcing variable)
crosses a known cutoff threshold.

# Fields
- `outcomes::Vector{T}`: Outcome variable (dependent variable)
- `running_var::Vector{T}`: Running variable (forcing variable, assignment variable)
- `treatment::Vector{Bool}`: Treatment indicator (D = 1 if running_var ≥ cutoff)
- `cutoff::T`: Threshold value for treatment assignment
- `covariates::Union{Nothing, Matrix{T}}`: Baseline covariates (optional, for balance checks)
- `parameters::P`: Analysis parameters (alpha, kernel type, bandwidth method, etc.)

# Sharp vs Fuzzy RDD
- **Sharp RDD**: Treatment deterministic at cutoff (P(D=1|X=c) jumps from 0 to 1)
  - `treatment[i] == true` if and only if `running_var[i] >= cutoff`
- **Fuzzy RDD**: Treatment probabilistic at cutoff (P(D=1|X=c) jumps but not to 1)
  - Implemented in Phase 4

# Constructor Validation
- Validates `length(outcomes) == length(running_var) == length(treatment)`
- Ensures `cutoff` is within range of `running_var`
- Checks treatment assignment consistency for Sharp RDD
- Validates `covariates` dimensions if provided

# Examples
```julia
using CausalEstimators

# Sharp RDD: Class size reduction for students scoring ≥ cutoff
test_scores = [45.2, 52.1, 48.9, 55.3, 60.1, 63.5, 58.7]  # Running variable
small_class = [false, false, false, true, true, true, true]  # Treatment
achievement = [3.2, 3.5, 3.3, 4.1, 4.3, 4.5, 4.2]  # Outcome

problem = RDDProblem(
    achievement,
    test_scores,
    small_class,
    55.0,  # Cutoff score
    nothing,  # No covariates
    (alpha=0.05,)
)

# With baseline covariates (for balance testing)
baseline_gpa = [2.8, 3.0, 2.9, 3.1, 3.2, 3.3, 3.0]
baseline_age = [18.0, 19.0, 18.5, 18.0, 19.0, 18.5, 19.0]
covariates = hcat(baseline_gpa, baseline_age)

problem_with_cov = RDDProblem(
    achievement, test_scores, small_class, 55.0,
    covariates,
    (alpha=0.05,)
)
```

# Theory
The sharp RDD estimator identifies the Average Treatment Effect (ATE) at the cutoff:

```math
τ_RDD = lim_{x↓c} E[Y|X=x] - lim_{x↑c} E[Y|X=x]
```

Under continuity of potential outcomes at the cutoff, this identifies the causal effect.

# References
- Imbens, G. W., & Lemieux, T. (2008). "Regression discontinuity designs: A guide to practice." *Journal of Econometrics*, 142(2), 615-635.
- Lee, D. S., & Lemieux, T. (2010). "Regression discontinuity designs in economics." *Journal of Economic Literature*, 48(2), 281-355.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2024). *A Practical Introduction to Regression Discontinuity Designs: Extensions*. Cambridge University Press.
"""
struct RDDProblem{T<:Real,P<:NamedTuple} <: AbstractRDDProblem{T,P}
    outcomes::AbstractVector{T}
    running_var::AbstractVector{T}
    treatment::AbstractVector{<:Union{Bool,T}}  # Bool for Sharp RDD, T for Fuzzy RDD
    cutoff::T
    covariates::Union{Nothing,AbstractMatrix{T}}
    parameters::P

    function RDDProblem(
        outcomes::AbstractVector{T},
        running_var::AbstractVector{T},
        treatment::AbstractVector{<:Union{Bool,T}},  # Accept Bool or numeric
        cutoff::T,
        covariates::Union{Nothing,AbstractMatrix{T}},
        parameters::P
    ) where {T<:Real,P<:NamedTuple}
        n = length(outcomes)

        # Validate dimensions
        if length(running_var) != n
            throw(ArgumentError("running_var must have same length as outcomes (got $(length(running_var)), expected $n)"))
        end
        if length(treatment) != n
            throw(ArgumentError("treatment must have same length as outcomes (got $(length(treatment)), expected $n)"))
        end

        # Validate covariates dimensions
        if !isnothing(covariates)
            if size(covariates, 1) != n
                throw(ArgumentError("covariates must have n rows (got $(size(covariates, 1)), expected $n)"))
            end
        end

        # Validate cutoff is within data range
        min_x, max_x = extrema(running_var)
        if cutoff < min_x || cutoff > max_x
            throw(ArgumentError("cutoff ($cutoff) must be within range of running_var [$min_x, $max_x]"))
        end

        # Check for observations on both sides of cutoff
        n_below = sum(running_var .< cutoff)
        n_above = sum(running_var .>= cutoff)
        if n_below == 0
            throw(ArgumentError("No observations below cutoff - cannot estimate RDD (need observations on both sides)"))
        end
        if n_above == 0
            throw(ArgumentError("No observations above cutoff - cannot estimate RDD (need observations on both sides)"))
        end

        # For Sharp RDD: validate treatment assignment consistency
        # Treatment should be 0 if X < c, 1 if X >= c
        # Skip this check for Fuzzy RDD (when treatment is numeric, not Bool)
        if eltype(treatment) <: Bool
            expected_treatment = running_var .>= cutoff
            if !all(treatment .== expected_treatment)
                n_inconsistent = sum(treatment .!= expected_treatment)
                @warn "Treatment assignment inconsistent with Sharp RDD at cutoff ($(n_inconsistent) violations). " *
                      "Expected treatment = 1 if running_var >= cutoff. " *
                      "For Fuzzy RDD, pass numeric treatment values (0.0/1.0)."
            end
        end

        new{T,P}(outcomes, running_var, treatment, cutoff, covariates, parameters)
    end
end

"""
    McCraryTest

McCrary (2008) density discontinuity test results.

Tests H₀: f(c⁺) = f(c⁻) (no manipulation at cutoff) where f is the density
of the running variable.

# Fields
- `p_value::Float64`: P-value for density discontinuity test
- `discontinuity_estimate::Float64`: Log density difference: log(f(c⁺)/f(c⁻))
- `se::Float64`: Standard error of discontinuity estimate
- `passes::Bool`: Test passes if p_value > α (no evidence of manipulation)

# Interpretation
- `passes == true`: No evidence of manipulation (good for RDD validity)
- `passes == false`: Evidence of manipulation (RDD identification questionable)
- Manipulation means units strategically sort across cutoff, violating continuity

# References
- McCrary, J. (2008). "Manipulation of the running variable in the regression discontinuity design." *Journal of Econometrics*, 142(2), 698-714.
- Cattaneo, M. D., Jansson, M., & Ma, X. (2020). "Simple local polynomial density estimators." *Journal of the American Statistical Association*, 115(531), 1449-1455.
"""
struct McCraryTest
    p_value::Float64
    discontinuity_estimate::Float64
    se::Float64
    passes::Bool

    function McCraryTest(p_value::Float64, discontinuity_estimate::Float64,
                         se::Float64, alpha::Float64=0.05)
        new(p_value, discontinuity_estimate, se, p_value > alpha)
    end
end

"""
    RDDSolution{T<:Real} <: AbstractRDDSolution

Solution from RDD estimator.

# Fields
- `estimate::T`: Point estimate of treatment effect at cutoff
- `se::T`: Standard error
- `ci_lower::T`: Lower confidence interval bound
- `ci_upper::T`: Upper confidence interval bound
- `p_value::T`: P-value for H₀: τ = 0
- `bandwidth::T`: Bandwidth used for local polynomial regression
- `bandwidth_bias::Union{Nothing, T}`: Bias-correction bandwidth (CCT only)
- `kernel::Symbol`: Kernel function used (:triangular, :uniform, :epanechnikov)
- `n_eff_left::Int`: Effective sample size left of cutoff (within bandwidth)
- `n_eff_right::Int`: Effective sample size right of cutoff
- `density_test::Union{Nothing, McCraryTest}`: McCrary density test results (if run)
- `bias_corrected::Bool`: Whether estimate uses bias correction (CCT)
- `retcode::Symbol`: Return code (:Success, :Warning, :Failure)

# Examples
```julia
# After solving RDD problem
solution = solve(problem, SharpRDD())

println("Treatment effect: \$(solution.estimate) ± \$(solution.se)")
println("95% CI: [\$(solution.ci_lower), \$(solution.ci_upper)]")
println("P-value: \$(solution.p_value)")
println("Bandwidth: \$(solution.bandwidth)")
println("McCrary test passed: \$(solution.density_test.passes)")
```

# References
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). "Robust nonparametric confidence intervals for regression-discontinuity designs." *Econometrica*, 82(6), 2295-2326.
"""
struct RDDSolution{T<:Real} <: AbstractRDDSolution
    estimate::T
    se::T
    ci_lower::T
    ci_upper::T
    p_value::T
    bandwidth::T
    bandwidth_bias::Union{Nothing,T}
    kernel::Symbol
    n_eff_left::Int
    n_eff_right::Int
    density_test::Union{Nothing,McCraryTest}
    bias_corrected::Bool
    retcode::Symbol

    function RDDSolution(;
        estimate::T,
        se::T,
        ci_lower::T,
        ci_upper::T,
        p_value::T,
        bandwidth::T,
        bandwidth_bias::Union{Nothing,T}=nothing,
        kernel::Symbol=:triangular,
        n_eff_left::Int,
        n_eff_right::Int,
        density_test::Union{Nothing,McCraryTest}=nothing,
        bias_corrected::Bool=false,
        retcode::Symbol=:Success
    ) where {T<:Real}
        new{T}(estimate, se, ci_lower, ci_upper, p_value, bandwidth, bandwidth_bias,
               kernel, n_eff_left, n_eff_right, density_test, bias_corrected, retcode)
    end
end

# =============================================================================
# Kernel Types
# =============================================================================

"""
    RDDKernel

Abstract type for RDD kernel functions.

Kernels weight observations based on distance from cutoff. Standard kernels:
- **Triangular** (default): K(u) = (1 - |u|) for |u| ≤ 1, 0 otherwise
- **Uniform**: K(u) = 0.5 for |u| ≤ 1, 0 otherwise
- **Epanechnikov**: K(u) = 0.75(1 - u²) for |u| ≤ 1, 0 otherwise

Choice of kernel has minimal impact on results (Imbens & Kalyanaraman 2012).
Triangular is IK/CCT default and recommended.
"""
abstract type RDDKernel end

"""
    TriangularKernel <: RDDKernel

Triangular kernel: K(u) = (1 - |u|) for |u| ≤ 1.

This is the default kernel for IK and CCT methods. It gives more weight to
observations near the cutoff, linearly decreasing to zero at bandwidth edges.

# Examples
```julia
kernel = TriangularKernel()
solution = solve(problem, SharpRDD(kernel=kernel))
```
"""
struct TriangularKernel <: RDDKernel end

"""
    UniformKernel <: RDDKernel

Uniform kernel: K(u) = 0.5 for |u| ≤ 1.

Equal weight to all observations within bandwidth. Rarely used in practice
as it gives equal weight to observations near and far from cutoff.
"""
struct UniformKernel <: RDDKernel end

"""
    EpanechnikovKernel <: RDDKernel

Epanechnikov kernel: K(u) = 0.75(1 - u²) for |u| ≤ 1.

Optimal in MSE sense for density estimation. Similar performance to triangular
for RDD in practice.
"""
struct EpanechnikovKernel <: RDDKernel end

"""
    kernel_function(kernel::RDDKernel, u::Real) -> Float64

Evaluate kernel function at u (normalized distance from cutoff).

# Arguments
- `kernel::RDDKernel`: Kernel type
- `u::Real`: Normalized distance: u = (X - c) / h where h is bandwidth

# Returns
- Kernel weight ∈ [0, 1] for |u| ≤ 1, 0 otherwise
"""
function kernel_function(::TriangularKernel, u::Real)
    abs(u) ≤ 1.0 ? (1.0 - abs(u)) : 0.0
end

function kernel_function(::UniformKernel, u::Real)
    abs(u) ≤ 1.0 ? 0.5 : 0.0
end

function kernel_function(::EpanechnikovKernel, u::Real)
    abs(u) ≤ 1.0 ? 0.75 * (1.0 - u^2) : 0.0
end

# =============================================================================
# Bandwidth Selector Types
# =============================================================================

"""
    AbstractBandwidthSelector

Abstract type for bandwidth selection methods in RDD.

Bandwidth h controls the local window around cutoff for estimation.
- Too small: High variance (few observations)
- Too large: High bias (non-local estimation)

Standard methods:
- **IK (Imbens-Kalyanaraman 2012)**: MSE-optimal bandwidth
- **CCT (Calonico-Cattaneo-Titiunik 2014)**: Coverage-error-optimal bandwidth
"""
abstract type AbstractBandwidthSelector end

"""
    IKBandwidth <: AbstractBandwidthSelector

Imbens-Kalyanaraman (2012) MSE-optimal bandwidth selection.

Minimizes asymptotic mean squared error of local linear estimator.

# Formula
```math
h_{IK} = C * n^{-1/5} * [σ²(c) / (f(c) * (m''(c⁺)² + m''(c⁻)²))]^{1/5}
```

where:
- σ²(c) = conditional variance at cutoff
- f(c) = density of running variable at cutoff
- m''(c±) = second derivative of regression function left/right of cutoff

# References
- Imbens, G., & Kalyanaraman, K. (2012). "Optimal bandwidth choice for the regression discontinuity estimator." *Review of Economic Studies*, 79(3), 933-959.

# Examples
```julia
solution = solve(problem, SharpRDD(bandwidth_method=IKBandwidth()))
```
"""
struct IKBandwidth <: AbstractBandwidthSelector end

"""
    CCTBandwidth <: AbstractBandwidthSelector

Calonico-Cattaneo-Titiunik (2014) coverage-error-optimal bandwidth.

Designed for bias-corrected robust inference. Produces valid confidence
intervals with correct coverage even with undersmoothing bias.

Uses two bandwidths:
- h (main): For point estimation
- b (bias): For bias correction (typically b > h)

# Advantages over IK
- Accounts for bias in coverage calculation
- Provides robust standard errors
- Better finite-sample coverage (closer to nominal 95%)

# References
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). "Robust nonparametric confidence intervals for regression-discontinuity designs." *Econometrica*, 82(6), 2295-2326.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2020). "On the effect of bias estimation on coverage accuracy in nonparametric inference." *Journal of the American Statistical Association*, 115(531), 1602-1614.

# Examples
```julia
# CCT is default
solution = solve(problem, SharpRDD())  # Uses CCT by default

# Explicit specification
solution = solve(problem, SharpRDD(bandwidth_method=CCTBandwidth()))
```
"""
struct CCTBandwidth <: AbstractBandwidthSelector end

# =============================================================================
# RDD Estimator Types
# =============================================================================

"""
    SharpRDD <: AbstractRDDEstimator

Sharp Regression Discontinuity Design estimator.

Estimates causal effect at cutoff using local polynomial regression.
Default: local linear regression with CCT bandwidth and robust inference.

# Fields
- `bandwidth_method::AbstractBandwidthSelector`: Bandwidth selection (default: CCT)
- `kernel::RDDKernel`: Kernel function (default: Triangular)
- `run_density_test::Bool`: Run McCrary test automatically (default: true)
- `polynomial_order::Int`: Local polynomial order (default: 1 = local linear)

# Local Linear Regression
Fits separate linear regressions left/right of cutoff:
```math
Y = α_L + β_L(X - c) + ε  for X < c (left)
Y = α_R + β_R(X - c) + ε  for X ≥ c (right)
```

Treatment effect: τ̂ = α_R - α_L (intercept difference at cutoff)

# Examples
```julia
using CausalEstimators

# Default: CCT bandwidth, triangular kernel, local linear, McCrary test
solution = solve(problem, SharpRDD())

# Custom: IK bandwidth, no McCrary test
solution = solve(problem, SharpRDD(
    bandwidth_method=IKBandwidth(),
    run_density_test=false
))

# Access results
println("ATE at cutoff: \$(solution.estimate) ± \$(solution.se)")
println("95% CI: [\$(solution.ci_lower), \$(solution.ci_upper)]")
println("Bandwidth: \$(solution.bandwidth)")
if !isnothing(solution.density_test)
    println("McCrary test p-value: \$(solution.density_test.p_value)")
end
```

# Interpretation
- `estimate`: Average treatment effect at cutoff (not population ATE)
- `ci_lower`, `ci_upper`: Bias-corrected robust CI (CCT 2014)
- `density_test.passes`: True if no evidence of manipulation

# References
- Hahn, J., Todd, P., & Van der Klaauw, W. (2001). "Identification and estimation of treatment effects with a regression-discontinuity design." *Econometrica*, 69(1), 201-209.
"""
Base.@kwdef struct SharpRDD <: AbstractRDDEstimator
    bandwidth_method::AbstractBandwidthSelector = CCTBandwidth()
    kernel::RDDKernel = TriangularKernel()
    run_density_test::Bool = true
    polynomial_order::Int = 1  # 1 = local linear (default), 2 = local quadratic (Phase 4)

    function SharpRDD(bandwidth_method, kernel, run_density_test, polynomial_order)
        if polynomial_order < 1
            throw(ArgumentError("polynomial_order must be ≥ 1 (got $polynomial_order)"))
        end
        if polynomial_order > 1
            @warn "polynomial_order > 1 (local quadratic) not yet implemented. Using local linear (order=1)."
            polynomial_order = 1
        end
        new(bandwidth_method, kernel, run_density_test, polynomial_order)
    end
end

# =============================================================================
# Fuzzy RDD Types
# =============================================================================

"""
    FuzzyRDD <: AbstractRDDEstimator

Fuzzy Regression Discontinuity Design estimator using 2SLS.

Fuzzy RDD occurs when treatment uptake is imperfect - crossing the cutoff
increases the probability of treatment but doesn't guarantee it.

# Method
Uses Two-Stage Least Squares (2SLS) with:
- **Instrument**: Z = 1{X ≥ c} (eligibility at cutoff)
- **Endogenous**: D (actual treatment received)
- **Local linear controls**: Separate slopes left/right of cutoff

# Estimand
The Local Average Treatment Effect (LATE) for compliers:
```math
τ_{LATE} = E[Y(1) - Y(0) | complier]
```

Compliers are units who take treatment if and only if they cross the cutoff.

# Fields
- `bandwidth_method::AbstractBandwidthSelector`: Bandwidth selection method (default: CCT)
- `kernel::RDDKernel`: Kernel function for weighting (default: Triangular)
- `run_density_test::Bool`: Run McCrary manipulation test (default: true)

# Example
```julia
using CausalEstimators

# Data: outcomes Y, running variable X, actual treatment D
# Treatment D is correlated with, but not determined by, crossing cutoff
problem = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))

# Estimate Fuzzy RDD (2SLS internally)
solution = solve(problem, FuzzyRDD())

println("LATE estimate: \$(solution.estimate)")
println("First-stage F: \$(solution.first_stage_fstat)")
println("Compliance rate: \$(solution.compliance_rate)")

if solution.weak_instrument_warning
    @warn "Weak first stage - LATE may be biased"
end
```

# Key Differences from Sharp RDD
| Aspect | Sharp RDD | Fuzzy RDD |
|--------|-----------|-----------|
| Treatment | Deterministic D=1{X≥c} | Stochastic D, P(D=1|X) jumps at c |
| Estimand | ATE at cutoff | LATE for compliers |
| Method | Local polynomial | 2SLS with local controls |
| First stage | Not needed | Critical: check F-stat |

# Warnings
- Weak first stage (F < 10): LATE estimate may be biased toward zero
- Low compliance rate: Large standard errors, imprecise estimates
- Very few compliers: Cannot identify LATE, results unreliable

# References
- Hahn, J., Todd, P., & Van der Klaauw, W. (2001). "Identification and estimation of
  treatment effects with a regression-discontinuity design." *Econometrica*, 69(1), 201-209.
- Imbens, G. W., & Angrist, J. D. (1994). "Identification and Estimation of Local
  Average Treatment Effects." *Econometrica*, 62(2), 467-475.
- Lee, D. S., & Lemieux, T. (2010). "Regression Discontinuity Designs in Economics."
  *Journal of Economic Literature*, 48(2), 281-355.
"""
Base.@kwdef struct FuzzyRDD <: AbstractRDDEstimator
    bandwidth_method::AbstractBandwidthSelector = CCTBandwidth()
    kernel::RDDKernel = TriangularKernel()
    run_density_test::Bool = true
end

"""
    FuzzyRDDSolution{T<:Real} <: AbstractRDDSolution

Solution from Fuzzy RDD estimation.

# Fields
- `estimate::T`: LATE estimate (treatment effect for compliers)
- `se::T`: Standard error (robust 2SLS)
- `ci_lower::T`: Lower confidence interval bound
- `ci_upper::T`: Upper confidence interval bound
- `p_value::T`: P-value for H₀: τ = 0
- `bandwidth::T`: Bandwidth used for local estimation
- `kernel::Symbol`: Kernel function used
- `n_eff_left::Int`: Effective sample size left of cutoff
- `n_eff_right::Int`: Effective sample size right of cutoff
- `first_stage_fstat::T`: First-stage F-statistic (instrument strength)
- `compliance_rate::T`: E[D|Z=1] - E[D|Z=0] (fraction of compliers)
- `weak_instrument_warning::Bool`: True if first-stage F < 10
- `density_test::Union{Nothing, McCraryTest}`: McCrary test results
- `retcode::Symbol`: Return code (:Success, :WeakInstrument, :LowCompliance)

# Diagnostics

**First-stage F-statistic**: Measures instrument strength
- F > 10: Acceptable (Stock-Yogo rule of thumb)
- F < 10: Weak instruments, LATE biased toward zero

**Compliance rate**: Fraction of units affected by cutoff
- High (>0.5): Strong instrument, precise estimates
- Low (<0.3): Weak instrument, imprecise LATE, warning issued
- Zero: No compliers, LATE not identified

# Example
```julia
solution = solve(problem, FuzzyRDD())

# Check diagnostics before trusting results
if solution.weak_instrument_warning
    println("⚠️  F-stat = \$(solution.first_stage_fstat) < 10")
end

if solution.compliance_rate < 0.3
    println("⚠️  Low compliance (\$(solution.compliance_rate))")
end
```
"""
struct FuzzyRDDSolution{T<:Real} <: AbstractRDDSolution
    estimate::T
    se::T
    ci_lower::T
    ci_upper::T
    p_value::T
    bandwidth::T
    kernel::Symbol
    n_eff_left::Int
    n_eff_right::Int
    first_stage_fstat::T
    compliance_rate::T
    weak_instrument_warning::Bool
    density_test::Union{Nothing,McCraryTest}
    retcode::Symbol

    function FuzzyRDDSolution(;
        estimate::T,
        se::T,
        ci_lower::T,
        ci_upper::T,
        p_value::T,
        bandwidth::T,
        kernel::Symbol=:triangular,
        n_eff_left::Int,
        n_eff_right::Int,
        first_stage_fstat::T,
        compliance_rate::T,
        weak_instrument_warning::Bool,
        density_test::Union{Nothing,McCraryTest}=nothing,
        retcode::Symbol=:Success
    ) where {T<:Real}
        new{T}(
            estimate, se, ci_lower, ci_upper, p_value, bandwidth, kernel,
            n_eff_left, n_eff_right, first_stage_fstat, compliance_rate,
            weak_instrument_warning, density_test, retcode
        )
    end
end

# Display methods for FuzzyRDD types
function Base.show(io::IO, solution::FuzzyRDDSolution{T}) where {T}
    println(io, "FuzzyRDDSolution{$T}")
    println(io, "  LATE estimate: $(round(solution.estimate, digits=4))")
    println(io, "  Std. Error: $(round(solution.se, digits=4))")
    println(io, "  95% CI: [$(round(solution.ci_lower, digits=4)), $(round(solution.ci_upper, digits=4))]")
    println(io, "  p-value: $(round(solution.p_value, digits=4))")
    println(io, "  Bandwidth: $(round(solution.bandwidth, digits=4))")
    println(io, "  First-stage F: $(round(solution.first_stage_fstat, digits=2))")
    println(io, "  Compliance rate: $(round(solution.compliance_rate, digits=3))")

    if solution.weak_instrument_warning
        println(io, "  ⚠️  Weak instruments detected (F < 10)")
    end

    if solution.compliance_rate < 0.3
        println(io, "  ⚠️  Low compliance rate")
    end

    if !isnothing(solution.density_test) && !solution.density_test.passes
        println(io, "  ⚠️  McCrary test suggests manipulation")
    end
end
