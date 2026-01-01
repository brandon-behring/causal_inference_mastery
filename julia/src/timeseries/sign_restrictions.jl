"""
Sign Restrictions for SVAR Identification (Uhlig 2005)

Session 162: Set-identified SVAR using sign restrictions on impulse responses.

Unlike point-identified methods (Cholesky, Long-Run), sign restrictions yield
a **set of valid structural matrices**. The algorithm samples random rotations
of the Cholesky factor and keeps those satisfying the sign constraints.

Algorithm (Uhlig 2005):
1. Start from reduced-form VAR: Σ_u = PP' (Cholesky decomposition)
2. For i = 1 to N_draws:
   a. Generate random orthogonal Q via Givens rotations
   b. Candidate impact matrix: B₀⁻¹ = P·Q
   c. Compute IRF for candidate
   d. If IRF satisfies all sign constraints → Accept
3. Report identified set bounds (percentiles across accepted draws)

References
----------
Uhlig, H. (2005). "What Are the Effects of Monetary Policy on Output?
Results from an Agnostic Identification Procedure." Journal of Monetary
Economics 52(2): 381-419.
"""
module SignRestrictions

using LinearAlgebra
using Statistics
using Random

using ..TimeSeriesTypes: VARResult
using ..SVARTypes: SVARResult, IRFResult, IdentificationMethod, SIGN, CHOLESKY
using ..SVAR: cholesky_svar, vma_coefficients

export SignRestrictionConstraint, SignRestrictionResult
export sign_restriction_svar
export create_monetary_policy_constraints, check_cholesky_in_set
export givens_rotation_matrix, random_orthogonal_givens, random_orthogonal_qr
export check_sign_constraints, validate_constraints, compute_irf_from_impact


"""
    SignRestrictionConstraint

Single sign constraint on an impulse response.

Specifies that the response of variable `response_idx` to a unit shock
in variable `shock_idx` at horizon `horizon` must be positive or negative.

# Fields
- `shock_idx::Int`: Index of the structural shock (1-indexed)
- `response_idx::Int`: Index of the response variable (1-indexed)
- `horizon::Int`: Horizon at which constraint applies (0 = impact)
- `sign::Int`: Required sign: +1 for positive, -1 for negative

# Example
```julia
# Money shock must increase output at impact
constraint = SignRestrictionConstraint(1, 2, 0, 1)
```
"""
struct SignRestrictionConstraint
    shock_idx::Int
    response_idx::Int
    horizon::Int
    sign::Int

    function SignRestrictionConstraint(shock_idx::Int, response_idx::Int, horizon::Int, sign::Int)
        if sign != 1 && sign != -1
            error("sign must be 1 or -1, got $sign")
        end
        if horizon < 0
            error("horizon must be >= 0, got $horizon")
        end
        if shock_idx < 1
            error("shock_idx must be >= 1, got $shock_idx")
        end
        if response_idx < 1
            error("response_idx must be >= 1, got $response_idx")
        end
        new(shock_idx, response_idx, horizon, sign)
    end
end

function Base.show(io::IO, c::SignRestrictionConstraint)
    sign_str = c.sign > 0 ? "+" : "-"
    print(io, "SignRestrictionConstraint(shock=$(c.shock_idx), response=$(c.response_idx), h=$(c.horizon), sign=$sign_str)")
end


"""
    SignRestrictionResult

Result from sign-restricted SVAR (set-identified).

Unlike point-identified SVARs, sign restrictions yield a set of valid
structural matrices. This result contains the full set and summary statistics.

# Fields
- `var_result::VARResult`: Underlying reduced-form VAR
- `B0_inv::Matrix{Float64}`: Median impact matrix from accepted draws
- `B0::Matrix{Float64}`: Inverse of B0_inv
- `structural_shocks::Matrix{Float64}`: Structural shocks using median B0
- `identification::IdentificationMethod`: Always SIGN
- `n_restrictions::Int`: Number of sign constraints imposed
- `constraints::Vector{SignRestrictionConstraint}`: Sign constraints imposed
- `B0_inv_set::Vector{Matrix{Float64}}`: All accepted impact matrices
- `irf_median::Array{Float64,3}`: Median IRF across accepted draws
- `irf_lower::Array{Float64,3}`: Lower percentile IRF
- `irf_upper::Array{Float64,3}`: Upper percentile IRF
- `n_draws::Int`: Total rotation draws attempted
- `n_accepted::Int`: Number of draws satisfying all constraints
- `acceptance_rate::Float64`: Fraction of draws accepted
- `rotation_method::String`: Method used for rotation generation
- `horizons::Int`: Maximum IRF horizon computed
"""
struct SignRestrictionResult
    var_result::VARResult
    B0_inv::Matrix{Float64}
    B0::Matrix{Float64}
    structural_shocks::Matrix{Float64}
    identification::IdentificationMethod
    n_restrictions::Int
    constraints::Vector{SignRestrictionConstraint}
    B0_inv_set::Vector{Matrix{Float64}}
    irf_median::Array{Float64,3}
    irf_lower::Array{Float64,3}
    irf_upper::Array{Float64,3}
    n_draws::Int
    n_accepted::Int
    acceptance_rate::Float64
    rotation_method::String
    horizons::Int
end

function SignRestrictionResult(;
    var_result::VARResult,
    B0_inv::Matrix{Float64},
    B0::Matrix{Float64},
    structural_shocks::Matrix{Float64},
    identification::IdentificationMethod=SIGN,
    n_restrictions::Int=0,
    constraints::Vector{SignRestrictionConstraint}=SignRestrictionConstraint[],
    B0_inv_set::Vector{Matrix{Float64}}=Matrix{Float64}[],
    irf_median::Array{Float64,3},
    irf_lower::Array{Float64,3},
    irf_upper::Array{Float64,3},
    n_draws::Int=0,
    n_accepted::Int=0,
    acceptance_rate::Float64=0.0,
    rotation_method::String="givens",
    horizons::Int=20
)
    SignRestrictionResult(
        var_result, B0_inv, B0, structural_shocks,
        identification, n_restrictions, constraints, B0_inv_set,
        irf_median, irf_lower, irf_upper,
        n_draws, n_accepted, acceptance_rate, rotation_method, horizons
    )
end

# Accessors
n_vars(r::SignRestrictionResult) = size(r.B0_inv, 1)
lags(r::SignRestrictionResult) = r.var_result.lags
var_names(r::SignRestrictionResult) = r.var_result.var_names

"""
    get_irf_bounds(r::SignRestrictionResult, response_idx::Int, shock_idx::Int)

Get IRF bounds for a specific response-shock pair.

Returns a NamedTuple with :median, :lower, :upper, :horizon.
"""
function get_irf_bounds(r::SignRestrictionResult, response_idx::Int, shock_idx::Int)
    (
        median = r.irf_median[response_idx, shock_idx, :],
        lower = r.irf_lower[response_idx, shock_idx, :],
        upper = r.irf_upper[response_idx, shock_idx, :],
        horizon = collect(0:r.horizons)
    )
end

"""
    to_irf_result(r::SignRestrictionResult)

Convert to standard IRFResult for compatibility.
"""
function to_irf_result(r::SignRestrictionResult)
    IRFResult(
        irf=r.irf_median,
        irf_lower=r.irf_lower,
        irf_upper=r.irf_upper,
        horizons=r.horizons,
        cumulative=false,
        orthogonalized=true,
        var_names=r.var_result.var_names,
        alpha=0.32,  # 16th/84th percentiles
        n_bootstrap=r.n_accepted
    )
end

function Base.show(io::IO, r::SignRestrictionResult)
    print(io, "SignRestrictionResult(n_vars=$(n_vars(r)), n_constraints=$(length(r.constraints)), ",
          "acceptance=$(round(r.acceptance_rate * 100, digits=1))%, n_accepted=$(r.n_accepted))")
end


# =============================================================================
# Rotation Matrix Generation
# =============================================================================

"""
    givens_rotation_matrix(n::Int, i::Int, j::Int, theta::Float64)

Create Givens rotation matrix G(i,j,θ).

Rotates in the (i,j) plane by angle θ.
- G[i,i] = G[j,j] = cos(θ)
- G[i,j] = -sin(θ)
- G[j,i] = sin(θ)
- All other diagonal = 1, off-diagonal = 0

# Arguments
- `n`: Matrix dimension
- `i`, `j`: Plane indices (i < j)
- `theta`: Rotation angle in radians

# Returns
n×n Givens rotation matrix
"""
function givens_rotation_matrix(n::Int, i::Int, j::Int, theta::Float64)::Matrix{Float64}
    G = Matrix{Float64}(I, n, n)
    c = cos(theta)
    s = sin(theta)
    G[i, i] = c
    G[j, j] = c
    G[i, j] = -s
    G[j, i] = s
    return G
end


"""
    random_orthogonal_givens(n::Int, rng::AbstractRNG)

Generate random orthogonal Q ∈ SO(n) via Givens rotations.

Composes n(n-1)/2 Givens rotations with random angles uniformly
distributed on [0, 2π). This produces matrices uniformly distributed
on the special orthogonal group SO(n).

# Arguments
- `n`: Matrix dimension
- `rng`: Random number generator

# Returns
n×n orthogonal matrix with det = +1
"""
function random_orthogonal_givens(n::Int, rng::AbstractRNG)::Matrix{Float64}
    Q = Matrix{Float64}(I, n, n)
    for i in 1:n
        for j in (i+1):n
            theta = rand(rng) * 2π
            G = givens_rotation_matrix(n, i, j, theta)
            Q = Q * G
        end
    end
    return Q
end


"""
    random_orthogonal_qr(n::Int, rng::AbstractRNG)

Generate random orthogonal matrix via QR decomposition.

Faster than Givens but produces matrices with det = ±1.
We correct for det = -1 by flipping a column.

# Arguments
- `n`: Matrix dimension
- `rng`: Random number generator

# Returns
n×n orthogonal matrix with det = +1
"""
function random_orthogonal_qr(n::Int, rng::AbstractRNG)::Matrix{Float64}
    # Random matrix with standard normal entries
    A = randn(rng, n, n)

    # QR decomposition
    Q_raw, R = qr(A)
    Q = Matrix(Q_raw)

    # Ensure Q has positive diagonal in R (standard form)
    d = diag(R)
    Q = Q * Diagonal(sign.(d))

    # Ensure det(Q) = +1 (special orthogonal group)
    if det(Q) < 0
        Q[:, 1] = -Q[:, 1]
    end

    return Q
end


# =============================================================================
# Constraint Checking
# =============================================================================

"""
    check_sign_constraints(irf::Array{Float64,3}, constraints::Vector{SignRestrictionConstraint})

Check if IRF satisfies all sign constraints.

# Arguments
- `irf`: Impulse response matrix (n_vars, n_vars, horizons+1)
- `constraints`: Sign constraints to check

# Returns
`true` if ALL constraints are satisfied
"""
function check_sign_constraints(
    irf::Array{Float64,3},
    constraints::Vector{SignRestrictionConstraint}
)::Bool
    for c in constraints
        # Note: Julia 1-indexed, horizon 0 is at index 1
        val = irf[c.response_idx, c.shock_idx, c.horizon + 1]

        if c.sign > 0 && val <= 0
            return false
        end
        if c.sign < 0 && val >= 0
            return false
        end
    end
    return true
end


"""
    validate_constraints(constraints::Vector{SignRestrictionConstraint}, n_vars::Int, horizons::Int)

Validate constraint specification.

Checks that all constraints reference valid variable indices and horizons.

# Raises
Error if any constraint references invalid indices or horizons
"""
function validate_constraints(
    constraints::Vector{SignRestrictionConstraint},
    n_vars::Int,
    horizons::Int
)
    for (i, c) in enumerate(constraints)
        if c.shock_idx > n_vars
            error("Constraint $i: shock_idx $(c.shock_idx) > n_vars $n_vars")
        end
        if c.response_idx > n_vars
            error("Constraint $i: response_idx $(c.response_idx) > n_vars $n_vars")
        end
        if c.horizon > horizons
            error("Constraint $i: horizon $(c.horizon) > max horizons $horizons")
        end
    end
end


"""
    compute_irf_from_impact(Phi::Array{Float64,3}, B0_inv::Matrix{Float64}, horizons::Int)

Compute IRF from VMA coefficients and impact matrix.

IRF_h = Φ_h × B₀⁻¹

# Arguments
- `Phi`: VMA coefficients, shape (n_vars, n_vars, horizons+1)
- `B0_inv`: Impact matrix, shape (n_vars, n_vars)
- `horizons`: Maximum horizon

# Returns
IRF matrix, shape (n_vars, n_vars, horizons+1)
"""
function compute_irf_from_impact(
    Phi::Array{Float64,3},
    B0_inv::Matrix{Float64},
    horizons::Int
)::Array{Float64,3}
    n_vars = size(B0_inv, 1)
    irf = zeros(n_vars, n_vars, horizons + 1)

    for h in 0:horizons
        irf[:, :, h + 1] = Phi[:, :, h + 1] * B0_inv
    end

    return irf
end


# =============================================================================
# Main Estimation Function
# =============================================================================

"""
    sign_restriction_svar(var_result, constraints; kwargs...)

SVAR identification via sign restrictions (Uhlig 2005).

Generates random orthogonal rotations of the Cholesky factor and keeps
those satisfying all sign constraints on the implied impulse responses.

# Arguments
- `var_result::VARResult`: Estimated reduced-form VAR model
- `constraints::Vector{SignRestrictionConstraint}`: Sign constraints on impulse responses

# Keyword Arguments
- `horizons::Int=20`: Maximum IRF horizon to compute
- `n_draws::Int=5000`: Number of random rotation draws to attempt
- `rotation_method::String="givens"`: Method for generating random orthogonal matrices
  - "givens": Compose Givens rotations (default, uniform on SO(n))
  - "qr": QR decomposition of random matrix (faster, approximate)
- `percentiles::Tuple{Float64,Float64}=(16.0, 84.0)`: Lower and upper percentiles
- `seed::Union{Int,Nothing}=nothing`: Random seed for reproducibility
- `min_acceptance_rate::Float64=0.01`: Minimum acceptable acceptance rate (warning if below)

# Returns
`SignRestrictionResult`: Sign-restricted SVAR results with identified set bounds

# Raises
- Error if no valid rotations found, or constraints are invalid

# Example
```julia
var_result = var_estimate(data, lags=4)
constraints = [
    SignRestrictionConstraint(1, 2, 0, 1),  # Shock 1 increases var 2 at impact
    SignRestrictionConstraint(1, 3, 0, -1), # Shock 1 decreases var 3 at impact
]
result = sign_restriction_svar(var_result, constraints, seed=42)
println("Acceptance rate: ", round(result.acceptance_rate * 100, digits=1), "%")
```
"""
function sign_restriction_svar(
    var_result::VARResult,
    constraints::Vector{SignRestrictionConstraint};
    horizons::Int=20,
    n_draws::Int=5000,
    rotation_method::String="givens",
    percentiles::Tuple{Float64,Float64}=(16.0, 84.0),
    seed::Union{Int,Nothing}=nothing,
    min_acceptance_rate::Float64=0.01
)::SignRestrictionResult

    n_vars_local = length(var_result.var_names)

    # Validate constraints
    validate_constraints(constraints, n_vars_local, horizons)

    if isempty(constraints)
        @warn "No constraints provided - all rotations will be accepted"
    end

    # Initialize random generator
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)

    # Get Cholesky starting point
    svar_chol = cholesky_svar(var_result)
    P = svar_chol.B0_inv  # Cholesky factor

    # Compute VMA coefficients for IRF calculation
    Phi = vma_coefficients(var_result, horizons)

    # Storage for accepted draws
    accepted_B0_inv = Matrix{Float64}[]
    accepted_irfs = Array{Float64,3}[]

    # Validate rotation method
    if rotation_method != "givens" && rotation_method != "qr"
        error("rotation_method must be 'givens' or 'qr', got '$rotation_method'")
    end

    # Main rotation loop
    for draw_idx in 1:n_draws
        # Generate random orthogonal matrix
        if rotation_method == "givens"
            Q = random_orthogonal_givens(n_vars_local, rng)
        else  # qr
            Q = random_orthogonal_qr(n_vars_local, rng)
        end

        # Candidate impact matrix: B₀⁻¹ = P × Q
        B0_inv_candidate = P * Q

        # Compute IRF for this candidate
        irf = compute_irf_from_impact(Phi, B0_inv_candidate, horizons)

        # Check sign constraints
        if check_sign_constraints(irf, constraints)
            push!(accepted_B0_inv, B0_inv_candidate)
            push!(accepted_irfs, irf)
        end
    end

    n_accepted = length(accepted_B0_inv)
    acceptance_rate = n_accepted / n_draws

    # Check acceptance
    if n_accepted == 0
        error("No rotations satisfied sign constraints (0/$n_draws). " *
              "Check if constraints are feasible or increase n_draws.")
    end

    if acceptance_rate < min_acceptance_rate
        @warn "Very low acceptance rate ($(round(acceptance_rate * 100, digits=2))%). " *
              "Constraints may be too restrictive or conflicting."
    end

    # Stack accepted IRFs for percentile computation
    # Shape: (n_accepted, n_vars, n_vars, horizons+1)
    irfs_array = zeros(n_accepted, n_vars_local, n_vars_local, horizons + 1)
    for (i, irf) in enumerate(accepted_irfs)
        irfs_array[i, :, :, :] = irf
    end

    # Compute percentile bounds
    lower_pct, upper_pct = percentiles
    irf_median = zeros(n_vars_local, n_vars_local, horizons + 1)
    irf_lower = zeros(n_vars_local, n_vars_local, horizons + 1)
    irf_upper = zeros(n_vars_local, n_vars_local, horizons + 1)

    for i in 1:n_vars_local
        for j in 1:n_vars_local
            for h in 1:(horizons + 1)
                vals = irfs_array[:, i, j, h]
                irf_median[i, j, h] = median(vals)
                irf_lower[i, j, h] = quantile(vals, lower_pct / 100)
                irf_upper[i, j, h] = quantile(vals, upper_pct / 100)
            end
        end
    end

    # Use median B0_inv as point estimate
    B0_inv_stacked = zeros(n_accepted, n_vars_local, n_vars_local)
    for (i, B) in enumerate(accepted_B0_inv)
        B0_inv_stacked[i, :, :] = B
    end

    B0_inv_median = zeros(n_vars_local, n_vars_local)
    for i in 1:n_vars_local
        for j in 1:n_vars_local
            B0_inv_median[i, j] = median(B0_inv_stacked[:, i, j])
        end
    end

    B0_median = inv(B0_inv_median)

    # Compute structural shocks using median impact matrix
    residuals = var_result.residuals
    structural_shocks = Matrix((B0_median * residuals')')

    SignRestrictionResult(
        var_result=var_result,
        B0_inv=B0_inv_median,
        B0=B0_median,
        structural_shocks=structural_shocks,
        identification=SIGN,
        n_restrictions=length(constraints),
        constraints=constraints,
        B0_inv_set=accepted_B0_inv,
        irf_median=irf_median,
        irf_lower=irf_lower,
        irf_upper=irf_upper,
        n_draws=n_draws,
        n_accepted=n_accepted,
        acceptance_rate=acceptance_rate,
        rotation_method=rotation_method,
        horizons=horizons
    )
end


# =============================================================================
# Utility Functions
# =============================================================================

"""
    create_monetary_policy_constraints(; kwargs...)

Create standard monetary policy sign restrictions.

Follows Uhlig (2005) baseline specification:
- Contractionary money shock increases interest rate
- Contractionary money shock decreases output
- Contractionary money shock decreases prices

# Keyword Arguments
- `money_shock_idx::Int=1`: Index of the monetary policy shock
- `output_idx::Int=2`: Index of the output variable
- `price_idx::Int=3`: Index of the price level variable
- `interest_idx::Union{Int,Nothing}=nothing`: Index of the interest rate variable
- `max_horizon::Int=4`: Maximum horizon for constraints

# Returns
`Vector{SignRestrictionConstraint}`: Sign constraints for monetary policy identification
"""
function create_monetary_policy_constraints(;
    money_shock_idx::Int=1,
    output_idx::Int=2,
    price_idx::Int=3,
    interest_idx::Union{Int,Nothing}=nothing,
    max_horizon::Int=4
)::Vector{SignRestrictionConstraint}

    constraints = SignRestrictionConstraint[]

    # Output declines after contractionary shock
    for h in 0:max_horizon
        push!(constraints, SignRestrictionConstraint(money_shock_idx, output_idx, h, -1))
    end

    # Prices decline after contractionary shock
    for h in 0:max_horizon
        push!(constraints, SignRestrictionConstraint(money_shock_idx, price_idx, h, -1))
    end

    # Interest rate rises after contractionary shock
    if interest_idx !== nothing
        for h in 0:max_horizon
            push!(constraints, SignRestrictionConstraint(money_shock_idx, interest_idx, h, 1))
        end
    end

    return constraints
end


"""
    check_cholesky_in_set(var_result, constraints; horizons=20)

Check if Cholesky identification satisfies sign constraints.

Useful diagnostic: if Cholesky satisfies constraints, it should
always be in the identified set.

# Arguments
- `var_result::VARResult`: VAR estimation result
- `constraints::Vector{SignRestrictionConstraint}`: Sign constraints

# Keyword Arguments
- `horizons::Int=20`: Maximum horizon

# Returns
`Bool`: `true` if Cholesky identification satisfies all constraints
"""
function check_cholesky_in_set(
    var_result::VARResult,
    constraints::Vector{SignRestrictionConstraint};
    horizons::Int=20
)::Bool
    svar_chol = cholesky_svar(var_result)
    Phi = vma_coefficients(var_result, horizons)
    irf = compute_irf_from_impact(Phi, svar_chol.B0_inv, horizons)

    return check_sign_constraints(irf, constraints)
end


end # module
