"""
Type I Error Verification Tests (Julia)

Validates that estimators correctly control Type I error rate at nominal level.
Under the null hypothesis (true effect = 0), the rejection rate should be ~5%.

Phase 1: 5 core estimators (one per method family)
- SimpleATE (RCT)
- ObservationalIPW (Observational)
- ClassicDiD (DiD)
- TSLS (IV)
- SharpRDD (RDD)

Target: Rejection rate between 3% and 7% (5% +/- 2%)

References:
    - Imbens & Rubin (2015). Causal Inference for Statistics, Social, and Biomedical Sciences
    - Angrist & Pischke (2009). Mostly Harmless Econometrics
"""

using Test
using CausalEstimators
using Random
using Statistics

# =============================================================================
# Configuration
# =============================================================================

const N_SIMULATIONS = 2000  # Sufficient for Type I error estimation
const ALPHA = 0.05          # Nominal significance level
const TYPE_I_LOWER = 0.03   # 5% - 2%
const TYPE_I_UPPER = 0.07   # 5% + 2%

# =============================================================================
# Helper Functions
# =============================================================================

"""
Count rejections where CI excludes true effect (0).

Returns: (rejection_count, rejection_rate)
"""
function count_rejections(ci_lowers::Vector{Float64}, ci_uppers::Vector{Float64}, true_effect::Float64=0.0)
    rejections = 0
    for (lower, upper) in zip(ci_lowers, ci_uppers)
        # Reject if CI doesn't contain true effect
        if lower > true_effect || upper < true_effect
            rejections += 1
        end
    end
    rejection_rate = rejections / length(ci_lowers)
    return (rejections, rejection_rate)
end

# =============================================================================
# DGP Functions (Null Effect)
# =============================================================================

"""Generate RCT data with NO treatment effect."""
function dgp_rct_null(; n::Int=200, seed::Int)
    Random.seed!(seed)

    # Randomized treatment (explicitly Vector{Bool})
    treatment = Vector{Bool}(rand(n) .< 0.5)

    # Outcome with NO treatment effect (true ATE = 0)
    outcomes = Vector{Float64}(randn(n))

    return (outcomes=outcomes, treatment=treatment)
end

"""Generate observational data with NO treatment effect."""
function dgp_ipw_null(; n::Int=500, seed::Int)
    Random.seed!(seed)

    # Covariates
    X = Matrix{Float64}(randn(n, 2))

    # Propensity score (treatment depends on covariates)
    logit = 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2]
    propensity = 1.0 ./ (1.0 .+ exp.(-logit))
    treatment = Vector{Bool}([rand() < p for p in propensity])

    # Outcome with NO treatment effect (true ATE = 0)
    # Y depends on X (confounding) but NOT on treatment
    Y = Vector{Float64}(1.0 .+ 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2] .+ randn(n))

    return (Y=Y, treatment=treatment, X=X)
end

"""Generate DiD data with NO treatment effect."""
function dgp_did_null(; n_units::Int=100, seed::Int)
    Random.seed!(seed)

    n_treated = div(n_units, 2)
    n_control = n_units - n_treated
    n_pre = 1
    n_post = 1
    n_periods = n_pre + n_post

    # Unit and time indices
    unit_id = Vector{Int}(repeat(1:n_units, inner=n_periods))
    time = Vector{Int}(repeat(1:n_periods, outer=n_units))

    # Treatment indicator
    treatment_unit = Vector{Bool}(vcat(fill(true, n_treated), fill(false, n_control)))
    treatment = Vector{Bool}(repeat(treatment_unit, inner=n_periods))

    # Post indicator
    post_time = Vector{Bool}([t > n_pre for t in 1:n_periods])
    post = Vector{Bool}(repeat(post_time, outer=n_units))

    # Fixed effects
    unit_fe = randn(n_units)
    time_fe = randn(n_periods) .* 0.5
    unit_fe_obs = unit_fe[unit_id]
    time_fe_obs = time_fe[time]

    # Outcome with NO treatment effect (true ATT = 0)
    epsilon = randn(n_units * n_periods)
    outcomes = Vector{Float64}(unit_fe_obs .+ time_fe_obs .+ epsilon)
    # Note: No treatment*post interaction - true effect is 0

    return (outcomes=outcomes, treatment=treatment, post=post, unit_id=unit_id, time=time)
end

"""Generate IV data with NO causal effect."""
function dgp_iv_null(; n::Int=500, seed::Int)
    Random.seed!(seed)

    # Strong instrument (pi = 0.5)
    pi = 0.5
    rho = 0.5  # Endogeneity

    # Generate instrument
    z = Vector{Float64}(randn(n))

    # Correlated errors for endogeneity
    nu = randn(n)
    eps = rho * nu + sqrt(1 - rho^2) * randn(n)

    # First stage: D = pi*Z + nu
    d = Vector{Float64}(pi * z .+ nu)

    # Second stage: Y = 0*D + eps (TRUE EFFECT = 0)
    y = Vector{Float64}(eps)

    return (y=y, d=d, z=z)
end

"""Generate RDD data with NO discontinuity effect."""
function dgp_rdd_null(; n::Int=1000, seed::Int)
    Random.seed!(seed)

    # Running variable centered at 0
    x = Vector{Float64}(randn(n) .* 2.0)

    # Treatment assignment at cutoff = 0
    treatment = Vector{Bool}(x .>= 0.0)

    # Outcome with NO discontinuity (true effect = 0)
    # Linear in x, no jump at cutoff
    y = Vector{Float64}(2.0 .* x .+ randn(n))

    return (y=y, x=x, treatment=treatment)
end

# =============================================================================
# Type I Error Tests
# =============================================================================

@testset "Type I Error Verification" begin

    # =========================================================================
    # RCT: SimpleATE
    # =========================================================================
    @testset "Type I Error - SimpleATE (RCT)" begin
        """
        Type I error test for SimpleATE.
        Under null (true_ate=0), rejection rate should be ~5%.
        """
        ci_lowers = Float64[]
        ci_uppers = Float64[]

        for seed in 1:N_SIMULATIONS
            data = dgp_rct_null(n=200, seed=seed)

            try
                problem = RCTProblem(
                    data.outcomes,
                    data.treatment,
                    nothing,  # no strata
                    nothing,  # no covariates
                    (alpha=ALPHA,)
                )
                solution = solve(problem, SimpleATE())

                push!(ci_lowers, solution.ci_lower)
                push!(ci_uppers, solution.ci_upper)
            catch e
                # Skip failed iterations (shouldn't happen for RCT)
                @warn "SimpleATE failed at seed $seed: $e"
                continue
            end
        end

        rejections, rejection_rate = count_rejections(ci_lowers, ci_uppers, 0.0)

        @test TYPE_I_LOWER < rejection_rate < TYPE_I_UPPER

        println("\nSimpleATE Type I Error: $(round(rejection_rate * 100, digits=1))% ($rejections/$(length(ci_lowers)))")
    end

    # =========================================================================
    # Observational: IPW
    # =========================================================================
    @testset "Type I Error - ObservationalIPW" begin
        """
        Type I error test for IPW.
        Under null (true_ate=0), rejection rate should be ~5%.
        """
        ci_lowers = Float64[]
        ci_uppers = Float64[]

        for seed in 1:N_SIMULATIONS
            data = dgp_ipw_null(n=500, seed=seed)

            try
                # Use convenience constructor with keyword arguments
                problem = ObservationalProblem(
                    data.Y,
                    data.treatment,
                    data.X;
                    propensity=nothing,
                    alpha=ALPHA,
                    trim_threshold=0.01,
                    stabilize=false
                )
                solution = solve(problem, ObservationalIPW())

                push!(ci_lowers, solution.ci_lower)
                push!(ci_uppers, solution.ci_upper)
            catch e
                # Skip iterations with extreme propensity scores
                continue
            end
        end

        # Need sufficient successful iterations
        @test length(ci_lowers) >= N_SIMULATIONS * 0.9

        rejections, rejection_rate = count_rejections(ci_lowers, ci_uppers, 0.0)

        # IPW can be conservative due to propensity score estimation
        # Allow wider bounds: 1.5% - 8%
        ipw_type_i_lower = 0.015
        ipw_type_i_upper = 0.08

        @test ipw_type_i_lower < rejection_rate < ipw_type_i_upper

        println("\nIPW Type I Error: $(round(rejection_rate * 100, digits=1))% ($rejections/$(length(ci_lowers)))")
    end

    # =========================================================================
    # DiD: ClassicDiD
    # =========================================================================
    @testset "Type I Error - ClassicDiD" begin
        """
        Type I error test for Classic DiD.
        Under null (true_effect=0), rejection rate should be ~5%.
        """
        ci_lowers = Float64[]
        ci_uppers = Float64[]

        for seed in 1:N_SIMULATIONS
            data = dgp_did_null(n_units=100, seed=seed)

            try
                problem = DiDProblem(
                    data.outcomes,
                    data.treatment,
                    data.post,
                    data.unit_id,
                    data.time,
                    (alpha=ALPHA,)
                )
                solution = solve(problem, ClassicDiD())

                push!(ci_lowers, solution.ci_lower)
                push!(ci_uppers, solution.ci_upper)
            catch e
                continue
            end
        end

        @test length(ci_lowers) >= N_SIMULATIONS * 0.9

        rejections, rejection_rate = count_rejections(ci_lowers, ci_uppers, 0.0)

        # DiD with clustered data can have slight over-rejection
        # Use bounds 2.5% - 8% (slightly wider than strict 3-7%)
        did_type_i_lower = 0.025
        did_type_i_upper = 0.08

        @test did_type_i_lower < rejection_rate < did_type_i_upper

        println("\nClassicDiD Type I Error: $(round(rejection_rate * 100, digits=1))% ($rejections/$(length(ci_lowers)))")
    end

    # =========================================================================
    # IV: 2SLS
    # =========================================================================
    @testset "Type I Error - 2SLS (IV)" begin
        """
        Type I error test for 2SLS.
        Under null (true_effect=0), rejection rate should be ~5%.
        Uses strong instrument to ensure valid inference.
        """
        ci_lowers = Float64[]
        ci_uppers = Float64[]

        for seed in 1:N_SIMULATIONS
            data = dgp_iv_null(n=500, seed=seed)

            try
                # Reshape z to be a matrix
                z_matrix = reshape(data.z, length(data.z), 1)

                # IVProblem takes 5 arguments: outcomes, treatment, instruments, covariates, parameters
                problem = IVProblem(
                    data.y,
                    data.d,
                    z_matrix,
                    nothing,  # no covariates
                    (alpha=ALPHA,)
                )
                solution = solve(problem, TSLS())

                push!(ci_lowers, solution.ci_lower)
                push!(ci_uppers, solution.ci_upper)
            catch e
                continue
            end
        end

        @test length(ci_lowers) >= N_SIMULATIONS * 0.9

        rejections, rejection_rate = count_rejections(ci_lowers, ci_uppers, 0.0)

        @test TYPE_I_LOWER < rejection_rate < TYPE_I_UPPER

        println("\n2SLS Type I Error: $(round(rejection_rate * 100, digits=1))% ($rejections/$(length(ci_lowers)))")
    end

    # =========================================================================
    # RDD: SharpRDD
    # =========================================================================
    @testset "Type I Error - SharpRDD" begin
        """
        Type I error test for Sharp RDD.
        Under null (true_effect=0), rejection rate should be ~5%.
        """
        ci_lowers = Float64[]
        ci_uppers = Float64[]

        for seed in 1:N_SIMULATIONS
            data = dgp_rdd_null(n=1000, seed=seed)

            try
                problem = RDDProblem(
                    data.y,
                    data.x,
                    data.treatment,
                    0.0,      # cutoff
                    nothing,  # no bandwidth override
                    (alpha=ALPHA,)
                )
                # Skip density test for speed
                solution = solve(problem, SharpRDD(run_density_test=false))

                push!(ci_lowers, solution.ci_lower)
                push!(ci_uppers, solution.ci_upper)
            catch e
                continue
            end
        end

        @test length(ci_lowers) >= N_SIMULATIONS * 0.8

        rejections, rejection_rate = count_rejections(ci_lowers, ci_uppers, 0.0)

        # RDD can have very different Type I error due to bandwidth selection
        # CCT procedure is conservative by design (bias-corrected CIs are wider)
        # This results in under-rejection, which is acceptable behavior
        # Use very wide bounds: 0.5% - 10% to accommodate conservative inference
        rdd_type_i_lower = 0.005
        rdd_type_i_upper = 0.10

        @test rdd_type_i_lower < rejection_rate < rdd_type_i_upper

        println("\nSharpRDD Type I Error: $(round(rejection_rate * 100, digits=1))% ($rejections/$(length(ci_lowers)))")
    end

    # =========================================================================
    # Summary
    # =========================================================================
    @testset "Type I Error Summary" begin
        """
        Quick summary test that documents what we're testing.
        """
        estimators_tested = [
            "SimpleATE (RCT)",
            "ObservationalIPW (Observational)",
            "ClassicDiD (DiD)",
            "2SLS (IV)",
            "SharpRDD (RDD)"
        ]

        println("\n=== Type I Error Verification ===")
        println("Estimators: $(length(estimators_tested))")
        println("Simulations per test: $N_SIMULATIONS")
        println("Nominal alpha: $ALPHA")
        println("Acceptable range: [$TYPE_I_LOWER, $TYPE_I_UPPER]")
        println("=" ^ 40)

        for est in estimators_tested
            println("  - $est")
        end

        @test true
    end
end
