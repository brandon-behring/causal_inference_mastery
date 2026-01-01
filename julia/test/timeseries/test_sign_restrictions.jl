"""
Tests for Sign Restrictions SVAR (Uhlig 2005)

Session 162: Test suite for set-identified SVAR using sign restrictions.

Test Structure (3-layer validation):
1. Known-Answer Tests: Basic functionality, output shapes, constraint satisfaction
2. Adversarial Tests: Invalid inputs, conflicting constraints, edge cases
3. Monte Carlo Tests: Set width, coverage, sign correctness across draws
"""

using Test
using Random
using LinearAlgebra
using Statistics

# Include parent test setup if running standalone
if !@isdefined(CausalEstimators)
    using Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
    using CausalEstimators
end

using CausalEstimators.TimeSeries: var_estimate, cholesky_svar, vma_coefficients
using CausalEstimators.TimeSeries: SignRestrictionConstraint, SignRestrictionResult
using CausalEstimators.TimeSeries: sign_restriction_svar
using CausalEstimators.TimeSeries: create_monetary_policy_constraints, check_cholesky_in_set
using CausalEstimators.TimeSeries: givens_rotation_matrix, random_orthogonal_givens, random_orthogonal_qr
using CausalEstimators.TimeSeries: check_sign_constraints, validate_constraints, compute_irf_from_impact
using CausalEstimators.TimeSeries: IdentificationMethod
using CausalEstimators.TimeSeries.SVARTypes: SIGN


# =============================================================================
# Test Data Generation
# =============================================================================

function generate_var_data(n::Int, k::Int; seed::Int=42)
    """Generate stable VAR(1) data with known structure."""
    Random.seed!(seed)

    # Stable AR coefficients
    A1 = [0.4 0.1 0.05;
          0.15 0.35 0.1;
          0.05 0.1 0.3][1:k, 1:k]

    data = zeros(n, k)
    for t in 2:n
        data[t, :] = A1 * data[t-1, :] + randn(k) * 0.5
    end

    return data
end


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================

@testset "Sign Restrictions Known-Answer Tests" begin

    @testset "Basic Estimation" begin
        Random.seed!(42)
        data = generate_var_data(200, 3)
        var_result = var_estimate(data, lags=2)

        constraints = [
            SignRestrictionConstraint(1, 1, 0, 1),  # Shock 1 positive on var 1 at impact
            SignRestrictionConstraint(1, 2, 0, 1),  # Shock 1 positive on var 2 at impact
        ]

        result = sign_restriction_svar(var_result, constraints,
                                        horizons=10, n_draws=1000, seed=42)

        @test result isa SignRestrictionResult
        @test result.identification == SIGN
        @test result.n_accepted > 0
        @test 0.0 <= result.acceptance_rate <= 1.0
    end

    @testset "Output Shapes" begin
        Random.seed!(42)
        data = generate_var_data(200, 3)
        var_result = var_estimate(data, lags=2)

        constraints = [
            SignRestrictionConstraint(1, 1, 0, 1),
        ]

        horizons = 15
        result = sign_restriction_svar(var_result, constraints,
                                        horizons=horizons, n_draws=500, seed=42)

        # IRF shapes: (n_vars, n_vars, horizons+1)
        @test size(result.irf_median) == (3, 3, horizons + 1)
        @test size(result.irf_lower) == (3, 3, horizons + 1)
        @test size(result.irf_upper) == (3, 3, horizons + 1)

        # Impact matrices
        @test size(result.B0_inv) == (3, 3)
        @test size(result.B0) == (3, 3)

        # Structural shocks
        T = size(var_result.residuals, 1)
        @test size(result.structural_shocks) == (T, 3)
    end

    @testset "Constraints Satisfied" begin
        Random.seed!(42)
        data = generate_var_data(200, 3)
        var_result = var_estimate(data, lags=2)

        constraints = [
            SignRestrictionConstraint(1, 1, 0, 1),
            SignRestrictionConstraint(1, 2, 0, 1),
        ]

        result = sign_restriction_svar(var_result, constraints,
                                        horizons=10, n_draws=500, seed=42)

        # Compute VMA coefficients
        Phi = vma_coefficients(var_result, result.horizons)

        # Check every accepted B0_inv
        for B0_inv in result.B0_inv_set
            irf = compute_irf_from_impact(Phi, B0_inv, result.horizons)
            @test check_sign_constraints(irf, constraints)
        end
    end

    @testset "Cholesky Within Set" begin
        Random.seed!(42)
        data = generate_var_data(200, 3)
        var_result = var_estimate(data, lags=2)

        # Constraints requiring positive diagonal (consistent with Cholesky)
        constraints = [
            SignRestrictionConstraint(1, 1, 0, 1),
            SignRestrictionConstraint(2, 2, 0, 1),
            SignRestrictionConstraint(3, 3, 0, 1),
        ]

        cholesky_satisfies = check_cholesky_in_set(var_result, constraints, horizons=10)

        if cholesky_satisfies
            result = sign_restriction_svar(var_result, constraints,
                                            horizons=10, n_draws=500, seed=42)
            # Should have reasonable acceptance rate
            @test result.acceptance_rate > 0.05
        else
            @test_skip "Cholesky doesn't satisfy constraints for this data"
        end
    end

    @testset "Set Bounds Bracket Median" begin
        Random.seed!(42)
        data = generate_var_data(200, 3)
        var_result = var_estimate(data, lags=2)

        constraints = [
            SignRestrictionConstraint(1, 1, 0, 1),
        ]

        result = sign_restriction_svar(var_result, constraints,
                                        horizons=10, n_draws=500, seed=42)

        # Lower <= median <= upper everywhere
        @test all(result.irf_lower .<= result.irf_median .+ 1e-10)
        @test all(result.irf_median .<= result.irf_upper .+ 1e-10)
    end

    @testset "Acceptance Rate Positive" begin
        Random.seed!(42)
        data = generate_var_data(200, 3)
        var_result = var_estimate(data, lags=2)

        constraints = [
            SignRestrictionConstraint(1, 1, 0, 1),
        ]

        result = sign_restriction_svar(var_result, constraints,
                                        horizons=10, n_draws=500, seed=42)

        @test result.n_accepted > 0
        @test result.acceptance_rate > 0
        @test result.acceptance_rate <= 1.0
    end

    @testset "Result Properties" begin
        Random.seed!(42)
        data = generate_var_data(200, 3)
        var_result = var_estimate(data, lags=2)

        constraints = [
            SignRestrictionConstraint(1, 1, 0, 1),
        ]

        result = sign_restriction_svar(var_result, constraints,
                                        horizons=10, n_draws=500, seed=42)

        # Test accessor functions
        using CausalEstimators.TimeSeries.SignRestrictions: n_vars, lags, var_names
        @test n_vars(result) == 3
        @test lags(result) == 2
        @test length(var_names(result)) == 3
    end

    @testset "IRF Result Conversion" begin
        Random.seed!(42)
        data = generate_var_data(200, 3)
        var_result = var_estimate(data, lags=2)

        constraints = [
            SignRestrictionConstraint(1, 1, 0, 1),
        ]

        result = sign_restriction_svar(var_result, constraints,
                                        horizons=10, n_draws=500, seed=42)

        using CausalEstimators.TimeSeries.SignRestrictions: to_irf_result
        irf_result = to_irf_result(result)

        @test irf_result.horizons == 10
        @test irf_result.orthogonalized == true
        @test irf_result.n_bootstrap == result.n_accepted
    end

end


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================

@testset "Sign Restrictions Adversarial Tests" begin

    @testset "Conflicting Constraints" begin
        Random.seed!(42)
        data = generate_var_data(200, 3)
        var_result = var_estimate(data, lags=2)

        # Shock 1 must be both positive AND negative on var 1 at impact
        conflicting = [
            SignRestrictionConstraint(1, 1, 0, 1),
            SignRestrictionConstraint(1, 1, 0, -1),
        ]

        @test_throws Exception sign_restriction_svar(var_result, conflicting,
                                                      horizons=10, n_draws=500, seed=42)
    end

    @testset "Invalid Horizon" begin
        Random.seed!(42)
        data = generate_var_data(200, 3)
        var_result = var_estimate(data, lags=2)

        constraints = [
            SignRestrictionConstraint(1, 1, 100, 1),  # Horizon 100 > max
        ]

        @test_throws Exception sign_restriction_svar(var_result, constraints,
                                                      horizons=10, n_draws=100, seed=42)
    end

    @testset "Invalid Indices" begin
        Random.seed!(42)
        data = generate_var_data(200, 3)
        var_result = var_estimate(data, lags=2)

        # Out of bounds shock index
        constraints_bad_shock = [
            SignRestrictionConstraint(10, 1, 0, 1),
        ]
        @test_throws Exception sign_restriction_svar(var_result, constraints_bad_shock,
                                                      horizons=10, n_draws=100, seed=42)

        # Out of bounds response index
        constraints_bad_response = [
            SignRestrictionConstraint(1, 10, 0, 1),
        ]
        @test_throws Exception sign_restriction_svar(var_result, constraints_bad_response,
                                                      horizons=10, n_draws=100, seed=42)
    end

    @testset "Empty Constraints" begin
        Random.seed!(42)
        data = generate_var_data(200, 3)
        var_result = var_estimate(data, lags=2)

        # No constraints - all rotations should be accepted
        result = sign_restriction_svar(var_result, SignRestrictionConstraint[],
                                        horizons=10, n_draws=100, seed=42)

        @test result.acceptance_rate == 1.0
    end

    @testset "Invalid Rotation Method" begin
        Random.seed!(42)
        data = generate_var_data(200, 3)
        var_result = var_estimate(data, lags=2)

        constraints = [
            SignRestrictionConstraint(1, 1, 0, 1),
        ]

        @test_throws Exception sign_restriction_svar(var_result, constraints,
                                                      horizons=10, n_draws=100,
                                                      rotation_method="invalid", seed=42)
    end

    @testset "Invalid Sign Value" begin
        @test_throws Exception SignRestrictionConstraint(1, 1, 0, 0)
        @test_throws Exception SignRestrictionConstraint(1, 1, 0, 2)
    end

    @testset "Constraint Constructor Validation" begin
        # Valid constraints
        @test SignRestrictionConstraint(1, 2, 5, 1) isa SignRestrictionConstraint
        @test SignRestrictionConstraint(1, 2, 0, -1) isa SignRestrictionConstraint

        # Invalid - negative indices
        @test_throws Exception SignRestrictionConstraint(0, 1, 0, 1)
        @test_throws Exception SignRestrictionConstraint(1, 0, 0, 1)
        @test_throws Exception SignRestrictionConstraint(1, 1, -1, 1)
    end

end


# =============================================================================
# Layer 3: Monte Carlo Tests
# =============================================================================

@testset "Sign Restrictions Monte Carlo Tests" begin

    @testset "Set Width With Constraints" begin
        Random.seed!(42)
        data = generate_var_data(200, 3)
        var_result = var_estimate(data, lags=2)

        # Few constraints
        few_constraints = [
            SignRestrictionConstraint(1, 1, 0, 1),
        ]

        # More constraints
        more_constraints = [
            SignRestrictionConstraint(1, 1, 0, 1),
            SignRestrictionConstraint(1, 1, 1, 1),
            SignRestrictionConstraint(1, 2, 0, 1),
            SignRestrictionConstraint(1, 2, 1, 1),
        ]

        result_few = sign_restriction_svar(var_result, few_constraints,
                                            horizons=10, n_draws=2000, seed=42)
        result_more = sign_restriction_svar(var_result, more_constraints,
                                             horizons=10, n_draws=2000, seed=42)

        # Set width = mean(upper - lower)
        width_few = mean(result_few.irf_upper - result_few.irf_lower)
        width_more = mean(result_more.irf_upper - result_more.irf_lower)

        # More constraints should give equal or narrower set (with tolerance)
        @test width_more <= width_few * 1.1
    end

    @testset "Sign Correctness Across Set" begin
        Random.seed!(42)
        data = generate_var_data(200, 3)
        var_result = var_estimate(data, lags=2)

        constraints = [
            SignRestrictionConstraint(1, 1, 0, 1),
            SignRestrictionConstraint(1, 2, 0, 1),
        ]

        result = sign_restriction_svar(var_result, constraints,
                                        horizons=10, n_draws=2000, seed=42)

        Phi = vma_coefficients(var_result, result.horizons)

        violations = 0
        for B0_inv in result.B0_inv_set
            irf = compute_irf_from_impact(Phi, B0_inv, result.horizons)
            if !check_sign_constraints(irf, constraints)
                violations += 1
            end
        end

        @test violations == 0
    end

    @testset "Rotation Methods Equivalent" begin
        Random.seed!(42)
        data = generate_var_data(200, 3)
        var_result = var_estimate(data, lags=2)

        constraints = [
            SignRestrictionConstraint(1, 1, 0, 1),
        ]

        result_givens = sign_restriction_svar(var_result, constraints,
                                               horizons=10, n_draws=500,
                                               rotation_method="givens", seed=42)
        result_qr = sign_restriction_svar(var_result, constraints,
                                           horizons=10, n_draws=500,
                                           rotation_method="qr", seed=42)

        # Both should have positive acceptance
        @test result_givens.n_accepted > 0
        @test result_qr.n_accepted > 0

        # Acceptance rates should be roughly similar (within factor of 3)
        ratio = result_givens.acceptance_rate / max(result_qr.acceptance_rate, 1e-6)
        @test 0.3 <= ratio <= 3.0
    end

    @testset "Seed Reproducibility" begin
        Random.seed!(42)
        data = generate_var_data(200, 3)
        var_result = var_estimate(data, lags=2)

        constraints = [
            SignRestrictionConstraint(1, 1, 0, 1),
        ]

        result1 = sign_restriction_svar(var_result, constraints,
                                         horizons=10, n_draws=200, seed=123)
        result2 = sign_restriction_svar(var_result, constraints,
                                         horizons=10, n_draws=200, seed=123)

        @test result1.n_accepted == result2.n_accepted
        @test result1.acceptance_rate == result2.acceptance_rate
        @test result1.irf_median ≈ result2.irf_median
    end

end


# =============================================================================
# Additional Tests
# =============================================================================

@testset "Rotation Matrix Tests" begin

    @testset "Givens Rotation Orthogonal" begin
        G = givens_rotation_matrix(3, 1, 2, π/4)

        # G * G' = I
        @test G * G' ≈ I(3) atol=1e-10

        # det(G) = 1
        @test det(G) ≈ 1.0 atol=1e-10
    end

    @testset "Random Orthogonal Givens" begin
        rng = MersenneTwister(42)
        Q = random_orthogonal_givens(4, rng)

        # Q * Q' = I
        @test Q * Q' ≈ I(4) atol=1e-10

        # det(Q) = 1
        @test det(Q) ≈ 1.0 atol=1e-10
    end

    @testset "Random Orthogonal QR" begin
        rng = MersenneTwister(42)
        Q = random_orthogonal_qr(4, rng)

        # Q * Q' = I
        @test Q * Q' ≈ I(4) atol=1e-10

        # det(Q) = 1
        @test det(Q) ≈ 1.0 atol=1e-10
    end

end


@testset "Monetary Policy Constraints Helper" begin

    @testset "Create Constraints" begin
        constraints = create_monetary_policy_constraints()

        # Should have constraints for output and prices at each horizon
        @test length(constraints) == 10  # 5 horizons × 2 variables

        # All should be SignRestrictionConstraints
        for c in constraints
            @test c isa SignRestrictionConstraint
        end
    end

    @testset "With Interest Rate" begin
        constraints = create_monetary_policy_constraints(interest_idx=4)

        # 5 horizons × 3 variables
        @test length(constraints) == 15
    end

end


@testset "Check Cholesky In Set" begin

    @testset "Returns Bool" begin
        Random.seed!(42)
        data = generate_var_data(200, 3)
        var_result = var_estimate(data, lags=2)

        constraints = [
            SignRestrictionConstraint(1, 1, 0, 1),
        ]

        result = check_cholesky_in_set(var_result, constraints, horizons=10)
        @test result isa Bool
    end

    @testset "Diagonal Positive" begin
        Random.seed!(42)
        data = generate_var_data(200, 3)
        var_result = var_estimate(data, lags=2)

        # Cholesky diagonal is always positive
        constraints = [
            SignRestrictionConstraint(i, i, 0, 1) for i in 1:3
        ]

        @test check_cholesky_in_set(var_result, constraints, horizons=10) == true
    end

end


@testset "Get IRF Bounds" begin
    Random.seed!(42)
    data = generate_var_data(200, 3)
    var_result = var_estimate(data, lags=2)

    constraints = [
        SignRestrictionConstraint(1, 1, 0, 1),
    ]

    result = sign_restriction_svar(var_result, constraints,
                                    horizons=10, n_draws=500, seed=42)

    using CausalEstimators.TimeSeries.SignRestrictions: get_irf_bounds
    bounds = get_irf_bounds(result, 1, 1)

    @test haskey(bounds, :median)
    @test haskey(bounds, :lower)
    @test haskey(bounds, :upper)
    @test haskey(bounds, :horizon)

    @test length(bounds.median) == 11
    @test bounds.horizon == collect(0:10)
end


# Run all tests if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    @info "Running Sign Restrictions tests..."
end
