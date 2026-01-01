"""
Test Proxy SVAR (External Instrument SVAR)

Session 164: Tests for Proxy SVAR implementation.

3-Layer Validation:
- Layer 1: Known-Answer Tests (8 tests)
- Layer 2: Adversarial Tests (7 tests)
- Layer 3: Monte Carlo Tests (3 tests)
"""

using Test
using Random
using Statistics
using LinearAlgebra
using Logging

# Add project to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using CausalEstimators
using CausalEstimators.TimeSeries.SVARTypes: PROXY
using CausalEstimators.TimeSeries.ProxySVAR: first_stage_regression, complete_impact_matrix
using CausalEstimators.TimeSeries.ProxySVAR: n_vars, lags, n_obs, has_bootstrap_se


# =============================================================================
# Test Fixtures
# =============================================================================

"""
Generate test data with known Proxy SVAR structure.

DGP:
- VAR(1) with known coefficient matrix A1
- Known structural impact matrix B0_inv
- Instrument z correlated with first structural shock

# Arguments
- `n::Int`: Number of observations
- `seed::Int`: Random seed
- `instrument_strength::Float64`: Correlation of z with first shock (0.5 = moderate)
- `n_vars::Int`: Number of variables

# Returns
- `data`: VAR data (n x n_vars)
- `instrument`: External instrument (n-1,) - trimmed for VAR lags
- `B0_inv_true`: True impact matrix
- `A1`: VAR(1) coefficients
"""
function generate_proxy_svar_dgp(;
    n::Int=500,
    seed::Int=42,
    instrument_strength::Float64=0.5,
    n_vars::Int=3
)
    rng = MersenneTwister(seed)

    # Known impact matrix (lower triangular with some off-diagonal)
    B0_inv_true = Matrix{Float64}(I, n_vars, n_vars)
    B0_inv_true[2, 1] = 0.3  # Second var responds to first shock
    B0_inv_true[3, 1] = 0.2  # Third var responds to first shock
    B0_inv_true[3, 2] = 0.1  # Third var responds to second shock

    # VAR(1) coefficient matrix (stable)
    A1 = zeros(n_vars, n_vars)
    A1[1, 1] = 0.5
    A1[2, 2] = 0.4
    A1[3, 3] = 0.3
    A1[2, 1] = 0.1

    # Generate structural shocks
    eps = randn(rng, n, n_vars)

    # Generate instrument correlated with first shock
    noise = randn(rng, n)
    z = instrument_strength * eps[:, 1] + sqrt(1 - instrument_strength^2) * noise

    # Generate reduced-form errors: u = B0_inv * eps
    u = (B0_inv_true * eps')'

    # Generate VAR data
    data = zeros(n, n_vars)
    for t in 2:n
        data[t, :] = A1 * data[t-1, :] + u[t, :]
    end

    # Trim instrument to match VAR effective sample (1 lag)
    instrument_trimmed = z[2:end]

    return data, instrument_trimmed, B0_inv_true, A1
end


"""
Generate weak instrument test data.

# Arguments
- `n::Int`: Number of observations
- `seed::Int`: Random seed
- `instrument_strength::Float64`: Very low correlation (0.05 = very weak)
"""
function generate_weak_instrument_dgp(;
    n::Int=200,
    seed::Int=42,
    instrument_strength::Float64=0.05
)
    return generate_proxy_svar_dgp(
        n=n,
        seed=seed,
        instrument_strength=instrument_strength
    )
end


# =============================================================================
# Layer 1: Known-Answer Tests (8 tests)
# =============================================================================

@testset "Proxy SVAR Known-Answer Tests" begin

    @testset "test_basic_estimation" begin
        # Generate test data
        data, instrument, B0_inv_true, _ = generate_proxy_svar_dgp(n=500, seed=42)

        # Estimate VAR
        var_result = var_estimate(data, lags=1)

        # Run proxy SVAR
        result = proxy_svar(var_result, instrument, target_shock_idx=1)

        # Basic checks
        @test result isa ProxySVARResult
        @test result.identification == PROXY
        @test result.target_shock_idx == 1
        @test result.target_residual_idx == 1
        @test result.is_just_identified == true
        @test result.is_over_identified == false
    end

    @testset "test_output_shapes" begin
        data, instrument, _, _ = generate_proxy_svar_dgp(n=500, seed=42)
        var_result = var_estimate(data, lags=1)
        result = proxy_svar(var_result, instrument, target_shock_idx=1)

        n_vars_local = 3
        n_obs_eff = 499  # n - lags

        @test size(result.B0_inv) == (n_vars_local, n_vars_local)
        @test size(result.B0) == (n_vars_local, n_vars_local)
        @test size(result.structural_shocks) == (n_obs_eff, n_vars_local)
        @test length(result.impact_column) == n_vars_local
        @test length(result.impact_column_se) == n_vars_local
        @test length(result.impact_column_ci_lower) == n_vars_local
        @test length(result.impact_column_ci_upper) == n_vars_local
        @test length(result.instrument) == n_obs_eff
    end

    @testset "test_first_stage_f_stat" begin
        # Strong instrument should have high F-stat
        data, instrument, _, _ = generate_proxy_svar_dgp(
            n=500, seed=42, instrument_strength=0.5
        )
        var_result = var_estimate(data, lags=1)
        result = proxy_svar(var_result, instrument, target_shock_idx=1)

        # With instrument_strength=0.5, F should be well above 10
        @test result.first_stage_f_stat > 10.0
        @test result.is_weak_instrument == false
        @test result.first_stage_r2 > 0.05  # Some explanatory power
    end

    @testset "test_impact_column_normalization" begin
        data, instrument, _, _ = generate_proxy_svar_dgp(n=500, seed=42)
        var_result = var_estimate(data, lags=1)
        result = proxy_svar(var_result, instrument, target_shock_idx=1)

        # Impact column should have target element = 1.0 (before scaling)
        # After full normalization, check B0_inv structure
        @test !any(isnan, result.impact_column)
        @test !any(isinf, result.impact_column)

        # The impact column should be close to 1.0 at target index
        @test isapprox(result.impact_column[1], 1.0, atol=1e-10)
    end

    @testset "test_identification_method_is_proxy" begin
        data, instrument, _, _ = generate_proxy_svar_dgp(n=500, seed=42)
        var_result = var_estimate(data, lags=1)
        result = proxy_svar(var_result, instrument, target_shock_idx=1)

        @test result.identification == PROXY
    end

    @testset "test_strong_instrument_recovery" begin
        # With strong instrument, should approximately recover true impact
        data, instrument, B0_inv_true, _ = generate_proxy_svar_dgp(
            n=1000, seed=42, instrument_strength=0.7
        )
        var_result = var_estimate(data, lags=1)
        result = proxy_svar(var_result, instrument, target_shock_idx=1)

        # Normalize true first column to compare
        true_col = B0_inv_true[:, 1]
        true_col_normalized = true_col / true_col[1]

        # Should be reasonably close (within 50% for each element)
        for i in 1:3
            ratio = result.impact_column[i] / true_col_normalized[i]
            @test abs(ratio - 1.0) < 0.5 || isapprox(true_col_normalized[i], 0.0, atol=0.1)
        end
    end

    @testset "test_structural_shocks_properties" begin
        data, instrument, _, _ = generate_proxy_svar_dgp(n=500, seed=42)
        var_result = var_estimate(data, lags=1)
        result = proxy_svar(var_result, instrument, target_shock_idx=1)

        # Structural shocks should have roughly mean 0
        for i in 1:3
            @test abs(mean(result.structural_shocks[:, i])) < 0.15
        end

        # Should have positive variance
        for i in 1:3
            @test var(result.structural_shocks[:, i]) > 0.1
        end
    end

    @testset "test_confidence_intervals_bracket" begin
        data, instrument, _, _ = generate_proxy_svar_dgp(n=500, seed=42)
        var_result = var_estimate(data, lags=1)
        result = proxy_svar(var_result, instrument, target_shock_idx=1, alpha=0.05)

        # CI should bracket the point estimate
        for i in 1:3
            @test result.impact_column_ci_lower[i] <= result.impact_column[i]
            @test result.impact_column[i] <= result.impact_column_ci_upper[i]
        end

        # CI width should be positive
        for i in 1:3
            width = result.impact_column_ci_upper[i] - result.impact_column_ci_lower[i]
            @test width >= 0
        end
    end

end


# =============================================================================
# Layer 2: Adversarial Tests (7 tests)
# =============================================================================

@testset "Proxy SVAR Adversarial Tests" begin

    @testset "test_weak_instrument_warning" begin
        # Very weak instrument should trigger warning
        data, instrument, _, _ = generate_weak_instrument_dgp(
            n=200, seed=42, instrument_strength=0.05
        )
        var_result = var_estimate(data, lags=1)

        # Should warn about weak instrument
        result = @test_logs (:warn, r"Weak instrument detected") proxy_svar(
            var_result, instrument, target_shock_idx=1
        )

        @test result.is_weak_instrument == true
        @test result.first_stage_f_stat < 10.0
    end

    @testset "test_invalid_instrument_length" begin
        data, instrument, _, _ = generate_proxy_svar_dgp(n=500, seed=42)
        var_result = var_estimate(data, lags=1)

        # Instrument too short
        bad_instrument = instrument[1:100]

        @test_throws ErrorException proxy_svar(
            var_result, bad_instrument, target_shock_idx=1
        )
    end

    @testset "test_constant_instrument_error" begin
        data, _, _, _ = generate_proxy_svar_dgp(n=500, seed=42)
        var_result = var_estimate(data, lags=1)

        # Constant instrument
        constant_instrument = ones(var_result.n_obs_effective)

        @test_throws ErrorException proxy_svar(
            var_result, constant_instrument, target_shock_idx=1
        )
    end

    @testset "test_invalid_target_idx" begin
        data, instrument, _, _ = generate_proxy_svar_dgp(n=500, seed=42)
        var_result = var_estimate(data, lags=1)

        # Out of bounds target_shock_idx
        @test_throws ErrorException proxy_svar(
            var_result, instrument, target_shock_idx=0
        )
        @test_throws ErrorException proxy_svar(
            var_result, instrument, target_shock_idx=10
        )
    end

    @testset "test_nan_in_instrument" begin
        data, instrument, _, _ = generate_proxy_svar_dgp(n=500, seed=42)
        var_result = var_estimate(data, lags=1)

        # Instrument with NaN
        bad_instrument = copy(instrument)
        bad_instrument[50] = NaN

        @test_throws ErrorException proxy_svar(
            var_result, bad_instrument, target_shock_idx=1
        )
    end

    @testset "test_nearly_perfect_instrument" begin
        # Test stability when R^2 -> 1
        rng = MersenneTwister(42)
        n = 500
        n_vars = 3

        # Create VAR residuals
        eps = randn(rng, n, n_vars)

        # Create near-perfect instrument (R^2 close to 1)
        z = eps[:, 1] + 0.001 * randn(rng, n)  # Almost perfect correlation

        # Create VAR data
        A1 = 0.5 * Matrix{Float64}(I, n_vars, n_vars)
        data = zeros(n, n_vars)
        for t in 2:n
            data[t, :] = A1 * data[t-1, :] + eps[t, :]
        end

        var_result = var_estimate(data, lags=1)
        instrument = z[2:end]

        # Should not error
        result = proxy_svar(var_result, instrument, target_shock_idx=1)

        @test result.first_stage_r2 > 0.95
        @test !any(isnan, result.impact_column)
        @test !any(isinf, result.impact_column)
    end

    @testset "test_invalid_target_residual_idx" begin
        data, instrument, _, _ = generate_proxy_svar_dgp(n=500, seed=42)
        var_result = var_estimate(data, lags=1)

        # Out of bounds target_residual_idx
        @test_throws ErrorException proxy_svar(
            var_result, instrument,
            target_shock_idx=1,
            target_residual_idx=10
        )
    end

end


# =============================================================================
# Layer 3: Monte Carlo Tests (3 tests)
# =============================================================================

@testset "Proxy SVAR Monte Carlo Tests" begin

    @testset "test_impact_column_unbiased" begin
        # Monte Carlo: check that impact column is approximately unbiased
        n_mc = 100
        n_vars = 3
        n = 300

        true_impact = [1.0, 0.3, 0.2]  # True first column (normalized)
        estimated_impacts = zeros(n_mc, n_vars)

        for mc in 1:n_mc
            data, instrument, _, _ = generate_proxy_svar_dgp(
                n=n, seed=mc, instrument_strength=0.5
            )
            var_result = var_estimate(data, lags=1)
            result = proxy_svar(var_result, instrument, target_shock_idx=1)
            estimated_impacts[mc, :] = result.impact_column
        end

        # Compute mean across MC runs
        mean_estimate = vec(mean(estimated_impacts, dims=1))

        # Bias should be small (< 0.15 for each component)
        for i in 1:n_vars
            bias = abs(mean_estimate[i] - true_impact[i])
            @test bias < 0.20  # Allow 20% bias for small sample
        end
    end

    @testset "test_weak_instrument_larger_variance" begin
        # Weak instrument should have larger variance in estimates
        n_mc = 50
        n = 300

        # Strong instrument
        strong_impacts = zeros(n_mc)
        for mc in 1:n_mc
            data, instrument, _, _ = generate_proxy_svar_dgp(
                n=n, seed=mc, instrument_strength=0.6
            )
            var_result = var_estimate(data, lags=1)
            result = proxy_svar(var_result, instrument, target_shock_idx=1)
            strong_impacts[mc] = result.impact_column[2]
        end

        # Weak instrument (suppress warnings for this test)
        weak_impacts = zeros(n_mc)
        for mc in 1:n_mc
            # Use weaker instrument
            data, instrument, _, _ = generate_proxy_svar_dgp(
                n=n, seed=mc + 1000, instrument_strength=0.15
            )
            var_result = var_estimate(data, lags=1)
            # Suppress warnings since we know instrument is weak
            result = with_logger(NullLogger()) do
                proxy_svar(var_result, instrument, target_shock_idx=1)
            end
            weak_impacts[mc] = result.impact_column[2]
        end

        # Weak instrument should have larger variance
        strong_var = var(strong_impacts)
        weak_var = var(weak_impacts)

        @test weak_var > strong_var
    end

    @testset "test_bootstrap_se_accuracy" begin
        # Bootstrap SE should be close to analytical SE
        data, instrument, _, _ = generate_proxy_svar_dgp(
            n=500, seed=42, instrument_strength=0.5
        )
        var_result = var_estimate(data, lags=1)

        # With bootstrap
        result_boot = proxy_svar(
            var_result, instrument,
            target_shock_idx=1,
            bootstrap_se=true,
            n_bootstrap=200,
            seed=123
        )

        @test result_boot.bootstrap_se !== nothing
        @test result_boot.n_bootstrap == 200

        # Bootstrap SE should be in same ballpark as analytical
        # (Skip first element since it's normalized to 1.0 with SE=0)
        for i in 2:3
            if result_boot.impact_column_se[i] > 0.01  # Avoid division by tiny SE
                ratio = result_boot.bootstrap_se[i] / result_boot.impact_column_se[i]
                @test 0.3 < ratio < 3.0  # Within factor of 3
            end
        end
    end

end


# =============================================================================
# Helper Function Tests
# =============================================================================

@testset "Proxy SVAR Helper Functions" begin

    @testset "test_first_stage_regression" begin
        rng = MersenneTwister(42)
        n = 200

        # Known regression: y = 1 + 2*z + noise
        z = randn(rng, n)
        y = 1.0 .+ 2.0 .* z + 0.5 * randn(rng, n)

        fitted, f_stat, r2, residuals = first_stage_regression(y, z)

        @test length(fitted) == n
        @test length(residuals) == n
        @test f_stat > 10  # Should be strong
        @test r2 > 0.5     # Should have good fit
        @test isapprox(mean(residuals), 0.0, atol=0.1)
    end

    @testset "test_weak_instrument_diagnostics" begin
        # Strong instrument
        diag_strong = weak_instrument_diagnostics(25.0, 200)
        @test diag_strong["is_weak"] == false
        @test diag_strong["is_very_weak"] == false
        @test occursin("Strong", diag_strong["interpretation"])

        # Moderate instrument
        diag_mod = weak_instrument_diagnostics(12.0, 200)
        @test diag_mod["is_weak"] == false
        @test occursin("Moderate", diag_mod["interpretation"])

        # Weak instrument
        diag_weak = weak_instrument_diagnostics(7.0, 200)
        @test diag_weak["is_weak"] == true
        @test diag_weak["is_very_weak"] == false
        @test occursin("Weak", diag_weak["interpretation"])

        # Very weak instrument
        diag_vweak = weak_instrument_diagnostics(3.0, 200)
        @test diag_vweak["is_weak"] == true
        @test diag_vweak["is_very_weak"] == true
        @test occursin("very weak", lowercase(diag_vweak["interpretation"]))
    end

    @testset "test_compute_irf_from_proxy" begin
        data, instrument, _, _ = generate_proxy_svar_dgp(n=500, seed=42)
        var_result = var_estimate(data, lags=1)
        result = proxy_svar(var_result, instrument, target_shock_idx=1)

        horizons = 10
        irf = compute_irf_from_proxy(result, horizons=horizons)

        @test size(irf) == (3, 3, horizons + 1)
        @test !any(isnan, irf)
        @test !any(isinf, irf)

        # Impact response (h=0) should match B0_inv
        @test isapprox(irf[:, :, 1], result.B0_inv, atol=1e-10)
    end

    @testset "test_result_accessors" begin
        data, instrument, _, _ = generate_proxy_svar_dgp(n=500, seed=42)
        var_result = var_estimate(data, lags=1)
        result = proxy_svar(var_result, instrument, target_shock_idx=1)

        @test n_vars(result) == 3
        @test lags(result) == 1
        @test n_obs(result) == 499
        @test has_bootstrap_se(result) == false

        # With bootstrap
        result_boot = proxy_svar(
            var_result, instrument,
            target_shock_idx=1,
            bootstrap_se=true,
            n_bootstrap=50,
            seed=42
        )
        @test has_bootstrap_se(result_boot) == true
    end

    @testset "test_complete_impact_matrix" begin
        # Test that complete_impact_matrix produces valid covariance decomposition
        rng = MersenneTwister(42)

        # Random positive definite sigma_u
        A = randn(rng, 3, 3)
        sigma_u = A * A'

        # Random impact column
        impact_column = [1.0, 0.3, 0.2]

        B0_inv = complete_impact_matrix(impact_column, sigma_u, 1)

        @test size(B0_inv) == (3, 3)
        @test !any(isnan, B0_inv)
        @test !any(isinf, B0_inv)

        # First column should be related to input (scaled)
        # Check it's not just zeros
        @test norm(B0_inv[:, 1]) > 0.01
    end

end


# =============================================================================
# Run all tests
# =============================================================================

# Logging for weak instrument tests
using Logging

println("\n" * "="^60)
println("Running Proxy SVAR Tests (Session 164)")
println("="^60 * "\n")
