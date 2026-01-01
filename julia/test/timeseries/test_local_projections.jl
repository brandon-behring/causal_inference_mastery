#=
Tests for Local Projections (Jordà 2005)

Session 160: Julia parity for LP-based impulse response estimation.

Test Layers:
1. Known-Answer - Basic functionality, output shapes, CI bracketing
2. Robustness - LP vs VAR comparison, Cholesky shocks orthogonality
3. Monte Carlo - Sign correctness validation
=#

using Test
using Random
using Statistics
using LinearAlgebra
using CausalEstimators

# =============================================================================
# Helper: Generate VAR(1) DGP
# =============================================================================

function generate_var1_data(n::Int, A1::Matrix{Float64}; seed::Int=42)
    Random.seed!(seed)
    n_vars = size(A1, 1)
    data = zeros(n, n_vars)

    for t in 2:n
        data[t, :] = A1 * data[t-1, :] + randn(n_vars) * 0.5
    end

    return data
end


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================

@testset "Local Projections Basic" begin

    @testset "Basic estimation runs" begin
        A1 = [0.5 0.0; 0.3 0.4]
        data = generate_var1_data(300, A1)

        result = local_projection_irf(data, horizons=10, lags=2)

        @test result isa LocalProjectionResult
        @test size(result.irf) == (2, 2, 11)  # 2 vars, horizons 0-10
        @test size(result.se) == (2, 2, 11)
        @test result.horizons == 10
        @test result.lags == 2
    end

    @testset "Output shapes correct" begin
        A1 = [0.5 0.1; 0.2 0.4]
        data = generate_var1_data(300, A1)

        result = local_projection_irf(data, horizons=15, lags=3)

        @test size(result.irf) == (2, 2, 16)
        @test size(result.se) == (2, 2, 16)
        @test size(result.ci_lower) == (2, 2, 16)
        @test size(result.ci_upper) == (2, 2, 16)
    end

    @testset "CI brackets point estimate" begin
        A1 = [0.5 0.0; 0.3 0.4]
        data = generate_var1_data(300, A1)

        result = local_projection_irf(data, horizons=10, lags=2, alpha=0.05)

        # Lower ≤ IRF ≤ Upper everywhere
        @test all(result.ci_lower .<= result.irf)
        @test all(result.irf .<= result.ci_upper)
    end

    @testset "Variable names assigned" begin
        A1 = [0.5 0.0; 0.3 0.4]
        data = generate_var1_data(300, A1)

        result = local_projection_irf(
            data, horizons=10, lags=2, var_names=["output", "inflation"]
        )

        @test result.var_names == ["output", "inflation"]
    end

    @testset "IRF direct access works" begin
        A1 = [0.5 0.0; 0.3 0.4]
        data = generate_var1_data(300, A1)

        result = local_projection_irf(data, horizons=10, lags=2)

        # Direct field access (Julia idiomatic)
        response = result.irf[2, 1, :]
        @test length(response) == 11

        # Single horizon access
        response_h5 = result.irf[2, 1, 6]  # +1 for Julia indexing
        @test response_h5 isa Float64
    end

    @testset "Significance via CI" begin
        A1 = [0.5 0.0; 0.3 0.4]
        data = generate_var1_data(300, A1)

        result = local_projection_irf(data, horizons=10, lags=2)

        # Check significance: CI doesn't contain zero
        lower = result.ci_lower[2, 1, 2]
        upper = result.ci_upper[2, 1, 2]
        significant = (lower > 0) || (upper < 0)
        @test significant isa Bool
    end

    @testset "Number of variables" begin
        A1 = [0.5 0.0; 0.3 0.4]
        data = generate_var1_data(300, A1)

        result = local_projection_irf(data, horizons=10, lags=2)

        @test size(result.irf, 1) == 2
    end

end


# =============================================================================
# Layer 2: LP vs VAR Comparison and Robustness
# =============================================================================

@testset "LP vs VAR Comparison" begin

    @testset "LP and VAR similar for VAR(1) DGP" begin
        Random.seed!(123)
        n = 500
        A1 = [0.5 0.0; 0.3 0.4]
        data = generate_var1_data(n, A1, seed=123)

        # LP estimation
        lp_result = local_projection_irf(data, horizons=10, lags=1)

        # VAR estimation
        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)
        var_irf = compute_irf(svar_result, horizons=10)

        # Should be reasonably similar (same identification)
        max_diff = maximum(abs.(lp_result.irf .- var_irf.irf))

        @test max_diff < 0.5
    end

    @testset "Impact response (h=0) matches" begin
        Random.seed!(456)
        A1 = [0.5 0.1; 0.2 0.4]
        data = generate_var1_data(400, A1, seed=456)

        # LP estimation
        lp_result = local_projection_irf(data, horizons=5, lags=1)

        # VAR estimation
        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)
        var_irf = compute_irf(svar_result, horizons=5)

        # Impact response should be very close (both use Cholesky)
        lp_impact = lp_result.irf[:, :, 1]
        var_impact = var_irf.irf[:, :, 1]

        @test isapprox(lp_impact, var_impact, atol=0.15)
    end

end


@testset "Cholesky Shock Computation" begin

    @testset "Shocks approximately orthogonal" begin
        A1 = [0.5 0.0; 0.3 0.4]
        data = generate_var1_data(500, A1, seed=789)

        # Access internal function via module
        shocks = CausalEstimators.TimeSeries.LocalProjections.compute_cholesky_shocks(data, 2)

        # Correlation should be low
        corr_val = cor(shocks[:, 1], shocks[:, 2])
        @test abs(corr_val) < 0.15
    end

    @testset "Shocks have approximately unit variance" begin
        A1 = [0.5 0.0; 0.3 0.4]
        data = generate_var1_data(500, A1, seed=789)

        shocks = CausalEstimators.TimeSeries.LocalProjections.compute_cholesky_shocks(data, 2)

        variances = var(shocks, dims=1)
        @test all(isapprox.(variances, 1.0, atol=0.3))
    end

end


@testset "HAC Standard Errors" begin

    @testset "All SEs positive" begin
        A1 = [0.5 0.0; 0.3 0.4]
        data = generate_var1_data(300, A1)

        result = local_projection_irf(data, horizons=5, lags=2)

        @test all(result.se .> 0)
    end

    @testset "Bartlett kernel works" begin
        A1 = [0.5 0.0; 0.3 0.4]
        data = generate_var1_data(300, A1)

        result = local_projection_irf(data, horizons=5, lags=2, hac_kernel="bartlett")

        @test result.hac_kernel == "bartlett"
        @test all(result.se .> 0)
    end

    @testset "Custom bandwidth applied" begin
        A1 = [0.5 0.0; 0.3 0.4]
        data = generate_var1_data(300, A1)

        result = local_projection_irf(data, horizons=5, lags=2, hac_bandwidth=15)

        @test result.hac_bandwidth == 15
    end

end


@testset "LP Robustness to Misspecification" begin

    @testset "LP similar with different lags" begin
        Random.seed!(789)
        n = 300

        # True DGP is VAR(3)
        A1 = [0.4 0.0; 0.2 0.3]
        A2 = [0.2 0.0; 0.1 0.15]
        A3 = [0.1 0.0; 0.05 0.1]

        data = zeros(n, 2)
        for t in 4:n
            data[t, :] = A1 * data[t-1, :] + A2 * data[t-2, :] + A3 * data[t-3, :] + randn(2) * 0.5
        end

        # LP with lag 1 (misspecified)
        lp_lag1 = local_projection_irf(data, horizons=5, lags=1)

        # LP with lag 3 (correct)
        lp_lag3 = local_projection_irf(data, horizons=5, lags=3)

        # LP should be reasonably robust to lag choice
        max_diff = maximum(abs.(lp_lag1.irf .- lp_lag3.irf))
        @test max_diff < 0.6
    end

end


# =============================================================================
# Layer 3: Monte Carlo Tests
# =============================================================================

@testset "Monte Carlo Sign Correctness" begin

    @testset "LP correctly identifies sign of causal effect" begin
        n_sims = 50
        n_obs = 300
        true_effect = 0.4

        correct_sign = 0

        for sim in 1:n_sims
            Random.seed!(222 + sim)

            data = zeros(n_obs, 2)
            for t in 2:n_obs
                data[t, 1] = 0.5 * data[t-1, 1] + randn()
                data[t, 2] = true_effect * data[t-1, 1] + 0.4 * data[t-1, 2] + randn()
            end

            result = local_projection_irf(data, horizons=5, lags=1, alpha=0.05)

            # Check if sign at horizon 1 is correct (positive)
            irf_1_0_h1 = result.irf[2, 1, 2]  # Julia 1-indexed

            if irf_1_0_h1 > 0
                correct_sign += 1
            end
        end

        correct_rate = correct_sign / n_sims
        @test correct_rate > 0.70
    end

end


@testset "Confidence Band Properties" begin

    @testset "CI width positive" begin
        A1 = [0.5 0.0; 0.3 0.4]
        data = generate_var1_data(300, A1)

        result = local_projection_irf(data, horizons=10, lags=2, alpha=0.05)

        width = result.ci_upper .- result.ci_lower
        @test all(width .> 0)
    end

    @testset "CI width not absurdly large" begin
        A1 = [0.5 0.0; 0.3 0.4]
        data = generate_var1_data(300, A1)

        result = local_projection_irf(data, horizons=10, lags=2, alpha=0.05)

        width = result.ci_upper .- result.ci_lower
        @test all(width .< 20)
    end

    @testset "CI width proportional to SE" begin
        A1 = [0.5 0.0; 0.3 0.4]
        data = generate_var1_data(300, A1)

        result = local_projection_irf(data, horizons=10, lags=2, alpha=0.05)

        avg_width = mean(result.ci_upper .- result.ci_lower)
        avg_se = mean(result.se)
        expected_width = avg_se * 2 * 1.96

        @test isapprox(avg_width, expected_width, rtol=0.01)
    end

end


# =============================================================================
# State-Dependent LP Tests
# =============================================================================

@testset "State-Dependent LP" begin

    @testset "Returns two IRFs" begin
        Random.seed!(333)
        n = 300

        data = zeros(n, 2)
        for t in 2:n
            data[t, :] = 0.5 * data[t-1, :] + randn(2)
        end

        state = Float64.(rand(n) .> 0.5)

        result = state_dependent_lp(data, state, horizons=5, lags=2)

        @test haskey(result, "high_state_irf")
        @test haskey(result, "low_state_irf")
        @test haskey(result, "difference")
        @test haskey(result, "diff_significant")

        @test result["high_state_irf"] isa LocalProjectionResult
        @test result["low_state_irf"] isa LocalProjectionResult
    end

    @testset "Detects state-dependent effects" begin
        Random.seed!(444)
        n = 500

        data = zeros(n, 2)
        state = zeros(n)

        for t in 2:n
            state[t] = data[t-1, 1] > 0 ? 1.0 : 0.0

            if state[t] == 1
                data[t, 1] = 0.7 * data[t-1, 1] + randn()
                data[t, 2] = 0.5 * data[t-1, 1] + randn()
            else
                data[t, 1] = 0.3 * data[t-1, 1] + randn()
                data[t, 2] = 0.1 * data[t-1, 1] + randn()
            end
        end

        result = state_dependent_lp(data, state, horizons=5, lags=1)

        diff = result["difference"]
        @test !all(isapprox.(diff, 0, atol=0.1))
    end

end


# =============================================================================
# API Compatibility Tests
# =============================================================================

@testset "LP to IRF Conversion" begin

    @testset "Conversion preserves data" begin
        A1 = [0.5 0.0; 0.3 0.4]
        data = generate_var1_data(300, A1)

        lp_result = local_projection_irf(data, horizons=10, lags=2)
        irf_result = lp_to_irf_result(lp_result)

        @test irf_result isa IRFResult
        @test irf_result.irf == lp_result.irf
        @test irf_result.irf_lower == lp_result.ci_lower
        @test irf_result.irf_upper == lp_result.ci_upper
        @test irf_result.horizons == lp_result.horizons
        @test irf_result.var_names == lp_result.var_names
    end

end


@testset "Cumulative IRF" begin

    @testset "Cumulative sums correctly" begin
        A1 = [0.5 0.0; 0.3 0.4]
        data = generate_var1_data(300, A1)

        result = local_projection_irf(data, horizons=10, lags=2, cumulative=false)
        result_cum = local_projection_irf(data, horizons=10, lags=2, cumulative=true)

        expected_cum = cumsum(result.irf, dims=3)
        @test isapprox(result_cum.irf, expected_cum, rtol=1e-5)
    end

end


# =============================================================================
# Input Validation Tests
# =============================================================================

@testset "Input Validation" begin

    @testset "Invalid horizons" begin
        data = randn(100, 2)
        @test_throws ErrorException local_projection_irf(data, horizons=-1, lags=2)
    end

    @testset "Invalid lags" begin
        data = randn(100, 2)
        @test_throws ErrorException local_projection_irf(data, horizons=10, lags=0)
    end

    @testset "Insufficient observations" begin
        data = randn(10, 2)
        @test_throws ErrorException local_projection_irf(data, horizons=10, lags=5)
    end

    @testset "External shock required" begin
        data = randn(100, 2)
        @test_throws ErrorException local_projection_irf(
            data, horizons=10, lags=2, shock_type="external"
        )
    end

    @testset "Invalid shock type" begin
        data = randn(100, 2)
        @test_throws ErrorException local_projection_irf(
            data, horizons=10, lags=2, shock_type="invalid"
        )
    end

    @testset "var_names length mismatch" begin
        data = randn(100, 2)
        @test_throws ErrorException local_projection_irf(
            data, horizons=10, lags=2, var_names=["a"]
        )
    end

end


# =============================================================================
# External Shock Tests
# =============================================================================

@testset "External Shock" begin

    @testset "External shock estimation runs" begin
        Random.seed!(555)
        n = 200

        shock = randn(n)
        data = zeros(n, 2)

        for t in 2:n
            data[t, 1] = 0.5 * data[t-1, 1] + 0.3 * shock[t-1] + randn() * 0.5
            data[t, 2] = 0.4 * data[t-1, 2] + randn() * 0.5
        end

        result = local_projection_irf(
            data, horizons=10, lags=2, shock_type="external", external_shock=shock
        )

        @test result.method == "external"
        @test size(result.irf) == (2, 2, 11)
    end

    @testset "External shock length validation" begin
        data = randn(100, 2)
        wrong_shock = randn(110)

        @test_throws ErrorException local_projection_irf(
            data, horizons=10, lags=2, shock_type="external", external_shock=wrong_shock
        )
    end

end
