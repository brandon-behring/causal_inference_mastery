"""
Tests for Julia Bootstrap IRF and FEVD

Session 147: Tests for Julia time-series parity.
"""

using Test
using Random
using CausalEstimators
using CausalEstimators.TimeSeries
using CausalEstimators.TimeSeries.SVARTypes: has_confidence_bands, validate_rows_sum_to_one


@testset "Bootstrap IRF/FEVD" begin

    @testset "Bootstrap IRF - Basic" begin
        Random.seed!(42)

        # Generate VAR data
        data = randn(200, 3)
        var_result = var_estimate(data; lags=2)
        svar_result = cholesky_svar(var_result)

        # Small bootstrap for speed
        irf_ci = bootstrap_irf(data, svar_result;
            horizons=10, n_bootstrap=50, alpha=0.05, seed=42)

        @test irf_ci isa IRFResult
        @test size(irf_ci.irf) == (3, 3, 11)
        @test size(irf_ci.irf_lower) == (3, 3, 11)
        @test size(irf_ci.irf_upper) == (3, 3, 11)
        @test has_confidence_bands(irf_ci) == true
        @test irf_ci.n_bootstrap == 50
        @test irf_ci.alpha == 0.05

        # Lower should be <= point <= upper (for most entries)
        @test all(irf_ci.irf_lower .<= irf_ci.irf_upper)
    end

    @testset "Bootstrap IRF - Wild Bootstrap" begin
        Random.seed!(42)

        data = randn(200, 2)
        var_result = var_estimate(data; lags=1)
        svar_result = cholesky_svar(var_result)

        irf_wild = bootstrap_irf(data, svar_result;
            horizons=5, n_bootstrap=30, method="wild", seed=42)

        @test has_confidence_bands(irf_wild)
    end

    @testset "Bootstrap IRF - Cumulative" begin
        Random.seed!(42)

        data = randn(200, 2)
        var_result = var_estimate(data; lags=1)
        svar_result = cholesky_svar(var_result)

        irf_cum = bootstrap_irf(data, svar_result;
            horizons=5, n_bootstrap=30, cumulative=true, seed=42)

        @test irf_cum.cumulative == true
        @test has_confidence_bands(irf_cum)
    end

    @testset "Moving Block Bootstrap IRF" begin
        Random.seed!(42)

        data = randn(200, 3)
        var_result = var_estimate(data; lags=2)
        svar_result = cholesky_svar(var_result)

        irf_mbb = moving_block_bootstrap_irf(data, svar_result;
            horizons=10, n_bootstrap=30, seed=42)

        @test irf_mbb isa IRFResult
        @test has_confidence_bands(irf_mbb)
        @test irf_mbb.n_bootstrap == 30

        # With custom block length
        irf_mbb2 = moving_block_bootstrap_irf(data, svar_result;
            horizons=5, n_bootstrap=20, block_length=10, seed=42)

        @test has_confidence_bands(irf_mbb2)
    end

    @testset "Joint Confidence Bands - Bonferroni" begin
        Random.seed!(42)

        # Create synthetic bootstrap samples
        n_bootstrap, n_vars, horizons = 100, 2, 10
        irf_boots = randn(n_bootstrap, n_vars, n_vars, horizons + 1)

        lower, upper = joint_confidence_bands(irf_boots;
            alpha=0.05, method="bonferroni")

        @test size(lower) == (n_vars, n_vars, horizons + 1)
        @test size(upper) == (n_vars, n_vars, horizons + 1)
        @test all(lower .<= upper)
    end

    @testset "Joint Confidence Bands - Sup" begin
        Random.seed!(42)

        n_bootstrap, n_vars, horizons = 100, 2, 5
        irf_boots = randn(n_bootstrap, n_vars, n_vars, horizons + 1)

        lower, upper = joint_confidence_bands(irf_boots;
            alpha=0.05, method="sup")

        @test size(lower) == (n_vars, n_vars, horizons + 1)
        @test all(lower .<= upper)
    end

    @testset "Joint Confidence Bands - Simes" begin
        Random.seed!(42)

        n_bootstrap, n_vars, horizons = 100, 2, 5
        irf_boots = randn(n_bootstrap, n_vars, n_vars, horizons + 1)

        lower, upper = joint_confidence_bands(irf_boots;
            alpha=0.05, method="simes")

        @test size(lower) == (n_vars, n_vars, horizons + 1)
    end

    @testset "MBB IRF Joint" begin
        Random.seed!(42)

        data = randn(200, 2)
        var_result = var_estimate(data; lags=1)
        svar_result = cholesky_svar(var_result)

        irf_joint = moving_block_bootstrap_irf_joint(data, svar_result;
            horizons=5, n_bootstrap=30, joint_method="bonferroni", seed=42)

        @test irf_joint isa IRFResult
        @test has_confidence_bands(irf_joint)
    end

    @testset "Bootstrap FEVD" begin
        Random.seed!(42)

        data = randn(200, 3)
        var_result = var_estimate(data; lags=2)
        svar_result = cholesky_svar(var_result)

        fevd_result, fevd_lower, fevd_upper = bootstrap_fevd(data, svar_result;
            horizons=10, n_bootstrap=30, seed=42)

        @test fevd_result isa FEVDResult
        @test size(fevd_result.fevd) == (3, 3, 11)
        @test size(fevd_lower) == (3, 3, 11)
        @test size(fevd_upper) == (3, 3, 11)

        # FEVD rows should sum to 1
        @test validate_rows_sum_to_one(fevd_result; tol=1e-6)

        # FEVD should be in [0, 1]
        @test all(0 .<= fevd_result.fevd .<= 1)
    end

    @testset "Input Validation" begin
        Random.seed!(42)

        data = randn(200, 2)
        var_result = var_estimate(data; lags=1)
        svar_result = cholesky_svar(var_result)

        # n_bootstrap must be >= 2
        @test_throws ErrorException bootstrap_irf(data, svar_result;
            horizons=5, n_bootstrap=1)

        # alpha must be in (0, 1)
        @test_throws ErrorException bootstrap_irf(data, svar_result;
            horizons=5, n_bootstrap=10, alpha=0.0)
        @test_throws ErrorException bootstrap_irf(data, svar_result;
            horizons=5, n_bootstrap=10, alpha=1.0)

        # Invalid method
        @test_throws ErrorException bootstrap_irf(data, svar_result;
            horizons=5, n_bootstrap=10, method="invalid")
    end

end
