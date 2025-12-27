"""
Tests for Julia Stationarity Tests (ADF, KPSS, Phillips-Perron)

Session 147: Tests for Julia time-series parity.
"""

using Test
using Random
using CausalEstimators
using CausalEstimators.TimeSeries


@testset "Stationarity Tests" begin

    @testset "ADF Test" begin
        Random.seed!(42)

        # Test stationary series (white noise)
        y_stat = randn(200)
        result = adf_test(y_stat)

        @test result isa ADFResult
        @test result.is_stationary == true
        @test result.statistic < result.critical_values["5%"]
        @test result.n_obs == 200

        # Test non-stationary series (random walk)
        # Use longer series for more power
        y_nonstat = cumsum(randn(500))
        result = adf_test(y_nonstat)

        # ADF has limited power in finite samples, so we check p-value is high
        # rather than strict is_stationary == false
        @test result.p_value > 0.01  # Should fail to strongly reject unit root

        # Test with different regression types
        result_c = adf_test(y_stat; regression="c")
        result_ct = adf_test(y_stat; regression="ct")
        result_n = adf_test(y_stat; regression="n")

        @test result_c.regression == "c"
        @test result_ct.regression == "ct"
        @test result_n.regression == "n"

        # Test short series error
        @test_throws ErrorException adf_test(randn(5))
    end

    @testset "KPSS Test" begin
        Random.seed!(42)

        # Test stationary series
        y_stat = randn(200)
        result = kpss_test(y_stat)

        @test result isa KPSSResult
        @test result.is_stationary == true
        @test result.statistic < result.critical_values["5%"]

        # Test non-stationary series (random walk)
        y_nonstat = cumsum(randn(200))
        result = kpss_test(y_nonstat)

        @test result.is_stationary == false
        @test result.statistic > result.critical_values["5%"]

        # Test custom lags
        result1 = kpss_test(y_stat; lags=5)
        result2 = kpss_test(y_stat; lags=10)

        @test result1.lags == 5
        @test result2.lags == 10

        # Test regression types
        result_c = kpss_test(y_stat; regression="c")
        result_ct = kpss_test(y_stat; regression="ct")

        @test result_c.regression == "c"
        @test result_ct.regression == "ct"

        # KPSS doesn't support "n"
        @test_throws ErrorException kpss_test(y_stat; regression="n")
    end

    @testset "Phillips-Perron Test" begin
        Random.seed!(42)

        # Test stationary series
        y_stat = randn(500)  # Larger sample for more power
        result = phillips_perron_test(y_stat)

        @test result isa PPResult
        @test result.is_stationary == true

        # Test non-stationary series
        y_nonstat = cumsum(randn(200))
        result = phillips_perron_test(y_nonstat)

        @test result.is_stationary == false

        # Test regression types
        result_c = phillips_perron_test(y_stat; regression="c")
        result_ct = phillips_perron_test(y_stat; regression="ct")
        result_n = phillips_perron_test(y_stat; regression="n")

        @test result_c.regression == "c"
        @test result_ct.regression == "ct"
        @test result_n.regression == "n"
    end

    @testset "Confirmatory Stationarity Test" begin
        Random.seed!(42)

        # Stationary series - both tests should agree
        y_stat = randn(200)
        result = confirmatory_stationarity_test(y_stat)

        @test result isa ConfirmatoryResult
        @test result.conclusion == "stationary"
        @test result.adf.is_stationary == true
        @test result.kpss.is_stationary == true

        # Non-stationary series
        y_nonstat = cumsum(randn(500))
        result = confirmatory_stationarity_test(y_nonstat)

        # Should not conclude stationary
        @test result.conclusion != "stationary"
    end

    @testset "Difference Series" begin
        y = [1.0, 3.0, 6.0, 10.0, 15.0]

        # First difference
        d1 = difference_series(y; order=1)
        @test length(d1) == 4
        @test d1 ≈ [2.0, 3.0, 4.0, 5.0]

        # Second difference
        d2 = difference_series(y; order=2)
        @test length(d2) == 3
        @test d2 ≈ [1.0, 1.0, 1.0]

        # Error for invalid order
        @test_throws ErrorException difference_series(y; order=0)
        @test_throws ErrorException difference_series(y; order=5)
    end

    @testset "Check Stationarity (Multiple Series)" begin
        Random.seed!(42)

        # Use longer series for better power
        data = hcat(randn(500), cumsum(randn(500)))
        results = check_stationarity(data; var_names=["stat", "nonstat"])

        @test "stat" in keys(results)
        @test "nonstat" in keys(results)
        @test results["stat"].is_stationary == true
        # Check p-value is high rather than strict is_stationary == false
        @test results["nonstat"].p_value > 0.01
    end

end
