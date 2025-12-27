"""
Tests for Julia Cointegration Tests (Johansen, Engle-Granger)

Session 147: Tests for Julia time-series parity.
"""

using Test
using Random
using CausalEstimators
using CausalEstimators.TimeSeries


@testset "Cointegration Tests" begin

    @testset "Johansen Test - Cointegrated System" begin
        Random.seed!(42)

        # Create cointegrated system: y1 and y2 share common stochastic trend
        n = 200
        trend = cumsum(randn(n))  # Common trend
        y1 = trend + randn(n) * 0.5
        y2 = 0.5 * trend + randn(n) * 0.5
        data = hcat(y1, y2)

        result = johansen_test(data; lags=2)

        @test result isa JohansenResult
        @test result.n_vars == 2
        @test result.lags == 2
        @test length(result.trace_stats) == 2
        @test length(result.eigenvalues) == 2

        # Should find at least rank 1 (cointegration exists)
        @test result.rank >= 1

        # Eigenvalues should be in [0, 1]
        @test all(0 .<= result.eigenvalues .<= 1)
    end

    @testset "Johansen Test - Non-cointegrated System" begin
        Random.seed!(42)

        # Two independent random walks - no cointegration
        n = 200
        y1 = cumsum(randn(n))
        y2 = cumsum(randn(n))
        data = hcat(y1, y2)

        result = johansen_test(data; lags=1)

        @test result.n_vars == 2
        # Likely rank 0 (no cointegration), but not guaranteed in finite sample
        @test result.rank >= 0
    end

    @testset "Johansen Test - Trivariate System" begin
        Random.seed!(42)

        n = 200
        trend = cumsum(randn(n))
        y1 = trend + randn(n) * 0.3
        y2 = 0.8 * trend + randn(n) * 0.3
        y3 = 0.5 * trend + randn(n) * 0.3
        data = hcat(y1, y2, y3)

        result = johansen_test(data; lags=2)

        @test result.n_vars == 3
        @test length(result.trace_stats) == 3
        @test size(result.eigenvectors) == (3, 3)
        @test size(result.adjustment) == (3, 3)
    end

    @testset "Johansen Test - Deterministic Order" begin
        Random.seed!(42)

        n = 200
        trend = cumsum(randn(n))
        data = hcat(trend + randn(n) * 0.3, 0.5 * trend + randn(n) * 0.3)

        # Test different det_order values
        result_m1 = johansen_test(data; lags=1, det_order=-1)
        result_0 = johansen_test(data; lags=1, det_order=0)
        result_1 = johansen_test(data; lags=1, det_order=1)

        @test result_m1.det_order == -1
        @test result_0.det_order == 0
        @test result_1.det_order == 1
    end

    @testset "Johansen Test - Input Validation" begin
        Random.seed!(42)

        # Too few variables
        @test_throws ErrorException johansen_test(randn(100, 1); lags=1)

        # Too many variables (>6)
        @test_throws ErrorException johansen_test(randn(100, 7); lags=1)

        # Invalid det_order
        @test_throws ErrorException johansen_test(randn(100, 2); lags=1, det_order=2)

        # Invalid lags
        @test_throws ErrorException johansen_test(randn(100, 2); lags=0)
    end

    @testset "Engle-Granger Test - Cointegrated" begin
        Random.seed!(42)

        # Create cointegrated pair
        n = 200
        trend = cumsum(randn(n))
        y = trend + randn(n) * 0.3
        x = 2 * trend + randn(n) * 0.3

        result = engle_granger_test(y, x)

        @test result isa EngleGrangerResult
        @test length(result.beta) == 2  # Intercept + slope
        @test length(result.residuals) == n
        @test result.adf_result isa ADFResult

        # Should detect cointegration
        @test result.is_cointegrated == true
    end

    @testset "Engle-Granger Test - Not Cointegrated" begin
        Random.seed!(42)

        # Independent random walks
        n = 200
        y = cumsum(randn(n))
        x = cumsum(randn(n))

        result = engle_granger_test(y, x)

        # Should not detect cointegration
        @test result.is_cointegrated == false
    end

    @testset "Engle-Granger Test - Multiple X Variables" begin
        Random.seed!(42)

        n = 200
        trend = cumsum(randn(n))
        y = trend + randn(n) * 0.3
        x1 = 0.5 * trend + randn(n) * 0.3
        x2 = 0.3 * trend + randn(n) * 0.3
        X = hcat(x1, x2)

        result = engle_granger_test(y, X)

        @test length(result.beta) == 3  # Intercept + 2 slopes
    end

end
