"""
Unit tests for bunching type definitions.

Tests BunchingProblem, SaezBunching, CounterfactualResult, BunchingSolution.
"""

using Test
using CausalEstimators

@testset "Bunching Types" begin

    @testset "BunchingProblem construction" begin
        @testset "basic construction" begin
            data = [1.0, 2.0, 3.0, 4.0, 5.0]
            problem = BunchingProblem(data, 3.0, 1.0)

            @test length(problem.data) == 5
            @test problem.kink_point == 3.0
            @test problem.bunching_width == 1.0
            @test isnothing(problem.t1_rate)
            @test isnothing(problem.t2_rate)
        end

        @testset "with tax rates" begin
            data = randn(100) .* 15 .+ 50
            problem = BunchingProblem(data, 50.0, 5.0; t1_rate=0.25, t2_rate=0.35)

            @test problem.kink_point == 50.0
            @test problem.bunching_width == 5.0
            @test problem.t1_rate == 0.25
            @test problem.t2_rate == 0.35
        end

        @testset "type stability" begin
            # Float64
            data64 = [1.0, 2.0, 3.0]
            problem64 = BunchingProblem(data64, 2.0, 0.5)
            @test eltype(problem64.data) == Float64

            # Float32
            data32 = Float32[1.0f0, 2.0f0, 3.0f0]
            problem32 = BunchingProblem(data32, 2.0f0, 0.5f0)
            @test eltype(problem32.data) == Float32
        end

        @testset "validation" begin
            # Empty data
            @test_throws ArgumentError BunchingProblem(Float64[], 1.0, 0.5)

            # Non-finite data
            @test_throws ArgumentError BunchingProblem([1.0, NaN, 3.0], 2.0, 0.5)
            @test_throws ArgumentError BunchingProblem([1.0, Inf, 3.0], 2.0, 0.5)

            # Non-positive bunching width
            @test_throws ArgumentError BunchingProblem([1.0, 2.0, 3.0], 2.0, 0.0)
            @test_throws ArgumentError BunchingProblem([1.0, 2.0, 3.0], 2.0, -1.0)

            # Invalid tax rates
            @test_throws ArgumentError BunchingProblem(
                [1.0, 2.0, 3.0], 2.0, 0.5; t1_rate=-0.1, t2_rate=0.3
            )
            @test_throws ArgumentError BunchingProblem(
                [1.0, 2.0, 3.0], 2.0, 0.5; t1_rate=1.0, t2_rate=0.5
            )
            @test_throws ArgumentError BunchingProblem(
                [1.0, 2.0, 3.0], 2.0, 0.5; t1_rate=0.4, t2_rate=0.3  # t2 <= t1
            )
        end
    end

    @testset "SaezBunching construction" begin
        @testset "default parameters" begin
            estimator = SaezBunching()
            @test estimator.n_bins == 50
            @test estimator.polynomial_order == 7
            @test estimator.n_bootstrap == 200
        end

        @testset "custom parameters" begin
            estimator = SaezBunching(n_bins=80, polynomial_order=5, n_bootstrap=500)
            @test estimator.n_bins == 80
            @test estimator.polynomial_order == 5
            @test estimator.n_bootstrap == 500
        end

        @testset "validation" begin
            @test_throws ArgumentError SaezBunching(n_bins=5)  # < 10
            @test_throws ArgumentError SaezBunching(polynomial_order=0)  # < 1
            @test_throws ArgumentError SaezBunching(n_bootstrap=5)  # < 10
        end
    end

    @testset "display methods" begin
        data = randn(100) .* 10 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0)

        # Test show methods don't error
        str = repr(problem)
        @test occursin("BunchingProblem", str)
        @test occursin("100", str)

        buf = IOBuffer()
        show(buf, MIME"text/plain"(), problem)
        detailed_str = String(take!(buf))
        @test occursin("Observations", detailed_str)
        @test occursin("Kink point", detailed_str)
    end

end
