"""
Unit tests for counterfactual estimation functions.

Tests polynomial_counterfactual, estimate_counterfactual, compute_excess_mass, compute_elasticity.
"""

using Test
using Statistics
using CausalEstimators

@testset "Counterfactual Estimation" begin

    @testset "polynomial_counterfactual" begin
        @testset "basic functionality" begin
            # Simple uniform distribution
            bin_centers = collect(1.0:1.0:20.0)
            counts = fill(10.0, 20)

            # Bunching region in middle
            counterfactual, coeffs, r2 = polynomial_counterfactual(
                bin_centers, counts, 9.0, 11.0; polynomial_order=3
            )

            @test length(counterfactual) == 20
            @test length(coeffs) == 4  # polynomial_order + 1
            @test r2 >= 0.0 && r2 <= 1.0
            # Flat counterfactual should approximate uniform
            @test all(c -> c >= 0.0, counterfactual)
        end

        @testset "excludes bunching region from fit" begin
            bin_centers = collect(1.0:1.0:20.0)
            # Create excess at bunching region
            counts = fill(10.0, 20)
            counts[9:11] .= 50.0  # Bunching

            counterfactual, _, r2 = polynomial_counterfactual(
                bin_centers, counts, 8.0, 12.0; polynomial_order=3
            )

            # Counterfactual should be smooth (not jump at bunching)
            outside_bunching = [i for i in 1:20 if i < 8 || i > 12]
            @test all(c -> c > 0.0, counterfactual[outside_bunching])
        end

        @testset "validation" begin
            bin_centers = collect(1.0:1.0:20.0)
            counts = fill(10.0, 20)

            # Length mismatch
            @test_throws ArgumentError polynomial_counterfactual(
                bin_centers, counts[1:15], 9.0, 11.0
            )

            # Invalid polynomial order
            @test_throws ArgumentError polynomial_counterfactual(
                bin_centers, counts, 9.0, 11.0; polynomial_order=0
            )

            # Insufficient bins outside bunching region
            @test_throws ArgumentError polynomial_counterfactual(
                bin_centers, counts, 0.0, 25.0; polynomial_order=7
            )
        end

        @testset "high polynomial order" begin
            bin_centers = collect(1.0:0.5:50.0)
            # Create smooth curved pattern (not constant)
            counts = 20.0 .+ 5.0 .* sin.(bin_centers .* 0.1) .+ 2.0 .* (bin_centers .- 25) .^ 2 ./ 100

            counterfactual, coeffs, r2 = polynomial_counterfactual(
                bin_centers, counts, 24.0, 26.0; polynomial_order=9
            )

            @test length(coeffs) == 10
            @test r2 >= 0.5  # Should fit reasonably well with variation
        end
    end

    @testset "estimate_counterfactual" begin
        @testset "basic functionality" begin
            # Generate simple data
            data = randn(500) .* 15 .+ 50
            problem = BunchingProblem(data, 50.0, 5.0)

            result = estimate_counterfactual(problem, 40; polynomial_order=5)

            @test length(result.bin_centers) == 40
            @test length(result.actual_counts) == 40
            @test length(result.counterfactual_counts) == 40
            @test result.n_bins == 40
            @test result.polynomial_order == 5
            @test result.r_squared >= 0.0 && result.r_squared <= 1.0
        end

        @testset "bunching region" begin
            data = randn(500) .* 15 .+ 50
            problem = BunchingProblem(data, 50.0, 5.0)

            result = estimate_counterfactual(problem, 50)

            lower, upper = result.bunching_region
            @test lower < upper
            @test lower <= 50.0 <= upper
        end

        @testset "validation" begin
            data = randn(100) .* 10 .+ 50
            problem = BunchingProblem(data, 50.0, 5.0)

            @test_throws ArgumentError estimate_counterfactual(problem, 5)  # n_bins < 10
        end
    end

    @testset "compute_excess_mass" begin
        @testset "with bunching" begin
            # Create result with clear bunching
            bin_centers = collect(1.0:1.0:20.0)
            actual = fill(10.0, 20)
            actual[9:11] .= 30.0  # Excess at bunching region
            counterfactual = fill(10.0, 20)

            result = CounterfactualResult(
                bin_centers,
                actual,
                counterfactual,
                [10.0],
                1,
                (8.0, 12.0),
                0.95,
                20,
                1.0,
            )

            excess_mass, excess_count, h0 = compute_excess_mass(result)

            @test excess_count > 0
            @test excess_mass > 0
            @test h0 == 10.0
            @test excess_count ≈ (30 - 10) * 3  # 3 bins * 20 excess
        end

        @testset "no bunching" begin
            bin_centers = collect(1.0:1.0:20.0)
            actual = fill(10.0, 20)
            counterfactual = fill(10.0, 20)

            result = CounterfactualResult(
                bin_centers,
                actual,
                counterfactual,
                [10.0],
                1,
                (8.0, 12.0),
                0.95,
                20,
                1.0,
            )

            excess_mass, excess_count, h0 = compute_excess_mass(result)

            @test excess_count ≈ 0.0 atol=1e-10
            @test excess_mass ≈ 0.0 atol=1e-10
        end

        @testset "negative bunching (hole)" begin
            bin_centers = collect(1.0:1.0:20.0)
            actual = fill(10.0, 20)
            actual[9:11] .= 5.0  # Deficit at bunching region
            counterfactual = fill(10.0, 20)

            result = CounterfactualResult(
                bin_centers,
                actual,
                counterfactual,
                [10.0],
                1,
                (8.0, 12.0),
                0.95,
                20,
                1.0,
            )

            excess_mass, excess_count, h0 = compute_excess_mass(result)

            @test excess_count < 0
            @test excess_mass < 0
        end
    end

    @testset "compute_elasticity" begin
        @testset "standard calculation" begin
            # e = b / ln((1-t1)/(1-t2))
            excess_mass = 5.0
            t1_rate = 0.25
            t2_rate = 0.35

            elasticity = compute_elasticity(excess_mass, t1_rate, t2_rate)

            expected = 5.0 / log((1 - 0.25) / (1 - 0.35))
            @test elasticity ≈ expected
        end

        @testset "zero excess mass gives zero elasticity" begin
            @test compute_elasticity(0.0, 0.25, 0.35) ≈ 0.0
        end

        @testset "validation" begin
            # t1 out of range
            @test_throws ArgumentError compute_elasticity(5.0, -0.1, 0.35)
            @test_throws ArgumentError compute_elasticity(5.0, 1.0, 0.35)

            # t2 out of range
            @test_throws ArgumentError compute_elasticity(5.0, 0.25, -0.1)
            @test_throws ArgumentError compute_elasticity(5.0, 0.25, 1.0)

            # t2 <= t1
            @test_throws ArgumentError compute_elasticity(5.0, 0.35, 0.25)
            @test_throws ArgumentError compute_elasticity(5.0, 0.25, 0.25)
        end

        @testset "different tax rates" begin
            # Small rate change
            e1 = compute_elasticity(1.0, 0.20, 0.21)
            # Large rate change
            e2 = compute_elasticity(1.0, 0.20, 0.40)

            # Same excess mass, smaller rate change → higher elasticity
            @test e1 > e2
        end
    end

end
