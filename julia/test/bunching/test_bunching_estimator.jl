"""
Unit tests for bunching estimation (solve function).

Tests solve(BunchingProblem, SaezBunching) and confidence interval functions.
"""

using Test
using Statistics
using Random
using CausalEstimators

@testset "Bunching Estimator" begin

    @testset "solve basic functionality" begin
        Random.seed!(42)
        # Generate data with bunching
        background = randn(800) .* 15 .+ 50
        bunchers = randn(200) .* 2 .+ 50
        data = vcat(background, bunchers)

        problem = BunchingProblem(data, 50.0, 5.0)
        estimator = SaezBunching(n_bins=50, polynomial_order=5, n_bootstrap=50)

        solution = solve(problem, estimator)

        @test isfinite(solution.excess_mass)
        @test isfinite(solution.excess_mass_se)
        @test isfinite(solution.excess_mass_count)
        @test solution.kink_point == 50.0
        @test solution.n_obs == length(data)
        @test solution.n_bootstrap == 50
        @test solution.convergence
    end

    @testset "solve with tax rates" begin
        Random.seed!(42)
        data = vcat(
            randn(800) .* 15 .+ 50,
            randn(200) .* 2 .+ 50
        )

        problem = BunchingProblem(data, 50.0, 5.0; t1_rate=0.25, t2_rate=0.35)
        estimator = SaezBunching(n_bins=50, n_bootstrap=30)

        solution = solve(problem, estimator)

        @test isfinite(solution.elasticity)
        @test isfinite(solution.elasticity_se)
        @test solution.t1_rate == 0.25
        @test solution.t2_rate == 0.35
    end

    @testset "solve without tax rates" begin
        Random.seed!(42)
        data = randn(500) .* 15 .+ 50

        problem = BunchingProblem(data, 50.0, 5.0)  # No rates
        estimator = SaezBunching(n_bins=50, n_bootstrap=30)

        solution = solve(problem, estimator)

        @test isnan(solution.elasticity)
        @test isnothing(solution.t1_rate)
        @test isnothing(solution.t2_rate)
    end

    @testset "detects bunching" begin
        Random.seed!(42)
        # Strong bunching signal
        background = randn(500) .* 20 .+ 50
        bunchers = randn(500) .* 1 .+ 50  # Half the sample bunches!
        data = vcat(background, bunchers)

        problem = BunchingProblem(data, 50.0, 3.0)
        estimator = SaezBunching(n_bins=60, polynomial_order=5, n_bootstrap=30)

        solution = solve(problem, estimator)

        @test solution.excess_mass > 0
        @test solution.excess_mass_count > 0
    end

    @testset "no bunching case" begin
        Random.seed!(42)
        # Uniform data - no bunching
        data = rand(1000) .* 100

        problem = BunchingProblem(data, 50.0, 5.0)
        estimator = SaezBunching(n_bins=50, polynomial_order=3, n_bootstrap=30)

        solution = solve(problem, estimator)

        # Excess mass should be small relative to sample
        @test abs(solution.excess_mass) < 5.0
    end

    @testset "solution contains counterfactual result" begin
        Random.seed!(42)
        data = randn(500) .* 15 .+ 50

        problem = BunchingProblem(data, 50.0, 5.0)
        estimator = SaezBunching(n_bins=40, polynomial_order=5, n_bootstrap=20)

        solution = solve(problem, estimator)

        cf = solution.counterfactual
        @test length(cf.bin_centers) == 40
        @test cf.polynomial_order == 5
        @test cf.n_bins == 40
    end

    @testset "different polynomial orders" begin
        Random.seed!(42)
        # Use wider spread data to ensure enough bins outside bunching
        data = randn(800) .* 20 .+ 50
        problem = BunchingProblem(data, 50.0, 3.0)  # Smaller bunching width

        # Low order
        est_low = SaezBunching(n_bins=60, polynomial_order=3, n_bootstrap=20)
        sol_low = solve(problem, est_low)

        # Medium-high order (7 is standard, avoid too high with limited data)
        est_high = SaezBunching(n_bins=60, polynomial_order=7, n_bootstrap=20)
        sol_high = solve(problem, est_high)

        # Both should complete successfully
        @test isfinite(sol_low.excess_mass)
        @test isfinite(sol_high.excess_mass)
        @test sol_low.counterfactual.polynomial_order == 3
        @test sol_high.counterfactual.polynomial_order == 7
    end

    @testset "display methods" begin
        Random.seed!(42)
        data = randn(500) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0; t1_rate=0.25, t2_rate=0.35)
        estimator = SaezBunching(n_bins=50, n_bootstrap=20)

        solution = solve(problem, estimator)

        # Test short form
        str = repr(solution)
        @test occursin("BunchingSolution", str)

        # Test detailed form
        buf = IOBuffer()
        show(buf, MIME"text/plain"(), solution)
        detailed = String(take!(buf))
        @test occursin("Excess mass", detailed)
        @test occursin("Elasticity", detailed)
    end

end

@testset "Confidence Intervals" begin

    @testset "bunching_confidence_interval" begin
        Random.seed!(42)
        data = randn(500) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0)
        estimator = SaezBunching(n_bins=50, n_bootstrap=50)

        solution = solve(problem, estimator)
        lower, upper = bunching_confidence_interval(solution; level=0.95)

        @test lower < solution.excess_mass < upper
        @test upper - lower > 0
    end

    @testset "different confidence levels" begin
        Random.seed!(42)
        data = randn(500) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0)
        estimator = SaezBunching(n_bins=50, n_bootstrap=50)

        solution = solve(problem, estimator)

        ci_90 = bunching_confidence_interval(solution; level=0.90)
        ci_95 = bunching_confidence_interval(solution; level=0.95)
        ci_99 = bunching_confidence_interval(solution; level=0.99)

        # Wider confidence levels should give wider intervals
        width_90 = ci_90[2] - ci_90[1]
        width_95 = ci_95[2] - ci_95[1]
        width_99 = ci_99[2] - ci_99[1]

        @test width_90 < width_95 < width_99
    end

    @testset "elasticity_confidence_interval with rates" begin
        Random.seed!(42)
        data = randn(500) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0; t1_rate=0.25, t2_rate=0.35)
        estimator = SaezBunching(n_bins=50, n_bootstrap=50)

        solution = solve(problem, estimator)
        lower, upper = elasticity_confidence_interval(solution; level=0.95)

        @test isfinite(lower) && isfinite(upper)
        @test lower < solution.elasticity < upper
    end

    @testset "elasticity_confidence_interval without rates" begin
        Random.seed!(42)
        data = randn(500) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0)  # No rates
        estimator = SaezBunching(n_bins=50, n_bootstrap=30)

        solution = solve(problem, estimator)
        lower, upper = elasticity_confidence_interval(solution)

        @test isnan(lower) && isnan(upper)
    end

    @testset "validation" begin
        Random.seed!(42)
        data = randn(500) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0)
        estimator = SaezBunching(n_bins=50, n_bootstrap=30)

        solution = solve(problem, estimator)

        @test_throws ArgumentError bunching_confidence_interval(solution; level=0.0)
        @test_throws ArgumentError bunching_confidence_interval(solution; level=1.0)
        @test_throws ArgumentError bunching_confidence_interval(solution; level=-0.5)
        @test_throws ArgumentError elasticity_confidence_interval(solution; level=1.5)
    end

end

@testset "Bootstrap Inference" begin

    @testset "standard errors positive" begin
        Random.seed!(42)
        data = vcat(
            randn(700) .* 15 .+ 50,
            randn(300) .* 2 .+ 50
        )

        problem = BunchingProblem(data, 50.0, 5.0; t1_rate=0.25, t2_rate=0.35)
        estimator = SaezBunching(n_bins=50, n_bootstrap=100)

        solution = solve(problem, estimator)

        @test solution.excess_mass_se > 0
        @test solution.elasticity_se > 0
    end

    @testset "more bootstrap iterations" begin
        Random.seed!(42)
        data = randn(500) .* 15 .+ 50
        problem = BunchingProblem(data, 50.0, 5.0)

        # Few bootstrap
        est_few = SaezBunching(n_bins=50, n_bootstrap=30)
        sol_few = solve(problem, est_few)

        # Many bootstrap
        est_many = SaezBunching(n_bins=50, n_bootstrap=200)
        sol_many = solve(problem, est_many)

        # Both should be finite
        @test isfinite(sol_few.excess_mass_se)
        @test isfinite(sol_many.excess_mass_se)

        # Note: More bootstrap doesn't necessarily mean smaller SE
        # It means more stable SE estimate
    end

end
