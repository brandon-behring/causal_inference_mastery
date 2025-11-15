"""
Tests for SimpleATE estimator.

Following test-first development with known-answer tests.
"""

using Test
using CausalEstimators

@testset "SimpleATE: Known Answer Tests" begin
    @testset "Perfect balance, simple numbers" begin
        # Hand-calculated: ATE = (7+5)/2 - (3+1)/2 = 6 - 2 = 4
        outcomes = [7.0, 5.0, 3.0, 1.0]
        treatment = [true, true, false, false]
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))

        solution = solve(problem, SimpleATE())

        @test solution.estimate ≈ 4.0 atol = 1e-10
        @test solution.n_treated == 2
        @test solution.n_control == 2
        @test solution.retcode == :Success
    end

    @testset "Zero treatment effect" begin
        outcomes = [5.0, 5.0, 5.0, 5.0]
        treatment = [true, true, false, false]
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))

        solution = solve(problem, SimpleATE())

        @test solution.estimate ≈ 0.0 atol = 1e-10
        @test solution.se ≈ 0.0 atol = 1e-10  # No variance
    end

    @testset "Negative treatment effect" begin
        # Control group has higher outcomes
        outcomes = [3.0, 4.0, 10.0, 12.0]
        treatment = [true, true, false, false]
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))

        solution = solve(problem, SimpleATE())

        # ATE = 3.5 - 11.0 = -7.5
        @test solution.estimate ≈ -7.5 atol = 1e-10
    end
end

@testset "SimpleATE: Statistical Properties" begin
    @testset "Confidence interval contains estimate" begin
        outcomes = [10.0, 12.0, 11.0, 4.0, 5.0, 3.0]
        treatment = [true, true, true, false, false, false]
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))

        solution = solve(problem, SimpleATE())

        @test solution.ci_lower < solution.estimate
        @test solution.estimate < solution.ci_upper
        @test solution.se > 0
    end

    @testset "Larger sample → smaller SE" begin
        using Random
        Random.seed!(42)

        # Small sample
        treatment_small = rand(Bool, 20)
        outcomes_small = treatment_small .* 5.0 .+ randn(20)
        problem_small = RCTProblem(outcomes_small, treatment_small, nothing, nothing, (alpha = 0.05,))
        solution_small = solve(problem_small, SimpleATE())

        # Large sample
        treatment_large = rand(Bool, 200)
        outcomes_large = treatment_large .* 5.0 .+ randn(200)
        problem_large = RCTProblem(outcomes_large, treatment_large, nothing, nothing, (alpha = 0.05,))
        solution_large = solve(problem_large, SimpleATE())

        # Larger sample should have smaller SE (approximately)
        @test solution_large.se < solution_small.se
    end
end

@testset "SimpleATE: Different Alpha Levels" begin
    outcomes = [10.0, 12.0, 4.0, 5.0]
    treatment = [true, true, false, false]

    # 95% CI (alpha = 0.05)
    problem_95 = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
    solution_95 = solve(problem_95, SimpleATE())

    # 99% CI (alpha = 0.01)
    problem_99 = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.01,))
    solution_99 = solve(problem_99, SimpleATE())

    # Same estimate
    @test solution_95.estimate == solution_99.estimate

    # Same SE
    @test solution_95.se == solution_99.se

    # 99% CI should be wider
    ci_width_95 = solution_95.ci_upper - solution_95.ci_lower
    ci_width_99 = solution_99.ci_upper - solution_99.ci_lower
    @test ci_width_99 > ci_width_95
end

@testset "SimpleATE: Type Stability" begin
    outcomes = [10.0, 12.0, 4.0, 5.0]
    treatment = [true, true, false, false]
    problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))

    solution = solve(problem, SimpleATE())

    # Check all fields have correct types
    @test solution.estimate isa Float64
    @test solution.se isa Float64
    @test solution.ci_lower isa Float64
    @test solution.ci_upper isa Float64
    @test solution.n_treated isa Int
    @test solution.n_control isa Int
    @test solution.retcode isa Symbol
end

@testset "SimpleATE: Adversarial Tests" begin
    @testset "n=1 (single observation)" begin
        # Should fail - cannot estimate variance with n=1
        outcomes = [10.0]
        treatment = [true]

        @test_throws ArgumentError RCTProblem(
            outcomes,
            treatment,
            nothing,
            nothing,
            (alpha = 0.05,),
        )
    end

    @testset "n=2 (undefined variance with single group)" begin
        # All treated → no control group
        outcomes = [10.0, 12.0]
        treatment = [true, true]

        @test_throws ArgumentError RCTProblem(
            outcomes,
            treatment,
            nothing,
            nothing,
            (alpha = 0.05,),
        )
    end

    @testset "All treated (no control group)" begin
        outcomes = [10.0, 12.0, 11.0, 13.0]
        treatment = [true, true, true, true]

        @test_throws ArgumentError RCTProblem(
            outcomes,
            treatment,
            nothing,
            nothing,
            (alpha = 0.05,),
        )
    end

    @testset "All control (no treatment group)" begin
        outcomes = [4.0, 5.0, 3.0, 6.0]
        treatment = [false, false, false, false]

        @test_throws ArgumentError RCTProblem(
            outcomes,
            treatment,
            nothing,
            nothing,
            (alpha = 0.05,),
        )
    end

    @testset "Zero variance within groups" begin
        # All treated outcomes identical, all control outcomes identical
        # This is valid - SE should be zero
        outcomes = [10.0, 10.0, 10.0, 5.0, 5.0, 5.0]
        treatment = [true, true, true, false, false, false]
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))

        solution = solve(problem, SimpleATE())

        @test solution.estimate ≈ 5.0  # 10 - 5 = 5
        @test solution.se ≈ 0.0  # No variance → SE = 0
        @test solution.retcode == :Success
    end

    @testset "Extreme outliers" begin
        # One extreme outlier in treatment group
        outcomes = [1e6, 10.0, 12.0, 4.0, 5.0, 3.0]
        treatment = [true, true, true, false, false, false]
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))

        solution = solve(problem, SimpleATE())

        # Should still compute (though estimate heavily influenced)
        @test solution.retcode == :Success
        @test !isnan(solution.estimate)
        @test !isnan(solution.se)
        @test !isinf(solution.estimate)
        @test !isinf(solution.se)
    end

    @testset "NaN in outcomes" begin
        outcomes = [10.0, NaN, 12.0, 4.0]
        treatment = [true, true, false, false]

        # Should fail during problem construction
        @test_throws ArgumentError RCTProblem(
            outcomes,
            treatment,
            nothing,
            nothing,
            (alpha = 0.05,),
        )
    end

    @testset "Inf in outcomes" begin
        outcomes = [10.0, Inf, 12.0, 4.0]
        treatment = [true, true, false, false]

        # Should fail during problem construction
        @test_throws ArgumentError RCTProblem(
            outcomes,
            treatment,
            nothing,
            nothing,
            (alpha = 0.05,),
        )
    end

    @testset "Mismatched lengths" begin
        outcomes = [10.0, 12.0, 4.0]
        treatment = [true, true]  # Length mismatch

        @test_throws ArgumentError RCTProblem(
            outcomes,
            treatment,
            nothing,
            nothing,
            (alpha = 0.05,),
        )
    end

    @testset "Empty arrays" begin
        outcomes = Float64[]
        treatment = Bool[]

        @test_throws ArgumentError RCTProblem(
            outcomes,
            treatment,
            nothing,
            nothing,
            (alpha = 0.05,),
        )
    end
end
