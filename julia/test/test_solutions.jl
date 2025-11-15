"""
Tests for solution types.
"""

using Test
using CausalEstimators

@testset "RCTSolution Construction" begin
    # Create a problem for testing
    outcomes = [10.0, 12.0, 4.0, 5.0]
    treatment = [true, true, false, false]
    problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))

    @testset "Valid construction" begin
        solution = RCTSolution(
            estimate = 5.0,
            se = 1.0,
            ci_lower = 3.04,
            ci_upper = 6.96,
            n_treated = 2,
            n_control = 2,
            retcode = :Success,
            original_problem = problem,
        )

        @test solution.estimate == 5.0
        @test solution.se == 1.0
        @test solution.ci_lower == 3.04
        @test solution.ci_upper == 6.96
        @test solution.n_treated == 2
        @test solution.n_control == 2
        @test solution.retcode == :Success
        @test solution.original_problem === problem
    end

    @testset "Invalid retcode fails" begin
        @test_throws ArgumentError RCTSolution(
            estimate = 5.0,
            se = 1.0,
            ci_lower = 3.04,
            ci_upper = 6.96,
            n_treated = 2,
            n_control = 2,
            retcode = :InvalidCode,  # Invalid
            original_problem = problem,
        )
    end

    @testset "Negative sample sizes fail" begin
        @test_throws ArgumentError RCTSolution(
            estimate = 5.0,
            se = 1.0,
            ci_lower = 3.04,
            ci_upper = 6.96,
            n_treated = -1,  # Invalid
            n_control = 2,
            retcode = :Success,
            original_problem = problem,
        )
    end
end

@testset "RCTSolution Display" begin
    outcomes = [10.0, 12.0, 4.0, 5.0]
    treatment = [true, true, false, false]
    problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))

    solution = RCTSolution(
        estimate = 5.0,
        se = 1.0,
        ci_lower = 3.04,
        ci_upper = 6.96,
        n_treated = 2,
        n_control = 2,
        retcode = :Success,
        original_problem = problem,
    )

    # Test that show() doesn't error
    io = IOBuffer()
    show(io, solution)
    output = String(take!(io))

    @test occursin("RCTSolution", output)
    @test occursin("Success", output)
    @test occursin("5.0", output)  # estimate
    @test occursin("95% CI", output)  # CI level
end
