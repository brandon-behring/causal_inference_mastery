"""
Tests for PermutationTest estimator.

Following test-first development with known-answer tests and property-based tests.
"""

using Test
using CausalEstimators
using Random

@testset "PermutationTest: Constructor" begin
    @testset "Default constructor (exact test)" begin
        estimator = PermutationTest()
        @test estimator.n_permutations === nothing
        @test estimator.random_seed === nothing
        @test estimator.alternative == "two-sided"
    end

    @testset "Monte Carlo constructor" begin
        estimator = PermutationTest(1000)
        @test estimator.n_permutations == 1000
        @test estimator.random_seed === nothing
        @test estimator.alternative == "two-sided"
    end

    @testset "Full constructor" begin
        estimator = PermutationTest(1000, 42, "greater")
        @test estimator.n_permutations == 1000
        @test estimator.random_seed == 42
        @test estimator.alternative == "greater"
    end

    @testset "Invalid n_permutations throws error" begin
        @test_throws ArgumentError PermutationTest(-1)
        @test_throws ArgumentError PermutationTest(0)
    end

    @testset "Invalid alternative throws error" begin
        @test_throws ArgumentError PermutationTest(1000, 42, "invalid")
    end
end

@testset "PermutationTest: Exact Test (Small Sample)" begin
    @testset "Moderate effect (p = 0.1)" begin
        # Treatment effect = 7, p-value = 0.1 (from golden reference)
        outcomes = [10.0, 12.0, 11.0, 4.0, 5.0, 3.0]
        treatment = [true, true, true, false, false, false]

        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        solution = solve(problem, PermutationTest())

        @test solution.p_value ≈ 0.1
        @test solution.observed_statistic ≈ 7.0
        @test solution.n_permutations == 20  # C(6,3) = 20
        @test solution.alternative == "two-sided"
        @test solution.retcode == :Success
    end

    @testset "No effect (p ≈ 1.0)" begin
        # Zero treatment effect - p-value should be large
        outcomes = [5.0, 6.0, 5.0, 4.0, 6.0, 4.0]
        treatment = [true, true, true, false, false, false]

        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        solution = solve(problem, PermutationTest())

        @test solution.p_value > 0.5
        @test abs(solution.observed_statistic) < 1.0
    end

    @testset "Permutation distribution length correct" begin
        outcomes = [10.0, 12.0, 4.0, 5.0]
        treatment = [true, true, false, false]

        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        solution = solve(problem, PermutationTest())

        # C(4,2) = 6 permutations
        @test length(solution.permutation_distribution) == 6
        @test solution.n_permutations == 6
    end
end

@testset "PermutationTest: Monte Carlo Test" begin
    @testset "Reproducibility with random_seed" begin
        Random.seed!(42)
        outcomes = randn(20) .+ [fill(5.0, 10); fill(0.0, 10)]  # Treatment effect = 5
        treatment = [fill(true, 10); fill(false, 10)]

        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))

        # Two runs with same seed should give identical results
        solution1 = solve(problem, PermutationTest(1000, 42))
        solution2 = solve(problem, PermutationTest(1000, 42))

        @test solution1.p_value == solution2.p_value
        @test solution1.permutation_distribution == solution2.permutation_distribution
    end

    @testset "Large sample Monte Carlo (faster than exact)" begin
        Random.seed!(42)
        n = 100
        outcomes = randn(n) .+ [fill(3.0, 50); fill(0.0, 50)]
        treatment = [fill(true, 50); fill(false, 50)]

        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        solution = solve(problem, PermutationTest(1000, 42))

        @test solution.n_permutations == 1000
        @test length(solution.permutation_distribution) == 1000
        @test solution.p_value < 0.05  # Should detect effect
    end
end

@testset "PermutationTest: Alternative Hypotheses" begin
    @testset "Two-sided (default)" begin
        outcomes = [10.0, 12.0, 4.0, 5.0]
        treatment = [true, true, false, false]

        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        solution = solve(problem, PermutationTest())

        @test solution.alternative == "two-sided"
        # Two-sided uses |stat| >= |observed|
    end

    @testset "Greater (one-sided)" begin
        outcomes = [10.0, 12.0, 4.0, 5.0]
        treatment = [true, true, false, false]

        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        solution = solve(problem, PermutationTest(nothing, nothing, "greater"))

        @test solution.alternative == "greater"
        # Greater uses stat >= observed (one-sided)
    end

    @testset "Less (one-sided)" begin
        outcomes = [4.0, 5.0, 10.0, 12.0]  # Control > Treatment
        treatment = [true, true, false, false]

        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        solution = solve(problem, PermutationTest(nothing, nothing, "less"))

        @test solution.alternative == "less"
        @test solution.observed_statistic < 0  # Negative effect
    end
end

@testset "PermutationTest: Statistical Properties" begin
    @testset "P-value in [0, 1]" begin
        Random.seed!(42)
        outcomes = randn(10)
        treatment = rand(Bool, 10)

        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        solution = solve(problem, PermutationTest(100, 42))

        @test 0.0 <= solution.p_value <= 1.0
    end

    @testset "Observed statistic in permutation distribution" begin
        outcomes = [10.0, 12.0, 4.0, 5.0]
        treatment = [true, true, false, false]

        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        solution = solve(problem, PermutationTest())

        # Observed statistic should appear in the permutation distribution
        # (since actual assignment is one permutation)
        @test solution.observed_statistic in solution.permutation_distribution
    end
end

@testset "PermutationTest: Type Stability" begin
    outcomes = [10.0, 12.0, 4.0, 5.0]
    treatment = [true, true, false, false]

    problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
    solution = solve(problem, PermutationTest())

    # Check all fields have correct types
    @test solution.p_value isa Float64
    @test solution.observed_statistic isa Float64
    @test solution.permutation_distribution isa Vector{Float64}
    @test solution.n_permutations isa Int
    @test solution.alternative isa String
    @test solution.n_treated isa Int
    @test solution.n_control isa Int
    @test solution.retcode isa Symbol
end

@testset "PermutationTest: Adversarial Tests" begin
    @testset "All outcomes identical (zero variance)" begin
        # No variation in outcomes → test statistic always zero
        outcomes = fill(10.0, 6)
        treatment = [true, true, true, false, false, false]

        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        solution = solve(problem, PermutationTest(100, 42))

        # Should compute: observed statistic = 0, p-value ≈ 1.0
        @test solution.retcode == :Success
        @test solution.observed_statistic ≈ 0.0
        # All permutations give same statistic → p-value should be 1.0
        @test solution.p_value ≈ 1.0 atol = 0.05
    end

    @testset "Very small sample (n=4)" begin
        # Minimal sample for permutation test
        outcomes = [10.0, 12.0, 4.0, 5.0]
        treatment = [true, true, false, false]

        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        
        # Exact test has only C(4,2) = 6 permutations
        solution = solve(problem, PermutationTest(6, 42))

        @test solution.retcode == :Success
        @test solution.n_permutations == 6
        @test 0.0 <= solution.p_value <= 1.0
    end

    @testset "Extreme outlier" begin
        outcomes = [1e10, 12.0, 4.0, 5.0]
        treatment = [true, true, false, false]

        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        solution = solve(problem, PermutationTest(100, 42))

        # Should handle outlier
        @test solution.retcode == :Success
        @test !isnan(solution.observed_statistic)
        @test !isinf(solution.observed_statistic)
    end
end
