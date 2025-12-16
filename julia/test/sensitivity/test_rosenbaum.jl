#=
Rosenbaum Bounds Sensitivity Analysis Tests

Test coverage:
- Layer 1: Known-answer tests (monotonicity, p-value properties)
- Layer 2: Robustness property tests
- Layer 3: Edge case tests
- Layer 4: Input validation tests
- Layer 5: Statistical property tests
- Layer 6: Integration tests
=#

using Test
using CausalEstimators
using Random

# ============================================================================
# Layer 1: Known-Answer Tests
# ============================================================================

@testset "Rosenbaum Known-Answer Tests" begin
    @testset "P-values increase with gamma" begin
        # Strong effect: all treated > control
        treated = [10.0, 12.0, 14.0, 16.0, 18.0]
        control = [5.0, 6.0, 7.0, 8.0, 9.0]

        problem = RosenbaumProblem(treated, control; gamma_range=(1.0, 3.0), n_gamma=10)
        solution = solve(problem, RosenbaumBounds())

        # P_upper should be monotonically non-decreasing with gamma
        for i in 2:length(solution.p_upper)
            @test solution.p_upper[i] >= solution.p_upper[i-1] - 1e-10
        end
    end

    @testset "At gamma=1, p_upper ≈ p_lower" begin
        treated = [10.0, 12.0, 14.0]
        control = [5.0, 6.0, 7.0]

        # Use very small gamma range near 1.0
        problem = RosenbaumProblem(treated, control; gamma_range=(1.0, 1.01), n_gamma=2)
        solution = solve(problem, RosenbaumBounds())

        # At gamma ≈ 1.0, upper and lower bounds should be very close
        @test solution.p_upper[1] ≈ solution.p_lower[1] atol=0.01
    end

    @testset "Perfect effect has low p-value at gamma=1" begin
        # Very strong effect
        treated = [100.0, 100.0, 100.0, 100.0, 100.0]
        control = [1.0, 1.0, 1.0, 1.0, 1.0]

        problem = RosenbaumProblem(treated, control; gamma_range=(1.0, 5.0))
        solution = solve(problem, RosenbaumBounds())

        @test solution.p_lower[1] < 0.05
    end

    @testset "Observed statistic is positive for treated > control" begin
        treated = [10.0, 12.0, 14.0, 16.0]
        control = [5.0, 6.0, 7.0, 8.0]

        problem = RosenbaumProblem(treated, control)
        solution = solve(problem, RosenbaumBounds())

        @test solution.observed_statistic > 0
    end

    @testset "N_pairs reflects non-zero differences" begin
        treated = [10.0, 10.0, 12.0]
        control = [5.0, 10.0, 7.0]  # Middle pair has zero difference

        problem = RosenbaumProblem(treated, control)
        solution = solve(problem, RosenbaumBounds())

        @test solution.n_pairs == 2  # One pair excluded (zero diff)
    end
end

# ============================================================================
# Layer 2: Robustness Property Tests
# ============================================================================

@testset "Rosenbaum Robustness Properties" begin
    @testset "Strong effect → high gamma_critical" begin
        Random.seed!(42)
        # Very strong effect
        treated = 50.0 .+ randn(20) .* 2
        control = 10.0 .+ randn(20) .* 2

        problem = RosenbaumProblem(treated, control; gamma_range=(1.0, 10.0))
        solution = solve(problem, RosenbaumBounds())

        # Strong effect should survive high gamma
        @test isnothing(solution.gamma_critical) || solution.gamma_critical > 3.0
    end

    @testset "Weak effect → low or missing gamma_critical" begin
        Random.seed!(123)
        # Weak effect with high noise
        treated = 10.5 .+ randn(10) .* 5
        control = 10.0 .+ randn(10) .* 5

        problem = RosenbaumProblem(treated, control; gamma_range=(1.0, 3.0))
        solution = solve(problem, RosenbaumBounds())

        # Weak effect should be sensitive
        if !isnothing(solution.gamma_critical)
            @test solution.gamma_critical < 2.0
        end
    end

    @testset "Larger sample → more robust inference" begin
        Random.seed!(456)

        # Small sample
        treated_small = 15.0 .+ randn(5)
        control_small = 10.0 .+ randn(5)
        prob_small = RosenbaumProblem(treated_small, control_small; gamma_range=(1.0, 3.0))
        sol_small = solve(prob_small, RosenbaumBounds())

        # Large sample with same effect
        treated_large = 15.0 .+ randn(50)
        control_large = 10.0 .+ randn(50)
        prob_large = RosenbaumProblem(treated_large, control_large; gamma_range=(1.0, 3.0))
        sol_large = solve(prob_large, RosenbaumBounds())

        # Larger sample should have more evidence (lower p at gamma=1)
        @test sol_large.p_lower[1] < sol_small.p_lower[1] + 0.3
    end
end

# ============================================================================
# Layer 3: Edge Case Tests
# ============================================================================

@testset "Rosenbaum Edge Cases" begin
    @testset "All ties → n_pairs = 0" begin
        treated = [10.0, 10.0, 10.0]
        control = [10.0, 10.0, 10.0]

        problem = RosenbaumProblem(treated, control)
        solution = solve(problem, RosenbaumBounds())

        @test solution.n_pairs == 0
        @test solution.gamma_critical ≈ 1.0 atol=1e-10
    end

    @testset "Single pair" begin
        treated = [10.0]
        control = [5.0]

        problem = RosenbaumProblem(treated, control)
        solution = solve(problem, RosenbaumBounds())

        @test solution.n_pairs == 1
        @test solution.observed_statistic == 1.0  # Rank 1
    end

    @testset "Custom gamma range" begin
        treated = [10.0, 12.0, 14.0]
        control = [5.0, 6.0, 7.0]

        problem = RosenbaumProblem(treated, control; gamma_range=(2.0, 5.0), n_gamma=5)
        solution = solve(problem, RosenbaumBounds())

        @test length(solution.gamma_values) == 5
        @test solution.gamma_values[1] ≈ 2.0 atol=1e-10
        @test solution.gamma_values[end] ≈ 5.0 atol=1e-10
    end

    @testset "Custom alpha" begin
        treated = [10.0, 12.0, 14.0, 16.0, 18.0]
        control = [5.0, 6.0, 7.0, 8.0, 9.0]

        problem = RosenbaumProblem(treated, control; alpha=0.10)
        solution = solve(problem, RosenbaumBounds())

        @test solution.alpha ≈ 0.10 atol=1e-10
    end

    @testset "Interpretation contains robustness assessment" begin
        treated = [10.0, 12.0, 14.0, 16.0]
        control = [5.0, 6.0, 7.0, 8.0]

        problem = RosenbaumProblem(treated, control)
        solution = solve(problem, RosenbaumBounds())

        @test occursin("robust", lowercase(solution.interpretation)) ||
              occursin("sensitive", lowercase(solution.interpretation))
    end
end

# ============================================================================
# Layer 4: Input Validation Tests
# ============================================================================

@testset "Rosenbaum Input Validation" begin
    @testset "Unequal lengths throws" begin
        @test_throws ArgumentError RosenbaumProblem([1.0, 2.0], [1.0])
    end

    @testset "Empty arrays throws" begin
        @test_throws ArgumentError RosenbaumProblem(Float64[], Float64[])
    end

    @testset "Gamma lower < 1 throws" begin
        @test_throws ArgumentError RosenbaumProblem([1.0], [0.5]; gamma_range=(0.5, 2.0))
    end

    @testset "Gamma upper <= lower throws" begin
        @test_throws ArgumentError RosenbaumProblem([1.0], [0.5]; gamma_range=(2.0, 1.5))
    end

    @testset "n_gamma < 2 throws" begin
        @test_throws ArgumentError RosenbaumProblem([1.0], [0.5]; n_gamma=1)
    end

    @testset "Invalid alpha throws" begin
        @test_throws ArgumentError RosenbaumProblem([1.0], [0.5]; alpha=0.0)
        @test_throws ArgumentError RosenbaumProblem([1.0], [0.5]; alpha=1.0)
        @test_throws ArgumentError RosenbaumProblem([1.0], [0.5]; alpha=-0.1)
    end
end

# ============================================================================
# Layer 5: Statistical Property Tests
# ============================================================================

@testset "Rosenbaum Statistical Properties" begin
    @testset "P-values bounded [0, 1]" begin
        Random.seed!(789)
        treated = randn(10) .+ 2.0
        control = randn(10)

        problem = RosenbaumProblem(treated, control)
        solution = solve(problem, RosenbaumBounds())

        @test all(0 .<= solution.p_upper .<= 1)
        @test all(0 .<= solution.p_lower .<= 1)
    end

    @testset "P_lower <= P_upper always" begin
        Random.seed!(111)
        treated = randn(15) .+ 1.0
        control = randn(15)

        problem = RosenbaumProblem(treated, control)
        solution = solve(problem, RosenbaumBounds())

        for i in 1:length(solution.p_upper)
            @test solution.p_lower[i] <= solution.p_upper[i] + 1e-10
        end
    end

    @testset "Gamma grid is evenly spaced" begin
        treated = [10.0, 12.0, 14.0]
        control = [5.0, 6.0, 7.0]

        problem = RosenbaumProblem(treated, control; gamma_range=(1.0, 3.0), n_gamma=5)
        solution = solve(problem, RosenbaumBounds())

        diffs = diff(solution.gamma_values)
        @test all(d ≈ diffs[1] for d in diffs)
    end

    @testset "Wilcoxon statistic computation" begin
        differences = [3.0, -1.0, 2.0]  # Ranks: 2, 1, 3 (by abs value: 1, 2, 3)
        # Sorted by |d|: -1.0 (rank 1), 2.0 (rank 2), 3.0 (rank 3)
        # T+ = sum of ranks for positive = 2 + 3 = 5
        T_plus, ranks, signs = compute_signed_rank_statistic(differences)

        @test T_plus ≈ 5.0 atol=1e-10
        @test length(ranks) == 3
    end
end

# ============================================================================
# Layer 6: Integration Tests
# ============================================================================

@testset "Rosenbaum Integration Tests" begin
    @testset "Typical PSM workflow" begin
        Random.seed!(222)
        # Simulated matched pairs after PSM
        treated = [15.2, 18.7, 12.3, 20.1, 16.5, 14.8, 19.2, 13.9, 17.4, 15.6]
        control = [12.1, 14.3, 11.8, 15.6, 13.2, 12.9, 16.1, 12.4, 14.8, 13.3]

        problem = RosenbaumProblem(treated, control; gamma_range=(1.0, 3.0), alpha=0.05)
        solution = solve(problem, RosenbaumBounds())

        # Verify structure
        @test solution isa RosenbaumSolution
        @test solution.n_pairs == 10
        @test length(solution.gamma_values) == 20
        @test length(solution.p_upper) == 20
        @test !isempty(solution.interpretation)
    end

    @testset "Problem-Estimator-Solution pattern" begin
        treated = [10.0, 12.0, 14.0]
        control = [5.0, 6.0, 7.0]

        problem = RosenbaumProblem(treated, control)
        estimator = RosenbaumBounds()
        solution = solve(problem, estimator)

        @test solution isa RosenbaumSolution
        @test solution.original_problem === problem
    end

    @testset "Deterministic results" begin
        treated = [10.0, 12.0, 14.0, 16.0, 18.0]
        control = [5.0, 6.0, 7.0, 8.0, 9.0]

        problem = RosenbaumProblem(treated, control)
        sol1 = solve(problem, RosenbaumBounds())
        sol2 = solve(problem, RosenbaumBounds())

        @test sol1.gamma_critical == sol2.gamma_critical
        @test sol1.observed_statistic == sol2.observed_statistic
        @test sol1.p_upper ≈ sol2.p_upper atol=1e-15
    end
end
