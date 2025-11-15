"""
Tests for StratifiedATE estimator.

Following test-first development with known-answer tests and property-based tests.
"""

using Test
using CausalEstimators
using Random

@testset "StratifiedATE: Known Answer Tests" begin
    @testset "Perfect balance, two strata with different baselines" begin
        # Stratum 1: baseline=10, treatment effect=5
        # Stratum 2: baseline=100, treatment effect=5
        # Both strata have same ATE=5, so overall should be 5
        outcomes = [15.0, 10.0, 105.0, 100.0]
        treatment = [true, false, true, false]
        strata = [1, 1, 2, 2]
        problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.05,))

        solution = solve(problem, StratifiedATE())

        @test solution.estimate ≈ 5.0 atol = 1e-10
        @test solution.n_treated == 2
        @test solution.n_control == 2
        @test solution.retcode == :Success
    end

    @testset "Equal sized strata" begin
        # Two strata: Stratum 1 has ATE=4, Stratum 2 has ATE=6
        # Equal weights (0.5 each), so overall ATE = 0.5*4 + 0.5*6 = 5
        outcomes = [14.0, 10.0, 16.0, 10.0]
        treatment = [true, false, true, false]
        strata = [1, 1, 2, 2]
        problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.05,))

        solution = solve(problem, StratifiedATE())

        @test solution.estimate ≈ 5.0 atol = 1e-10
    end

    @testset "Unequal sized strata" begin
        # Stratum 1 (n=2): ATE=10 (weight=2/6)
        # Stratum 2 (n=4): ATE=5  (weight=4/6)
        # Overall ATE = (2/6)*10 + (4/6)*5 = 20/6 + 20/6 = 40/6 ≈ 6.667
        outcomes = [20.0, 10.0, 15.0, 10.0, 15.0, 10.0]
        treatment = [true, false, true, false, true, false]
        strata = [1, 1, 2, 2, 2, 2]
        problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.05,))

        solution = solve(problem, StratifiedATE())

        expected_ate = (2 / 6) * 10 + (4 / 6) * 5
        @test solution.estimate ≈ expected_ate atol = 1e-10
    end

    @testset "Zero treatment effect in both strata" begin
        outcomes = [5.0, 5.0, 10.0, 10.0]
        treatment = [true, false, true, false]
        strata = [1, 1, 2, 2]
        problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.05,))

        solution = solve(problem, StratifiedATE())

        @test solution.estimate ≈ 0.0 atol = 1e-10
    end
end

@testset "StratifiedATE: Error Handling" begin
    @testset "Missing strata throws error" begin
        outcomes = [10.0, 12.0, 4.0, 5.0]
        treatment = [true, true, false, false]
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))

        @test_throws ArgumentError solve(problem, StratifiedATE())
    end

    # Note: Per-stratum variation checks happen in RCTProblem constructor (validation.jl:187-205)
    # So we don't test those here - they're tested in test_problems.jl
end

@testset "StratifiedATE: Statistical Properties" begin
    @testset "Confidence interval contains estimate" begin
        outcomes = [15.0, 10.0, 14.0, 9.0, 25.0, 20.0, 24.0, 19.0]
        treatment = [true, false, true, false, true, false, true, false]
        strata = [1, 1, 1, 1, 2, 2, 2, 2]
        problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.05,))

        solution = solve(problem, StratifiedATE())

        @test solution.ci_lower < solution.estimate
        @test solution.estimate < solution.ci_upper
        @test solution.se > 0
    end

    @testset "Stratification reduces variance vs unstratified" begin
        using Random
        Random.seed!(42)

        # Create data where outcomes vary substantially by stratum
        n_per_stratum = 50
        strata = vcat(fill(1, n_per_stratum), fill(2, n_per_stratum))
        treatment = rand(Bool, 100)

        # Stratum 1: baseline=10, Stratum 2: baseline=100
        # Both have same treatment effect = 5
        outcomes = zeros(100)
        for i in 1:100
            baseline = strata[i] == 1 ? 10.0 : 100.0
            effect = treatment[i] ? 5.0 : 0.0
            outcomes[i] = baseline + effect + randn()
        end

        # Stratified estimator
        problem_strat = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.05,))
        solution_strat = solve(problem_strat, StratifiedATE())

        # Unstratified estimator (ignore strata)
        problem_unstrat = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        solution_unstrat = solve(problem_unstrat, SimpleATE())

        # Stratified should have smaller SE (approximately, due to randomness)
        # This may fail occasionally with random seed changes
        @test solution_strat.se < solution_unstrat.se
    end

    @testset "Different alpha levels" begin
        # Need multiple observations per group to have non-zero variance
        outcomes = [15.0, 14.0, 10.0, 11.0, 105.0, 104.0, 100.0, 101.0]
        treatment = [true, true, false, false, true, true, false, false]
        strata = [1, 1, 1, 1, 2, 2, 2, 2]

        # 95% CI
        problem_95 = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.05,))
        solution_95 = solve(problem_95, StratifiedATE())

        # 99% CI
        problem_99 = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.01,))
        solution_99 = solve(problem_99, StratifiedATE())

        # Same estimate
        @test solution_95.estimate == solution_99.estimate

        # Same SE
        @test solution_95.se == solution_99.se

        # 99% CI should be wider
        ci_width_95 = solution_95.ci_upper - solution_95.ci_lower
        ci_width_99 = solution_99.ci_upper - solution_99.ci_lower
        @test ci_width_99 > ci_width_95
    end
end

@testset "StratifiedATE: Type Stability" begin
    outcomes = [15.0, 10.0, 105.0, 100.0]
    treatment = [true, false, true, false]
    strata = [1, 1, 2, 2]
    problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.05,))

    solution = solve(problem, StratifiedATE())

    # Check all fields have correct types
    @test solution.estimate isa Float64
    @test solution.se isa Float64
    @test solution.ci_lower isa Float64
    @test solution.ci_upper isa Float64
    @test solution.n_treated isa Int
    @test solution.n_control isa Int
    @test solution.retcode isa Symbol
end

@testset "StratifiedATE: Adversarial Tests" begin
    @testset "Stratum with all treated (caught by validation)" begin
        outcomes = [15.0, 14.0, 105.0, 100.0]
        treatment = [true, true, true, false]  # Stratum 1 all treated
        strata = [1, 1, 2, 2]

        # Should fail - stratum 1 has no control units
        @test_throws ArgumentError RCTProblem(
            outcomes,
            treatment,
            nothing,
            strata,
            (alpha = 0.05,),
        )
    end

    @testset "Stratum with all control (caught by validation)" begin
        outcomes = [15.0, 14.0, 105.0, 100.0]
        treatment = [false, false, true, false]  # Stratum 1 all control
        strata = [1, 1, 2, 2]

        # Should fail - stratum 1 has no treated units
        @test_throws ArgumentError RCTProblem(
            outcomes,
            treatment,
            nothing,
            strata,
            (alpha = 0.05,),
        )
    end

    @testset "Zero strata value (invalid)" begin
        outcomes = [15.0, 10.0, 105.0, 100.0]
        treatment = [true, false, true, false]
        strata = [0, 0, 1, 1]  # Stratum 0 is invalid

        @test_throws ArgumentError RCTProblem(
            outcomes,
            treatment,
            nothing,
            strata,
            (alpha = 0.05,),
        )
    end

    @testset "Negative strata value (invalid)" begin
        outcomes = [15.0, 10.0, 105.0, 100.0]
        treatment = [true, false, true, false]
        strata = [-1, -1, 1, 1]  # Negative stratum is invalid

        @test_throws ArgumentError RCTProblem(
            outcomes,
            treatment,
            nothing,
            strata,
            (alpha = 0.05,),
        )
    end

    @testset "Very imbalanced strata sizes" begin
        # Stratum 1: 2 units, Stratum 2: 98 units
        outcomes = vcat([15.0, 10.0], randn(98) .+ 100.0)
        treatment = vcat([true, false], rand(Bool, 98))
        strata = vcat([1, 1], fill(2, 98))

        problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.05,))
        solution = solve(problem, StratifiedATE())

        # Should still compute
        @test solution.retcode == :Success
        @test !isnan(solution.estimate)
        @test !isnan(solution.se)
    end

    @testset "Zero variance within stratum" begin
        # All outcomes identical within each stratum+treatment group
        outcomes = [15.0, 15.0, 10.0, 10.0, 105.0, 105.0, 100.0, 100.0]
        treatment = [true, true, false, false, true, true, false, false]
        strata = [1, 1, 1, 1, 2, 2, 2, 2]

        problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.05,))
        solution = solve(problem, StratifiedATE())

        # Should still compute (SE may be zero or near-zero)
        @test solution.retcode == :Success
        @test !isnan(solution.estimate)
        # SE might be zero if no variance within groups
        @test solution.se >= 0.0
    end

    @testset "Single large stratum (equivalent to SimpleATE)" begin
        # All units in same stratum → should match SimpleATE
        outcomes = [15.0, 14.0, 16.0, 10.0, 11.0, 9.0]
        treatment = [true, true, true, false, false, false]
        strata = fill(1, 6)  # All same stratum

        problem_strat = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.05,))
        solution_strat = solve(problem_strat, StratifiedATE())

        problem_simple = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        solution_simple = solve(problem_simple, SimpleATE())

        # Should match SimpleATE exactly
        @test solution_strat.estimate ≈ solution_simple.estimate
        @test solution_strat.se ≈ solution_simple.se
    end

    @testset "Many strata (n_strata = n/2)" begin
        # Extreme: Half as many strata as observations
        # Each stratum has exactly 2 units (1 treated, 1 control)
        n = 20
        n_strata = 10
        outcomes = randn(n) .+ 50.0
        # Ensure each stratum has 1 treated and 1 control
        treatment = repeat([true, false], n_strata)
        strata = repeat(1:n_strata, inner = 2)

        problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.05,))
        solution = solve(problem, StratifiedATE())

        # Should still compute
        @test solution.retcode == :Success
        @test !isnan(solution.estimate)
        @test !isnan(solution.se)
    end

    @testset "Extreme outlier in single stratum" begin
        # One extreme value in stratum 1
        outcomes = [1e6, 10.0, 105.0, 104.0, 100.0, 101.0]
        treatment = [true, false, true, true, false, false]
        strata = [1, 1, 2, 2, 2, 2]

        problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.05,))
        solution = solve(problem, StratifiedATE())

        # Should still compute (weighted average dampens outlier impact)
        @test solution.retcode == :Success
        @test !isnan(solution.estimate)
        @test !isnan(solution.se)
        @test !isinf(solution.estimate)
        @test !isinf(solution.se)
    end
end
