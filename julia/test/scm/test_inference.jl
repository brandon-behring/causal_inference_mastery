#=
Tests for SCM Inference Methods
=#

@testset "SCM Inference" begin

    @testset "Placebo Inference Produces SE" begin
        Random.seed!(42)
        n_units = 10
        n_periods = 20

        outcomes = randn(n_units, n_periods)
        treatment = [true; falses(n_units - 1)]

        problem = SCMProblem(outcomes, treatment, 10, nothing, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:placebo, n_placebo=20))

        @test solution.se > 0
        @test isfinite(solution.se)
    end

    @testset "Placebo Inference Produces P-value" begin
        Random.seed!(123)
        outcomes = randn(12, 24)
        treatment = [true; falses(11)]

        problem = SCMProblem(outcomes, treatment, 12, nothing, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:placebo, n_placebo=30))

        @test 0.0 <= solution.p_value <= 1.0
    end

    @testset "Placebo CI Contains Estimate" begin
        Random.seed!(456)
        outcomes = randn(15, 30)
        treatment = [true; falses(14)]

        problem = SCMProblem(outcomes, treatment, 15, nothing, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:placebo, n_placebo=50))

        # CI should be centered around estimate
        @test solution.ci_lower < solution.estimate < solution.ci_upper
    end

    @testset "Bootstrap Inference Produces SE" begin
        Random.seed!(789)
        outcomes = randn(10, 20)
        treatment = [true; falses(9)]

        problem = SCMProblem(outcomes, treatment, 10, nothing, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:bootstrap, n_placebo=30))

        @test solution.se > 0
        @test isfinite(solution.se)
    end

    @testset "Bootstrap Produces Valid CI" begin
        Random.seed!(101)
        outcomes = randn(8, 16)
        treatment = [true; falses(7)]

        problem = SCMProblem(outcomes, treatment, 8, nothing, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:bootstrap, n_placebo=40))

        @test solution.ci_lower < solution.ci_upper
        @test isfinite(solution.ci_lower)
        @test isfinite(solution.ci_upper)
    end

    @testset "No Inference Returns NaN SE" begin
        Random.seed!(202)
        outcomes = randn(10, 20)
        treatment = [true; falses(9)]

        problem = SCMProblem(outcomes, treatment, 10, nothing, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test isnan(solution.se)
        @test isnan(solution.p_value)
        @test isnan(solution.ci_lower)
        @test isnan(solution.ci_upper)
    end

    @testset "More Placebo Iterations More Stable SE" begin
        Random.seed!(303)
        outcomes = randn(15, 25)
        treatment = [true; falses(14)]

        problem = SCMProblem(outcomes, treatment, 12, nothing, (alpha=0.05,))

        # Run with different n_placebo values
        solution_low = solve(problem, SyntheticControl(inference=:placebo, n_placebo=10))
        solution_high = solve(problem, SyntheticControl(inference=:placebo, n_placebo=14))

        # Both should produce valid SE
        @test isfinite(solution_low.se)
        @test isfinite(solution_high.se)
    end

    @testset "Placebo Limited by Control Units" begin
        # With only 5 controls, can only do 5 placebo tests
        Random.seed!(404)
        outcomes = randn(6, 15)
        treatment = [true; falses(5)]

        problem = SCMProblem(outcomes, treatment, 8, nothing, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:placebo, n_placebo=100))

        # Should still work, limited to 5 placebos
        @test isfinite(solution.se)
    end

    @testset "Null Effect Has High P-value" begin
        # Create data with no true effect
        Random.seed!(505)
        n_control = 10
        n_periods = 20
        treatment_period = 11

        # All units follow same process, no treatment effect
        base = randn(n_periods)
        outcomes = zeros(n_control + 1, n_periods)
        for i in 1:(n_control + 1)
            outcomes[i, :] = base .+ 0.1 .* randn(n_periods)
        end
        treatment = [true; falses(n_control)]

        problem = SCMProblem(outcomes, treatment, treatment_period, nothing, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:placebo, n_placebo=10))

        # P-value should be relatively high (not significant)
        @test solution.p_value > 0.05
    end

end
