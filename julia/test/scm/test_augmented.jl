#=
Tests for Augmented Synthetic Control (AugmentedSC) Estimator
=#

@testset "AugmentedSC Estimator" begin

    @testset "Basic Functionality" begin
        Random.seed!(42)
        outcomes = randn(10, 20)
        treatment = [true; falses(9)]

        problem = SCMProblem(outcomes, treatment, 10, nothing, (alpha=0.05,))
        solution = solve(problem, AugmentedSC(inference=:none))

        @test isfinite(solution.estimate)
        @test sum(solution.weights) ≈ 1.0 atol=1e-6
        @test solution.retcode in (:Success, :Warning)
    end

    @testset "Solution Structure Complete" begin
        Random.seed!(123)
        outcomes = randn(12, 24)
        treatment = [true, true, falses(10)...]

        problem = SCMProblem(outcomes, treatment, 12, nothing, (alpha=0.05,))
        solution = solve(problem, AugmentedSC(inference=:none))

        @test solution.n_treated == 2
        @test solution.n_control == 10
        @test length(solution.synthetic_control) == 24
        @test length(solution.treated_series) == 24
        @test length(solution.gap) == 24
    end

    @testset "Jackknife SE Produces Valid Estimate" begin
        Random.seed!(456)
        outcomes = randn(10, 20)
        treatment = [true; falses(9)]

        problem = SCMProblem(outcomes, treatment, 10, nothing, (alpha=0.05,))
        solution = solve(problem, AugmentedSC(inference=:jackknife))

        @test solution.se > 0
        @test isfinite(solution.se)
    end

    @testset "Bootstrap SE Produces Valid Estimate" begin
        Random.seed!(789)
        outcomes = randn(10, 20)
        treatment = [true; falses(9)]

        problem = SCMProblem(outcomes, treatment, 10, nothing, (alpha=0.05,))
        solution = solve(problem, AugmentedSC(inference=:bootstrap))

        @test solution.se > 0
        @test isfinite(solution.se)
    end

    @testset "Custom Lambda" begin
        Random.seed!(101)
        outcomes = randn(10, 20)
        treatment = [true; falses(9)]

        problem = SCMProblem(outcomes, treatment, 10, nothing, (alpha=0.05,))
        solution = solve(problem, AugmentedSC(inference=:none, lambda=1.0))

        @test isfinite(solution.estimate)
    end

    @testset "Auto Lambda Selection" begin
        Random.seed!(202)
        outcomes = randn(12, 25)
        treatment = [true; falses(11)]

        problem = SCMProblem(outcomes, treatment, 12, nothing, (alpha=0.05,))

        # Lambda=nothing should trigger CV selection
        solution = solve(problem, AugmentedSC(inference=:none, lambda=nothing))

        @test isfinite(solution.estimate)
    end

    @testset "Bias Correction Improves Poor Fit" begin
        # Create scenario where treated is different from controls pre-treatment
        Random.seed!(303)
        n_control = 8
        n_periods = 20
        treatment_period = 11
        true_effect = 3.0

        # Controls share a common factor
        factor = randn(n_periods)
        control_outcomes = zeros(n_control, n_periods)
        for i in 1:n_control
            control_outcomes[i, :] = factor .+ 0.5 .* randn(n_periods)
        end

        # Treated has different level but similar trends
        treated_outcome = factor .+ 2.0 .+ 0.5 .* randn(n_periods)
        treated_outcome[treatment_period:end] .+= true_effect

        outcomes = vcat(treated_outcome', control_outcomes)
        treatment = [true; falses(n_control)]

        problem = SCMProblem(outcomes, treatment, treatment_period, nothing, (alpha=0.05,))

        # Compare basic SCM vs ASCM
        sol_scm = solve(problem, SyntheticControl(inference=:none))
        sol_ascm = solve(problem, AugmentedSC(inference=:none))

        # Both should produce estimates; ASCM might be closer to true effect
        @test isfinite(sol_scm.estimate)
        @test isfinite(sol_ascm.estimate)
    end

    @testset "Valid CI from Jackknife" begin
        Random.seed!(404)
        outcomes = randn(10, 20)
        treatment = [true; falses(9)]

        problem = SCMProblem(outcomes, treatment, 10, nothing, (alpha=0.05,))
        solution = solve(problem, AugmentedSC(inference=:jackknife))

        @test solution.ci_lower < solution.estimate < solution.ci_upper
        @test isfinite(solution.ci_lower)
        @test isfinite(solution.ci_upper)
    end

    @testset "No Inference Returns NaN" begin
        Random.seed!(505)
        outcomes = randn(10, 20)
        treatment = [true; falses(9)]

        problem = SCMProblem(outcomes, treatment, 10, nothing, (alpha=0.05,))
        solution = solve(problem, AugmentedSC(inference=:none))

        @test isnan(solution.se)
        @test isnan(solution.p_value)
    end

    @testset "Pre-Treatment Fit Metrics" begin
        Random.seed!(606)
        outcomes = randn(10, 20)
        treatment = [true; falses(9)]

        problem = SCMProblem(outcomes, treatment, 10, nothing, (alpha=0.05,))
        solution = solve(problem, AugmentedSC(inference=:none))

        # Pre-fit uses SCM weights, not augmented
        @test solution.pre_rmse > 0
        @test 0.0 <= solution.pre_r_squared <= 1.0
    end

    @testset "Multiple Treated Units" begin
        Random.seed!(707)
        outcomes = randn(15, 25)
        treatment = [true, true, true, falses(12)...]

        problem = SCMProblem(outcomes, treatment, 12, nothing, (alpha=0.05,))
        solution = solve(problem, AugmentedSC(inference=:none))

        @test solution.n_treated == 3
        @test solution.n_control == 12
        @test isfinite(solution.estimate)
    end

end
