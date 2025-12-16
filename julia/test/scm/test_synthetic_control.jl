#=
Tests for SyntheticControl Estimator
=#

@testset "SyntheticControl Estimator" begin

    @testset "Known Answer: Perfect Match" begin
        # DGP: Treated unit is weighted combination of controls in pre-period
        # Treatment effect = 5.0 in post-period
        Random.seed!(42)
        n_control = 5
        n_periods = 20
        treatment_period = 11

        # Control outcomes with time trends
        control_outcomes = zeros(n_control, n_periods)
        base_trend = collect(1.0:n_periods) ./ n_periods
        for i in 1:n_control
            control_outcomes[i, :] .= 10.0 .+ i .+ base_trend .+ 0.1 .* randn(n_periods)
        end

        # Treated = uniform average of controls pre-treatment, +5 post-treatment
        treated_outcome = vec(mean(control_outcomes, dims=1))
        treated_outcome[treatment_period:end] .+= 5.0

        outcomes = vcat(treated_outcome', control_outcomes)
        treatment = [true; Vector{Bool}(falses(n_control))]

        problem = SCMProblem(outcomes, treatment, treatment_period, nothing, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        # Should recover treatment effect approximately (some noise in DGP)
        @test isapprox(solution.estimate, 5.0, atol=0.5)
        @test solution.pre_r_squared > 0.90
        @test solution.retcode == :Success
    end

    @testset "Solution Structure Complete" begin
        Random.seed!(123)
        n_units = 8
        n_periods = 15
        treatment_period = 8

        outcomes = randn(n_units, n_periods)
        treatment = [true; falses(n_units - 1)]

        problem = SCMProblem(outcomes, treatment, treatment_period, nothing, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        # Check all fields are populated
        @test isfinite(solution.estimate)
        @test isfinite(solution.pre_rmse)
        @test isfinite(solution.pre_r_squared)
        @test sum(solution.weights) ≈ 1.0 atol=1e-6
        @test all(solution.weights .>= -1e-6)  # Non-negative (small numerical tolerance)
        @test solution.n_treated == 1
        @test solution.n_control == n_units - 1
        @test solution.n_pre_periods == treatment_period - 1
        @test solution.n_post_periods == n_periods - treatment_period + 1
        @test length(solution.synthetic_control) == n_periods
        @test length(solution.treated_series) == n_periods
        @test length(solution.gap) == n_periods
    end

    @testset "Weights Sum to One" begin
        Random.seed!(456)
        outcomes = randn(10, 20)
        treatment = [true; falses(9)]

        problem = SCMProblem(outcomes, treatment, 10, nothing, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test sum(solution.weights) ≈ 1.0 atol=1e-6
    end

    @testset "Weights Non-negative" begin
        Random.seed!(789)
        outcomes = randn(15, 25)
        treatment = [true, true, falses(13)...]

        problem = SCMProblem(outcomes, treatment, 12, nothing, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test all(solution.weights .>= -1e-8)
    end

    @testset "Gap Equals Treated Minus Synthetic" begin
        Random.seed!(101)
        outcomes = randn(8, 16)
        treatment = [true; falses(7)]

        problem = SCMProblem(outcomes, treatment, 8, nothing, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        expected_gap = solution.treated_series .- solution.synthetic_control
        @test maximum(abs.(solution.gap .- expected_gap)) < 1e-10
    end

    @testset "Estimate is Mean Post-Treatment Gap" begin
        Random.seed!(202)
        n_periods = 20
        treatment_period = 11

        outcomes = randn(10, n_periods)
        treatment = [true; falses(9)]

        problem = SCMProblem(outcomes, treatment, treatment_period, nothing, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        expected_estimate = mean(solution.gap[treatment_period:end])
        @test solution.estimate ≈ expected_estimate atol=1e-10
    end

    @testset "Multiple Treated Units Averaged" begin
        Random.seed!(303)
        outcomes = randn(10, 20)
        treatment = [true, true, true, falses(7)...]  # 3 treated units

        problem = SCMProblem(outcomes, treatment, 10, nothing, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test solution.n_treated == 3
        @test solution.n_control == 7
        @test isfinite(solution.estimate)
    end

    @testset "Poor Pre-Fit Warning" begin
        # Create data where treated is very different from controls
        Random.seed!(404)
        n_control = 5
        n_periods = 15

        control_outcomes = randn(n_control, n_periods)
        treated_outcome = 100.0 .+ 10.0 .* randn(n_periods)  # Far from controls

        outcomes = vcat(treated_outcome', control_outcomes)
        treatment = [true; falses(n_control)]

        problem = SCMProblem(outcomes, treatment, 8, nothing, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        # Should still run, might have warning retcode
        @test isfinite(solution.estimate)
        @test solution.pre_r_squared < 0.5  # Poor fit expected
    end

    @testset "Pre-Treatment Fit Metrics" begin
        Random.seed!(505)
        outcomes = randn(8, 16)
        treatment = [true; falses(7)]

        problem = SCMProblem(outcomes, treatment, 10, nothing, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test solution.pre_rmse > 0
        @test 0.0 <= solution.pre_r_squared <= 1.0
    end

    @testset "With Covariates" begin
        Random.seed!(606)
        n_units = 10
        n_periods = 20
        n_covariates = 3

        outcomes = randn(n_units, n_periods)
        treatment = [true; falses(n_units - 1)]
        covariates = randn(n_units, n_covariates)

        problem = SCMProblem(outcomes, treatment, 10, covariates, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none, covariate_weight=0.5))

        @test isfinite(solution.estimate)
        @test sum(solution.weights) ≈ 1.0 atol=1e-6
    end

    @testset "Different Alpha Levels" begin
        Random.seed!(707)
        outcomes = randn(10, 20)
        treatment = [true; falses(9)]

        for alpha in [0.01, 0.05, 0.10]
            problem = SCMProblem(outcomes, treatment, 10, nothing, (alpha=alpha,))
            solution = solve(problem, SyntheticControl(inference=:none))

            @test isfinite(solution.estimate)
        end
    end

end
