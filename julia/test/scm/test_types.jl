#=
Tests for SCM Type Definitions
=#

@testset "SCM Types" begin

    @testset "SCMProblem Construction" begin
        # Valid problem
        outcomes = randn(10, 20)  # 10 units, 20 periods
        treatment = [true; falses(9)]
        treatment_period = 10

        problem = SCMProblem(outcomes, treatment, treatment_period, nothing, (alpha=0.05,))

        @test problem.outcomes == outcomes
        @test problem.treatment == treatment
        @test problem.treatment_period == treatment_period
        @test problem.covariates === nothing
        @test problem.parameters.alpha == 0.05
    end

    @testset "SCMProblem with Covariates" begin
        outcomes = randn(8, 15)
        treatment = [true, true, falses(6)...]
        treatment_period = 8
        covariates = randn(8, 3)

        problem = SCMProblem(outcomes, treatment, treatment_period, covariates, (alpha=0.10,))

        @test problem.covariates !== nothing
        @test size(problem.covariates) == (8, 3)
    end

    @testset "SCMProblem Validation - Dimension Mismatch" begin
        outcomes = randn(10, 20)
        treatment = [true; falses(5)]  # Wrong length

        @test_throws ArgumentError SCMProblem(outcomes, treatment, 10, nothing, (alpha=0.05,))
    end

    @testset "SCMProblem Validation - No Treated Units" begin
        outcomes = randn(10, 20)
        treatment = Vector{Bool}(falses(10))  # No treated

        @test_throws ArgumentError SCMProblem(outcomes, treatment, 10, nothing, (alpha=0.05,))
    end

    @testset "SCMProblem Validation - No Control Units" begin
        outcomes = randn(10, 20)
        treatment = Vector{Bool}(trues(10))  # No controls

        @test_throws ArgumentError SCMProblem(outcomes, treatment, 10, nothing, (alpha=0.05,))
    end

    @testset "SCMProblem Validation - Invalid Treatment Period" begin
        outcomes = randn(10, 20)
        treatment = [true; falses(9)]

        # Too early
        @test_throws ArgumentError SCMProblem(outcomes, treatment, 1, nothing, (alpha=0.05,))

        # Too late
        @test_throws ArgumentError SCMProblem(outcomes, treatment, 21, nothing, (alpha=0.05,))
    end

    @testset "SCMProblem Validation - NaN in Outcomes" begin
        outcomes = randn(10, 20)
        outcomes[1, 1] = NaN
        treatment = [true; falses(9)]

        @test_throws ArgumentError SCMProblem(outcomes, treatment, 10, nothing, (alpha=0.05,))
    end

    @testset "SyntheticControl Constructor Defaults" begin
        estimator = SyntheticControl()

        @test estimator.inference == :placebo
        @test estimator.n_placebo == 100
        @test estimator.covariate_weight == 1.0
    end

    @testset "SyntheticControl Custom Options" begin
        estimator = SyntheticControl(inference=:bootstrap, n_placebo=200, covariate_weight=2.0)

        @test estimator.inference == :bootstrap
        @test estimator.n_placebo == 200
        @test estimator.covariate_weight == 2.0
    end

    @testset "SyntheticControl Invalid Inference" begin
        @test_throws ArgumentError SyntheticControl(inference=:invalid)
    end

    @testset "SyntheticControl Invalid n_placebo" begin
        @test_throws ArgumentError SyntheticControl(n_placebo=0)
        @test_throws ArgumentError SyntheticControl(n_placebo=-5)
    end

    @testset "AugmentedSC Constructor Defaults" begin
        estimator = AugmentedSC()

        @test estimator.inference == :jackknife
        @test estimator.lambda === nothing
    end

    @testset "AugmentedSC Custom Lambda" begin
        estimator = AugmentedSC(lambda=1.5)

        @test estimator.lambda == 1.5
    end

    @testset "AugmentedSC Invalid Inference" begin
        @test_throws ArgumentError AugmentedSC(inference=:invalid)
    end

end
