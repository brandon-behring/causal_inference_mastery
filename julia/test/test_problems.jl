"""
Tests for problem construction and validation.
"""

using Test
using CausalEstimators

@testset "RCTProblem Construction" begin
    @testset "Valid construction" begin
        outcomes = [10.0, 12.0, 4.0, 5.0]
        treatment = [true, true, false, false]

        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))

        @test problem.outcomes == outcomes
        @test problem.treatment == treatment
        @test isnothing(problem.covariates)
        @test isnothing(problem.strata)
        @test problem.parameters.alpha == 0.05
    end

    @testset "Empty arrays fail fast" begin
        @test_throws ArgumentError RCTProblem(Float64[], Bool[], nothing, nothing, (alpha = 0.05,))
    end

    @testset "Mismatched lengths fail fast" begin
        outcomes = [10.0, 12.0]
        treatment = [true, true, false, false]
        @test_throws ArgumentError RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
    end

    @testset "NaN in outcomes fails fast" begin
        outcomes = [10.0, NaN, 4.0, 5.0]
        treatment = [true, true, false, false]
        @test_throws ArgumentError RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
    end

    @testset "No treatment variation fails fast" begin
        outcomes = [10.0, 12.0, 4.0, 5.0]

        # All treated
        treatment_all_1 = [true, true, true, true]
        @test_throws ArgumentError RCTProblem(outcomes, treatment_all_1, nothing, nothing, (alpha = 0.05,))

        # All control
        treatment_all_0 = [false, false, false, false]
        @test_throws ArgumentError RCTProblem(outcomes, treatment_all_0, nothing, nothing, (alpha = 0.05,))
    end

    @testset "Covariate matrix validation" begin
        outcomes = [10.0, 12.0, 4.0, 5.0]
        treatment = [true, true, false, false]

        # Valid covariates
        X = [5.0 2.0; 6.0 3.0; 5.5 2.5; 4.5 2.0]
        problem = RCTProblem(outcomes, treatment, X, nothing, (alpha = 0.05,))
        @test size(problem.covariates) == (4, 2)

        # Wrong number of rows
        X_bad = [5.0 2.0; 6.0 3.0]
        @test_throws ArgumentError RCTProblem(outcomes, treatment, X_bad, nothing, (alpha = 0.05,))

        # Empty columns
        X_empty = Matrix{Float64}(undef, 4, 0)
        @test_throws ArgumentError RCTProblem(outcomes, treatment, X_empty, nothing, (alpha = 0.05,))

        # NaN in covariates
        X_nan = [5.0 NaN; 6.0 3.0; 5.5 2.5; 4.5 2.0]
        @test_throws ArgumentError RCTProblem(outcomes, treatment, X_nan, nothing, (alpha = 0.05,))
    end

    @testset "Strata validation" begin
        outcomes = [10.0, 12.0, 4.0, 5.0]
        treatment = [true, false, true, false]

        # Valid strata
        strata = [1, 1, 2, 2]
        problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.05,))
        @test problem.strata == strata

        # Wrong length
        strata_bad = [1, 1]
        @test_throws ArgumentError RCTProblem(outcomes, treatment, nothing, strata_bad, (alpha = 0.05,))

        # Invalid strata values (zero or negative)
        strata_zero = [1, 1, 0, 0]
        @test_throws ArgumentError RCTProblem(outcomes, treatment, nothing, strata_zero, (alpha = 0.05,))

        # Stratum lacks variation
        treatment_no_var = [true, true, false, false]
        strata_no_var = [1, 1, 2, 2]
        @test_throws ArgumentError RCTProblem(outcomes, treatment_no_var, nothing, strata_no_var, (alpha = 0.05,))
    end
end

@testset "remake()" begin
    outcomes = [10.0, 12.0, 4.0, 5.0]
    treatment = [true, true, false, false]
    original = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))

    @testset "Change alpha" begin
        new_problem = remake(original, parameters = (alpha = 0.01,))
        @test new_problem.parameters.alpha == 0.01
        @test new_problem.outcomes == original.outcomes
        @test new_problem.treatment == original.treatment
    end

    @testset "Add covariates" begin
        X = reshape([5.0; 6.0; 5.5; 4.5], 4, 1)  # Matrix, not vector
        new_problem = remake(original, covariates = X)
        @test new_problem.covariates == X
        @test new_problem.parameters == original.parameters
    end
end
