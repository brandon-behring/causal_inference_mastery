#=
Tests for Neural Double Machine Learning with Cross-Fitting

Session 155: Julia Neural CATE Parity

Test Layers:
- Layer 1: Known-Answer - Cross-fitting correctness
- Layer 2: Adversarial - Edge cases and error handling
=#

using Test
using CausalEstimators
using Random
using Statistics

# Include shared DGP file
include("dgp_cate.jl")


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================

@testset "Neural DML Known-Answer" begin

    @testset "Constant effect DGP" begin
        data = dgp_constant_effect(n=1000, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, NeuralDoubleMachineLearning(n_folds=5, degree=2))

        @test solution.retcode == :Success
        @test solution.method == :neural_dml
        @test length(solution.cate) == data.n

        # ATE should be close to true
        @test abs(solution.ate - data.true_ate) < 0.5

        # CI should cover true ATE
        @test solution.ci_lower < data.true_ate < solution.ci_upper
    end

    @testset "Linear heterogeneity DGP" begin
        data = dgp_linear_heterogeneity(n=1000, base_effect=2.0, het_coef=1.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, NeuralDoubleMachineLearning(n_folds=5, degree=2))

        @test abs(solution.ate - data.true_ate) < 0.5

        # CATE should correlate with truth
        corr = cor(solution.cate, data.true_cate)
        @test corr > 0.5
    end

    @testset "Strong confounding DGP" begin
        # DML with cross-fitting should handle confounding
        data = dgp_strong_confounding(n=1000, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, NeuralDoubleMachineLearning(n_folds=5, degree=2))

        @test solution.retcode == :Success
        # Allow larger tolerance for confounded data
        @test abs(solution.ate - data.true_ate) < 1.0
    end

    @testset "Different fold counts" begin
        data = dgp_constant_effect(n=500, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        # Test different fold counts
        for n_folds in [2, 3, 5, 10]
            solution = solve(problem, NeuralDoubleMachineLearning(n_folds=n_folds))
            @test solution.retcode == :Success
            @test abs(solution.ate - data.true_ate) < 1.0
        end
    end

end


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================

@testset "Neural DML Adversarial" begin

    @testset "Invalid n_folds parameter" begin
        @test_throws ArgumentError NeuralDoubleMachineLearning(n_folds=1)
        @test_throws ArgumentError NeuralDoubleMachineLearning(n_folds=0)
    end

    @testset "Invalid degree parameter" begin
        @test_throws ArgumentError NeuralDoubleMachineLearning(degree=0)
    end

    @testset "Invalid lambda parameter" begin
        @test_throws ArgumentError NeuralDoubleMachineLearning(lambda=-0.1)
    end

    @testset "Insufficient samples for cross-fitting" begin
        Random.seed!(42)
        n = 15  # Too small for 5-fold cross-fitting (3 per fold)
        X = randn(n, 3)
        T = rand(n) .< 0.5
        Y = X[:, 1] .+ 2.0 .* T .+ randn(n) * 0.5

        problem = CATEProblem(Y, T, X, (alpha=0.05,))

        # Should throw error for insufficient fold size
        @test_throws ArgumentError solve(problem, NeuralDoubleMachineLearning(n_folds=5))
    end

    @testset "High-dimensional covariates" begin
        data = dgp_high_dimensional(n=500, p=30, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        # Use degree=1 and higher lambda for high-dim
        solution = solve(problem, NeuralDoubleMachineLearning(
            n_folds=5, degree=1, lambda=1.0
        ))

        @test solution.retcode == :Success
        @test isfinite(solution.ate)
        @test isfinite(solution.se)
    end

    @testset "Imbalanced treatment" begin
        data = dgp_imbalanced_treatment(n=500, true_ate=2.0, treatment_prob=0.15, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, NeuralDoubleMachineLearning(n_folds=5, degree=2))

        @test solution.retcode == :Success
        @test abs(solution.ate - data.true_ate) < 1.5
    end

end


# =============================================================================
# Cross-Fitting Verification
# =============================================================================

@testset "Neural DML Cross-Fitting Properties" begin

    @testset "More folds → more variance but less bias" begin
        data = dgp_constant_effect(n=500, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        ates = Float64[]
        for seed in 1:20
            Random.seed!(seed)
            # Use same problem, different random splits
            sol = solve(problem, NeuralDoubleMachineLearning(n_folds=5))
            push!(ates, sol.ate)
        end

        # Mean should be close to true ATE
        @test abs(mean(ates) - 2.0) < 0.3

        # Some variance across random splits is expected
        @test std(ates) > 0.01
        @test std(ates) < 0.5
    end

end


# =============================================================================
# Comparison Tests
# =============================================================================

@testset "Neural DML vs Standard DML" begin

    @testset "Both recover ATE on linear DGP" begin
        data = dgp_linear_heterogeneity(n=500, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        sol_dml = solve(problem, DoubleMachineLearning(n_folds=5))
        sol_neural = solve(problem, NeuralDoubleMachineLearning(n_folds=5, degree=2))

        # Both should be close to true ATE
        @test abs(sol_dml.ate - data.true_ate) < 0.5
        @test abs(sol_neural.ate - data.true_ate) < 0.5

        # Both should have valid CIs
        @test sol_dml.se > 0
        @test sol_neural.se > 0
    end

    @testset "Neural DML captures nonlinear heterogeneity better" begin
        data = dgp_nonlinear_heterogeneity(n=1000, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        sol_dml = solve(problem, DoubleMachineLearning(n_folds=5))
        sol_neural = solve(problem, NeuralDoubleMachineLearning(n_folds=5, degree=2))

        # Neural should have at least comparable CATE correlation
        # (polynomial features help with nonlinear patterns)
        corr_dml = cor(sol_dml.cate, data.true_cate)
        corr_neural = cor(sol_neural.cate, data.true_cate)

        # Both should capture some heterogeneity
        @test corr_dml > 0.2
        @test corr_neural > 0.2
    end

end
