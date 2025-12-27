#=
Tests for Neural Network-Style CATE Meta-Learners

Session 155: Julia Neural CATE Parity

Test Layers:
- Layer 1: Known-Answer - Linear CATE DGP with known ground truth
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

@testset "Neural S-Learner Known-Answer" begin

    @testset "Constant effect DGP" begin
        data = dgp_constant_effect(n=1000, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, NeuralSLearner(degree=2, lambda=0.01))

        @test solution.retcode == :Success
        @test solution.method == :neural_s_learner
        @test length(solution.cate) == data.n

        # ATE should be close to true (bias < 0.5)
        @test abs(solution.ate - data.true_ate) < 0.5

        # CI should cover true ATE
        @test solution.ci_lower < data.true_ate < solution.ci_upper
    end

    @testset "Linear heterogeneity DGP" begin
        data = dgp_linear_heterogeneity(n=1000, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, NeuralSLearner(degree=2, lambda=0.01))

        # ATE should be reasonably close
        @test abs(solution.ate - data.true_ate) < 0.5

        # Note: S-Learner gives constant CATE by design when treatment enters linearly
        # The CATE is just the coefficient on T, so CATE variance is near zero
        # This is expected behavior, not a bug
        @test std(solution.cate) < 0.1  # Constant CATE
    end

end


@testset "Neural T-Learner Known-Answer" begin

    @testset "Constant effect DGP" begin
        data = dgp_constant_effect(n=1000, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, NeuralTLearner(degree=2, lambda=0.01))

        @test solution.retcode == :Success
        @test solution.method == :neural_t_learner
        @test abs(solution.ate - data.true_ate) < 0.5
    end

    @testset "Linear heterogeneity captures trend" begin
        data = dgp_linear_heterogeneity(n=1000, base_effect=2.0, het_coef=1.5, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, NeuralTLearner(degree=2, lambda=0.01))

        # CATE should correlate with truth
        corr = cor(solution.cate, data.true_cate)
        @test corr > 0.5
    end

end


@testset "Neural X-Learner Known-Answer" begin

    @testset "Constant effect DGP" begin
        data = dgp_constant_effect(n=1000, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, NeuralXLearner(degree=2, lambda=0.01))

        @test solution.retcode == :Success
        @test solution.method == :neural_x_learner
        @test abs(solution.ate - data.true_ate) < 0.5
    end

    @testset "Imbalanced treatment DGP" begin
        # X-learner should handle imbalanced treatment well
        data = dgp_imbalanced_treatment(n=1000, true_ate=2.0, treatment_prob=0.2, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, NeuralXLearner(degree=2, lambda=0.01))

        @test solution.retcode == :Success
        @test abs(solution.ate - data.true_ate) < 1.0
    end

end


@testset "Neural R-Learner Known-Answer" begin

    @testset "Constant effect DGP" begin
        data = dgp_constant_effect(n=1000, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, NeuralRLearner(degree=2, lambda=0.01))

        @test solution.retcode == :Success
        @test solution.method == :neural_r_learner
        @test abs(solution.ate - data.true_ate) < 0.5
    end

    @testset "Strong confounding DGP" begin
        # R-learner should handle confounding via Robinson transformation
        data = dgp_strong_confounding(n=1000, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, NeuralRLearner(degree=2, lambda=0.01))

        @test solution.retcode == :Success
        # Allow larger bias with strong confounding
        @test abs(solution.ate - data.true_ate) < 1.0
    end

    @testset "Linear heterogeneity recovery" begin
        data = dgp_linear_heterogeneity(n=1000, base_effect=2.0, het_coef=1.5, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, NeuralRLearner(degree=2, lambda=0.01))

        # CATE correlation
        corr = cor(solution.cate, data.true_cate)
        @test corr > 0.5
    end

end


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================

@testset "Neural Meta-Learners Adversarial" begin

    @testset "Invalid degree parameter" begin
        @test_throws ArgumentError NeuralSLearner(degree=0)
        @test_throws ArgumentError NeuralTLearner(degree=-1)
    end

    @testset "Invalid lambda parameter" begin
        @test_throws ArgumentError NeuralSLearner(lambda=-0.1)
        @test_throws ArgumentError NeuralRLearner(lambda=-1.0)
    end

    @testset "Small sample size" begin
        # With degree=2, polynomial features grow quadratically
        # Small n should still work with ridge regularization
        Random.seed!(42)
        n = 50
        X = randn(n, 3)
        T = rand(n) .< 0.5
        Y = X[:, 1] .+ 2.0 .* T .+ randn(n) * 0.5

        problem = CATEProblem(Y, T, X, (alpha=0.05,))

        # All estimators should handle small samples
        for estimator in [NeuralSLearner(), NeuralTLearner(), NeuralXLearner(), NeuralRLearner()]
            solution = solve(problem, estimator)
            @test solution.retcode == :Success
            @test isfinite(solution.ate)
            @test isfinite(solution.se)
        end
    end

    @testset "High-dimensional covariates" begin
        data = dgp_high_dimensional(n=500, p=30, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        # Should handle high-dim with regularization
        solution = solve(problem, NeuralRLearner(degree=1, lambda=1.0))
        @test solution.retcode == :Success
        @test isfinite(solution.ate)
    end

    @testset "Degree=1 gives linear features" begin
        data = dgp_constant_effect(n=500, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        # Degree=1 should still work (no polynomial expansion)
        solution = solve(problem, NeuralSLearner(degree=1, lambda=0.01))
        @test solution.retcode == :Success
        @test abs(solution.ate - data.true_ate) < 1.0
    end

    @testset "Insufficient samples per treatment group" begin
        Random.seed!(42)
        n = 20
        X = randn(n, 3)
        T = vcat(trues(19), falses(1))  # Very imbalanced: 19 treated, 1 control
        Y = X[:, 1] .+ 2.0 .* T .+ randn(n) * 0.5

        problem = CATEProblem(Y, T, X, (alpha=0.05,))

        # T-learner and X-learner should fail with < 2 in one group
        @test_throws ArgumentError solve(problem, NeuralTLearner())
        @test_throws ArgumentError solve(problem, NeuralXLearner())
    end

end


# =============================================================================
# Comparison Tests
# =============================================================================

@testset "Neural vs Standard Meta-Learners" begin

    @testset "Neural R-Learner captures heterogeneity" begin
        data = dgp_linear_heterogeneity(n=500, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        # Standard R-learner
        sol_r = solve(problem, RLearner())

        # Neural R-learner
        sol_nr = solve(problem, NeuralRLearner(degree=2))

        # Both should recover reasonable ATE
        @test abs(sol_r.ate - data.true_ate) < 1.0
        @test abs(sol_nr.ate - data.true_ate) < 1.0

        # R-learner should capture heterogeneity (unlike S-learner)
        corr_r = cor(sol_r.cate, data.true_cate)
        corr_nr = cor(sol_nr.cate, data.true_cate)

        @test corr_r > 0.3
        @test corr_nr > 0.3
    end

    @testset "Neural T-Learner captures heterogeneity" begin
        data = dgp_linear_heterogeneity(n=500, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        sol_t = solve(problem, TLearner())
        sol_nt = solve(problem, NeuralTLearner(degree=2))

        # Both should recover reasonable ATE
        @test abs(sol_t.ate - data.true_ate) < 1.0
        @test abs(sol_nt.ate - data.true_ate) < 1.0

        # T-learner should capture heterogeneity
        corr_t = cor(sol_t.cate, data.true_cate)
        corr_nt = cor(sol_nt.cate, data.true_cate)

        @test corr_t > 0.3
        @test corr_nt > 0.3
    end

end
