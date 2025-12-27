#=
Tests for Latent CATE Methods

Session 156: Latent CATE Julia Parity

Test Layers:
- Layer 1: Known-Answer - Constant effect DGPs with known ground truth
- Layer 2: Adversarial - Edge cases and error handling
=#

using Test
using CausalEstimators
using Random
using Statistics

# Include shared DGP file
include("dgp_cate.jl")


# =============================================================================
# Layer 1: Known-Answer Tests - Factor Analysis CATE
# =============================================================================

@testset "Factor Analysis CATE Known-Answer" begin

    @testset "Constant effect DGP" begin
        data = dgp_constant_effect(n=1000, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, FactorAnalysisCATEEstimator(n_latent=3))

        @test solution.retcode == :Success
        @test solution.method == :factor_analysis_cate
        @test length(solution.cate) == data.n

        # ATE should be close to true (bias < 0.5)
        @test abs(solution.ate - data.true_ate) < 0.5

        # CI should cover true ATE
        @test solution.ci_lower < data.true_ate < solution.ci_upper
    end

    @testset "Linear heterogeneity DGP" begin
        data = dgp_linear_heterogeneity(n=1000, base_effect=2.0, het_coef=1.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, FactorAnalysisCATEEstimator(n_latent=3))

        # ATE should be reasonably close
        @test abs(solution.ate - data.true_ate) < 0.5

        # CATE should have some variation (FA captures heterogeneity)
        @test std(solution.cate) > 0.01
    end

    @testset "With R-learner base" begin
        data = dgp_constant_effect(n=500, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, FactorAnalysisCATEEstimator(n_latent=2, base_learner=:r_learner))

        @test solution.retcode == :Success
        @test solution.method == :factor_analysis_cate
        @test abs(solution.ate - data.true_ate) < 1.0
    end

end


# =============================================================================
# Layer 1: Known-Answer Tests - PPCA CATE
# =============================================================================

@testset "PPCA CATE Known-Answer" begin

    @testset "Constant effect DGP" begin
        data = dgp_constant_effect(n=1000, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, PPCACATEEstimator(n_components=3))

        @test solution.retcode == :Success
        @test solution.method == :ppca_cate
        @test length(solution.cate) == data.n

        # ATE should be close to true
        @test abs(solution.ate - data.true_ate) < 0.5
    end

    @testset "Linear heterogeneity DGP" begin
        data = dgp_linear_heterogeneity(n=1000, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, PPCACATEEstimator(n_components=3))

        @test abs(solution.ate - data.true_ate) < 0.5
    end

    @testset "Dimensionality reduction verification" begin
        # PCA should work with fewer components than features
        data = dgp_constant_effect(n=500, true_ate=2.0, p=10, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, PPCACATEEstimator(n_components=5))

        @test solution.retcode == :Success
        @test isfinite(solution.ate)
    end

    @testset "With R-learner base" begin
        data = dgp_constant_effect(n=500, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, PPCACATEEstimator(n_components=2, base_learner=:r_learner))

        @test solution.retcode == :Success
        @test abs(solution.ate - data.true_ate) < 1.0
    end

end


# =============================================================================
# Layer 1: Known-Answer Tests - GMM Stratified CATE
# =============================================================================

@testset "GMM Stratified CATE Known-Answer" begin

    @testset "Constant effect DGP" begin
        data = dgp_constant_effect(n=1000, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, GMMStratifiedCATEEstimator(n_strata=3))

        @test solution.retcode == :Success
        @test solution.method == :gmm_stratified_cate
        @test length(solution.cate) == data.n

        # ATE should be close to true (allow more tolerance for stratification)
        @test abs(solution.ate - data.true_ate) < 1.0
    end

    @testset "Linear heterogeneity DGP" begin
        data = dgp_linear_heterogeneity(n=1000, base_effect=2.0, het_coef=1.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, GMMStratifiedCATEEstimator(n_strata=3))

        # ATE should be reasonably close
        @test abs(solution.ate - data.true_ate) < 1.0

        # Different strata may have different CATEs
        @test std(solution.cate) > 0.01
    end

    @testset "Different strata counts" begin
        data = dgp_constant_effect(n=500, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        for n_strata in [2, 3, 5]
            solution = solve(problem, GMMStratifiedCATEEstimator(n_strata=n_strata))
            @test solution.retcode == :Success
            @test abs(solution.ate - data.true_ate) < 1.5
        end
    end

    @testset "With R-learner base" begin
        data = dgp_constant_effect(n=500, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, GMMStratifiedCATEEstimator(n_strata=3, base_learner=:r_learner))

        @test solution.retcode == :Success
        @test abs(solution.ate - data.true_ate) < 1.5
    end

end


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================

@testset "Latent CATE Adversarial" begin

    @testset "Invalid n_latent parameter" begin
        @test_throws ArgumentError FactorAnalysisCATEEstimator(n_latent=0)
        @test_throws ArgumentError FactorAnalysisCATEEstimator(n_latent=-1)
    end

    @testset "Invalid n_components parameter" begin
        @test_throws ArgumentError PPCACATEEstimator(n_components=0)
    end

    @testset "Invalid n_strata parameter" begin
        @test_throws ArgumentError GMMStratifiedCATEEstimator(n_strata=1)
        @test_throws ArgumentError GMMStratifiedCATEEstimator(n_strata=0)
    end

    @testset "Invalid base_learner parameter" begin
        @test_throws ArgumentError FactorAnalysisCATEEstimator(base_learner=:invalid)
        @test_throws ArgumentError PPCACATEEstimator(base_learner=:invalid)
        @test_throws ArgumentError GMMStratifiedCATEEstimator(base_learner=:invalid)
    end

    @testset "Small sample size" begin
        Random.seed!(42)
        n = 50
        X = randn(n, 3)
        T = rand(n) .< 0.5
        Y = X[:, 1] .+ 2.0 .* T .+ randn(n) * 0.5

        problem = CATEProblem(Y, T, X, (alpha=0.05,))

        # All estimators should handle small samples
        for estimator in [
            FactorAnalysisCATEEstimator(n_latent=2),
            PPCACATEEstimator(n_components=2),
            GMMStratifiedCATEEstimator(n_strata=2)
        ]
            solution = solve(problem, estimator)
            @test solution.retcode == :Success
            @test isfinite(solution.ate)
            @test isfinite(solution.se)
        end
    end

    @testset "High-dimensional covariates" begin
        data = dgp_high_dimensional(n=500, p=30, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        # FA and PCA should handle high-dim
        sol_fa = solve(problem, FactorAnalysisCATEEstimator(n_latent=5))
        @test sol_fa.retcode == :Success

        sol_pca = solve(problem, PPCACATEEstimator(n_components=5))
        @test sol_pca.retcode == :Success

        # GMM may struggle with high-dim but should not crash
        sol_gmm = solve(problem, GMMStratifiedCATEEstimator(n_strata=3))
        @test sol_gmm.retcode == :Success
    end

    @testset "n_latent exceeds p-1" begin
        # Should cap at p-1 automatically
        Random.seed!(42)
        n = 100
        p = 3  # Only 3 covariates
        X = randn(n, p)
        T = rand(n) .< 0.5
        Y = X[:, 1] .+ 2.0 .* T .+ randn(n) * 0.5

        problem = CATEProblem(Y, T, X, (alpha=0.05,))

        # Request more factors than possible
        solution = solve(problem, FactorAnalysisCATEEstimator(n_latent=10))
        @test solution.retcode == :Success
        @test isfinite(solution.ate)
    end

    @testset "Imbalanced treatment" begin
        data = dgp_imbalanced_treatment(n=500, true_ate=2.0, treatment_prob=0.15, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        # All should handle imbalanced treatment
        sol_fa = solve(problem, FactorAnalysisCATEEstimator(n_latent=3))
        @test sol_fa.retcode == :Success

        sol_pca = solve(problem, PPCACATEEstimator(n_components=3))
        @test sol_pca.retcode == :Success

        sol_gmm = solve(problem, GMMStratifiedCATEEstimator(n_strata=3))
        @test sol_gmm.retcode == :Success
    end

end


# =============================================================================
# Comparison Tests
# =============================================================================

@testset "Latent CATE vs Standard Meta-Learners" begin

    @testset "Latent methods comparable to T-learner" begin
        data = dgp_constant_effect(n=500, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        sol_t = solve(problem, TLearner())
        sol_fa = solve(problem, FactorAnalysisCATEEstimator(n_latent=3))
        sol_pca = solve(problem, PPCACATEEstimator(n_components=3))

        # All should be reasonably close to true ATE
        @test abs(sol_t.ate - data.true_ate) < 0.5
        @test abs(sol_fa.ate - data.true_ate) < 0.5
        @test abs(sol_pca.ate - data.true_ate) < 0.5
    end

    @testset "Strong confounding DGP" begin
        data = dgp_strong_confounding(n=1000, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        # Latent methods may help capture confounding structure
        sol_fa = solve(problem, FactorAnalysisCATEEstimator(n_latent=3))
        sol_pca = solve(problem, PPCACATEEstimator(n_components=3))
        sol_gmm = solve(problem, GMMStratifiedCATEEstimator(n_strata=3))

        # All should have finite estimates (may have larger bias due to confounding)
        @test isfinite(sol_fa.ate)
        @test isfinite(sol_pca.ate)
        @test isfinite(sol_gmm.ate)

        # Allow larger tolerance for confounded data
        @test abs(sol_fa.ate - data.true_ate) < 1.5
        @test abs(sol_pca.ate - data.true_ate) < 1.5
    end

end
