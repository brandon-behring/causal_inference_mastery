#=
Tests for Double Machine Learning with Continuous Treatment (Session 116)

Tests mirror Python tests in tests/test_cate/test_dml_continuous.py
=#

@testset "DML Continuous Tests" begin

    # =========================================================================
    # Known-Answer Tests
    # =========================================================================

    @testset "Known Answer: Constant Effect" begin
        Random.seed!(42)
        n = 500
        X = randn(n, 3)
        # Continuous treatment: D depends on X
        D = X[:, 1] .+ randn(n)
        true_effect = 2.0
        Y = 1.0 .+ X[:, 1] .+ true_effect .* D .+ randn(n)

        result = dml_continuous(Y, D, X)

        @test result.method == :dml_continuous
        @test abs(result.ate - true_effect) < 0.3
        @test result.ci_lower < true_effect < result.ci_upper
    end

    @testset "Known Answer: Zero Effect" begin
        Random.seed!(123)
        n = 400
        X = randn(n, 2)
        D = X[:, 1] .+ randn(n)
        true_effect = 0.0
        Y = 1.0 .+ X[:, 1] .+ true_effect .* D .+ randn(n)

        result = dml_continuous(Y, D, X)

        @test abs(result.ate) < 0.3  # Close to 0
        @test result.ci_lower < 0.0 < result.ci_upper
    end

    @testset "Known Answer: Negative Effect" begin
        Random.seed!(456)
        n = 400
        X = randn(n, 2)
        D = X[:, 1] .+ randn(n)
        true_effect = -1.5
        Y = 1.0 .+ X[:, 1] .+ true_effect .* D .+ randn(n)

        result = dml_continuous(Y, D, X)

        @test abs(result.ate - true_effect) < 0.4
        @test result.ci_lower < true_effect < result.ci_upper
    end

    @testset "Known Answer: Large Effect" begin
        Random.seed!(789)
        n = 500
        X = randn(n, 3)
        D = 0.5 .* X[:, 1] .+ randn(n)
        true_effect = 5.0
        Y = X[:, 1] .+ true_effect .* D .+ randn(n)

        result = dml_continuous(Y, D, X)

        @test abs(result.ate - true_effect) < 0.5
    end

    # =========================================================================
    # Heterogeneous Effects Tests
    # =========================================================================

    @testset "CATE Shape Matches N" begin
        Random.seed!(101)
        n = 200
        X = randn(n, 2)
        D = X[:, 1] .+ randn(n)
        Y = 1.0 .+ 2.0 .* D .+ randn(n)

        result = dml_continuous(Y, D, X)

        @test length(result.cate) == n
        @test all(isfinite.(result.cate))
    end

    @testset "CATE Mean Close to ATE" begin
        Random.seed!(202)
        n = 300
        X = randn(n, 2)
        D = X[:, 1] .+ randn(n)
        Y = 1.0 .+ 2.0 .* D .+ randn(n)

        result = dml_continuous(Y, D, X)

        # Mean of CATE should be close to ATE
        @test abs(mean(result.cate) - result.ate) < 0.3
    end

    # =========================================================================
    # Adversarial Tests
    # =========================================================================

    @testset "High-Dimensional Covariates" begin
        Random.seed!(303)
        n = 400
        p = 20  # High-dimensional
        X = randn(n, p)
        D = X[:, 1] .+ 0.5 .* randn(n)
        Y = 1.0 .+ X[:, 1] .+ 1.5 .* D .+ randn(n)

        # Should still estimate without error
        result = dml_continuous(Y, D, X, model=:ridge)

        @test isfinite(result.ate)
        @test isfinite(result.ate_se)
        @test result.ate_se > 0
    end

    @testset "Strong Confounding" begin
        Random.seed!(404)
        n = 500
        X = randn(n, 3)
        # Strong confounding: treatment heavily depends on X
        D = 2.0 .* X[:, 1] .+ 0.5 .* randn(n)
        true_effect = 1.0
        # Outcome also heavily depends on X
        Y = 3.0 .* X[:, 1] .+ true_effect .* D .+ randn(n)

        result = dml_continuous(Y, D, X)

        # Should still recover effect reasonably
        @test abs(result.ate - true_effect) < 0.5
    end

    @testset "Two Folds" begin
        Random.seed!(505)
        n = 200
        X = randn(n, 2)
        D = X[:, 1] .+ randn(n)
        Y = 1.0 .+ 2.0 .* D .+ randn(n)

        result = dml_continuous(Y, D, X, n_folds=2)

        @test result.n_folds == 2
        @test length(result.fold_estimates) == 2
        @test isfinite(result.ate)
    end

    @testset "Many Folds" begin
        Random.seed!(606)
        n = 500
        X = randn(n, 2)
        D = X[:, 1] .+ randn(n)
        Y = 1.0 .+ 2.0 .* D .+ randn(n)

        result = dml_continuous(Y, D, X, n_folds=10)

        @test result.n_folds == 10
        @test length(result.fold_estimates) == 10
    end

    # =========================================================================
    # Diagnostics Tests
    # =========================================================================

    @testset "Result Fields Complete" begin
        Random.seed!(707)
        n = 200
        X = randn(n, 2)
        D = X[:, 1] .+ randn(n)
        Y = 1.0 .+ 2.0 .* D .+ randn(n)

        result = dml_continuous(Y, D, X, n_folds=5)

        @test result.method == :dml_continuous
        @test result.n == n
        @test result.n_folds == 5
        @test 0.0 <= result.outcome_r2 <= 1.0
        @test 0.0 <= result.treatment_r2 <= 1.0
        @test length(result.fold_estimates) == 5
        @test length(result.fold_ses) == 5
        @test all(isfinite.(result.fold_estimates))
        @test all(isfinite.(result.fold_ses))
    end

    @testset "SE Positive and Finite" begin
        Random.seed!(808)
        n = 300
        X = randn(n, 2)
        D = X[:, 1] .+ randn(n)
        Y = 1.0 .+ 2.0 .* D .+ randn(n)

        result = dml_continuous(Y, D, X)

        @test result.ate_se > 0
        @test isfinite(result.ate_se)
        @test result.ci_lower < result.ci_upper
    end

    # =========================================================================
    # Input Validation Tests
    # =========================================================================

    @testset "Input Validation: Length Mismatch" begin
        n = 100
        X = randn(n, 2)
        D = randn(n)
        Y_wrong = randn(n + 10)  # Wrong length

        @test_throws ArgumentError dml_continuous(Y_wrong, D, X)
    end

    @testset "Input Validation: No Treatment Variation" begin
        n = 100
        X = randn(n, 2)
        D = fill(1.0, n)  # No variation
        Y = randn(n)

        @test_throws ArgumentError dml_continuous(Y, D, X)
    end

    @testset "Input Validation: n_folds Too Small" begin
        n = 100
        X = randn(n, 2)
        D = X[:, 1] .+ randn(n)
        Y = 1.0 .+ 2.0 .* D .+ randn(n)

        @test_throws ArgumentError dml_continuous(Y, D, X, n_folds=1)
    end

    @testset "Input Validation: Invalid Model" begin
        n = 100
        X = randn(n, 2)
        D = X[:, 1] .+ randn(n)
        Y = 1.0 .+ 2.0 .* D .+ randn(n)

        @test_throws ArgumentError dml_continuous(Y, D, X, model=:invalid)
    end

    # =========================================================================
    # Model Variants Tests
    # =========================================================================

    @testset "OLS Model" begin
        Random.seed!(909)
        n = 300
        X = randn(n, 3)
        D = X[:, 1] .+ randn(n)
        Y = 1.0 .+ X[:, 1] .+ 2.0 .* D .+ randn(n)

        result = dml_continuous(Y, D, X, model=:ols)

        @test isfinite(result.ate)
        @test abs(result.ate - 2.0) < 0.5
    end

    @testset "Ridge Model" begin
        Random.seed!(1010)
        n = 300
        X = randn(n, 3)
        D = X[:, 1] .+ randn(n)
        Y = 1.0 .+ X[:, 1] .+ 2.0 .* D .+ randn(n)

        result = dml_continuous(Y, D, X, model=:ridge)

        @test isfinite(result.ate)
        @test abs(result.ate - 2.0) < 0.5
    end

    # =========================================================================
    # Monte Carlo Test (Basic Unbiasedness)
    # =========================================================================

    @testset "Monte Carlo: Unbiased Estimation" begin
        # Run multiple simulations to check bias
        true_effect = 2.0
        n_sims = 100
        estimates = zeros(n_sims)

        for i in 1:n_sims
            Random.seed!(1000 + i)
            n = 200
            X = randn(n, 2)
            D = X[:, 1] .+ randn(n)
            Y = 1.0 .+ X[:, 1] .+ true_effect .* D .+ randn(n)

            result = dml_continuous(Y, D, X, n_folds=3)
            estimates[i] = result.ate
        end

        # Check bias < 0.15
        bias = mean(estimates) - true_effect
        @test abs(bias) < 0.15
    end

    @testset "Monte Carlo: Coverage Rate" begin
        # Check that 95% CI has ~95% coverage
        true_effect = 2.0
        n_sims = 100
        covers = falses(n_sims)

        for i in 1:n_sims
            Random.seed!(2000 + i)
            n = 200
            X = randn(n, 2)
            D = X[:, 1] .+ randn(n)
            Y = 1.0 .+ X[:, 1] .+ true_effect .* D .+ randn(n)

            result = dml_continuous(Y, D, X, n_folds=3)
            covers[i] = result.ci_lower < true_effect < result.ci_upper
        end

        # Coverage should be approximately 95%, allow 85-99% range
        coverage = mean(covers)
        @test 0.85 < coverage < 0.99
    end

end
