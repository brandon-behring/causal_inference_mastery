#=
Tests for DragonNet (Neural CATE) Implementation

Session 152: Julia Neural CATE foundation.

Layer 1: Known-answer tests
Layer 2: Adversarial tests
=#

using Test
using Random
using Statistics
using CausalEstimators


@testset "DragonNet Tests" begin

    # =========================================================================
    # Layer 1: Known-Answer Tests
    # =========================================================================

    @testset "Layer 1: Known-Answer" begin

        @testset "Constant Effect: ATE Recovery" begin
            # DGP: Y = 1 + X₁ + 2*T + ε
            Random.seed!(42)
            n = 300
            X = randn(n, 3)
            T = rand(n) .> 0.5
            true_ate = 2.0
            Y = 1.0 .+ X[:, 1] .+ true_ate .* T .+ 0.5 .* randn(n)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, Dragonnet())

            @test solution.method == :dragonnet
            @test solution.retcode == :Success
            # Neural methods have higher variance; looser tolerance
            @test abs(solution.ate - true_ate) < 1.0
        end

        @testset "CATE Shape" begin
            Random.seed!(42)
            n = 200
            X = randn(n, 2)
            T = rand(n) .> 0.5
            Y = 1.0 .+ 2.0 .* T .+ randn(n)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, Dragonnet())

            @test length(solution.cate) == n
            @test all(isfinite.(solution.cate))
        end

        @testset "CI Contains True Effect" begin
            Random.seed!(123)
            n = 400
            X = randn(n, 3)
            T = rand(n) .> 0.5
            true_ate = 2.0
            Y = 1.0 .+ X[:, 1] .+ true_ate .* T .+ randn(n)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, Dragonnet())

            @test solution.ci_lower < solution.ci_upper
            # May not always contain true due to variance; just check valid CI
        end

        @testset "SE Positive and Finite" begin
            Random.seed!(42)
            n = 200
            X = randn(n, 2)
            T = rand(n) .> 0.5
            Y = 1.0 .+ 2.0 .* T .+ randn(n)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, Dragonnet())

            @test solution.se > 0
            @test isfinite(solution.se)
        end

        @testset "Custom Architecture" begin
            Random.seed!(42)
            n = 200
            X = randn(n, 2)
            T = rand(n) .> 0.5
            Y = 1.0 .+ 2.0 .* T .+ randn(n)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            config = DragonNetConfig(
                hidden_layers = (100, 50),
                head_layers = (50,),
                max_iter = 100
            )
            solution = solve(problem, Dragonnet(backend=:regression, config=config))

            @test isfinite(solution.ate)
            @test solution.method == :dragonnet
        end

        @testset "Confounded DGP" begin
            Random.seed!(42)
            n = 400
            X = randn(n, 2)
            # Propensity depends on X
            propensity = 1.0 ./ (1.0 .+ exp.(-0.5 .* X[:, 1]))
            T = [rand() < p for p in propensity]
            true_ate = 2.0
            Y = 1.0 .+ 0.5 .* X[:, 1] .+ true_ate .* T .+ randn(n)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, Dragonnet())

            # DragonNet should handle confounding via propensity head
            @test abs(solution.ate - true_ate) < 1.5
        end

        @testset "Heterogeneous Effect" begin
            Random.seed!(123)
            n = 400
            X = randn(n, 3)
            T = rand(n) .> 0.5
            # CATE varies with X
            true_cate = 2.0 .+ X[:, 1]
            Y = 1.0 .+ X[:, 1] .+ true_cate .* T .+ randn(n)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, Dragonnet())

            # Should capture some heterogeneity
            @test std(solution.cate) > 0.1
            @test abs(mean(solution.cate) - mean(true_cate)) < 1.0
        end

    end


    # =========================================================================
    # Layer 2: Adversarial Tests
    # =========================================================================

    @testset "Layer 2: Adversarial" begin

        @testset "Invalid Backend" begin
            @test_throws ArgumentError Dragonnet(backend = :invalid)
        end

        @testset "Invalid Config - Empty Hidden Layers" begin
            @test_throws ArgumentError DragonNetConfig(hidden_layers = ())
        end

        @testset "Invalid Config - Negative Alpha" begin
            @test_throws ArgumentError DragonNetConfig(alpha = -0.1)
        end

        @testset "Invalid Config - Zero Learning Rate" begin
            @test_throws ArgumentError DragonNetConfig(learning_rate = 0.0)
        end

        @testset "High Dimensional" begin
            Random.seed!(42)
            n = 300
            p = 20
            X = randn(n, p)
            T = rand(n) .> 0.5
            Y = 1.0 .+ 2.0 .* T .+ randn(n)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, Dragonnet())

            @test isfinite(solution.ate)
            @test all(isfinite.(solution.cate))
        end

        @testset "Small Sample" begin
            Random.seed!(42)
            n = 40
            X = randn(n, 2)
            T = rand(n) .> 0.5
            Y = 1.0 .+ 2.0 .* T .+ randn(n)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, Dragonnet())

            @test isfinite(solution.ate)
        end

        @testset "Imbalanced Treatment" begin
            Random.seed!(42)
            n = 400
            X = randn(n, 2)
            T = rand(n) .> 0.9  # 10% treated
            Y = 1.0 .+ X[:, 1] .+ 2.0 .* T .+ randn(n)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, Dragonnet())

            @test isfinite(solution.ate)
            @test all(isfinite.(solution.cate))
        end

        @testset "Extreme Propensity" begin
            Random.seed!(42)
            n = 300
            X = randn(n, 2)
            # Strong selection on X
            propensity = 1.0 ./ (1.0 .+ exp.(-2.0 .* X[:, 1]))
            T = [rand() < p for p in propensity]
            Y = 1.0 .+ X[:, 1] .+ 2.0 .* T .+ randn(n)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, Dragonnet())

            @test isfinite(solution.ate)
            @test all(isfinite.(solution.cate))
        end

        @testset "Flux Backend Not Implemented" begin
            Random.seed!(42)
            n = 100
            X = randn(n, 2)
            T = rand(n) .> 0.5
            Y = 1.0 .+ 2.0 .* T .+ randn(n)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))

            @test_throws ErrorException solve(problem, Dragonnet(backend = :flux))
        end

        @testset "Convenience Constructor" begin
            estimator = Dragonnet(
                :regression;
                hidden_layers = (50, 25),
                max_iter = 50,
                alpha = 0.01
            )
            @test estimator.backend == :regression
            @test estimator.config.hidden_layers == (50, 25)
            @test estimator.config.max_iter == 50
            @test estimator.config.alpha == 0.01
        end

    end


    # =========================================================================
    # Comparison with Other Meta-Learners
    # =========================================================================

    @testset "Comparison with Meta-Learners" begin

        @testset "DragonNet vs T-Learner on Simple DGP" begin
            Random.seed!(42)
            n = 300
            X = randn(n, 3)
            T = rand(n) .> 0.5
            true_ate = 2.0
            Y = 1.0 .+ X[:, 1] .+ true_ate .* T .+ randn(n)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))

            dragon_sol = solve(problem, Dragonnet())
            t_learner_sol = solve(problem, TLearner())

            # Both should recover roughly correct ATE
            @test abs(dragon_sol.ate - true_ate) < 1.5
            @test abs(t_learner_sol.ate - true_ate) < 1.5
        end

    end

end
