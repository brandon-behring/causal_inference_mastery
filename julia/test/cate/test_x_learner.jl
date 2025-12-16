#=
Tests for X-Learner (Cross-Learner) Implementation
=#

@testset "X-Learner Tests" begin

    @testset "Known Answer: Constant Effect" begin
        Random.seed!(42)
        n = 200
        X = randn(n, 3)
        T = rand(n) .> 0.5
        true_ate = 2.0
        Y = 1.0 .+ X[:, 1] .+ true_ate .* T .+ 0.5 .* randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, XLearner())

        @test solution.method == :x_learner
        @test solution.retcode == :Success
        @test abs(solution.ate - true_ate) < 0.8
    end

    @testset "Imbalanced Groups" begin
        # X-learner should handle imbalanced treatment groups well
        Random.seed!(123)
        n = 300
        X = randn(n, 2)
        # Imbalanced: 80% treated
        T = rand(n) .> 0.2
        true_ate = 1.5
        Y = 1.0 .+ true_ate .* T .+ randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, XLearner())

        @test abs(solution.ate - true_ate) < 1.0
    end

    @testset "CI Contains True Effect" begin
        Random.seed!(888)
        n = 400
        X = randn(n, 2)
        T = rand(n) .> 0.5
        true_ate = 2.0
        Y = 0.5 .+ true_ate .* T .+ 0.8 .* randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, XLearner())

        @test solution.ci_lower < true_ate < solution.ci_upper
    end

    @testset "SE Positive and Finite" begin
        Random.seed!(789)
        n = 100
        X = randn(n, 2)
        T = rand(n) .> 0.5
        Y = 1.0 .+ 2.0 .* T .+ randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, XLearner())

        @test solution.se > 0
        @test isfinite(solution.se)
    end

    @testset "CATE Shape Correct" begin
        Random.seed!(101)
        n = 50
        X = randn(n, 3)
        T = rand(n) .> 0.5
        Y = 1.0 .+ 2.0 .* T .+ randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, XLearner())

        @test size(solution.cate) == (n,)
        @test all(isfinite.(solution.cate))
    end

end
