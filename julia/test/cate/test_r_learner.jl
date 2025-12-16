#=
Tests for R-Learner (Robinson Transformation) Implementation
=#

@testset "R-Learner Tests" begin

    @testset "Known Answer: Constant Effect" begin
        Random.seed!(42)
        n = 300
        X = randn(n, 3)
        T = rand(n) .> 0.5
        true_ate = 2.0
        Y = 1.0 .+ X[:, 1] .+ true_ate .* T .+ 0.5 .* randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, RLearner())

        @test solution.method == :r_learner
        @test solution.retcode == :Success
        @test abs(solution.ate - true_ate) < 0.5
    end

    @testset "Heterogeneous Effect: Correlation" begin
        Random.seed!(123)
        n = 400
        X = randn(n, 3)
        T = rand(n) .> 0.5
        # τ(x) = 2 + 0.5*x₁
        true_cate = 2.0 .+ 0.5 .* X[:, 1]
        Y = 1.0 .+ X[:, 1] .+ true_cate .* T .+ randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, RLearner())

        # ATE should be close to mean of true_cate
        true_ate = mean(true_cate)
        @test abs(solution.ate - true_ate) < 0.5
    end

    @testset "CI Contains True Effect" begin
        Random.seed!(456)
        n = 300
        X = randn(n, 2)
        T = rand(n) .> 0.5
        true_ate = 1.5
        Y = 0.5 .+ true_ate .* T .+ randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, RLearner())

        @test solution.ci_lower < true_ate < solution.ci_upper
    end

    @testset "SE Positive and Finite" begin
        Random.seed!(789)
        n = 150
        X = randn(n, 2)
        T = rand(n) .> 0.5
        Y = 1.0 .+ 2.0 .* T .+ randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, RLearner())

        @test solution.se > 0
        @test isfinite(solution.se)
    end

    @testset "CATE Shape Correct" begin
        Random.seed!(101)
        n = 80
        X = randn(n, 3)
        T = rand(n) .> 0.5
        Y = 1.0 .+ 2.0 .* T .+ randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, RLearner())

        # R-learner returns constant CATE (homogeneous)
        @test size(solution.cate) == (n,)
        @test all(isfinite.(solution.cate))
    end

end
