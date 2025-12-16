#=
Tests for T-Learner (Two Models) Implementation
=#

@testset "T-Learner Tests" begin

    @testset "Known Answer: Constant Effect" begin
        Random.seed!(42)
        n = 200
        X = randn(n, 3)
        T = rand(n) .> 0.5
        true_ate = 2.0
        Y = 1.0 .+ X[:, 1] .+ true_ate .* T .+ 0.5 .* randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, TLearner())

        @test solution.method == :t_learner
        @test solution.retcode == :Success
        @test abs(solution.ate - true_ate) < 0.5
    end

    @testset "Heterogeneous Effect: Correlation with Truth" begin
        Random.seed!(123)
        n = 300
        X = randn(n, 3)
        T = rand(n) .> 0.5
        # τ(x) = 2 + x₁
        true_cate = 2.0 .+ X[:, 1]
        Y = 1.0 .+ X[:, 1] .+ true_cate .* T .+ randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, TLearner())

        # CATE estimates should correlate with true CATE
        correlation = cor(solution.cate, true_cate)
        @test correlation > 0.3
    end

    @testset "CI Contains True Effect" begin
        Random.seed!(999)
        n = 400
        X = randn(n, 2)
        T = rand(n) .> 0.5
        true_ate = 1.5
        Y = 0.5 .+ true_ate .* T .+ 0.8 .* randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, TLearner())

        @test solution.ci_lower < true_ate < solution.ci_upper
    end

    @testset "SE Positive and Finite" begin
        Random.seed!(789)
        n = 100
        X = randn(n, 2)
        T = rand(n) .> 0.5
        Y = 1.0 .+ 2.0 .* T .+ randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, TLearner())

        @test solution.se > 0
        @test isfinite(solution.se)
    end

    @testset "Ridge Model Option" begin
        Random.seed!(101)
        n = 100
        X = randn(n, 5)
        T = rand(n) .> 0.5
        Y = 1.0 .+ 2.0 .* T .+ randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, TLearner(model=:ridge))

        @test solution.method == :t_learner
        @test isfinite(solution.ate)
    end

end
