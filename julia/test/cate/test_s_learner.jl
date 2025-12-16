#=
Tests for S-Learner (Single Model) Implementation
=#

@testset "S-Learner Tests" begin

    @testset "Known Answer: Constant Effect" begin
        # DGP: Y = 1 + X₁ + 2*T + ε
        Random.seed!(42)
        n = 200
        X = randn(n, 3)
        T = rand(n) .> 0.5
        true_ate = 2.0
        Y = 1.0 .+ X[:, 1] .+ true_ate .* T .+ 0.5 .* randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, SLearner())

        @test solution.method == :s_learner
        @test solution.retcode == :Success
        @test abs(solution.ate - true_ate) < 0.5
        @test length(solution.cate) == n
    end

    @testset "CI Contains True Effect" begin
        Random.seed!(123)
        n = 300
        X = randn(n, 2)
        T = rand(n) .> 0.5
        true_ate = 1.5
        Y = 0.5 .+ X[:, 1] .+ true_ate .* T .+ randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, SLearner())

        @test solution.ci_lower < true_ate < solution.ci_upper
    end

    @testset "SE Positive and Finite" begin
        Random.seed!(456)
        n = 100
        X = randn(n, 2)
        T = rand(n) .> 0.5
        Y = 1.0 .+ 2.0 .* T .+ randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, SLearner())

        @test solution.se > 0
        @test isfinite(solution.se)
    end

    @testset "Ridge Model Option" begin
        Random.seed!(789)
        n = 100
        X = randn(n, 5)
        T = rand(n) .> 0.5
        Y = 1.0 .+ 2.0 .* T .+ randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, SLearner(model=:ridge))

        @test solution.method == :s_learner
        @test isfinite(solution.ate)
    end

    @testset "CATE Shape Correct" begin
        Random.seed!(101)
        n = 50
        X = randn(n, 3)
        T = rand(n) .> 0.5
        Y = 1.0 .+ 2.0 .* T .+ randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, SLearner())

        @test size(solution.cate) == (n,)
        @test all(isfinite.(solution.cate))
    end

end
