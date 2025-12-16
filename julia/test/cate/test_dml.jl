#=
Tests for Double Machine Learning (DML) Implementation
=#

@testset "DML Tests" begin

    @testset "Known Answer: Constant Effect" begin
        Random.seed!(42)
        n = 400
        X = randn(n, 3)
        T = rand(n) .> 0.5
        true_ate = 2.0
        Y = 1.0 .+ X[:, 1] .+ true_ate .* T .+ 0.5 .* randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, DoubleMachineLearning())

        @test solution.method == :dml
        @test solution.retcode == :Success
        @test abs(solution.ate - true_ate) < 0.6
    end

    @testset "Cross-Fitting Reduces Bias" begin
        # DML with cross-fitting should work with complex DGP
        Random.seed!(123)
        n = 500
        X = randn(n, 4)
        # Confounded: treatment depends on X
        propensity = 1 ./ (1 .+ exp.(-0.5 .* X[:, 1]))
        T = rand(n) .< propensity
        true_ate = 1.5
        # Outcome also depends on X
        Y = 0.5 .+ X[:, 1] .+ X[:, 2] .+ true_ate .* T .+ randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, DoubleMachineLearning(n_folds=5))

        # Should recover approximately correct ATE despite confounding
        @test abs(solution.ate - true_ate) < 1.0
    end

    @testset "CI Contains True Effect" begin
        Random.seed!(456)
        n = 350
        X = randn(n, 2)
        T = rand(n) .> 0.5
        true_ate = 2.0
        Y = 0.5 .+ true_ate .* T .+ randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, DoubleMachineLearning())

        @test solution.ci_lower < true_ate < solution.ci_upper
    end

    @testset "SE Positive and Finite" begin
        Random.seed!(789)
        n = 200
        X = randn(n, 2)
        T = rand(n) .> 0.5
        Y = 1.0 .+ 2.0 .* T .+ randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, DoubleMachineLearning())

        @test solution.se > 0
        @test isfinite(solution.se)
    end

    @testset "Custom Fold Count" begin
        Random.seed!(101)
        n = 300
        X = randn(n, 3)
        T = rand(n) .> 0.5
        Y = 1.0 .+ 2.0 .* T .+ randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        # Test with 3 folds instead of default 5
        solution = solve(problem, DoubleMachineLearning(n_folds=3))

        @test solution.method == :dml
        @test isfinite(solution.ate)
        @test isfinite(solution.se)
    end

    @testset "CATE Shape Correct" begin
        Random.seed!(202)
        n = 100
        X = randn(n, 2)
        T = rand(n) .> 0.5
        Y = 1.0 .+ 2.0 .* T .+ randn(n)

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, DoubleMachineLearning())

        @test size(solution.cate) == (n,)
        @test all(isfinite.(solution.cate))
    end

end
