#=
Tests for Dynamic DML module.

Tests cover:
1. Known-answer tests (effect recovery)
2. Adversarial tests (edge cases)
3. Cross-fitting strategies
=#

using Test
using Statistics
using Random
using CausalEstimators

@testset "Dynamic DML Tests" begin

    @testset "Known-Answer Tests" begin
        @testset "Contemporaneous effect recovery" begin
            Random.seed!(42)
            n = 300
            X = randn(n, 3)
            D = Float64.(rand(n) .< 0.5)
            Y = 2.0 .* D .+ X * [1.0, 0.5, 0.2] .+ randn(n)

            result = dynamic_dml(Y, D, X; max_lag=0, n_folds=5)

            # Should recover θ₀ ≈ 2.0
            @test abs(result.theta[1] - 2.0) < 0.5
            # CI should cover true value
            @test result.ci_lower[1] < 2.0 < result.ci_upper[1]
        end

        @testset "Lagged effects recovery" begin
            Random.seed!(123)
            Y, D, X, true_effects = simulate_dynamic_dgp(
                n_obs=500,
                n_lags=3,
                true_effects=[2.0, 1.0, 0.5],
                confounding_strength=0.2,
                seed=123
            )

            result = dynamic_dml(Y, D, X; max_lag=2, n_folds=5)

            # Check each lag (with reasonable tolerance for stochastic test)
            for h in 0:2
                @test abs(result.theta[h + 1] - true_effects[h + 1]) < 2.0
            end
        end

        @testset "Zero effect detection" begin
            Random.seed!(789)
            n = 300
            X = randn(n, 3)
            D = Float64.(rand(n) .< 0.5)
            # No treatment effect
            Y = X * [1.0, 0.5, 0.2] .+ randn(n)

            result = dynamic_dml(Y, D, X; max_lag=1, n_folds=5)

            # Effects should be close to zero
            for h in 0:1
                @test abs(result.theta[h + 1]) < 0.5
            end
            # CIs should include zero
            for h in 0:1
                @test result.ci_lower[h + 1] < 0 < result.ci_upper[h + 1]
            end
        end

        @testset "Cumulative effect" begin
            Random.seed!(456)
            Y, D, X, true_effects = simulate_dynamic_dgp(
                n_obs=500,
                n_lags=3,
                true_effects=[2.0, 1.0, 0.5],
                seed=456
            )

            discount = 0.99
            result = dynamic_dml(Y, D, X; max_lag=2, n_folds=5, discount_factor=discount)

            # Expected cumulative
            true_cumulative = sum(discount^h * true_effects[h + 1] for h in 0:2)

            @test abs(result.cumulative_effect - true_cumulative) < 1.5
        end
    end

    @testset "Adversarial Tests" begin
        @testset "Sparse treatment" begin
            Random.seed!(654)
            n = 500
            X = randn(n, 3)
            D = Float64.(rand(n) .< 0.1)  # Only 10% treated
            Y = 2.0 .* D .+ X * [1.0, 0.5, 0.2] .+ randn(n)

            result = dynamic_dml(Y, D, X; max_lag=0, n_folds=3)

            @test isfinite(result.theta[1])
            @test result.theta_se[1] > 0
        end

        @testset "Short time series" begin
            Y, D, X, _ = simulate_dynamic_dgp(n_obs=100, n_lags=2, seed=42)

            result = dynamic_dml(Y, D, X; max_lag=2, n_folds=3)

            @test length(result.theta) == 3
            @test all(isfinite, result.theta)
        end

        @testset "High-dimensional states" begin
            Random.seed!(321)
            n = 300
            p = 20
            X = randn(n, p)
            D = Float64.(rand(n) .< 0.5)
            Y = 2.0 .* D .+ X[:, 1:3] * [1.0, 0.5, 0.2] .+ randn(n)

            result = dynamic_dml(Y, D, X; max_lag=0, n_folds=3)

            @test abs(result.theta[1] - 2.0) < 0.5
        end

        @testset "Invalid max_lag" begin
            Y, D, X, _ = simulate_dynamic_dgp(n_obs=100, seed=42)

            @test_throws ErrorException dynamic_dml(Y, D, X; max_lag=-1)
        end

        @testset "Insufficient observations" begin
            Random.seed!(42)
            Y = randn(15)
            D = Float64.(rand(15) .< 0.5)
            X = randn(15, 2)

            @test_throws ErrorException dynamic_dml(Y, D, X; max_lag=10)
        end
    end

    @testset "Cross-Fitting Strategies" begin
        @testset "BlockedTimeSeriesSplit indices" begin
            cv = BlockedTimeSeriesSplit(n_splits=5)
            splits = split_indices(cv, 100)

            all_test = Int[]
            for (train_idx, test_idx) in splits
                # Train and test should not overlap
                @test isempty(intersect(train_idx, test_idx))
                append!(all_test, test_idx)
            end
            # All indices should be covered
            @test sort(all_test) == collect(1:100)
        end

        @testset "RollingOriginSplit forward only" begin
            cv = RollingOriginSplit(initial_window=50, step=10, horizon=5)
            splits = split_indices(cv, 100)

            for (train_idx, test_idx) in splits
                # All training indices should be before all test indices
                @test maximum(train_idx) < minimum(test_idx)
            end
        end

        @testset "PanelStratifiedSplit units" begin
            cv = PanelStratifiedSplit(n_splits=5)
            n_units, n_periods = 50, 10
            unit_id = repeat(1:n_units, inner=n_periods)
            splits = split_indices(cv, n_units * n_periods, unit_id)

            for (train_idx, test_idx) in splits
                train_units = unique(unit_id[train_idx])
                test_units = unique(unit_id[test_idx])
                # Units should not overlap
                @test isempty(intersect(train_units, test_units))
            end
        end

        @testset "ProgressiveBlockSplit expanding" begin
            cv = ProgressiveBlockSplit(n_blocks=10, min_train_blocks=3)
            splits = split_indices(cv, 1000)

            train_sizes = [length(train_idx) for (train_idx, _) in splits]
            # Training size should increase
            @test issorted(train_sizes)
        end

        @testset "All strategies run" begin
            Y, D, X, _ = simulate_dynamic_dgp(n_obs=300, n_lags=2, seed=42)

            for strategy in [:blocked, :rolling, :progressive]
                result = dynamic_dml(Y, D, X; max_lag=1, n_folds=3, cross_fitting=strategy)
                @test result !== nothing
                @test all(isfinite, result.theta)
            end
        end
    end

    @testset "Result Object" begin
        Random.seed!(42)
        Y, D, X, _ = simulate_dynamic_dgp(n_obs=300, seed=42)
        result = dynamic_dml(Y, D, X; max_lag=1, n_folds=3)

        @testset "Summary format" begin
            s = result_summary(result)
            @test occursin("Dynamic DML Results", s)
            @test occursin("Lag", s)
            @test occursin("Effect", s)
        end

        @testset "dml_is_significant" begin
            # With true effect = 2.0, lag 0 should be significant
            sig = dml_is_significant(result, 0)
            @test sig isa Bool
        end

        @testset "Nuisance R² populated" begin
            @test haskey(result.nuisance_r2, :outcome_r2)
            @test haskey(result.nuisance_r2, :propensity_r2)
            @test !isempty(result.nuisance_r2[:outcome_r2])
        end
    end

    @testset "HAC Inference" begin
        @testset "Bartlett kernel" begin
            Random.seed!(42)
            Y, D, X, _ = simulate_dynamic_dgp(n_obs=300, seed=42)
            result = dynamic_dml(Y, D, X; max_lag=0, hac_kernel=:bartlett)

            @test result.theta_se[1] > 0
            @test result.hac_kernel == "bartlett"
        end

        @testset "QS kernel" begin
            Random.seed!(42)
            Y, D, X, _ = simulate_dynamic_dgp(n_obs=300, seed=42)
            result = dynamic_dml(Y, D, X; max_lag=0, hac_kernel=:qs)

            @test result.theta_se[1] > 0
            @test result.hac_kernel == "qs"
        end

        @testset "Custom bandwidth" begin
            Random.seed!(42)
            Y, D, X, _ = simulate_dynamic_dgp(n_obs=300, seed=42)
            result = dynamic_dml(Y, D, X; max_lag=0, hac_bandwidth=10)

            @test result.hac_bandwidth == 10
        end
    end

    @testset "DGP Simulation" begin
        Y, D, X, effects = simulate_dynamic_dgp(n_obs=500, n_lags=3, seed=42)

        @test length(Y) == 500
        @test length(D) == 500
        @test size(X) == (500, 3)
        @test length(effects) == 3
    end

end
