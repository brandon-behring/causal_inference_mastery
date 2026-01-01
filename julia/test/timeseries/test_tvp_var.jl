#=
Tests for TVP-VAR (Time-Varying Parameter VAR) module.

Session 165: Kalman filter estimation with time-varying coefficients.

Tests cover:
1. Known-Answer: Constant VAR DGP (coefficients should stabilize)
2. Known-Answer: Structural Break DGP (coefficients should adapt)
3. IRF computation at different times
4. Stability checking
5. Coefficient change test
6. Edge cases and adversarial inputs
=#

using Test
using Statistics
using Random
using LinearAlgebra
using CausalEstimators

@testset "TVP-VAR Tests" begin

    @testset "Basic Estimation" begin
        @testset "Constant VAR DGP - coefficients should be stable" begin
            Random.seed!(42)
            n = 200
            n_vars = 2

            # Generate data from constant VAR(1)
            true_A = [0.5 0.1; 0.2 0.4]
            true_c = [0.1, 0.2]
            data = zeros(n, n_vars)
            data[1, :] = randn(n_vars)

            for t in 2:n
                data[t, :] = true_c + true_A * data[t - 1, :] + randn(n_vars) * 0.5
            end

            result = tvp_var_estimate(data; lags=1, smooth=true)

            # Check basic properties
            @test result.n_vars == n_vars
            @test result.lags == 1
            @test result.n_obs == n
            @test result.n_obs_effective == n - 1
            @test result.state_dim == n_vars * (n_vars * 1 + 1)  # 2 * 3 = 6

            # Smoothed coefficients should be close to true values
            @test has_smoothed(result)
            @test size(result.coefficients_smoothed) == (n - 1, n_vars, n_vars + 1)

            # Check coefficient variance is small (constant DGP)
            # Time trajectory should be relatively stable
            trajectory_1_1 = coefficient_trajectory(result, 1, 2; smoothed=true)  # A[1,1]
            coef_var = var(trajectory_1_1)
            @test coef_var < 0.5  # Should have low variance for constant DGP

            # Mean coefficient should be close to true value
            mean_A11 = mean(trajectory_1_1)
            @test abs(mean_A11 - 0.5) < 0.3  # Reasonable tolerance
        end

        @testset "Structural break DGP - coefficients should adapt" begin
            Random.seed!(123)
            n = 300
            n_vars = 2
            break_point = 150

            # Generate data with structural break
            true_A_before = [0.5 0.1; 0.2 0.4]
            true_A_after = [0.2 0.3; 0.4 0.3]
            true_c = [0.0, 0.0]

            data = zeros(n, n_vars)
            data[1, :] = randn(n_vars)

            for t in 2:n
                A = t <= break_point ? true_A_before : true_A_after
                data[t, :] = true_c + A * data[t - 1, :] + randn(n_vars) * 0.3
            end

            result = tvp_var_estimate(data; lags=1, Q_scale=0.01, smooth=true)

            # Coefficient trajectory should show change
            trajectory = coefficient_trajectory(result, 1, 2; smoothed=true)

            # Early period mean vs late period mean should differ
            early_mean = mean(trajectory[1:100])
            late_mean = mean(trajectory[200:end])

            # The difference should be substantial (break is 0.3)
            @test abs(late_mean - early_mean) > 0.1  # Some adaptation occurred
        end

        @testset "Filtered vs Smoothed estimates" begin
            Random.seed!(456)
            data = randn(100, 2)

            result = tvp_var_estimate(data; lags=1, smooth=true)

            # Both should exist
            @test !isnothing(result.coefficients_filtered)
            @test !isnothing(result.coefficients_smoothed)

            # Smoothed should have lower variance
            filt_var = var(coefficient_trajectory(result, 1, 1; smoothed=false))
            smooth_var = var(coefficient_trajectory(result, 1, 1; smoothed=true))

            # Smoothed typically has lower variance
            @test smooth_var <= filt_var * 1.5  # Allow some tolerance
        end

        @testset "No smoothing option" begin
            Random.seed!(789)
            data = randn(100, 2)

            result = tvp_var_estimate(data; lags=1, smooth=false)

            @test !isnothing(result.coefficients_filtered)
            @test isnothing(result.coefficients_smoothed)
            @test !has_smoothed(result)
        end
    end

    @testset "Initialization Methods" begin
        Random.seed!(42)
        data = randn(100, 2)

        @testset "OLS initialization (default)" begin
            result = tvp_var_estimate(data; lags=1, initialization=:ols)
            @test result.initialization == "ols"
            @test isfinite(result.log_likelihood)
        end

        @testset "Diffuse initialization" begin
            result = tvp_var_estimate(data; lags=1, initialization=:diffuse)
            @test result.initialization == "diffuse"
            @test isfinite(result.log_likelihood)
        end

        @testset "Custom initialization" begin
            state_dim = 2 * 3  # n_vars * n_params_per_eq
            beta_init = zeros(state_dim)
            P_init = Matrix{Float64}(I, state_dim, state_dim) * 1.0

            result = tvp_var_estimate(
                data;
                lags=1,
                initialization=:custom,
                beta_init=beta_init,
                P_init=P_init
            )
            @test result.initialization == "custom"
            @test isfinite(result.log_likelihood)
        end
    end

    @testset "IRF Computation" begin
        Random.seed!(42)
        data = randn(100, 2)
        result = tvp_var_estimate(data; lags=1)

        @testset "IRF at specific time" begin
            irf = compute_tvp_irf(result, 50; horizons=10, shock_idx=1)

            @test size(irf) == (2, 11)  # n_vars x (horizons+1)
            @test all(isfinite.(irf))

            # Impact at h=0 should be non-zero for shocked variable
            @test abs(irf[1, 1]) > 0
        end

        @testset "IRF with different shock indices" begin
            irf1 = compute_tvp_irf(result, 50; horizons=5, shock_idx=1)
            irf2 = compute_tvp_irf(result, 50; horizons=5, shock_idx=2)

            @test size(irf1) == size(irf2)
            @test irf1 != irf2  # Different shocks should give different responses
        end

        @testset "IRF all times" begin
            irf_all = compute_tvp_irf_all_times(result; horizons=10, shock_idx=1)

            @test size(irf_all) == (result.n_obs_effective, 2, 11)
            @test all(isfinite.(irf_all))
        end

        @testset "Non-orthogonalized IRF" begin
            irf_orth = compute_tvp_irf(result, 50; horizons=5, orthogonalize=true)
            irf_non = compute_tvp_irf(result, 50; horizons=5, orthogonalize=false)

            # Should be different
            @test irf_orth != irf_non
        end
    end

    @testset "Stability Checking" begin
        @testset "Stable VAR" begin
            Random.seed!(42)
            # Generate stable data
            n_vars = 2
            data = zeros(100, n_vars)
            true_A = [0.3 0.1; 0.1 0.3]  # Stable coefficients

            data[1, :] = randn(n_vars)
            for t in 2:100
                data[t, :] = true_A * data[t - 1, :] + randn(n_vars) * 0.5
            end

            result = tvp_var_estimate(data; lags=1)

            is_stable, eigvals = check_tvp_stability(
                get_coefficients_at_time(result, 50),
                result.lags,
                result.n_vars
            )

            @test is_stable
            @test all(abs.(eigvals) .< 1.0)
        end

        @testset "Stability all times" begin
            Random.seed!(42)
            data = randn(100, 2) * 0.5  # Small variance for stability

            result = tvp_var_estimate(data; lags=1, Q_scale=0.0001)

            is_stable, max_mod = check_tvp_stability_all_times(result)

            @test length(is_stable) == result.n_obs_effective
            @test length(max_mod) == result.n_obs_effective
            @test all(isfinite.(max_mod))
        end
    end

    @testset "Coefficient Change Test" begin
        Random.seed!(42)
        data = randn(100, 2)
        result = tvp_var_estimate(data; lags=1)

        variance_ratio, p_value = coefficient_change_test(result, 1, 1)

        @test isfinite(variance_ratio)
        @test variance_ratio >= 0
        @test 0 <= p_value <= 1
    end

    @testset "Accessor Functions" begin
        Random.seed!(42)
        data = randn(100, 2)
        result = tvp_var_estimate(data; lags=1)

        @testset "get_coefficients_at_time" begin
            coef = get_coefficients_at_time(result, 50)
            @test size(coef) == (2, 3)  # (n_vars, n_params_per_eq)
            @test all(isfinite.(coef))
        end

        @testset "get_lag_matrix_at_time" begin
            A1 = get_lag_matrix_at_time(result, 50, 1)
            @test size(A1) == (2, 2)  # (n_vars, n_vars)
            @test all(isfinite.(A1))
        end

        @testset "get_intercepts_at_time" begin
            c = get_intercepts_at_time(result, 50)
            @test length(c) == 2
            @test all(isfinite.(c))
        end

        @testset "coefficient_trajectory" begin
            traj = coefficient_trajectory(result, 1, 1)
            @test length(traj) == result.n_obs_effective
            @test all(isfinite.(traj))
        end
    end

    @testset "Information Criteria" begin
        Random.seed!(42)
        data = randn(100, 2)
        result = tvp_var_estimate(data; lags=1)

        @test isfinite(result.log_likelihood)
        @test isfinite(result.aic)
        @test isfinite(result.bic)

        # BIC should be larger than AIC (more penalty)
        @test result.bic > result.aic
    end

    @testset "Edge Cases" begin
        @testset "Minimum observations" begin
            Random.seed!(42)
            data = randn(15, 2)  # Small sample

            result = tvp_var_estimate(data; lags=1)
            @test result.n_obs_effective == 14
            @test isfinite(result.log_likelihood)
        end

        @testset "Multiple lags" begin
            Random.seed!(42)
            data = randn(100, 2)

            result = tvp_var_estimate(data; lags=3)
            @test result.lags == 3
            @test result.state_dim == 2 * (2 * 3 + 1)  # n_vars * (n_vars*lags + 1)
            @test isfinite(result.log_likelihood)
        end

        @testset "Higher-dimensional VAR" begin
            Random.seed!(42)
            data = randn(100, 4)

            result = tvp_var_estimate(data; lags=1)
            @test result.n_vars == 4
            @test isfinite(result.log_likelihood)
        end

        @testset "Custom Q scale" begin
            Random.seed!(42)
            data = randn(100, 2)

            # Small Q = smoother coefficients
            result_small = tvp_var_estimate(data; lags=1, Q_scale=0.0001)
            # Large Q = more volatile coefficients
            result_large = tvp_var_estimate(data; lags=1, Q_scale=0.1)

            traj_small = coefficient_trajectory(result_small, 1, 1)
            traj_large = coefficient_trajectory(result_large, 1, 1)

            # Larger Q should lead to more coefficient variance
            @test var(traj_small) < var(traj_large)
        end
    end

    @testset "Error Handling" begin
        @testset "Invalid lags" begin
            Random.seed!(42)
            data = randn(100, 2)

            @test_throws ErrorException tvp_var_estimate(data; lags=0)
            @test_throws ErrorException tvp_var_estimate(data; lags=-1)
        end

        @testset "Insufficient observations" begin
            Random.seed!(42)
            data = randn(5, 2)  # Too few observations

            @test_throws ErrorException tvp_var_estimate(data; lags=1)
        end

        @testset "Invalid time index for accessors" begin
            Random.seed!(42)
            data = randn(100, 2)
            result = tvp_var_estimate(data; lags=1)

            @test_throws ErrorException get_coefficients_at_time(result, 0)
            @test_throws ErrorException get_coefficients_at_time(result, 1000)
        end

        @testset "Invalid shock index for IRF" begin
            Random.seed!(42)
            data = randn(100, 2)
            result = tvp_var_estimate(data; lags=1)

            @test_throws ErrorException compute_tvp_irf(result, 50; shock_idx=0)
            @test_throws ErrorException compute_tvp_irf(result, 50; shock_idx=10)
        end

        @testset "Custom init without required params" begin
            Random.seed!(42)
            data = randn(100, 2)

            @test_throws ErrorException tvp_var_estimate(
                data;
                lags=1,
                initialization=:custom
            )
        end
    end

    @testset "Kalman Filter Properties" begin
        Random.seed!(42)
        data = randn(100, 2)
        result = tvp_var_estimate(data; lags=1)

        @testset "Innovations" begin
            @test size(result.innovations) == (result.n_obs_effective, result.n_vars)
            @test all(isfinite.(result.innovations))

            # Innovations should be roughly zero-mean
            @test abs(mean(result.innovations)) < 0.5
        end

        @testset "Innovation covariance" begin
            @test size(result.innovation_covariance) == (
                result.n_obs_effective,
                result.n_vars,
                result.n_vars
            )
            @test all(isfinite.(result.innovation_covariance))
        end

        @testset "Kalman gain" begin
            @test size(result.kalman_gain) == (
                result.n_obs_effective,
                result.state_dim,
                result.n_vars
            )
            @test all(isfinite.(result.kalman_gain))
        end

        @testset "Covariances positive definite" begin
            # Check a few time points
            for t in [1, 50, result.n_obs_effective]
                P = result.covariance_filtered[t, :, :]
                eigvals_P = eigvals(Symmetric(P))
                @test all(eigvals_P .> -1e-10)  # Should be PSD
            end
        end
    end

    @testset "Numerical Stability" begin
        @testset "Large values" begin
            Random.seed!(42)
            data = randn(100, 2) * 1000  # Large scale

            result = tvp_var_estimate(data; lags=1)
            @test all(isfinite.(result.coefficients_filtered))
            @test isfinite(result.log_likelihood)
        end

        @testset "Small values" begin
            Random.seed!(42)
            data = randn(100, 2) * 0.001  # Small scale

            result = tvp_var_estimate(data; lags=1)
            @test all(isfinite.(result.coefficients_filtered))
            @test isfinite(result.log_likelihood)
        end
    end

end  # @testset "TVP-VAR Tests"
