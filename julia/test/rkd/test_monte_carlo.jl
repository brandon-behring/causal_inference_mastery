"""
Monte Carlo validation tests for Julia RKD module.

Following the 6-layer validation architecture:
- Layer 3: Monte Carlo simulations to verify statistical properties
  - Bias: estimate should be unbiased (mean close to true value)
  - Coverage: CI should contain true value ~95% of the time
  - SE accuracy: SE should match empirical SD of estimates

Test parameters calibrated for computational efficiency while maintaining
statistical power to detect meaningful deviations.
"""

using Test
using CausalEstimators
using Random
using Statistics
using Distributions

# =============================================================================
# DGP Functions
# =============================================================================

"""Generate Sharp RKD data with known kink effect."""
function generate_sharp_rkd_data(;
    n::Int=500,
    cutoff::Float64=0.0,
    slope_left::Float64=0.5,
    slope_right::Float64=1.5,
    noise_sd::Float64=1.0,
    seed::Union{Nothing,Int}=nothing
)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    # Running variable centered at cutoff
    X = rand(Uniform(-5, 5), n)

    # Treatment with kink at cutoff (deterministic for Sharp RKD)
    D = [x < cutoff ? slope_left * (x - cutoff) : slope_right * (x - cutoff) for x in X]

    # Outcome is linear function of treatment
    Y = D .+ noise_sd .* randn(n)

    true_kink = slope_right - slope_left

    return Y, X, D, true_kink
end

"""Generate Fuzzy RKD data with known LATE."""
function generate_fuzzy_rkd_data(;
    n::Int=500,
    cutoff::Float64=0.0,
    slope_left_d::Float64=0.5,
    slope_right_d::Float64=1.5,
    true_effect::Float64=2.0,
    noise_d::Float64=0.3,
    noise_y::Float64=1.0,
    seed::Union{Nothing,Int}=nothing
)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    X = rand(Uniform(-5, 5), n)

    # Expected treatment with kink
    D_expected = [x < cutoff ? slope_left_d * (x - cutoff) : slope_right_d * (x - cutoff) for x in X]

    # Fuzzy treatment: add noise
    D = D_expected .+ noise_d .* randn(n)

    # Outcome
    Y = true_effect .* D .+ noise_y .* randn(n)

    first_stage_kink = slope_right_d - slope_left_d

    return Y, X, D, true_effect, first_stage_kink
end

# =============================================================================
# Sharp RKD Monte Carlo Tests
# =============================================================================

@testset "Sharp RKD Monte Carlo Validation" begin

    @testset "Bias: estimate unbiased for true kink" begin
        # Parameters
        n_simulations = 500
        n_obs = 400
        true_kink = 1.0  # slope_right - slope_left
        bandwidth = 2.5

        estimates = Float64[]

        for sim in 1:n_simulations
            Y, X, D, _ = generate_sharp_rkd_data(
                n=n_obs,
                slope_left=0.5,
                slope_right=1.5,
                noise_sd=1.0,
                seed=sim
            )
            problem = RKDProblem(Y, X, D, 0.0)
            solution = solve(problem, SharpRKD(bandwidth=bandwidth))

            if solution.retcode in [:Success, :Warning] && isfinite(solution.estimate)
                push!(estimates, solution.estimate)
            end
        end

        # Compute bias
        mean_estimate = mean(estimates)
        bias = mean_estimate - true_kink

        # Bias should be small (< 0.15 for RKD with local polynomial)
        @test abs(bias) < 0.15
        @test length(estimates) > n_simulations * 0.95  # >95% should succeed
    end

    @testset "Coverage: 95% CI contains true value" begin
        n_simulations = 500
        n_obs = 400
        true_kink = 1.0
        bandwidth = 2.5

        coverage_count = 0
        valid_sims = 0

        for sim in 1:n_simulations
            Y, X, D, _ = generate_sharp_rkd_data(
                n=n_obs,
                slope_left=0.5,
                slope_right=1.5,
                noise_sd=1.0,
                seed=sim
            )
            problem = RKDProblem(Y, X, D, 0.0)
            solution = solve(problem, SharpRKD(bandwidth=bandwidth))

            if solution.retcode in [:Success, :Warning] && isfinite(solution.ci_lower)
                valid_sims += 1
                if solution.ci_lower <= true_kink <= solution.ci_upper
                    coverage_count += 1
                end
            end
        end

        coverage = coverage_count / valid_sims

        # Coverage should be close to 95% (allow 90-99% range)
        @test 0.90 <= coverage <= 0.99
    end

    @testset "SE accuracy: SE ≈ empirical SD" begin
        n_simulations = 500
        n_obs = 400
        bandwidth = 2.5

        estimates = Float64[]
        reported_ses = Float64[]

        for sim in 1:n_simulations
            Y, X, D, _ = generate_sharp_rkd_data(
                n=n_obs,
                slope_left=0.5,
                slope_right=1.5,
                noise_sd=1.0,
                seed=sim
            )
            problem = RKDProblem(Y, X, D, 0.0)
            solution = solve(problem, SharpRKD(bandwidth=bandwidth))

            if solution.retcode in [:Success, :Warning] && isfinite(solution.estimate)
                push!(estimates, solution.estimate)
                push!(reported_ses, solution.se)
            end
        end

        empirical_sd = std(estimates)
        mean_reported_se = mean(reported_ses)

        # SE ratio should be close to 1 (within 30%)
        se_ratio = mean_reported_se / empirical_sd
        @test 0.7 < se_ratio < 1.3
    end

    @testset "Different sample sizes" begin
        sample_sizes = [200, 500, 1000]
        true_kink = 1.0

        for n_obs in sample_sizes
            estimates = Float64[]

            for sim in 1:100
                Y, X, D, _ = generate_sharp_rkd_data(
                    n=n_obs,
                    slope_left=0.5,
                    slope_right=1.5,
                    noise_sd=1.0,
                    seed=sim
                )
                problem = RKDProblem(Y, X, D, 0.0)
                solution = solve(problem, SharpRKD(bandwidth=2.5))

                if solution.retcode in [:Success, :Warning] && isfinite(solution.estimate)
                    push!(estimates, solution.estimate)
                end
            end

            bias = abs(mean(estimates) - true_kink)
            @test bias < 0.25  # More lenient for smaller n
        end
    end

end

# =============================================================================
# Fuzzy RKD Monte Carlo Tests
# =============================================================================

@testset "Fuzzy RKD Monte Carlo Validation" begin

    @testset "Bias: LATE estimate unbiased" begin
        n_simulations = 400
        n_obs = 600
        true_effect = 2.0
        bandwidth = 2.5

        estimates = Float64[]

        for sim in 1:n_simulations
            Y, X, D, _, _ = generate_fuzzy_rkd_data(
                n=n_obs,
                slope_left_d=0.5,
                slope_right_d=1.5,
                true_effect=true_effect,
                noise_d=0.3,
                noise_y=1.0,
                seed=sim
            )
            problem = RKDProblem(Y, X, D, 0.0)
            solution = solve(problem, FuzzyRKD(bandwidth=bandwidth))

            if solution.retcode in [:Success, :Warning] && isfinite(solution.estimate)
                push!(estimates, solution.estimate)
            end
        end

        mean_estimate = mean(estimates)
        bias = mean_estimate - true_effect

        # Fuzzy RKD has more variance, allow larger bias tolerance
        @test abs(bias) < 0.3
        @test length(estimates) > n_simulations * 0.90
    end

    @testset "Coverage: 95% CI contains true LATE" begin
        n_simulations = 400
        n_obs = 600
        true_effect = 2.0
        bandwidth = 2.5

        coverage_count = 0
        valid_sims = 0

        for sim in 1:n_simulations
            Y, X, D, _, _ = generate_fuzzy_rkd_data(
                n=n_obs,
                slope_left_d=0.5,
                slope_right_d=1.5,
                true_effect=true_effect,
                noise_d=0.3,
                noise_y=1.0,
                seed=sim
            )
            problem = RKDProblem(Y, X, D, 0.0)
            solution = solve(problem, FuzzyRKD(bandwidth=bandwidth))

            if solution.retcode in [:Success, :Warning] && isfinite(solution.ci_lower)
                valid_sims += 1
                if solution.ci_lower <= true_effect <= solution.ci_upper
                    coverage_count += 1
                end
            end
        end

        coverage = coverage_count / valid_sims

        # Coverage should be close to 95%
        @test 0.88 <= coverage <= 0.99
    end

    @testset "First stage F-statistic distribution" begin
        n_simulations = 200
        n_obs = 500

        f_stats = Float64[]

        for sim in 1:n_simulations
            Y, X, D, _, _ = generate_fuzzy_rkd_data(
                n=n_obs,
                slope_left_d=0.5,
                slope_right_d=1.5,
                true_effect=2.0,
                noise_d=0.3,
                seed=sim
            )
            problem = RKDProblem(Y, X, D, 0.0)
            solution = solve(problem, FuzzyRKD(bandwidth=2.5))

            if isfinite(solution.first_stage_f_stat)
                push!(f_stats, solution.first_stage_f_stat)
            end
        end

        # Most should have strong first stage
        @test mean(f_stats .> 10) > 0.80  # >80% have F > 10
        @test median(f_stats) > 15  # Median F should be well above 10
    end

    @testset "Weak first stage detection" begin
        n_simulations = 100

        weak_detected = 0

        for sim in 1:n_simulations
            X = rand(Uniform(-5, 5), 500)

            # Weak first stage: only 0.05 slope difference
            D_expected = [x < 0 ? 1.0 * x : 1.05 * x for x in X]
            D = D_expected .+ 2.0 .* randn(500)  # High noise
            Y = 2.0 .* D .+ randn(500)

            problem = RKDProblem(Y, X, D, 0.0)
            solution = solve(problem, FuzzyRKD(bandwidth=2.5))

            if solution.weak_first_stage
                weak_detected += 1
            end
        end

        # Should detect weak first stage in many cases
        @test weak_detected > n_simulations * 0.50
    end

end

# =============================================================================
# Bandwidth Sensitivity Monte Carlo Tests
# =============================================================================

@testset "Bandwidth Sensitivity Monte Carlo" begin

    @testset "Results stable across reasonable bandwidths" begin
        n_simulations = 100
        bandwidths = [1.5, 2.0, 2.5, 3.0, 3.5]
        true_kink = 1.0

        # Collect mean estimates for each bandwidth
        mean_estimates = Float64[]

        for bw in bandwidths
            estimates = Float64[]

            for sim in 1:n_simulations
                Y, X, D, _ = generate_sharp_rkd_data(
                    n=500,
                    slope_left=0.5,
                    slope_right=1.5,
                    noise_sd=1.0,
                    seed=sim
                )
                problem = RKDProblem(Y, X, D, 0.0)
                solution = solve(problem, SharpRKD(bandwidth=bw))

                if solution.retcode in [:Success, :Warning] && isfinite(solution.estimate)
                    push!(estimates, solution.estimate)
                end
            end

            push!(mean_estimates, mean(estimates))
        end

        # All should be close to true kink
        for est in mean_estimates
            @test abs(est - true_kink) < 0.20
        end

        # Estimates should be stable across bandwidths
        @test std(mean_estimates) < 0.10
    end

end

# =============================================================================
# Polynomial Order Sensitivity Tests
# =============================================================================

@testset "Polynomial Order Monte Carlo" begin

    @testset "Linear vs quadratic comparison" begin
        n_simulations = 100
        true_kink = 1.0

        estimates_linear = Float64[]
        estimates_quadratic = Float64[]

        for sim in 1:n_simulations
            Y, X, D, _ = generate_sharp_rkd_data(
                n=600,
                slope_left=0.5,
                slope_right=1.5,
                noise_sd=1.0,
                seed=sim
            )
            problem = RKDProblem(Y, X, D, 0.0)

            sol1 = solve(problem, SharpRKD(bandwidth=2.5, polynomial_order=1))
            sol2 = solve(problem, SharpRKD(bandwidth=2.5, polynomial_order=2))

            if sol1.retcode in [:Success, :Warning] && isfinite(sol1.estimate)
                push!(estimates_linear, sol1.estimate)
            end
            if sol2.retcode in [:Success, :Warning] && isfinite(sol2.estimate)
                push!(estimates_quadratic, sol2.estimate)
            end
        end

        bias_linear = abs(mean(estimates_linear) - true_kink)
        bias_quadratic = abs(mean(estimates_quadratic) - true_kink)

        # Both should be reasonably unbiased
        @test bias_linear < 0.20
        @test bias_quadratic < 0.25  # Quadratic has more variance
    end

end
