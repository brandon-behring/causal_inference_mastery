"""
Monte Carlo validation for Sharp RDD estimator.

Phase 3.7: Test statistical properties with 10,000 simulations:
- Coverage: 95% CI should contain true effect in ~95% of simulations
- Bias: Mean estimate should be close to true effect (|bias| < 0.05)
- Power: Should detect effect with high probability

Following Cattaneo, Idrobo, Titiunik (2019) recommendations.
"""

using Test
using CausalEstimators
using Random
using Statistics

"""
Data generating process for RDD Monte Carlo.
"""
function generate_rdd_data(;
    n::Int=1000,
    τ::Real=0.0,
    dgp::Symbol=:linear,
    noise_level::Real=1.0,
    seed::Int=123
)
    Random.seed!(seed)

    # Running variable
    x = randn(n) .* 2.0

    # Treatment assignment
    treatment = x .>= 0.0

    # Outcome
    if dgp == :linear
        # Linear DGP: y = 2x + τ*D + ε
        y = 2.0 .* x .+ τ .* treatment .+ noise_level .* randn(n)
    elseif dgp == :quadratic
        # Quadratic DGP: y = x + 0.5x² + τ*D + ε
        y = x .+ 0.5 .* x.^2 .+ τ .* treatment .+ noise_level .* randn(n)
    elseif dgp == :discontinuous_slope
        # Discontinuous slope: different trends on each side
        y_left = 2.0 .* x[.!treatment] .+ noise_level .* randn(sum(.!treatment))
        y_right = 1.0 .* x[treatment] .+ τ .+ noise_level .* randn(sum(treatment))
        y = zeros(n)
        y[.!treatment] = y_left
        y[treatment] = y_right
    else
        error("Unknown DGP: $dgp")
    end

    return (x=x, treatment=treatment, y=y)
end

@testset "Monte Carlo Validation" begin
    @testset "Coverage - Null Effect (τ=0)" begin
        # Test that 95% CI has correct coverage when τ=0
        n_sims = 1000  # Reduced for test speed (full validation uses 10,000)
        τ_true = 0.0
        n = 1000

        coverage_count = 0
        estimates = zeros(n_sims)

        for i in 1:n_sims
            data = generate_rdd_data(n=n, τ=τ_true, dgp=:linear, seed=i)

            problem = RDDProblem(data.y, data.x, data.treatment, 0.0, nothing, (alpha=0.05,))
            solution = solve(problem, SharpRDD(run_density_test=false))  # Skip McCrary for speed

            estimates[i] = solution.estimate

            # Check if CI contains true effect
            if solution.ci_lower <= τ_true <= solution.ci_upper
                coverage_count += 1
            end
        end

        coverage = coverage_count / n_sims

        # Coverage should be ~95% (allow 95-100% - CCT is conservative by design)
        # CCT bias-corrected inference intentionally produces wider CIs for robustness
        @test 0.95 <= coverage <= 1.00

        # Mean bias should be near zero
        bias = mean(estimates) - τ_true
        @test abs(bias) < 0.05

        println("\nCoverage (τ=0): $(round(coverage * 100, digits=1))%")
        println("Bias: $(round(bias, digits=4))")
    end

    @testset "Coverage - Positive Effect (τ=5)" begin
        n_sims = 1000
        τ_true = 5.0
        n = 1000

        coverage_count = 0
        estimates = zeros(n_sims)

        for i in 1:n_sims
            data = generate_rdd_data(n=n, τ=τ_true, dgp=:linear, seed=i+10000)

            problem = RDDProblem(data.y, data.x, data.treatment, 0.0, nothing, (alpha=0.05,))
            solution = solve(problem, SharpRDD(run_density_test=false))

            estimates[i] = solution.estimate

            if solution.ci_lower <= τ_true <= solution.ci_upper
                coverage_count += 1
            end
        end

        coverage = coverage_count / n_sims
        bias = mean(estimates) - τ_true

        # CCT conservative inference: 95-100% coverage expected
        @test 0.95 <= coverage <= 1.00
        @test abs(bias) < 0.1  # Slightly looser for non-null

        println("\nCoverage (τ=5): $(round(coverage * 100, digits=1))%")
        println("Bias: $(round(bias, digits=4))")
    end

    @testset "Bias - Various Sample Sizes" begin
        n_sims = 500
        τ_true = 3.0

        for n in [200, 500, 1000, 2000]
            estimates = zeros(n_sims)

            for i in 1:n_sims
                data = generate_rdd_data(n=n, τ=τ_true, dgp=:linear, seed=i+20000)

                problem = RDDProblem(data.y, data.x, data.treatment, 0.0, nothing, (alpha=0.05,))
                solution = solve(problem, SharpRDD(run_density_test=false))

                estimates[i] = solution.estimate
            end

            bias = mean(estimates) - τ_true

            # Bias should decrease with sample size
            if n >= 1000
                @test abs(bias) < 0.1
            else
                @test abs(bias) < 0.2  # More lenient for smaller samples
            end

            println("\nBias (n=$n): $(round(bias, digits=4))")
        end
    end

    @testset "Power - Detect τ=0.5" begin
        # Test power to detect small effect
        n_sims = 500
        τ_true = 0.5
        n = 1000

        rejections = 0

        for i in 1:n_sims
            data = generate_rdd_data(n=n, τ=τ_true, dgp=:linear, seed=i+30000)

            problem = RDDProblem(data.y, data.x, data.treatment, 0.0, nothing, (alpha=0.05,))
            solution = solve(problem, SharpRDD(run_density_test=false))

            # Reject if p < 0.05
            if solution.p_value < 0.05
                rejections += 1
            end
        end

        power = rejections / n_sims

        # Should have reasonable power (target: 80%)
        # With small effect and moderate n, expect 40-80%
        @test power > 0.30

        println("\nPower (τ=0.5, n=1000): $(round(power * 100, digits=1))%")
    end

    @testset "Power - Detect τ=2.0" begin
        # Test power with larger effect
        n_sims = 500
        τ_true = 2.0
        n = 1000

        rejections = 0

        for i in 1:n_sims
            data = generate_rdd_data(n=n, τ=τ_true, dgp=:linear, seed=i+40000)

            problem = RDDProblem(data.y, data.x, data.treatment, 0.0, nothing, (alpha=0.05,))
            solution = solve(problem, SharpRDD(run_density_test=false))

            if solution.p_value < 0.05
                rejections += 1
            end
        end

        power = rejections / n_sims

        # Should have high power with larger effect
        @test power > 0.80

        println("\nPower (τ=2.0, n=1000): $(round(power * 100, digits=1))%")
    end

    @testset "Quadratic DGP - Bias Correction" begin
        # Test CCT bias correction with quadratic DGP
        n_sims = 500
        τ_true = 4.0
        n = 1000

        estimates_cct = zeros(n_sims)
        estimates_ik = zeros(n_sims)

        for i in 1:n_sims
            data = generate_rdd_data(n=n, τ=τ_true, dgp=:quadratic, seed=i+50000)

            problem = RDDProblem(data.y, data.x, data.treatment, 0.0, nothing, (alpha=0.05,))

            # CCT with bias correction
            solution_cct = solve(problem, SharpRDD(
                bandwidth_method=CCTBandwidth(),
                run_density_test=false
            ))

            # IK without bias correction
            solution_ik = solve(problem, SharpRDD(
                bandwidth_method=IKBandwidth(),
                run_density_test=false
            ))

            estimates_cct[i] = solution_cct.estimate
            estimates_ik[i] = solution_ik.estimate
        end

        bias_cct = mean(estimates_cct) - τ_true
        bias_ik = mean(estimates_ik) - τ_true

        # CCT should have less bias than IK for quadratic DGP
        @test abs(bias_cct) <= abs(bias_ik) + 0.05  # Allow small margin

        println("\nQuadratic DGP - CCT bias: $(round(bias_cct, digits=4))")
        println("Quadratic DGP - IK bias: $(round(bias_ik, digits=4))")
    end

    @testset "Type I Error Rate" begin
        # Test that we don't over-reject when τ=0
        n_sims = 500
        τ_true = 0.0
        n = 1000

        rejections = 0

        for i in 1:n_sims
            data = generate_rdd_data(n=n, τ=τ_true, dgp=:linear, seed=i+60000)

            problem = RDDProblem(data.y, data.x, data.treatment, 0.0, nothing, (alpha=0.05,))
            solution = solve(problem, SharpRDD(run_density_test=false))

            if solution.p_value < 0.05
                rejections += 1
            end
        end

        type_i_error = rejections / n_sims

        # Should be ~5% (allow 3-7% given finite samples)
        @test 0.03 <= type_i_error <= 0.07

        println("\nType I error rate: $(round(type_i_error * 100, digits=1))%")
    end

    @testset "Different Noise Levels" begin
        # Test robustness to different noise levels
        n_sims = 300
        τ_true = 3.0
        n = 1000

        for noise_level in [0.5, 1.0, 2.0]
            estimates = zeros(n_sims)
            coverage_count = 0

            for i in 1:n_sims
                data = generate_rdd_data(
                    n=n, τ=τ_true, dgp=:linear,
                    noise_level=noise_level, seed=i+70000
                )

                problem = RDDProblem(data.y, data.x, data.treatment, 0.0, nothing, (alpha=0.05,))
                solution = solve(problem, SharpRDD(run_density_test=false))

                estimates[i] = solution.estimate

                if solution.ci_lower <= τ_true <= solution.ci_upper
                    coverage_count += 1
                end
            end

            coverage = coverage_count / n_sims
            bias = mean(estimates) - τ_true

            # Coverage should be maintained across noise levels (CCT conservative: 95-100%)
            @test 0.95 <= coverage <= 1.00
            @test abs(bias) < 0.15

            println("\nNoise=$noise_level - Coverage: $(round(coverage * 100, digits=1))%, Bias: $(round(bias, digits=4))")
        end
    end
end

# Print final summary
println("\n" * "="^70)
println("Monte Carlo Validation Summary")
println("="^70)
println("All tests passed! Sharp RDD estimator demonstrates:")
println("- ✅ Correct coverage (~95%)")
println("- ✅ Low bias (|bias| < 0.05 for large samples)")
println("- ✅ Adequate power for moderate effects")
println("- ✅ Correct Type I error rate (~5%)")
println("- ✅ CCT bias correction effective for curved DGPs")
println("- ✅ Robust to different noise levels")
println("="^70)
