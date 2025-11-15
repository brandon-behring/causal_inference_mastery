"""
Tests for Two-Stage Least Squares (2SLS) estimator.

Phase 4.3: 2SLS implementation
"""

using Test
using CausalEstimators
using LinearAlgebra
using Statistics
using Random

@testset "2SLS Estimator" begin
    @testset "2SLS - Simple Known-Answer Test" begin
        # Simple IV setup with known effect
        Random.seed!(123)
        n = 1000

        # Instrument
        z = randn(n)

        # Endogenous treatment: D = 0.8Z + ε_d
        d = 0.8 * z + 0.5 * randn(n)

        # Outcome: Y = 2D + ε_y (true effect β = 2.0)
        y = 2.0 * d + randn(n)

        Z = reshape(z, n, 1)

        # Create problem and solve
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
        solution = solve(problem, TSLS())

        # Should recover β ≈ 2.0
        @test abs(solution.estimate - 2.0) < 0.2
        @test solution.n == n
        @test solution.n_instruments == 1
        @test solution.n_covariates == 0
        @test solution.estimator_name == "2SLS"
        @test solution.first_stage_fstat > 10.0  # Strong instrument
        @test !solution.weak_iv_warning

        println("2SLS estimate: $(round(solution.estimate, digits=3)), True: 2.0")
    end

    @testset "2SLS - Robust vs Classical SEs" begin
        Random.seed!(456)
        n = 500
        z = randn(n)
        d = 0.7 * z + randn(n)

        # Heteroskedastic errors
        y = 3.0 * d + (1.0 .+ 0.5 * abs.(d)) .* randn(n)

        Z = reshape(z, n, 1)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))

        # Robust SEs (default)
        sol_robust = solve(problem, TSLS(robust=true))

        # Classical SEs
        sol_classic = solve(problem, TSLS(robust=false))

        # Both should give same point estimate
        @test abs(sol_robust.estimate - sol_classic.estimate) < 1e-10

        # Standard errors should differ (robust typically larger with heteroskedasticity)
        @test sol_robust.se != sol_classic.se

        # CI widths should differ
        ci_width_robust = sol_robust.ci_upper - sol_robust.ci_lower
        ci_width_classic = sol_classic.ci_upper - sol_classic.ci_lower
        @test ci_width_robust != ci_width_classic

        println("Robust SE: $(round(sol_robust.se, digits=3)), Classical SE: $(round(sol_classic.se, digits=3))")
    end

    @testset "2SLS - With Covariates" begin
        Random.seed!(789)
        n = 500
        z = randn(n)
        x = randn(n, 2)  # 2 covariates

        # D depends on Z and X
        d = 0.6 * z + 0.3 * x[:, 1] + 0.2 * x[:, 2] + randn(n)

        # Y depends on D and X
        y = 2.5 * d + 0.5 * x[:, 1] - 0.3 * x[:, 2] + randn(n)

        Z = reshape(z, n, 1)

        # With covariates
        problem = IVProblem(y, d, Z, x, (alpha=0.05,))
        solution = solve(problem, TSLS())

        @test abs(solution.estimate - 2.5) < 0.3
        @test solution.n_covariates == 2
        @test solution.first_stage_fstat > 10.0

        println("2SLS with covariates: $(round(solution.estimate, digits=3)), True: 2.5")
    end

    @testset "2SLS - Multiple Instruments" begin
        Random.seed!(101112)
        n = 500
        z1 = randn(n)
        z2 = randn(n)

        # D depends on both instruments
        d = 0.5 * z1 + 0.4 * z2 + randn(n)

        # Y depends on D
        y = 1.5 * d + randn(n)

        Z = hcat(z1, z2)

        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
        solution = solve(problem, TSLS())

        @test abs(solution.estimate - 1.5) < 0.2
        @test solution.n_instruments == 2
        @test solution.first_stage_fstat > 10.0

        # Overidentified: should have Sargan test
        @test !isnothing(solution.overid_pvalue)
        @test 0.0 <= solution.overid_pvalue <= 1.0

        println("2SLS (K=2): $(round(solution.estimate, digits=3)), Overid p-value: $(round(solution.overid_pvalue, digits=3))")
    end

    @testset "2SLS - Weak IV Warning" begin
        Random.seed!(131415)
        n = 200
        z = randn(n)

        # Weak first stage: D barely depends on Z
        d = 0.05 * z + randn(n)
        y = 2.0 * d + randn(n)

        Z = reshape(z, n, 1)

        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
        solution = solve(problem, TSLS())

        # Should detect weak instruments
        @test solution.first_stage_fstat < 10.0
        @test solution.weak_iv_warning

        println("Weak IV F-stat: $(round(solution.first_stage_fstat, digits=2))")
    end

    @testset "2SLS - Confidence Interval Coverage" begin
        # Monte Carlo test: CI should contain true value ~95% of time
        Random.seed!(161718)
        n_sims = 100
        true_beta = 3.0
        coverage_count = 0

        for i in 1:n_sims
            n = 500
            z = randn(n)
            d = 0.7 * z + randn(n)
            y = true_beta * d + randn(n)

            Z = reshape(z, n, 1)
            problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
            solution = solve(problem, TSLS())

            if solution.ci_lower <= true_beta <= solution.ci_upper
                coverage_count += 1
            end
        end

        coverage = coverage_count / n_sims

        # Should be close to 95% (allow 85-100% given finite samples)
        @test coverage >= 0.85

        println("CI coverage: $(round(coverage * 100, digits=1))%")
    end

    @testset "2SLS - P-Value Consistency" begin
        Random.seed!(192021)
        n = 500
        z = randn(n)
        d = 0.7 * z + randn(n)
        y = 2.0 * d + randn(n)

        Z = reshape(z, n, 1)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
        solution = solve(problem, TSLS())

        # P-value should be consistent with CI
        # If p < 0.05, then 0 should NOT be in 95% CI
        if solution.p_value < 0.05
            @test !(solution.ci_lower <= 0.0 <= solution.ci_upper)
        else
            @test (solution.ci_lower <= 0.0 <= solution.ci_upper)
        end

        # P-value should be in [0, 1]
        @test 0.0 <= solution.p_value <= 1.0

        println("P-value: $(round(solution.p_value, digits=4))")
    end

    @testset "2SLS - Sargan Overidentification Test" begin
        Random.seed!(222324)
        n = 500

        # Valid instruments (exogenous)
        z1 = randn(n)
        z2 = randn(n)

        # D depends on both
        d = 0.6 * z1 + 0.5 * z2 + randn(n)

        # Y depends on D (instruments are exogenous → Sargan should not reject)
        y = 2.0 * d + randn(n)

        Z = hcat(z1, z2)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
        solution = solve(problem, TSLS())

        # With valid instruments, Sargan test should not reject at 5%
        @test !isnothing(solution.overid_pvalue)
        @test solution.overid_pvalue > 0.05  # Should not reject exogeneity

        println("Sargan p-value (valid instruments): $(round(solution.overid_pvalue, digits=3))")
    end

    @testset "2SLS - Diagnostics Fields" begin
        Random.seed!(252627)
        n = 500
        z = randn(n)
        d = 0.7 * z + randn(n)
        y = 2.0 * d + randn(n)

        Z = reshape(z, n, 1)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
        solution = solve(problem, TSLS())

        # Check diagnostics contain expected fields
        @test haskey(solution.diagnostics, :first_stage_coef)
        @test haskey(solution.diagnostics, :first_stage_se)
        @test haskey(solution.diagnostics, :cragg_donald)
        @test haskey(solution.diagnostics, :robust_se)
        @test haskey(solution.diagnostics, :second_stage_coef)
        @test haskey(solution.diagnostics, :n_instruments)
        @test haskey(solution.diagnostics, :n_covariates)

        # Verify values
        @test length(solution.diagnostics.first_stage_coef) == 1  # K=1
        @test solution.diagnostics.cragg_donald > 0.0
        @test solution.diagnostics.robust_se == true  # Default
        @test solution.diagnostics.n_instruments == 1
        @test solution.diagnostics.n_covariates == 0
    end

    @testset "2SLS - Type Stability" begin
        n = 100
        z = randn(n)
        d = randn(n)
        y = randn(n)
        Z = reshape(z, n, 1)

        # Float64
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
        solution = @inferred solve(problem, TSLS())

        @test solution isa IVSolution{Float64}
        @test solution.estimate isa Float64
        @test solution.se isa Float64
    end

    @testset "2SLS - Null Effect Detection" begin
        # Test that we correctly detect when true effect is zero
        Random.seed!(282930)
        n = 500
        z = randn(n)
        d = 0.8 * z + randn(n)

        # Null effect: Y independent of D
        y = randn(n)

        Z = reshape(z, n, 1)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
        solution = solve(problem, TSLS())

        # Estimate should be close to zero
        @test abs(solution.estimate) < 0.3

        # P-value should be large (fail to reject H₀: β = 0)
        @test solution.p_value > 0.05

        # CI should contain zero
        @test solution.ci_lower <= 0.0 <= solution.ci_upper

        println("Null effect estimate: $(round(solution.estimate, digits=3)), p-value: $(round(solution.p_value, digits=3))")
    end

    @testset "2SLS - Strong IV Properties" begin
        # With strong IV, 2SLS should have good properties
        Random.seed!(313233)
        n = 1000
        z = randn(n)

        # Very strong first stage
        d = 0.9 * z + 0.1 * randn(n)
        y = 2.5 * d + randn(n)

        Z = reshape(z, n, 1)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
        solution = solve(problem, TSLS())

        # Strong instrument
        @test solution.first_stage_fstat > 100.0
        @test !solution.weak_iv_warning

        # Good precision
        @test solution.se < 0.5

        # Accurate estimate
        @test abs(solution.estimate - 2.5) < 0.2

        println("Strong IV: F=$(round(solution.first_stage_fstat, digits=1)), SE=$(round(solution.se, digits=3))")
    end

    @testset "2SLS - Integration with Weak IV Diagnostics" begin
        Random.seed!(343536)
        n = 300
        z = randn(n)
        d = 0.15 * z + randn(n)  # Moderate weakness
        y = 2.0 * d + randn(n)

        Z = reshape(z, n, 1)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
        solution = solve(problem, TSLS())

        # Diagnostics should be populated
        @test solution.first_stage_fstat > 0.0
        @test haskey(solution.diagnostics, :cragg_donald)

        # Weak IV warning should match F-stat
        if solution.first_stage_fstat < 10.0
            @test solution.weak_iv_warning
        end

        println("F-stat: $(round(solution.first_stage_fstat, digits=2)), Weak? $(solution.weak_iv_warning)")
    end
end

println("\n" * "="^70)
println("2SLS Estimator Tests Complete")
println("="^70)
println("All 2SLS functionality validated:")
println("- ✅ Point estimation")
println("- ✅ Robust and classical standard errors")
println("- ✅ Confidence intervals and p-values")
println("- ✅ Multiple instruments and overidentification tests")
println("- ✅ Weak IV diagnostics integration")
println("- ✅ Covariates handling")
println("="^70)
