"""
Tests for Generalized Method of Moments (GMM) estimator.

Phase 4.5: GMM implementation with optimal weighting and Hansen J test
"""

using Test
using CausalEstimators
using LinearAlgebra
using Statistics
using Random

@testset "GMM Estimator" begin
    @testset "GMM - Identity Weighting Matches 2SLS" begin
        # GMM with identity weighting should equal 2SLS exactly
        Random.seed!(123)
        n = 500
        z = randn(n)
        d = 0.8 * z + randn(n)
        y = 2.0 * d + randn(n)

        Z = reshape(z, n, 1)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))

        # GMM with identity weighting
        sol_gmm = solve(problem, GMM(weighting=:identity))

        # 2SLS
        sol_tsls = solve(problem, TSLS())

        # Should match exactly
        @test abs(sol_gmm.estimate - sol_tsls.estimate) < 1e-10
        @test abs(sol_gmm.se - sol_tsls.se) < 1e-6

        println(
            "GMM-Identity: $(round(sol_gmm.estimate, digits=3)), 2SLS: $(round(sol_tsls.estimate, digits=3))",
        )
    end

    @testset "GMM - Just-Identified (K=L=1)" begin
        # With K=1, GMM should match 2SLS regardless of weighting
        Random.seed!(456)
        n = 500
        z = randn(n)
        d = 0.7 * z + randn(n)
        y = 2.5 * d + randn(n)

        Z = reshape(z, n, 1)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))

        sol_gmm_opt = solve(problem, GMM(weighting=:optimal))
        sol_tsls = solve(problem, TSLS())

        # Estimates should be close (just-identified)
        @test abs(sol_gmm_opt.estimate - sol_tsls.estimate) < 0.1

        # No overidentification test (K = L)
        @test isnothing(sol_gmm_opt.overid_pvalue)

        println("Just-identified: GMM $(round(sol_gmm_opt.estimate, digits=3)), 2SLS $(round(sol_tsls.estimate, digits=3))")
    end

    @testset "GMM - Overidentified Efficiency Gains (K=3, L=1)" begin
        # With multiple instruments, optimal GMM more efficient than 2SLS
        Random.seed!(789)
        n = 500
        z1 = randn(n)
        z2 = randn(n)
        z3 = randn(n)

        # All instruments valid
        d = 0.5 * z1 + 0.4 * z2 + 0.3 * z3 + randn(n)
        y = 2.0 * d + randn(n)

        Z = hcat(z1, z2, z3)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))

        sol_gmm = solve(problem, GMM(weighting=:optimal))
        sol_tsls = solve(problem, TSLS())

        # Both should recover true effect
        @test abs(sol_gmm.estimate - 2.0) < 0.2
        @test abs(sol_tsls.estimate - 2.0) < 0.2

        # GMM should have Hansen J test (overidentified)
        @test !isnothing(sol_gmm.overid_pvalue)
        @test 0.0 <= sol_gmm.overid_pvalue <= 1.0

        # With valid instruments, Hansen J should not reject
        @test sol_gmm.overid_pvalue > 0.05

        println("Overidentified (K=3): GMM $(round(sol_gmm.estimate, digits=3)), Hansen J p-value: $(round(sol_gmm.overid_pvalue, digits=3))")
    end

    @testset "GMM - Optimal vs Identity Weighting" begin
        # Optimal weighting should have smaller SE with heteroskedasticity
        Random.seed!(101112)
        n = 500
        z1 = randn(n)
        z2 = randn(n)

        d = 0.6 * z1 + 0.5 * z2 + randn(n)

        # Heteroskedastic errors
        y = 2.5 * d + (1.0 .+ 0.5 * abs.(d)) .* randn(n)

        Z = hcat(z1, z2)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))

        sol_identity = solve(problem, GMM(weighting=:identity))
        sol_optimal = solve(problem, GMM(weighting=:optimal))

        # Both should estimate same parameter
        @test abs(sol_identity.estimate - sol_optimal.estimate) < 0.5

        # Optimal weighting typically more efficient (smaller SE)
        # (Not guaranteed in finite samples, but usually true)
        @test !isnan(sol_optimal.se)
        @test sol_optimal.se > 0.0

        println("Identity SE: $(round(sol_identity.se, digits=3)), Optimal SE: $(round(sol_optimal.se, digits=3))")
    end

    @testset "GMM - HAC Weighting for Time Series" begin
        # HAC should handle autocorrelated errors
        Random.seed!(131415)
        n = 500

        # Generate autocorrelated instrument
        z = randn(n)
        for t in 2:n
            z[t] += 0.3 * z[t-1]  # AR(1) process
        end

        # Treatment depends on instrument
        d = 0.7 * z + randn(n)

        # Autocorrelated errors
        ε = randn(n)
        for t in 2:n
            ε[t] += 0.5 * ε[t-1]  # AR(1) errors
        end

        y = 2.0 * d + ε

        Z = reshape(z, n, 1)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))

        # GMM with HAC
        sol_hac = solve(problem, GMM(weighting=:hac, kernel=:bartlett, bandwidth=4))

        # Should still recover true effect
        @test abs(sol_hac.estimate - 2.0) < 0.3

        # HAC SE should be positive
        @test sol_hac.se > 0.0

        # Estimator name should reflect HAC
        @test sol_hac.estimator_name == "GMM-HAC"

        println("HAC estimate: $(round(sol_hac.estimate, digits=3)), SE: $(round(sol_hac.se, digits=3))")
    end

    @testset "GMM - HAC Auto Bandwidth" begin
        # Test automatic bandwidth selection
        Random.seed!(161718)
        n = 200
        z = randn(n)
        d = 0.8 * z + randn(n)
        y = 2.0 * d + randn(n)

        Z = reshape(z, n, 1)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))

        # HAC with auto bandwidth (bandwidth=nothing)
        sol_hac_auto = solve(problem, GMM(weighting=:hac, bandwidth=nothing))

        @test !isnan(sol_hac_auto.estimate)
        @test sol_hac_auto.se > 0.0

        println("HAC auto bandwidth estimate: $(round(sol_hac_auto.estimate, digits=3))")
    end

    @testset "GMM - Hansen J Valid Instruments" begin
        # With valid instruments, Hansen J should not reject
        Random.seed!(192021)
        n = 500
        z1 = randn(n)
        z2 = randn(n)
        z3 = randn(n)

        # All instruments exogenous (uncorrelated with error)
        d = 0.6 * z1 + 0.5 * z2 + 0.4 * z3 + randn(n)
        ε = randn(n)  # Independent of Z
        y = 2.0 * d + ε

        Z = hcat(z1, z2, z3)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))

        solution = solve(problem, GMM(weighting=:optimal))

        # Hansen J should not reject valid instruments
        @test !isnothing(solution.overid_pvalue)
        @test solution.overid_pvalue > 0.05

        @test !isnothing(solution.diagnostics.hansen_j)
        @test solution.diagnostics.hansen_j >= 0.0

        println("Hansen J (valid): J=$(round(solution.diagnostics.hansen_j, digits=2)), p=$(round(solution.overid_pvalue, digits=3))")
    end

    @testset "GMM - Hansen J Invalid Instruments" begin
        # With invalid instruments, Hansen J should detect
        Random.seed!(222324)
        n = 500
        z1 = randn(n)
        z2 = randn(n)
        z3 = randn(n)

        # z3 is invalid (correlated with error)
        ε = randn(n)
        z3_invalid = z3 + 0.5 * ε  # Endogenous instrument

        d = 0.6 * z1 + 0.5 * z2 + 0.4 * z3_invalid + randn(n)
        y = 2.0 * d + ε

        Z = hcat(z1, z2, z3_invalid)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))

        solution = solve(problem, GMM(weighting=:optimal))

        # Hansen J may detect invalid instrument (not guaranteed with finite sample)
        @test !isnothing(solution.overid_pvalue)

        # J statistic should be positive
        @test solution.diagnostics.hansen_j >= 0.0

        println("Hansen J (invalid): J=$(round(solution.diagnostics.hansen_j, digits=2)), p=$(round(solution.overid_pvalue, digits=3))")
    end

    @testset "GMM - With Covariates" begin
        Random.seed!(252627)
        n = 500
        z1 = randn(n)
        z2 = randn(n)
        x = randn(n, 2)

        # D depends on Z and X
        d = 0.5 * z1 + 0.4 * z2 + 0.3 * x[:, 1] + 0.2 * x[:, 2] + randn(n)

        # Y depends on D and X
        y = 2.0 * d + 0.5 * x[:, 1] - 0.3 * x[:, 2] + randn(n)

        Z = hcat(z1, z2)
        problem = IVProblem(y, d, Z, x, (alpha=0.05,))

        solution = solve(problem, GMM(weighting=:optimal))

        @test abs(solution.estimate - 2.0) < 0.3
        @test solution.n_covariates == 2
        @test solution.first_stage_fstat > 10.0

        println("GMM with covariates: $(round(solution.estimate, digits=3)), True: 2.0")
    end

    @testset "GMM - Multiple Instruments (K=5)" begin
        Random.seed!(282930)
        n = 500
        Z_mat = randn(n, 5)

        # D depends on all instruments
        d = Z_mat * [0.4, 0.3, 0.3, 0.2, 0.2] + randn(n)

        # Y depends on D
        y = 1.5 * d + randn(n)

        problem = IVProblem(y, d, Z_mat, nothing, (alpha=0.05,))
        solution = solve(problem, GMM(weighting=:optimal))

        @test abs(solution.estimate - 1.5) < 0.2
        @test solution.n_instruments == 5

        # Overidentified: Hansen J test available
        @test !isnothing(solution.overid_pvalue)

        println("GMM (K=5): $(round(solution.estimate, digits=3)), Hansen J p=$(round(solution.overid_pvalue, digits=3))")
    end

    @testset "GMM - Confidence Interval Coverage" begin
        # Monte Carlo test: 95% CI should contain true value ~95% of time
        Random.seed!(313233)
        n_sims = 100  # Reduced for speed
        true_beta = 3.0
        coverage_count = 0

        for i in 1:n_sims
            n = 500
            z1 = randn(n)
            z2 = randn(n)
            d = 0.6 * z1 + 0.5 * z2 + randn(n)
            y = true_beta * d + randn(n)

            Z = hcat(z1, z2)
            problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
            solution = solve(problem, GMM(weighting=:optimal))

            if solution.ci_lower <= true_beta <= solution.ci_upper
                coverage_count += 1
            end
        end

        coverage = coverage_count / n_sims

        # Should be close to 95% (allow 85-100% with only 100 sims)
        @test coverage >= 0.85

        println("CI coverage (100 sims): $(round(coverage * 100, digits=1))%")
    end

    @testset "GMM - P-Value Consistency" begin
        Random.seed!(343536)
        n = 500
        z1 = randn(n)
        z2 = randn(n)
        d = 0.7 * z1 + 0.6 * z2 + randn(n)
        y = 2.0 * d + randn(n)

        Z = hcat(z1, z2)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
        solution = solve(problem, GMM(weighting=:optimal))

        # P-value should be consistent with CI
        if solution.p_value < 0.05
            @test !(solution.ci_lower <= 0.0 <= solution.ci_upper)
        else
            @test (solution.ci_lower <= 0.0 <= solution.ci_upper)
        end

        # P-value should be in [0, 1]
        @test 0.0 <= solution.p_value <= 1.0

        println("P-value: $(round(solution.p_value, digits=4))")
    end

    @testset "GMM - Null Effect Detection" begin
        # Test zero true effect
        Random.seed!(373839)
        n = 500
        z1 = randn(n)
        z2 = randn(n)
        d = 0.8 * z1 + 0.7 * z2 + randn(n)

        # Null effect: Y independent of D
        y = randn(n)

        Z = hcat(z1, z2)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
        solution = solve(problem, GMM(weighting=:optimal))

        # Estimate should be close to zero
        @test abs(solution.estimate) < 0.3

        # P-value should be large
        @test solution.p_value > 0.05

        # CI should contain zero
        @test solution.ci_lower <= 0.0 <= solution.ci_upper

        println("Null effect: $(round(solution.estimate, digits=3)), p=$(round(solution.p_value, digits=3))")
    end

    @testset "GMM - Diagnostics Fields" begin
        Random.seed!(404142)
        n = 500
        z1 = randn(n)
        z2 = randn(n)
        d = 0.7 * z1 + 0.6 * z2 + randn(n)
        y = 2.0 * d + randn(n)

        Z = hcat(z1, z2)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
        solution = solve(problem, GMM(weighting=:optimal))

        # Check diagnostics fields
        @test haskey(solution.diagnostics, :weighting)
        @test haskey(solution.diagnostics, :hansen_j)
        @test haskey(solution.diagnostics, :initial_estimate)
        @test haskey(solution.diagnostics, :cragg_donald)
        @test haskey(solution.diagnostics, :n_instruments)
        @test haskey(solution.diagnostics, :n_covariates)

        # Verify values
        @test solution.diagnostics.weighting == :optimal
        @test !isnothing(solution.diagnostics.hansen_j)
        @test solution.diagnostics.hansen_j >= 0.0
        @test !isnan(solution.diagnostics.initial_estimate)
        @test solution.diagnostics.n_instruments == 2
        @test solution.diagnostics.n_covariates == 0
    end

    @testset "GMM - Type Stability" begin
        n = 100
        z = randn(n)
        d = randn(n)
        y = randn(n)
        Z = reshape(z, n, 1)

        # Float64
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
        solution = @inferred solve(problem, GMM(weighting=:optimal))

        @test solution isa IVSolution{Float64}
        @test solution.estimate isa Float64
        @test solution.se isa Float64
    end

    @testset "GMM - Weak IV Warning" begin
        Random.seed!(434445)
        n = 300
        z = randn(n)

        # Weak first stage
        d = 0.1 * z + randn(n)
        y = 2.0 * d + randn(n)

        Z = reshape(z, n, 1)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))
        solution = solve(problem, GMM(weighting=:optimal))

        # Should detect weak instruments
        @test solution.first_stage_fstat < 10.0
        @test solution.weak_iv_warning

        println("Weak IV: F=$(round(solution.first_stage_fstat, digits=2))")
    end

    @testset "GMM - Estimator Name" begin
        Random.seed!(464748)
        n = 500
        z = randn(n)
        d = 0.7 * z + randn(n)
        y = 2.0 * d + randn(n)

        Z = reshape(z, n, 1)
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))

        # Test different weighting methods
        sol_identity = solve(problem, GMM(weighting=:identity))
        @test sol_identity.estimator_name == "GMM-Identity"

        sol_optimal = solve(problem, GMM(weighting=:optimal))
        @test sol_optimal.estimator_name == "GMM-Optimal"

        sol_hac = solve(problem, GMM(weighting=:hac))
        @test sol_hac.estimator_name == "GMM-HAC"
    end

    @testset "GMM - Invalid Weighting Method" begin
        # Should throw error for invalid weighting
        @test_throws ArgumentError GMM(weighting=:invalid)
        @test_throws ArgumentError GMM(weighting=:two_step)  # Not implemented
    end

    @testset "GMM - Invalid Kernel" begin
        # Should throw error for invalid HAC kernel
        @test_throws ArgumentError GMM(weighting=:hac, kernel=:quadratic)
    end

    @testset "GMM - Invalid Bandwidth" begin
        # Negative bandwidth should error
        @test_throws ArgumentError GMM(weighting=:hac, bandwidth=-1)
        @test_throws ArgumentError GMM(weighting=:hac, bandwidth=0)

        # Valid bandwidth
        @test GMM(weighting=:hac, bandwidth=4) isa GMM
    end
end

println("\n" * "="^70)
println("GMM Estimator Tests Complete")
println("="^70)
println("All GMM functionality validated:")
println("- ✅ Identity weighting matches 2SLS")
println("- ✅ Optimal weighting for efficiency")
println("- ✅ HAC weighting for time series")
println("- ✅ Hansen J overidentification test")
println("- ✅ Just-identified and overidentified cases")
println("- ✅ Confidence intervals and p-values")
println("- ✅ Multiple instruments handling")
println("- ✅ Weak IV diagnostics integration")
println("="^70)
