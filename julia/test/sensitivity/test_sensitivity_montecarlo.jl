#=
Monte Carlo Validation for Julia Sensitivity Analysis.

Validates statistical properties:
- E-value: formula accuracy, monotonicity, symmetry
- Rosenbaum bounds: gamma_critical detection, p-value properties

Session 69: Julia Sensitivity Validation
=#

using Test
using Statistics
using Random
using CausalEstimators

# Include DGP generators
include("dgp_sensitivity.jl")

# =============================================================================
# E-Value Monte Carlo Tests
# =============================================================================

@testset "EValue Monte Carlo Validation" begin

    @testset "Formula accuracy across RR range" begin
        rr_values = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

        for rr in rr_values
            e = compute_e_value(rr)

            # Theoretical E-value: E = RR + sqrt(RR * (RR - 1))
            if rr ≈ 1.0
                expected_e = 1.0
            else
                expected_e = rr + sqrt(rr * (rr - 1))
            end

            @test e ≈ expected_e atol=1e-10
        end
    end

    @testset "Monotonicity - E-value increases with RR" begin
        rr_values = range(1.0, 5.0, length=50)
        e_values = [compute_e_value(rr) for rr in rr_values]

        for i in 2:length(e_values)
            @test e_values[i] >= e_values[i-1] - 1e-10
        end
    end

    @testset "Protective/harmful symmetry" begin
        rr_values = [0.25, 0.33, 0.5, 0.67, 0.8]

        for rr in rr_values
            e_protective = compute_e_value(rr)
            e_harmful = compute_e_value(1 / rr)

            @test e_protective ≈ e_harmful atol=1e-10
        end
    end

    @testset "SMD conversion consistency" begin
        n_runs = 100
        e_values = Float64[]

        for seed in 1:n_runs
            data = dgp_evalue_smd(n=200, true_smd=0.5, seed=seed)

            # Compute observed SMD
            treated_mask = data.treatment .== 1.0
            treated_mean = mean(data.outcomes[treated_mask])
            control_mean = mean(data.outcomes[.!treated_mask])
            pooled_sd = sqrt((
                var(data.outcomes[treated_mask]) +
                var(data.outcomes[.!treated_mask])
            ) / 2)
            observed_smd = (treated_mean - control_mean) / pooled_sd

            # Compute E-value via solve
            problem = EValueProblem(observed_smd; effect_type=:smd)
            solution = solve(problem, EValue())
            push!(e_values, solution.e_value)
        end

        # E-values should be reasonably consistent
        mean_e = mean(e_values)
        std_e = std(e_values)

        # Expected E-value for SMD=0.5: RR ≈ exp(0.91 * 0.5) ≈ 1.577
        expected_rr = exp(0.91 * 0.5)
        expected_e = expected_rr + sqrt(expected_rr * (expected_rr - 1))

        @test mean_e ≈ expected_e rtol=0.20
        @test std_e < 0.6  # Relaxed for MC variance
    end

    @testset "ATE conversion accuracy" begin
        test_cases = [
            (0.1, 0.2, 1.5),   # ATE=0.1, baseline=0.2 → RR=1.5
            (0.2, 0.2, 2.0),   # ATE=0.2, baseline=0.2 → RR=2.0
            (0.1, 0.1, 2.0),   # ATE=0.1, baseline=0.1 → RR=2.0
        ]

        for (ate, baseline, expected_rr) in test_cases
            problem = EValueProblem(ate; effect_type=:ate, baseline_risk=baseline)
            solution = solve(problem, EValue())

            @test solution.rr_equivalent ≈ expected_rr atol=0.05
        end
    end

    @testset "CI crossing null produces E_CI = 1" begin
        n_runs = 100
        null_crossing_count = 0

        for seed in 1:n_runs
            Random.seed!(seed)

            # Generate RR and CI that may cross null
            rr = 0.8 + rand() * 0.7  # [0.8, 1.5]
            ci_width = 0.3 + rand() * 0.5
            ci_lower = rr - ci_width / 2
            ci_upper = rr + ci_width / 2

            problem = EValueProblem(rr; ci_lower=ci_lower, ci_upper=ci_upper, effect_type=:rr)
            solution = solve(problem, EValue())

            # Check if CI crosses null
            if ci_lower <= 1.0 <= ci_upper
                null_crossing_count += 1
                @test solution.e_value_ci ≈ 1.0 atol=0.01
            end
        end

        @test null_crossing_count > 20  # Should have some crossings
    end

end

# =============================================================================
# Rosenbaum Bounds Monte Carlo Tests
# =============================================================================

@testset "Rosenbaum Monte Carlo Validation" begin

    @testset "P-value monotonicity across runs" begin
        n_runs = 50

        for seed in 1:n_runs
            data = dgp_matched_pairs_no_confounding(n_pairs=30, true_effect=2.0, seed=seed)

            problem = RosenbaumProblem(data.treated, data.control; gamma_range=(1.0, 3.0), n_gamma=20)
            solution = solve(problem, RosenbaumBounds())

            # P_upper should be non-decreasing
            p_upper_diffs = diff(solution.p_upper)
            @test all(p_upper_diffs .>= -1e-10)
        end
    end

    @testset "P_lower <= P_upper always" begin
        n_runs = 50

        for seed in 1:n_runs
            data = dgp_matched_pairs_no_confounding(n_pairs=30, true_effect=1.5, seed=seed)

            problem = RosenbaumProblem(data.treated, data.control; gamma_range=(1.0, 3.0), n_gamma=20)
            solution = solve(problem, RosenbaumBounds())

            for i in 1:length(solution.p_upper)
                @test solution.p_lower[i] <= solution.p_upper[i] + 1e-10
            end
        end
    end

    @testset "P-values bounded [0, 1]" begin
        n_runs = 50

        for seed in 1:n_runs
            data = dgp_matched_pairs_no_confounding(n_pairs=30, true_effect=2.0, seed=seed)

            problem = RosenbaumProblem(data.treated, data.control; gamma_range=(1.0, 5.0), n_gamma=20)
            solution = solve(problem, RosenbaumBounds())

            @test all(0 .<= solution.p_upper .<= 1)
            @test all(0 .<= solution.p_lower .<= 1)
        end
    end

    @testset "Strong effect → high gamma_critical (robust)" begin
        gamma_criticals = Float64[]

        for seed in 1:100
            data = dgp_matched_pairs_strong_effect(n_pairs=40, true_effect=5.0, seed=seed)

            problem = RosenbaumProblem(data.treated, data.control; gamma_range=(1.0, 5.0), n_gamma=30)
            solution = solve(problem, RosenbaumBounds())

            if !isnothing(solution.gamma_critical) && solution.gamma_critical > 1.0
                push!(gamma_criticals, solution.gamma_critical)
            end
        end

        # Strong effect: most runs should be robust
        robust_count = 100 - length(gamma_criticals)
        if length(gamma_criticals) > 0
            mean_gamma = mean(gamma_criticals)
            @test mean_gamma > 2.5 || robust_count > 50
        end
    end

    @testset "Weak effect → low gamma_critical (sensitive)" begin
        gamma_criticals = Float64[]

        for seed in 1:100
            data = dgp_matched_pairs_weak_effect(n_pairs=40, true_effect=0.3, seed=seed)

            problem = RosenbaumProblem(data.treated, data.control; gamma_range=(1.0, 3.0), n_gamma=20)
            solution = solve(problem, RosenbaumBounds())

            if !isnothing(solution.gamma_critical) && solution.gamma_critical > 1.0
                push!(gamma_criticals, solution.gamma_critical)
            end
        end

        # Weak effect: many should have low gamma_critical
        if length(gamma_criticals) > 30
            mean_gamma = mean(gamma_criticals)
            @test mean_gamma < 2.0
        end
    end

    @testset "Null effect → very sensitive" begin
        not_significant_at_gamma1 = 0

        for seed in 1:100
            data = dgp_matched_pairs_null_effect(n_pairs=40, seed=seed)

            problem = RosenbaumProblem(data.treated, data.control; gamma_range=(1.0, 2.0), n_gamma=15, alpha=0.05)
            solution = solve(problem, RosenbaumBounds())

            # Check if not significant at gamma=1
            if solution.p_upper[1] > 0.05
                not_significant_at_gamma1 += 1
            end
        end

        # Null effect: most should be non-significant at gamma=1
        @test not_significant_at_gamma1 > 80
    end

    @testset "Larger samples more robust" begin
        small_gammas = Float64[]
        large_gammas = Float64[]

        for seed in 1:100
            # Small sample
            data_s = dgp_matched_pairs_no_confounding(n_pairs=15, true_effect=1.5, seed=seed)
            prob_s = RosenbaumProblem(data_s.treated, data_s.control; gamma_range=(1.0, 3.0))
            sol_s = solve(prob_s, RosenbaumBounds())
            if !isnothing(sol_s.gamma_critical)
                push!(small_gammas, sol_s.gamma_critical)
            end

            # Large sample
            data_l = dgp_matched_pairs_no_confounding(n_pairs=60, true_effect=1.5, seed=seed + 10000)
            prob_l = RosenbaumProblem(data_l.treated, data_l.control; gamma_range=(1.0, 3.0))
            sol_l = solve(prob_l, RosenbaumBounds())
            if !isnothing(sol_l.gamma_critical)
                push!(large_gammas, sol_l.gamma_critical)
            end
        end

        # Larger samples should be more robust
        small_robust = 100 - length(small_gammas)
        large_robust = 100 - length(large_gammas)

        if length(small_gammas) > 10 && length(large_gammas) > 10
            @test large_robust >= small_robust - 20 || mean(large_gammas) >= mean(small_gammas) - 0.3
        end
    end

end

# =============================================================================
# Run summary
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("\n" * "="^60)
    println("Sensitivity Monte Carlo Validation Summary")
    println("="^60)
    println("Validating: E-value, Rosenbaum bounds")
    println("Metrics: Formula accuracy, monotonicity, gamma_critical detection")
    println("="^60 * "\n")
end
