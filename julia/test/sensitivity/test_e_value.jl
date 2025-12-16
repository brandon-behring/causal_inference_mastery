#=
E-Value Sensitivity Analysis Tests

Test coverage:
- Layer 1: Known-answer tests (formula validation)
- Layer 2: CI handling tests
- Layer 3: Effect type conversion tests
- Layer 4: Edge case tests
- Layer 5: Input validation tests
- Layer 6: Integration tests
=#

using Test
using CausalEstimators

# ============================================================================
# Layer 1: Known-Answer Tests
# ============================================================================

@testset "EValue Known-Answer Tests" begin
    @testset "E-value formula: RR = 1.5" begin
        # E = RR + sqrt(RR * (RR - 1)) = 1.5 + sqrt(1.5 * 0.5) = 1.5 + 0.866 = 2.366
        e = compute_e_value(1.5)
        expected = 1.5 + sqrt(1.5 * 0.5)
        @test e ≈ expected atol=1e-10
        @test e ≈ 2.366 atol=0.001
    end

    @testset "E-value formula: RR = 2.0" begin
        # E = 2 + sqrt(2 * 1) = 2 + sqrt(2) = 3.414
        e = compute_e_value(2.0)
        expected = 2.0 + sqrt(2.0)
        @test e ≈ expected atol=1e-10
        @test e ≈ 3.414 atol=0.001
    end

    @testset "E-value formula: RR = 3.0" begin
        # E = 3 + sqrt(3 * 2) = 3 + sqrt(6) = 5.449
        e = compute_e_value(3.0)
        expected = 3.0 + sqrt(6.0)
        @test e ≈ expected atol=1e-10
        @test e ≈ 5.449 atol=0.001
    end

    @testset "E-value for null effect: RR = 1.0" begin
        e = compute_e_value(1.0)
        @test e ≈ 1.0 atol=1e-10
    end

    @testset "E-value via solve()" begin
        problem = EValueProblem(2.0; effect_type=:rr)
        solution = solve(problem, EValue())
        @test solution.e_value ≈ 2.0 + sqrt(2.0) atol=1e-10
        @test solution.rr_equivalent ≈ 2.0 atol=1e-10
    end

    @testset "Protective effect: RR = 0.5 inverts to RR = 2.0" begin
        e = compute_e_value(0.5)
        e_inverted = compute_e_value(2.0)
        @test e ≈ e_inverted atol=1e-10
    end
end

# ============================================================================
# Layer 2: CI Handling Tests
# ============================================================================

@testset "EValue CI Tests" begin
    @testset "CI includes null → E_CI = 1.0" begin
        problem = EValueProblem(1.5; ci_lower=0.8, ci_upper=2.5, effect_type=:rr)
        solution = solve(problem, EValue())
        @test solution.e_value_ci ≈ 1.0 atol=1e-10
    end

    @testset "CI excludes null (harmful effect) → E_CI from lower bound" begin
        problem = EValueProblem(2.5; ci_lower=1.8, ci_upper=3.5, effect_type=:rr)
        solution = solve(problem, EValue())
        # For harmful effects (RR > 1), lower bound closest to null
        expected_e_ci = compute_e_value(1.8)
        @test solution.e_value_ci ≈ expected_e_ci atol=1e-10
    end

    @testset "CI excludes null (protective effect) → E_CI from upper bound" begin
        problem = EValueProblem(0.4; ci_lower=0.2, ci_upper=0.7, effect_type=:rr)
        solution = solve(problem, EValue())
        # For protective effects (RR < 1), upper bound closest to null
        expected_e_ci = compute_e_value(0.7)
        @test solution.e_value_ci ≈ expected_e_ci atol=1e-10
    end

    @testset "No CI provided → E_CI equals E" begin
        problem = EValueProblem(2.0; effect_type=:rr)
        solution = solve(problem, EValue())
        @test solution.e_value_ci ≈ solution.e_value atol=1e-10
    end
end

# ============================================================================
# Layer 3: Effect Type Conversion Tests
# ============================================================================

@testset "EValue Effect Conversions" begin
    @testset "SMD to RR conversion" begin
        # RR ≈ exp(0.91 * d)
        d = 0.5
        expected_rr = exp(0.91 * 0.5)
        @test smd_to_rr(d) ≈ expected_rr atol=1e-10

        problem = EValueProblem(0.5; effect_type=:smd)
        solution = solve(problem, EValue())
        @test solution.rr_equivalent ≈ expected_rr atol=1e-10
    end

    @testset "ATE to RR conversion" begin
        # RR = (baseline + ATE) / baseline
        ate = 0.1
        baseline = 0.2
        expected_rr = (0.2 + 0.1) / 0.2  # = 1.5
        @test ate_to_rr(ate, baseline) ≈ expected_rr atol=1e-10

        problem = EValueProblem(0.1; effect_type=:ate, baseline_risk=0.2)
        solution = solve(problem, EValue())
        @test solution.rr_equivalent ≈ 1.5 atol=1e-10
    end

    @testset "OR treated as RR" begin
        problem = EValueProblem(2.0; effect_type=:or)
        solution = solve(problem, EValue())
        @test solution.rr_equivalent ≈ 2.0 atol=1e-10
    end

    @testset "HR treated as RR" begin
        problem = EValueProblem(1.8; effect_type=:hr)
        solution = solve(problem, EValue())
        @test solution.rr_equivalent ≈ 1.8 atol=1e-10
    end

    @testset "Symbol effect_type works" begin
        for (sym, expected) in [(:rr, RR), (:or, OR), (:hr, HR), (:smd, SMD)]
            @test effect_type_from_symbol(sym) == expected
        end
    end
end

# ============================================================================
# Layer 4: Edge Case Tests
# ============================================================================

@testset "EValue Edge Cases" begin
    @testset "Very large RR" begin
        e = compute_e_value(10.0)
        @test e > 10.0  # E-value > RR always
        @test isfinite(e)
    end

    @testset "RR very close to 1" begin
        e = compute_e_value(1.001)
        @test e ≈ 1.0 atol=0.1
    end

    @testset "Very small protective effect" begin
        e = compute_e_value(0.1)
        @test e > 10  # Inverted to RR = 10
        @test isfinite(e)
    end

    @testset "Interpretation contains robustness assessment" begin
        problem = EValueProblem(3.0; effect_type=:rr)
        solution = solve(problem, EValue())
        @test occursin("robust", lowercase(solution.interpretation))
    end

    @testset "Interpretation mentions effect type" begin
        problem = EValueProblem(2.0; ci_lower=1.5, ci_upper=2.8, effect_type=:smd)
        solution = solve(problem, EValue())
        @test occursin("SMD", solution.interpretation)
    end
end

# ============================================================================
# Layer 5: Input Validation Tests
# ============================================================================

@testset "EValue Input Validation" begin
    @testset "Negative RR throws" begin
        @test_throws ArgumentError compute_e_value(-1.0)
    end

    @testset "Zero RR throws" begin
        @test_throws ArgumentError compute_e_value(0.0)
    end

    @testset "ATE without baseline_risk throws" begin
        @test_throws ArgumentError EValueProblem(0.1; effect_type=:ate)
    end

    @testset "Invalid baseline_risk throws" begin
        @test_throws ArgumentError EValueProblem(0.1; effect_type=:ate, baseline_risk=0.0)
        @test_throws ArgumentError EValueProblem(0.1; effect_type=:ate, baseline_risk=1.0)
        @test_throws ArgumentError EValueProblem(0.1; effect_type=:ate, baseline_risk=-0.1)
    end

    @testset "ATE creating invalid risk throws" begin
        # baseline = 0.2, ATE = 0.9 → new risk = 1.1 > 1
        @test_throws ArgumentError EValueProblem(0.9; effect_type=:ate, baseline_risk=0.2)
    end

    @testset "Invalid effect_type symbol throws" begin
        @test_throws ArgumentError effect_type_from_symbol(:invalid)
    end

    @testset "CI ordering validation" begin
        @test_throws ArgumentError EValueProblem(2.0; ci_lower=3.0, ci_upper=1.5)
    end
end

# ============================================================================
# Layer 6: Integration Tests
# ============================================================================

@testset "EValue Integration Tests" begin
    @testset "Typical OR with CI" begin
        # Typical epidemiological study: OR = 2.3, 95% CI [1.4, 3.8]
        problem = EValueProblem(2.3; ci_lower=1.4, ci_upper=3.8, effect_type=:or)
        solution = solve(problem, EValue())

        @test solution.e_value > 2.0
        @test solution.e_value_ci > 1.0
        @test solution.rr_equivalent ≈ 2.3 atol=1e-10
        @test solution.effect_type == OR
        @test !isempty(solution.interpretation)
    end

    @testset "Typical SMD with CI" begin
        # Cohen's d = 0.6 with CI [0.3, 0.9]
        problem = EValueProblem(0.6; ci_lower=0.3, ci_upper=0.9, effect_type=:smd)
        solution = solve(problem, EValue())

        @test solution.e_value > 1.0
        @test solution.rr_equivalent ≈ exp(0.91 * 0.6) atol=1e-10
        @test solution.effect_type == SMD
    end

    @testset "Problem-Estimator-Solution pattern" begin
        problem = EValueProblem(2.0; effect_type=:rr)
        estimator = EValue()
        solution = solve(problem, estimator)

        @test solution isa EValueSolution
        @test solution.original_problem === problem
    end
end
