#=
Adversarial Tests for Julia Sensitivity Analysis Module.

Tests edge cases, boundary conditions, and error handling for:
- E-value: Invalid inputs, extreme values, edge case conversions
- Rosenbaum bounds: Invalid inputs, degenerate data, numerical stability

Session 69: Julia Sensitivity Validation
=#

using Test
using Random
using Statistics
using CausalEstimators

# =============================================================================
# E-Value Input Validation
# =============================================================================

@testset "EValue Input Validation" begin

    @testset "Negative RR throws" begin
        @test_throws ArgumentError compute_e_value(-1.0)
        @test_throws ArgumentError compute_e_value(-0.5)
    end

    @testset "Zero RR throws" begin
        @test_throws ArgumentError compute_e_value(0.0)
    end

    @testset "Invalid effect_type symbol throws" begin
        @test_throws ArgumentError effect_type_from_symbol(:invalid)
        @test_throws ArgumentError effect_type_from_symbol(:unknown)
    end

    @testset "ATE without baseline_risk throws" begin
        @test_throws ArgumentError EValueProblem(0.1; effect_type=:ate)
    end

    @testset "Invalid baseline_risk throws" begin
        @test_throws ArgumentError EValueProblem(0.1; effect_type=:ate, baseline_risk=0.0)
        @test_throws ArgumentError EValueProblem(0.1; effect_type=:ate, baseline_risk=1.0)
        @test_throws ArgumentError EValueProblem(0.1; effect_type=:ate, baseline_risk=-0.1)
        @test_throws ArgumentError EValueProblem(0.1; effect_type=:ate, baseline_risk=1.5)
    end

    @testset "ATE creating invalid treated risk throws" begin
        # baseline = 0.2, ATE = 0.9 → treated_risk = 1.1 > 1
        @test_throws ArgumentError EValueProblem(0.9; effect_type=:ate, baseline_risk=0.2)

        # baseline = 0.2, ATE = -0.3 → treated_risk = -0.1 < 0
        @test_throws ArgumentError EValueProblem(-0.3; effect_type=:ate, baseline_risk=0.2)
    end

    @testset "CI ordering validation" begin
        @test_throws ArgumentError EValueProblem(2.0; ci_lower=3.0, ci_upper=1.5)
    end

end

# =============================================================================
# E-Value Edge Cases
# =============================================================================

@testset "EValue Edge Cases" begin

    @testset "RR exactly 1.0 gives E = 1.0" begin
        e = compute_e_value(1.0)
        @test e ≈ 1.0 atol=1e-10
    end

    @testset "RR very close to 1.0" begin
        e1 = compute_e_value(1.0 + 1e-10)
        e2 = compute_e_value(1.0 - 1e-10)

        @test e1 ≈ 1.0 atol=0.01
        @test e2 ≈ 1.0 atol=0.01
    end

    @testset "Very large RR" begin
        e = compute_e_value(100.0)
        expected = 100.0 + sqrt(100.0 * 99.0)
        @test e ≈ expected rtol=1e-6
        @test isfinite(e)
    end

    @testset "Very small protective RR" begin
        e = compute_e_value(0.01)
        # Should invert to RR = 100
        expected = 100.0 + sqrt(100.0 * 99.0)
        @test e ≈ expected rtol=1e-6
    end

    @testset "SMD = 0 gives E = 1" begin
        problem = EValueProblem(0.0; effect_type=:smd)
        solution = solve(problem, EValue())
        @test solution.e_value ≈ 1.0 atol=0.01
    end

    @testset "Very large positive SMD" begin
        problem = EValueProblem(5.0; effect_type=:smd)
        solution = solve(problem, EValue())

        expected_rr = exp(0.91 * 5.0)
        expected_e = expected_rr + sqrt(expected_rr * (expected_rr - 1))
        @test solution.e_value ≈ expected_e rtol=0.01
    end

    @testset "Very large negative SMD (protective)" begin
        problem = EValueProblem(-5.0; effect_type=:smd)
        solution = solve(problem, EValue())

        # exp(0.91 * -5) ≈ 0.01, inverted to 100
        @test solution.e_value > 10.0
    end

    @testset "ATE = 0 gives RR = 1, E = 1" begin
        problem = EValueProblem(0.0; effect_type=:ate, baseline_risk=0.5)
        solution = solve(problem, EValue())

        @test solution.rr_equivalent ≈ 1.0 atol=0.01
        @test solution.e_value ≈ 1.0 atol=0.01
    end

    @testset "Very small baseline_risk" begin
        problem = EValueProblem(0.01; effect_type=:ate, baseline_risk=0.01)
        solution = solve(problem, EValue())

        # RR = (0.01 + 0.01) / 0.01 = 2.0
        @test solution.rr_equivalent ≈ 2.0 atol=0.01
    end

end

# =============================================================================
# E-Value Interpretation
# =============================================================================

@testset "EValue Interpretation" begin

    @testset "Interpretation always present and non-empty" begin
        test_cases = [
            (1.0, :rr, nothing),
            (2.0, :rr, nothing),
            (0.5, :rr, nothing),
            (0.5, :smd, nothing),
            (0.1, :ate, 0.3),
        ]

        for (estimate, effect_type, baseline) in test_cases
            kwargs = isnothing(baseline) ? () : (baseline_risk=baseline,)
            problem = EValueProblem(estimate; effect_type=effect_type, kwargs...)
            solution = solve(problem, EValue())

            @test !isempty(solution.interpretation)
            @test length(solution.interpretation) > 20
        end
    end

    @testset "Interpretation contains robustness assessment" begin
        problem = EValueProblem(3.0; effect_type=:rr)
        solution = solve(problem, EValue())

        @test occursin("robust", lowercase(solution.interpretation))
    end

end

# =============================================================================
# E-Value Data Types
# =============================================================================

@testset "EValue Data Types" begin

    @testset "Integer estimate works" begin
        e = compute_e_value(2)
        @test e ≈ 2.0 + sqrt(2.0) rtol=0.01
    end

    @testset "Float32 estimate works" begin
        e = compute_e_value(Float32(2.0))
        @test e ≈ 2.0 + sqrt(2.0) rtol=0.01
    end

end

# =============================================================================
# Rosenbaum Bounds Input Validation
# =============================================================================

@testset "Rosenbaum Input Validation" begin

    @testset "Unequal lengths throws" begin
        @test_throws ArgumentError RosenbaumProblem([1.0, 2.0], [1.0])
        @test_throws ArgumentError RosenbaumProblem([1.0], [1.0, 2.0])
    end

    @testset "Empty arrays throws" begin
        @test_throws ArgumentError RosenbaumProblem(Float64[], Float64[])
    end

    @testset "Gamma lower < 1 throws" begin
        @test_throws ArgumentError RosenbaumProblem([1.0], [0.5]; gamma_range=(0.5, 2.0))
        @test_throws ArgumentError RosenbaumProblem([1.0], [0.5]; gamma_range=(0.0, 2.0))
        @test_throws ArgumentError RosenbaumProblem([1.0], [0.5]; gamma_range=(-1.0, 2.0))
    end

    @testset "Gamma upper <= lower throws" begin
        @test_throws ArgumentError RosenbaumProblem([1.0], [0.5]; gamma_range=(2.0, 1.5))
        @test_throws ArgumentError RosenbaumProblem([1.0], [0.5]; gamma_range=(2.0, 2.0))
    end

    @testset "n_gamma < 2 throws" begin
        @test_throws ArgumentError RosenbaumProblem([1.0], [0.5]; n_gamma=1)
        @test_throws ArgumentError RosenbaumProblem([1.0], [0.5]; n_gamma=0)
    end

    @testset "Invalid alpha throws" begin
        @test_throws ArgumentError RosenbaumProblem([1.0], [0.5]; alpha=0.0)
        @test_throws ArgumentError RosenbaumProblem([1.0], [0.5]; alpha=1.0)
        @test_throws ArgumentError RosenbaumProblem([1.0], [0.5]; alpha=-0.1)
        @test_throws ArgumentError RosenbaumProblem([1.0], [0.5]; alpha=1.5)
    end

end

# =============================================================================
# Rosenbaum Bounds Edge Cases
# =============================================================================

@testset "Rosenbaum Edge Cases" begin

    @testset "Single pair" begin
        problem = RosenbaumProblem([10.0], [5.0])
        solution = solve(problem, RosenbaumBounds())

        @test solution.n_pairs == 1
        @test solution.observed_statistic == 1.0
    end

    @testset "All ties (zero differences)" begin
        problem = RosenbaumProblem([10.0, 10.0, 10.0], [10.0, 10.0, 10.0])
        solution = solve(problem, RosenbaumBounds())

        @test solution.n_pairs == 0
        @test solution.gamma_critical ≈ 1.0 atol=1e-10
    end

    @testset "Custom gamma range" begin
        treated = [10.0, 12.0, 14.0]
        control = [5.0, 6.0, 7.0]

        problem = RosenbaumProblem(treated, control; gamma_range=(2.0, 5.0), n_gamma=5)
        solution = solve(problem, RosenbaumBounds())

        @test length(solution.gamma_values) == 5
        @test solution.gamma_values[1] ≈ 2.0 atol=1e-10
        @test solution.gamma_values[end] ≈ 5.0 atol=1e-10
    end

    @testset "Custom alpha" begin
        treated = [10.0, 12.0, 14.0, 16.0, 18.0]
        control = [5.0, 6.0, 7.0, 8.0, 9.0]

        problem = RosenbaumProblem(treated, control; alpha=0.10)
        solution = solve(problem, RosenbaumBounds())

        @test solution.alpha ≈ 0.10 atol=1e-10
    end

    @testset "All positive differences (strong effect)" begin
        Random.seed!(42)
        n = 20
        treated = randn(n) .+ 10.0  # Large positive effect
        control = randn(n)

        problem = RosenbaumProblem(treated, control; gamma_range=(1.0, 5.0))
        solution = solve(problem, RosenbaumBounds())

        # Should be robust
        @test isnothing(solution.gamma_critical) || solution.gamma_critical > 3.0
    end

    @testset "All negative differences (control higher)" begin
        Random.seed!(42)
        n = 20
        treated = randn(n)
        control = randn(n) .+ 10.0  # Control much higher

        problem = RosenbaumProblem(treated, control; gamma_range=(1.0, 5.0))
        solution = solve(problem, RosenbaumBounds())

        # P_upper at gamma=1 should be high (effect in wrong direction)
        @test solution.p_upper[1] > 0.5
    end

    @testset "Mixed sign differences" begin
        treated = [5.0, 1.0, 5.0, 1.0, 5.0, 1.0]
        control = [0.0, 6.0, 0.0, 6.0, 0.0, 6.0]

        problem = RosenbaumProblem(treated, control)
        solution = solve(problem, RosenbaumBounds())

        @test length(solution.gamma_values) > 0
    end

    @testset "Interpretation contains robustness assessment" begin
        treated = [10.0, 12.0, 14.0, 16.0]
        control = [5.0, 6.0, 7.0, 8.0]

        problem = RosenbaumProblem(treated, control)
        solution = solve(problem, RosenbaumBounds())

        @test occursin("robust", lowercase(solution.interpretation)) ||
              occursin("sensitive", lowercase(solution.interpretation))
    end

end

# =============================================================================
# Rosenbaum Numerical Stability
# =============================================================================

@testset "Rosenbaum Numerical Stability" begin

    @testset "Very large outcomes" begin
        scale = 1e8
        Random.seed!(42)
        treated = randn(20) .* scale .+ scale
        control = randn(20) .* scale

        problem = RosenbaumProblem(treated, control)
        solution = solve(problem, RosenbaumBounds())

        @test !any(isnan, solution.p_upper)
        @test !any(isnan, solution.p_lower)
    end

    @testset "Very small outcomes" begin
        scale = 1e-8
        Random.seed!(42)
        treated = randn(20) .* scale .+ 2 * scale
        control = randn(20) .* scale

        problem = RosenbaumProblem(treated, control)
        solution = solve(problem, RosenbaumBounds())

        @test !any(isnan, solution.p_upper)
        @test !any(isnan, solution.p_lower)
    end

    @testset "Mixed scale outcomes" begin
        treated = [1e10, 1e-10, 1e5, 1e-5, 1.0, 0.5, 2.0, 3.0, 4.0, 5.0]
        control = zeros(10)

        problem = RosenbaumProblem(treated, control)
        solution = solve(problem, RosenbaumBounds())

        @test length(solution.gamma_values) > 0
    end

    @testset "Nearly identical pairs" begin
        eps = 1e-15
        treated = [1.0 + eps, 2.0 + eps, 3.0 + eps, 4.0 + eps, 5.0 + eps]
        control = [1.0, 2.0, 3.0, 4.0, 5.0]

        problem = RosenbaumProblem(treated, control)
        solution = solve(problem, RosenbaumBounds())

        @test length(solution.gamma_values) > 0
    end

end

# =============================================================================
# Rosenbaum Gamma Range Handling
# =============================================================================

@testset "Rosenbaum Gamma Range" begin

    @testset "Wide gamma range" begin
        Random.seed!(42)
        treated = randn(20) .+ 2.0
        control = randn(20)

        problem = RosenbaumProblem(treated, control; gamma_range=(1.0, 100.0), n_gamma=50)
        solution = solve(problem, RosenbaumBounds())

        @test length(solution.gamma_values) == 50
        @test solution.gamma_values[1] ≈ 1.0 atol=1e-10
        @test solution.gamma_values[end] ≈ 100.0 atol=1e-10
    end

    @testset "Narrow gamma range" begin
        Random.seed!(42)
        treated = randn(20) .+ 2.0
        control = randn(20)

        problem = RosenbaumProblem(treated, control; gamma_range=(1.0, 1.01), n_gamma=10)
        solution = solve(problem, RosenbaumBounds())

        @test length(solution.gamma_values) == 10
    end

    @testset "Large n_gamma" begin
        Random.seed!(42)
        treated = randn(20) .+ 2.0
        control = randn(20)

        problem = RosenbaumProblem(treated, control; gamma_range=(1.0, 3.0), n_gamma=100)
        solution = solve(problem, RosenbaumBounds())

        @test length(solution.gamma_values) == 100
    end

    @testset "Gamma grid is evenly spaced" begin
        treated = [10.0, 12.0, 14.0]
        control = [5.0, 6.0, 7.0]

        problem = RosenbaumProblem(treated, control; gamma_range=(1.0, 3.0), n_gamma=5)
        solution = solve(problem, RosenbaumBounds())

        diffs = diff(solution.gamma_values)
        @test all(d ≈ diffs[1] for d in diffs)
    end

end

# =============================================================================
# Rosenbaum Data Types
# =============================================================================

@testset "Rosenbaum Data Types" begin

    @testset "Integer arrays" begin
        treated = [10, 12, 14, 16, 18]
        control = [5, 6, 7, 8, 9]

        problem = RosenbaumProblem(Float64.(treated), Float64.(control))
        solution = solve(problem, RosenbaumBounds())

        @test solution.n_pairs == 5
    end

    @testset "Float32 arrays converted" begin
        treated = Float32[10.0, 12.0, 14.0, 16.0, 18.0]
        control = Float32[5.0, 6.0, 7.0, 8.0, 9.0]

        problem = RosenbaumProblem(Float64.(treated), Float64.(control))
        solution = solve(problem, RosenbaumBounds())

        @test solution.n_pairs == 5
    end

end

# =============================================================================
# Result Structure Tests
# =============================================================================

@testset "Solution Structure" begin

    @testset "EValueSolution has all fields" begin
        problem = EValueProblem(2.0; ci_lower=1.5, ci_upper=2.5, effect_type=:rr)
        solution = solve(problem, EValue())

        @test hasfield(typeof(solution), :e_value)
        @test hasfield(typeof(solution), :e_value_ci)
        @test hasfield(typeof(solution), :rr_equivalent)
        @test hasfield(typeof(solution), :effect_type)
        @test hasfield(typeof(solution), :interpretation)
        @test hasfield(typeof(solution), :original_problem)
    end

    @testset "RosenbaumSolution has all fields" begin
        problem = RosenbaumProblem([10.0, 12.0, 14.0], [5.0, 6.0, 7.0])
        solution = solve(problem, RosenbaumBounds())

        @test hasfield(typeof(solution), :gamma_values)
        @test hasfield(typeof(solution), :p_upper)
        @test hasfield(typeof(solution), :p_lower)
        @test hasfield(typeof(solution), :gamma_critical)
        @test hasfield(typeof(solution), :observed_statistic)
        @test hasfield(typeof(solution), :n_pairs)
        @test hasfield(typeof(solution), :alpha)
        @test hasfield(typeof(solution), :interpretation)
        @test hasfield(typeof(solution), :original_problem)
    end

    @testset "Array lengths match n_gamma" begin
        n_gamma = 25
        problem = RosenbaumProblem([10.0, 12.0, 14.0], [5.0, 6.0, 7.0]; n_gamma=n_gamma)
        solution = solve(problem, RosenbaumBounds())

        @test length(solution.gamma_values) == n_gamma
        @test length(solution.p_upper) == n_gamma
        @test length(solution.p_lower) == n_gamma
    end

    @testset "Alpha recorded in solution" begin
        for alpha in [0.01, 0.05, 0.10]
            problem = RosenbaumProblem([10.0, 12.0, 14.0], [5.0, 6.0, 7.0]; alpha=alpha)
            solution = solve(problem, RosenbaumBounds())

            @test solution.alpha ≈ alpha atol=1e-10
        end
    end

end

# =============================================================================
# Determinism Tests
# =============================================================================

@testset "Determinism" begin

    @testset "E-value is deterministic" begin
        problem = EValueProblem(2.5; ci_lower=1.8, ci_upper=3.5, effect_type=:rr)

        sol1 = solve(problem, EValue())
        sol2 = solve(problem, EValue())

        @test sol1.e_value ≈ sol2.e_value atol=1e-15
        @test sol1.e_value_ci ≈ sol2.e_value_ci atol=1e-15
    end

    @testset "Rosenbaum bounds are deterministic" begin
        treated = [10.0, 12.0, 14.0, 16.0, 18.0]
        control = [5.0, 6.0, 7.0, 8.0, 9.0]

        problem = RosenbaumProblem(treated, control)
        sol1 = solve(problem, RosenbaumBounds())
        sol2 = solve(problem, RosenbaumBounds())

        @test sol1.gamma_critical == sol2.gamma_critical
        @test sol1.observed_statistic == sol2.observed_statistic
        @test sol1.p_upper ≈ sol2.p_upper atol=1e-15
    end

end
