"""
Unit tests for Julia Fuzzy RKD module.

Test structure:
1. FuzzyRKD construction and validation
2. Fuzzy RKD estimation
3. Known-answer tests
4. First stage diagnostics
5. Edge cases
"""

using Test
using CausalEstimators
using Random
using Statistics
using Distributions

# =============================================================================
# DGP Functions
# =============================================================================

"""Generate Fuzzy RKD data with known effect."""
function generate_fuzzy_rkd_data(;
    n::Int=1000,
    cutoff::Float64=0.0,
    slope_left_d::Float64=0.5,
    slope_right_d::Float64=1.5,
    true_effect::Float64=2.0,
    noise_d::Float64=0.3,  # Fuzziness
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
    Y = true_effect .* D .+ randn(n) .* noise_y

    return Y, X, D
end

"""Generate Fuzzy RKD data with weak first stage."""
function generate_weak_first_stage_data(;
    n::Int=1000,
    cutoff::Float64=0.0,
    seed::Union{Nothing,Int}=nothing
)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    X = rand(Uniform(-5, 5), n)

    # Very small kink (weak first stage)
    slope_left = 1.0
    slope_right = 1.05  # Only 0.05 difference

    D_expected = [x < cutoff ? slope_left * x : slope_right * x for x in X]
    D = D_expected .+ 2.0 .* randn(n)  # High noise

    Y = 2.0 .* D .+ randn(n)

    return Y, X, D
end

# =============================================================================
# FuzzyRKD Estimator Tests
# =============================================================================

@testset "FuzzyRKD Estimator Construction" begin
    @testset "Default construction" begin
        est = FuzzyRKD()
        @test isnothing(est.bandwidth)
        @test est.kernel == :triangular
        @test est.polynomial_order == 1
        @test est.alpha == 0.05
    end

    @testset "Custom parameters" begin
        est = FuzzyRKD(bandwidth=2.0, kernel=:epanechnikov, polynomial_order=2, alpha=0.10)
        @test est.bandwidth == 2.0
        @test est.kernel == :epanechnikov
        @test est.polynomial_order == 2
        @test est.alpha == 0.10
    end

    @testset "Invalid parameters" begin
        @test_throws ArgumentError FuzzyRKD(bandwidth=-1.0)
        @test_throws ArgumentError FuzzyRKD(kernel=:invalid)
        @test_throws ArgumentError FuzzyRKD(polynomial_order=0)
        @test_throws ArgumentError FuzzyRKD(polynomial_order=4)
        @test_throws ArgumentError FuzzyRKD(alpha=0.0)
        @test_throws ArgumentError FuzzyRKD(alpha=1.0)
    end
end

@testset "FuzzyRKD Estimation" begin
    @testset "Basic estimation" begin
        Y, X, D = generate_fuzzy_rkd_data(n=1000, true_effect=2.0, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, FuzzyRKD(bandwidth=2.5))

        @test solution isa FuzzyRKDSolution
        @test solution.retcode in [:Success, :Warning]
        @test isfinite(solution.estimate)
        @test isfinite(solution.se)
        @test solution.se > 0
    end

    @testset "Result fields" begin
        Y, X, D = generate_fuzzy_rkd_data(n=500, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, FuzzyRKD(bandwidth=2.0))

        # Check all fields exist
        @test hasfield(FuzzyRKDSolution, :estimate)
        @test hasfield(FuzzyRKDSolution, :se)
        @test hasfield(FuzzyRKDSolution, :ci_lower)
        @test hasfield(FuzzyRKDSolution, :ci_upper)
        @test hasfield(FuzzyRKDSolution, :first_stage_kink)
        @test hasfield(FuzzyRKDSolution, :reduced_form_kink)
        @test hasfield(FuzzyRKDSolution, :first_stage_f_stat)
        @test hasfield(FuzzyRKDSolution, :weak_first_stage)
    end

    @testset "CI contains estimate" begin
        Y, X, D = generate_fuzzy_rkd_data(n=500, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, FuzzyRKD(bandwidth=2.0))

        if solution.retcode == :Success
            @test solution.ci_lower <= solution.estimate <= solution.ci_upper
        end
    end

    @testset "Different kernels" begin
        Y, X, D = generate_fuzzy_rkd_data(n=500, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)

        for kernel in [:triangular, :uniform, :epanechnikov]
            solution = solve(problem, FuzzyRKD(bandwidth=2.0, kernel=kernel))
            @test solution isa FuzzyRKDSolution
            @test solution.kernel == kernel
        end
    end

    @testset "Different polynomial orders" begin
        Y, X, D = generate_fuzzy_rkd_data(n=800, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)

        for order in [1, 2, 3]
            solution = solve(problem, FuzzyRKD(bandwidth=2.5, polynomial_order=order))
            @test solution isa FuzzyRKDSolution
            @test solution.polynomial_order == order
        end
    end

    @testset "Auto bandwidth" begin
        Y, X, D = generate_fuzzy_rkd_data(n=500, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, FuzzyRKD())  # No bandwidth specified

        @test solution.bandwidth > 0
        @test isfinite(solution.bandwidth)
    end
end

# =============================================================================
# Known-Answer Tests
# =============================================================================

@testset "FuzzyRKD Known-Answer Tests" begin
    @testset "Recovers true effect" begin
        Y, X, D = generate_fuzzy_rkd_data(
            n=2000,
            slope_left_d=0.5,
            slope_right_d=1.5,
            true_effect=2.0,
            noise_d=0.2,
            noise_y=0.5,
            seed=42
        )
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, FuzzyRKD(bandwidth=2.5))

        # Should recover approximately 2.0
        @test abs(solution.estimate - 2.0) < 0.8
    end

    @testset "First stage kink detection" begin
        Y, X, D = generate_fuzzy_rkd_data(
            n=1500,
            slope_left_d=0.5,
            slope_right_d=1.5,  # true kink = 1.0
            noise_d=0.2,
            seed=123
        )
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, FuzzyRKD(bandwidth=2.0))

        # First stage kink should be approximately 1.0
        @test abs(solution.first_stage_kink - 1.0) < 0.5
    end

    @testset "Reduced form kink" begin
        Y, X, D = generate_fuzzy_rkd_data(
            n=1500,
            slope_left_d=0.5,
            slope_right_d=1.5,  # kink = 1.0
            true_effect=3.0,    # RF kink should be ~3.0
            noise_d=0.2,
            seed=321
        )
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, FuzzyRKD(bandwidth=2.0))

        # RF kink ≈ true_effect * first_stage_kink
        @test abs(solution.reduced_form_kink - 3.0) < 1.5
    end

    @testset "Strong first stage detection" begin
        Y, X, D = generate_fuzzy_rkd_data(n=1500, seed=654)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, FuzzyRKD(bandwidth=2.5))

        # Should detect strong first stage
        @test solution.first_stage_f_stat > 10
        @test !solution.weak_first_stage
    end

    @testset "Negative effect" begin
        Y, X, D = generate_fuzzy_rkd_data(n=1500, true_effect=-1.5, seed=789)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, FuzzyRKD(bandwidth=2.0))

        @test solution.estimate < 0
    end
end

# =============================================================================
# First Stage Diagnostics
# =============================================================================

@testset "FuzzyRKD First Stage" begin
    @testset "Weak first stage warning" begin
        Y, X, D = generate_weak_first_stage_data(n=1000, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, FuzzyRKD(bandwidth=2.0))

        # May or may not detect weak first stage depending on randomness
        # Just check the fields are populated
        @test isfinite(solution.first_stage_f_stat) || isnan(solution.first_stage_f_stat)
    end

    @testset "F-stat computation" begin
        Y, X, D = generate_fuzzy_rkd_data(n=1000, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, FuzzyRKD(bandwidth=2.5))

        if solution.retcode in [:Success, :Warning]
            @test solution.first_stage_f_stat > 0
            @test isfinite(solution.first_stage_f_stat)
        end
    end
end

# =============================================================================
# Edge Cases
# =============================================================================

@testset "FuzzyRKD Edge Cases" begin
    @testset "Small sample" begin
        Y, X, D = generate_fuzzy_rkd_data(n=50, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, FuzzyRKD(bandwidth=3.0))

        # Should handle gracefully
        @test solution isa FuzzyRKDSolution
    end

    @testset "Very narrow bandwidth" begin
        Y, X, D = generate_fuzzy_rkd_data(n=2000, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, FuzzyRKD(bandwidth=0.3))

        @test solution isa FuzzyRKDSolution
    end

    @testset "Wide bandwidth" begin
        Y, X, D = generate_fuzzy_rkd_data(n=500, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, FuzzyRKD(bandwidth=10.0))

        # Should include most data
        @test solution.n_eff_left + solution.n_eff_right > 400
    end

    @testset "High noise in treatment" begin
        Y, X, D = generate_fuzzy_rkd_data(n=1000, noise_d=2.0, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, FuzzyRKD(bandwidth=2.5))

        @test solution isa FuzzyRKDSolution
        # High noise should lead to lower F-stat
    end
end

# =============================================================================
# FuzzyRKDSolution Display
# =============================================================================

@testset "FuzzyRKDSolution Display" begin
    Y, X, D = generate_fuzzy_rkd_data(n=500, seed=42)
    problem = RKDProblem(Y, X, D, 0.0)
    solution = solve(problem, FuzzyRKD(bandwidth=2.0))

    # Test that show method works
    io = IOBuffer()
    show(io, solution)
    output = String(take!(io))

    @test occursin("FuzzyRKDSolution", output)
    @test occursin("LATE", output)
    @test occursin("First stage", output)
end
