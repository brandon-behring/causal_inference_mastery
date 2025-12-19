"""
Unit tests for Julia RKD module.

Test structure:
1. RKDProblem construction and validation
2. Kernel functions
3. Bandwidth selection
4. Sharp RKD estimation
5. Known-answer tests
"""

using Test
using CausalEstimators
using Random
using Statistics
using Distributions

# =============================================================================
# DGP Functions
# =============================================================================

"""Generate Sharp RKD data with known effect."""
function generate_rkd_data(;
    n::Int=1000,
    cutoff::Float64=0.0,
    slope_left_d::Float64=0.5,
    slope_right_d::Float64=1.5,
    true_effect::Float64=2.0,
    noise_y::Float64=1.0,
    seed::Union{Nothing,Int}=nothing
)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    X = rand(Uniform(-5, 5), n)

    # Treatment with kink at cutoff
    D = [x < cutoff ? slope_left_d * x : slope_right_d * x for x in X]

    # Outcome
    Y = true_effect .* D .+ randn(n) .* noise_y

    return Y, X, D
end

"""Generate RKD data with no kink (for null tests)."""
function generate_no_kink_data(;
    n::Int=500,
    cutoff::Float64=0.0,
    seed::Union{Nothing,Int}=nothing
)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    X = rand(Uniform(-5, 5), n)
    D = 1.0 .* X .+ randn(n) .* 0.5  # Linear D, no kink
    Y = 2.0 .* D .+ randn(n)

    return Y, X, D
end

# =============================================================================
# RKDProblem Tests
# =============================================================================

@testset "RKDProblem Construction" begin
    Y, X, D = generate_rkd_data(n=100, seed=42)

    @testset "Basic construction" begin
        problem = RKDProblem(Y, X, D, 0.0)
        @test problem isa RKDProblem
        @test length(problem.outcomes) == 100
        @test problem.cutoff == 0.0
    end

    @testset "With parameters" begin
        problem = RKDProblem(Y, X, D, 0.0, nothing, (alpha=0.10,))
        @test problem.parameters.alpha == 0.10
    end

    @testset "With covariates" begin
        covariates = randn(100, 2)
        problem = RKDProblem(Y, X, D, 0.0, covariates)
        @test !isnothing(problem.covariates)
        @test size(problem.covariates, 2) == 2
    end

    @testset "Type conversion" begin
        # Int to Float
        Y_int = round.(Int, Y .* 10)
        X_int = round.(Int, X .* 10)
        D_int = round.(Int, D .* 10)
        problem = RKDProblem(Float64.(Y_int), Float64.(X_int), Float64.(D_int), 0.0)
        @test problem isa RKDProblem{Float64}
    end
end

@testset "RKDProblem Validation" begin
    @testset "Length mismatch" begin
        Y = randn(100)
        X = randn(100)
        D = randn(99)  # Wrong length
        @test_throws ArgumentError RKDProblem(Y, X, D, 0.0)
    end

    @testset "NaN in data" begin
        Y = randn(100)
        Y[50] = NaN
        X = randn(100)
        D = randn(100)
        @test_throws ArgumentError RKDProblem(Y, X, D, 0.0)
    end

    @testset "Inf in data" begin
        Y = randn(100)
        X = randn(100)
        X[25] = Inf
        D = randn(100)
        @test_throws ArgumentError RKDProblem(Y, X, D, 0.0)
    end

    @testset "Cutoff out of range" begin
        Y = randn(100)
        X = rand(Uniform(-5, 5), 100)
        D = randn(100)
        @test_throws ArgumentError RKDProblem(Y, X, D, 10.0)  # Out of range
    end

    @testset "Insufficient observations on one side" begin
        Y = randn(100)
        X = rand(Uniform(0, 5), 100)  # All positive
        D = randn(100)
        # Cutoff at -1 means no data below
        @test_throws ArgumentError RKDProblem(Y, X, D, -1.0)
    end

    @testset "Covariate dimension mismatch" begin
        Y = randn(100)
        X = randn(100)
        D = randn(100)
        covariates = randn(50, 2)  # Wrong number of rows
        @test_throws ArgumentError RKDProblem(Y, X, D, 0.0, covariates)
    end
end

# =============================================================================
# Kernel Tests
# =============================================================================

@testset "RKD Kernels" begin
    @testset "Triangular kernel" begin
        kernel = TriangularRKDKernel()
        @test rkd_kernel_function(kernel, 0.0) == 1.0
        @test rkd_kernel_function(kernel, 0.5) == 0.5
        @test rkd_kernel_function(kernel, 1.0) == 0.0
        @test rkd_kernel_function(kernel, 1.5) == 0.0
        @test rkd_kernel_function(kernel, -0.5) == 0.5
    end

    @testset "Uniform kernel" begin
        kernel = UniformRKDKernel()
        @test rkd_kernel_function(kernel, 0.0) == 0.5
        @test rkd_kernel_function(kernel, 0.5) == 0.5
        @test rkd_kernel_function(kernel, 1.0) == 0.5
        @test rkd_kernel_function(kernel, 1.01) == 0.0
    end

    @testset "Epanechnikov kernel" begin
        kernel = EpanechnikovRKDKernel()
        @test rkd_kernel_function(kernel, 0.0) == 0.75
        @test rkd_kernel_function(kernel, 1.0) == 0.0
        @test rkd_kernel_function(kernel, 1.5) == 0.0
    end

    @testset "get_rkd_kernel" begin
        @test get_rkd_kernel(:triangular) isa TriangularRKDKernel
        @test get_rkd_kernel(:uniform) isa UniformRKDKernel
        @test get_rkd_kernel(:rectangular) isa UniformRKDKernel
        @test get_rkd_kernel(:epanechnikov) isa EpanechnikovRKDKernel
        @test_throws ArgumentError get_rkd_kernel(:invalid)
    end
end

# =============================================================================
# Bandwidth Tests
# =============================================================================

@testset "RKD Bandwidth Selection" begin
    Y, X, D = generate_rkd_data(n=500, seed=42)
    problem = RKDProblem(Y, X, D, 0.0)

    @testset "IK bandwidth" begin
        h = rkd_ik_bandwidth(Y, X, 0.0)
        @test h > 0
        @test isfinite(h)
        # RKD bandwidth should be reasonable relative to data spread
        @test h < 10 * std(X)
    end

    @testset "ROT bandwidth" begin
        h = rkd_rot_bandwidth(X, 0.0)
        @test h > 0
        @test isfinite(h)
    end

    @testset "select_rkd_bandwidth" begin
        h_ik = select_rkd_bandwidth(problem, :ik)
        h_rot = select_rkd_bandwidth(problem, :rot)
        @test h_ik > 0
        @test h_rot > 0
        @test_throws ArgumentError select_rkd_bandwidth(problem, :invalid)
    end

    @testset "Larger n gives narrower bandwidth (ROT)" begin
        # Use ROT bandwidth which has cleaner n^{-1/9} rate
        X1 = rand(Uniform(-5, 5), 100)
        X2 = rand(Uniform(-5, 5), 5000)

        h1 = rkd_rot_bandwidth(X1, 0.0)
        h2 = rkd_rot_bandwidth(X2, 0.0)

        # With much more data, bandwidth should be noticeably smaller
        @test h2 < h1
    end
end

# =============================================================================
# SharpRKD Estimator Tests
# =============================================================================

@testset "SharpRKD Estimator Construction" begin
    @testset "Default construction" begin
        est = SharpRKD()
        @test isnothing(est.bandwidth)
        @test est.kernel == :triangular
        @test est.polynomial_order == 1
        @test est.alpha == 0.05
    end

    @testset "Custom parameters" begin
        est = SharpRKD(bandwidth=2.0, kernel=:epanechnikov, polynomial_order=2, alpha=0.10)
        @test est.bandwidth == 2.0
        @test est.kernel == :epanechnikov
        @test est.polynomial_order == 2
        @test est.alpha == 0.10
    end

    @testset "Invalid parameters" begin
        @test_throws ArgumentError SharpRKD(bandwidth=-1.0)
        @test_throws ArgumentError SharpRKD(kernel=:invalid)
        @test_throws ArgumentError SharpRKD(polynomial_order=0)
        @test_throws ArgumentError SharpRKD(polynomial_order=4)
        @test_throws ArgumentError SharpRKD(alpha=0.0)
        @test_throws ArgumentError SharpRKD(alpha=1.0)
    end
end

@testset "SharpRKD Estimation" begin
    @testset "Basic estimation" begin
        Y, X, D = generate_rkd_data(n=1000, true_effect=2.0, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=2.5))

        @test solution isa RKDSolution
        @test solution.retcode == :Success
        @test isfinite(solution.estimate)
        @test isfinite(solution.se)
        @test solution.se > 0
    end

    @testset "Result fields" begin
        Y, X, D = generate_rkd_data(n=500, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=2.0))

        # Check all fields exist
        @test hasfield(RKDSolution, :estimate)
        @test hasfield(RKDSolution, :se)
        @test hasfield(RKDSolution, :ci_lower)
        @test hasfield(RKDSolution, :ci_upper)
        @test hasfield(RKDSolution, :t_stat)
        @test hasfield(RKDSolution, :p_value)
        @test hasfield(RKDSolution, :bandwidth)
        @test hasfield(RKDSolution, :outcome_kink)
        @test hasfield(RKDSolution, :treatment_kink)
    end

    @testset "CI contains estimate" begin
        Y, X, D = generate_rkd_data(n=500, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=2.0))

        if solution.retcode == :Success
            @test solution.ci_lower <= solution.estimate <= solution.ci_upper
        end
    end

    @testset "Different kernels" begin
        Y, X, D = generate_rkd_data(n=500, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)

        for kernel in [:triangular, :uniform, :epanechnikov]
            solution = solve(problem, SharpRKD(bandwidth=2.0, kernel=kernel))
            @test solution isa RKDSolution
            @test solution.kernel == kernel
        end
    end

    @testset "Different polynomial orders" begin
        Y, X, D = generate_rkd_data(n=800, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)

        for order in [1, 2, 3]
            solution = solve(problem, SharpRKD(bandwidth=2.5, polynomial_order=order))
            @test solution isa RKDSolution
            @test solution.polynomial_order == order
        end
    end

    @testset "Auto bandwidth" begin
        Y, X, D = generate_rkd_data(n=500, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD())  # No bandwidth specified

        @test solution.bandwidth > 0
        @test isfinite(solution.bandwidth)
    end
end

# =============================================================================
# Known-Answer Tests
# =============================================================================

@testset "SharpRKD Known-Answer Tests" begin
    @testset "Recovers true effect" begin
        # Generate data with true effect = 2.0
        Y, X, D = generate_rkd_data(
            n=2000,
            slope_left_d=0.5,
            slope_right_d=1.5,  # kink = 1.0
            true_effect=2.0,
            noise_y=0.5,
            seed=42
        )
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=2.5))

        # Should recover approximately 2.0
        @test abs(solution.estimate - 2.0) < 0.6
    end

    @testset "Treatment kink detection" begin
        Y, X, D = generate_rkd_data(
            n=1500,
            slope_left_d=0.5,
            slope_right_d=1.5,  # true kink = 1.0
            seed=123
        )
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=2.0))

        # Treatment kink should be approximately 1.0
        @test abs(solution.treatment_kink - 1.0) < 0.4
    end

    @testset "Negative effect" begin
        Y, X, D = generate_rkd_data(n=1500, true_effect=-1.5, seed=321)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=2.0))

        @test solution.estimate < 0
    end

    @testset "Large effect" begin
        Y, X, D = generate_rkd_data(n=2000, true_effect=5.0, seed=654)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=2.5))

        @test abs(solution.estimate - 5.0) < 1.5
    end

    @testset "Non-zero cutoff" begin
        Y, X, D = generate_rkd_data(n=1000, cutoff=2.0, seed=789)
        problem = RKDProblem(Y, X, D, 2.0)
        solution = solve(problem, SharpRKD(bandwidth=2.0))

        @test solution.retcode in [:Success, :Warning]
    end
end

# =============================================================================
# Edge Cases
# =============================================================================

@testset "SharpRKD Edge Cases" begin
    @testset "Small sample" begin
        Y, X, D = generate_rkd_data(n=50, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=3.0))

        # Should handle gracefully
        @test solution isa RKDSolution
        if solution.retcode == :Warning
            @test occursin("Small sample", solution.message)
        end
    end

    @testset "Very narrow bandwidth" begin
        Y, X, D = generate_rkd_data(n=2000, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=0.3))

        @test solution isa RKDSolution
    end

    @testset "Wide bandwidth" begin
        Y, X, D = generate_rkd_data(n=500, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=10.0))

        # Should include most data
        @test solution.n_eff_left + solution.n_eff_right > 400
    end

    @testset "No treatment kink" begin
        Y, X, D = generate_no_kink_data(n=500, seed=42)
        problem = RKDProblem(Y, X, D, 0.0)
        solution = solve(problem, SharpRKD(bandwidth=2.5))

        # Should return failure or have small kink
        if solution.retcode == :Failure
            @test occursin("No kink", solution.message)
        else
            @test abs(solution.treatment_kink) < 0.5
        end
    end
end

# =============================================================================
# RKDSolution Display
# =============================================================================

@testset "RKDSolution Display" begin
    Y, X, D = generate_rkd_data(n=500, seed=42)
    problem = RKDProblem(Y, X, D, 0.0)
    solution = solve(problem, SharpRKD(bandwidth=2.0))

    # Test that show method works
    io = IOBuffer()
    show(io, solution)
    output = String(take!(io))

    @test occursin("RKDSolution", output)
    @test occursin("Estimate", output)
    @test occursin("Std. Error", output)
end
