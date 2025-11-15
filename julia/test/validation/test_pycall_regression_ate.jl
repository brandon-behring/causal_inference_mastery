"""
PyCall validation for RegressionATE during development.

This script directly calls Python implementation to validate Julia results.
Run manually during development for immediate feedback.

DO NOT include in main test suite (avoids Python dependency).

Usage:
    cd julia
    julia --project=. test/validation/test_pycall_regression_ate.jl
"""

using PyCall
using CausalEstimators
using Test

# Import Python implementation
py"""
import sys
sys.path.insert(0, '/home/brandon_behring/Claude/causal_inference_mastery/src')
from causal_inference.rct.estimators_regression import regression_adjusted_ate
"""

@testset "PyCall Validation: RegressionATE" begin
    @testset "Test case 1: Single covariate, perfect linear" begin
        # Y = 2 + 5*T + 3*X
        X = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        treatment = [true, true, true, true, false, false, false, false]
        outcomes = 2.0 .+ 5.0 .* treatment .+ 3.0 .* X

        # Julia result
        X_matrix = reshape(X, length(X), 1)
        problem = RCTProblem(outcomes, treatment, X_matrix, nothing, (alpha = 0.05,))
        jl_solution = solve(problem, RegressionATE())

        # Python result
        py_result = py"regression_adjusted_ate"(outcomes, Int.(treatment), X)

        # Compare (should match to 10 decimal places)
        @test jl_solution.estimate ≈ py_result["estimate"] rtol = 1e-10
        @test jl_solution.se ≈ py_result["se"] rtol = 1e-10
        @test jl_solution.ci_lower ≈ py_result["ci_lower"] rtol = 1e-10
        @test jl_solution.ci_upper ≈ py_result["ci_upper"] rtol = 1e-10
        @test jl_solution.n_treated == py_result["n_treated"]
        @test jl_solution.n_control == py_result["n_control"]

        println("✓ Test case 1: Julia and Python agree to 10 decimal places")
    end

    @testset "Test case 2: Multiple covariates" begin
        using Random
        Random.seed!(123)

        # Y = 1 + 3*T + 2*X1 + 4*X2 + small noise
        # Use varied values to avoid singularity
        X1 = [1.0, 2.5, 3.2, 1.8, 2.1, 3.5]
        X2 = [0.5, 1.8, 2.3, 0.7, 1.4, 2.9]
        treatment = [true, true, true, false, false, false]
        outcomes = 1.0 .+ 3.0 .* treatment .+ 2.0 .* X1 .+ 4.0 .* X2 .+ 0.01 .* randn(6)

        # Julia result
        X_matrix = hcat(X1, X2)
        problem = RCTProblem(outcomes, treatment, X_matrix, nothing, (alpha = 0.05,))
        jl_solution = solve(problem, RegressionATE())

        # Python result
        py_result = py"regression_adjusted_ate"(outcomes, Int.(treatment), X_matrix)

        # Compare
        @test jl_solution.estimate ≈ py_result["estimate"] rtol = 1e-10
        @test jl_solution.se ≈ py_result["se"] rtol = 1e-10
        @test jl_solution.ci_lower ≈ py_result["ci_lower"] rtol = 1e-10
        @test jl_solution.ci_upper ≈ py_result["ci_upper"] rtol = 1e-10

        println("✓ Test case 2: Julia and Python agree to 10 decimal places")
    end

    @testset "Test case 3: Random data with noise" begin
        using Random
        Random.seed!(42)

        n = 50
        X = randn(n)
        treatment_bool = rand(Bool, n)
        treatment_int = Int.(treatment_bool)
        outcomes = 5.0 .+ 3.0 .* treatment_bool .+ 2.0 .* X .+ 0.5 .* randn(n)

        # Julia result
        X_matrix = reshape(X, n, 1)
        problem = RCTProblem(outcomes, treatment_bool, X_matrix, nothing, (alpha = 0.05,))
        jl_solution = solve(problem, RegressionATE())

        # Python result
        py_result = py"regression_adjusted_ate"(outcomes, treatment_int, X)

        # Compare
        @test jl_solution.estimate ≈ py_result["estimate"] rtol = 1e-10
        @test jl_solution.se ≈ py_result["se"] rtol = 1e-10
        @test jl_solution.ci_lower ≈ py_result["ci_lower"] rtol = 1e-10
        @test jl_solution.ci_upper ≈ py_result["ci_upper"] rtol = 1e-10

        println("✓ Test case 3: Julia and Python agree to 10 decimal places")
    end

    @testset "Test case 4: Different alpha (99% CI)" begin
        X = [1.0, 2.0, 3.0, 4.0]
        treatment = [true, true, false, false]
        outcomes = [10.0, 14.0, 7.0, 11.0]

        # Julia result with alpha=0.01
        X_matrix = reshape(X, length(X), 1)
        problem = RCTProblem(outcomes, treatment, X_matrix, nothing, (alpha = 0.01,))
        jl_solution = solve(problem, RegressionATE())

        # Python result with alpha=0.01
        py_result = py"regression_adjusted_ate"(outcomes, Int.(treatment), X, alpha = 0.01)

        # Compare
        @test jl_solution.estimate ≈ py_result["estimate"] rtol = 1e-10
        @test jl_solution.se ≈ py_result["se"] rtol = 1e-10
        @test jl_solution.ci_lower ≈ py_result["ci_lower"] rtol = 1e-10
        @test jl_solution.ci_upper ≈ py_result["ci_upper"] rtol = 1e-10

        println("✓ Test case 4: Julia and Python agree to 10 decimal places")
    end

    @testset "Test case 5: Larger sample" begin
        using Random
        Random.seed!(123)

        n = 100
        X1 = randn(n)
        X2 = randn(n)
        treatment_bool = rand(Bool, n)
        treatment_int = Int.(treatment_bool)
        outcomes = 2.0 .+ 4.0 .* treatment_bool .+ 3.0 .* X1 .+ 1.5 .* X2 .+ randn(n)

        # Julia result
        X_matrix = hcat(X1, X2)
        problem = RCTProblem(outcomes, treatment_bool, X_matrix, nothing, (alpha = 0.05,))
        jl_solution = solve(problem, RegressionATE())

        # Python result
        py_result = py"regression_adjusted_ate"(outcomes, treatment_int, X_matrix)

        # Compare
        @test jl_solution.estimate ≈ py_result["estimate"] rtol = 1e-10
        @test jl_solution.se ≈ py_result["se"] rtol = 1e-10

        println("✓ Test case 5: Julia and Python agree to 10 decimal places")
    end
end

println("\n" * "="^80)
println("PyCall validation complete: All tests passed!")
println("Julia implementation matches Python to machine precision.")
println("="^80)
