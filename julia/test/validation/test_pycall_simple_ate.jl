"""
PyCall validation for SimpleATE during development.

This script directly calls Python implementation to validate Julia results.
Run manually during development for immediate feedback.

DO NOT include in main test suite (avoids Python dependency).

Usage:
    cd julia
    julia --project=. test/validation/test_pycall_simple_ate.jl
"""

using PyCall
using CausalEstimators
using Test

# Import Python implementation
py"""
import sys
sys.path.insert(0, '/home/brandon_behring/Claude/causal_inference_mastery/src')
from causal_inference.rct.estimators import simple_ate
"""

@testset "PyCall Validation: SimpleATE" begin
    @testset "Test case 1: Simple balanced RCT" begin
        outcomes = [7.0, 5.0, 3.0, 1.0]
        treatment = [true, true, false, false]

        # Julia result
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        jl_solution = solve(problem, SimpleATE())

        # Python result
        py_result = py"simple_ate"(outcomes, Int.(treatment))

        # Compare (should match to 10 decimal places)
        @test jl_solution.estimate ≈ py_result["estimate"] rtol = 1e-10
        @test jl_solution.se ≈ py_result["se"] rtol = 1e-10
        @test jl_solution.ci_lower ≈ py_result["ci_lower"] rtol = 1e-10
        @test jl_solution.ci_upper ≈ py_result["ci_upper"] rtol = 1e-10
        @test jl_solution.n_treated == py_result["n_treated"]
        @test jl_solution.n_control == py_result["n_control"]

        println("✓ Test case 1: Julia and Python agree to 10 decimal places")
    end

    @testset "Test case 2: Larger sample" begin
        using Random
        Random.seed!(42)

        n = 100
        treatment = rand(Bool, n)
        outcomes = treatment .* 5.0 .+ randn(n)

        # Julia result
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        jl_solution = solve(problem, SimpleATE())

        # Python result
        py_result = py"simple_ate"(outcomes, Int.(treatment))

        # Compare
        @test jl_solution.estimate ≈ py_result["estimate"] rtol = 1e-10
        @test jl_solution.se ≈ py_result["se"] rtol = 1e-10
        @test jl_solution.ci_lower ≈ py_result["ci_lower"] rtol = 1e-10
        @test jl_solution.ci_upper ≈ py_result["ci_upper"] rtol = 1e-10

        println("✓ Test case 2: Julia and Python agree to 10 decimal places")
    end

    @testset "Test case 3: Negative treatment effect" begin
        outcomes = [3.0, 4.0, 10.0, 12.0]
        treatment = [true, true, false, false]

        # Julia result
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        jl_solution = solve(problem, SimpleATE())

        # Python result
        py_result = py"simple_ate"(outcomes, Int.(treatment))

        # Compare
        @test jl_solution.estimate ≈ py_result["estimate"] rtol = 1e-10
        @test jl_solution.se ≈ py_result["se"] rtol = 1e-10

        println("✓ Test case 3: Julia and Python agree to 10 decimal places")
    end

    @testset "Test case 4: Different alpha" begin
        outcomes = [10.0, 12.0, 4.0, 5.0]
        treatment = [true, true, false, false]

        # Test alpha = 0.01 (99% CI)
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.01,))
        jl_solution = solve(problem, SimpleATE())

        py_result = py"simple_ate"(outcomes, Int.(treatment), alpha = 0.01)

        @test jl_solution.estimate ≈ py_result["estimate"] rtol = 1e-10
        @test jl_solution.se ≈ py_result["se"] rtol = 1e-10
        @test jl_solution.ci_lower ≈ py_result["ci_lower"] rtol = 1e-10
        @test jl_solution.ci_upper ≈ py_result["ci_upper"] rtol = 1e-10

        println("✓ Test case 4: Julia and Python agree to 10 decimal places")
    end
end

println("\n" * "="^80)
println("PyCall validation complete: All tests passed!")
println("Julia implementation matches Python to machine precision.")
println("="^80)
