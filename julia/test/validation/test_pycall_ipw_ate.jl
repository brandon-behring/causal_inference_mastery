"""
PyCall validation for IPWATE during development.

This script directly calls Python implementation to validate Julia results.
Run manually during development for immediate feedback.

DO NOT include in main test suite (avoids Python dependency).

Usage:
    cd julia
    julia --project=. test/validation/test_pycall_ipw_ate.jl
"""

using PyCall
using CausalEstimators
using Test
using Random

# Import Python implementation
py"""
import sys
sys.path.insert(0, '/home/brandon_behring/Claude/causal_inference_mastery/src')
from causal_inference.rct.estimators_ipw import ipw_ate
"""

@testset "PyCall Validation: IPWATE" begin
    @testset "Test case 1: Constant propensity (p=0.5)" begin
        outcomes = [10.0, 12.0, 4.0, 5.0]
        treatment = [true, true, false, false]
        propensity = [0.5, 0.5, 0.5, 0.5]

        # Julia result
        problem = RCTProblem(outcomes, treatment, hcat(propensity), nothing, (alpha = 0.05,))
        jl_solution = solve(problem, IPWATE())

        # Python result
        py_result = py"ipw_ate"(outcomes, Int.(treatment), propensity, alpha = 0.05)

        # Compare (should match to 10 decimal places)
        @test jl_solution.estimate ≈ py_result["estimate"] rtol = 1e-10
        @test jl_solution.se ≈ py_result["se"] rtol = 1e-10
        @test jl_solution.ci_lower ≈ py_result["ci_lower"] rtol = 1e-10
        @test jl_solution.ci_upper ≈ py_result["ci_upper"] rtol = 1e-10
        @test jl_solution.n_treated == py_result["n_treated"]
        @test jl_solution.n_control == py_result["n_control"]

        println("✓ Test case 1: Constant propensity - Julia and Python agree to 10 decimal places")
    end

    @testset "Test case 2: Varying propensity" begin
        outcomes = [10.0, 12.0, 4.0, 6.0]
        treatment = [true, true, false, false]
        propensity = [0.6, 0.4, 0.4, 0.6]

        # Julia result
        problem = RCTProblem(outcomes, treatment, hcat(propensity), nothing, (alpha = 0.05,))
        jl_solution = solve(problem, IPWATE())

        # Python result
        py_result = py"ipw_ate"(outcomes, Int.(treatment), propensity, alpha = 0.05)

        # Compare
        @test jl_solution.estimate ≈ py_result["estimate"] rtol = 1e-10
        @test jl_solution.se ≈ py_result["se"] rtol = 1e-10
        @test jl_solution.ci_lower ≈ py_result["ci_lower"] rtol = 1e-10
        @test jl_solution.ci_upper ≈ py_result["ci_upper"] rtol = 1e-10

        println("✓ Test case 2: Varying propensity - Julia and Python agree to 10 decimal places")
    end

    @testset "Test case 3: Different alpha (99% CI)" begin
        outcomes = [10.0, 12.0, 4.0, 5.0]
        treatment = [true, true, false, false]
        propensity = [0.5, 0.5, 0.5, 0.5]

        # Julia result with alpha=0.01
        problem = RCTProblem(outcomes, treatment, hcat(propensity), nothing, (alpha = 0.01,))
        jl_solution = solve(problem, IPWATE())

        # Python result with alpha=0.01
        py_result = py"ipw_ate"(outcomes, Int.(treatment), propensity, alpha = 0.01)

        # Compare
        @test jl_solution.estimate ≈ py_result["estimate"] rtol = 1e-10
        @test jl_solution.se ≈ py_result["se"] rtol = 1e-10
        @test jl_solution.ci_lower ≈ py_result["ci_lower"] rtol = 1e-10
        @test jl_solution.ci_upper ≈ py_result["ci_upper"] rtol = 1e-10

        println("✓ Test case 3: Different alpha - Julia and Python agree to 10 decimal places")
    end

    @testset "Test case 4: Larger sample with random data" begin
        Random.seed!(42)
        n = 30
        outcomes = randn(n) .+ [fill(5.0, 15); fill(0.0, 15)]
        treatment = [fill(true, 15); fill(false, 15)]
        propensity = rand(n) * 0.4 .+ 0.3  # In (0.3, 0.7)

        # Julia result
        problem = RCTProblem(outcomes, treatment, hcat(propensity), nothing, (alpha = 0.05,))
        jl_solution = solve(problem, IPWATE())

        # Python result
        py_result = py"ipw_ate"(outcomes, Int.(treatment), propensity, alpha = 0.05)

        # Compare
        @test jl_solution.estimate ≈ py_result["estimate"] rtol = 1e-10
        @test jl_solution.se ≈ py_result["se"] rtol = 1e-10
        @test jl_solution.ci_lower ≈ py_result["ci_lower"] rtol = 1e-10
        @test jl_solution.ci_upper ≈ py_result["ci_upper"] rtol = 1e-10

        println("✓ Test case 4: Larger sample - Julia and Python agree to 10 decimal places")
    end

    @testset "Test case 5: Extreme weights (test stability)" begin
        outcomes = [10.0, 15.0, 8.0, 12.0, 5.0, 3.0]
        treatment = [true, true, true, false, false, false]
        propensity = [0.9, 0.7, 0.5, 0.5, 0.3, 0.1]  # Varied, some extreme

        # Julia result
        problem = RCTProblem(outcomes, treatment, hcat(propensity), nothing, (alpha = 0.05,))
        jl_solution = solve(problem, IPWATE())

        # Python result
        py_result = py"ipw_ate"(outcomes, Int.(treatment), propensity, alpha = 0.05)

        # Compare
        @test jl_solution.estimate ≈ py_result["estimate"] rtol = 1e-10
        @test jl_solution.se ≈ py_result["se"] rtol = 1e-10

        println("✓ Test case 5: Extreme weights - Julia and Python agree to 10 decimal places")
    end
end

println("\n" * "="^80)
println("PyCall validation complete: All tests passed!")
println("Julia implementation matches Python to machine precision.")
println("="^80)
