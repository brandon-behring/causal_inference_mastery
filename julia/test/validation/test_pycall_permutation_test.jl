"""
PyCall validation for PermutationTest during development.

This script directly calls Python implementation to validate Julia results.
Run manually during development for immediate feedback.

DO NOT include in main test suite (avoids Python dependency).

Usage:
    cd julia
    julia --project=. test/validation/test_pycall_permutation_test.jl
"""

using PyCall
using CausalEstimators
using Test
using Random

# Import Python implementation
py"""
import sys
sys.path.insert(0, '/home/brandon_behring/Claude/causal_inference_mastery/src')
from causal_inference.rct.estimators_permutation import permutation_test
"""

@testset "PyCall Validation: PermutationTest" begin
    @testset "Test case 1: Exact test (small sample)" begin
        outcomes = [10.0, 12.0, 11.0, 4.0, 5.0, 3.0]
        treatment = [true, true, true, false, false, false]

        # Julia result
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        jl_solution = solve(problem, PermutationTest())

        # Python result
        py_result = py"permutation_test"(
            outcomes,
            Int.(treatment),
            n_permutations = nothing,
            alternative = "two-sided",
            random_seed = nothing,
        )

        # Compare (should match to 10 decimal places)
        @test jl_solution.p_value ≈ py_result["p_value"] rtol = 1e-10
        @test jl_solution.observed_statistic ≈ py_result["observed_statistic"] rtol = 1e-10
        @test jl_solution.n_permutations == py_result["n_permutations"]
        @test jl_solution.alternative == py_result["alternative"]

        println("✓ Test case 1: Exact test - Julia and Python agree to 10 decimal places")
    end

    @testset "Test case 2: Monte Carlo with seed" begin
        Random.seed!(42)
        outcomes = randn(20) .+ [fill(5.0, 10); fill(0.0, 10)]
        treatment = [fill(true, 10); fill(false, 10)]

        # Julia result
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        jl_solution = solve(problem, PermutationTest(1000, 42))

        # Python result
        py_result = py"permutation_test"(
            outcomes,
            Int.(treatment),
            n_permutations = 1000,
            alternative = "two-sided",
            random_seed = 42,
        )

        # Compare
        @test jl_solution.p_value ≈ py_result["p_value"] rtol = 1e-10
        @test jl_solution.observed_statistic ≈ py_result["observed_statistic"] rtol = 1e-10
        @test jl_solution.n_permutations == py_result["n_permutations"]

        println("✓ Test case 2: Monte Carlo with seed - Julia and Python agree to 10 decimal places")
    end

    @testset "Test case 3: One-sided greater" begin
        outcomes = [10.0, 12.0, 4.0, 5.0]
        treatment = [true, true, false, false]

        # Julia result
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        jl_solution = solve(problem, PermutationTest(nothing, nothing, "greater"))

        # Python result
        py_result = py"permutation_test"(
            outcomes,
            Int.(treatment),
            n_permutations = nothing,
            alternative = "greater",
            random_seed = nothing,
        )

        # Compare
        @test jl_solution.p_value ≈ py_result["p_value"] rtol = 1e-10
        @test jl_solution.observed_statistic ≈ py_result["observed_statistic"] rtol = 1e-10

        println("✓ Test case 3: One-sided greater - Julia and Python agree to 10 decimal places")
    end

    @testset "Test case 4: One-sided less" begin
        outcomes = [4.0, 5.0, 10.0, 12.0]
        treatment = [true, true, false, false]

        # Julia result
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        jl_solution = solve(problem, PermutationTest(nothing, nothing, "less"))

        # Python result
        py_result = py"permutation_test"(
            outcomes,
            Int.(treatment),
            n_permutations = nothing,
            alternative = "less",
            random_seed = nothing,
        )

        # Compare
        @test jl_solution.p_value ≈ py_result["p_value"] rtol = 1e-10
        @test jl_solution.observed_statistic ≈ py_result["observed_statistic"] rtol = 1e-10

        println("✓ Test case 4: One-sided less - Julia and Python agree to 10 decimal places")
    end

    @testset "Test case 5: Large Monte Carlo sample" begin
        Random.seed!(123)
        n = 50
        outcomes = randn(n) .+ [fill(3.0, 25); fill(0.0, 25)]
        treatment = [fill(true, 25); fill(false, 25)]

        # Julia result
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
        jl_solution = solve(problem, PermutationTest(5000, 123))

        # Python result
        py_result = py"permutation_test"(
            outcomes,
            Int.(treatment),
            n_permutations = 5000,
            alternative = "two-sided",
            random_seed = 123,
        )

        # Compare
        @test jl_solution.p_value ≈ py_result["p_value"] rtol = 1e-10
        @test jl_solution.observed_statistic ≈ py_result["observed_statistic"] rtol = 1e-10

        println("✓ Test case 5: Large Monte Carlo - Julia and Python agree to 10 decimal places")
    end
end

println("\n" * "="^80)
println("PyCall validation complete: All tests passed!")
println("Julia implementation matches Python to machine precision.")
println("="^80)
