"""
PyCall validation for StratifiedATE during development.

This script directly calls Python implementation to validate Julia results.
Run manually during development for immediate feedback.

DO NOT include in main test suite (avoids Python dependency).

Usage:
    cd julia
    julia --project=. test/validation/test_pycall_stratified_ate.jl
"""

using PyCall
using CausalEstimators
using Test

# Import Python implementation
py"""
import sys
sys.path.insert(0, '/home/brandon_behring/Claude/causal_inference_mastery/src')
from causal_inference.rct.estimators_stratified import stratified_ate
"""

@testset "PyCall Validation: StratifiedATE" begin
    @testset "Test case 1: Two equal-sized strata" begin
        outcomes = [15.0, 10.0, 105.0, 100.0]
        treatment = [true, false, true, false]
        strata = [1, 1, 2, 2]

        # Julia result
        problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.05,))
        jl_solution = solve(problem, StratifiedATE())

        # Python result
        py_result = py"stratified_ate"(outcomes, Int.(treatment), strata)

        # Compare (should match to 10 decimal places)
        @test jl_solution.estimate ≈ py_result["estimate"] rtol = 1e-10
        @test jl_solution.se ≈ py_result["se"] rtol = 1e-10
        @test jl_solution.ci_lower ≈ py_result["ci_lower"] rtol = 1e-10
        @test jl_solution.ci_upper ≈ py_result["ci_upper"] rtol = 1e-10
        @test jl_solution.n_treated == py_result["n_treated"]
        @test jl_solution.n_control == py_result["n_control"]

        println("✓ Test case 1: Julia and Python agree to 10 decimal places")
    end

    @testset "Test case 2: Unequal-sized strata" begin
        # Stratum 1 (n=2): ATE=10
        # Stratum 2 (n=4): ATE=5
        outcomes = [20.0, 10.0, 15.0, 10.0, 15.0, 10.0]
        treatment = [true, false, true, false, true, false]
        strata = [1, 1, 2, 2, 2, 2]

        # Julia result
        problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.05,))
        jl_solution = solve(problem, StratifiedATE())

        # Python result
        py_result = py"stratified_ate"(outcomes, Int.(treatment), strata)

        # Compare
        @test jl_solution.estimate ≈ py_result["estimate"] rtol = 1e-10
        @test jl_solution.se ≈ py_result["se"] rtol = 1e-10
        @test jl_solution.ci_lower ≈ py_result["ci_lower"] rtol = 1e-10
        @test jl_solution.ci_upper ≈ py_result["ci_upper"] rtol = 1e-10

        println("✓ Test case 2: Julia and Python agree to 10 decimal places")
    end

    @testset "Test case 3: Larger sample with randomness" begin
        using Random
        Random.seed!(42)

        n_per_stratum = 50
        strata = vcat(fill(1, n_per_stratum), fill(2, n_per_stratum))
        treatment_bool = rand(Bool, 100)
        treatment_int = Int.(treatment_bool)

        # Stratum 1: baseline=10, Stratum 2: baseline=100
        # Both have treatment effect = 5
        outcomes = zeros(100)
        for i in 1:100
            baseline = strata[i] == 1 ? 10.0 : 100.0
            effect = treatment_bool[i] ? 5.0 : 0.0
            outcomes[i] = baseline + effect + randn()
        end

        # Julia result
        problem = RCTProblem(outcomes, treatment_bool, nothing, strata, (alpha = 0.05,))
        jl_solution = solve(problem, StratifiedATE())

        # Python result
        py_result = py"stratified_ate"(outcomes, treatment_int, strata)

        # Compare
        @test jl_solution.estimate ≈ py_result["estimate"] rtol = 1e-10
        @test jl_solution.se ≈ py_result["se"] rtol = 1e-10
        @test jl_solution.ci_lower ≈ py_result["ci_lower"] rtol = 1e-10
        @test jl_solution.ci_upper ≈ py_result["ci_upper"] rtol = 1e-10

        println("✓ Test case 3: Julia and Python agree to 10 decimal places")
    end

    @testset "Test case 4: Different alpha (99% CI)" begin
        outcomes = [15.0, 10.0, 105.0, 100.0]
        treatment = [true, false, true, false]
        strata = [1, 1, 2, 2]

        # Julia result with alpha=0.01
        problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.01,))
        jl_solution = solve(problem, StratifiedATE())

        # Python result with alpha=0.01
        py_result = py"stratified_ate"(outcomes, Int.(treatment), strata, alpha = 0.01)

        # Compare
        @test jl_solution.estimate ≈ py_result["estimate"] rtol = 1e-10
        @test jl_solution.se ≈ py_result["se"] rtol = 1e-10
        @test jl_solution.ci_lower ≈ py_result["ci_lower"] rtol = 1e-10
        @test jl_solution.ci_upper ≈ py_result["ci_upper"] rtol = 1e-10

        println("✓ Test case 4: Julia and Python agree to 10 decimal places")
    end

    @testset "Test case 5: Three strata" begin
        outcomes = [15.0, 10.0, 25.0, 20.0, 35.0, 30.0]
        treatment = [true, false, true, false, true, false]
        strata = [1, 1, 2, 2, 3, 3]

        # Julia result
        problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.05,))
        jl_solution = solve(problem, StratifiedATE())

        # Python result
        py_result = py"stratified_ate"(outcomes, Int.(treatment), strata)

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
