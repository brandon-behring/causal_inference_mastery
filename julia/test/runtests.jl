"""
Main test runner for CausalEstimators.jl

Following SciML pattern with SafeTestsets for test isolation.

Run with: julia --project test/runtests.jl
"""

using Test
using SafeTestsets

@info "Starting CausalEstimators.jl test suite"

# Test module loading
@safetestset "Module Loading" begin
    using CausalEstimators
    @test true  # If module loads, test passes
end

# Test problem construction
@safetestset "Problem Construction" begin
    include("test_problems.jl")
end

# Test solution types
@safetestset "Solution Types" begin
    include("test_solutions.jl")
end

# RCT Estimators
@safetestset "RCT Estimators" begin
    @safetestset "SimpleATE" begin include("rct/test_simple_ate.jl") end
    @safetestset "StratifiedATE" begin include("rct/test_stratified_ate.jl") end
    @safetestset "RegressionATE" begin include("rct/test_regression_ate.jl") end
    @safetestset "PermutationTest" begin include("rct/test_permutation_test.jl") end
    @safetestset "IPWATE" begin include("rct/test_ipw_ate.jl") end
end

# PSM Estimators
@safetestset "PSM Estimators" begin
    @safetestset "Propensity Estimation" begin include("estimators/psm/test_propensity.jl") end
    @safetestset "Matching Algorithm" begin include("estimators/psm/test_matching.jl") end
    @safetestset "Balance Diagnostics" begin include("estimators/psm/test_balance.jl") end
    @safetestset "NearestNeighborPSM" begin include("estimators/psm/test_nearest_neighbor_psm.jl") end
    @safetestset "Monte Carlo Validation" begin include("estimators/psm/test_monte_carlo.jl") end
end

# Golden Reference Validation
@safetestset "Golden Reference Validation" begin
    include("rct/test_golden_reference.jl")
end

# RDD Estimators (Phase 3)
@safetestset "RDD Estimators" begin
    include("rdd/runtests.jl")
end

# Observational IPW/DR (Session 32+)
@safetestset "Observational Estimators" begin
    include("observational/runtests.jl")
end

@info "Test suite complete"
