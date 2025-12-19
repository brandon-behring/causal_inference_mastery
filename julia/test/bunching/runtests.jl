#=
Bunching estimation test runner.

Test Categories:
1. Unit tests (test_bunching_types.jl, test_bunching_counterfactual.jl, test_bunching_estimator.jl)
2. Monte Carlo validation (test_bunching_montecarlo.jl)
3. Adversarial edge cases (test_bunching_adversarial.jl)

Session 81: Added Monte Carlo and Adversarial validation tests.
=#

using Test
using SafeTestsets

@safetestset "Bunching Types" begin
    include("test_bunching_types.jl")
end

@safetestset "Counterfactual Estimation" begin
    include("test_bunching_counterfactual.jl")
end

@safetestset "Bunching Estimator" begin
    include("test_bunching_estimator.jl")
end

@safetestset "Bunching Monte Carlo" begin
    include("test_bunching_montecarlo.jl")
end

@safetestset "Bunching Adversarial" begin
    include("test_bunching_adversarial.jl")
end
