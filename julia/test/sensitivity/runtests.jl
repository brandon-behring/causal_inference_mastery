#=
Sensitivity Analysis Test Suite

Runs all tests for E-value and Rosenbaum bounds sensitivity analysis.
=#

using SafeTestsets

# Unit tests
@safetestset "E-Value Tests" begin
    include("test_e_value.jl")
end

@safetestset "Rosenbaum Bounds Tests" begin
    include("test_rosenbaum.jl")
end

# Validation tests (Monte Carlo and Adversarial)
@safetestset "Sensitivity Monte Carlo Validation" begin
    include("test_sensitivity_montecarlo.jl")
end

@safetestset "Sensitivity Adversarial Tests" begin
    include("test_sensitivity_adversarial.jl")
end
