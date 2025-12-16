#=
Sensitivity Analysis Test Suite

Runs all tests for E-value and Rosenbaum bounds sensitivity analysis.
=#

using SafeTestsets

@safetestset "E-Value Tests" begin
    include("test_e_value.jl")
end

@safetestset "Rosenbaum Bounds Tests" begin
    include("test_rosenbaum.jl")
end
