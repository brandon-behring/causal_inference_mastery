"""
RKD test runner.

Runs all RKD tests:
- Unit tests (Sharp RKD, Fuzzy RKD, diagnostics)
- Monte Carlo validation (bias, coverage, SE accuracy)
- Adversarial tests (edge cases, boundary conditions)
"""

using Test
using CausalEstimators
using Random
using Statistics
using Distributions

@testset "RKD Module Tests" begin
    @testset "Sharp RKD" begin
        include("test_rkd.jl")
    end
    @testset "Fuzzy RKD" begin
        include("test_fuzzy_rkd.jl")
    end
    @testset "RKD Diagnostics" begin
        include("test_diagnostics.jl")
    end
    @testset "Monte Carlo Validation" begin
        include("test_monte_carlo.jl")
    end
    @testset "Adversarial Tests" begin
        include("test_adversarial.jl")
    end
end
