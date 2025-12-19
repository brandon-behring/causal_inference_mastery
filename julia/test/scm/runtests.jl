#=
Synthetic Control Methods Test Runner

Tests for SCM (Synthetic Control Methods):
- SyntheticControl: Abadie/Gardeazabal/Hainmueller basic SCM
- AugmentedSC: Ben-Michael et al. (2021) bias-corrected estimator
=#

using Test
using CausalEstimators
using Random
using Statistics
using LinearAlgebra

@testset "SCM Module Tests" begin
    # Unit tests
    include("test_types.jl")
    include("test_synthetic_control.jl")
    include("test_inference.jl")
    include("test_augmented.jl")

    # Validation tests (Monte Carlo and Adversarial)
    @testset "SCM Monte Carlo Validation" begin
        include("test_scm_montecarlo.jl")
    end
    @testset "SCM Adversarial Tests" begin
        include("test_scm_adversarial.jl")
    end
end
