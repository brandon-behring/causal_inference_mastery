#=
Observational Module Test Runner

Runs all tests for IPW and DR estimators for observational studies.
=#

using Test
using CausalEstimators

@testset "Observational Tests" begin
    include("test_ipw.jl")
    include("test_doubly_robust.jl")
end
