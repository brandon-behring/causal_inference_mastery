"""
Test runner for DiD estimators.

Includes unit tests, Monte Carlo validation, and adversarial tests.
"""

using Test
using SafeTestsets

@info "Running DiD test suite"

# Unit tests
@safetestset "Classic DiD" begin include("test_classic_did.jl") end
@safetestset "Event Study" begin include("test_event_study.jl") end
@safetestset "Staggered DiD" begin include("test_staggered_did.jl") end

# Validation tests (Monte Carlo and Adversarial)
# Note: These are comprehensive and may take longer to run
@safetestset "DiD Monte Carlo Validation" begin include("test_did_montecarlo.jl") end
@safetestset "DiD Adversarial Tests" begin include("test_did_adversarial.jl") end

@info "DiD test suite complete"
