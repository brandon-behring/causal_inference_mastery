"""
Tests for RDD bandwidth selection methods.

Phase 3.2-3.4: IK and CCT bandwidth selection
"""

using Test
using CausalEstimators
using Random
using Statistics

@testset "Bandwidth Selection" begin
    @testset "IK Bandwidth - Basic Functionality" begin
        # Simple RDD data with known effect
        Random.seed!(123)
        n = 1000
        x = randn(n) .* 2.0  # Running variable: ~ N(0, 4)
        treatment = x .>= 0.0

        # Sharp RDD: y = 2x + 5*treatment + noise
        y = 2.0 .* x .+ 5.0 .* treatment .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))

        # Select IK bandwidth
        h_ik = select_bandwidth(problem, IKBandwidth())

        # Basic validation
        @test h_ik > 0.0
        @test isfinite(h_ik)
        @test h_ik < maximum(x) - minimum(x)  # Should be less than range

        # Reasonable magnitude (typically n^(-1/5) scaling)
        expected_order = n^(-1/5) * std(x)
        @test h_ik > 0.1 * expected_order
        @test h_ik < 10.0 * expected_order
    end

    @testset "CCT Bandwidth - Two Bandwidths" begin
        Random.seed!(456)
        n = 500
        x = randn(n)
        treatment = x .>= 0.0
        y = randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))

        # Select CCT bandwidths
        h_main, h_bias = select_bandwidth(problem, CCTBandwidth())

        # Basic validation
        @test h_main > 0.0
        @test h_bias > 0.0
        @test isfinite(h_main)
        @test isfinite(h_bias)

        # CCT property: h_bias > h_main (for bias correction)
        @test h_bias > h_main

        # Typical ratio: h_bias ≈ 2 * h_main
        ratio = h_bias / h_main
        @test ratio > 1.5
        @test ratio < 3.0
    end

    @testset "IK vs CCT - Main Bandwidth Relationship" begin
        Random.seed!(789)
        n = 800
        x = randn(n) .* 1.5
        treatment = x .>= 0.0
        y = 3.0 .* x .+ 4.0 .* treatment .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))

        h_ik = select_bandwidth(problem, IKBandwidth())
        h_main, h_bias = select_bandwidth(problem, CCTBandwidth())

        # CCT uses IK as starting point with conservative adjustment
        # Typically h_main ≈ h_ik (or slightly different based on coverage adjustment)
        @test isapprox(h_main, h_ik, rtol=0.5)  # Within 50%
    end

    @testset "Bandwidth with Covariates" begin
        Random.seed!(321)
        n = 600
        x = randn(n)
        treatment = x .>= 0.0
        covariates = randn(n, 3)
        y = 2.0 .* x .+ 5.0 .* treatment .+ sum(covariates, dims=2)[:] .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, covariates, (alpha=0.05,))

        h_ik = select_bandwidth(problem, IKBandwidth())
        h_main, h_bias = select_bandwidth(problem, CCTBandwidth())

        # Should still work with covariates
        @test h_ik > 0.0
        @test h_main > 0.0
        @test h_bias > h_main
    end

    @testset "Bandwidth Stability - Small Changes in Data" begin
        # Bandwidth should be relatively stable to small perturbations
        Random.seed!(111)
        n = 500
        x = randn(n)
        treatment = x .>= 0.0
        y_base = 2.0 .* x .+ 5.0 .* treatment .+ randn(n)

        problem_base = RDDProblem(y_base, x, treatment, 0.0, nothing, (alpha=0.05,))
        h_base = select_bandwidth(problem_base, IKBandwidth())

        # Small perturbation to outcomes
        y_perturbed = y_base .+ 0.1 .* randn(n)
        problem_perturbed = RDDProblem(y_perturbed, x, treatment, 0.0, nothing, (alpha=0.05,))
        h_perturbed = select_bandwidth(problem_perturbed, IKBandwidth())

        # Should be similar (within 30%)
        @test isapprox(h_perturbed, h_base, rtol=0.3)
    end

    @testset "Bandwidth Edge Cases" begin
        Random.seed!(222)

        # Small sample
        n_small = 100
        x_small = randn(n_small)
        treatment_small = x_small .>= 0.0
        y_small = randn(n_small)

        problem_small = RDDProblem(y_small, x_small, treatment_small, 0.0, nothing, (alpha=0.05,))

        # Function should work even with small samples
        h_small = select_bandwidth(problem_small, IKBandwidth())
        @test h_small > 0.0

        # Large sample
        n_large = 5000
        x_large = randn(n_large)
        treatment_large = x_large .>= 0.0
        y_large = randn(n_large)

        problem_large = RDDProblem(y_large, x_large, treatment_large, 0.0, nothing, (alpha=0.05,))
        h_large = select_bandwidth(problem_large, IKBandwidth())

        # Larger sample should have smaller bandwidth (n^(-1/5) scaling)
        @test h_large < h_small
    end

    @testset "Bandwidth Numerical Stability" begin
        Random.seed!(333)
        n = 500

        # Test with different scales of running variable
        for scale in [0.1, 1.0, 10.0, 100.0]
            x = randn(n) .* scale
            treatment = x .>= 0.0
            y = randn(n) .* scale

            problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))
            h = select_bandwidth(problem, IKBandwidth())

            # Should scale roughly with data
            @test h > 0.0
            @test isfinite(h)
            @test h > 0.01 * scale
            @test h < 10.0 * scale
        end
    end

    @testset "Bandwidth with Non-Zero Cutoff" begin
        Random.seed!(444)
        n = 600
        cutoff = 2.0
        x = randn(n) .* 2.0 .+ cutoff  # Center around cutoff
        treatment = x .>= cutoff
        y = 2.0 .* (x .- cutoff) .+ 5.0 .* treatment .+ randn(n)

        problem = RDDProblem(y, x, treatment, cutoff, nothing, (alpha=0.05,))

        h_ik = select_bandwidth(problem, IKBandwidth())
        h_main, h_bias = select_bandwidth(problem, CCTBandwidth())

        # Should work with non-zero cutoff
        @test h_ik > 0.0
        @test h_main > 0.0
        @test h_bias > h_main
    end
end
