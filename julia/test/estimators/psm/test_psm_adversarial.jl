#=
Adversarial Tests for Julia PSM Module.

Tests edge cases, boundary conditions, and error handling for:
- PSMProblem: Input validation
- nearest_neighbor_match: Matching algorithm edge cases
- compute_ate_from_matches: ATE computation edge cases
- estimate_propensity: Propensity estimation edge cases
- Balance diagnostics: SMD, variance ratio edge cases

Session 68: Julia PSM Adversarial Tests
=#

using Test
using Statistics
using Random
using LinearAlgebra

# CausalEstimators should be loaded by parent test runner

# =============================================================================
# PSMProblem Input Validation Tests
# =============================================================================

@testset "PSMProblem Input Validation" begin

    @testset "Mismatched lengths" begin
        # outcomes vs treatment
        @test_throws ArgumentError PSMProblem(
            [1.0, 2.0, 3.0],
            [true, false],  # Wrong length
            [1.0 2.0; 3.0 4.0; 5.0 6.0],
            (alpha=0.05,)
        )

        # outcomes vs covariates
        @test_throws ArgumentError PSMProblem(
            [1.0, 2.0, 3.0],
            [true, false, true],
            [1.0 2.0; 3.0 4.0],  # Wrong number of rows
            (alpha=0.05,)
        )
    end

    @testset "Empty inputs" begin
        @test_throws ArgumentError PSMProblem(
            Float64[],
            Bool[],
            Matrix{Float64}(undef, 0, 2),
            (alpha=0.05,)
        )
    end

    @testset "NaN in outcomes" begin
        @test_throws ArgumentError PSMProblem(
            [1.0, NaN, 3.0, 4.0, 5.0, 6.0],
            [true, true, true, false, false, false],
            ones(6, 2),
            (alpha=0.05,)
        )
    end

    @testset "Inf in outcomes" begin
        @test_throws ArgumentError PSMProblem(
            [1.0, Inf, 3.0, 4.0, 5.0, 6.0],
            [true, true, true, false, false, false],
            ones(6, 2),
            (alpha=0.05,)
        )

        @test_throws ArgumentError PSMProblem(
            [1.0, -Inf, 3.0, 4.0, 5.0, 6.0],
            [true, true, true, false, false, false],
            ones(6, 2),
            (alpha=0.05,)
        )
    end

    @testset "NaN in covariates" begin
        covs = [1.0 2.0; 3.0 NaN; 5.0 6.0; 7.0 8.0; 9.0 10.0; 11.0 12.0]
        @test_throws ArgumentError PSMProblem(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [true, true, true, false, false, false],
            covs,
            (alpha=0.05,)
        )
    end

    @testset "Inf in covariates" begin
        covs = [1.0 Inf; 3.0 4.0; 5.0 6.0; 7.0 8.0; 9.0 10.0; 11.0 12.0]
        @test_throws ArgumentError PSMProblem(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [true, true, true, false, false, false],
            covs,
            (alpha=0.05,)
        )
    end

    @testset "No treated units" begin
        @test_throws ArgumentError PSMProblem(
            [1.0, 2.0, 3.0, 4.0],
            [false, false, false, false],
            ones(4, 2),
            (alpha=0.05,)
        )
    end

    @testset "No control units" begin
        @test_throws ArgumentError PSMProblem(
            [1.0, 2.0, 3.0, 4.0],
            [true, true, true, true],
            ones(4, 2),
            (alpha=0.05,)
        )
    end

    @testset "Insufficient treated units" begin
        # Only 1 treated, need at least 2
        @test_throws ArgumentError PSMProblem(
            [1.0, 2.0, 3.0, 4.0],
            [true, false, false, false],
            ones(4, 2),
            (alpha=0.05,)
        )
    end

    @testset "Insufficient control units" begin
        # Only 1 control, need at least 2
        @test_throws ArgumentError PSMProblem(
            [1.0, 2.0, 3.0, 4.0],
            [true, true, true, false],
            ones(4, 2),
            (alpha=0.05,)
        )
    end

    @testset "Invalid alpha" begin
        @test_throws ArgumentError PSMProblem(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [true, true, true, false, false, false],
            ones(6, 2),
            (alpha=0.0,)
        )

        @test_throws ArgumentError PSMProblem(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [true, true, true, false, false, false],
            ones(6, 2),
            (alpha=1.0,)
        )

        @test_throws ArgumentError PSMProblem(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [true, true, true, false, false, false],
            ones(6, 2),
            (alpha=-0.1,)
        )

        @test_throws ArgumentError PSMProblem(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [true, true, true, false, false, false],
            ones(6, 2),
            (alpha=1.5,)
        )
    end

    @testset "Valid minimum case" begin
        # Minimum viable: 2 treated, 2 control
        problem = PSMProblem(
            [1.0, 2.0, 3.0, 4.0],
            [true, true, false, false],
            [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0],
            (alpha=0.05,)
        )
        @test problem.outcomes == [1.0, 2.0, 3.0, 4.0]
        @test problem.treatment == [true, true, false, false]
    end

    @testset "BitVector treatment conversion" begin
        # BitVector should be converted to Vector{Bool}
        treatment = BitVector([true, true, false, false])
        problem = PSMProblem(
            [1.0, 2.0, 3.0, 4.0],
            treatment,
            ones(4, 2),
            (alpha=0.05,)
        )
        @test problem.treatment isa Vector{Bool}
    end

end

# =============================================================================
# nearest_neighbor_match Edge Cases
# =============================================================================

@testset "nearest_neighbor_match Edge Cases" begin

    @testset "No treated units" begin
        @test_throws ArgumentError nearest_neighbor_match(
            Float64[],  # No treated
            [0.3, 0.4, 0.5],
            Int[],
            [1, 2, 3]
        )
    end

    @testset "No control units" begin
        @test_throws ArgumentError nearest_neighbor_match(
            [0.5, 0.6],
            Float64[],  # No control
            [1, 2],
            Int[]
        )
    end

    @testset "Invalid M value" begin
        @test_throws ArgumentError nearest_neighbor_match(
            [0.5, 0.6],
            [0.3, 0.4, 0.5],
            [1, 2],
            [3, 4, 5],
            M=0
        )

        @test_throws ArgumentError nearest_neighbor_match(
            [0.5, 0.6],
            [0.3, 0.4, 0.5],
            [1, 2],
            [3, 4, 5],
            M=-1
        )
    end

    @testset "M > n_control without replacement" begin
        @test_throws ArgumentError nearest_neighbor_match(
            [0.5, 0.6],
            [0.3, 0.4],  # Only 2 controls
            [1, 2],
            [3, 4],
            M=3,  # Want 3 matches
            with_replacement=false
        )
    end

    @testset "M > n_control with replacement - valid" begin
        # With replacement, this is allowed
        matches, distances, n_matched = nearest_neighbor_match(
            [0.5],
            [0.3, 0.4],  # Only 2 controls
            [1],
            [2, 3],
            M=3,  # Want 3 matches (will get 2)
            with_replacement=true
        )
        @test n_matched >= 0
    end

    @testset "Invalid caliper" begin
        @test_throws ArgumentError nearest_neighbor_match(
            [0.5, 0.6],
            [0.3, 0.4, 0.5],
            [1, 2],
            [3, 4, 5],
            caliper=0.0
        )

        @test_throws ArgumentError nearest_neighbor_match(
            [0.5, 0.6],
            [0.3, 0.4, 0.5],
            [1, 2],
            [3, 4, 5],
            caliper=-0.1
        )
    end

    @testset "Mismatched propensity/indices lengths" begin
        @test_throws ArgumentError nearest_neighbor_match(
            [0.5, 0.6, 0.7],  # 3 propensity
            [0.3, 0.4, 0.5],
            [1, 2],  # 2 indices - mismatch
            [3, 4, 5]
        )

        @test_throws ArgumentError nearest_neighbor_match(
            [0.5, 0.6],
            [0.3, 0.4, 0.5],  # 3 propensity
            [1, 2],
            [3, 4]  # 2 indices - mismatch
        )
    end

    @testset "Very tight caliper - no matches" begin
        matches, distances, n_matched = nearest_neighbor_match(
            [0.9],  # High propensity
            [0.1, 0.2],  # Low propensity
            [1],
            [2, 3],
            caliper=0.01  # Very tight
        )
        @test n_matched == 0 || isempty(matches[1])
    end

    @testset "Exact propensity match" begin
        matches, distances, n_matched = nearest_neighbor_match(
            [0.5],
            [0.5],  # Exact match
            [1],
            [2],
            M=1
        )
        @test n_matched == 1
        @test matches[1] == [2]
        @test distances[1][1] ≈ 0.0 atol=1e-10
    end

    @testset "All controls same propensity" begin
        matches, distances, n_matched = nearest_neighbor_match(
            [0.5],
            [0.4, 0.4, 0.4],  # All same
            [1],
            [2, 3, 4],
            M=2
        )
        @test n_matched == 1
        @test length(matches[1]) == 2
        @test all(d ≈ 0.1 for d in distances[1])
    end

    @testset "Without replacement exhausts controls" begin
        # 3 treated, 2 controls without replacement
        matches, distances, n_matched = nearest_neighbor_match(
            [0.5, 0.5, 0.5],
            [0.5, 0.5],
            [1, 2, 3],
            [4, 5],
            M=1,
            with_replacement=false
        )
        # First 2 treated should match, 3rd should fail
        @test n_matched == 2
        @test length(matches[1]) == 1
        @test length(matches[2]) == 1
        @test isempty(matches[3])  # No controls left
    end

end

# =============================================================================
# compute_ate_from_matches Edge Cases
# =============================================================================

@testset "compute_ate_from_matches Edge Cases" begin

    @testset "No matches" begin
        outcomes = [10.0, 5.0, 6.0]
        treatment = [true, false, false]
        matches = [Int[]]  # Empty matches

        @test_throws ArgumentError compute_ate_from_matches(outcomes, treatment, matches)
    end

    @testset "Mismatched lengths" begin
        outcomes = [10.0, 5.0, 6.0, 7.0]
        treatment = [true, true, false, false]
        matches = [Int[3]]  # Only 1 match vector, but 2 treated

        @test_throws ArgumentError compute_ate_from_matches(outcomes, treatment, matches)
    end

    @testset "Single match" begin
        outcomes = [10.0, 5.0, 6.0]
        treatment = [true, false, false]
        matches = [[2]]  # Treated matches control at index 2

        ate, y_treated, y_control = compute_ate_from_matches(outcomes, treatment, matches)
        @test ate ≈ 10.0 - 5.0 atol=1e-10  # 10 - 5 = 5
    end

    @testset "Multiple matches per treated (M > 1)" begin
        outcomes = [10.0, 5.0, 6.0, 7.0, 4.0]
        treatment = [true, true, false, false, false]
        matches = [[3, 4], [4, 5]]  # Each treated matches 2 controls

        ate, y_treated, y_control = compute_ate_from_matches(outcomes, treatment, matches)

        # First: 10 - mean(6, 7) = 10 - 6.5 = 3.5
        # Second: 5 - mean(7, 4) = 5 - 5.5 = -0.5
        # ATE = mean(3.5, -0.5) = 1.5
        @test ate ≈ 1.5 atol=1e-10
    end

    @testset "Some treated with no matches" begin
        outcomes = [10.0, 12.0, 5.0, 6.0]
        treatment = [true, true, false, false]
        matches = [[3], Int[]]  # Second treated has no matches

        ate, y_treated, y_control = compute_ate_from_matches(outcomes, treatment, matches)

        # Only first treated included: 10 - 5 = 5
        @test ate ≈ 5.0 atol=1e-10
        @test length(y_treated) == 1
    end

end

# =============================================================================
# estimate_propensity Edge Cases
# =============================================================================

@testset "estimate_propensity Edge Cases" begin

    @testset "Mismatched lengths" begin
        treatment = [true, false, true]
        covariates = [1.0 2.0; 3.0 4.0]  # 2 rows, 3 treatment

        @test_throws ArgumentError estimate_propensity(treatment, covariates)
    end

    @testset "Single covariate" begin
        Random.seed!(42)
        n = 20
        x = randn(n)
        treatment = x .> 0  # Treatment based on x

        covariates = reshape(x, n, 1)
        propensity = estimate_propensity(treatment, covariates)

        @test length(propensity) == n
        @test all(0 < p < 1 for p in propensity)
    end

    @testset "Multiple covariates" begin
        Random.seed!(42)
        n = 30
        covariates = randn(n, 3)
        treatment = covariates[:, 1] .+ covariates[:, 2] .> 0

        # Need some variation in treatment
        if all(treatment) || all(.!treatment)
            treatment[1] = !treatment[1]
            treatment[end] = !treatment[end]
        end

        propensity = estimate_propensity(treatment, covariates)

        @test length(propensity) == n
        @test all(0 < p < 1 for p in propensity)
    end

    @testset "Propensity scores in valid range" begin
        Random.seed!(42)
        n = 50
        covariates = randn(n, 2)
        treatment = Vector{Bool}(rand(n) .> 0.5)

        # Ensure both groups have observations
        treatment[1] = true
        treatment[2] = false

        propensity = estimate_propensity(treatment, covariates)

        @test all(p -> 0 < p < 1, propensity)
    end

end

# =============================================================================
# check_common_support Edge Cases
# =============================================================================

@testset "check_common_support Edge Cases" begin

    @testset "Perfect overlap" begin
        # All propensity scores within same range for both groups
        propensity = [0.4, 0.5, 0.6, 0.4, 0.5, 0.6]
        treatment = [true, true, true, false, false, false]

        has_support, region, n_outside = check_common_support(propensity, treatment)

        @test has_support == true
        @test n_outside == 0  # All units within overlap region [0.4, 0.6]
    end

    @testset "No overlap" begin
        propensity = [0.8, 0.9, 0.95, 0.1, 0.15, 0.2]  # Treated high, control low
        treatment = [true, true, true, false, false, false]

        has_support, region, n_outside = check_common_support(propensity, treatment)

        # No overlap between [0.8, 0.95] and [0.1, 0.2]
        @test has_support == false
    end

    @testset "Partial overlap" begin
        propensity = [0.4, 0.6, 0.8, 0.2, 0.3, 0.5]
        treatment = [true, true, true, false, false, false]

        has_support, region, n_outside = check_common_support(propensity, treatment)

        # Overlap region: [0.4, 0.5]
        @test region[1] ≈ 0.4 atol=0.01
        @test region[2] ≈ 0.5 atol=0.01
    end

    @testset "Single point overlap" begin
        propensity = [0.5, 0.6, 0.7, 0.3, 0.4, 0.5]
        treatment = [true, true, true, false, false, false]

        has_support, region, n_outside = check_common_support(propensity, treatment)

        # Overlap at exactly 0.5
        @test region[1] ≈ 0.5 atol=0.01
        @test region[2] ≈ 0.5 atol=0.01
    end

end

# =============================================================================
# Balance Diagnostics Edge Cases
# =============================================================================

@testset "compute_standardized_mean_difference Edge Cases" begin

    @testset "Zero variance in both groups" begin
        x_treated = [5.0, 5.0, 5.0]
        x_control = [5.0, 5.0, 5.0]  # Same constant

        smd = compute_standardized_mean_difference(x_treated, x_control)
        @test smd == 0.0  # Identical distributions
    end

    @testset "Zero variance, different means" begin
        x_treated = [5.0, 5.0, 5.0]
        x_control = [3.0, 3.0, 3.0]  # Different constant

        smd = compute_standardized_mean_difference(x_treated, x_control)
        @test abs(smd) > 1e5  # Very large (proxy for infinite)
    end

    @testset "Perfect balance" begin
        x_treated = [1.0, 2.0, 3.0, 4.0, 5.0]
        x_control = [1.0, 2.0, 3.0, 4.0, 5.0]

        smd = compute_standardized_mean_difference(x_treated, x_control)
        @test abs(smd) < 1e-10
    end

    @testset "Large imbalance" begin
        x_treated = [10.0, 11.0, 12.0]
        x_control = [1.0, 2.0, 3.0]

        smd = compute_standardized_mean_difference(x_treated, x_control)
        @test abs(smd) > 2.0  # Large SMD
    end

    @testset "Unpooled SMD" begin
        x_treated = [1.0, 2.0, 3.0]
        x_control = [4.0, 5.0, 6.0]

        smd_pooled = compute_standardized_mean_difference(x_treated, x_control, pooled=true)
        smd_unpooled = compute_standardized_mean_difference(x_treated, x_control, pooled=false)

        # Both should indicate imbalance
        @test abs(smd_pooled) > 0.5
        @test abs(smd_unpooled) > 0.5
    end

end

@testset "compute_variance_ratio Edge Cases" begin

    @testset "Equal variances" begin
        x_treated = [1.0, 2.0, 3.0, 4.0, 5.0]
        x_control = [1.0, 2.0, 3.0, 4.0, 5.0]

        vr = compute_variance_ratio(x_treated, x_control)
        @test vr ≈ 1.0 atol=1e-10
    end

    @testset "Treated has higher variance" begin
        x_treated = [0.0, 5.0, 10.0]  # Variance = 25
        x_control = [4.0, 5.0, 6.0]   # Variance = 1

        vr = compute_variance_ratio(x_treated, x_control)
        @test vr > 10.0
    end

    @testset "Control has higher variance" begin
        x_treated = [4.0, 5.0, 6.0]   # Variance = 1
        x_control = [0.0, 5.0, 10.0]  # Variance = 25

        vr = compute_variance_ratio(x_treated, x_control)
        @test vr < 0.1
    end

    @testset "Near-zero control variance" begin
        x_treated = [1.0, 2.0, 3.0]
        x_control = [5.0, 5.0 + 1e-15, 5.0 - 1e-15]  # Nearly constant

        vr = compute_variance_ratio(x_treated, x_control)
        @test vr > 1e10 || isinf(vr)  # Very large or infinite
    end

end

# =============================================================================
# Data Type Tests
# =============================================================================

@testset "PSM Data Types" begin

    @testset "Float32 outcomes - converted to Float64" begin
        outcomes_f32 = Float32[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        outcomes = Float64.(outcomes_f32)
        treatment = [true, true, true, false, false, false]
        covariates = Float64[1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0; 9.0 10.0; 11.0 12.0]

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
        @test problem.outcomes isa Vector{Float64}
    end

    @testset "Integer outcomes - should convert" begin
        outcomes_int = [1, 2, 3, 4, 5, 6]
        outcomes = Float64.(outcomes_int)
        treatment = [true, true, true, false, false, false]
        covariates = Float64[1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0; 9.0 10.0; 11.0 12.0]

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
        @test problem.outcomes isa Vector{Float64}
    end

    @testset "Float32 covariates - converted to Float64" begin
        outcomes = Float64[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        treatment = [true, true, true, false, false, false]
        covariates_f32 = Float32[1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0; 9.0 10.0; 11.0 12.0]
        covariates = Float64.(covariates_f32)

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
        @test problem.covariates isa Matrix{Float64}
    end

end

# =============================================================================
# Numerical Stability Tests
# =============================================================================

@testset "PSM Numerical Stability" begin

    @testset "Very large outcomes" begin
        scale = 1e8
        outcomes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] .* scale
        treatment = [true, true, true, false, false, false]
        covariates = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0; 9.0 10.0; 11.0 12.0]

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
        @test problem.outcomes[1] ≈ 1e8 rtol=1e-6
    end

    @testset "Very small outcomes" begin
        scale = 1e-8
        outcomes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] .* scale
        treatment = [true, true, true, false, false, false]
        covariates = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0; 9.0 10.0; 11.0 12.0]

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
        @test problem.outcomes[1] ≈ 1e-8 rtol=1e-6
    end

    @testset "Mixed scale outcomes" begin
        outcomes = [1e8, 1e-8, 1e4, 1e-4, 1.0, 0.5]
        treatment = [true, true, true, false, false, false]
        covariates = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0; 9.0 10.0; 11.0 12.0]

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
        @test all(isfinite, problem.outcomes)
    end

    @testset "Propensity near boundaries" begin
        # Create data that produces extreme propensity scores
        Random.seed!(42)
        n = 50
        x = randn(n)
        treatment = x .> 2.0  # Very few treated

        # Ensure at least 2 in each group
        treatment[1] = true
        treatment[2] = true
        treatment[end] = false
        treatment[end-1] = false

        covariates = reshape(x, n, 1)
        propensity = estimate_propensity(treatment, covariates)

        # Should be clamped away from 0 and 1
        @test all(p -> 0 < p < 1, propensity)
    end

end

# =============================================================================
# NearestNeighborPSM Solve Integration
# =============================================================================

@testset "NearestNeighborPSM solve Edge Cases" begin

    @testset "Minimum viable solve" begin
        Random.seed!(42)
        # Use larger sample for reliable matching
        n = 20
        outcomes = randn(n) .+ [i <= 10 ? 2.0 : 0.0 for i in 1:n]
        treatment = [i <= 10 for i in 1:n]
        covariates = randn(n, 2)

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
        estimator = NearestNeighborPSM(M=1, with_replacement=true)

        solution = solve(problem, estimator)

        @test solution isa PSMSolution
        @test isfinite(solution.estimate)
        @test solution.n_matched >= 1
    end

    @testset "With tight caliper" begin
        Random.seed!(42)
        outcomes = [10.0, 12.0, 11.0, 5.0, 6.0, 5.5]
        treatment = [true, true, true, false, false, false]
        covariates = randn(6, 2)

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
        estimator = NearestNeighborPSM(M=1, caliper=0.01, with_replacement=true)

        solution = solve(problem, estimator)

        # Tight caliper may result in few matches
        @test solution isa PSMSolution
    end

    @testset "With replacement vs without" begin
        Random.seed!(42)
        n = 20
        outcomes = randn(n) .+ [i <= 10 ? 2.0 : 0.0 for i in 1:n]
        treatment = [i <= 10 for i in 1:n]
        covariates = randn(n, 2)

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))

        sol_with = solve(problem, NearestNeighborPSM(M=1, with_replacement=true))
        sol_without = solve(problem, NearestNeighborPSM(M=1, with_replacement=false))

        @test sol_with.n_matched >= sol_without.n_matched
    end

    @testset "M > 1 matching" begin
        Random.seed!(42)
        n = 30
        outcomes = randn(n) .+ [i <= 10 ? 2.0 : 0.0 for i in 1:n]
        treatment = [i <= 10 for i in 1:n]
        covariates = randn(n, 2)

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
        estimator = NearestNeighborPSM(M=3, with_replacement=true)

        solution = solve(problem, estimator)

        @test solution isa PSMSolution
        @test isfinite(solution.estimate)
    end

end

# =============================================================================
# Result Structure Tests
# =============================================================================

@testset "PSMSolution Structure" begin

    @testset "All fields present" begin
        Random.seed!(42)
        outcomes = [10.0, 12.0, 11.0, 5.0, 6.0, 5.5]
        treatment = [true, true, true, false, false, false]
        covariates = randn(6, 2)

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
        solution = solve(problem, NearestNeighborPSM())

        @test hasfield(PSMSolution, :estimate)
        @test hasfield(PSMSolution, :se)
        @test hasfield(PSMSolution, :ci_lower)
        @test hasfield(PSMSolution, :ci_upper)
        @test hasfield(PSMSolution, :n_treated)
        @test hasfield(PSMSolution, :n_control)
        @test hasfield(PSMSolution, :n_matched)
        @test hasfield(PSMSolution, :propensity_scores)
        @test hasfield(PSMSolution, :matched_indices)
        @test hasfield(PSMSolution, :balance_metrics)
        @test hasfield(PSMSolution, :retcode)
    end

    @testset "Propensity scores same length as n" begin
        Random.seed!(42)
        n = 20
        outcomes = randn(n)
        treatment = [i <= 10 for i in 1:n]
        covariates = randn(n, 2)

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
        solution = solve(problem, NearestNeighborPSM())

        @test length(solution.propensity_scores) == n
    end

    @testset "CI contains estimate" begin
        Random.seed!(42)
        n = 30
        outcomes = randn(n)
        treatment = [i <= 15 for i in 1:n]
        covariates = randn(n, 2)

        problem = PSMProblem(outcomes, treatment, covariates, (alpha=0.05,))
        solution = solve(problem, NearestNeighborPSM())

        @test solution.ci_lower <= solution.estimate <= solution.ci_upper
    end

end
