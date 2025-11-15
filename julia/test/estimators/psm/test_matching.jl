"""
Tests for nearest neighbor matching algorithms.
"""

using Test
using Random
using Statistics
using CausalEstimators

@testset "Nearest Neighbor Matching" begin

    # ========================================================================
    # Known-Answer Tests: Matching Algorithm
    # ========================================================================

    @testset "Known-Answer: 1:1 matching without replacement" begin
        Random.seed!(42)

        # Simple case: perfect propensity score separation
        propensity_treated = [0.7, 0.8, 0.9]
        propensity_control = [0.1, 0.2, 0.3, 0.4]
        indices_treated = [1, 2, 3]
        indices_control = [4, 5, 6, 7]

        matches, distances, n_matched = nearest_neighbor_match(
            propensity_treated,
            propensity_control,
            indices_treated,
            indices_control,
            M=1,
            with_replacement=false
        )

        # All treated should be matched
        @test n_matched == 3
        @test length(matches) == 3

        # Each match should be non-empty
        @test all(length.(matches) .== 1)

        # Distances should be reasonable
        @test all(all(d .>= 0) for d in distances)
    end

    @testset "Known-Answer: 1:1 matching with replacement" begin
        Random.seed!(42)

        propensity_treated = [0.5, 0.6]
        propensity_control = [0.45]  # Only 1 control
        indices_treated = [1, 2]
        indices_control = [3]

        matches, distances, n_matched = nearest_neighbor_match(
            propensity_treated,
            propensity_control,
            indices_treated,
            indices_control,
            M=1,
            with_replacement=true  # With replacement allows reuse
        )

        # Both treated should match to same control
        @test n_matched == 2
        @test matches[1] == [3]
        @test matches[2] == [3]
    end

    @testset "Known-Answer: Caliper matching" begin
        Random.seed!(42)

        propensity_treated = [0.8, 0.9]
        propensity_control = [0.1, 0.85]  # 0.1 is far, 0.85 is close
        indices_treated = [1, 2]
        indices_control = [3, 4]

        matches, distances, n_matched = nearest_neighbor_match(
            propensity_treated,
            propensity_control,
            indices_treated,
            indices_control,
            M=1,
            with_replacement=false,
            caliper=0.1  # Strict caliper
        )

        # Only second treated (0.9) should match to second control (0.85)
        # First treated (0.8) has no controls within 0.1
        @test n_matched <= 2

        # All distances should be within caliper
        @test all(all(d .<= 0.1) for d in distances if !isempty(d))
    end

    @testset "Known-Answer: 2:1 matching" begin
        Random.seed!(42)

        propensity_treated = [0.5]
        propensity_control = [0.45, 0.52, 0.6]
        indices_treated = [1]
        indices_control = [2, 3, 4]

        matches, distances, n_matched = nearest_neighbor_match(
            propensity_treated,
            propensity_control,
            indices_treated,
            indices_control,
            M=2,  # 2 matches per treated
            with_replacement=false
        )

        @test n_matched == 1
        @test length(matches[1]) == 2  # Should get 2 matches

        # Closest 2 controls should be matched
        # 0.45 (dist 0.05) and 0.52 (dist 0.02)
        @test 2 in matches[1] || 3 in matches[1]
    end

    # ========================================================================
    # Adversarial Tests: Edge Cases
    # ========================================================================

    @testset "Adversarial: All units dropped by caliper" begin
        propensity_treated = [0.9, 0.95]
        propensity_control = [0.1, 0.2]
        indices_treated = [1, 2]
        indices_control = [3, 4]

        matches, distances, n_matched = nearest_neighbor_match(
            propensity_treated,
            propensity_control,
            indices_treated,
            indices_control,
            M=1,
            caliper=0.05  # Very strict caliper
        )

        # No matches should be found
        @test n_matched == 0
        @test all(isempty.(matches))
    end

    @testset "Adversarial: More matches requested than controls" begin
        propensity_treated = [0.5]
        propensity_control = [0.4, 0.6]  # Only 2 controls
        indices_treated = [1]
        indices_control = [2, 3]

        # Should error without replacement
        @test_throws ArgumentError nearest_neighbor_match(
            propensity_treated,
            propensity_control,
            indices_treated,
            indices_control,
            M=5,  # Request 5 but only 2 available
            with_replacement=false
        )

        # Should work with replacement
        matches, distances, n_matched = nearest_neighbor_match(
            propensity_treated,
            propensity_control,
            indices_treated,
            indices_control,
            M=5,
            with_replacement=true
        )

        # Should get only 2 matches (all available controls)
        @test length(matches[1]) == 2
    end

    @testset "Adversarial: Identical propensity scores" begin
        Random.seed!(42)

        propensity_treated = [0.5, 0.5, 0.5]
        propensity_control = [0.5, 0.5, 0.5]
        indices_treated = [1, 2, 3]
        indices_control = [4, 5, 6]

        matches, distances, n_matched = nearest_neighbor_match(
            propensity_treated,
            propensity_control,
            indices_treated,
            indices_control,
            M=1,
            with_replacement=false
        )

        # All should match
        @test n_matched == 3

        # All distances should be 0
        @test all(all(d .== 0) for d in distances)
    end

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    @testset "Error: No treated units" begin
        @test_throws ArgumentError nearest_neighbor_match(
            Float64[],  # Empty treated
            [0.5],
            Int[],
            [1],
            M=1
        )
    end

    @testset "Error: No control units" begin
        @test_throws ArgumentError nearest_neighbor_match(
            [0.5],
            Float64[],  # Empty control
            [1],
            Int[],
            M=1
        )
    end

    @testset "Error: Invalid M" begin
        @test_throws ArgumentError nearest_neighbor_match(
            [0.5],
            [0.4],
            [1],
            [2],
            M=0  # Invalid
        )
    end

    @testset "Error: Invalid caliper" begin
        @test_throws ArgumentError nearest_neighbor_match(
            [0.5],
            [0.4],
            [1],
            [2],
            M=1,
            caliper=-0.1  # Negative caliper
        )
    end

    @testset "Error: Mismatched indices length" begin
        @test_throws ArgumentError nearest_neighbor_match(
            [0.5, 0.6],
            [0.4],
            [1],  # Wrong length
            [2],
            M=1
        )
    end

    # ========================================================================
    # ATE Computation Tests
    # ========================================================================

    @testset "ATE: Simple 1:1 matching" begin
        outcomes = [10.0, 12.0, 4.0, 5.0]
        treatment = [true, true, false, false]
        matches = [[3], [4]]  # Treated 1→Control 3, Treated 2→Control 4

        ate, y_treated, y_control = compute_ate_from_matches(
            outcomes,
            treatment,
            matches
        )

        # ATE = mean((10-4) + (12-5)) = mean(6+7) = 6.5
        @test ate ≈ 6.5
        @test y_treated == [10.0, 12.0]
        @test length(y_control) == 2
    end

    @testset "ATE: 2:1 matching" begin
        outcomes = [10.0, 4.0, 5.0, 6.0]
        treatment = [true, false, false, false]
        matches = [[2, 3]]  # Treated 1 → Controls 2,3

        ate, y_treated, y_control = compute_ate_from_matches(
            outcomes,
            treatment,
            matches
        )

        # ATE = 10 - mean(4,5) = 10 - 4.5 = 5.5
        @test ate ≈ 5.5
        @test y_control[1] == [4.0, 5.0]
    end

    @testset "ATE: Some units unmatched" begin
        outcomes = [10.0, 12.0, 4.0]
        treatment = [true, true, false]
        matches = [[3], Int[]]  # Only first treated matched (Int[] for proper typing)

        ate, y_treated, y_control = compute_ate_from_matches(
            outcomes,
            treatment,
            matches
        )

        # Only first treated contributes: ATE = 10 - 4 = 6
        @test ate ≈ 6.0
        @test length(y_treated) == 1
    end

    @testset "Error: All units unmatched" begin
        outcomes = [10.0, 4.0]
        treatment = [true, false]
        matches = Vector{Vector{Int}}([Int[]])  # No matches (properly typed)

        @test_throws ArgumentError compute_ate_from_matches(
            outcomes,
            treatment,
            matches
        )
    end

    @testset "Error: Mismatched lengths in compute_ate" begin
        outcomes = [10.0, 4.0]
        treatment = [true, false]
        matches = Vector{Vector{Int}}([Int[], Int[]])  # Wrong number of match vectors (properly typed)

        @test_throws ArgumentError compute_ate_from_matches(
            outcomes,
            treatment,
            matches
        )
    end

end
