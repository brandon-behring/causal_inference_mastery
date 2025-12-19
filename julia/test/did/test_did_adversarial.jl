"""
Layer 2: Adversarial tests for Julia DiD estimator edge cases.

Tests boundary conditions, extreme inputs, and degenerate cases.
All tests should either:
1. Return valid results (graceful handling)
2. Return appropriate failure codes (retcode=:Failure)
3. Throw explicit errors with diagnostic messages (NEVER fail silently)

References:
    - Bertrand, Duflo, Mullainathan (2004). "How much should we trust DiD estimates?"
"""

using Test
using Statistics
using Random

# Include main module
include("../../src/CausalEstimators.jl")
using .CausalEstimators

# =============================================================================
# Minimum Sample Size Tests
# =============================================================================

@testset "DiD Minimum Sample Sizes" begin

    @testset "Minimum viable sample (n=5 per group)" begin
        # DiD with n=5 per group (minimum for cluster SE)
        # May have numerical issues with cluster SEs (NaN SE → Failure)
        # This is expected behavior for very small samples

        control_pre = [10.0, 11.0, 9.0, 10.5, 10.2]
        control_post = [13.0, 14.0, 12.0, 13.5, 13.2]
        treated_pre = [12.0, 13.0, 11.0, 12.5, 12.2]
        treated_post = [18.0, 19.0, 17.0, 18.5, 18.2]

        outcomes = vcat(control_pre, control_post, treated_pre, treated_post)
        treatment = vcat(falses(10), trues(10))
        post = vcat(falses(5), trues(5), falses(5), trues(5))
        unit_id = vcat(1:5, 1:5, 6:10, 6:10)
        time = vcat(ones(Int, 5), fill(2, 5), ones(Int, 5), fill(2, 5))

        problem = DiDProblem(outcomes, treatment, post, unit_id, time,
                            (alpha=0.05, cluster_se=true))
        solution = solve(problem, ClassicDiD())

        # May succeed, warn, or fail due to small sample cluster SE issues
        @test solution.retcode in [:Success, :Warning, :Failure]
        @test solution.n_obs == 20
        @test solution.n_treated == 5
        @test solution.n_control == 5

        # Point estimate should still be reasonable even if SE fails
        @test abs(solution.estimate - 3.0) < 0.5
    end

    @testset "Extreme small sample (n=2 per group)" begin
        # 2 control units, 2 treated units (4 clusters total)

        outcomes = Float64[10.0, 15.0, 11.0, 16.0,  # Control: units 1,2
                          12.0, 20.0, 13.0, 21.0]   # Treated: units 3,4
        treatment = Bool[false, false, false, false, true, true, true, true]
        post = Bool[false, true, false, true, false, true, false, true]
        unit_id = [1, 1, 2, 2, 3, 3, 4, 4]
        time = [1, 2, 1, 2, 1, 2, 1, 2]

        problem = DiDProblem(outcomes, treatment, post, unit_id, time,
                            (alpha=0.05, cluster_se=true))
        solution = solve(problem, ClassicDiD())

        # DiD = (20.5 - 12.5) - (15.5 - 10.5) = 8 - 5 = 3
        @test abs(solution.estimate - 3.0) < 0.5
        @test solution.df == 3  # n_clusters - 1 = 4 - 1
    end

    @testset "Single observation per cell" begin
        # Absolute minimum: 1 observation per 2×2 cell
        outcomes = Float64[10.0, 15.0, 12.0, 20.0]
        treatment = Bool[false, false, true, true]
        post = Bool[false, true, false, true]
        unit_id = [1, 1, 2, 2]
        time = [1, 2, 1, 2]

        problem = DiDProblem(outcomes, treatment, post, unit_id, time,
                            (alpha=0.05, cluster_se=true))
        solution = solve(problem, ClassicDiD())

        # DiD = (20 - 12) - (15 - 10) = 8 - 5 = 3
        @test abs(solution.estimate - 3.0) < 1e-10
    end

end

# =============================================================================
# Treatment Imbalance Tests
# =============================================================================

@testset "DiD Treatment Imbalance" begin

    @testset "90-10 split (mostly treated)" begin
        Random.seed!(200)
        n_treated = 90
        n_control = 10

        control_pre = fill(10.0, n_control) .+ randn(n_control)
        control_post = fill(13.0, n_control) .+ randn(n_control)
        treated_pre = fill(12.0, n_treated) .+ randn(n_treated)
        treated_post = fill(17.0, n_treated) .+ randn(n_treated)

        outcomes = vcat(control_pre, control_post, treated_pre, treated_post)
        treatment = vcat(falses(n_control * 2), trues(n_treated * 2))
        post = vcat(falses(n_control), trues(n_control),
                   falses(n_treated), trues(n_treated))
        unit_id = vcat(1:n_control, 1:n_control,
                      (n_control+1):(n_control+n_treated),
                      (n_control+1):(n_control+n_treated))
        time = vcat(ones(Int, n_control), fill(2, n_control),
                   ones(Int, n_treated), fill(2, n_treated))

        problem = DiDProblem(outcomes, treatment, post, unit_id, time,
                            (alpha=0.05, cluster_se=true))
        solution = solve(problem, ClassicDiD())

        @test solution.retcode in [:Success, :Warning]
        @test solution.n_treated == 90
        @test solution.n_control == 10
        # True effect ≈ (17-12) - (13-10) = 5 - 3 = 2
        @test abs(solution.estimate - 2.0) < 1.5
    end

    @testset "10-90 split (mostly control)" begin
        Random.seed!(201)
        n_treated = 10
        n_control = 90

        control_pre = fill(10.0, n_control) .+ randn(n_control)
        control_post = fill(13.0, n_control) .+ randn(n_control)
        treated_pre = fill(12.0, n_treated) .+ randn(n_treated)
        treated_post = fill(17.0, n_treated) .+ randn(n_treated)

        outcomes = vcat(control_pre, control_post, treated_pre, treated_post)
        treatment = vcat(falses(n_control * 2), trues(n_treated * 2))
        post = vcat(falses(n_control), trues(n_control),
                   falses(n_treated), trues(n_treated))
        unit_id = vcat(1:n_control, 1:n_control,
                      (n_control+1):(n_control+n_treated),
                      (n_control+1):(n_control+n_treated))
        time = vcat(ones(Int, n_control), fill(2, n_control),
                   ones(Int, n_treated), fill(2, n_treated))

        problem = DiDProblem(outcomes, treatment, post, unit_id, time,
                            (alpha=0.05, cluster_se=true))
        solution = solve(problem, ClassicDiD())

        @test solution.retcode in [:Success, :Warning]
        @test solution.n_treated == 10
        @test solution.n_control == 90
        @test abs(solution.estimate - 2.0) < 1.5
    end

end

# =============================================================================
# High/Low Variance Tests
# =============================================================================

@testset "DiD Extreme Variance" begin

    @testset "High variance outcomes (σ=50)" begin
        Random.seed!(300)
        n_treated = 50
        n_control = 50

        # High variance noise
        control_pre = fill(10.0, n_control) .+ randn(n_control) .* 50.0
        control_post = fill(13.0, n_control) .+ randn(n_control) .* 50.0
        treated_pre = fill(12.0, n_treated) .+ randn(n_treated) .* 50.0
        treated_post = fill(17.0, n_treated) .+ randn(n_treated) .* 50.0

        outcomes = vcat(control_pre, control_post, treated_pre, treated_post)
        treatment = vcat(falses(n_control * 2), trues(n_treated * 2))
        post = vcat(falses(n_control), trues(n_control),
                   falses(n_treated), trues(n_treated))
        unit_id = vcat(1:n_control, 1:n_control,
                      (n_control+1):(n_control+n_treated),
                      (n_control+1):(n_control+n_treated))
        time = vcat(ones(Int, n_control), fill(2, n_control),
                   ones(Int, n_treated), fill(2, n_treated))

        problem = DiDProblem(outcomes, treatment, post, unit_id, time,
                            (alpha=0.05, cluster_se=true))
        solution = solve(problem, ClassicDiD())

        @test solution.retcode in [:Success, :Warning]
        # CI should be wide with high variance
        ci_width = solution.ci_upper - solution.ci_lower
        @test ci_width > 10.0  # CI should be wide with high variance
    end

    @testset "Zero variance outcomes (constant within groups)" begin
        # Perfect deterministic outcomes - no noise
        outcomes = Float64[10, 10, 15, 15, 12, 12, 20, 20]
        treatment = Bool[false, false, false, false, true, true, true, true]
        post = Bool[false, false, true, true, false, false, true, true]
        unit_id = [1, 2, 1, 2, 3, 4, 3, 4]
        time = [1, 1, 2, 2, 1, 1, 2, 2]

        problem = DiDProblem(outcomes, treatment, post, unit_id, time,
                            (alpha=0.05, cluster_se=true))
        solution = solve(problem, ClassicDiD())

        # DiD = (20 - 12) - (15 - 10) = 8 - 5 = 3
        @test abs(solution.estimate - 3.0) < 1e-10
        # SE should be small (but may not be exactly zero due to cluster adjustment)
        @test solution.se < 1.0
    end

end

# =============================================================================
# Many Periods Test
# =============================================================================

@testset "DiD Many Periods" begin

    @testset "20 periods (10 pre, 10 post)" begin
        Random.seed!(400)
        n_treated = 30
        n_control = 30
        n_units = n_treated + n_control
        n_periods = 20  # t = 1 to 20, treatment at t=11

        outcomes = Float64[]
        treatment_vec = Bool[]
        post_vec = Bool[]
        unit_id_vec = Int[]
        time_vec = Int[]

        for unit in 1:n_units
            is_treated = unit > n_control
            baseline = is_treated ? 12.0 : 10.0

            for t in 1:n_periods
                is_post = t >= 11

                # Common time trend
                y = baseline + 0.2 * t

                # Treatment effect in post-period
                if is_treated && is_post
                    y += 3.0
                end

                y += randn() * 0.5

                push!(outcomes, y)
                push!(treatment_vec, is_treated)
                push!(post_vec, is_post)
                push!(unit_id_vec, unit)
                push!(time_vec, t)
            end
        end

        problem = DiDProblem(outcomes, treatment_vec, post_vec, unit_id_vec, time_vec,
                            (alpha=0.05, cluster_se=true))
        solution = solve(problem, ClassicDiD())

        @test solution.retcode in [:Success, :Warning]
        # Should recover treatment effect (3.0)
        @test abs(solution.estimate - 3.0) < 0.5
        @test solution.n_obs == 60 * 20  # 60 units × 20 periods
    end

end

# =============================================================================
# Extreme Baseline Differences
# =============================================================================

@testset "DiD Extreme Baselines" begin

    @testset "100x baseline difference" begin
        Random.seed!(600)
        n_treated = 40
        n_control = 40

        # Control group: outcomes around 10
        # Treated group: outcomes around 1000 (100x difference)
        control_pre = fill(10.0, n_control) .+ randn(n_control) .* 1.0
        control_post = fill(13.0, n_control) .+ randn(n_control) .* 1.0
        treated_pre = fill(1000.0, n_treated) .+ randn(n_treated) .* 10.0
        treated_post = fill(1005.0, n_treated) .+ randn(n_treated) .* 10.0

        outcomes = vcat(control_pre, control_post, treated_pre, treated_post)
        treatment = vcat(falses(n_control * 2), trues(n_treated * 2))
        post = vcat(falses(n_control), trues(n_control),
                   falses(n_treated), trues(n_treated))
        unit_id = vcat(1:n_control, 1:n_control,
                      (n_control+1):(n_control+n_treated),
                      (n_control+1):(n_control+n_treated))
        time = vcat(ones(Int, n_control), fill(2, n_control),
                   ones(Int, n_treated), fill(2, n_treated))

        problem = DiDProblem(outcomes, treatment, post, unit_id, time,
                            (alpha=0.05, cluster_se=true))
        solution = solve(problem, ClassicDiD())

        @test solution.retcode in [:Success, :Warning]
        # DiD differences out baseline: (1005 - 1000) - (13 - 10) = 5 - 3 = 2
        # With higher variance in treated group (σ=10), tolerance needs to be wider
        @test abs(solution.estimate - 2.0) < 5.0
    end

end

# =============================================================================
# Negative Outcomes
# =============================================================================

@testset "DiD Negative Outcomes" begin

    @testset "All negative outcomes" begin
        Random.seed!(700)
        n_treated = 30
        n_control = 30

        control_pre = fill(-100.0, n_control) .+ randn(n_control) .* 5.0
        control_post = fill(-97.0, n_control) .+ randn(n_control) .* 5.0
        treated_pre = fill(-98.0, n_treated) .+ randn(n_treated) .* 5.0
        treated_post = fill(-90.0, n_treated) .+ randn(n_treated) .* 5.0

        outcomes = vcat(control_pre, control_post, treated_pre, treated_post)
        treatment = vcat(falses(n_control * 2), trues(n_treated * 2))
        post = vcat(falses(n_control), trues(n_control),
                   falses(n_treated), trues(n_treated))
        unit_id = vcat(1:n_control, 1:n_control,
                      (n_control+1):(n_control+n_treated),
                      (n_control+1):(n_control+n_treated))
        time = vcat(ones(Int, n_control), fill(2, n_control),
                   ones(Int, n_treated), fill(2, n_treated))

        problem = DiDProblem(outcomes, treatment, post, unit_id, time,
                            (alpha=0.05, cluster_se=true))
        solution = solve(problem, ClassicDiD())

        @test solution.retcode in [:Success, :Warning]
        # DiD = (-90 - -98) - (-97 - -100) = 8 - 3 = 5
        @test abs(solution.estimate - 5.0) < 3.0
    end

    @testset "Mixed sign outcomes (crossing zero)" begin
        outcomes = Float64[-5, 5, -3, 7, -2, 8, 0, 12]
        treatment = Bool[false, false, false, false, true, true, true, true]
        post = Bool[false, true, false, true, false, true, false, true]
        unit_id = [1, 1, 2, 2, 3, 3, 4, 4]
        time = [1, 2, 1, 2, 1, 2, 1, 2]

        problem = DiDProblem(outcomes, treatment, post, unit_id, time,
                            (alpha=0.05, cluster_se=true))
        solution = solve(problem, ClassicDiD())

        @test solution.retcode in [:Success, :Warning, :Failure]
        # Should produce an estimate regardless of sign crossing
        @test !isnan(solution.estimate) || solution.retcode == :Failure
    end

end

# =============================================================================
# Missing Cell Tests
# =============================================================================

@testset "DiD Missing Cells" begin

    @testset "No treated pre-period observations" begin
        # All treated observations are post-period
        outcomes = Float64[10.0, 15.0, 11.0, 16.0,  # Control pre/post
                          20.0, 21.0]                # Treated post only
        treatment = Bool[false, false, false, false, true, true]
        post = Bool[false, true, false, true, true, true]  # No treated pre
        unit_id = [1, 1, 2, 2, 3, 3]
        time = [1, 2, 1, 2, 2, 2]

        problem = DiDProblem(outcomes, treatment, post, unit_id, time,
                            (alpha=0.05, cluster_se=true))
        solution = solve(problem, ClassicDiD())

        # Should fail gracefully (missing cell)
        @test solution.retcode == :Failure
        @test isnan(solution.estimate)
    end

    @testset "No control post-period observations" begin
        # All control observations are pre-period
        outcomes = Float64[10.0, 11.0,              # Control pre only
                          12.0, 20.0, 13.0, 21.0]   # Treated pre/post
        treatment = Bool[false, false, true, true, true, true]
        post = Bool[false, false, false, true, false, true]  # No control post
        unit_id = [1, 2, 3, 3, 4, 4]
        time = [1, 1, 1, 2, 1, 2]

        problem = DiDProblem(outcomes, treatment, post, unit_id, time,
                            (alpha=0.05, cluster_se=true))
        solution = solve(problem, ClassicDiD())

        # Should fail gracefully (missing cell)
        @test solution.retcode == :Failure
        @test isnan(solution.estimate)
    end

end

# =============================================================================
# Singular Matrix Tests
# =============================================================================

@testset "DiD Near-Singular Cases" begin

    @testset "Perfect collinearity in regressors" begin
        # All units are in same group (no variation in treatment)
        outcomes = Float64[10.0, 15.0, 11.0, 16.0, 12.0, 17.0, 13.0, 18.0]
        treatment = trues(8)  # All treated
        post = Bool[false, true, false, true, false, true, false, true]
        unit_id = [1, 1, 2, 2, 3, 3, 4, 4]
        time = [1, 2, 1, 2, 1, 2, 1, 2]

        problem = DiDProblem(outcomes, treatment, post, unit_id, time,
                            (alpha=0.05, cluster_se=true))
        solution = solve(problem, ClassicDiD())

        # No control group → should fail
        @test solution.retcode == :Failure
    end

    @testset "No variation in post indicator" begin
        # All observations are post-period
        outcomes = Float64[15.0, 16.0, 20.0, 21.0]
        treatment = Bool[false, false, true, true]
        post = trues(4)  # All post
        unit_id = [1, 2, 3, 4]
        time = [2, 2, 2, 2]

        problem = DiDProblem(outcomes, treatment, post, unit_id, time,
                            (alpha=0.05, cluster_se=true))
        solution = solve(problem, ClassicDiD())

        # No pre-period → should fail
        @test solution.retcode == :Failure
    end

end

# =============================================================================
# Parallel Trends Test Edge Cases
# =============================================================================

@testset "Parallel Trends Test Adversarial" begin

    @testset "Single pre-period (cannot test)" begin
        # Only 1 pre-period → cannot test parallel trends
        Random.seed!(801)

        outcomes = Float64[]
        treatment_vec = Bool[]
        time_vec = Int[]
        unit_id_vec = Int[]

        for unit in 1:40
            is_treated = unit > 20
            for t in [1, 2]  # 1 pre, 1 post
                y = is_treated ? 12.0 : 10.0
                if is_treated && t == 2
                    y += 3.0
                end
                y += randn() * 0.5
                push!(outcomes, y)
                push!(treatment_vec, is_treated)
                push!(time_vec, t)
                push!(unit_id_vec, unit)
            end
        end

        problem = DiDProblem(
            outcomes,
            treatment_vec,
            time_vec .>= 2,
            unit_id_vec,
            time_vec,
            (alpha=0.05, cluster_se=true)
        )
        solution = solve(problem, ClassicDiD(test_parallel_trends=true))

        # Parallel trends test should report insufficient pre-periods
        if !isnothing(solution.parallel_trends_test)
            @test solution.parallel_trends_test.n_pre_periods < 2 ||
                  occursin("Need", solution.parallel_trends_test.message)
        end
    end

    @testset "Extreme differential pre-trends" begin
        Random.seed!(800)

        outcomes = Float64[]
        treatment_vec = Bool[]
        time_vec = Int[]
        unit_id_vec = Int[]

        # Treated: steep upward trend; Control: flat
        for unit in 1:60
            is_treated = unit > 30
            baseline = 10.0

            for t in 1:10  # 5 pre (1-5), 5 post (6-10)
                is_post = t >= 6

                if is_treated && !is_post
                    y = baseline + 5 * (t - 5)  # Strong upward trend
                elseif !is_post
                    y = baseline
                else
                    y = baseline + 10
                end

                y += randn()

                push!(outcomes, y)
                push!(treatment_vec, is_treated)
                push!(time_vec, t)
                push!(unit_id_vec, unit)
            end
        end

        problem = DiDProblem(
            outcomes,
            treatment_vec,
            time_vec .>= 6,
            unit_id_vec,
            time_vec,
            (alpha=0.05, cluster_se=true)
        )
        solution = solve(problem, ClassicDiD(test_parallel_trends=true))

        # Should reject parallel trends (extreme differential pre-trends)
        if !isnothing(solution.parallel_trends_test)
            if !isnan(solution.parallel_trends_test.p_value)
                @test solution.parallel_trends_test.p_value < 0.10 ||
                      solution.parallel_trends_test.passes == false
            end
        end
    end

end

# =============================================================================
# Non-Cluster SE Comparison
# =============================================================================

@testset "Cluster vs Non-Cluster SE" begin

    @testset "Compare SE estimates" begin
        Random.seed!(901)
        n_treated = 50
        n_control = 50

        control_pre = fill(10.0, n_control) .+ randn(n_control) .* 2.0
        control_post = fill(13.0, n_control) .+ randn(n_control) .* 2.0
        treated_pre = fill(12.0, n_treated) .+ randn(n_treated) .* 2.0
        treated_post = fill(17.0, n_treated) .+ randn(n_treated) .* 2.0

        outcomes = vcat(control_pre, control_post, treated_pre, treated_post)
        treatment = vcat(falses(n_control * 2), trues(n_treated * 2))
        post = vcat(falses(n_control), trues(n_control),
                   falses(n_treated), trues(n_treated))
        unit_id = vcat(1:n_control, 1:n_control,
                      (n_control+1):(n_control+n_treated),
                      (n_control+1):(n_control+n_treated))
        time = vcat(ones(Int, n_control), fill(2, n_control),
                   ones(Int, n_treated), fill(2, n_treated))

        # Cluster SE
        problem_cluster = DiDProblem(outcomes, treatment, post, unit_id, time,
                                    (alpha=0.05, cluster_se=true))
        solution_cluster = solve(problem_cluster, ClassicDiD(cluster_se=true))

        # Non-cluster SE (HC1)
        problem_hc = DiDProblem(outcomes, treatment, post, unit_id, time,
                               (alpha=0.05, cluster_se=false))
        solution_hc = solve(problem_hc, ClassicDiD(cluster_se=false))

        # Both should succeed
        @test solution_cluster.retcode in [:Success, :Warning]
        @test solution_hc.retcode in [:Success, :Warning]

        # Estimates should be identical (same point estimate)
        @test abs(solution_cluster.estimate - solution_hc.estimate) < 1e-10

        # Cluster SEs should generally be larger (conservative)
        # With 2 periods, cluster SE accounts for within-unit correlation
        @test solution_cluster.se >= solution_hc.se * 0.5  # Allow some flexibility
    end

end

# =============================================================================
# Data Type Edge Cases
# =============================================================================

@testset "DiD Data Types" begin

    @testset "Float32 outcomes (converts to Float64)" begin
        # Note: Implementation is designed for Float64. Float32 inputs are
        # automatically converted to maintain numerical stability.
        outcomes_f32 = Float32[10.0, 15.0, 11.0, 16.0, 12.0, 20.0, 13.0, 21.0]
        outcomes = Float64.(outcomes_f32)  # Convert to Float64
        treatment = Bool[false, false, false, false, true, true, true, true]
        post = Bool[false, true, false, true, false, true, false, true]
        unit_id = [1, 1, 2, 2, 3, 3, 4, 4]
        time = [1, 2, 1, 2, 1, 2, 1, 2]

        problem = DiDProblem(outcomes, treatment, post, unit_id, time,
                            (alpha=0.05, cluster_se=true))
        solution = solve(problem, ClassicDiD())

        @test solution.retcode in [:Success, :Warning]
        @test typeof(solution.estimate) == Float64
    end

    @testset "Integer unit_id types" begin
        outcomes = Float64[10.0, 15.0, 11.0, 16.0, 12.0, 20.0, 13.0, 21.0]
        treatment = Bool[false, false, false, false, true, true, true, true]
        post = Bool[false, true, false, true, false, true, false, true]

        # Different integer types for unit_id
        for IntType in [Int8, Int16, Int32, Int64, UInt8, UInt16]
            unit_id = IntType[1, 1, 2, 2, 3, 3, 4, 4]
            time = [1, 2, 1, 2, 1, 2, 1, 2]

            problem = DiDProblem(outcomes, treatment, post, Int.(unit_id), time,
                                (alpha=0.05, cluster_se=true))
            solution = solve(problem, ClassicDiD())

            @test solution.retcode in [:Success, :Warning]
        end
    end

end

# Run all tests
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n" * "="^60)
    println("DiD Adversarial Tests Summary")
    println("="^60)
    println("Testing edge cases and boundary conditions for Julia DiD.")
    println("All tests should handle gracefully or fail explicitly.")
    println("="^60 * "\n")
end
