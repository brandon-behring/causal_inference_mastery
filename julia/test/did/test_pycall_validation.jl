"""
PyCall validation tests for DiD estimators.

Validates Julia DiD implementation against Python implementation using identical data.
"""

using Test
using CausalEstimators
using PyCall
using Statistics
using Random

# Add Python project root to sys.path
pushfirst!(PyVector(pyimport("sys")."path"), "/home/brandon_behring/Claude/causal_inference_mastery")

# Import Python DiD module
const did_py = pyimport("src.causal_inference.did.did_estimator")

@testset "PyCall DiD Validation" begin

    @testset "Classic DiD - Hand Calculation Match" begin
        # Test with simple hand-calculable values from Python test
        # Control: pre=[10, 10], post=[15, 15] (change = +5)
        # Treated: pre=[12, 12], post=[20, 20] (change = +8)
        # DiD = 8 - 5 = 3

        outcomes = [10.0, 10.0, 15.0, 15.0, 12.0, 12.0, 20.0, 20.0]
        treatment = [false, false, false, false, true, true, true, true]
        post = [false, false, true, true, false, false, true, true]
        unit_id = [1, 2, 1, 2, 3, 4, 3, 4]

        # Julia implementation
        problem_jl = DiDProblem(outcomes, treatment, post, unit_id, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, ClassicDiD())

        # Python implementation
        result_py = did_py.did_2x2(
            outcomes=outcomes,
            treatment=Int.(treatment),
            post=Int.(post),
            unit_id=unit_id .- 1,  # Python uses 0-indexed
            cluster_se=true
        )

        # Compare estimates
        @test abs(solution_jl.estimate - result_py["estimate"]) < 1e-10
        @test abs(solution_jl.estimate - 3.0) < 1e-10  # Known answer

        # Compare standard errors (allow for perfect fit edge case)
        # Julia may get exact 0.0, Python gets near-zero (9.78e-15)
        @test abs(solution_jl.se - result_py["se"]) < 1e-6 || (solution_jl.se == 0.0 && result_py["se"] < 1e-10)

        # Compare confidence intervals (if SEs are near-zero, CIs collapse to point estimate)
        if solution_jl.se < 1e-10 && result_py["se"] < 1e-10
            # Perfect fit: CIs should equal estimate
            @test abs(solution_jl.ci_lower - solution_jl.estimate) < 1e-6
            @test abs(solution_jl.ci_upper - solution_jl.estimate) < 1e-6
        else
            @test abs(solution_jl.ci_lower - result_py["ci_lower"]) < 1e-6
            @test abs(solution_jl.ci_upper - result_py["ci_upper"]) < 1e-6
        end

        # Compare p-values (skip if SE = 0 causing NaN or Inf)
        if !isnan(solution_jl.p_value) && !isinf(solution_jl.p_value)
            @test abs(solution_jl.p_value - result_py["p_value"]) < 1e-6
        else
            # Perfect fit: Julia gets NaN/Inf, Python gets very small p-value
            @test result_py["p_value"] < 1e-10 || isnan(result_py["p_value"])
        end

        # Compare diagnostics
        @test solution_jl.n_treated == result_py["n_treated"]
        @test solution_jl.n_control == result_py["n_control"]
        @test solution_jl.df == result_py["df"]
    end

    @testset "Classic DiD - Zero Treatment Effect" begin
        # Generate data with no treatment effect
        Random.seed!(123)
        n_control = 50
        n_treated = 50

        control_pre = fill(10.0, n_control) .+ randn(n_control) .* 0.5
        control_post = fill(12.0, n_control) .+ randn(n_control) .* 0.5
        treated_pre = fill(11.0, n_treated) .+ randn(n_treated) .* 0.5
        treated_post = fill(13.0, n_treated) .+ randn(n_treated) .* 0.5  # Same change as control

        outcomes = vcat(control_pre, control_post, treated_pre, treated_post)
        treatment = vcat(fill(false, 2*n_control), fill(true, 2*n_treated))
        post = vcat(
            fill(false, n_control), fill(true, n_control),
            fill(false, n_treated), fill(true, n_treated)
        )
        unit_id = vcat(
            1:n_control, 1:n_control,
            (n_control+1):(n_control+n_treated), (n_control+1):(n_control+n_treated)
        )

        # Julia implementation
        problem_jl = DiDProblem(outcomes, treatment, post, unit_id, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, ClassicDiD())

        # Python implementation
        result_py = did_py.did_2x2(
            outcomes=outcomes,
            treatment=Int.(treatment),
            post=Int.(post),
            unit_id=unit_id .- 1,
            cluster_se=true
        )

        # Both should estimate ~0
        @test abs(solution_jl.estimate) < 0.5
        @test abs(result_py["estimate"]) < 0.5

        # Estimates should match
        @test abs(solution_jl.estimate - result_py["estimate"]) < 1e-10

        # Standard errors should match
        @test abs(solution_jl.se - result_py["se"]) < 1e-6

        # CI should contain zero in both
        @test solution_jl.ci_lower <= 0.0 <= solution_jl.ci_upper
        @test result_py["ci_lower"] <= 0.0 <= result_py["ci_upper"]
    end

    @testset "Classic DiD - Positive Treatment Effect" begin
        # Generate data with positive treatment effect
        Random.seed!(456)
        n_control = 50
        n_treated = 50
        true_effect = 2.0

        control_pre = fill(10.0, n_control) .+ randn(n_control) .* 0.5
        control_post = fill(12.0, n_control) .+ randn(n_control) .* 0.5
        treated_pre = fill(11.0, n_treated) .+ randn(n_treated) .* 0.5
        treated_post = fill(15.0, n_treated) .+ randn(n_treated) .* 0.5  # +2 effect

        outcomes = vcat(control_pre, control_post, treated_pre, treated_post)
        treatment = vcat(fill(false, 2*n_control), fill(true, 2*n_treated))
        post = vcat(
            fill(false, n_control), fill(true, n_control),
            fill(false, n_treated), fill(true, n_treated)
        )
        unit_id = vcat(
            1:n_control, 1:n_control,
            (n_control+1):(n_control+n_treated), (n_control+1):(n_control+n_treated)
        )

        # Julia implementation
        problem_jl = DiDProblem(outcomes, treatment, post, unit_id, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, ClassicDiD())

        # Python implementation
        result_py = did_py.did_2x2(
            outcomes=outcomes,
            treatment=Int.(treatment),
            post=Int.(post),
            unit_id=unit_id .- 1,
            cluster_se=true
        )

        # Both should estimate ~2.0
        @test abs(solution_jl.estimate - true_effect) < 0.5
        @test abs(result_py["estimate"] - true_effect) < 0.5

        # Estimates should match exactly
        @test abs(solution_jl.estimate - result_py["estimate"]) < 1e-10

        # Standard errors should match
        @test abs(solution_jl.se - result_py["se"]) < 1e-6

        # Both should detect significance
        @test solution_jl.p_value < 0.05
        @test result_py["p_value"] < 0.05
    end

    @testset "Classic DiD - Cluster vs Heteroskedasticity-Robust SE" begin
        # Same data, different SE methods
        Random.seed!(789)
        n_control = 30
        n_treated = 30

        control_pre = fill(8.0, n_control) .+ randn(n_control) .* 1.0
        control_post = fill(10.0, n_control) .+ randn(n_control) .* 1.0
        treated_pre = fill(9.0, n_treated) .+ randn(n_treated) .* 1.0
        treated_post = fill(14.0, n_treated) .+ randn(n_treated) .* 1.0

        outcomes = vcat(control_pre, control_post, treated_pre, treated_post)
        treatment = vcat(fill(false, 2*n_control), fill(true, 2*n_treated))
        post = vcat(
            fill(false, n_control), fill(true, n_control),
            fill(false, n_treated), fill(true, n_treated)
        )
        unit_id = vcat(
            1:n_control, 1:n_control,
            (n_control+1):(n_control+n_treated), (n_control+1):(n_control+n_treated)
        )

        # Julia - Cluster SE
        problem_jl = DiDProblem(outcomes, treatment, post, unit_id, nothing, (alpha=0.05,))
        solution_cluster_jl = solve(problem_jl, ClassicDiD(cluster_se=true))
        solution_hc_jl = solve(problem_jl, ClassicDiD(cluster_se=false))

        # Python - Cluster SE
        result_cluster_py = did_py.did_2x2(
            outcomes=outcomes,
            treatment=Int.(treatment),
            post=Int.(post),
            unit_id=unit_id .- 1,
            cluster_se=true
        )

        # Python - HC SE
        result_hc_py = did_py.did_2x2(
            outcomes=outcomes,
            treatment=Int.(treatment),
            post=Int.(post),
            unit_id=unit_id .- 1,
            cluster_se=false
        )

        # Point estimates should match (SE method doesn't affect estimate)
        @test abs(solution_cluster_jl.estimate - result_cluster_py["estimate"]) < 1e-10
        @test abs(solution_hc_jl.estimate - result_hc_py["estimate"]) < 1e-10
        @test abs(solution_cluster_jl.estimate - solution_hc_jl.estimate) < 1e-10

        # Cluster SEs should match between Julia and Python
        @test abs(solution_cluster_jl.se - result_cluster_py["se"]) < 1e-6

        # HC SEs should be similar (Julia uses HC1, Python uses HC3)
        # Relax tolerance due to different HC formulas
        @test abs(solution_hc_jl.se - result_hc_py["se"]) < 1e-2

        # Cluster SE should be >= HC SE (usually, with serial correlation)
        @test solution_cluster_jl.se >= solution_hc_jl.se * 0.9  # Allow 10% tolerance
    end

    @testset "Classic DiD - Negative Treatment Effect" begin
        # Test negative effects
        Random.seed!(999)
        n_control = 40
        n_treated = 40
        true_effect = -3.0

        control_pre = fill(15.0, n_control) .+ randn(n_control) .* 0.8
        control_post = fill(17.0, n_control) .+ randn(n_control) .* 0.8
        treated_pre = fill(16.0, n_treated) .+ randn(n_treated) .* 0.8
        treated_post = fill(16.0, n_treated) .+ randn(n_treated) .* 0.8  # -1 relative change

        outcomes = vcat(control_pre, control_post, treated_pre, treated_post)
        treatment = vcat(fill(false, 2*n_control), fill(true, 2*n_treated))
        post = vcat(
            fill(false, n_control), fill(true, n_control),
            fill(false, n_treated), fill(true, n_treated)
        )
        unit_id = vcat(
            1:n_control, 1:n_control,
            (n_control+1):(n_control+n_treated), (n_control+1):(n_control+n_treated)
        )

        # Julia implementation
        problem_jl = DiDProblem(outcomes, treatment, post, unit_id, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, ClassicDiD())

        # Python implementation
        result_py = did_py.did_2x2(
            outcomes=outcomes,
            treatment=Int.(treatment),
            post=Int.(post),
            unit_id=unit_id .- 1,
            cluster_se=true
        )

        # Both should be negative
        @test solution_jl.estimate < 0
        @test result_py["estimate"] < 0

        # Estimates should match
        @test abs(solution_jl.estimate - result_py["estimate"]) < 1e-10

        # CI upper bounds should be similar (some tolerance due to adjustment differences)
        @test abs(solution_jl.ci_upper - result_py["ci_upper"]) < 0.05
    end

    @testset "Classic DiD - Multiple Pre/Post Periods" begin
        # Test with 3 pre-periods and 2 post-periods
        Random.seed!(1010)
        n_control = 20
        n_treated = 20
        n_pre = 3
        n_post = 2

        outcomes = Float64[]
        treatment_vec = Bool[]
        post_vec = Bool[]
        unit_id_vec = Int[]

        for unit in 1:(n_control + n_treated)
            is_treated = unit > n_control
            baseline = is_treated ? 12.0 : 10.0

            for t in 1:(n_pre + n_post)
                is_post = t > n_pre
                y = baseline + 0.5 * t  # Time trend

                if is_treated && is_post
                    y += 5.0  # Treatment effect
                end

                y += randn() * 0.5

                push!(outcomes, y)
                push!(treatment_vec, is_treated)
                push!(post_vec, is_post)
                push!(unit_id_vec, unit)
            end
        end

        # Julia implementation
        problem_jl = DiDProblem(outcomes, treatment_vec, post_vec, unit_id_vec, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, ClassicDiD())

        # Python implementation
        result_py = did_py.did_2x2(
            outcomes=outcomes,
            treatment=Int.(treatment_vec),
            post=Int.(post_vec),
            unit_id=unit_id_vec .- 1,
            cluster_se=true
        )

        # Estimates should match
        @test abs(solution_jl.estimate - result_py["estimate"]) < 1e-10

        # Should detect effect
        @test abs(solution_jl.estimate - 5.0) < 1.0
        @test solution_jl.p_value < 0.05
        @test result_py["p_value"] < 0.05

        # Diagnostics should match
        @test solution_jl.n_obs == length(outcomes)
        @test solution_jl.n_treated == n_treated
        @test solution_jl.n_control == n_control
    end

end  # PyCall DiD Validation
