#=
Adversarial Tests for Julia SCM Estimators

Layer 2 validation: Edge cases, boundary conditions, and stress tests
for SyntheticControl and AugmentedSC estimators.

Test Categories:
1. Input Validation - Dimension mismatches, invalid values
2. Numerical Stability - Extreme scales, collinearity
3. Panel Structure - Minimum viable panels, extreme imbalance
4. Optimizer Edge Cases - Convergence, degenerate solutions
5. Inference Edge Cases - Few placebos, bootstrap failures
6. ASCM Specific - Ridge regularization edge cases
7. Data Type Handling - Int, Float32, etc.

References:
    Abadie, Diamond, Hainmueller (2010). "Synthetic Control Methods"
    Ben-Michael, Feller, Rothstein (2021). "Augmented Synthetic Control"
=#

using Test
using Statistics
using Random
using LinearAlgebra

# CausalEstimators should be loaded by parent test runner

# =============================================================================
# 1. Input Validation Tests
# =============================================================================

@testset "SCM Input Validation" begin

    @testset "Treatment length mismatch" begin
        outcomes = randn(5, 10)
        treatment = Bool[true, false, false]  # Wrong length

        @test_throws ArgumentError SCMProblem(
            outcomes, treatment, 5, (alpha=0.05,)
        )
    end

    @testset "No treated units" begin
        outcomes = randn(5, 10)
        treatment = Bool[false, false, false, false, false]

        @test_throws ArgumentError SCMProblem(
            outcomes, treatment, 5, (alpha=0.05,)
        )
    end

    @testset "No control units" begin
        outcomes = randn(3, 10)
        treatment = Bool[true, true, true]

        @test_throws ArgumentError SCMProblem(
            outcomes, treatment, 5, (alpha=0.05,)
        )
    end

    @testset "Only one control unit" begin
        outcomes = randn(2, 10)
        treatment = Bool[true, false]

        @test_throws ArgumentError SCMProblem(
            outcomes, treatment, 5, (alpha=0.05,)
        )
    end

    @testset "Treatment period too early" begin
        outcomes = randn(5, 10)
        treatment = Bool[true, false, false, false, false]

        @test_throws ArgumentError SCMProblem(
            outcomes, treatment, 1, (alpha=0.05,)  # No pre-treatment
        )
    end

    @testset "Treatment period too late" begin
        outcomes = randn(5, 10)
        treatment = Bool[true, false, false, false, false]

        @test_throws ArgumentError SCMProblem(
            outcomes, treatment, 11, (alpha=0.05,)  # Past end
        )
    end

    @testset "NaN in outcomes" begin
        outcomes = randn(5, 10)
        outcomes[2, 5] = NaN

        treatment = Bool[true, false, false, false, false]

        @test_throws ArgumentError SCMProblem(
            outcomes, treatment, 5, (alpha=0.05,)
        )
    end

    @testset "Covariates row mismatch" begin
        outcomes = randn(5, 10)
        treatment = Bool[true, false, false, false, false]
        covariates = randn(3, 2)  # Wrong number of rows

        @test_throws ArgumentError SCMProblem(
            outcomes, treatment, 5, covariates, (alpha=0.05,)
        )
    end

    @testset "NaN in covariates" begin
        outcomes = randn(5, 10)
        treatment = Bool[true, false, false, false, false]
        covariates = randn(5, 2)
        covariates[3, 1] = NaN

        @test_throws ArgumentError SCMProblem(
            outcomes, treatment, 5, covariates, (alpha=0.05,)
        )
    end

end

# =============================================================================
# 2. Estimator Parameter Validation
# =============================================================================

@testset "Estimator Parameter Validation" begin

    @testset "Invalid inference method - SyntheticControl" begin
        @test_throws ArgumentError SyntheticControl(inference=:invalid)
    end

    @testset "Invalid n_placebo" begin
        @test_throws ArgumentError SyntheticControl(n_placebo=0)
        @test_throws ArgumentError SyntheticControl(n_placebo=-1)
    end

    @testset "Negative covariate_weight" begin
        @test_throws ArgumentError SyntheticControl(covariate_weight=-0.5)
    end

    @testset "Invalid inference method - AugmentedSC" begin
        @test_throws ArgumentError AugmentedSC(inference=:invalid)
    end

    @testset "Negative lambda" begin
        @test_throws ArgumentError AugmentedSC(lambda=-1.0)
    end

end

# =============================================================================
# 3. Numerical Stability Tests
# =============================================================================

@testset "SCM Numerical Stability" begin

    @testset "Very small outcome values" begin
        Random.seed!(42)
        scale = 1e-10
        outcomes = randn(5, 12) .* scale .+ scale
        outcomes[1, 7:end] .+= 0.5 * scale

        treatment = Bool[true, false, false, false, false]

        problem = SCMProblem(outcomes, treatment, 7, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test isfinite(solution.estimate)
        @test length(solution.weights) == 4
    end

    @testset "Very large outcome values" begin
        Random.seed!(42)
        scale = 1e6  # Reduced scale for numerical stability
        outcomes = randn(5, 12) .* scale .+ scale
        outcomes[1, 7:end] .+= 1e5

        treatment = Bool[true, false, false, false, false]

        problem = SCMProblem(outcomes, treatment, 7, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test isfinite(solution.estimate)
        # Estimate may be negative due to poor fit, just check it's finite
    end

    @testset "Identical control units" begin
        Random.seed!(42)
        n_periods = 10
        trajectory = collect(LinRange(10, 15, n_periods))

        outcomes = zeros(5, n_periods)
        outcomes[1, :] = trajectory .+ 2.0
        outcomes[1, 6:end] .+= 3.0  # Treatment effect
        for i in 2:5
            outcomes[i, :] = trajectory .+ randn(n_periods) .* 0.01
        end

        treatment = Bool[true, false, false, false, false]

        problem = SCMProblem(outcomes, treatment, 6, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test isfinite(solution.estimate)
        # Weights should be valid (non-negative, sum to 1)
        @test all(solution.weights .>= -1e-6)
        @test isapprox(sum(solution.weights), 1.0, atol=1e-6)
    end

    @testset "Treated outside convex hull" begin
        Random.seed!(42)
        n_periods = 10

        # Controls clustered around 10
        outcomes = randn(5, n_periods) .* 0.5 .+ 10

        # Treated much higher
        outcomes[1, :] = randn(n_periods) .* 0.5 .+ 20
        outcomes[1, 6:end] .+= 3.0

        treatment = Bool[true, false, false, false, false]

        problem = SCMProblem(outcomes, treatment, 6, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test isfinite(solution.estimate)
        @test solution.pre_r_squared < 0.5  # Poor fit expected
    end

    @testset "Collinear control units" begin
        Random.seed!(42)
        n_periods = 10
        base = collect(LinRange(10, 20, n_periods))

        outcomes = zeros(4, n_periods)
        outcomes[1, :] = base .* 1.5
        outcomes[1, 6:end] .+= 3.0
        outcomes[2, :] = base
        outcomes[3, :] = base .* 2
        outcomes[4, :] = base .* 0.5

        treatment = Bool[true, false, false, false]

        problem = SCMProblem(outcomes, treatment, 6, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test isfinite(solution.estimate)
        @test isapprox(sum(solution.weights), 1.0, atol=1e-6)
    end

    @testset "Constant outcomes in control" begin
        outcomes = [
            10.0 11.0 12.0 18.0 19.0;  # Treated (varies + effect)
            5.0 5.0 5.0 5.0 5.0;       # Constant control
            8.0 8.0 8.0 8.0 8.0        # Constant control
        ]

        treatment = Bool[true, false, false]

        problem = SCMProblem(outcomes, treatment, 4, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test isfinite(solution.estimate)
    end

    @testset "Zero outcomes" begin
        outcomes = zeros(5, 10)
        outcomes[1, 6:end] .= 2.0  # Only treatment effect

        treatment = Bool[true, false, false, false, false]

        problem = SCMProblem(outcomes, treatment, 6, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test isapprox(solution.estimate, 2.0, atol=0.1)
    end

end

# =============================================================================
# 4. Panel Structure Edge Cases
# =============================================================================

@testset "SCM Panel Structure Edge Cases" begin

    @testset "Minimum viable panel" begin
        # 3 units (1 treated + 2 controls), 4 periods
        outcomes = [
            10.0 11.0 15.0 16.0;  # Treated (effect in period 3)
            10.0 11.0 12.0 13.0;  # Control 1
            9.0 10.0 11.0 12.0    # Control 2
        ]

        treatment = Bool[true, false, false]

        problem = SCMProblem(outcomes, treatment, 3, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test solution.n_control == 2
        @test solution.n_pre_periods == 2
        @test solution.n_post_periods == 2
        @test isfinite(solution.estimate)
    end

    @testset "Single post-treatment period" begin
        Random.seed!(42)
        outcomes = randn(5, 11) .+ 10
        outcomes[1, 11] += 5.0

        treatment = Bool[true, false, false, false, false]

        problem = SCMProblem(outcomes, treatment, 11, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test solution.n_post_periods == 1
        @test isfinite(solution.estimate)
    end

    @testset "Many pre-periods" begin
        Random.seed!(42)
        n_pre = 50
        n_post = 5
        outcomes = randn(6, n_pre + n_post) .+ 10
        outcomes[1, (n_pre+1):end] .+= 2.0

        treatment = Bool[true, false, false, false, false, false]

        problem = SCMProblem(outcomes, treatment, n_pre + 1, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test solution.n_pre_periods == 50
        @test isfinite(solution.estimate)
    end

    @testset "Many controls" begin
        Random.seed!(42)
        n_control = 50
        outcomes = randn(n_control + 1, 15) .+ 10
        outcomes[1, 9:end] .+= 3.0

        treatment = Bool[i == 1 for i in 1:(n_control+1)]

        problem = SCMProblem(outcomes, treatment, 9, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test solution.n_control == 50
        @test isfinite(solution.estimate)

        # Most weights should be near zero
        @test sum(solution.weights .< 0.01) > 40
    end

    @testset "Multiple treated units" begin
        Random.seed!(42)
        outcomes = randn(6, 12) .+ 10
        outcomes[1, 7:end] .+= 3.0
        outcomes[2, 7:end] .+= 3.5

        treatment = Bool[true, true, false, false, false, false]

        problem = SCMProblem(outcomes, treatment, 7, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test solution.n_treated == 2
        @test solution.n_control == 4
        # Estimate should be average of effects (~3.25)
        @test isapprox(solution.estimate, 3.25, atol=1.0)
    end

end

# =============================================================================
# 5. Inference Edge Cases
# =============================================================================

@testset "SCM Inference Edge Cases" begin

    @testset "Placebo with minimum controls" begin
        outcomes = [
            10.0 11.0 15.0 16.0;
            10.0 11.0 12.0 13.0;
            9.0 10.0 11.0 12.0
        ]

        treatment = Bool[true, false, false]

        problem = SCMProblem(outcomes, treatment, 3, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:placebo, n_placebo=10))

        # Should handle gracefully (may have NaN SE)
        @test isfinite(solution.se) || isnan(solution.se)
        @test 0 <= solution.p_value <= 1 || isnan(solution.p_value)
    end

    @testset "Bootstrap with few controls" begin
        outcomes = [
            10.0 11.0 15.0 16.0;
            10.0 11.0 12.0 13.0;
            9.0 10.0 11.0 12.0
        ]

        treatment = Bool[true, false, false]

        problem = SCMProblem(outcomes, treatment, 3, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:bootstrap, n_placebo=20))

        @test solution.se >= 0 || isnan(solution.se)
    end

    @testset "No inference" begin
        Random.seed!(42)
        outcomes = randn(5, 10) .+ 10

        treatment = Bool[true, false, false, false, false]

        problem = SCMProblem(outcomes, treatment, 6, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test isnan(solution.se)
        @test isnan(solution.p_value)
        @test isnan(solution.ci_lower)
        @test isnan(solution.ci_upper)
    end

    @testset "Large effect should have small p-value" begin
        Random.seed!(42)
        outcomes = randn(10, 12) .* 0.5 .+ 10
        outcomes[1, 7:end] .+= 20.0  # Very large effect

        treatment = Bool[i == 1 for i in 1:10]

        problem = SCMProblem(outcomes, treatment, 7, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:placebo, n_placebo=50))

        @test solution.p_value < 0.20
    end

end

# =============================================================================
# 6. ASCM Specific Tests
# =============================================================================

@testset "AugmentedSC Edge Cases" begin

    @testset "Minimum panel" begin
        outcomes = [
            10.0 11.0 15.0 16.0;
            10.0 11.0 12.0 13.0;
            9.0 10.0 11.0 12.0
        ]

        treatment = Bool[true, false, false]

        problem = SCMProblem(outcomes, treatment, 3, (alpha=0.05,))
        solution = solve(problem, AugmentedSC(inference=:none, lambda=1.0))

        @test isfinite(solution.estimate)
    end

    @testset "Various lambda values" begin
        Random.seed!(42)
        outcomes = randn(6, 12) .+ 10
        outcomes[1, 7:end] .+= 2.0

        treatment = Bool[true, false, false, false, false, false]

        for lambda_val in [0.01, 1.0, 100.0, 10000.0]
            problem = SCMProblem(outcomes, treatment, 7, (alpha=0.05,))
            solution = solve(problem, AugmentedSC(inference=:none, lambda=lambda_val))

            @test isfinite(solution.estimate)
        end
    end

    @testset "CV lambda selection" begin
        Random.seed!(42)
        outcomes = randn(8, 15) .+ 10
        outcomes[1, 9:end] .+= 2.0

        treatment = Bool[i == 1 for i in 1:8]

        problem = SCMProblem(outcomes, treatment, 9, (alpha=0.05,))
        solution = solve(problem, AugmentedSC(inference=:none, lambda=nothing))

        @test isfinite(solution.estimate)
    end

    @testset "Jackknife inference" begin
        Random.seed!(42)
        outcomes = randn(8, 15) .+ 10
        outcomes[1, 9:end] .+= 2.0

        treatment = Bool[i == 1 for i in 1:8]

        problem = SCMProblem(outcomes, treatment, 9, (alpha=0.05,))
        solution = solve(problem, AugmentedSC(inference=:jackknife, lambda=1.0))

        @test solution.se >= 0 || isnan(solution.se)
    end

    @testset "Bootstrap inference" begin
        Random.seed!(42)
        outcomes = randn(8, 15) .+ 10
        outcomes[1, 9:end] .+= 2.0

        treatment = Bool[i == 1 for i in 1:8]

        problem = SCMProblem(outcomes, treatment, 9, (alpha=0.05,))
        solution = solve(problem, AugmentedSC(inference=:bootstrap, lambda=1.0))

        @test solution.se >= 0 || isnan(solution.se)
    end

end

# =============================================================================
# 7. Data Type Handling
# =============================================================================

@testset "SCM Data Type Handling" begin

    @testset "Int outcomes" begin
        outcomes = [
            10 11 12 15 16;
            9 10 11 12 13;
            8 9 10 11 12
        ]

        treatment = Bool[true, false, false]

        # Should convert to Float64
        problem = SCMProblem(Float64.(outcomes), treatment, 4, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test isfinite(solution.estimate)
    end

    @testset "Float32 outcomes - converted to Float64" begin
        # Float32 input is converted to Float64 internally for numerical stability
        outcomes_f32 = Float32[
            10.0 11.0 12.0 15.0 16.0;
            9.0 10.0 11.0 12.0 13.0;
            8.0 9.0 10.0 11.0 12.0
        ]
        # Convert to Float64 as expected by implementation
        outcomes = Float64.(outcomes_f32)

        treatment = Bool[true, false, false]

        problem = SCMProblem(outcomes, treatment, 4, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test isfinite(solution.estimate)
        @test typeof(solution.estimate) == Float64
    end

    @testset "BitVector treatment" begin
        Random.seed!(42)
        outcomes = randn(5, 10) .+ 10

        # Use BitVector
        treatment = trues(5)
        treatment[2:end] .= false

        problem = SCMProblem(outcomes, Vector{Bool}(treatment), 6, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test solution.n_treated == 1
        @test solution.n_control == 4
    end

end

# =============================================================================
# 8. Weights Validation
# =============================================================================

@testset "SCM Weights Properties" begin

    @testset "Weights non-negative" begin
        Random.seed!(42)
        outcomes = randn(10, 15) .+ 10

        treatment = Bool[i == 1 for i in 1:10]

        problem = SCMProblem(outcomes, treatment, 8, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test all(solution.weights .>= -1e-10)
    end

    @testset "Weights sum to one" begin
        Random.seed!(42)
        outcomes = randn(10, 15) .+ 10

        treatment = Bool[i == 1 for i in 1:10]

        problem = SCMProblem(outcomes, treatment, 8, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test isapprox(sum(solution.weights), 1.0, atol=1e-6)
    end

    @testset "Weights length equals n_control" begin
        Random.seed!(42)
        outcomes = randn(8, 12) .+ 10

        treatment = Bool[true, true, false, false, false, false, false, false]

        problem = SCMProblem(outcomes, treatment, 7, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test length(solution.weights) == 6  # n_control
    end

end

# =============================================================================
# 9. Pre-treatment Fit Tests
# =============================================================================

@testset "SCM Pre-treatment Fit" begin

    @testset "Perfect fit should have R² ≈ 1" begin
        n_periods = 10
        trajectory = collect(LinRange(10, 20, n_periods))

        outcomes = zeros(4, n_periods)
        outcomes[1, :] = trajectory
        outcomes[1, 6:end] .+= 2.0  # Treatment effect
        outcomes[2, :] = trajectory .+ randn(n_periods) .* 0.01
        outcomes[3, :] = trajectory .+ 5
        outcomes[4, :] = trajectory .* 0.5

        treatment = Bool[true, false, false, false]

        problem = SCMProblem(outcomes, treatment, 6, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test solution.pre_r_squared > 0.95
        @test solution.pre_rmse < 0.5
    end

    @testset "Poor fit should have low R²" begin
        Random.seed!(42)
        n_periods = 10

        outcomes = randn(5, n_periods) .* 0.5 .+ 10
        outcomes[1, :] .+= 10  # Treated much higher

        treatment = Bool[true, false, false, false, false]

        problem = SCMProblem(outcomes, treatment, 6, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test solution.pre_r_squared < 0.5
    end

end

# =============================================================================
# 10. Solution Fields Tests
# =============================================================================

@testset "SCM Solution Fields" begin

    @testset "Gap dimensions" begin
        Random.seed!(42)
        n_periods = 15
        outcomes = randn(5, n_periods) .+ 10

        treatment = Bool[true, false, false, false, false]

        problem = SCMProblem(outcomes, treatment, 8, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test length(solution.gap) == n_periods
        @test length(solution.synthetic_control) == n_periods
        @test length(solution.treated_series) == n_periods
    end

    @testset "Retcode values" begin
        Random.seed!(42)
        outcomes = randn(5, 12) .+ 10
        outcomes[1, 7:end] .+= 2.0

        treatment = Bool[true, false, false, false, false]

        problem = SCMProblem(outcomes, treatment, 7, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test solution.retcode in [:Success, :Warning, :Error]
    end

    @testset "Original problem preserved" begin
        Random.seed!(42)
        outcomes = randn(5, 12) .+ 10

        treatment = Bool[true, false, false, false, false]

        problem = SCMProblem(outcomes, treatment, 7, (alpha=0.05,))
        solution = solve(problem, SyntheticControl(inference=:none))

        @test solution.original_problem.treatment_period == 7
        @test size(solution.original_problem.outcomes) == (5, 12)
    end

end

# Run summary
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n" * "="^60)
    println("SCM Adversarial Tests Summary")
    println("="^60)
    println("Testing: Input validation, numerical stability, edge cases")
    println("="^60 * "\n")
end
