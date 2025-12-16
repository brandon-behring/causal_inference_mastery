"""
Golden reference validation against Python results.

These tests validate Julia implementation against captured Python results
(from python_golden_results.json). This ensures cross-language agreement
and prevents regressions.

Tolerance: rtol < 1e-10 (match Python to 10 decimal places)
"""

using Test
using JSON3
using CausalEstimators

# Load golden results
golden_path = joinpath(@__DIR__, "../golden_results/python_golden_results.json")
golden = JSON3.read(read(golden_path, String))

@testset "Golden Reference: SimpleATE" begin
    # Note: Julia uses t-distribution for CIs (Satterthwaite df), golden reference
    # was generated with normal approximation. CI tolerance is relaxed to 1e-2
    # while estimate/SE remain at 1e-10 to validate core calculation.

    @testset "Balanced RCT" begin
        data = golden.balanced_rct.data
        expected = golden.balanced_rct.simple_ate

        problem = RCTProblem(
            Float64.(data.outcomes),
            Vector{Bool}(data.treatment),
            nothing,
            nothing,
            (alpha = 0.05,),
        )

        solution = solve(problem, SimpleATE())

        @test solution.estimate ≈ expected.estimate rtol = 1e-10
        @test solution.se ≈ expected.se rtol = 1e-10
        @test solution.ci_lower ≈ expected.ci_lower rtol = 1e-2  # t vs z distribution
        @test solution.ci_upper ≈ expected.ci_upper rtol = 1e-2
        @test solution.n_treated == expected.n_treated
        @test solution.n_control == expected.n_control
    end

    @testset "Stratified RCT (simple_ate result)" begin
        data = golden.stratified_rct.data
        expected = golden.stratified_rct.simple_ate

        problem = RCTProblem(
            Float64.(data.outcomes),
            Vector{Bool}(data.treatment),
            nothing,
            Int.(data.strata),  # Provided but not used by SimpleATE
            (alpha = 0.05,),
        )

        solution = solve(problem, SimpleATE())

        @test solution.estimate ≈ expected.estimate rtol = 1e-10
        @test solution.se ≈ expected.se rtol = 1e-10
        @test solution.ci_lower ≈ expected.ci_lower rtol = 2e-2  # t vs z, small sample
        @test solution.ci_upper ≈ expected.ci_upper rtol = 2e-2
    end

    @testset "Regression RCT (simple_ate result)" begin
        data = golden.regression_rct.data
        expected = golden.regression_rct.simple_ate

        problem = RCTProblem(
            Float64.(data.outcomes),
            Vector{Bool}(data.treatment),
            nothing,  # Covariates provided but not used by SimpleATE
            nothing,
            (alpha = 0.05,),
        )

        solution = solve(problem, SimpleATE())

        @test solution.estimate ≈ expected.estimate rtol = 1e-10
        @test solution.se ≈ expected.se rtol = 1e-10
    end

    @testset "Small sample (permutation test data)" begin
        data = golden.permutation_small.data
        expected = golden.permutation_small.simple_ate

        problem = RCTProblem(
            Float64.(data.outcomes),
            Vector{Bool}(data.treatment),
            nothing,
            nothing,
            (alpha = 0.05,),
        )

        solution = solve(problem, SimpleATE())

        @test solution.estimate ≈ expected.estimate rtol = 1e-10
        @test solution.se ≈ expected.se rtol = 1e-10
    end

    @testset "IPW varying propensity (simple_ate result)" begin
        data = golden.ipw_varying.data
        expected = golden.ipw_varying.simple_ate

        problem = RCTProblem(
            Float64.(data.outcomes),
            Vector{Bool}(data.treatment),
            nothing,
            nothing,
            (alpha = 0.05,),
        )

        solution = solve(problem, SimpleATE())

        @test solution.estimate ≈ expected.estimate rtol = 1e-10
        @test solution.se ≈ expected.se rtol = 1e-10
    end

    @testset "Large sample" begin
        data = golden.large_sample.data
        expected = golden.large_sample.simple_ate

        problem = RCTProblem(
            Float64.(data.outcomes),
            Vector{Bool}(data.treatment),
            nothing,
            nothing,
            (alpha = 0.05,),
        )

        solution = solve(problem, SimpleATE())

        @test solution.estimate ≈ expected.estimate rtol = 1e-10
        @test solution.se ≈ expected.se rtol = 1e-10
        @test solution.ci_lower ≈ expected.ci_lower rtol = 1e-2  # t vs z distribution
        @test solution.ci_upper ≈ expected.ci_upper rtol = 1e-2
    end
end

@testset "Golden Reference: StratifiedATE" begin
    @testset "Stratified RCT (stratified_ate result)" begin
        data = golden.stratified_rct.data
        expected = golden.stratified_rct.stratified_ate

        problem = RCTProblem(
            Float64.(data.outcomes),
            Vector{Bool}(data.treatment),
            nothing,
            Int.(data.strata),
            (alpha = 0.05,),
        )

        solution = solve(problem, StratifiedATE())

        @test solution.estimate ≈ expected.estimate rtol = 1e-10
        @test solution.se ≈ expected.se rtol = 1e-10
        @test solution.ci_lower ≈ expected.ci_lower rtol = 1e-10
        @test solution.ci_upper ≈ expected.ci_upper rtol = 1e-10
        @test solution.n_treated == expected.n_treated
        @test solution.n_control == expected.n_control
    end
end

@testset "Golden Reference: RegressionATE" begin
    @testset "Regression RCT (regression_adjusted_ate result)" begin
        data = golden.regression_rct.data
        expected = golden.regression_rct.regression_adjusted_ate

        # Covariate is provided as a 1D array in JSON (single covariate)
        covariate = Float64.(data.covariate)
        covariate_matrix = reshape(covariate, length(covariate), 1)

        problem = RCTProblem(
            Float64.(data.outcomes),
            Vector{Bool}(data.treatment),
            covariate_matrix,
            nothing,
            (alpha = 0.05,),
        )

        solution = solve(problem, RegressionATE())

        @test solution.estimate ≈ expected.estimate rtol = 1e-10
        @test solution.se ≈ expected.se rtol = 1e-10
        @test solution.ci_lower ≈ expected.ci_lower rtol = 1e-10
        @test solution.ci_upper ≈ expected.ci_upper rtol = 1e-10
        @test solution.n_treated == expected.n_treated
        @test solution.n_control == expected.n_control
    end
end

@testset "Golden Reference: PermutationTest" begin
    @testset "Exact test" begin
        data = golden.permutation_small.data
        expected = golden.permutation_small.permutation_test_exact

        problem = RCTProblem(
            Float64.(data.outcomes),
            Vector{Bool}(data.treatment),
            nothing,
            nothing,
            (alpha = 0.05,),
        )

        solution = solve(problem, PermutationTest())

        @test solution.p_value ≈ expected.p_value rtol = 1e-10
        @test solution.observed_statistic ≈ expected.observed_statistic rtol = 1e-10
        @test solution.n_permutations == expected.n_permutations
        @test solution.alternative == expected.alternative
    end

    @testset "Monte Carlo test" begin
        data = golden.permutation_small.data
        expected = golden.permutation_small.permutation_test_monte_carlo

        problem = RCTProblem(
            Float64.(data.outcomes),
            Vector{Bool}(data.treatment),
            nothing,
            nothing,
            (alpha = 0.05,),
        )

        solution = solve(problem, PermutationTest(1000, 42))

        # Monte Carlo p-value may differ due to different RNGs in Julia vs Python
        # (same seed doesn't guarantee same random sequence across languages)
        # Test that p-value is reasonably close (within 0.05 is acceptable)
        @test abs(solution.p_value - expected.p_value) < 0.05
        @test solution.observed_statistic ≈ expected.observed_statistic rtol = 1e-10
        @test solution.n_permutations == expected.n_permutations
        @test solution.alternative == expected.alternative
    end
end

@testset "Golden Reference: IPWATE" begin
    @testset "Varying propensity RCT" begin
        data = golden.ipw_varying.data
        expected = golden.ipw_varying.ipw_ate

        problem = RCTProblem(
            Float64.(data.outcomes),
            Vector{Bool}(data.treatment),
            hcat(Float64.(data.propensity)),  # Propensity in first column
            nothing,
            (alpha = 0.05,),
        )

        solution = solve(problem, IPWATE())

        @test solution.estimate ≈ expected.estimate rtol = 1e-10
        @test solution.se ≈ expected.se rtol = 1e-10
        @test solution.ci_lower ≈ expected.ci_lower rtol = 1e-10
        @test solution.ci_upper ≈ expected.ci_upper rtol = 1e-10
        @test solution.n_treated == expected.n_treated
        @test solution.n_control == expected.n_control
    end
end
