"""
Tests for RDD type definitions and constructors.

Phase 3.1: Foundation tests
"""

using Test
using CausalEstimators

@testset "RDD Types" begin
    @testset "RDDProblem - Valid Construction" begin
        # Simple valid RDD data
        n = 100
        x = randn(n)
        treatment = x .>= 0.0
        y = 2.0 .+ 3.0 .* x .+ 5.0 .* treatment .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))

        @test problem.outcomes == y
        @test problem.running_var == x
        @test problem.treatment == treatment
        @test problem.cutoff == 0.0
        @test isnothing(problem.covariates)
        @test problem.parameters.alpha == 0.05
    end

    @testset "RDDProblem - With Covariates" begin
        n = 100
        x = randn(n)
        treatment = x .>= 0.0
        y = randn(n)
        covariates = randn(n, 3)

        problem = RDDProblem(y, x, treatment, 0.0, covariates, (alpha=0.05,))

        @test size(problem.covariates) == (n, 3)
    end

    @testset "RDDProblem - Validation Errors" begin
        n = 100
        x = randn(n)
        treatment = x .>= 0.0
        y = randn(n)

        # Mismatched dimensions
        @test_throws ArgumentError RDDProblem(y[1:50], x, treatment, 0.0, nothing, (alpha=0.05,))
        @test_throws ArgumentError RDDProblem(y, x[1:50], treatment, 0.0, nothing, (alpha=0.05,))
        @test_throws ArgumentError RDDProblem(y, x, treatment[1:50], 0.0, nothing, (alpha=0.05,))

        # Cutoff out of range
        @test_throws ArgumentError RDDProblem(y, x, treatment, 100.0, nothing, (alpha=0.05,))
        @test_throws ArgumentError RDDProblem(y, x, treatment, -100.0, nothing, (alpha=0.05,))

        # No observations on one side
        x_positive = abs.(randn(n)) .+ 1.0  # All > 0
        treatment_positive = fill(true, n)
        @test_throws ArgumentError RDDProblem(y, x_positive, treatment_positive, 0.0, nothing, (alpha=0.05,))

        x_negative = -(abs.(randn(n)) .+ 1.0)  # All < 0
        treatment_negative = fill(false, n)
        @test_throws ArgumentError RDDProblem(y, x_negative, treatment_negative, 0.0, nothing, (alpha=0.05,))
    end

    @testset "RDDProblem - Fuzzy RDD Warning" begin
        # Fuzzy RDD (treatment not perfectly aligned with cutoff)
        n = 100
        x = randn(n)
        treatment = rand(Bool, n)  # Random treatment assignment
        y = randn(n)

        # Should warn about Fuzzy RDD
        @test_logs (:warn, r"Treatment assignment inconsistent") RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))
    end

    @testset "Kernel Functions" begin
        # Triangular kernel
        tri = TriangularKernel()
        @test kernel_function(tri, 0.0) == 1.0
        @test kernel_function(tri, 0.5) == 0.5
        @test kernel_function(tri, 1.0) == 0.0
        @test kernel_function(tri, -0.5) == 0.5
        @test kernel_function(tri, 1.5) == 0.0

        # Uniform kernel
        uni = UniformKernel()
        @test kernel_function(uni, 0.0) == 0.5
        @test kernel_function(uni, 0.5) == 0.5
        @test kernel_function(uni, 1.0) == 0.5
        @test kernel_function(uni, 1.5) == 0.0

        # Epanechnikov kernel
        epan = EpanechnikovKernel()
        @test kernel_function(epan, 0.0) == 0.75
        @test kernel_function(epan, 1.0) == 0.0
        @test kernel_function(epan, 1.5) == 0.0
        @test isapprox(kernel_function(epan, 0.5), 0.75 * 0.75, atol=1e-10)
    end

    @testset "McCraryTest Construction" begin
        # Test passes (p > 0.05)
        test_pass = McCraryTest(0.25, 0.05, 0.03)
        @test test_pass.p_value == 0.25
        @test test_pass.passes == true

        # Test fails (p < 0.05)
        test_fail = McCraryTest(0.01, 0.15, 0.04)
        @test test_fail.p_value == 0.01
        @test test_fail.passes == false
    end

    @testset "Bandwidth Selectors" begin
        ik = IKBandwidth()
        cct = CCTBandwidth()

        @test ik isa AbstractBandwidthSelector
        @test cct isa AbstractBandwidthSelector
    end

    @testset "SharpRDD Estimator" begin
        # Default construction
        estimator = SharpRDD()
        @test estimator.bandwidth_method isa CCTBandwidth
        @test estimator.kernel isa TriangularKernel
        @test estimator.run_density_test == true
        @test estimator.polynomial_order == 1

        # Custom construction
        estimator_custom = SharpRDD(
            bandwidth_method=IKBandwidth(),
            kernel=UniformKernel(),
            run_density_test=false,
            polynomial_order=1
        )
        @test estimator_custom.bandwidth_method isa IKBandwidth
        @test estimator_custom.kernel isa UniformKernel
        @test estimator_custom.run_density_test == false

        # Invalid polynomial order
        @test_throws ArgumentError SharpRDD(polynomial_order=0)
    end

    @testset "RDDSolution Construction" begin
        solution = RDDSolution(
            estimate=5.0,
            se=0.8,
            ci_lower=3.4,
            ci_upper=6.6,
            p_value=0.001,
            bandwidth=0.5,
            n_eff_left=50,
            n_eff_right=45
        )

        @test solution.estimate == 5.0
        @test solution.se == 0.8
        @test solution.ci_lower == 3.4
        @test solution.ci_upper == 6.6
        @test solution.p_value == 0.001
        @test solution.bandwidth == 0.5
        @test solution.n_eff_left == 50
        @test solution.n_eff_right == 45
        @test solution.retcode == :Success
        @test solution.bias_corrected == false
        @test isnothing(solution.density_test)

        # With McCrary test
        mccrary = McCraryTest(0.35, 0.02, 0.01)
        solution_with_mccrary = RDDSolution(
            estimate=5.0,
            se=0.8,
            ci_lower=3.4,
            ci_upper=6.6,
            p_value=0.001,
            bandwidth=0.5,
            n_eff_left=50,
            n_eff_right=45,
            density_test=mccrary,
            bias_corrected=true
        )

        @test !isnothing(solution_with_mccrary.density_test)
        @test solution_with_mccrary.density_test.passes == true
        @test solution_with_mccrary.bias_corrected == true
    end
end
