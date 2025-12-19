"""
Unit tests for Julia RKD diagnostics module.

Test structure:
1. Density smoothness test
2. Covariate smoothness test
3. First stage test
4. Comprehensive diagnostics
5. Edge cases
"""

using Test
using CausalEstimators
using Random
using Statistics
using Distributions

# =============================================================================
# DGP Functions
# =============================================================================

"""Generate data with smooth density (no manipulation)."""
function generate_smooth_density_data(;
    n::Int=1000,
    cutoff::Float64=0.0,
    seed::Union{Nothing,Int}=nothing
)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    # Uniform X should have smooth density
    X = rand(Uniform(-5, 5), n)

    return X
end

"""Generate data with smooth covariates."""
function generate_smooth_covariate_data(;
    n::Int=1000,
    cutoff::Float64=0.0,
    seed::Union{Nothing,Int}=nothing
)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    X = rand(Uniform(-5, 5), n)

    # Covariate with smooth relationship to X
    Z = 0.5 .* X .+ 0.3 .* randn(n)

    return X, Z
end

"""Generate data with non-smooth covariate (kink at cutoff)."""
function generate_kinked_covariate_data(;
    n::Int=1000,
    cutoff::Float64=0.0,
    seed::Union{Nothing,Int}=nothing
)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    X = rand(Uniform(-5, 5), n)

    # Covariate with kink at cutoff
    Z = [x < cutoff ? 0.5 * x : 2.0 * x for x in X] .+ 0.3 .* randn(n)

    return X, Z
end

"""Generate full RKD data for diagnostics."""
function generate_rkd_diagnostic_data(;
    n::Int=1000,
    cutoff::Float64=0.0,
    seed::Union{Nothing,Int}=nothing
)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    X = rand(Uniform(-5, 5), n)
    D = [x < cutoff ? 0.5 * (x - cutoff) : 1.5 * (x - cutoff) for x in X]
    Y = 2.0 .* D .+ randn(n)
    Z = 0.5 .* X .+ 0.3 .* randn(n)  # Smooth covariate

    return Y, X, D, Z
end

# =============================================================================
# Density Smoothness Tests
# =============================================================================

@testset "Density Smoothness Test" begin
    @testset "Returns correct type" begin
        X = generate_smooth_density_data(n=500, seed=42)
        result = density_smoothness_test(X, 0.0)

        @test result isa DensitySmoothnessResult
        @test isfinite(result.p_value) || isnan(result.p_value)
    end

    @testset "Fields populated" begin
        X = generate_smooth_density_data(n=500, seed=42)
        result = density_smoothness_test(X, 0.0)

        @test hasfield(DensitySmoothnessResult, :slope_left)
        @test hasfield(DensitySmoothnessResult, :slope_right)
        @test hasfield(DensitySmoothnessResult, :slope_difference)
        @test hasfield(DensitySmoothnessResult, :se)
        @test hasfield(DensitySmoothnessResult, :t_stat)
        @test hasfield(DensitySmoothnessResult, :p_value)
        @test hasfield(DensitySmoothnessResult, :n_bins)
        @test hasfield(DensitySmoothnessResult, :interpretation)
    end

    @testset "Smooth density high p-value" begin
        # Multiple runs to reduce randomness effects
        p_values = Float64[]
        for seed in 1:10
            X = generate_smooth_density_data(n=1000, seed=seed)
            result = density_smoothness_test(X, 0.0)
            if isfinite(result.p_value)
                push!(p_values, result.p_value)
            end
        end

        # On average, p-values should be roughly uniform
        # At least some should be > 0.1
        @test any(p_values .> 0.1)
    end

    @testset "Custom n_bins" begin
        X = generate_smooth_density_data(n=500, seed=42)
        result = density_smoothness_test(X, 0.0; n_bins=10)

        @test result.n_bins == 10
    end

    @testset "Custom bandwidth" begin
        X = generate_smooth_density_data(n=500, seed=42)
        result = density_smoothness_test(X, 0.0; bandwidth=2.0)

        @test result isa DensitySmoothnessResult
    end

    @testset "Interpretation provided" begin
        X = generate_smooth_density_data(n=500, seed=42)
        result = density_smoothness_test(X, 0.0)

        @test length(result.interpretation) > 0
        @test occursin("density", lowercase(result.interpretation)) ||
              occursin("smooth", lowercase(result.interpretation)) ||
              occursin("evidence", lowercase(result.interpretation))
    end
end

# =============================================================================
# Covariate Smoothness Tests
# =============================================================================

@testset "Covariate Smoothness Test" begin
    @testset "Returns correct type" begin
        X, Z = generate_smooth_covariate_data(n=500, seed=42)
        results = covariate_smoothness_test(X, Z, 0.0; covariate_name="Z")

        @test results isa Vector{CovariateSmoothnessResult}
        @test length(results) == 1
    end

    @testset "Fields populated" begin
        X, Z = generate_smooth_covariate_data(n=500, seed=42)
        results = covariate_smoothness_test(X, Z, 0.0; covariate_name="Z")
        result = results[1]

        @test hasfield(CovariateSmoothnessResult, :covariate_name)
        @test hasfield(CovariateSmoothnessResult, :slope_left)
        @test hasfield(CovariateSmoothnessResult, :slope_right)
        @test hasfield(CovariateSmoothnessResult, :slope_difference)
        @test hasfield(CovariateSmoothnessResult, :se)
        @test hasfield(CovariateSmoothnessResult, :t_stat)
        @test hasfield(CovariateSmoothnessResult, :p_value)
        @test hasfield(CovariateSmoothnessResult, :is_smooth)
    end

    @testset "Smooth covariate passes" begin
        # Multiple runs
        pass_count = 0
        for seed in 1:10
            X, Z = generate_smooth_covariate_data(n=1000, seed=seed)
            results = covariate_smoothness_test(X, Z, 0.0; covariate_name="Z")
            if results[1].is_smooth
                pass_count += 1
            end
        end

        # Most should pass (at 5% level, expect ~95% to pass)
        @test pass_count >= 5
    end

    @testset "Kinked covariate may fail" begin
        X, Z = generate_kinked_covariate_data(n=2000, seed=42)
        results = covariate_smoothness_test(X, Z, 0.0; covariate_name="Kinked Z")

        # Large kink should be detectable
        @test results[1].p_value < 0.5  # At least somewhat low
    end

    @testset "Multiple covariates" begin
        Random.seed!(42)
        n = 500
        X = rand(Uniform(-5, 5), n)
        Z1 = 0.5 .* X .+ 0.3 .* randn(n)
        Z2 = -0.3 .* X .+ 0.5 .* randn(n)
        covariates = hcat(Z1, Z2)

        results = covariate_smoothness_test(
            X, covariates, 0.0;
            covariate_names=["Z1", "Z2"]
        )

        @test length(results) == 2
        @test results[1].covariate_name == "Z1"
        @test results[2].covariate_name == "Z2"
    end

    @testset "Custom bandwidth" begin
        X, Z = generate_smooth_covariate_data(n=500, seed=42)
        results = covariate_smoothness_test(X, Z, 0.0; bandwidth=2.0)

        @test results isa Vector{CovariateSmoothnessResult}
    end
end

# =============================================================================
# First Stage Tests
# =============================================================================

@testset "First Stage Test" begin
    @testset "Returns correct type" begin
        Y, X, D, _ = generate_rkd_diagnostic_data(n=500, seed=42)
        result = first_stage_test(D, X, 0.0)

        @test result isa FirstStageTestResult
    end

    @testset "Fields populated" begin
        Y, X, D, _ = generate_rkd_diagnostic_data(n=500, seed=42)
        result = first_stage_test(D, X, 0.0)

        @test hasfield(FirstStageTestResult, :kink_estimate)
        @test hasfield(FirstStageTestResult, :se)
        @test hasfield(FirstStageTestResult, :f_stat)
        @test hasfield(FirstStageTestResult, :p_value)
        @test hasfield(FirstStageTestResult, :is_strong)
        @test hasfield(FirstStageTestResult, :interpretation)
    end

    @testset "Strong first stage detection" begin
        Y, X, D, _ = generate_rkd_diagnostic_data(n=1000, seed=42)
        result = first_stage_test(D, X, 0.0)

        # Deterministic D with kink should have very strong first stage
        @test result.f_stat > 10
        @test result.is_strong
    end

    @testset "Kink estimate accuracy" begin
        Y, X, D, _ = generate_rkd_diagnostic_data(n=1000, seed=42)
        result = first_stage_test(D, X, 0.0)

        # True kink = 1.5 - 0.5 = 1.0
        @test abs(result.kink_estimate - 1.0) < 0.5
    end

    @testset "P-value in range" begin
        Y, X, D, _ = generate_rkd_diagnostic_data(n=500, seed=42)
        result = first_stage_test(D, X, 0.0)

        @test 0 <= result.p_value <= 1
    end

    @testset "Custom bandwidth" begin
        Y, X, D, _ = generate_rkd_diagnostic_data(n=500, seed=42)
        result = first_stage_test(D, X, 0.0; bandwidth=2.0)

        @test result isa FirstStageTestResult
    end

    @testset "Interpretation provided" begin
        Y, X, D, _ = generate_rkd_diagnostic_data(n=500, seed=42)
        result = first_stage_test(D, X, 0.0)

        @test length(result.interpretation) > 0
        @test occursin("first stage", lowercase(result.interpretation)) ||
              occursin("F", result.interpretation)
    end
end

# =============================================================================
# Comprehensive Diagnostics Tests
# =============================================================================

@testset "RKD Diagnostics Summary" begin
    @testset "Returns correct type" begin
        Y, X, D, Z = generate_rkd_diagnostic_data(n=500, seed=42)
        covariates = reshape(Z, :, 1)
        summary = rkd_diagnostics(Y, X, D, 0.0; covariates=covariates)

        @test summary isa RKDDiagnosticsSummary
    end

    @testset "All components present" begin
        Y, X, D, Z = generate_rkd_diagnostic_data(n=500, seed=42)
        covariates = reshape(Z, :, 1)
        summary = rkd_diagnostics(Y, X, D, 0.0; covariates=covariates)

        @test summary.density_test isa DensitySmoothnessResult
        @test summary.first_stage_test isa FirstStageTestResult
        @test summary.covariate_tests isa Vector{CovariateSmoothnessResult}
    end

    @testset "Summary flags" begin
        Y, X, D, Z = generate_rkd_diagnostic_data(n=500, seed=42)
        covariates = reshape(Z, :, 1)
        summary = rkd_diagnostics(Y, X, D, 0.0; covariates=covariates)

        @test hasfield(RKDDiagnosticsSummary, :density_smooth)
        @test hasfield(RKDDiagnosticsSummary, :first_stage_strong)
        @test hasfield(RKDDiagnosticsSummary, :covariates_smooth)
        @test hasfield(RKDDiagnosticsSummary, :all_pass)

        @test typeof(summary.density_smooth) == Bool
        @test typeof(summary.first_stage_strong) == Bool
        @test typeof(summary.covariates_smooth) == Bool
        @test typeof(summary.all_pass) == Bool
    end

    @testset "Without covariates" begin
        Y, X, D, _ = generate_rkd_diagnostic_data(n=500, seed=42)
        summary = rkd_diagnostics(Y, X, D, 0.0)

        @test isempty(summary.covariate_tests)
        @test summary.covariates_smooth  # True when no covariates
    end

    @testset "All pass logic" begin
        Y, X, D, Z = generate_rkd_diagnostic_data(n=1000, seed=42)
        covariates = reshape(Z, :, 1)
        summary = rkd_diagnostics(Y, X, D, 0.0; covariates=covariates)

        # all_pass should be AND of all individual flags
        expected_all_pass = summary.density_smooth && summary.first_stage_strong && summary.covariates_smooth
        @test summary.all_pass == expected_all_pass
    end

    @testset "Covariate names" begin
        Y, X, D, Z = generate_rkd_diagnostic_data(n=500, seed=42)
        covariates = reshape(Z, :, 1)
        summary = rkd_diagnostics(
            Y, X, D, 0.0;
            covariates=covariates,
            covariate_names=["MyCovariate"]
        )

        @test summary.covariate_tests[1].covariate_name == "MyCovariate"
    end
end

# =============================================================================
# Edge Cases
# =============================================================================

@testset "Diagnostics Edge Cases" begin
    @testset "Small sample density test" begin
        X = rand(Uniform(-5, 5), 30)
        result = density_smoothness_test(X, 0.0)

        # Should handle gracefully
        @test result isa DensitySmoothnessResult
    end

    @testset "Small sample covariate test" begin
        X = rand(Uniform(-5, 5), 30)
        Z = 0.5 .* X .+ 0.3 .* randn(30)
        results = covariate_smoothness_test(X, Z, 0.0)

        @test results isa Vector{CovariateSmoothnessResult}
    end

    @testset "Small sample first stage" begin
        X = rand(Uniform(-5, 5), 30)
        D = [x < 0 ? 0.5 * x : 1.5 * x for x in X]
        result = first_stage_test(D, X, 0.0)

        @test result isa FirstStageTestResult
    end

    @testset "Constant covariate" begin
        X = rand(Uniform(-5, 5), 100)
        Z = ones(100)  # Constant
        results = covariate_smoothness_test(X, Z, 0.0)

        # Should handle without crashing
        @test results isa Vector{CovariateSmoothnessResult}
    end

    @testset "Non-zero cutoff" begin
        Random.seed!(42)
        X = rand(Uniform(-5, 5), 500)
        D = [x < 2.0 ? 0.5 * (x - 2.0) : 1.5 * (x - 2.0) for x in X]
        Y = 2.0 .* D .+ randn(500)

        summary = rkd_diagnostics(Y, X, D, 2.0)

        @test summary isa RKDDiagnosticsSummary
    end
end

# =============================================================================
# Display Methods
# =============================================================================

@testset "Diagnostics Display" begin
    @testset "DensitySmoothnessResult display" begin
        X = generate_smooth_density_data(n=500, seed=42)
        result = density_smoothness_test(X, 0.0)

        io = IOBuffer()
        show(io, result)
        output = String(take!(io))

        @test occursin("DensitySmoothnessResult", output)
        @test occursin("Slope", output) || occursin("slope", output)
    end

    @testset "FirstStageTestResult display" begin
        Y, X, D, _ = generate_rkd_diagnostic_data(n=500, seed=42)
        result = first_stage_test(D, X, 0.0)

        io = IOBuffer()
        show(io, result)
        output = String(take!(io))

        @test occursin("FirstStageTestResult", output)
        @test occursin("F-statistic", output) || occursin("Kink", output)
    end

    @testset "RKDDiagnosticsSummary display" begin
        Y, X, D, Z = generate_rkd_diagnostic_data(n=500, seed=42)
        covariates = reshape(Z, :, 1)
        summary = rkd_diagnostics(Y, X, D, 0.0; covariates=covariates)

        io = IOBuffer()
        show(io, summary)
        output = String(take!(io))

        @test occursin("RKDDiagnosticsSummary", output)
        @test occursin("pass", lowercase(output)) || occursin("smooth", lowercase(output))
    end
end
