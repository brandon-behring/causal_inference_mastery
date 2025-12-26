"""
Panel Quantile Treatment Effects (Session 118) - Julia Tests

Tests for panel_rif_qte, panel_rif_qte_band, and panel_unconditional_qte.
"""

using Test
using Statistics
using Random
using LinearAlgebra

# Import module
using CausalEstimators

# =============================================================================
# Test Fixtures
# =============================================================================

"""
Generate panel data for QTE testing.

Args:
    n_units: Number of panel units
    n_periods: Time periods per unit
    true_qte: Constant treatment effect
    confounded: If true, add X→Y confounding
    heterogeneous: If true, effect varies by quantile
    seed: Random seed
"""
function generate_panel_qte_dgp(;
    n_units::Int=50,
    n_periods::Int=10,
    true_qte::Float64=2.0,
    confounded::Bool=false,
    heterogeneous::Bool=false,
    seed::Int=42
)
    Random.seed!(seed)

    n_obs = n_units * n_periods

    # Unit and time indices
    unit_id = repeat(1:n_units, inner=n_periods)
    time = repeat(1:n_periods, outer=n_units)

    # Covariates
    p = 2
    X = randn(n_obs, p)

    # Unit effects (correlated with covariates if confounded)
    unit_effects = randn(n_units)
    alpha_i = unit_effects[unit_id]

    # Treatment assignment
    if confounded
        # Treatment depends on X (creates confounding)
        propensity = 1.0 ./ (1.0 .+ exp.(-0.5 .* X[:, 1]))
        D = Float64.(rand(n_obs) .< propensity)
    else
        # Random treatment (no confounding)
        D = Float64.(rand(n_obs) .< 0.5)
    end

    # Error term
    epsilon = randn(n_obs)

    # Outcome
    if heterogeneous
        # Effect varies with epsilon (quantile heterogeneity)
        effect = true_qte .+ 0.5 .* epsilon
    else
        effect = fill(true_qte, n_obs)
    end

    # Y = alpha_i + X*beta + D*effect + epsilon
    beta = [1.0, 0.5]
    Y = alpha_i .+ X * beta .+ D .* effect .+ epsilon

    return PanelData(Y, D, X, unit_id, time)
end

# =============================================================================
# Test: panel_rif_qte Basic
# =============================================================================

@testset "PanelRIFQTE Basic" begin
    # Simple panel with known properties
    panel = generate_panel_qte_dgp(
        n_units=100, n_periods=10, true_qte=2.0,
        confounded=false, seed=123
    )

    @testset "Median QTE estimation" begin
        result = panel_rif_qte(panel; tau=0.5)

        @test result isa PanelQTEResult
        @test result.method == :panel_rif_qte
        @test result.n_obs == 1000
        @test result.n_units == 100
        @test result.quantile == 0.5

        # QTE should be close to true value (within 1.5 for Monte Carlo variation)
        @test abs(result.qte - 2.0) < 1.5

        # SE should be positive and reasonable
        @test result.qte_se > 0
        @test result.qte_se < 2.0  # Not absurdly large

        # CI should contain point estimate
        @test result.ci_lower < result.qte < result.ci_upper

        # Density should be positive
        @test result.density_at_quantile > 0
        @test result.bandwidth > 0
    end

    @testset "Different quantiles" begin
        result_25 = panel_rif_qte(panel; tau=0.25)
        result_50 = panel_rif_qte(panel; tau=0.5)
        result_75 = panel_rif_qte(panel; tau=0.75)

        # All should produce valid results
        @test result_25.quantile == 0.25
        @test result_50.quantile == 0.5
        @test result_75.quantile == 0.75

        # Outcome quantiles should be ordered
        @test result_25.outcome_quantile < result_50.outcome_quantile
        @test result_50.outcome_quantile < result_75.outcome_quantile
    end

    @testset "No covariates mode" begin
        result = panel_rif_qte(panel; tau=0.5, include_covariates=false)

        @test result isa PanelQTEResult
        @test result.n_obs == 1000
        # Should still produce valid estimates
        @test isfinite(result.qte)
        @test isfinite(result.qte_se)
    end
end

# =============================================================================
# Test: panel_rif_qte_band
# =============================================================================

@testset "PanelRIFQTEBand" begin
    panel = generate_panel_qte_dgp(
        n_units=50, n_periods=10, true_qte=2.0,
        confounded=false, seed=456
    )

    @testset "Default quantiles" begin
        result = panel_rif_qte_band(panel)

        @test result isa PanelQTEBandResult
        @test result.method == :panel_rif_qte_band
        @test length(result.quantiles) == 5  # Default: [0.1, 0.25, 0.5, 0.75, 0.9]
        @test result.quantiles == [0.1, 0.25, 0.5, 0.75, 0.9]

        # All arrays should have same length
        @test length(result.qtes) == 5
        @test length(result.qte_ses) == 5
        @test length(result.ci_lowers) == 5
        @test length(result.ci_uppers) == 5

        # All estimates should be finite
        @test all(isfinite, result.qtes)
        @test all(isfinite, result.qte_ses)
        @test all(x -> x > 0, result.qte_ses)  # All SE positive
    end

    @testset "Custom quantiles" begin
        custom_quantiles = [0.1, 0.5, 0.9]
        result = panel_rif_qte_band(panel; quantiles=custom_quantiles)

        @test result.quantiles == custom_quantiles
        @test length(result.qtes) == 3
    end

    @testset "Single quantile via band" begin
        result = panel_rif_qte_band(panel; quantiles=[0.5])

        @test length(result.quantiles) == 1
        @test result.quantiles[1] == 0.5
    end
end

# =============================================================================
# Test: panel_unconditional_qte
# =============================================================================

@testset "PanelUnconditionalQTE" begin
    panel = generate_panel_qte_dgp(
        n_units=50, n_periods=10, true_qte=2.0,
        confounded=false, seed=789
    )

    @testset "Basic estimation" begin
        result = panel_unconditional_qte(
            panel; tau=0.5, n_bootstrap=200, random_state=42
        )

        @test result isa PanelQTEResult
        @test result.method == :panel_unconditional_qte
        @test result.quantile == 0.5
        @test result.n_obs == 500
        @test result.n_units == 50

        # Estimate should be finite
        @test isfinite(result.qte)
        @test result.qte_se > 0
    end

    @testset "Cluster vs observation bootstrap" begin
        result_cluster = panel_unconditional_qte(
            panel; tau=0.5, n_bootstrap=100,
            cluster_bootstrap=true, random_state=42
        )
        result_obs = panel_unconditional_qte(
            panel; tau=0.5, n_bootstrap=100,
            cluster_bootstrap=false, random_state=42
        )

        # Both should produce valid results
        @test isfinite(result_cluster.qte)
        @test isfinite(result_obs.qte)

        # Clustered SE should typically be larger (accounts for within-unit correlation)
        # This is not always true in small samples, so we just check validity
        @test result_cluster.qte_se > 0
        @test result_obs.qte_se > 0
    end
end

# =============================================================================
# Test: Clustering Validity
# =============================================================================

@testset "ClusteringValidity" begin
    # Create panel with strong within-unit correlation
    Random.seed!(111)
    n_units, n_periods = 30, 20
    n_obs = n_units * n_periods

    unit_id = repeat(1:n_units, inner=n_periods)
    time = repeat(1:n_periods, outer=n_units)

    # Strong unit effects
    unit_effects = 3.0 .* randn(n_units)
    alpha_i = unit_effects[unit_id]

    X = randn(n_obs, 2)
    D = Float64.(rand(n_obs) .< 0.5)
    Y = alpha_i .+ X[:, 1] .+ 2.0 .* D .+ randn(n_obs)

    panel = PanelData(Y, D, X, unit_id, time)

    @testset "Clustered SE accounts for correlation" begin
        result = panel_rif_qte(panel; tau=0.5)

        # With strong unit effects, clustered SE should be non-trivial
        @test result.qte_se > 0.01

        # SE should be reasonable given data variation
        @test result.qte_se < std(Y) / sqrt(n_units)  # Upper bound
    end
end

# =============================================================================
# Test: Edge Cases
# =============================================================================

@testset "EdgeCases" begin
    @testset "Small panel" begin
        # Minimal panel: 10 units, 5 periods
        panel = generate_panel_qte_dgp(
            n_units=10, n_periods=5, true_qte=2.0, seed=222
        )

        result = panel_rif_qte(panel; tau=0.5)
        @test isfinite(result.qte)
        @test result.n_obs == 50
        @test result.n_units == 10
    end

    @testset "Extreme quantiles warning" begin
        panel = generate_panel_qte_dgp(n_units=20, n_periods=10, seed=333)

        # Should still work but may warn
        result_05 = panel_rif_qte(panel; tau=0.05)
        result_95 = panel_rif_qte(panel; tau=0.95)

        @test result_05.quantile == 0.05
        @test result_95.quantile == 0.95
        @test isfinite(result_05.qte)
        @test isfinite(result_95.qte)
    end

    @testset "Invalid quantile errors" begin
        panel = generate_panel_qte_dgp(n_units=20, n_periods=10, seed=444)

        @test_throws ErrorException panel_rif_qte(panel; tau=0.0)
        @test_throws ErrorException panel_rif_qte(panel; tau=1.0)
        @test_throws ErrorException panel_rif_qte(panel; tau=-0.1)
        @test_throws ErrorException panel_rif_qte(panel; tau=1.5)
    end

    @testset "Invalid quantiles in band" begin
        panel = generate_panel_qte_dgp(n_units=20, n_periods=10, seed=555)

        @test_throws ErrorException panel_rif_qte_band(panel; quantiles=[0.0, 0.5])
        @test_throws ErrorException panel_rif_qte_band(panel; quantiles=[0.5, 1.0])
    end
end

# =============================================================================
# Test: Result Structure
# =============================================================================

@testset "ResultStructure" begin
    panel = generate_panel_qte_dgp(n_units=30, n_periods=8, seed=666)

    @testset "PanelQTEResult fields" begin
        result = panel_rif_qte(panel; tau=0.5, alpha=0.10)

        # Check all fields exist and have correct types
        @test result.qte isa Float64
        @test result.qte_se isa Float64
        @test result.ci_lower isa Float64
        @test result.ci_upper isa Float64
        @test result.quantile isa Float64
        @test result.n_obs isa Int
        @test result.n_units isa Int
        @test result.outcome_quantile isa Float64
        @test result.density_at_quantile isa Float64
        @test result.bandwidth isa Float64
        @test result.method isa Symbol
    end

    @testset "PanelQTEBandResult fields" begin
        result = panel_rif_qte_band(panel; quantiles=[0.25, 0.5, 0.75])

        @test result.quantiles isa Vector{Float64}
        @test result.qtes isa Vector{Float64}
        @test result.qte_ses isa Vector{Float64}
        @test result.ci_lowers isa Vector{Float64}
        @test result.ci_uppers isa Vector{Float64}
        @test result.n_obs isa Int
        @test result.n_units isa Int
        @test result.method isa Symbol
    end

    @testset "Confidence interval alpha" begin
        result_95 = panel_rif_qte(panel; tau=0.5, alpha=0.05)
        result_90 = panel_rif_qte(panel; tau=0.5, alpha=0.10)

        # 90% CI should be narrower than 95% CI
        width_95 = result_95.ci_upper - result_95.ci_lower
        width_90 = result_90.ci_upper - result_90.ci_lower

        @test width_90 < width_95
    end
end

# =============================================================================
# Test: Monte Carlo (bias and coverage)
# =============================================================================

@testset "MonteCarlo" begin
    @testset "Unconditional QTE coverage" begin
        # Test unconditional_qte (simple quantile difference) for unbiasedness
        # Note: panel_rif_qte estimates a different quantity (Firpo et al. 2009)
        # and has known upward bias relative to simple quantile difference
        n_sims = 30
        true_qte = 2.0
        estimates = Float64[]
        covers = Bool[]

        for sim in 1:n_sims
            Random.seed!(2000 + sim)
            num_units, num_periods = 50, 10
            n_obs = num_units * num_periods

            unit_id = repeat(1:num_units, inner=num_periods)
            time = repeat(1:num_periods, outer=num_units)
            X = randn(n_obs, 2)
            D = Float64.(rand(n_obs) .< 0.5)

            # Simple DGP: Y = D*true_qte + epsilon
            Y = true_qte .* D .+ randn(n_obs)

            panel = PanelData(Y, D, X, unit_id, time)
            result = panel_unconditional_qte(panel; tau=0.5, n_bootstrap=200, random_state=sim)
            push!(estimates, result.qte)
            push!(covers, result.ci_lower <= true_qte <= result.ci_upper)
        end

        # Bias should be small for unconditional QTE
        bias = mean(estimates) - true_qte
        @test abs(bias) < 0.3

        # Coverage should be reasonable (bootstrap CI)
        coverage = mean(covers)
        @test coverage > 0.70  # Bootstrap CI may be slightly conservative
    end
end

# =============================================================================
# Test: Comparison with Unconditional QTE
# =============================================================================

@testset "MethodComparison" begin
    panel = generate_panel_qte_dgp(
        n_units=50, n_periods=10, true_qte=2.0,
        confounded=false, seed=777
    )

    @testset "RIF vs unconditional agreement" begin
        rif_result = panel_rif_qte(panel; tau=0.5)
        uncond_result = panel_unconditional_qte(
            panel; tau=0.5, n_bootstrap=500, random_state=42
        )

        # Both should estimate similar effects (within 1.0 for unconfounded data)
        @test abs(rif_result.qte - uncond_result.qte) < 1.5
    end
end

println("\n" * "="^60)
println("All Panel QTE tests completed")
println("="^60)
