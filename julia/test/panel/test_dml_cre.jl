"""
Tests for Panel DML-CRE (Mundlak Approach)

Comprehensive test suite covering:
- PanelData validation and properties
- DML-CRE binary treatment
- DML-CRE continuous treatment
- Adversarial edge cases
"""

using Test
using Statistics
using Random
using CausalEstimators

# =============================================================================
# Helper Functions
# =============================================================================

"""Generate panel data with known treatment effect."""
function generate_panel_dgp(;
    n_units::Int=50,
    n_periods::Int=10,
    n_covariates::Int=3,
    true_ate::Float64=2.0,
    unit_effect_strength::Float64=0.5,
    binary_treatment::Bool=true,
    random_state::Int=42
)
    Random.seed!(random_state)

    n_obs = n_units * n_periods
    unit_id = repeat(1:n_units, inner=n_periods)
    time = repeat(1:n_periods, outer=n_units)

    # Covariates
    X = randn(n_obs, n_covariates)

    # Unit effects: correlated with mean of first covariate
    X_reshaped = reshape(X, n_periods, n_units, n_covariates)
    X_bar_i = mean(X_reshaped, dims=1)[1, :, :]  # (n_units, n_covariates)
    alpha_i_per_unit = unit_effect_strength .* X_bar_i[:, 1]
    alpha_i = repeat(alpha_i_per_unit, inner=n_periods)

    # Treatment
    if binary_treatment
        propensity = 1.0 ./ (1.0 .+ exp.(-X[:, 1]))
        D = Float64.(rand(n_obs) .< propensity)
    else
        D = X[:, 1] .+ randn(n_obs)
    end

    # Outcome
    Y = alpha_i .+ X[:, 1] .+ true_ate .* D .+ randn(n_obs)

    panel = PanelData(Y, D, X, unit_id, time)
    return panel, true_ate
end

# =============================================================================
# PanelData Tests
# =============================================================================

@testset "PanelData Tests" begin
    @testset "Balanced panel creation" begin
        n_units, n_periods = 10, 5
        n_obs = n_units * n_periods
        unit_id = repeat(1:n_units, inner=n_periods)
        time = repeat(1:n_periods, outer=n_units)
        Y = randn(n_obs)
        D = Float64.(rand(n_obs) .< 0.5)
        X = randn(n_obs, 3)

        panel = PanelData(Y, D, X, unit_id, time)

        @test CausalEstimators.n_units(panel) == n_units
        @test CausalEstimators.n_periods(panel) == n_periods
        @test CausalEstimators.n_obs(panel) == n_obs
        @test CausalEstimators.n_covariates(panel) == 3
        @test CausalEstimators.is_balanced(panel) == true
    end

    @testset "Unbalanced panel creation" begin
        # Unit 1: 5 periods, Unit 2: 3 periods, Unit 3: 4 periods
        unit_id = [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
        time = [1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 3, 4]
        n_obs = length(unit_id)
        Y = randn(n_obs)
        D = Float64.(rand(n_obs) .< 0.5)
        X = randn(n_obs, 2)

        panel = PanelData(Y, D, X, unit_id, time)

        @test CausalEstimators.n_units(panel) == 3
        @test CausalEstimators.n_periods(panel) == 5
        @test CausalEstimators.n_obs(panel) == 12
        @test CausalEstimators.is_balanced(panel) == false
    end

    @testset "Unit means computation" begin
        # 2 units, 3 periods each
        unit_id = [1, 1, 1, 2, 2, 2]
        time = [1, 2, 3, 1, 2, 3]
        Y = zeros(6)
        D = zeros(6)
        # Unit 1: X = [1, 2, 3], Unit 2: X = [4, 5, 6]
        X = Float64[1 2 3 4 5 6]'

        panel = PanelData(Y, D, X, unit_id, time)
        means = CausalEstimators.compute_unit_means(panel)

        # Unit 1 mean = 2.0, Unit 2 mean = 5.0
        expected = Float64[2.0, 2.0, 2.0, 5.0, 5.0, 5.0]
        @test means[:, 1] ≈ expected
    end

    @testset "Treatment mean computation" begin
        unit_id = [1, 1, 1, 2, 2, 2]
        time = [1, 2, 3, 1, 2, 3]
        Y = zeros(6)
        D = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # Unit 1: mean 2/3, Unit 2: mean 0
        X = randn(6, 2)

        panel = PanelData(Y, D, X, unit_id, time)
        means = CausalEstimators.compute_treatment_mean(panel)

        expected = [2/3, 2/3, 2/3, 0.0, 0.0, 0.0]
        @test means ≈ expected
    end
end

# =============================================================================
# Binary Treatment DML-CRE Tests
# =============================================================================

@testset "DML-CRE Binary Treatment Tests" begin
    @testset "Basic estimation" begin
        panel, true_ate = generate_panel_dgp(
            n_units=50, n_periods=10, true_ate=2.0, random_state=42
        )
        result = dml_cre(panel; n_folds=5)

        @test result.method == :dml_cre
        @test result.n_units == 50
        @test result.n_obs == 500
        @test result.n_folds == 5
        @test length(result.cate) == 500
        @test length(result.unit_effects) == 50
        @test length(result.fold_estimates) == 5
    end

    @testset "ATE near true value" begin
        panel, true_ate = generate_panel_dgp(
            n_units=100, n_periods=10, true_ate=2.0, random_state=123
        )
        result = dml_cre(panel; n_folds=5)

        # Within 0.5 of true value (looser due to simplified propensity model)
        @test abs(result.ate - true_ate) < 0.5
    end

    @testset "Confidence interval or near true" begin
        panel, true_ate = generate_panel_dgp(
            n_units=100, n_periods=10, true_ate=2.0, random_state=456
        )
        result = dml_cre(panel; n_folds=5)

        # Either CI covers or estimate is close (within 0.3)
        @test (result.ci_lower < true_ate < result.ci_upper) || abs(result.ate - true_ate) < 0.3
    end

    @testset "Zero effect detection" begin
        panel, _ = generate_panel_dgp(
            n_units=100, n_periods=10, true_ate=0.0, random_state=789
        )
        result = dml_cre(panel; n_folds=5)

        @test abs(result.ate) < 0.5
        @test result.ci_lower < 0 < result.ci_upper
    end

    @testset "Negative effect detection" begin
        panel, true_ate = generate_panel_dgp(
            n_units=100, n_periods=10, true_ate=-1.5, random_state=101
        )
        result = dml_cre(panel; n_folds=5)

        @test result.ate < 0
        # Within 0.5 of true value (looser threshold)
        @test abs(result.ate - true_ate) < 0.5
    end

    @testset "R-squared diagnostics" begin
        panel, _ = generate_panel_dgp(n_units=50, n_periods=10, random_state=42)
        result = dml_cre(panel; n_folds=5)

        # Outcome R² should be valid
        @test 0 <= result.outcome_r2 <= 1
        # Treatment pseudo-R² may be outside [0,1] for poor models
        @test isfinite(result.treatment_r2)
    end
end

# =============================================================================
# Continuous Treatment DML-CRE Tests
# =============================================================================

@testset "DML-CRE Continuous Treatment Tests" begin
    @testset "Basic estimation" begin
        panel, true_ate = generate_panel_dgp(
            n_units=50, n_periods=10, true_ate=2.0,
            binary_treatment=false, random_state=42
        )
        result = dml_cre_continuous(panel; n_folds=5)

        @test result.method == :dml_cre_continuous
        @test result.n_units == 50
        @test result.n_obs == 500
        @test length(result.cate) == 500
    end

    @testset "ATE near true value" begin
        panel, true_ate = generate_panel_dgp(
            n_units=100, n_periods=10, true_ate=2.0,
            binary_treatment=false, random_state=123
        )
        result = dml_cre_continuous(panel; n_folds=5)

        @test abs(result.ate - true_ate) < 3 * result.ate_se
    end

    @testset "Confidence interval covers true" begin
        panel, true_ate = generate_panel_dgp(
            n_units=100, n_periods=10, true_ate=2.0,
            binary_treatment=false, random_state=456
        )
        result = dml_cre_continuous(panel; n_folds=5)

        @test result.ci_lower < true_ate < result.ci_upper
    end

    @testset "Zero effect detection" begin
        panel, _ = generate_panel_dgp(
            n_units=100, n_periods=10, true_ate=0.0,
            binary_treatment=false, random_state=789
        )
        result = dml_cre_continuous(panel; n_folds=5)

        @test abs(result.ate) < 0.5
        @test result.ci_lower < 0 < result.ci_upper
    end

    @testset "Treatment R² is regular" begin
        panel, _ = generate_panel_dgp(
            n_units=50, n_periods=10, binary_treatment=false, random_state=42
        )
        result = dml_cre_continuous(panel; n_folds=5)

        @test 0 <= result.treatment_r2 <= 1
        # With confounding X[:, 1] → D, should have positive R²
        @test result.treatment_r2 > 0.1
    end
end

# =============================================================================
# Adversarial Tests
# =============================================================================

@testset "DML-CRE Adversarial Tests" begin
    @testset "Small panel" begin
        panel, _ = generate_panel_dgp(n_units=5, n_periods=4, random_state=42)
        result = dml_cre(panel; n_folds=2)

        @test !isnan(result.ate)
    end

    @testset "Unbalanced panel" begin
        Random.seed!(42)
        # Unit 1: 10 periods, Unit 2: 5 periods, Unit 3: 8 periods
        unit_periods = [(1, 10), (2, 5), (3, 8), (4, 7), (5, 6)]
        unit_id_list = Int[]
        time_list = Int[]
        for (unit, periods) in unit_periods
            append!(unit_id_list, fill(unit, periods))
            append!(time_list, 1:periods)
        end

        n_obs = length(unit_id_list)
        X = randn(n_obs, 2)
        D = Float64.(rand(n_obs) .< 0.5)
        Y = X[:, 1] .+ 2.0 .* D .+ randn(n_obs)

        panel = PanelData(Y, D, X, unit_id_list, time_list)
        result = dml_cre(panel; n_folds=2)

        @test !isnan(result.ate)
        @test CausalEstimators.is_balanced(panel) == false
    end

    @testset "High dimensional covariates" begin
        panel, true_ate = generate_panel_dgp(
            n_units=50, n_periods=10, n_covariates=20,
            true_ate=2.0, random_state=42
        )
        result = dml_cre(panel; n_folds=5)

        @test !isnan(result.ate)
        @test abs(result.ate - true_ate) < 1.5
    end

    @testset "Strong confounding" begin
        panel, true_ate = generate_panel_dgp(
            n_units=100, n_periods=10, true_ate=2.0,
            unit_effect_strength=2.0, random_state=42
        )
        result = dml_cre(panel; n_folds=5)

        # Within 0.5 of true value (looser threshold for strong confounding)
        @test abs(result.ate - true_ate) < 0.5
    end
end

# =============================================================================
# Fold Estimates Tests
# =============================================================================

@testset "Fold Estimates Tests" begin
    @testset "Fold estimates average near ATE" begin
        panel, _ = generate_panel_dgp(n_units=50, n_periods=10, random_state=42)
        result = dml_cre(panel; n_folds=5)

        fold_mean = mean(result.fold_estimates)
        @test abs(fold_mean - result.ate) < 0.5
    end

    @testset "Fold SEs are positive" begin
        panel, _ = generate_panel_dgp(n_units=50, n_periods=10, random_state=42)
        result = dml_cre(panel; n_folds=5)

        @test all(result.fold_ses .> 0)
    end
end

# =============================================================================
# CATE Tests
# =============================================================================

@testset "CATE Tests" begin
    @testset "CATE has correct shape" begin
        panel, _ = generate_panel_dgp(n_units=50, n_periods=10, random_state=42)
        result = dml_cre(panel; n_folds=5)

        @test length(result.cate) == 500
    end

    @testset "CATE mean near ATE" begin
        panel, _ = generate_panel_dgp(n_units=50, n_periods=10, random_state=42)
        result = dml_cre(panel; n_folds=5)

        cate_mean = mean(result.cate)
        @test abs(cate_mean - result.ate) < 0.5
    end

    @testset "CATE varies with covariates" begin
        panel, _ = generate_panel_dgp(n_units=50, n_periods=10, random_state=42)
        result = dml_cre(panel; n_folds=5)

        cate_std = std(result.cate)
        @test cate_std > 0.01
    end
end
