"""
Comprehensive performance benchmarking suite for CausalEstimators.jl

Benchmarks all 5 RCT estimators across multiple sample sizes with:
- Execution time measurement (BenchmarkTools)
- Type stability verification (@code_warntype)
- Memory allocation tracking
- Scaling analysis (log-log plots)
- Julia vs Python performance comparison

Usage:
    cd julia
    julia --project=. benchmark/run_benchmarks.jl
"""

using BenchmarkTools
using CausalEstimators
using Random
using Statistics
using Printf
using Dates
using PyCall
using InteractiveUtils  # For @code_warntype

# Import Python implementation for comparison
py"""
import sys
sys.path.insert(0, '/home/brandon_behring/Claude/causal_inference_mastery/src')
from causal_inference.rct.estimators import simple_ate
from causal_inference.rct.estimators_stratified import stratified_ate
from causal_inference.rct.estimators_regression import regression_adjusted_ate
from causal_inference.rct.estimators_permutation import permutation_test
from causal_inference.rct.estimators_ipw import ipw_ate
"""

# ============================================================================
# Data Generation Utilities
# ============================================================================

"""Generate balanced RCT data with given sample size."""
function generate_rct_data(n::Int; effect_size::Float64 = 0.5, seed::Int = 42)
    Random.seed!(seed)
    n_treated = div(n, 2)
    n_control = n - n_treated

    outcomes = vcat(
        randn(n_treated) .+ effect_size,  # Treated (shifted by effect)
        randn(n_control)                   # Control
    )
    treatment = vcat(fill(true, n_treated), fill(false, n_control))

    return outcomes, treatment
end

"""Generate stratified RCT data."""
function generate_stratified_data(n::Int; n_strata::Int = 5, seed::Int = 42)
    Random.seed!(seed)
    strata = repeat(1:n_strata, outer = div(n, n_strata))
    strata = vcat(strata, fill(n_strata, n - length(strata)))  # Fill remainder

    outcomes = randn(n)
    treatment = rand(Bool, n)

    return outcomes, treatment, strata
end

"""Generate regression RCT data with covariate."""
function generate_regression_data(n::Int; seed::Int = 42)
    Random.seed!(seed)
    covariate = randn(n)
    treatment = rand(Bool, n)
    outcomes = 0.5 .* treatment .+ 0.3 .* covariate .+ randn(n) .* 0.5

    return outcomes, treatment, reshape(covariate, n, 1)
end

"""Generate IPW data with varying propensity."""
function generate_ipw_data(n::Int; seed::Int = 42)
    Random.seed!(seed)
    propensity = rand(n) .* 0.6 .+ 0.2  # In (0.2, 0.8)
    treatment = Vector{Bool}(rand(n) .< propensity)  # Convert BitVector to Vector{Bool}
    outcomes = randn(n)

    return outcomes, treatment, reshape(propensity, n, 1)
end

# ============================================================================
# Type Stability Verification
# ============================================================================

"""Verify type stability of estimator using @code_warntype."""
function check_type_stability(
    estimator_name::String,
    problem::RCTProblem,
    estimator,
)::Bool
    # Use code_typed instead of @code_warntype for programmatic checking
    # code_typed returns the type-inferred lowered form
    result = @code_typed solve(problem, estimator)
    return_type = result[2]

    # Check if return type is concrete (not abstract, not Union, not Any)
    is_stable = isconcretetype(return_type)

    if !is_stable
        println("  ⚠️  Type instability detected in $estimator_name")
        println("     Return type: $return_type (not concrete)")
    end

    return is_stable
end

# ============================================================================
# Benchmarking Functions
# ============================================================================

"""Benchmark single estimator at given sample size."""
function benchmark_estimator(
    estimator_name::String,
    estimator,
    problem::RCTProblem,
    n::Int,
)::NamedTuple
    # Warmup
    solve(problem, estimator)

    # Benchmark
    b = @benchmark solve($problem, $estimator) samples = 100 evals = 1

    median_time_ms = median(b).time / 1e6  # Convert ns to ms
    min_time_ms = minimum(b).time / 1e6
    max_time_ms = maximum(b).time / 1e6
    allocs = b.allocs
    memory_kb = b.memory / 1024  # Convert bytes to KB

    return (
        estimator = estimator_name,
        n = n,
        median_time_ms = median_time_ms,
        min_time_ms = min_time_ms,
        max_time_ms = max_time_ms,
        allocs = allocs,
        memory_kb = memory_kb,
    )
end

"""Benchmark Python implementation for comparison."""
function benchmark_python(
    estimator_name::String,
    py_function,
    outcomes,
    treatment,
    extra_args...,
)::Float64
    # Warmup
    py_function(outcomes, Int.(treatment), extra_args...)

    # Benchmark (manual timing since BenchmarkTools doesn't work with PyCall)
    times = Float64[]
    for _ in 1:100
        t_start = time_ns()
        py_function(outcomes, Int.(treatment), extra_args...)
        t_end = time_ns()
        push!(times, (t_end - t_start) / 1e6)  # Convert to ms
    end

    return median(times)
end

# ============================================================================
# Main Benchmarking Suite
# ============================================================================

function run_comprehensive_benchmarks()
    println("=" ^ 80)
    println("CausalEstimators.jl Comprehensive Performance Benchmark")
    println("=" ^ 80)
    println("Julia Version: ", VERSION)
    println("Date: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("=" ^ 80)
    println()

    # Sample sizes to test
    sample_sizes = [100, 500, 1000, 5000, 10000]

    # Storage for results
    all_results = []
    type_stability_results = Dict{String,Bool}()

    println("Testing sample sizes: ", sample_sizes)
    println()

    # ========================================================================
    # 1. SimpleATE Benchmarks
    # ========================================================================
    println("─" ^ 80)
    println("1. SimpleATE Benchmarks")
    println("─" ^ 80)

    for n in sample_sizes
        outcomes, treatment = generate_rct_data(n)
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))

        # Julia benchmark
        result = benchmark_estimator("SimpleATE", SimpleATE(), problem, n)
        push!(all_results, result)

        # Python benchmark
        py_time = benchmark_python("SimpleATE", py"simple_ate", outcomes, treatment)
        speedup = py_time / result.median_time_ms

        @printf(
            "  n=%5d: Julia %.3f ms | Python %.3f ms | Speedup: %.2fx | Allocs: %d | Memory: %.1f KB\n",
            n,
            result.median_time_ms,
            py_time,
            speedup,
            result.allocs,
            result.memory_kb
        )
    end

    # Type stability check (do once at n=1000)
    outcomes, treatment = generate_rct_data(1000)
    problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
    type_stability_results["SimpleATE"] =
        check_type_stability("SimpleATE", problem, SimpleATE())
    println()

    # ========================================================================
    # 2. StratifiedATE Benchmarks
    # ========================================================================
    println("─" ^ 80)
    println("2. StratifiedATE Benchmarks")
    println("─" ^ 80)

    for n in sample_sizes
        outcomes, treatment, strata = generate_stratified_data(n)
        problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.05,))

        # Julia benchmark
        result = benchmark_estimator("StratifiedATE", StratifiedATE(), problem, n)
        push!(all_results, result)

        # Python benchmark
        py_time =
            benchmark_python("StratifiedATE", py"stratified_ate", outcomes, treatment, strata)
        speedup = py_time / result.median_time_ms

        @printf(
            "  n=%5d: Julia %.3f ms | Python %.3f ms | Speedup: %.2fx | Allocs: %d | Memory: %.1f KB\n",
            n,
            result.median_time_ms,
            py_time,
            speedup,
            result.allocs,
            result.memory_kb
        )
    end

    # Type stability check
    outcomes, treatment, strata = generate_stratified_data(1000)
    problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha = 0.05,))
    type_stability_results["StratifiedATE"] =
        check_type_stability("StratifiedATE", problem, StratifiedATE())
    println()

    # ========================================================================
    # 3. RegressionATE Benchmarks
    # ========================================================================
    println("─" ^ 80)
    println("3. RegressionATE Benchmarks")
    println("─" ^ 80)

    for n in sample_sizes
        outcomes, treatment, covariates = generate_regression_data(n)
        problem = RCTProblem(outcomes, treatment, covariates, nothing, (alpha = 0.05,))

        # Julia benchmark
        result = benchmark_estimator("RegressionATE", RegressionATE(), problem, n)
        push!(all_results, result)

        # Python benchmark
        py_time = benchmark_python(
            "RegressionATE",
            py"regression_adjusted_ate",
            outcomes,
            treatment,
            covariates,
        )
        speedup = py_time / result.median_time_ms

        @printf(
            "  n=%5d: Julia %.3f ms | Python %.3f ms | Speedup: %.2fx | Allocs: %d | Memory: %.1f KB\n",
            n,
            result.median_time_ms,
            py_time,
            speedup,
            result.allocs,
            result.memory_kb
        )
    end

    # Type stability check
    outcomes, treatment, covariates = generate_regression_data(1000)
    problem = RCTProblem(outcomes, treatment, covariates, nothing, (alpha = 0.05,))
    type_stability_results["RegressionATE"] =
        check_type_stability("RegressionATE", problem, RegressionATE())
    println()

    # ========================================================================
    # 4. PermutationTest Benchmarks (Monte Carlo only, exact is too slow)
    # ========================================================================
    println("─" ^ 80)
    println("4. PermutationTest Benchmarks (Monte Carlo, 1000 permutations)")
    println("─" ^ 80)

    for n in sample_sizes
        outcomes, treatment = generate_rct_data(n)
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))

        # Julia benchmark (Monte Carlo with 1000 permutations)
        estimator = PermutationTest(1000, 42)
        result = benchmark_estimator("PermutationTest", estimator, problem, n)
        push!(all_results, result)

        # Python benchmark (use keyword args to avoid signature mismatch)
        py_time_sum = 0.0
        for _ in 1:100
            t_start = time_ns()
            py"permutation_test"(
                outcomes,
                Int.(treatment);
                n_permutations = 1000,
                random_seed = 42,
            )
            t_end = time_ns()
            py_time_sum += (t_end - t_start) / 1e6
        end
        py_time = py_time_sum / 100
        speedup = py_time / result.median_time_ms

        @printf(
            "  n=%5d: Julia %.3f ms | Python %.3f ms | Speedup: %.2fx | Allocs: %d | Memory: %.1f KB\n",
            n,
            result.median_time_ms,
            py_time,
            speedup,
            result.allocs,
            result.memory_kb
        )
    end

    # Type stability check
    outcomes, treatment = generate_rct_data(1000)
    problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha = 0.05,))
    type_stability_results["PermutationTest"] =
        check_type_stability("PermutationTest", problem, PermutationTest(1000, 42))
    println()

    # ========================================================================
    # 5. IPWATE Benchmarks
    # ========================================================================
    println("─" ^ 80)
    println("5. IPWATE Benchmarks")
    println("─" ^ 80)

    for n in sample_sizes
        outcomes, treatment, propensity = generate_ipw_data(n)
        problem = RCTProblem(outcomes, treatment, propensity, nothing, (alpha = 0.05,))

        # Julia benchmark
        result = benchmark_estimator("IPWATE", IPWATE(), problem, n)
        push!(all_results, result)

        # Python benchmark
        py_time = benchmark_python(
            "IPWATE",
            py"ipw_ate",
            outcomes,
            treatment,
            vec(propensity),
        )
        speedup = py_time / result.median_time_ms

        @printf(
            "  n=%5d: Julia %.3f ms | Python %.3f ms | Speedup: %.2fx | Allocs: %d | Memory: %.1f KB\n",
            n,
            result.median_time_ms,
            py_time,
            speedup,
            result.allocs,
            result.memory_kb
        )
    end

    # Type stability check
    outcomes, treatment, propensity = generate_ipw_data(1000)
    problem = RCTProblem(outcomes, treatment, propensity, nothing, (alpha = 0.05,))
    type_stability_results["IPWATE"] =
        check_type_stability("IPWATE", problem, IPWATE())
    println()

    # ========================================================================
    # Summary
    # ========================================================================
    println("=" ^ 80)
    println("SUMMARY")
    println("=" ^ 80)
    println()

    println("Type Stability Results:")
    for (estimator, is_stable) in sort(collect(type_stability_results))
        status = is_stable ? "✅ STABLE" : "❌ UNSTABLE"
        println("  $estimator: $status")
    end
    println()

    println("Performance Summary (median times at n=10000):")
    for result in all_results
        if result.n == 10000
            @printf(
                "  %-20s: %8.3f ms | %6d allocs | %8.1f KB\n",
                result.estimator,
                result.median_time_ms,
                result.allocs,
                result.memory_kb
            )
        end
    end
    println()

    # ========================================================================
    # Save Results
    # ========================================================================
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    results_file = "benchmark/results/benchmark_results_$timestamp.txt"

    open(results_file, "w") do io
        println(io, "CausalEstimators.jl Benchmark Results")
        println(io, "=" ^ 80)
        println(io, "Date: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, "Julia Version: ", VERSION)
        println(io, "=" ^ 80)
        println(io)

        println(io, "Full Results Table:")
        println(io, "-" ^ 80)
        @printf(
            io,
            "%-20s | %6s | %10s | %8s | %10s\n",
            "Estimator",
            "n",
            "Time (ms)",
            "Allocs",
            "Memory (KB)"
        )
        println(io, "-" ^ 80)

        for result in all_results
            @printf(
                io,
                "%-20s | %6d | %10.3f | %8d | %10.1f\n",
                result.estimator,
                result.n,
                result.median_time_ms,
                result.allocs,
                result.memory_kb
            )
        end
        println(io, "-" ^ 80)
        println(io)

        println(io, "Type Stability:")
        for (estimator, is_stable) in sort(collect(type_stability_results))
            status = is_stable ? "✅ STABLE" : "❌ UNSTABLE"
            println(io, "  $estimator: $status")
        end
    end

    println("Results saved to: $results_file")
    println()
    println("=" ^ 80)
    println("Benchmark Complete!")
    println("=" ^ 80)
end

# ============================================================================
# Run Benchmarks
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    run_comprehensive_benchmarks()
end
