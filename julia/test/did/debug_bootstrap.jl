"""
Debug script for Callaway-Sant'Anna bootstrap failure.

Adds diagnostic logging to understand why 100% of bootstrap samples fail.
"""

using CausalEstimators
using Random

# Create test data (same as smoke test)
Random.seed!(123)
n_periods = 8
unit_id = repeat(0:29, inner=n_periods)
time = repeat(0:(n_periods-1), outer=30)
treatment_time = vcat(fill(3.0, 10), fill(5.0, 10), fill(Inf, 10))

treatment = zeros(Bool, length(unit_id))
for i in 1:length(unit_id)
    uid = unit_id[i]
    t = time[i]
    tt = treatment_time[uid + 1]
    treatment[i] = isfinite(tt) && (t >= tt)
end

outcomes = 10.0 .+ 5.0 * treatment .+ randn(length(unit_id))

problem = StaggeredDiDProblem(outcomes, treatment, time, unit_id, treatment_time, (alpha=0.05,))

println("=" ^ 70)
println("BOOTSTRAP DIAGNOSTIC")
println("=" ^ 70)
println()

println("Original Data:")
println("  - Total units: ", length(unique(unit_id)))
println("  - Treatment cohorts: ", sort(unique(treatment_time[isfinite.(treatment_time)])))
println("  - Never-treated: ", sum(isinf.(treatment_time)))
println("  - Time periods: ", sort(unique(time)))
println()

# Manually test one bootstrap resample
println("Testing single bootstrap resample...")
rng = Random.MersenneTwister(123)

# Call internal bootstrap function (access via problem type)
# This won't work directly, so let me just try to run the estimator with verbose error handling

estimator = CallawaySantAnna(
    aggregation = :simple,
    control_group = :nevertreated,
    n_bootstrap = 50,
    random_seed = 123
)

println("Attempting Callaway-Sant'Anna solve...")
try
    result = solve(problem, estimator)
    println("✓ Success! ATT = ", result.att)
catch e
    println("✗ Error: ", e)
    println()
    println("Stack trace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end
