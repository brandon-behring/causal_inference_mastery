"""
Bunching estimation module.

Implements bunching estimation (Saez 2010) for detecting behavioral responses
at kinks in budget constraints.

# Components
- `BunchingProblem`: Problem specification (data, kink, rates)
- `SaezBunching`: Estimator configuration
- `BunchingSolution`: Full solution with inference
- `solve`: Main estimation function

# Example
```julia
using CausalEstimators.Bunching

# Generate data with bunching at kink
data = vcat(randn(800) .* 15 .+ 50, randn(200) .* 2 .+ 50)

# Create problem
problem = BunchingProblem(data, 50.0, 5.0; t1_rate=0.25, t2_rate=0.35)

# Solve
estimator = SaezBunching(n_bins=50, polynomial_order=7, n_bootstrap=200)
solution = solve(problem, estimator)

# Results
println("Excess mass: ", solution.excess_mass, " (SE: ", solution.excess_mass_se, ")")
println("Elasticity: ", solution.elasticity, " (SE: ", solution.elasticity_se, ")")
```

# References
- Saez (2010) - Original bunching methodology
- Chetty et al. (2011) - Integration constraint
- Kleven (2016) - Bunching estimation review
"""
module Bunching

# Dependencies
using Statistics
using LinearAlgebra
using Random

# Include component files
include("types.jl")
include("counterfactual.jl")
include("estimator.jl")

# Export types
export BunchingProblem, SaezBunching, CounterfactualResult, BunchingSolution

# Export counterfactual functions
export polynomial_counterfactual, estimate_counterfactual
export compute_excess_mass, compute_elasticity

# Export main solver
export solve

# Export confidence interval functions
export bunching_confidence_interval, elasticity_confidence_interval

end # module
