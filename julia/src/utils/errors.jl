"""
Custom error types for causal estimation.

Following Brandon's principle: NEVER FAIL SILENTLY.
All errors provide diagnostic information.
"""

"""
    EstimationError

Custom exception for estimation failures.

Provides diagnostic information about what went wrong during estimation.

# Fields
- `estimator::String`: Name of estimator that failed
- `message::String`: Error description
- `details::Dict{String,Any}`: Additional diagnostic info

# Examples

```julia
throw(EstimationError(
    "SimpleATE",
    "Singular covariance matrix",
    Dict("n_treated" => 2, "variance" => 0.0)
))
```
"""
struct EstimationError <: Exception
    estimator::String
    message::String
    details::Dict{String,Any}
end

function Base.showerror(io::IO, e::EstimationError)
    println(io, "ESTIMATION ERROR in $(e.estimator):")
    println(io, "  $(e.message)")
    if !isempty(e.details)
        println(io, "  Details:")
        for (k, v) in e.details
            println(io, "    $(k) = $(v)")
        end
    end
end
