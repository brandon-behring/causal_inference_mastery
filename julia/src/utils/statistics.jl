"""
Statistical utility functions for causal estimation.

Helper functions used across multiple estimators.
"""

"""
    neyman_variance(y1::Vector{T}, y0::Vector{T}) where T<:Real

Compute Neyman heteroskedasticity-robust variance for ATE.

# Formula

```math
Var(\\hat{\\tau}) = \\frac{s_1^2}{n_1} + \\frac{s_0^2}{n_0}
```

where ``s_t^2`` is the sample variance for treatment group ``t``.

# Arguments
- `y1`: Outcomes for treated units
- `y0`: Outcomes for control units

# Returns
- `Float64`: Variance estimate

# References
- Neyman (1923): On the Application of Probability Theory to Agricultural Experiments
"""
function neyman_variance(y1::Vector{T}, y0::Vector{T}) where {T<:Real}
    n1 = length(y1)
    n0 = length(y0)

    if n1 == 0 || n0 == 0
        throw(ArgumentError("Cannot compute Neyman variance with empty groups"))
    end

    var1 = var(y1)
    var0 = var(y0)

    return var1 / n1 + var0 / n0
end

"""
    confidence_interval(estimate::T, se::T, alpha::Real) where T<:Real

Compute confidence interval using normal approximation.

# Arguments
- `estimate`: Point estimate
- `se`: Standard error
- `alpha`: Significance level (e.g., 0.05 for 95% CI)

# Returns
- `Tuple{T,T}`: (lower_bound, upper_bound)

# Examples

```julia
ci_lower, ci_upper = confidence_interval(5.0, 1.0, 0.05)
# Returns approximately (3.04, 6.96) for 95% CI
```
"""
function confidence_interval(estimate::T, se::T, alpha::Real) where {T<:Real}
    if alpha <= 0 || alpha >= 1
        throw(ArgumentError("alpha must be in (0, 1), got $(alpha)"))
    end

    z_critical = quantile(Normal(), 1 - alpha / 2)
    lower = estimate - z_critical * se
    upper = estimate + z_critical * se

    return (lower, upper)
end

"""
    robust_se_hc3(residuals::Vector{T}, X::Matrix{T}) where T<:Real

Compute HC3 heteroskedasticity-robust standard errors for regression.

HC3 provides better small-sample performance than HC0/HC1/HC2.

# Formula

```math
SE_{HC3} = \\sqrt{(X'X)^{-1} X' \\text{diag}(e_i^2 / (1-h_i)^2) X (X'X)^{-1}}
```

where ``h_i`` is the leverage of observation ``i``.

# Arguments
- `residuals`: Regression residuals (n-vector)
- `X`: Design matrix (n × p)

# Returns
- `Vector{T}`: Standard errors for each coefficient (p-vector)

# References
- MacKinnon & White (1985): Some Heteroskedasticity-Consistent Covariance Matrix Estimators
"""
function robust_se_hc3(residuals::Vector{T}, X::Matrix{T}) where {T<:Real}
    n, p = size(X)

    if length(residuals) != n
        throw(ArgumentError("Residuals length must match number of rows in X"))
    end

    # Compute leverage (diagonal of hat matrix H = X(X'X)^{-1}X')
    XtX_inv = inv(X' * X)
    leverage = [dot(X[i, :], XtX_inv * X[i, :]) for i = 1:n]

    # HC3 weights: e_i^2 / (1 - h_i)^2
    weights = residuals .^ 2 ./ (1 .- leverage) .^ 2

    # Variance-covariance matrix
    vcov = XtX_inv * (X' * Diagonal(weights) * X) * XtX_inv

    # Standard errors are sqrt of diagonal
    return sqrt.(diag(vcov))
end
