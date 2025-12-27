"""
    Score Functions for Score-Based Causal Discovery

Session 151: BIC, AIC, and local scores for GES algorithm.

Score-based methods search for the DAG that maximizes a penalized
likelihood score. Common choices:

- **BIC (Bayesian Information Criterion)**: log L - (k/2) log n
- **AIC (Akaike Information Criterion)**: log L - k
- **BGe (Bayesian Gaussian equivalent)**: Marginal likelihood

For Gaussian linear models:
    log L = -n/2 * [log(2π) + log(σ²) + 1]

where σ² is the residual variance from regressing X_i on its parents.

References
----------
- Chickering (2002). Optimal structure identification with greedy search.
- Schwarz (1978). Estimating the dimension of a model.
"""
module ScoreFunctions

using LinearAlgebra
using Statistics

export ScoreType, BIC, AIC, BIC_G
export LocalScore, GESResult
export compute_rss, local_score_bic, local_score_aic, local_score
export total_score, score_delta_add, score_delta_remove


"""
    ScoreType

Score function types for GES algorithm.

Values:
- `BIC`: Bayesian Information Criterion
- `AIC`: Akaike Information Criterion
- `BIC_G`: BIC with Gaussian assumption (same as BIC)
"""
@enum ScoreType begin
    BIC = 1
    AIC = 2
    BIC_G = 3
end


"""
    LocalScore

Score for a single node given its parents.

# Fields
- `node::Int`: Node index (1-based)
- `parents::Set{Int}`: Parent node indices
- `score::Float64`: Score value (higher is better)
- `n_params::Int`: Number of parameters
- `rss::Float64`: Residual sum of squares
"""
struct LocalScore
    node::Int
    parents::Set{Int}
    score::Float64
    n_params::Int
    rss::Float64
end


"""
    GESResult

Result from GES algorithm.

# Fields
- `cpdag::CPDAG`: Learned equivalence class
- `score::Float64`: Final score
- `n_forward_steps::Int`: Number of edges added in forward phase
- `n_backward_steps::Int`: Number of edges removed in backward phase
- `forward_scores::Vector{Float64}`: Score after each forward step
- `backward_scores::Vector{Float64}`: Score after each backward step
- `n_vars::Int`: Number of variables
- `n_samples::Int`: Number of samples
- `score_type::ScoreType`: Score function used
"""
struct GESResult
    cpdag::Any  # CPDAG type from types.jl
    score::Float64
    n_forward_steps::Int
    n_backward_steps::Int
    forward_scores::Vector{Float64}
    backward_scores::Vector{Float64}
    n_vars::Int
    n_samples::Int
    score_type::ScoreType
end


"""
    compute_rss(data, node, parents)

Compute residual sum of squares for node given parents.

# Arguments
- `data::Matrix{Float64}`: (n_samples, n_vars) data matrix
- `node::Int`: Target node index (1-based)
- `parents::Set{Int}`: Parent node indices

# Returns
- `Float64`: Residual sum of squares
"""
function compute_rss(data::Matrix{Float64}, node::Int, parents::Set{Int})::Float64
    n_samples = size(data, 1)
    y = data[:, node]

    if isempty(parents)
        # No parents: RSS = sum of squared deviations from mean
        return sum((y .- mean(y)).^2)
    end

    # Regress node on parents
    parent_list = collect(parents)
    X = data[:, parent_list]
    if ndims(X) == 1
        X = reshape(X, :, 1)
    end

    # Add intercept column
    X_with_intercept = hcat(ones(n_samples), X)

    # OLS via least squares (\ operator handles it)
    beta = X_with_intercept \ y

    # Compute residuals
    y_pred = X_with_intercept * beta
    rss = sum((y .- y_pred).^2)

    return rss
end


"""
    local_score_bic(data, node, parents; cache=nothing)

Compute local BIC score for a node given its parents.

BIC = log L - (k/2) * log(n)

For Gaussian linear model:
    log L = -n/2 * [log(2π) + log(σ²) + 1]

So: BIC = -n/2 * log(RSS/n) - (k/2) * log(n) + const

We drop constants and use: -n * log(RSS/n) - k * log(n)
(Higher is better)

# Arguments
- `data::Matrix{Float64}`: (n_samples, n_vars) data matrix
- `node::Int`: Target node index (1-based)
- `parents::Set{Int}`: Parent node indices
- `cache::Union{Dict,Nothing}`: Optional cache for computed scores

# Returns
- `LocalScore`: Local score result
"""
function local_score_bic(
    data::Matrix{Float64},
    node::Int,
    parents::Set{Int};
    cache::Union{Dict{Tuple{Int,Set{Int}},LocalScore},Nothing} = nothing
)::LocalScore
    n_samples = size(data, 1)

    # Check cache
    cache_key = (node, parents)
    if cache !== nothing && haskey(cache, cache_key)
        return cache[cache_key]
    end

    # Compute RSS
    rss = compute_rss(data, node, parents)

    # Number of parameters: intercept + parents
    n_params = 1 + length(parents)

    # BIC score (higher is better)
    # -n/2 * log(RSS/n) - k/2 * log(n)
    if rss <= 0
        rss = 1e-10  # Numerical stability
    end

    log_likelihood = -n_samples / 2 * log(rss / n_samples)
    penalty = n_params / 2 * log(n_samples)
    score = log_likelihood - penalty

    result = LocalScore(node, parents, score, n_params, rss)

    # Cache result
    if cache !== nothing
        cache[cache_key] = result
    end

    return result
end


"""
    local_score_aic(data, node, parents; cache=nothing)

Compute local AIC score for a node given its parents.

AIC = log L - k

(Higher is better)
"""
function local_score_aic(
    data::Matrix{Float64},
    node::Int,
    parents::Set{Int};
    cache::Union{Dict{Tuple{Int,Set{Int}},LocalScore},Nothing} = nothing
)::LocalScore
    n_samples = size(data, 1)

    cache_key = (node, parents)
    if cache !== nothing && haskey(cache, cache_key)
        return cache[cache_key]
    end

    rss = compute_rss(data, node, parents)
    n_params = 1 + length(parents)

    if rss <= 0
        rss = 1e-10
    end

    log_likelihood = -n_samples / 2 * log(rss / n_samples)
    penalty = Float64(n_params)
    score = log_likelihood - penalty

    result = LocalScore(node, parents, score, n_params, rss)

    if cache !== nothing
        cache[cache_key] = result
    end

    return result
end


"""
    local_score(data, node, parents, score_type; cache=nothing)

Compute local score for a node given its parents.

# Arguments
- `data::Matrix{Float64}`: (n_samples, n_vars) data matrix
- `node::Int`: Target node index (1-based)
- `parents::Set{Int}`: Parent node indices
- `score_type::ScoreType`: Score function to use
- `cache::Union{Dict,Nothing}`: Optional cache for computed scores

# Returns
- `LocalScore`: Local score result
"""
function local_score(
    data::Matrix{Float64},
    node::Int,
    parents::Set{Int},
    score_type::ScoreType = BIC;
    cache::Union{Dict{Tuple{Int,Set{Int}},LocalScore},Nothing} = nothing
)::LocalScore
    if score_type in (BIC, BIC_G)
        return local_score_bic(data, node, parents; cache=cache)
    elseif score_type == AIC
        return local_score_aic(data, node, parents; cache=cache)
    else
        error("Unknown score type: $score_type")
    end
end


"""
    total_score(data, adjacency, score_type; cache=nothing)

Compute total score for a DAG.

Total score = sum of local scores.

# Arguments
- `data::Matrix{Float64}`: (n_samples, n_vars) data matrix
- `adjacency::Matrix`: (n_vars, n_vars) adjacency matrix where adj[i,j]=1 means i→j
- `score_type::ScoreType`: Score function to use
- `cache::Union{Dict,Nothing}`: Optional cache for computed scores

# Returns
- `Float64`: Total score (higher is better)
"""
function total_score(
    data::Matrix{Float64},
    adjacency::AbstractMatrix,
    score_type::ScoreType = BIC;
    cache::Union{Dict{Tuple{Int,Set{Int}},LocalScore},Nothing} = nothing
)::Float64
    n_vars = size(data, 2)
    total = 0.0

    for node in 1:n_vars
        # Find parents of this node (1-based indexing)
        parents = Set(findall(x -> x == 1, adjacency[:, node]))
        ls = local_score(data, node, parents, score_type; cache=cache)
        total += ls.score
    end

    return total
end


"""
    score_delta_add(data, node, current_parents, new_parent, score_type; cache=nothing)

Compute score change from adding an edge.

Returns score(new) - score(old), positive if improvement.
"""
function score_delta_add(
    data::Matrix{Float64},
    node::Int,
    current_parents::Set{Int},
    new_parent::Int,
    score_type::ScoreType = BIC;
    cache::Union{Dict{Tuple{Int,Set{Int}},LocalScore},Nothing} = nothing
)::Float64
    old_score = local_score(data, node, current_parents, score_type; cache=cache)
    new_parents = union(current_parents, Set([new_parent]))
    new_score = local_score(data, node, new_parents, score_type; cache=cache)
    return new_score.score - old_score.score
end


"""
    score_delta_remove(data, node, current_parents, parent_to_remove, score_type; cache=nothing)

Compute score change from removing an edge.

Returns score(new) - score(old), positive if improvement.
"""
function score_delta_remove(
    data::Matrix{Float64},
    node::Int,
    current_parents::Set{Int},
    parent_to_remove::Int,
    score_type::ScoreType = BIC;
    cache::Union{Dict{Tuple{Int,Set{Int}},LocalScore},Nothing} = nothing
)::Float64
    old_score = local_score(data, node, current_parents, score_type; cache=cache)
    new_parents = setdiff(current_parents, Set([parent_to_remove]))
    new_score = local_score(data, node, new_parents, score_type; cache=cache)
    return new_score.score - old_score.score
end


end # module
