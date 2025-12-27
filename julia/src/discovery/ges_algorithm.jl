"""
    GES Algorithm (Greedy Equivalence Search)

Session 151: Score-based causal discovery.

GES (Chickering, 2002) searches for the DAG that maximizes a
decomposable score (e.g., BIC). It proceeds in two phases:

1. **Forward Phase**: Start with empty graph, greedily add edges
   that improve the score until no improvement possible.

2. **Backward Phase**: Greedily remove edges that improve the score
   until no improvement possible.

The output is a CPDAG representing the Markov equivalence class.

References
----------
- Chickering (2002). Optimal structure identification with greedy search.
- Meek (1997). Graphical models: Selecting causal and statistical models.
"""
module GESAlgorithm

using Combinatorics
using ..DiscoveryTypes
using ..ScoreFunctions

export ges_algorithm, ges_forward, ges_backward
export get_neighbors, get_parents, get_children, is_clique
export is_valid_insert, is_valid_delete
export apply_insert!, apply_delete!


# =============================================================================
# Graph Helper Functions
# =============================================================================

"""
    get_neighbors(adjacency, node)

Get neighbors (connected by undirected edge) of a node.
Undirected edge: adj[i,j] == 1 && adj[j,i] == 1
"""
function get_neighbors(adjacency::AbstractMatrix, node::Int)::Set{Int}
    n = size(adjacency, 1)
    neighbors = Set{Int}()
    for j in 1:n
        if j != node
            # Undirected edge: both directions present
            if adjacency[node, j] == 1 && adjacency[j, node] == 1
                push!(neighbors, j)
            end
        end
    end
    return neighbors
end


"""
    get_parents(adjacency, node)

Get parents (directed edges into node) of a node.
Directed edge j -> node: adj[j, node] = 1, adj[node, j] = 0
"""
function get_parents(adjacency::AbstractMatrix, node::Int)::Set{Int}
    n = size(adjacency, 1)
    parents = Set{Int}()
    for j in 1:n
        if j != node
            if adjacency[j, node] == 1 && adjacency[node, j] == 0
                push!(parents, j)
            end
        end
    end
    return parents
end


"""
    get_children(adjacency, node)

Get children (directed edges out of node) of a node.
Directed edge node -> j: adj[node, j] = 1, adj[j, node] = 0
"""
function get_children(adjacency::AbstractMatrix, node::Int)::Set{Int}
    n = size(adjacency, 1)
    children = Set{Int}()
    for j in 1:n
        if j != node
            if adjacency[node, j] == 1 && adjacency[j, node] == 0
                push!(children, j)
            end
        end
    end
    return children
end


"""
    is_clique(adjacency, nodes)

Check if a set of nodes forms a clique (all connected).
"""
function is_clique(adjacency::AbstractMatrix, nodes::Set{Int})::Bool
    node_list = collect(nodes)
    for i in 1:length(node_list)
        for j in i+1:length(node_list)
            u, v = node_list[i], node_list[j]
            # Must be connected (either direction or undirected)
            if adjacency[u, v] == 0 && adjacency[v, u] == 0
                return false
            end
        end
    end
    return true
end


"""
    has_path(adjacency, from, to)

Check if there is a directed path from `from` to `to` using BFS.
Used for cycle detection.
"""
function has_path(adjacency::AbstractMatrix, from::Int, to::Int)::Bool
    n = size(adjacency, 1)
    visited = Set{Int}()
    queue = [from]

    while !isempty(queue)
        node = popfirst!(queue)
        if node in visited
            continue
        end
        push!(visited, node)

        if node == to
            return true
        end

        # Follow directed edges and undirected edges (outgoing)
        for j in 1:n
            if adjacency[node, j] == 1 && j ∉ visited
                push!(queue, j)
            end
        end
    end

    return false
end


# =============================================================================
# Insert/Delete Validation
# =============================================================================

"""
    is_valid_insert(adjacency, x, y, T)

Check if insert(x, y, T) is valid.

Insert adds edge x -> y and orients all edges from T to y.

Valid if:
1. x and y are not adjacent
2. T ⊆ neighbors(y)
3. Na(y) ∪ {x} forms a clique (Na = neighbors + parents)
4. No new cycles (x not reachable from y)
"""
function is_valid_insert(
    adjacency::AbstractMatrix,
    x::Int,
    y::Int,
    T::Set{Int}
)::Bool
    # 1. x and y not adjacent
    if adjacency[x, y] == 1 || adjacency[y, x] == 1
        return false
    end

    # 2. T subset of neighbors(y)
    neighbors_y = get_neighbors(adjacency, y)
    if !issubset(T, neighbors_y)
        return false
    end

    # 3. Na(y) ∪ {x} is clique
    Na_y = union(neighbors_y, get_parents(adjacency, y))
    clique_check = union(Na_y, Set([x]))
    if !is_clique(adjacency, clique_check)
        return false
    end

    # 4. No cycle: x not reachable from y
    if has_path(adjacency, y, x)
        return false
    end

    return true
end


"""
    is_valid_delete(adjacency, x, y, H)

Check if delete(x, y, H) is valid.

Delete removes edge x - y and orients edges from Na(y) \\ H to y.

Valid if:
1. x and y are adjacent
2. H ⊆ neighbors(y)
3. Na(y) \\ H is a clique
"""
function is_valid_delete(
    adjacency::AbstractMatrix,
    x::Int,
    y::Int,
    H::Set{Int}
)::Bool
    # 1. x and y adjacent
    if adjacency[x, y] == 0 && adjacency[y, x] == 0
        return false
    end

    # 2. H subset of neighbors(y)
    neighbors_y = get_neighbors(adjacency, y)
    if !issubset(H, neighbors_y)
        return false
    end

    # 3. Na(y) \ H is clique
    Na_y = union(neighbors_y, get_parents(adjacency, y))
    remaining = setdiff(Na_y, H, Set([x]))
    if !is_clique(adjacency, remaining)
        return false
    end

    return true
end


# =============================================================================
# Apply Operations
# =============================================================================

"""
    apply_insert!(adjacency, x, y, T)

Apply insert(x, y, T) operation in-place.

1. Add edge x -> y
2. For each t in T, orient t - y as t -> y

Returns modified adjacency.
"""
function apply_insert!(
    adjacency::AbstractMatrix,
    x::Int,
    y::Int,
    T::Set{Int}
)
    adjacency[x, y] = 1  # x -> y
    adjacency[y, x] = 0

    for t in T
        adjacency[t, y] = 1  # t -> y
        adjacency[y, t] = 0
    end

    return adjacency
end


"""
    apply_delete!(adjacency, x, y, H)

Apply delete(x, y, H) operation in-place.

1. Remove edge x - y
2. For each h in H, orient edges as h -> y

Returns modified adjacency.
"""
function apply_delete!(
    adjacency::AbstractMatrix,
    x::Int,
    y::Int,
    H::Set{Int}
)
    adjacency[x, y] = 0
    adjacency[y, x] = 0

    for h in H
        adjacency[h, y] = 1
        adjacency[y, h] = 0
    end

    return adjacency
end


# =============================================================================
# Scoring Operations
# =============================================================================

"""
    score_insert(data, adjacency, x, y, T, score_type; cache=nothing)

Compute score change from insert(x, y, T).
"""
function score_insert(
    data::Matrix{Float64},
    adjacency::AbstractMatrix,
    x::Int,
    y::Int,
    T::Set{Int},
    score_type::ScoreType;
    cache::Union{Dict{Tuple{Int,Set{Int}},LocalScore},Nothing} = nothing
)::Float64
    # Current parents of y
    old_parents = get_parents(adjacency, y)

    # New parents of y: old + x + T (T becomes parents from neighbors)
    new_parents = union(old_parents, Set([x]), T)

    # Score change for y
    old_score = local_score(data, y, old_parents, score_type; cache=cache)
    new_score = local_score(data, y, new_parents, score_type; cache=cache)

    return new_score.score - old_score.score
end


"""
    score_delete(data, adjacency, x, y, H, score_type; cache=nothing)

Compute score change from delete(x, y, H).
"""
function score_delete(
    data::Matrix{Float64},
    adjacency::AbstractMatrix,
    x::Int,
    y::Int,
    H::Set{Int},
    score_type::ScoreType;
    cache::Union{Dict{Tuple{Int,Set{Int}},LocalScore},Nothing} = nothing
)::Float64
    # Current parents of y
    old_parents = get_parents(adjacency, y)
    neighbors_y = get_neighbors(adjacency, y)

    # If x -> y (directed), remove x from parents
    # If x - y (undirected), no change to parents
    if x in old_parents
        base_parents = setdiff(old_parents, Set([x]))
    else
        base_parents = old_parents
    end

    # New parents: base + (Na_y \ H) oriented as parents
    Na_y = union(neighbors_y, old_parents)
    new_directed = intersect(setdiff(Na_y, H, Set([x])), neighbors_y)
    new_parents = union(base_parents, new_directed)

    old_score = local_score(data, y, old_parents, score_type; cache=cache)
    new_score = local_score(data, y, new_parents, score_type; cache=cache)

    return new_score.score - old_score.score
end


# =============================================================================
# Main Algorithm Phases
# =============================================================================

"""
    ges_forward(data, adjacency, score_type; cache=nothing, max_parents=10)

GES forward phase: greedily add edges.

# Arguments
- `data::Matrix{Float64}`: (n_samples, n_vars) data matrix
- `adjacency::Matrix`: Current adjacency matrix
- `score_type::ScoreType`: Score function to use
- `cache::Union{Dict,Nothing}`: Cache for local scores
- `max_parents::Int`: Maximum number of parents per node

# Returns
- `Tuple{Matrix, Int, Vector{Float64}}`: (final adjacency, n_steps, scores)
"""
function ges_forward(
    data::Matrix{Float64},
    adjacency::AbstractMatrix,
    score_type::ScoreType = BIC;
    cache::Union{Dict{Tuple{Int,Set{Int}},LocalScore},Nothing} = nothing,
    max_parents::Int = 10
)
    if cache === nothing
        cache = Dict{Tuple{Int,Set{Int}},LocalScore}()
    end

    n_vars = size(data, 2)
    adj = copy(adjacency)
    n_steps = 0
    scores = Float64[]

    while true
        best_delta = 0.0
        best_op = nothing

        # Try all possible insert(x, y, T) operations
        for x in 1:n_vars
            for y in 1:n_vars
                if x == y
                    continue
                end

                # Check if already adjacent
                if adj[x, y] == 1 || adj[y, x] == 1
                    continue
                end

                # Check parent limit
                if length(get_parents(adj, y)) >= max_parents
                    continue
                end

                # Try all subsets T of neighbors(y)
                neighbors_y = get_neighbors(adj, y)
                neighbors_list = collect(neighbors_y)

                for T_size in 0:length(neighbors_list)
                    for T_tuple in combinations(neighbors_list, T_size)
                        T = Set(T_tuple)
                        if is_valid_insert(adj, x, y, T)
                            delta = score_insert(data, adj, x, y, T, score_type; cache=cache)
                            if delta > best_delta
                                best_delta = delta
                                best_op = (x, y, T)
                            end
                        end
                    end
                end
            end
        end

        if best_op === nothing
            break
        end

        # Apply best operation
        x, y, T = best_op
        apply_insert!(adj, x, y, T)
        n_steps += 1
        push!(scores, total_score(data, adj, score_type; cache=cache))
    end

    return adj, n_steps, scores
end


"""
    ges_backward(data, adjacency, score_type; cache=nothing)

GES backward phase: greedily remove edges.

# Arguments
- `data::Matrix{Float64}`: (n_samples, n_vars) data matrix
- `adjacency::Matrix`: Current adjacency matrix
- `score_type::ScoreType`: Score function to use
- `cache::Union{Dict,Nothing}`: Cache for local scores

# Returns
- `Tuple{Matrix, Int, Vector{Float64}}`: (final adjacency, n_steps, scores)
"""
function ges_backward(
    data::Matrix{Float64},
    adjacency::AbstractMatrix,
    score_type::ScoreType = BIC;
    cache::Union{Dict{Tuple{Int,Set{Int}},LocalScore},Nothing} = nothing
)
    if cache === nothing
        cache = Dict{Tuple{Int,Set{Int}},LocalScore}()
    end

    n_vars = size(data, 2)
    adj = copy(adjacency)
    n_steps = 0
    scores = Float64[]

    while true
        best_delta = 0.0
        best_op = nothing

        # Try all possible delete(x, y, H) operations
        for x in 1:n_vars
            for y in 1:n_vars
                if x == y
                    continue
                end

                # Check if adjacent
                if adj[x, y] == 0 && adj[y, x] == 0
                    continue
                end

                # Try all subsets H of neighbors(y)
                neighbors_y = get_neighbors(adj, y)
                neighbors_list = collect(neighbors_y)

                for H_size in 0:length(neighbors_list)
                    for H_tuple in combinations(neighbors_list, H_size)
                        H = Set(H_tuple)
                        if is_valid_delete(adj, x, y, H)
                            delta = score_delete(data, adj, x, y, H, score_type; cache=cache)
                            if delta > best_delta
                                best_delta = delta
                                best_op = (x, y, H)
                            end
                        end
                    end
                end
            end
        end

        if best_op === nothing
            break
        end

        # Apply best operation
        x, y, H = best_op
        apply_delete!(adj, x, y, H)
        n_steps += 1
        push!(scores, total_score(data, adj, score_type; cache=cache))
    end

    return adj, n_steps, scores
end


# =============================================================================
# Adjacency to CPDAG Conversion
# =============================================================================

"""
    adjacency_to_cpdag(adjacency; var_names=nothing)

Convert adjacency matrix to CPDAG object.
"""
function adjacency_to_cpdag(
    adjacency::AbstractMatrix;
    var_names::Union{Vector{String},Nothing} = nothing
)
    n = size(adjacency, 1)
    if var_names === nothing
        var_names = ["X$i" for i in 0:n-1]
    end

    cpdag = CPDAG(n; node_names=var_names)

    for i in 1:n
        for j in i+1:n
            if adjacency[i, j] == 1 && adjacency[j, i] == 1
                # Undirected edge
                add_undirected_edge!(cpdag, i, j)
            elseif adjacency[i, j] == 1 && adjacency[j, i] == 0
                # Directed edge i -> j
                add_directed_edge!(cpdag, i, j)
            elseif adjacency[i, j] == 0 && adjacency[j, i] == 1
                # Directed edge j -> i
                add_directed_edge!(cpdag, j, i)
            end
        end
    end

    return cpdag
end


# =============================================================================
# Main Entry Point
# =============================================================================

"""
    ges_algorithm(data; score_type=:bic, var_names=nothing, max_parents=10, verbose=false)

Greedy Equivalence Search for causal discovery.

# Arguments
- `data::Matrix{Float64}`: (n_samples, n_vars) data matrix
- `score_type::Union{Symbol,ScoreType}`: Score function: :bic (default), :aic
- `var_names::Union{Vector{String},Nothing}`: Variable names
- `max_parents::Int`: Maximum number of parents per node
- `verbose::Bool`: Print progress

# Returns
- `GESResult`: GES algorithm result

# Example
```julia
using Random
Random.seed!(42)
n = 500
X1 = randn(n)
X2 = 0.8 .* X1 .+ randn(n) .* 0.5
X3 = 0.6 .* X2 .+ randn(n) .* 0.5
data = hcat(X1, X2, X3)
result = ges_algorithm(data; score_type=:bic)
println("Forward steps: \$(result.n_forward_steps)")
```
"""
function ges_algorithm(
    data::Matrix{Float64};
    score_type::Union{Symbol,ScoreType} = :bic,
    var_names::Union{Vector{String},Nothing} = nothing,
    max_parents::Int = 10,
    verbose::Bool = false
)
    # Validate inputs
    if ndims(data) != 2
        error("data must be 2D, got $(ndims(data))D")
    end

    n_samples, n_vars = size(data)

    if n_samples < 10
        error("Too few samples: $n_samples < 10")
    end

    if n_vars < 2
        error("Need at least 2 variables, got $n_vars")
    end

    # Parse score type
    if score_type isa Symbol
        if score_type == :bic
            st = BIC
        elseif score_type == :aic
            st = AIC
        else
            error("Unknown score_type: $score_type. Use :bic or :aic")
        end
    else
        st = score_type
    end

    if var_names === nothing
        var_names = ["X$i" for i in 0:n_vars-1]
    end

    # Initialize empty graph
    adjacency = zeros(Float64, n_vars, n_vars)
    cache = Dict{Tuple{Int,Set{Int}},LocalScore}()

    # Forward phase
    if verbose
        println("GES Forward Phase...")
    end
    adj_fwd, n_fwd, scores_fwd = ges_forward(
        data, adjacency, st; cache=cache, max_parents=max_parents
    )

    if verbose
        println("  Added $n_fwd edges")
    end

    # Backward phase
    if verbose
        println("GES Backward Phase...")
    end
    adj_final, n_bwd, scores_bwd = ges_backward(
        data, adj_fwd, st; cache=cache
    )

    if verbose
        println("  Removed $n_bwd edges")
    end

    # Compute final score
    final_score = total_score(data, adj_final, st; cache=cache)

    # Convert to CPDAG
    cpdag = adjacency_to_cpdag(adj_final; var_names=var_names)

    return GESResult(
        cpdag,
        final_score,
        n_fwd,
        n_bwd,
        scores_fwd,
        scores_bwd,
        n_vars,
        n_samples,
        st
    )
end


end # module
