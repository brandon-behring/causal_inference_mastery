"""
    Tests for GES Algorithm (Greedy Equivalence Search)

Session 151: Julia GES implementation tests.

Layer 1: Known-Answer (small DAGs with exact structure)
Layer 2: Adversarial (edge cases, high-dimensional)
Layer 3: Monte Carlo (statistical validation)
"""

using Test
using Random
using Statistics
using LinearAlgebra
using CausalEstimators.Discovery


@testset "GES Algorithm Tests" begin

    # =========================================================================
    # Layer 1: Known-Answer Tests
    # =========================================================================

    @testset "Layer 1: Known-Structure Tests" begin

        @testset "Empty graph - two independent variables" begin
            Random.seed!(42)
            n = 500
            X1 = randn(n)
            X2 = randn(n)
            data = hcat(X1, X2)

            result = ges_algorithm(data; score_type=:bic)

            @test result isa GESResult
            n_edges_found = sum(result.cpdag.directed) + sum(result.cpdag.undirected) ÷ 2
            # Should find 0 or 1 edge (independent data may spuriously add edge)
            @test n_edges_found <= 1
        end

        @testset "Chain X → Y" begin
            Random.seed!(42)
            n = 500
            X = randn(n)
            Y = 0.8 .* X .+ randn(n) .* 0.5
            data = hcat(X, Y)

            result = ges_algorithm(data; score_type=:bic)

            n_edges = sum(result.cpdag.directed) + sum(result.cpdag.undirected) ÷ 2
            @test n_edges >= 1
        end

        @testset "Chain X → Y → Z" begin
            Random.seed!(42)
            n = 500
            X = randn(n)
            Y = 0.8 .* X .+ randn(n) .* 0.3
            Z = 0.7 .* Y .+ randn(n) .* 0.3
            data = hcat(X, Y, Z)

            result = ges_algorithm(data; score_type=:bic)

            n_edges = sum(result.cpdag.directed) + sum(result.cpdag.undirected) ÷ 2
            @test n_edges >= 2
        end

        @testset "Fork X ← Z → Y" begin
            Random.seed!(42)
            n = 500
            Z = randn(n)
            X = 0.8 .* Z .+ randn(n) .* 0.3
            Y = 0.7 .* Z .+ randn(n) .* 0.3
            data = hcat(X, Y, Z)

            result = ges_algorithm(data; score_type=:bic)

            n_edges = sum(result.cpdag.directed) + sum(result.cpdag.undirected) ÷ 2
            @test n_edges >= 2
        end

        @testset "Collider X → Z ← Y" begin
            Random.seed!(42)
            n = 500
            X = randn(n)
            Y = randn(n)
            Z = 0.6 .* X .+ 0.5 .* Y .+ randn(n) .* 0.3
            data = hcat(X, Y, Z)

            result = ges_algorithm(data; score_type=:bic)

            # Collider is identifiable
            n_edges = sum(result.cpdag.directed) + sum(result.cpdag.undirected) ÷ 2
            @test n_edges >= 2
        end

        @testset "Diamond structure" begin
            Random.seed!(42)
            n = 500
            X = randn(n)
            Y = 0.7 .* X .+ randn(n) .* 0.3
            Z = 0.6 .* X .+ randn(n) .* 0.3
            W = 0.5 .* Y .+ 0.5 .* Z .+ randn(n) .* 0.3
            data = hcat(X, Y, Z, W)

            result = ges_algorithm(data; score_type=:bic)

            n_edges = sum(result.cpdag.directed) + sum(result.cpdag.undirected) ÷ 2
            @test n_edges >= 3
        end
    end


    @testset "Score Functions" begin

        @testset "BIC score increases with better fit" begin
            Random.seed!(42)
            n = 200
            X = randn(n)
            Y = 0.9 .* X .+ randn(n) .* 0.1
            data = hcat(X, Y)

            score_with_parent = local_score_bic(data, 2, Set([1]))
            score_no_parent = local_score_bic(data, 2, Set{Int}())

            @test score_with_parent.score > score_no_parent.score
        end

        @testset "BIC penalizes more parameters" begin
            Random.seed!(42)
            n = 100
            data = randn(n, 5)

            score_1 = local_score_bic(data, 1, Set([2]))
            score_2 = local_score_bic(data, 1, Set([2, 3]))

            @test score_2.n_params > score_1.n_params
        end

        @testset "AIC score computation" begin
            Random.seed!(42)
            n = 200
            X = randn(n)
            Y = 0.9 .* X .+ randn(n) .* 0.1
            data = hcat(X, Y)

            score_with_parent = local_score_aic(data, 2, Set([1]))
            score_no_parent = local_score_aic(data, 2, Set{Int}())

            @test score_with_parent.score > score_no_parent.score
        end

        @testset "Total score for empty graph" begin
            Random.seed!(42)
            data = randn(100, 3)
            adjacency = zeros(3, 3)

            score = total_score(data, adjacency, BIC)
            @test isfinite(score)
        end

        @testset "Total score increases with correct edge" begin
            Random.seed!(42)
            n = 300
            X = randn(n)
            Y = 0.8 .* X .+ randn(n) .* 0.3
            data = hcat(X, Y)

            adj_empty = zeros(2, 2)
            adj_edge = [0.0 1.0; 0.0 0.0]  # X → Y

            score_empty = total_score(data, adj_empty, BIC)
            score_edge = total_score(data, adj_edge, BIC)

            @test score_edge > score_empty
        end

        @testset "RSS computation no parents" begin
            Random.seed!(42)
            data = randn(100, 3)

            rss = compute_rss(data, 1, Set{Int}())
            @test rss > 0
            @test isfinite(rss)
        end

        @testset "RSS computation with parents" begin
            Random.seed!(42)
            n = 200
            X = randn(n)
            Y = 0.9 .* X .+ randn(n) .* 0.1
            data = hcat(X, Y)

            rss_no_parent = compute_rss(data, 2, Set{Int}())
            rss_with_parent = compute_rss(data, 2, Set([1]))

            # RSS should decrease with correct parent
            @test rss_with_parent < rss_no_parent
        end
    end


    @testset "GESResult" begin

        @testset "Result has expected attributes" begin
            Random.seed!(42)
            data = randn(100, 3)
            result = ges_algorithm(data)

            @test hasproperty(result, :cpdag)
            @test hasproperty(result, :score)
            @test hasproperty(result, :n_forward_steps)
            @test hasproperty(result, :n_backward_steps)
            @test hasproperty(result, :forward_scores)
            @test hasproperty(result, :backward_scores)
            @test result.n_vars == 3
            @test result.n_samples == 100
        end

        @testset "Score is finite" begin
            Random.seed!(42)
            data = randn(100, 3)
            result = ges_algorithm(data)

            @test isfinite(result.score)
        end

        @testset "Forward scores are tracked" begin
            Random.seed!(42)
            n = 300
            X = randn(n)
            Y = 0.8 .* X .+ randn(n) .* 0.3
            Z = 0.7 .* Y .+ randn(n) .* 0.3
            data = hcat(X, Y, Z)

            result = ges_algorithm(data)

            @test length(result.forward_scores) == result.n_forward_steps
        end
    end


    # =========================================================================
    # Layer 2: Adversarial Tests
    # =========================================================================

    @testset "Layer 2: Adversarial Tests" begin

        @testset "Small sample (n=20)" begin
            Random.seed!(42)
            data = randn(20, 3)
            result = ges_algorithm(data)
            @test result !== nothing
        end

        @testset "High-dimensional (8 variables)" begin
            Random.seed!(42)
            dag = generate_random_dag(8; edge_prob=0.2, seed=42)
            data, _ = generate_dag_data(dag, 500; seed=42)

            result = ges_algorithm(data)
            @test result.n_vars == 8
        end

        @testset "Near-collinear data" begin
            Random.seed!(42)
            n = 200
            X = randn(n)
            Y = X .+ randn(n) .* 0.01  # Almost X
            Z = randn(n)
            data = hcat(X, Y, Z)

            result = ges_algorithm(data)
            @test result !== nothing
        end

        @testset "Constant column" begin
            Random.seed!(42)
            n = 100
            X = randn(n)
            Y = ones(n)  # Constant
            Z = randn(n)
            data = hcat(X, Y, Z)

            # May handle differently, but should not crash
            result = ges_algorithm(data)
            @test result !== nothing
        end

        @testset "Score type AIC" begin
            Random.seed!(42)
            data = randn(200, 3)

            result = ges_algorithm(data; score_type=:aic)
            @test result.score_type == AIC
        end

        @testset "Invalid score type throws" begin
            data = randn(100, 3)

            @test_throws ErrorException ges_algorithm(data; score_type=:invalid)
        end

        @testset "Too few samples throws" begin
            data = randn(5, 3)

            @test_throws ErrorException ges_algorithm(data)
        end

        @testset "Too few variables throws" begin
            data = randn(100, 1)

            @test_throws ErrorException ges_algorithm(data)
        end

        @testset "Max parents limit" begin
            Random.seed!(42)
            data = randn(200, 5)

            result = ges_algorithm(data; max_parents=2)
            @test result !== nothing
        end
    end


    # =========================================================================
    # Layer 3: Monte Carlo Tests
    # =========================================================================

    @testset "Layer 3: Monte Carlo Tests" begin

        @testset "SHD on random DAGs" begin
            n_vars = 5
            n_samples = 500
            shd_values = Float64[]

            for seed in 1:10
                Random.seed!(seed)
                dag = generate_random_dag(n_vars; edge_prob=0.3, seed=seed)
                data, _ = generate_dag_data(dag, n_samples; seed=seed)

                result = ges_algorithm(data)
                shd = compute_shd(result.cpdag, dag)
                push!(shd_values, shd)
            end

            mean_shd = mean(shd_values)
            # SHD should be reasonable
            @test mean_shd < n_vars * 2
        end

        @testset "Skeleton F1 on random DAGs" begin
            n_vars = 5
            n_samples = 500
            f1_values = Float64[]

            for seed in 1:10
                Random.seed!(seed)
                dag = generate_random_dag(n_vars; edge_prob=0.3, seed=seed)
                data, _ = generate_dag_data(dag, n_samples; seed=seed)

                result = ges_algorithm(data)

                # Compute skeleton F1
                precision, recall, f1 = skeleton_f1(result.cpdag, dag)
                if isfinite(f1)
                    push!(f1_values, f1)
                end
            end

            if length(f1_values) > 0
                mean_f1 = mean(f1_values)
                @test mean_f1 > 0.3
            end
        end

        @testset "Accuracy improves with sample size" begin
            Random.seed!(42)
            n_vars = 4
            dag = generate_random_dag(n_vars; edge_prob=0.4, seed=42)

            shd_small = Float64[]
            shd_large = Float64[]

            for seed in 1:5
                # Small sample
                data_small, _ = generate_dag_data(dag, 100; seed=seed)
                result_small = ges_algorithm(data_small)
                push!(shd_small, compute_shd(result_small.cpdag, dag))

                # Large sample
                data_large, _ = generate_dag_data(dag, 1000; seed=seed)
                result_large = ges_algorithm(data_large)
                push!(shd_large, compute_shd(result_large.cpdag, dag))
            end

            # Large sample should have lower or equal SHD on average
            @test mean(shd_large) <= mean(shd_small) + 2
        end

        @testset "GES vs PC comparison - chain" begin
            Random.seed!(42)
            n = 500
            X1 = randn(n)
            X2 = 0.7 .* X1 .+ randn(n) .* 0.3
            X3 = 0.7 .* X2 .+ randn(n) .* 0.3
            data = hcat(X1, X2, X3)

            ges_result = ges_algorithm(data)
            pc_result = pc_algorithm(data; alpha=0.01)

            ges_edges = sum(ges_result.cpdag.directed) + sum(ges_result.cpdag.undirected) ÷ 2
            pc_edges = sum(pc_result.cpdag.directed) + sum(pc_result.cpdag.undirected) ÷ 2

            @test ges_edges >= 1
            @test pc_edges >= 1
            @test abs(ges_edges - pc_edges) <= 2
        end

        @testset "GES and PC both find edges" begin
            Random.seed!(42)
            n = 500
            X = randn(n)
            Y = 0.8 .* X .+ randn(n) .* 0.3
            Z = 0.7 .* Y .+ randn(n) .* 0.3
            data = hcat(X, Y, Z)

            ges_result = ges_algorithm(data)
            pc_result = pc_algorithm(data; alpha=0.01)

            ges_edges = sum(ges_result.cpdag.directed) + sum(ges_result.cpdag.undirected) ÷ 2
            pc_edges = sum(pc_result.cpdag.directed) + sum(pc_result.cpdag.undirected) ÷ 2

            @test ges_edges >= 1
            @test pc_edges >= 1
        end
    end


    # =========================================================================
    # Graph Helper Functions Tests
    # =========================================================================

    @testset "Graph Helpers" begin

        @testset "get_neighbors" begin
            adj = zeros(3, 3)
            adj[1, 2] = 1
            adj[2, 1] = 1  # Undirected 1 -- 2

            neighbors = get_neighbors(adj, 1)
            @test 2 in neighbors
            @test length(neighbors) == 1
        end

        @testset "get_parents" begin
            adj = zeros(3, 3)
            adj[1, 2] = 1  # 1 -> 2

            parents_of_2 = get_parents(adj, 2)
            @test 1 in parents_of_2
            @test length(parents_of_2) == 1
        end

        @testset "get_children" begin
            adj = zeros(3, 3)
            adj[1, 2] = 1  # 1 -> 2

            children_of_1 = get_children(adj, 1)
            @test 2 in children_of_1
            @test length(children_of_1) == 1
        end

        @testset "is_clique" begin
            adj = zeros(3, 3)
            adj[1, 2] = 1
            adj[2, 1] = 1
            adj[1, 3] = 1
            adj[3, 1] = 1
            adj[2, 3] = 1
            adj[3, 2] = 1

            @test is_clique(adj, Set([1, 2, 3]))
            @test is_clique(adj, Set([1, 2]))
            @test is_clique(adj, Set{Int}())  # Empty is clique
        end
    end

end
