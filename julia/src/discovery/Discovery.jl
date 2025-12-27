"""
    Discovery Module

Session 133-134: Causal discovery algorithms for structure learning.
Session 151: GES algorithm (Greedy Equivalence Search).

# Submodules
- `DiscoveryTypes`: Graph, DAG, CPDAG, PAG, result types
- `IndependenceTests`: Conditional independence tests
- `PCAlgorithm`: PC algorithm
- `FCIAlgorithm`: FCI algorithm (handles latent confounders)
- `LiNGAM`: Linear Non-Gaussian Acyclic Model
- `ScoreFunctions`: BIC, AIC score functions for GES
- `GESAlgorithm`: Greedy Equivalence Search
- `DiscoveryUtils`: DAG generation, metrics

# Main Functions
- `pc_algorithm`: Learn CPDAG from observational data
- `fci_algorithm`: Learn PAG allowing for latent confounders
- `direct_lingam`: Learn unique DAG via non-Gaussianity
- `ges_algorithm`: Learn CPDAG via score optimization (GES)
- `generate_random_dag`: Generate test DAGs
- `generate_dag_data`: Generate data from SCM
"""
module Discovery

# Include submodules
include("types.jl")
include("independence_tests.jl")
include("pc_algorithm.jl")
include("fci_algorithm.jl")
include("lingam.jl")
include("score_functions.jl")
include("ges_algorithm.jl")
include("utils.jl")

# Re-export from submodules
using .DiscoveryTypes
using .IndependenceTests
using .PCAlgorithm
using .FCIAlgorithm
using .LiNGAM
using .ScoreFunctions
using .GESAlgorithm
using .DiscoveryUtils

# Types
export Graph, DAG, CPDAG, PCResult, LiNGAMResult, CITestResult
export EdgeMark, PAG, FCIResult
export NONE, TAIL, ARROW, CIRCLE  # EdgeMark values

# PC Algorithm
export pc_algorithm, pc_skeleton, pc_orient

# FCI Algorithm
export fci_algorithm, fci_orient

# LiNGAM
export direct_lingam, ica_lingam

# GES Algorithm (Session 151)
export ScoreType, BIC, AIC, BIC_G
export LocalScore, GESResult
export ges_algorithm, ges_forward, ges_backward
export local_score, local_score_bic, local_score_aic, total_score
export compute_rss, score_delta_add, score_delta_remove
export get_neighbors, get_parents, get_children, is_clique

# Utilities
export generate_random_dag, generate_dag_data
export skeleton_f1, compute_shd, dag_to_cpdag
export partial_correlation, fisher_z_test, ci_test

# PAG utilities
export n_directed_edges, n_bidirected_edges, n_circle_edges

end # module
