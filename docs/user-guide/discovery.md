# Causal Discovery

Learn causal graph structure from observational data — discover *which* variables cause *which*.

## Algorithms

| Algorithm | Assumptions | Output |
|-----------|-------------|--------|
| **PC** | Causal sufficiency, faithfulness | CPDAG (equivalence class) |
| **FCI** | Allows latent confounders | PAG (partial ancestral graph) |
| **GES** | Greedy search, BIC scoring | CPDAG |
| **LiNGAM** | Linear non-Gaussian | Unique DAG |

## Example

```python
from causal_inference.discovery import PC

pc = PC(alpha=0.05)
result = pc.fit(data[["x1", "x2", "x3", "y"]])
print(f"Edges: {result.edges}")
print(f"Undirected: {result.undirected_edges}")
```

## Interpretation

- **Directed edge** ($X \to Y$): $X$ causes $Y$
- **Undirected edge** ($X - Y$): Causal direction cannot be determined from data alone
- **Bi-directed edge** ($X \leftrightarrow Y$): Latent common cause (FCI only)

## Limitations

- Sample size requirements grow with graph complexity
- Faithfulness assumption can fail in practice
- Cannot distinguish Markov-equivalent DAGs without additional assumptions

See {doc}`/api/discovery` for full API reference.
