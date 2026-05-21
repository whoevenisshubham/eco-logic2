# Phase 3

This phase adds an AST-GNN regression pipeline on top of the Tree-sitter graph builder.

## What it does

- Converts each snippet into a graph using Tree-sitter.
- Uses node features for:
  - node type id
  - token hash embedding
  - depth
  - positional index
  - line / byte position
  - subtree and leaf indicators
- Adds edge types for:
  - parent-child
  - next-token preorder links
  - optional repeated-identifier data-flow links
- Trains a pure PyTorch GraphSAGE-style regressor to predict energy in Joules.
- Splits by `snippet_id` so train/test leakage is avoided.
- Compares the GNN against the Phase 1 RandomForest baseline on the same test split.

## Run the demo

```bash
python tests/phase3_gnn_demo.py
```

## Run a longer experiment

```bash
python phase3_ast_gnn.py --epochs 12 --limit 200 --batch-size 8
```

## Ablations supported

- `--no-data-flow`
- `--no-cache-locality`

## Notes

- The implementation is intentionally pure PyTorch so it runs in this workspace without PyTorch Geometric.
- If you later want a research-grade version, the same graph builder can be ported to PyTorch Geometric or DGL.
