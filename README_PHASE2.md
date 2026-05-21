# Phase 2

This phase replaces the line-based parser with a Tree-sitter-backed analysis layer and expands the feature space beyond the original binary flags.

## What changed

- Tree-sitter parsing for Python and C++ when grammars are available.
- Rich feature extraction:
  - cyclomatic complexity
  - Halstead-style operator and operand counts
  - branching-factor statistics
  - loop nesting depth
  - allocation pressure
  - cache-locality stride penalty
- Compatibility vector preserved for the existing RandomForest baseline.
- Fallback parser retained if Tree-sitter cannot parse a snippet.

## Run the validation demo

```bash
python tests/phase2_treesitter_demo.py
```

## Notes

- `phase1_model.pkl` still expects 9 input features, so the app uses a compatibility projection while exposing the richer Tree-sitter metrics in the UI.
- To retrain a Phase 2 model later, use the rich metrics in `phase2_features.py` as the new feature set.
