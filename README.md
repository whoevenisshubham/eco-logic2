# Freshstart — Energy-aware Refactor Agent

This repository implements a closed-loop refactoring agent that: parses code, extracts features, predicts energy use, measures runtime when possible, converts energy to carbon, and visualizes an Energy–Time Pareto frontier. It includes a RandomForest Phase‑1/Phase‑2 baseline and a Phase‑3 AST‑GNN regressor.

This README covers how to set up, run, test, and push changes to Git.

## Quick setup

1. Clone or open the repo in your workspace.
2. Create a Python virtual environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

If `requirements.txt` does not exist, ensure you have these (typical):

```bash
pip install streamlit plotly scikit-learn torch torchvision torchaudio tree_sitter pandas numpy matplotlib seaborn
```

## Environment variables

Create a `.env` or export environment variables in your shell. Useful keys:

- `GROQ_API_KEY` — API key for Groq/GPT provider wrapper used by `groq_client.py` (if applicable).
- `ELECTRICITY_MAPS_API_KEY` — (Optional) for `carbon_providers.ElectricityMapsAdapter`.
- `WATTTIME_TOKEN`, `WATTTIME_BA` — (Optional) for `WattTimeAdapter`.

Load `.env` via `load_env.py` if present, or export manually.

## Run the Streamlit app (dashboard)

From the repository root:

```bash
streamlit run app.py
```

The UI shows: input code, generated LLM refactor candidates, heuristic candidates, predicted energy (J), predicted carbon (gCO2eq), and an interactive Pareto frontier. Measured runtimes (Phase‑1.5) replace proxy runtime when available.

## Scripts and common commands

- Evaluate refactor candidates (LLM + heuristics):

```bash
PYTHONPATH=. python scripts/eval_refactor_candidates.py
```

- Train Phase‑3 AST‑GNN model (example):

```bash
python phase3_ast_gnn.py --data-file data/train_dataset.parquet --epochs 30 --save-path phase3_model.pth
```

- Evaluate saved Phase‑3 model:

```bash
python evaluate_phase3_saved.py --model-path phase3_model.pth
```

## Models and artifacts

- `phase1_model.pkl`, `phase2_model.pkl` — scikit-learn RandomForest baselines (legacy 9‑feature compatibility).
- `phase3_model.pth` — PyTorch AST‑GNN regressor checkpoint.
- `eco_logic_synthetic_benchmark.csv` — benchmark dataset used for initial prototyping.

Keep large artifacts out of Git (add them to `.gitignore`) and store model checkpoints in a separate release storage if required.

## How to accept the optimized refactor into the UI (manual)

1. Run evaluation script to generate candidate JSON: `scripts/eval_refactor_candidates_output.json`.
2. In the Streamlit UI, select the candidate you want to accept (heuristic or LLM). The UI will display the accepted candidate and update the Pareto.
3. To persist accepted code, update `st.session_state` usage in `app.py` or save the chosen candidate to `data/accepted_refactor.cpp` (or similar) via the UI controls.

## Git: commit & push (recommended workflow)

Use the following sequence for a clean commit and push. Replace `origin` and `main` as appropriate.

```bash
# show status
git status

# create a feature branch
git checkout -b feat/accept-early-exit-bubble

# stage changes
git add .

# commit with concise message
git commit -m "feat: accept early-exit bubble refactor; update prompts and eval script"

# push branch
git push -u origin feat/accept-early-exit-bubble
```

If you need to squash or rebase before merging into `main`, use your project's PR workflow.

## Tests & verification

Run the evaluation script and `evaluate_phase3_saved.py` to sanity check models.

Optional quick smoke tests

```bash
python -m pytest -q tests || true
```

## Developer notes

- LLM outputs are strictly validated by `has_valid_function_body()` and `validate_feature_vector_9()` to ensure algorithmic implementations (no library-only shortcuts) before accepting them.
- Measured runtime (`runtime_harness.py`) is preferred over proxy predictions when available; UI Pareto uses measured values when present.

## Licensing and contribution

Add your preferred license in `LICENSE` and contribution guide in `CONTRIBUTING.md` if you intend to accept external PRs.

---
If you'd like, I can also add a `requirements.txt`, `CONTRIBUTING.md`, or a small test that verifies the `early_exit_bubble` heuristic outperforms the original for the provided benchmark.
