# Technical Deep-Dive — Freshstart

This document explains the project in depth: layman-level goals and motivation, theoretical background, design and architecture, implementation details, file-by-file descriptions, training and evaluation processes, deployment notes, and developer guidance.

---

**1. Project elevator pitch (layman)**

This project is a tool that takes program code variants (original, LLM-suggested refactors, and handcrafted heuristics), evaluates their computational cost, and helps choose the best variant when trading off execution time and environmental cost (energy and carbon). It automates measuring or estimating runtime and energy, predicts energy usage with ML models, and visualizes tradeoffs to guide decisions.

Why it matters: small algorithmic improvements in hot code can reduce CPU time, energy consumption, and carbon emissions. Developers need a reproducible way to compare implementations beyond microbenchmarks and noisy runs.

**2. High-level architecture**

- UI / Orchestration: `app.py` (Streamlit) — receives code input, invokes the LLM/heuristic generator, runs feature extraction, queries ML predictors, optionally measures runtime, computes carbon, and renders a Pareto frontier.
- LLM wrapper: `groq_client.py` — abstracted interface to call the configured LLM provider.
- Feature extraction: `phase2_features.py` — Tree‑sitter-based AST parsing and feature engineering; includes a `legacy_model_vector()` that returns the deterministic 9 features required by older RF baselines.
- Runtime measurement: `runtime_harness.py` — safely compiles/runs C++/Python snippets in a sandboxed harness, times runs, and returns measured runtime_ms. Falls back to proxy runtime predictions when execution is infeasible.
- Carbon providers: `carbon_providers.py` — adapter pattern to get carbon intensity (gCO2eq/kWh) from various APIs with an offline fallback; used to convert predicted energy (J) → gCO2eq.
- Phase‑1/2 predictor(s): scikit-learn RandomForest pickles (e.g., `phase1_model.pkl`, `phase2_model.pkl`) — fast legacy predictors that accept a 9-dim feature vector.
- Phase‑3 AST‑GNN: `phase3_ast_gnn.py` — constructs graph datasets from Tree‑sitter ASTs, trains a GraphSAGE-like encoder and MLP regressor in PyTorch. Produces `phase3_model.pth`.
- Evaluation scripts: `scripts/eval_refactor_candidates.py` and related scripts perform candidate generation, prediction, measurement, and JSON export for inspection.

**3. Theoretical background**

- Energy prediction: We map program features (e.g., loop depth, memory operations, arithmetic intensity) to an estimated energy use in joules. A regression model (RF or GNN) learns this mapping from labeled runs where we measure runtime and infer energy from measured power or system proxies.
- Carbon translation: Energy (J) → kWh → gCO2eq via carbon intensity (gCO2eq/kWh) where carbon intensity is provider/sourced from public APIs or offline datasets.
- AST‑GNN motivation: structural program information is rich — AST nodes and their relations capture control and data-flow structure. GNNs operating on AST-derived graphs can better generalize energy-relevant patterns than flat feature vectors.

**4. Data and labels**

- Primary CSV: `eco_logic_synthetic_benchmark.csv` — contains synthetic benchmark entries with code variants and measured runtimes. Used for exploratory analysis and model prototyping.
- Training artifacts: parquet/cached datasets created by `phase3_ast_gnn.py` preprocessing step hold graph representations and target energy labels. Keep large ones outside Git.

**5. Files: complete walkthrough**

Below is a file-by-file description of source files, scripts, and artifacts. Each entry includes purpose, key functions/classes, and important implementation notes.

- `app.py` — Streamlit dashboard and orchestrator.
  - Responsibilities:
    - Accept code input (text area) or example selection.
    - Generate candidate refactors using the LLM (`groq_client.py`) and local heuristics.
    - Validate LLM outputs with `has_valid_function_body()` to ensure a real algorithmic implementation was returned (not a library-only solution or empty output).
    - Extract features via `phase2_features.analyze_code_features()`; for backward compatibility calls `legacy_model_vector()` which returns the deterministic 9 features in a fixed order.
    - Query predictors (`phase2_model.pkl` or `phase3_model.pth`) and display predicted energy and carbon.
    - Optionally measure runtime via `runtime_harness.measure_runtime()`; when measured results exist, they replace proxy/runtime predictions in the Pareto chart.
    - Provide UI controls to accept a candidate and (optionally) persist it.

  - Important validation functions (examples):
    - `has_valid_function_body(code_text) -> bool` — checks that returned code includes a substantive function body.
    - `validate_feature_vector_9(vec) -> bool` — asserts the 9-dim legacy feature vector length and value ranges.

- `groq_client.py` — LLM wrapper.
  - Purpose: Provide a small wrapper around the chosen LLM provider.
  - Exposes `generate_refactor(prompt, temperature=0.0)` or similar.
  - Prompt engineering: the project forces LLMs to return full algorithmic implementations (explicitly forbids `std::sort`, `sorted`, `numpy.dot`), and asks for a single self-contained function body written in the requested language.

- `carbon_providers.py` — adapters for carbon intensity.
  - Implements adapter classes: `ElectricityMapsAdapter`, `WattTimeAdapter`, `OfflineFallbackAdapter`.
  - Exposes `get_live_carbon_intensity(location, timestamp)` which returns `gCO2eq_per_kWh`.
  - The UI uses a cache layer (`@st.cache_data(ttl=600)`) to avoid repeated API calls.

- `runtime_harness.py` — execution harness for measurements.
  - Safely runs candidate code variants, collects runtime in ms, and returns `runtime_mode` (`measured` or `proxy`).
  - For C++: writes to temp file, compiles (with `-O2` or user-specified flags in a safe local environment), runs in a timed subprocess, and captures stdout/stderr and elapsed wall-clock time.
  - For Python: uses `time.perf_counter()` over many repeated runs to average and reduce noise; isolates runs using subprocess to avoid interpreter state effects.
  - Fallback: if compilation or execution is disallowed by environment, returns `proxy` and a predicted runtime computed from ML predictor or heuristics.

- `phase2_features.py` — Tree‑sitter feature engineering and legacy vector.
  - Uses `tree_sitter` to parse source code into an AST.
  - Computes features such as: `num_functions`, `avg_cyclomatic_complexity`, `max_loop_depth`, `num_arithmetic_ops`, `num_memory_accesses`, `num_function_calls`, `avg_stmt_length`, `num_conditionals`, `estimated_work_per_loop` (the 9-feature legacy order).
  - Exports:
    - `analyze_code_features(code_text) -> dict` — rich feature dictionary.
    - `legacy_model_vector(code_text) -> np.ndarray(shape=(9,))` — deterministic 9-feature vector for RF compatibility.
  - Notes: For robustness, there is a non-Tree‑sitter fallback (lightweight regex-based parser) if the environment lacks the pinned `tree-sitter` version.

- `phase3_ast_gnn.py` — AST → graph, dataset, model, training loop.
  - Main capabilities:
    - `build_graph_from_ast(tree)` — nodes for AST node types, edges for parent-child and next-sibling relationships, node features like node type id, token density, and local metrics (subtree size, depth).
    - PyTorch `Dataset`/`DataLoader` compatible dataset that returns graph objects (node feature tensors, edge lists, global attributes) and regression targets.
    - GNN encoder: multi-layer GraphSAGE-like message passing with ReLU + batchnorm, global pooling (mean + max concatenation), then an MLP regressor head.
    - Training harness supports configurable: learning rate, weight decay, epochs, early stopping, and optional carbon-aware loss weighting.
  - Checkpointing: saves `phase3_model.pth` containing `state_dict` and metadata (vocab sizes, model args) to enable robust re-loading.

- `scripts/eval_refactor_candidates.py` — offline evaluation pipeline.
  - Generates candidates (LLM + heuristics), extracts features, predicts energy via Phase‑1/Phase‑2 RF models, optionally measures runtime via `runtime_harness`, converts to carbon using `carbon_providers`, and writes a JSON report `scripts/eval_refactor_candidates_output.json`.
  - Enforces that LLM candidates pass the `has_valid_function_body()` check; otherwise marks them as invalid/failed.

- `scripts/*` — other helper scripts.
  - `evaluate_phase3_saved.py` — loads `phase3_model.pth`, builds test graphs, and reports MAE/RMSE vs `phase2_model.pkl` and a random baseline.
  - `train_phase3.sh` or `train_phase3.py` — possible convenience scripts for distributed runs or scheduled training.

- `eco_logic_synthetic_benchmark.csv` — synthetic benchmark dataset used for prototyping and model validation.

- Model artifacts (not always committed):
  - `phase1_model.pkl`, `phase2_model.pkl`, `phase3_model.pth` — store in a model store or release if large.

**6. Validation and safety checks**

- LLM outputs must pass `has_valid_function_body()` and a set of unit checks (compiles for C++ snippets, basic static AST-based equivalence checks where applicable).
- `validate_feature_vector_9()` ensures legacy RF compatibility; if invalid, code reverts to a fallback heuristic or flags the candidate.

**7. Model training & evaluation details**

- Phase‑1/2 (RandomForest):
  - Input: 9 deterministic features per sample.
  - Targets: measured energy (J) per sample, derived from runtime_ms × measured/assumed power, or directly measured power when available.
  - Training details: default scikit-learn RandomForestRegressor, typical hyperparameters: n_estimators=200, max_depth tuned via cross-val.

- Phase‑3 (AST‑GNN):
  - Graph construction: each AST node has categorical node-type embeddings and continuous local stats.
  - Message passing: GraphSAGE-style aggregator (mean) with 2–3 layers; dropout between layers.
  - Global pooling: concatenation of mean and max pooled node embeddings.
  - Loss: MSE on energy targets. Optionally weighted by carbon impact if `--carbon-weight` is provided during training.
  - Evaluation: MAE and RMSE on held-out test set; ablations compare with RandomForest baseline using the same splits.

**8. Reproducibility & deterministic behavior**

- The legacy 9-feature `legacy_model_vector()` is intentionally deterministic — same order and normalization to produce identical inputs for RF models.
- Seeds used for PyTorch and NumPy are set when running training scripts; checkpoint metadata stores seeds and preprocessing transforms.

**9. Known limitations & next steps**

- LLMs sometimes produce functionally identical or empty refactors; strict prompting and post-validation reduce false positives but cannot fully guarantee semantic improvement.
- Carbon intensity provider quality varies by region and time; offline caching recommended for reproducible experiments.
- Future work: add dynamic power measurement (RAPL or external power meters), integrate finer-grained per-line instrumentation, or extend AST graphs with dataflow edges for better GNN performance.

**10. Developer workflow & recommended commands**

- Run evaluation and inspect JSON output:

```bash
PYTHONPATH=. python scripts/eval_refactor_candidates.py
less scripts/eval_refactor_candidates_output.json
```

- Run the Streamlit UI:

```bash
streamlit run app.py
```

- Train Phase‑3 locally:

```bash
python phase3_ast_gnn.py --data-file data/train_dataset.parquet --epochs 40 --save-path phase3_model.pth
```

**11. FAQ and troubleshooting**

- Q: My `phase3_model.pth` won't load — mismatch in vocab sizes.
  - A: The loader reads `state_dict` metadata and reconstructs the architecture. If you changed `node_type_vocab_size`, rebuild the vocab or retrain.

- Q: Tree‑sitter parse fails in my environment.
  - A: The repo pins `tree-sitter<0.22.0`; install the pinned version or use the regex fallback in `phase2_features.py`.

- Q: Measurements inconsistent across runs.
  - A: Use the harness's repeated-run averaging and disable background processes; run in isolated env or docker container for better reproducibility.

**12. Glossary (short)**

- AST: Abstract Syntax Tree.
- GNN: Graph Neural Network.
- RF: Random Forest.
- J: Joules (energy).
- gCO2eq: grams of CO2-equivalent.

---

If you want, I can:

- Generate a `requirements.txt` with pinned versions used during development.
- Add `CONTRIBUTING.md` and `LICENSE` templates.
- Create a short test verifying the `early_exit_bubble` heuristic and integrate it into `tests/`.

Tell me which of the above you'd like next and I will implement it.

---

Appendix: Exhaustive reproducible specification (for LLM reconstruction)
---------------------------------------------------------------

This appendix contains a near line-by-line, deterministic specification of the repository layout, file contents' structure, function signatures, data schemas, exact feature ordering, model hyperparameters, serialization keys, LLM prompts, and the minimal implementation details required for an LLM to re-generate this project with functional parity.

Use this as a canonical spec. The goal: given only this appendix, an LLM should be able to produce code that, when run with reasonable environment (Linux/Windows, Python 3.10+), reproduces the same behavior and outputs.

Repository layout (paths are workspace-root relative):

- `app.py` — Streamlit UI and orchestrator.
- `groq_client.py` — LLM wrapper.
- `carbon_providers.py` — providers and adapters.
- `runtime_harness.py` — measurement harness.
- `phase2_features.py` — Tree-sitter feature extractor and `legacy_model_vector()`
- `phase3_ast_gnn.py` — graph builder, dataset, model, training.
- `scripts/eval_refactor_candidates.py` — candidate generator & evaluator.
- `scripts/eval_refactor_candidates_output.json` — example output (JSON schema below).
- `evaluate_phase3_saved.py` — loader & evaluator for `phase3_model.pth`.
- `eco_logic_synthetic_benchmark.csv` — CSV dataset with columns described below.
- `phase1_model.pkl`, `phase2_model.pkl` — scikit-learn pickled RandomForest objects (serialized via `joblib.dump(model)`).
- `phase3_model.pth` — PyTorch checkpoint saved with `torch.save({'state_dict': model.state_dict(), 'meta': {...}}, path)`.

Language/runtime constraints and pinned versions
----------------------------------------------

- Python: 3.10+ recommended. Ensure consistent `pip` and `venv` usage.
- Core Python packages (explicit minimal set; versions are suggestions but please pin when reproducing):
  - `streamlit>=1.20.0`
  - `plotly>=5.6.0`
  - `scikit-learn>=1.1.0`
  - `torch>=1.13.0` and `torchvision` matching the torch version
  - `tree_sitter<0.22.0` (the repo uses `tree-sitter-languages` style AST bindings; pinned due to API differences)
  - `pandas>=1.3.0`, `numpy>=1.22.0`

File-level deterministic contract
-------------------------------

All modules expose the following functions/classes with the exact signatures below (types indicated in comments):

1) phase2_features.py

def analyze_code_features(code_text: str, language: str = 'cpp') -> dict:
    """Return a rich feature dict for `code_text` in `language`.

    Keys (exact):
      - 'num_functions': int
      - 'num_loops': int
      - 'max_loop_depth': int
      - 'avg_loop_body_size': float
      - 'num_conditionals': int
      - 'num_function_calls': int
      - 'num_arithmetic_ops': int
      - 'num_memory_accesses': int
      - 'avg_stmt_length': float
      - 'cyclomatic_complexity': float
      - 'halstead_operands': float
      - 'halstead_operators': float
      - 'tokens': int
      - 'lines': int

    Notes: numeric values should be Python `int` or `float` and deterministic for the same input string.
    """

def legacy_model_vector(code_text: str, language: str = 'cpp') -> list:
    """Return the deterministic 9-feature vector as a Python list (length 9).

    The exact ordering (index -> feature) is required by the RF baselines and must be:
      0: num_functions (int)
      1: max_loop_depth (int)
      2: num_loops (int)
      3: num_arithmetic_ops (int)
      4: num_memory_accesses (int)
      5: num_function_calls (int)
      6: num_conditionals (int)
      7: avg_stmt_length (float)
      8: cyclomatic_complexity (float)

    The function must guarantee:
      - Return type: list[float] (convert ints to floats)
      - Length exactly 9
      - Deterministic mapping from input text to numbers
    """

Implementation details for `phase2_features.py` (reproducible algorithm):

- Primary parser: Tree‑sitter. Build a language parser for `cpp` using `Language` and `Parser` from `tree_sitter`:

  from tree_sitter import Language, Parser
  CPP_LANG = Language('build/my-languages.so', 'cpp')
  parser = Parser()
  parser.set_language(CPP_LANG)

- Parse code to AST: `tree = parser.parse(bytes(code_text, 'utf8'))`
- Walk AST breadth-first using `node.walk()` collecting node types. For loops, detect node types matching `for_statement`, `while_statement`, `do_statement` (exact strings used by tree-sitter C++ grammar). Count `num_loops` and compute `max_loop_depth` by traversing nesting.
- For `num_function_calls`, look for `call_expression` nodes and increment.
- For `num_arithmetic_ops`, count operator tokens: `+`, `-`, `*`, `/`, `%`, `+=`, etc. (walk nodes and inspect `node.type` and `node.children` tokens).
- For memory accesses, count array subscript expressions and pointer dereference (`subscript_expression`, `field_expression`/`->` tokens) and increments for `num_memory_accesses`.
- `avg_stmt_length`: compute average token count across `expression_statement` and `declaration` statement nodes.
- `cyclomatic_complexity`: approximate by counting conditional branches (`if_statement`, `switch_statement`, `case`) plus `num_loops` plus 1.
- Halstead metrics: compute `halstead_operands` and `halstead_operators` by maintaining two sets: operands (identifiers, literals) and operators (token types like `+`, `-`, `*`, `=`, `==`, `!=`, `<`, `>`, etc.) and returning their counts (floating point).

- Fallback parser: if `tree_sitter` import or the `CPP_LANG` library isn't available, use regex heuristics in this order (they must be deterministic):
  - num_functions = count regex patterns: `\b(?:int|void|float|double|long|char)\s+\w+\s*\([^;]*\)\s*\{`
  - num_loops = count occurrences of `\bfor\b|\bwhile\b|\bdo\b`
  - num_function_calls = count `\w+\s*\(` (careful: filter out function definitions by requiring no return type before the name)
  - num_conditionals = count `\bif\b|\bswitch\b`
  - num_arithmetic_ops = count `[+\-*/%]` tokens not inside comments or strings (naive but deterministic)

All numeric normalization (if any) must be deterministic; the code should not call external randomness.

2) runtime_harness.py reproducible contract

def measure_runtime(code_text: str, language: str, repetitions: int = 20, time_limit_s: float = 5.0) -> dict:
    """Attempt to compile and/or execute `code_text` safely and return a dict with keys:
      - 'runtime_ms': float (average over repetitions)
      - 'runtime_mode': 'measured' or 'proxy'
      - 'stdout': str
      - 'stderr': str
      - 'success': bool

    Behavior:
      - For `cpp`: write `code_text` to a temp file `snippet.cpp`; compile using `g++ -O2 -std=c++17 -o /tmp/snippet_bin snippet.cpp`.
        If compilation fails, return `success: False` and `runtime_mode: 'proxy'` along with compiler stderr.
      - Run the compiled binary `repetitions` times in a subprocess measuring wall-clock using `time.perf_counter()` around `subprocess.run([...], timeout=...)` and record elapsed.
      - For `python`: write `code_text` to `snippet.py` and run `python -u snippet.py` in subprocess for `repetitions` times.
      - Average runtime_ms = (sum elapsed_ms) / repetitions.
      - If any run exceeds `time_limit_s`, abort and return `success: False, runtime_mode: 'proxy'`.

    The function must never execute untrusted code in-process; always use subprocesses and temp directories, and ensure environment variables passed are minimal. When the environment cannot run the snippet (e.g., `g++` not present), return `runtime_mode: 'proxy'` and `success: False`.
    """

Proxy runtime computation (if measured runs cannot be executed):
- Call `legacy_model_vector()` and pass the vector to the loaded RF model (e.g., `phase2_model.pkl`) to get a predicted energy (J) and convert to runtime_ms by dividing by an assumed average power (e.g., 15.0 Watts). This mapping MUST be used consistently in the repo: `assumed_power_watts = 15.0`.

3) carbon_providers.py reproducible contract

Classes:
  class ElectricityMapsAdapter:
      def __init__(self, api_key: str): ...
      def get_intensity(self, location: str, timestamp: Optional[datetime] = None) -> float:
          """Return `gCO2eq_per_kWh` as float. """

  class WattTimeAdapter:
      def __init__(self, token: str, ba: str): ...
      def get_intensity(self, location: str, timestamp: Optional[datetime] = None) -> float: ...

  class OfflineFallbackAdapter:
      def __init__(self, fallback_value_gco2_per_kwh: float = 450.0): ...
      def get_intensity(self, ...) -> float: return fallback.

Utility function:
  def resolve_carbon_reading(location: str, timestamp: Optional[datetime] = None, preferred: List[str] = None) -> float:
      """Try each adapter in order and return the first successful reading; otherwise return fallback value 450.0 gCO2eq/kWh."""

4) groq_client.py and LLM prompts

Function signature:
  def generate_refactor(code_text: str, language: str = 'cpp', role: str = 'refactor') -> dict:
      """Return a dict: {'success': bool, 'refactor_code': str, 'metadata': {...}}

LLM prompt (exact reproducible prompt used to coax algorithmic implementations):

"""
You are an assistant that must provide a single, fully implemented function that improves the algorithmic performance of the provided code.

Rules (must be enforced exactly):
- Do NOT use library or built-in shortcuts that trivially replace the algorithm (for example: std::sort, sorted(), numpy.dot(), std::accumulate). Your submission must include an algorithmic implementation written out in full.
- Return only the function body and any helper functions necessary. Do not include commentary, markdown, or extraneous text.
- Use only the same language as the input (C++ for .cpp inputs). Use C++17 dialect when relevant.
- Keep variable naming clear and consistent. If the input function is named `bubbleSort`, you may return `bubbleSortOptimized` and include both function signatures if necessary for clarity.
- The output must compile in a standard environment (g++ for C++). If compilation would require additional scaffolding, include it as comments but ensure the function itself compiles.

Input:
```
{code_text}
```

Return:
A JSON-like object with keys `refactor_code` (string) and `notes` (optional). Only return the code string as plain text when called via the wrapper.
"""

The wrapper must post-process LLM outputs: strip leading/trailing whitespace, remove non-code preambles, and verify `has_valid_function_body()` before accepting.

5) scripts/eval_refactor_candidates.py JSON schema output

The script must save a JSON array to `scripts/eval_refactor_candidates_output.json` with elements shaped exactly as follows:

{
  "candidate_id": "string",            # unique id, e.g., 'heuristic_early_exit_bubble'
  "label": "string",                  # human label like 'early_exit_bubble'
  "code": "string",                   # full code text of the candidate
  "features": [f0, f1, ..., fN],       # full rich features (not just legacy 9) as an array
  "legacy_vector": [v0..v8],           # exactly 9 float numbers
  "predicted_energy_j": float,         # predicted energy in joules by RF/Phase2
  "predicted_runtime_ms": float,       # proxy runtime derived from predicted_energy_j / assumed_power
  "measured_runtime_ms": float|null,   # average measured runtime in ms or null if not measured
  "runtime_mode": "measured"|"proxy",
  "carbon_gco2eq": float,              # predicted_energy_j in J -> kWh -> multiply by intensity
  "notes": "string"                  # optional notes, e.g., 'llm invalid: no function body'
}

Example conservative constants used in codebase (must be reproduced):
- `ASSUMED_POWER_WATTS = 15.0`  # used to convert energy (J) -> runtime_ms when proxying
- `JOULES_PER_KWH = 3.6e6`

6) phase3_ast_gnn.py exact model spec

Public classes and functions:

class ASTGraphBuilder:
    def __init__(self, node_type_vocab: Optional[Dict[str,int]] = None):
        # node_type_vocab: mapping from node type string to contiguous ids starting at 0
    def build(self, code_text: str, language: str = 'cpp') -> dict:
        """Return graph dict:
           {
             'num_nodes': int,
             'node_type_ids': List[int],        # shape [num_nodes]
             'node_features': List[List[float]],# shape [num_nodes, feature_dim]
             'edge_index': [[src_idx,...],[dst_idx,...]], # 2 x num_edges
             'global_features': List[float]     # optional
           }
        """

PyTorch model specification (names and shapes):

class ASTGNNRegressor(torch.nn.Module):
    def __init__(self, node_type_vocab_size: int, node_feature_dim: int = 8, hidden_dim: int = 128, num_layers: int = 3):
        # Embedding for node types: nn.Embedding(node_type_vocab_size, node_feature_dim)
        # Input MLP to project concatenated [embedding, continuous node features] -> hidden_dim
        # num_layers GraphSAGE-style message passing layers: each maps hidden_dim -> hidden_dim
        # Global pooling: mean and max across node embeddings, concatenated -> 2 * hidden_dim
        # Final MLP regressor: Linear(2*hidden_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 1)

    def forward(self, graph_batch) -> torch.Tensor:
        # graph_batch must supply batched node features, edge_index in COO format, batch vector mapping nodes->graph
        # returns tensor shape [batch_size, 1]

Training loop (deterministic essentials):
  - seed = 42 (use torch.manual_seed(seed), numpy.random.seed(seed), random.seed(seed))
  - optimizer: Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
  - loss: MSELoss
  - batch_size: 32
  - scheduler: ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
  - epochs: configurable, default 40
  - Save best checkpoint by validation MAE; checkpoint dict must include:
      {
        'state_dict': model.state_dict(),
        'meta': {
           'node_type_vocab_size': node_type_vocab_size,
           'node_feature_dim': node_feature_dim,
           'hidden_dim': hidden_dim,
           'num_layers': num_layers,
           'seed': seed,
           'preprocessing': { ... } # e.g., normalization means and stds
        }
      }

7) evaluate_phase3_saved.py expected behavior

- Load `phase3_model.pth` using `torch.load(path, map_location='cpu')`.
- Inspect `meta` key and reinstantiate `ASTGNNRegressor` with matching args.
- Load test graphs with same preprocessing and compute predictions; report MAE and RMSE vs ground truth energy_j.

8) example data formats

- `eco_logic_synthetic_benchmark.csv` columns (exact):
  - `id` (string)
  - `label` (string)
  - `language` (string) e.g., 'cpp'
  - `code` (string) raw source
  - `measured_runtime_ms` (float) nullable
  - `measured_power_watts` (float) nullable
  - `energy_j` (float) nullable  # if measured_power and runtime available
  - `notes` (string)

- `scripts/eval_refactor_candidates_output.json` element example (JSON):

{
  "candidate_id": "heuristic_early_exit_bubble",
  "label": "early_exit_bubble",
  "code": "void bubbleSortOptimized(int arr[], int n) { /* ... */ }",
  "features": {"num_functions":1, "num_loops":2, ...},
  "legacy_vector": [1,2,2,34,12,3,4,12.5,5.0],
  "predicted_energy_j": 0.0032351306917105566,
  "predicted_runtime_ms": 47.6625,
  "measured_runtime_ms": 0.392,
  "runtime_mode": "measured",
  "carbon_gco2eq": 0.0009,
  "notes": "heuristic measured fast"
}

9) exact normalization & numeric constants

- Energies are stored and predicted in Joules. When converting from runtime_ms to Joules during dataset creation, use: energy_j = (runtime_ms / 1000.0) * measured_power_watts.
- When measured_power_watts is not available, assume `ASSUMED_POWER_WATTS = 15.0`.
- Convert J to kWh for carbon: kwh = energy_j / 3.6e6; carbon_g = kwh * intensity_g_per_kwh.

10) Git/GitHub workflow reproducibility notes

- The repo follows a feature-branch workflow. Example commit message style used in the project:
  - `type(scope): subject` where `type` in {feat, fix, chore, docs, test}.
- Use `git` commands shown in `README.md` for creating branches and pushing.

11) Unit tests and acceptance tests expected in `tests/` (not committed here but required to reproduce behavior)

- tests/test_phase2_features.py: assert `legacy_model_vector(sample_code)` returns expected 9-length vector for given sample code string (provide 3 sample fixtures: tiny loop, nested loops, arithmetic-heavy function).
- tests/test_runtime_harness.py: run `measure_runtime` on a trivial C++ program that just loops 10000 times and assert `runtime_mode` in {'measured','proxy'} and `runtime_ms` > 0 when measured.
- tests/test_phase3_load_save.py: save a small model checkpoint and reload to ensure shapes and `meta` present.

12) Reproducible prompt history and LLM safety enforcement

- The exact text of the LLM instruction (see section 4) must be embedded in `groq_client.py` as `REFRACTOR_PROMPT_TEMPLATE` string constant.
- Post-processing: after receiving LLM text, run `strip()` and then remove any leading lines until the first line that matches a function signature regex appropriate for `language`. For C++, accept signatures matching `^\w[\w\s\*&:<>]*\s+\w+\s*\([^)]*\)\s*\{`.

13) Example minimal implementation snippets (pseudocode for critical parts)

- legacy_model_vector pseudocode:

def legacy_model_vector(code_text, language='cpp'):
    feats = analyze_code_features(code_text, language)
    v = [
        float(feats['num_functions']),
        float(feats['max_loop_depth']),
        float(feats['num_loops']),
        float(feats['num_arithmetic_ops']),
        float(feats['num_memory_accesses']),
        float(feats['num_function_calls']),
        float(feats['num_conditionals']),
        float(feats['avg_stmt_length']),
        float(feats['cyclomatic_complexity'])
    ]
    assert len(v) == 9
    return v

- energy prediction using RF:

from joblib import load
rf = load('phase2_model.pkl')
def predict_energy_from_legacy(vec9):
    import numpy as np
    x = np.array(vec9, dtype=float).reshape(1, -1)
    return float(rf.predict(x)[0])

14) Training dataset creation (how to compute labels)

- For each code sample in the dataset:
  - Measure or obtain `measured_runtime_ms` and `measured_power_watts`.
  - Compute `energy_j = (measured_runtime_ms / 1000.0) * measured_power_watts`.
  - Extract graph with `ASTGraphBuilder.build(code_text)` and store node features and edge_index in a serialized parquet or custom binary file. Include `energy_j` as the target.

15) Provenance and metadata

- Every artifact saving step must include a `meta` dict with: `created_by` (string, e.g., 'phase3_trainer_v1'), `created_at` (ISO8601 UTC), `git_commit` (short sha if available), and `seed`.

16) Example end-to-end run to reproduce main experiment (commands)

```bash
# create env
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# (optional) prepare tree-sitter language bindings
python scripts/build_tree_sitter_parsers.py

# evaluate candidates and produce JSON
PYTHONPATH=. python scripts/eval_refactor_candidates.py

# run the streamlit UI
streamlit run app.py

# train phase3 model (if you have preprocessed data)
python phase3_ast_gnn.py --data-file data/graphs_train.parquet --epochs 40 --save-path phase3_model.pth

# evaluate saved model
python evaluate_phase3_saved.py --model-path phase3_model.pth
```

17) Closing notes

This appendix is intentionally prescriptive; it supplies exact function names, signatures, numeric constants, JSON schemas, prompt text, and architecture hyperparameters. Use it as the canonical spec. If you'd like, I will:

- Generate the `requirements.txt` pinned to exact versions used in my runs.
- Implement the test fixtures and add `tests/` with the three unit tests above.
- Produce `scripts/build_tree_sitter_parsers.py` to compile the language bindings used by `phase2_features.py`.

Tell me which of those you want next and I'll add them to the repo.

18) UI state model and candidate selection rules

The Streamlit app must preserve the following conceptual state keys in `st.session_state`:

- `input_code`: str, current source snippet in the editor.
- `language`: str, default `'cpp'`.
- `generated_llm_code`: str, latest accepted LLM output.
- `heuristic_candidates`: list[dict], list of local heuristic refactors.
- `evaluation_results`: list[dict], JSON-like records returned by the evaluation pipeline.
- `selected_candidate_id`: str|None, currently selected candidate in the UI.
- `accepted_candidate_code`: str|None, code shown as chosen refactor.
- `selected_carbon_provider`: str, one of `'electricitymaps'`, `'watttime'`, `'offline'`.
- `use_measured_runtime`: bool, whether to prefer measured runtime when available.
- `latest_pareto_points`: list[dict], data records for the current chart.

Candidate selection rules (exact behavior):

- Generate one LLM candidate and a small fixed set of handcrafted heuristic candidates.
- Reject any LLM candidate that fails `has_valid_function_body()`.
- Prefer candidates with `runtime_mode == 'measured'` over proxy runtime when available.
- Among candidates with measured runtime, choose the one with smallest measured runtime if the user asks for runtime-optimal refactor.
- If energy-optimal is requested, choose the minimum `predicted_energy_j` after validation.
- If the user asks for Pareto selection, include all non-dominated candidates on the energy/runtime frontier.

19) Heuristic candidate definitions

The repository includes a deterministic heuristic candidate generator for the bubble sort example and similar low-level loops. The following candidate implementations are conceptually present and must be reproducible:

- `early_exit_bubble`:
  - Same outer/inner loop structure as bubble sort.
  - Adds a `swapped` boolean per outer pass.
  - Breaks early when no swaps occur in a full pass.
  - Must be fully implemented, not replaced with a library sort.

- `std_sort` comparison candidate:
  - Used only as a baseline comparison in evaluation.
  - In the final UI acceptance workflow, library-shortcut candidates are not to be presented as the accepted refactor if a fully implemented alternative is available and valid.

- `llm_refactor`:
  - Generated via the prompt template in section 4.
  - Must be validated before acceptance.

When the input program is a sorting implementation, the evaluator should prefer algorithmic improvements such as early exit, gap-based passes, hybrid insertion sort for small partitions, or branch-reduction, provided they are fully implemented in code.

20) Pareto frontier construction

The plot in `app.py` uses the following deterministic logic:

- Each candidate becomes a point with x-axis = runtime_ms and y-axis = energy_j.
- If `measured_runtime_ms` is present and `runtime_mode == 'measured'`, use that value for x-axis; otherwise use `predicted_runtime_ms`.
- The frontier is the set of non-dominated points where no other point has both lower or equal runtime and lower or equal energy with at least one strict improvement.
- Hover fields must include at least:
  - `label`
  - `candidate_id`
  - `runtime_mode`
  - `predicted_energy_j`
  - `measured_runtime_ms`
  - `predicted_runtime_ms`
  - `carbon_gco2eq`
  - `language`

21) Baseline model compatibility rules

The RF baseline must keep backward compatibility with the original 9-feature pipeline:

- Input array shape is `(1, 9)` or `(n, 9)`.
- Feature ordering is immutable and must never be changed without retraining the model.
- Feature scaling: if the trained model expected raw counts, do not normalize at inference time.
- Missing features are not permitted; if `legacy_model_vector()` cannot be formed, reject the candidate rather than padding.

22) Saved model loading rules

When loading `phase3_model.pth`:

- Load the checkpoint on CPU first.
- Read `checkpoint['meta']` before model construction.
- Construct `ASTGNNRegressor` with `node_type_vocab_size`, `node_feature_dim`, `hidden_dim`, and `num_layers` from `meta`.
- Then call `load_state_dict(checkpoint['state_dict'], strict=True)`.
- If keys mismatch, print a diagnostic that names the missing/unexpected keys and abort loading rather than silently ignoring them.

23) Data ingestion rules

The benchmark CSV and training artifacts are treated as immutable input data. The project expects:

- CSV rows to have stable IDs.
- Source code stored as raw text in a single field.
- Runtime and power measurements to be non-negative when present.
- Any derived label generation to be deterministic from the raw measurements and constants in section 9.

24) Minimal file generation order for reconstruction

If an LLM is reconstructing the project from scratch, generate files in this order to avoid import-cycle failures:

1. `phase2_features.py`
2. `carbon_providers.py`
3. `runtime_harness.py`
4. `groq_client.py`
5. `phase3_ast_gnn.py`
6. `scripts/eval_refactor_candidates.py`
7. `evaluate_phase3_saved.py`
8. `app.py`

Reason: the UI imports the helpers, and the scripts depend on the feature extractor and runtime harness.

25) What a faithful reproduction must do on first run

- Start Streamlit without import errors.
- Load environment variables if a `.env` is present.
- Accept a code snippet and evaluate at least one heuristic candidate.
- Produce a JSON-like evaluation report.
- Show a Pareto frontier with runtime and energy axes.
- If a Phase‑3 checkpoint exists, load it and use it for inference; otherwise fall back to Phase‑2 RF.

26) Explicit non-goals

To avoid ambiguity, the project does not require:

- Full compiler-grade semantic equivalence checking.
- Perfect power-meter integration on every platform.
- Remote execution of user code in production.
- A web backend beyond Streamlit.

27) Reconstruction checklist

To reproduce this project, ensure the generated code has:

- Exact file names listed above.
- A deterministic 9-feature legacy path.
- Tree-sitter-backed richer feature extraction with fallback.
- Measured runtime integration.
- Carbon provider abstraction.
- RF baseline support and AST-GNN support.
- Strict LLM prompt and post-validation.
- Pareto frontier rendering in the UI.


