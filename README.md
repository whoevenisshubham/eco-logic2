**EcoLogic**

Introduction
============

EcoLogic is a lightweight tool for exploring how small changes to program code affect execution time, estimated energy consumption, and approximate carbon emissions. It is intended to make optimization outcomes accessible to non-technical stakeholders while also providing a modest amount of developer-facing information for reproducibility and local use.

Overview (User-facing)
----------------------

At a high level, EcoLogic does the following when you run an analysis session:

1. Accepts a single source file or a project (folder/zip/repo) as input.
2. Generates candidate refactors using built-in heuristics and, optionally, an LLM assistant.
3. Scores candidates using a fast energy predictor and a proxy runtime model.
4. When possible, performs measured runtime profiling for the original and the best candidate.
5. Produces an easy-to-read dashboard with numeric summaries and a PDF certificate you can download.

Why this matters to non-developers
---------------------------------

- Energy and carbon numbers give a concise, comparable metric for software-related efficiency improvements.
- The PDF certificate provides a sharable artifact for reporting and audits.
- You don't need to be a developer to see whether a proposed change is likely to improve performance or reduce energy.

Quick User Walkthrough
----------------------

1. Open the app in your browser (it runs as a local web dashboard).
2. Provide the code to analyze:
	- Paste the source file into the main editor area, or
	- Use the workspace intake panel to scan a local folder, upload a ZIP, or provide a Git URL.
3. Configure simple knobs in the sidebar (Input scale N, cores, and electricity intensity) if desired.
4. Click the "Run closed-loop optimization" button.
5. Inspect the dashboard: energy estimates, measured runtime (if available), Pareto chart, and rounds table.
6. Download the PDF certificate if you want to share the result.

User-facing UI Elements Explained
--------------------------------

- Source code editor: paste or edit the original code you want to score.
- LLM output box: optionally paste a suggested refactor from any external tool; the app can also call the integrated assistant.
- Run button: starts the candidate generation, evaluation, and profiling loop.
- Metrics tiles: show energy (J), delta, estimated carbon, and loop runtime.
- Pareto plot: benchmark points vs. original vs. optimized candidate.
- Certificates panel (sidebar): lists previously generated certificates and allows downloads.

Certificates and Evidence
-------------------------

Each run can produce a small PDF certificate summarizing:

- Generated at (timestamp), project/file, input settings (N, cores, TDP), and carbon intensity used.
- Tabulated numeric comparisons (energy, proxy runtime, measured runtime if available).
- A short code excerpt for the original and the selected best candidate.
- High-level selection reason (e.g., lower predicted energy, measured runtime improvement).

Certificates are saved locally under the `certificates/` directory, and you can also download them directly from the UI.

Interpreting the Numbers
--------------------------------------------

- Treat measured runtime as stronger evidence than model-based proxy runtime when available.
- Small energy differences (a few percent) are noisy with model-based predictions; prefer measured runtime differences greater than a few percent for operational changes.
- Use the Pareto chart to compare both time and energy; a point lower-left of another is strictly better for both dimensions.

Short Example Scenario
--------------------------------------

Imagine you run the tool on a sorting function used in a data pipeline. The app generates an optimized candidate that replaces a loop-based sort with a standard library sort. The app will show:

- Predicted energy decreased by X J (model estimate).
- Measured runtime (if available) decreased from A ms to B ms.
- The certificate will state that the selected candidate was chosen for lower predicted energy and confirmed by a measured speedup of Y%.

Developer / Setup Information
-------------------------------------

The following section contains minimal developer/setup steps for local use. Non-developers can skip this section.

1) Python runtime

- EcoLogic runs on Python 3.10+ but is commonly used with Python 3.11. Use a virtual environment for local installs.

2) Basic installation

From the repository root, the minimal steps are:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1    # Windows PowerShell
pip install -r requirements.txt
```

If you prefer POSIX shells (macOS / Linux):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- The app uses `streamlit` for the UI and a small scikit-learn RandomForest as a fallback predictor.
- If you cannot install dependencies, the app still functions in a limited form: the UI will still accept pasted code and show stored artifacts, but runtime profiling or LLM calls will be disabled or limited.

3) .NET (optional, for C# project profiling)

If you want the app to run measured profiles for C#/.NET projects, install the .NET SDK (recommended 8.x or later). On Windows use the official installer or package manager; ensure the `dotnet` CLI is on your PATH.

If `dotnet` is not available, the app will fall back to model-based estimates for .NET code.

4) Environment variables (optional)

- `GROQ_API_KEY` — optional API key for the integrated LLM assistant (if you configure an external LLM provider).
- `ELECTRICITY_MAPS_API_KEY` — optional for fetching live carbon intensity from Electricity Maps.
- `WATTTIME_USERNAME` / `WATTTIME_PASSWORD` — optional for WattTime.

If no API keys are provided the app will use a fallback static intensity value that you can adjust in the UI sidebar.

5) Running the app locally

Start Streamlit from the repository root:

```powershell
streamlit run app.py --server.port 8504
```

Open the URL shown in the terminal (usually `http://localhost:8504`).

6) Files and layout (what's important)

- `app.py` — main Streamlit application and orchestration.
- `runtime_harness.py` — code that builds, executes, and measures runtime for language-specific snippets.
- `project_ingestion.py` — scans folders, ZIPs, and repos to detect target files and .NET projects.
- `groq_client.py` — optional wrapper for calling an LLM provider to request refactor candidates.
- `eco_logic_synthetic_benchmark.csv` — a small benchmark dataset used to train the fallback model if no saved model artifact exists.
- `certificates/` — folder where generated PDFs are stored.

7) Training a fallback model (developer note)

If the application does not find a saved model artifact it will attempt to train a fallback RandomForest from the CSV benchmark. This is done automatically on first run and produces a modest-sized model suitable for interactive use.

8) Extending or debugging (developer note)

- The app uses small, test-friendly components so you can modify evaluation heuristics in `app.py` or add new language support in `runtime_harness.py`.
- To validate Python syntax quickly, run:

```powershell
python -m py_compile app.py
```

- If you add new dependencies, update `requirements.txt` so local setup remains reproducible.

Troubleshooting (practical tips)
-------------------------------

- If measured runtimes are failing for C#: verify `dotnet --version` runs and returns a version.
- If the LLM assistant returns no output, check that any API key is loaded into environment variables (or paste LLM output into the LLM output box manually).
- If the certificate download button is missing, check the app UI panel labeled "Certificates" in the sidebar — generated PDFs are stored there.

Privacy & Security
--------------------------

- EcoLogic runs locally by default and analyzes only files you provide. If you configure an external LLM service, be mindful of what source code you transmit to that service.