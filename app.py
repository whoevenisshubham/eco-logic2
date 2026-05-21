import math
import os
import re
import time
from typing import Dict, List, Tuple

import joblib
import ast
import load_env
from groq_client import generate_code_explanation, generate_refactor
from explainability import LEGACY_FEATURE_NAMES, ast_to_dot, build_ast_preview, compute_shap_summary
from runtime_harness import measure_runtime
from feature_engineering import analyze_code_features, legacy_model_vector, rich_feature_rows
from carbon_providers import (
    CarbonProviderError,
    DEFAULT_MAHARASHTRA_ZONE,
    DEFAULT_OFFLINE_INTENSITY,
    resolve_carbon_reading,
)
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Cool Ecologic",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load environment from .env if present
try:
    load_env.load()
except Exception:
    # non-fatal; environment may be set in shell
    pass


@st.cache_resource
def load_model():
    model_files = [
        "baseline_rf_model.pkl",
        "high_precision_energy_model.pkl",
        "energy_predictor_model.pkl",
    ]
    for filename in model_files:
        if os.path.exists(filename):
            try:
                model = joblib.load(filename)
                return model, filename
            except Exception:
                continue
    return None, None


@st.cache_data
def load_dataset():
    dataset_files = [
        "eco_logic_synthetic_benchmark.csv",
        "eco_logic_ast_dataset.csv",
    ]
    for filename in dataset_files:
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                return normalize_dataset(df), filename
            except Exception:
                continue
    return None, None


def normalize_dataset(df):
    out = df.copy()
    defaults = {
        "algorithm_class": "unknown",
        "source_code": "def solve(arr):\n    return arr",
        "input_scale_N": 10000,
        "hardware_tdp": 45.0,
        "hardware_cores": 4,
        "target_energy_joules": 1.0,
    }
    for col, default_val in defaults.items():
        if col not in out.columns:
            out[col] = default_val
    out["input_scale_N"] = pd.to_numeric(out["input_scale_N"], errors="coerce").fillna(10000)
    out["hardware_tdp"] = pd.to_numeric(out["hardware_tdp"], errors="coerce").fillna(45.0)
    out["hardware_cores"] = pd.to_numeric(out["hardware_cores"], errors="coerce").fillna(4)
    out["target_energy_joules"] = pd.to_numeric(
        out["target_energy_joules"], errors="coerce"
    ).fillna(1.0)
    out["runtime_ms_proxy"] = out.apply(runtime_proxy_from_row, axis=1)
    return out


def init_session_state():
    defaults = {
        "original_code_text": "",
        "llm_output": "",
        "generate_groq_pending": False,
        "run_pipeline_pending": False,
        "generated_llm_raw": "",
        "generated_llm_code": "",
        "xai_explanation": "",
        "shap_cache": {},
        "auto_xai_groq": True,
        "xai_generation_requested": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


class ASTNode:
    def __init__(self, node_type, expression, line_no, indentation):
        self.node_type = node_type
        self.expression = expression.strip()
        self.line_no = line_no
        self.indentation = indentation
        self.children = []


def parse_partial_ast(code_str):
    lines = code_str.split("\n")
    root = ASTNode("Root", "Root", 0, -1)
    stack = [root]

    for idx, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("//"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip())
        lowered = stripped.lower()

        node_type = "Statement"
        if any(tok in lowered for tok in ["for ", "for(", "while ", "while("]):
            node_type = "Loop"
        elif any(tok in lowered for tok in ["if ", "if(", "elif ", "else:"]):
            node_type = "Branch"
        elif any(tok in lowered for tok in ["append(", "push(", "malloc", "new "]):
            node_type = "Allocation"
        elif any(tok in lowered for tok in ["lock", "mutex", "synchronized"]):
            node_type = "Sync"

        node = ASTNode(node_type, stripped, idx, indent)
        while len(stack) > 1 and stack[-1].indentation >= indent:
            stack.pop()
        stack[-1].children.append(node)
        if node_type in ["Loop", "Branch"]:
            stack.append(node)

    return root


def extract_features(root, input_n, tdp, cores, expected_dim=9):
    loop_depth = 0
    busy_wait = 0
    inner_loop_alloc = 0
    strided_access = 0
    thread_lock = 0
    has_vector_ops = 0

    def walk(node, depth):
        nonlocal loop_depth
        nonlocal busy_wait
        nonlocal inner_loop_alloc
        nonlocal strided_access
        nonlocal thread_lock
        nonlocal has_vector_ops

        new_depth = depth + 1 if node.node_type == "Loop" else depth
        loop_depth = max(loop_depth, new_depth)

        expr = node.expression.lower()
        if node.node_type == "Loop" and any(tok in expr for tok in ["pass", "continue"]):
            busy_wait = 1
        if node.node_type == "Allocation" and depth > 0:
            inner_loop_alloc = 1
        if any(tok in expr for tok in ["[j][i]", "[col][row]", "stride"]):
            strided_access = 1
        if node.node_type == "Sync":
            thread_lock = 1
        if any(tok in expr for tok in ["np.", "matmul", "dot(", "linalg"]):
            has_vector_ops = 1

        for child in node.children:
            walk(child, new_depth)

    walk(root, 0)

    base = [
        min(loop_depth, 3),
        busy_wait,
        inner_loop_alloc,
        strided_access,
        thread_lock,
        has_vector_ops,
        float(input_n),
    ]
    if expected_dim == 7:
        return base
    return base + [float(tdp), float(cores)]


def normalize_code_text(llm_text):
    if not llm_text:
        return ""
    fenced = re.findall(r"```(?:python)?\s*([\s\S]*?)```", llm_text, flags=re.IGNORECASE)
    if fenced:
        return fenced[0].strip()
    return llm_text.strip()


def has_valid_function_body(code_text: str, language_name: str) -> bool:
    code_text = (code_text or "").strip()
    if not code_text:
        return False

    if language_name == "Python":
        try:
            tree = ast.parse(code_text)
        except Exception:
            return False
        return any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))

    if language_name == "C++":
        signature = r"[A-Za-z_][A-Za-z0-9_:<>\*&\s]*\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^\)]*\)\s*\{"
        return bool(re.search(signature, code_text))

    if language_name == "Java":
        signature = r"(public|private|protected)?\s*(static\s+)?[A-Za-z_][A-Za-z0-9_<>,\[\]\s]*\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^\)]*\)\s*\{"
        return bool(re.search(signature, code_text))

    return bool(re.search(r"\{[\s\S]*\}", code_text) or "def " in code_text)


def validate_feature_vector_9(features: List[float]) -> None:
    if len(features) != 9:
        raise ValueError(f"Feature vector must contain exactly 9 values, got {len(features)}")


def infer_algorithm_class(code_text):
    text = code_text.lower()
    if "bubble_sort" in text or ("for" in text and "for" in text and "swap" in text):
        return "bubble_sort"
    if "quick_sort" in text or "pivot" in text:
        return "quick_sort"
    if "np.dot" in text or "matmul" in text:
        return "matrix_mult_optimized"
    if "matrix" in text and "for" in text and "for" in text:
        return "matrix_mult_naive"
    if "while" in text and any(tok in text for tok in ["pass", "poll", "sleep"]):
        return "busy_wait_anomaly"
    return "unknown"


def detect_language(code_text: str) -> str:
    low = code_text.lower()
    if "#include" in low or "std::" in low or "cout" in low or "using namespace" in low or "::" in low:
        return "C++"
    if "def " in low or "import " in low or "numpy" in low or "pandas" in low:
        return "Python"
    if "public static void main" in low or "system.out.println" in low:
        return "Java"
    return "Python"


def code_language_token(language_name: str) -> str:
    mapping = {
        "C++": "cpp",
        "Python": "python",
        "Java": "java",
    }
    return mapping.get(language_name, "python")


def runtime_proxy(algorithm_class, n_val, cores):
    n = max(float(n_val), 1.0)
    core_factor = max(float(cores), 1.0)

    if algorithm_class == "bubble_sort":
        base = (n ** 2) / 2.2e7
    elif algorithm_class == "quick_sort":
        base = (n * math.log2(n + 1.0)) / 5.0e5
    elif algorithm_class == "matrix_mult_naive":
        cube = n ** 3
        base = cube / 2.8e11
    elif algorithm_class == "matrix_mult_optimized":
        base = (n ** 2.2) / 5.0e8
    elif algorithm_class == "busy_wait_anomaly":
        base = n / 6.0e3
    else:
        base = (n * math.log2(n + 1.0)) / 4.0e5

    runtime_ms = 1000.0 * (base / math.sqrt(core_factor))
    return float(max(runtime_ms, 0.01))


def runtime_proxy_from_row(row):
    algo = str(row.get("algorithm_class", "unknown"))
    n_val = row.get("input_scale_N", 10000)
    cores = row.get("hardware_cores", 4)
    return runtime_proxy(algo, n_val, cores)


def compute_pareto_frontier(points_df):
    ordered = points_df.sort_values("runtime_ms")
    best_energy = float("inf")
    frontier = []
    for _, row in ordered.iterrows():
        if row["energy_j"] <= best_energy:
            frontier.append(row)
            best_energy = row["energy_j"]
    if not frontier:
        return pd.DataFrame(columns=points_df.columns)
    return pd.DataFrame(frontier)


def calculate_emissions_gco2eq(energy_j, intensity_g_per_kwh):
    kwh = float(energy_j) / 3_600_000.0
    return kwh * float(intensity_g_per_kwh)


def build_ast_chart(code_text: str, language_name: str):
    preview = build_ast_preview(code_text, language_name=language_name)
    return ast_to_dot(preview["root"]), preview


def shap_summary_frame(summary: Dict[str, object]) -> pd.DataFrame:
    contributions = summary.get("contributions", []) if isinstance(summary, dict) else []
    rows = []
    for item in contributions:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "feature": item.get("feature", ""),
                "value": float(item.get("value", 0.0) or 0.0),
                "contribution": float(item.get("contribution", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(rows)


def local_xai_summary(base_eval: Dict[str, object], best_eval: Dict[str, object]) -> str:
    base_energy = float(base_eval.get("energy_j", 0.0) or 0.0)
    best_energy = float(best_eval.get("energy_j", 0.0) or 0.0)
    base_runtime = float(base_eval.get("runtime_ms", 0.0) or 0.0)
    best_runtime = float(best_eval.get("runtime_ms", 0.0) or 0.0)
    delta_energy = base_energy - best_energy
    delta_runtime = base_runtime - best_runtime
    return (
        f"Energy dropped from {base_energy:.6f} J to {best_energy:.6f} J. "
        f"Runtime changed from {base_runtime:.3f} ms to {best_runtime:.3f} ms. "
        f"That is a delta of {delta_energy:.6f} J and {delta_runtime:.3f} ms. "
        f"Primary improvement is the reduction of unnecessary work in the inner loop structure."
    )


@st.cache_data(ttl=600)
def get_live_carbon_intensity(provider_name, electricity_maps_key, electricity_maps_zone, watttime_token, watttime_ba, offline_intensity):
    reading = resolve_carbon_reading(
        provider_name=provider_name,
        electricity_maps_key=electricity_maps_key,
        electricity_maps_zone=electricity_maps_zone,
        watttime_token=watttime_token,
        watttime_ba=watttime_ba,
        offline_intensity=offline_intensity,
    )
    return {
        "intensity": reading.intensity_g_per_kwh,
        "source": reading.source,
        "mode": reading.mode,
        "zone": reading.zone,
    }


def make_refactor_candidates(original_code, llm_output, language_name):
    candidates = []
    seen = set()

    def add_candidate(label, code):
        code_clean = normalize_code_text(code)
        if not code_clean:
            return
        key = code_clean.strip()
        if key in seen:
            return
        seen.add(key)
        candidates.append({"label": label, "code": code_clean})

    if llm_output.strip():
        add_candidate("LLM candidate", llm_output)

    lower = original_code.lower()
    if any(token in lower for token in ["bubble_sort", "bubblesort", "cubicbubblesort"]) or (
        "swap(arr[j], arr[j + 1])" in lower and "vector<int>" in lower
    ):
        if language_name == "C++":
            add_candidate(
                "Heuristic: early-exit bubble",
                "#include <iostream>\n#include <vector>\nusing namespace std;\n\nvoid optimizedBubbleSort(vector<int>& arr) {\n    int n = (int)arr.size();\n    for (int i = 0; i < n - 1; ++i) {\n        bool swapped = false;\n        for (int j = 0; j < n - i - 1; ++j) {\n            if (arr[j] > arr[j + 1]) {\n                swap(arr[j], arr[j + 1]);\n                swapped = true;\n            }\n        }\n        if (!swapped) {\n            break;\n        }\n    }\n}\n\nint main() {\n    vector<int> arr = {5, 2, 8, 1, 3};\n    optimizedBubbleSort(arr);\n    for (int x : arr) {\n        cout << x << \" \";\n    }\n    return 0;\n}",
            )
        else:
            add_candidate(
                "Heuristic: early-exit bubble",
                "def bubble_sort_optimized(arr):\n    n = len(arr)\n    for i in range(n - 1):\n        swapped = False\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n                swapped = True\n        if not swapped:\n            break\n    return arr",
            )
    if "matrix" in lower and "for" in lower:
        add_candidate(
            "Heuristic: numpy dot",
            "import numpy as np\ndef matrix_multiply_opt(A, B):\n    return np.dot(A, B)",
        )
    if "while" in lower and any(tok in lower for tok in ["pass", "poll", "busy"]):
        add_candidate(
            "Heuristic: sleep backoff",
            "import time\ndef poll_sensor(target_time, n):\n    while time.time() < target_time:\n        time.sleep(0.001)\n    return n",
        )

    if not candidates:
        add_candidate(
            "Heuristic: vectorized placeholder",
            "def optimized_code(data):\n    return [x for x in data]",
        )
    return candidates


def rank_candidate_priority(original_code: str, candidate: Dict[str, str]) -> int:
    label = candidate.get("label", "")
    code = candidate.get("code", "")
    lowered = original_code.lower()
    code_lower = code.lower()

    if "bubble_sort" in lowered:
        if "sorted(" in code_lower or "std::sort" in code_lower:
            return 100
        if "swapped = false" in code_lower or "not swapped" in code_lower:
            return 0
        if code_lower == lowered:
            return 80
    if label.lower().startswith("heuristic"):
        return 10
    return 50


def evaluate_code(model, code_text, input_n, tdp, cores):
    expected_dim = int(getattr(model, "n_features_in_", 9))
    feature_bundle = analyze_code_features(code_text, input_n=input_n, tdp=tdp, cores=cores)
    if feature_bundle.get("parse_error"):
        raise ValueError("parse fail: code could not be parsed reliably")
    features = legacy_model_vector(feature_bundle)
    validate_feature_vector_9(features)
    if len(features) != expected_dim:
        raise ValueError(
            f"Feature length mismatch: got {len(features)} expected {expected_dim}"
        )
    pred = float(model.predict([features])[0])
    algo = infer_algorithm_class(code_text)
    language_name = feature_bundle.get("language", detect_language(code_text))
    runtime_profile = measure_runtime(code_text, language_name, algo, input_n)
    runtime_adjustment = float(runtime_profile["runtime_ms"]) * 0.001
    return {
        "energy_j": pred + runtime_adjustment,
        "model_energy_j": pred,
        "runtime_energy_j": runtime_adjustment,
        "features": features,
        "algorithm_class": algo,
        "runtime_ms": runtime_profile["runtime_ms"],
        "runtime_mode": runtime_profile["mode"],
        "runtime_detail": runtime_profile["detail"],
        "language": language_name,
        "parser_backend": feature_bundle.get("parser_backend", "unknown"),
        "feature_bundle": feature_bundle,
    }


def run_closed_loop(model, original_code, llm_output, input_n, tdp, cores, max_rounds=3):
    base_eval = evaluate_code(model, original_code, input_n, tdp, cores)
    best = {
        "label": "Original",
        "code": original_code,
        "eval": base_eval,
    }
    rounds = []

    language_name = detect_language(original_code)
    candidates = make_refactor_candidates(original_code, llm_output, language_name)
    candidates = sorted(
        candidates,
        key=lambda cand: (
            rank_candidate_priority(original_code, cand),
            len(cand.get("code", "")),
        ),
    )
    for idx, cand in enumerate(candidates[:max_rounds], start=1):
        round_entry = {
            "round": idx,
            "label": cand["label"],
            "status": "ok",
            "error": "",
            "energy_j": None,
            "runtime_ms": None,
            "runtime_mode": None,
            "delta_j": None,
            "code": cand["code"],
        }
        try:
            if not has_valid_function_body(cand["code"], language_name):
                raise ValueError("invalid syntax/function body: no valid function detected")
            cand_eval = evaluate_code(model, cand["code"], input_n, tdp, cores)
            delta_j = base_eval["energy_j"] - cand_eval["energy_j"]
            round_entry["energy_j"] = cand_eval["energy_j"]
            round_entry["runtime_ms"] = cand_eval["runtime_ms"]
            round_entry["runtime_mode"] = cand_eval["runtime_mode"]
            round_entry["delta_j"] = delta_j
            if cand_eval["energy_j"] < best["eval"]["energy_j"]:
                best = {
                    "label": cand["label"],
                    "code": cand["code"],
                    "eval": cand_eval,
                }
            elif (
                cand_eval["energy_j"] == best["eval"]["energy_j"]
                and cand["code"].strip() != original_code.strip()
                and cand_eval["runtime_ms"] < best["eval"]["runtime_ms"]
            ):
                best = {
                    "label": cand["label"],
                    "code": cand["code"],
                    "eval": cand_eval,
                }
            else:
                round_entry["status"] = "rejected"
                round_entry["error"] = "worse energy than current best"
        except Exception as exc:
            round_entry["status"] = "rejected"
            round_entry["error"] = str(exc)
        rounds.append(round_entry)

    return base_eval, best, rounds


def main():
    init_session_state()
    model, model_name = load_model()
    dataset, dataset_name = load_dataset()

    st.title("Cool Ecologic: Code Refactor Studio")
    st.caption(
        "Closed-loop refactoring agent + carbon translation + energy-time Pareto frontier"
    )

    if model is None:
        st.error("No model file found. Add baseline_rf_model.pkl in the project root.")
        st.stop()

    if dataset is None:
        st.error("No benchmark dataset found. Add eco_logic_synthetic_benchmark.csv.")
        st.stop()

    st.sidebar.header("Runtime Settings")
    input_n = st.sidebar.number_input("Input scale N", min_value=1, value=10000, step=1)
    tdp = st.sidebar.number_input("Hardware TDP (W)", min_value=1.0, value=45.0, step=1.0)
    cores = st.sidebar.number_input("CPU cores", min_value=1, value=8, step=1)
    max_rounds = st.sidebar.slider("Max agent rounds", min_value=1, max_value=5, value=3)

    st.sidebar.subheader("Carbon Telemetry")
    provider_label = st.sidebar.selectbox(
        "Carbon provider",
        ["Electricity Maps", "WattTime", "Offline fallback"],
        index=0,
    )
    provider_name = {
        "Electricity Maps": "electricity_maps",
        "WattTime": "watttime",
        "Offline fallback": "offline",
    }[provider_label]

    st.sidebar.caption("Default Maharashtra zone code: IN-WE (configurable)")
    electricity_maps_zone = st.sidebar.text_input("Electricity Maps zone", value=os.getenv("ELECTRICITY_MAPS_ZONE", DEFAULT_MAHARASHTRA_ZONE))
    electricity_maps_key = st.sidebar.text_input(
        "Electricity Maps API key",
        value=os.getenv("ELECTRICITY_MAPS_API_KEY", ""),
        type="password",
    )
    watttime_token = st.sidebar.text_input(
        "WattTime bearer token",
        value=os.getenv("WATTTIME_TOKEN", ""),
        type="password",
    )
    watttime_ba = st.sidebar.text_input(
        "WattTime BA code",
        value=os.getenv("WATTTIME_BA", "CAISO_NORTH"),
    )
    offline_intensity = st.sidebar.number_input(
        "Offline intensity (gCO2eq/kWh)",
        min_value=1.0,
        value=float(os.getenv("OFFLINE_GRID_INTENSITY", DEFAULT_OFFLINE_INTENSITY)),
        step=1.0,
    )

    try:
        payload = get_live_carbon_intensity(
            provider_name,
            electricity_maps_key,
            electricity_maps_zone,
            watttime_token,
            watttime_ba,
            offline_intensity,
        )
        intensity = payload["intensity"]
        intensity_source = payload["source"]
        intensity_mode = payload.get("mode", "unknown")
    except CarbonProviderError as exc:
        st.sidebar.error(f"Carbon source error: {exc}")
        st.stop()
    except Exception as exc:
        st.sidebar.error(f"Failed to fetch carbon intensity: {exc}")
        st.stop()
    st.sidebar.info(f"Carbon source: {intensity_source} ({intensity:.2f} gCO2eq/kWh)")
    st.sidebar.caption(f"Source mode: {intensity_mode}")
    objective_text = st.sidebar.text_input("Optimization objective", value="reduce energy")

    st.sidebar.subheader("LLM Settings")
    groq_model = st.sidebar.selectbox(
        "Groq model",
        [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "llama3-8b-8192",
            "llama3-70b-8192",
        ],
        index=0,
    )
    st.sidebar.write(f"Model: {model_name}")
    st.sidebar.write(f"Dataset: {dataset_name}")

    detected_language = detect_language(st.session_state.original_code_text)
    if st.session_state.original_code_text.strip():
        st.sidebar.success(f"Detected language: {detected_language}")
    else:
        st.sidebar.info("Paste code to detect language and run optimization.")

    if st.session_state.generate_groq_pending:
        try:
            lang = detected_language
            prompt = (
                (
                    f"You are an expert code optimizer. Improve the following {lang} code to {objective_text}. "
                    "Do NOT replace the algorithm by calling a library routine that merely delegates to a built-in (for example: do not use `std::sort`, `sorted`, `numpy.dot`, `Collections.sort`, or similar shortcuts). "
                    "Return a complete, self-contained implementation that improves the algorithm or technique. "
                    "If the code is recursive/backtracking, convert it to iterative, memoized, or dynamic-programming form when appropriate. "
                    "If it has redundant nested loops, remove extra work, tighten bounds, or add pruning / early-exit. "
                    "Preserve behavior, input/output, and language. Return ONLY the improved code inside a single fenced code block (```" + lang.lower() + "\n...```). Do not include explanations or additional prose.\n\n"
                )
                + st.session_state.original_code_text
            )
            gen_text = generate_refactor(prompt, model=groq_model)
            gen_code = normalize_code_text(gen_text)
            try:
                ast.parse(gen_code)
                st.session_state.generated_llm_code = gen_code
                st.session_state.llm_output = gen_code
            except Exception:
                st.session_state.generated_llm_code = gen_code
                st.session_state.llm_output = gen_text
            st.session_state.generated_llm_raw = gen_text
        except Exception as exc:
            st.sidebar.warning(f"Groq generation failed: {exc}")
        finally:
            st.session_state.generate_groq_pending = False

    original_code = st.text_area(
        "Original code (paste C++, Python, etc.)",
        key="original_code_text",
        height=260,
        placeholder=(
            "Paste any code here: C++, Java, Python, or another language.\n"
            "Examples: backtracking, recursion, nested loops, tree traversal, DP, parsing, graph code, etc."
        ),
    )

    auto_xai = st.sidebar.checkbox("Auto Groq explanation after run", value=st.session_state.auto_xai_groq)
    st.session_state.auto_xai_groq = auto_xai

    run_clicked = st.button("Run closed-loop optimization", type="primary")

    if run_clicked:
        if not st.session_state.original_code_text.strip():
            st.error("Paste code first. The dashboard no longer ships with a default example.")
            st.stop()
        st.session_state.generate_groq_pending = True
        st.session_state.run_pipeline_pending = True
        if auto_xai:
            st.session_state.xai_generation_requested = True
        st.rerun()

    if st.session_state.run_pipeline_pending or run_clicked:
        if not st.session_state.original_code_text.strip():
            st.warning("Paste code first to run the optimizer.")
            st.stop()
        if not st.session_state.llm_output.strip():
            st.warning("Generating the optimized candidate from the pasted code...")
            st.stop()
        with st.spinner("Running objective-driven loop..."):
            llm_output = st.session_state.llm_output
            started = time.time()
            base_eval, best, rounds = run_closed_loop(
                model,
                st.session_state.original_code_text,
                llm_output,
                input_n,
                tdp,
                cores,
                max_rounds=max_rounds,
            )
            elapsed_ms = (time.time() - started) * 1000.0
        st.session_state.run_pipeline_pending = False

        base_energy = base_eval["energy_j"]
        best_energy = best["eval"]["energy_j"]
        delta_energy = base_energy - best_energy
        pct = (delta_energy / base_energy * 100.0) if base_energy > 0 else 0.0

        base_carbon = calculate_emissions_gco2eq(base_energy, intensity)
        best_carbon = calculate_emissions_gco2eq(best_energy, intensity)
        runtime_mode_label = f"{base_eval.get('runtime_mode', 'proxy')} → {best['eval'].get('runtime_mode', 'proxy')}"
        base_shap = compute_shap_summary(model, base_eval["features"], LEGACY_FEATURE_NAMES)
        best_shap = compute_shap_summary(model, best["eval"]["features"], LEGACY_FEATURE_NAMES)

        st.caption(f"Parser backend: {base_eval.get('parser_backend', 'unknown')}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Energy", f"{base_energy:.6f} J", f"{best_energy:.6f} J best")
        m2.metric("Delta", f"{delta_energy:.6f} J", f"{pct:.2f}% reduction")
        m3.metric("Carbon", f"{base_carbon:.6f} gCO2eq", f"{best_carbon:.6f} gCO2eq")
        m4.metric("Loop runtime", f"{elapsed_ms:.1f} ms", f"{len(rounds)} rounds")
        st.caption(f"Runtime measurement mode: {runtime_mode_label}")

        xai_section = st.container(border=True)
        with xai_section:
            st.subheader("Groq XAI explanation")
            xai_refresh = st.button("Generate / refresh Groq explanation", key="groq_xai_refresh")
            should_generate_xai = xai_refresh or st.session_state.xai_generation_requested or not st.session_state.xai_explanation
            if should_generate_xai:
                try:
                    metrics_summary = (
                        f"Energy: {base_eval['energy_j']:.6f} J -> {best['eval']['energy_j']:.6f} J; "
                            f"Runtime: {base_eval['runtime_ms']:.3f} ms -> {best['eval']['runtime_ms']:.3f} ms; "
                            f"Mode: {base_eval.get('runtime_mode', 'proxy')} -> {best['eval'].get('runtime_mode', 'proxy')}."
                    )
                    shap_summary_text = (
                        f"Original top features: {', '.join(item['feature'] for item in base_shap['top_contributions'][:3])}. "
                        f"Optimized top features: {', '.join(item['feature'] for item in best_shap['top_contributions'][:3])}."
                    )
                    with st.spinner("Asking Groq to explain the improvement..."):
                        explanation = generate_code_explanation(
                            original_code=st.session_state.original_code_text,
                            refactored_code=best["code"],
                            language=detected_language,
                            objective=objective_text,
                            metrics_summary=metrics_summary,
                            shap_summary=shap_summary_text,
                            model=groq_model,
                        )
                    st.session_state.xai_explanation = explanation
                except Exception as exc:
                    st.session_state.xai_explanation = local_xai_summary(base_eval, best["eval"])
                    st.warning(f"Groq explanation failed, using local fallback: {exc}")
                finally:
                    st.session_state.xai_generation_requested = False

            if st.session_state.xai_explanation:
                st.write(st.session_state.xai_explanation)
            else:
                st.info("Groq explanation will appear here after generation.")

        st.subheader("Side-by-side optimization outcome")
        st.write(f"{base_energy:.6f} J -> {best_energy:.6f} J")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Original code**")
            st.code(st.session_state.original_code_text, language=code_language_token(detected_language))
            st.caption(f"Inferred class: {base_eval['algorithm_class']}")
        with col_b:
            st.markdown(f"**Best candidate ({best['label']})**")
            st.code(best["code"], language=code_language_token(detected_language))
            st.caption(f"Inferred class: {best['eval']['algorithm_class']}")

        with st.expander("AST preview"):
            st.caption(
                "The AST view is a compact structural preview of the code that highlights loops, branches, allocations, and sync points."
            )
            ast_left, ast_right = st.columns(2)
            with ast_left:
                st.markdown("**Original AST**")
                original_dot, original_ast_preview = build_ast_chart(st.session_state.original_code_text, detected_language)
                st.caption(
                    f"Nodes: {original_ast_preview['node_count']} | Max depth: {original_ast_preview['max_depth']} | Language: {original_ast_preview['language']}"
                )
                st.graphviz_chart(original_dot, use_container_width=True)
            with ast_right:
                st.markdown("**Optimized AST**")
                optimized_dot, optimized_ast_preview = build_ast_chart(best["code"], detected_language)
                st.caption(
                    f"Nodes: {optimized_ast_preview['node_count']} | Max depth: {optimized_ast_preview['max_depth']} | Language: {optimized_ast_preview['language']}"
                )
                st.graphviz_chart(optimized_dot, use_container_width=True)

        with st.expander("SHAP analysis"):
            st.caption(
                "SHAP explains which model inputs pushed the RandomForest energy prediction up or down. If the optional `shap` package is missing, the app falls back to a feature-importance approximation."
            )
            shap_left, shap_right = st.columns(2)
            for container, title, summary in [
                (shap_left, "Original", base_shap),
                (shap_right, "Optimized", best_shap),
            ]:
                with container:
                    st.markdown(f"**{title} prediction explanation**")
                    st.caption(f"Method: {summary['method']} | Expected value: {summary['expected_value']:.6f}")
                    shap_df = shap_summary_frame(summary).head(8)
                    if not shap_df.empty:
                        shap_fig = go.Figure(
                            go.Bar(
                                x=shap_df["contribution"],
                                y=shap_df["feature"],
                                orientation="h",
                                marker_color=["#2ca02c" if val >= 0 else "#d62728" for val in shap_df["contribution"]],
                            )
                        )
                        shap_fig.update_layout(
                            title=f"Top SHAP contributions - {title}",
                            xaxis_title="Contribution",
                            yaxis_title="Feature",
                            template="plotly_white",
                            height=360,
                            margin={"l": 10, "r": 10, "t": 40, "b": 10},
                        )
                        st.plotly_chart(shap_fig, use_container_width=True)
                        st.dataframe(shap_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No SHAP data available for this model instance.")

        with st.expander("Feature engineering metrics"):
            left_rows = rich_feature_rows(base_eval["feature_bundle"])
            right_rows = rich_feature_rows(best["eval"]["feature_bundle"])
            left_df = pd.DataFrame(left_rows, columns=["feature", "original"])
            right_df = pd.DataFrame(right_rows, columns=["feature", "optimized"])
            feature_compare = left_df.merge(right_df, on="feature", how="outer")
            st.dataframe(feature_compare, use_container_width=True)
            st.caption(
                "Tree-sitter now supplies cyclomatic complexity, Halstead-style counts, branch factors, "
                "allocation pressure, and cache-locality stride penalty."
            )

        st.subheader("Closed-loop rounds")
        rounds_df = pd.DataFrame(rounds)
        st.dataframe(rounds_df, use_container_width=True)

        st.subheader("Energy-Time Pareto Frontier")
        benchmark_neighbors = dataset.copy()
        target_algo = base_eval["algorithm_class"]
        same_algo = benchmark_neighbors[benchmark_neighbors["algorithm_class"] == target_algo]
        if len(same_algo) >= 40:
            benchmark_neighbors = same_algo.sample(40, random_state=42)
        else:
            benchmark_neighbors = benchmark_neighbors.sample(min(60, len(benchmark_neighbors)), random_state=42)

        benchmark_rows: List[Dict[str, float]] = []
        for _, row in benchmark_neighbors.iterrows():
            runtime_profile = measure_runtime(
                str(row.get("source_code", "")),
                detect_language(str(row.get("source_code", ""))),
                str(row.get("algorithm_class", "unknown")),
                float(row.get("input_scale_N", input_n)),
            )
            runtime_ms = runtime_profile["runtime_ms"]
            runtime_mode = runtime_profile.get("mode", "proxy")
            energy_j = float(row.get("target_energy_joules", 0.0))
            benchmark_rows.append(
                {
                    "runtime_ms": runtime_ms,
                    "energy_j": energy_j,
                    "algorithm_class": str(row.get("algorithm_class", "unknown")),
                    "label": "benchmark",
                    "input_n": float(row.get("input_scale_N", input_n)),
                    "tdp": float(row.get("hardware_tdp", tdp)),
                    "cores": float(row.get("hardware_cores", cores)),
                    "carbon_g": calculate_emissions_gco2eq(energy_j, intensity),
                    "runtime_mode": runtime_mode,
                }
            )
        sampled_points = pd.DataFrame(benchmark_rows)

        custom_points = pd.DataFrame(
            [
                {
                    "runtime_ms": base_eval["runtime_ms"],
                    "energy_j": base_eval["energy_j"],
                    "algorithm_class": base_eval["algorithm_class"],
                    "label": "original",
                    "input_n": float(input_n),
                    "tdp": float(tdp),
                    "cores": float(cores),
                    "carbon_g": calculate_emissions_gco2eq(base_eval["energy_j"], intensity),
                    "runtime_mode": base_eval.get("runtime_mode", "proxy"),
                },
                {
                    "runtime_ms": best["eval"]["runtime_ms"],
                    "energy_j": best["eval"]["energy_j"],
                    "algorithm_class": best["eval"]["algorithm_class"],
                    "label": "optimized",
                    "input_n": float(input_n),
                    "tdp": float(tdp),
                    "cores": float(cores),
                    "carbon_g": calculate_emissions_gco2eq(best["eval"]["energy_j"], intensity),
                    "runtime_mode": best["eval"].get("runtime_mode", "proxy"),
                },
            ]
        )

        all_points = pd.concat([sampled_points, custom_points], ignore_index=True)
        frontier = compute_pareto_frontier(all_points[["runtime_ms", "energy_j"]].copy())

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sampled_points["runtime_ms"],
                y=sampled_points["energy_j"],
                mode="markers",
                name="Benchmark",
                marker={"size": 6, "opacity": 0.35, "color": "#2b7bba"},
                customdata=np.stack(
                    [
                        sampled_points["algorithm_class"],
                        sampled_points["input_n"],
                        sampled_points["tdp"],
                        sampled_points["cores"],
                        sampled_points["carbon_g"],
                        sampled_points["runtime_mode"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "runtime=%{x:.3f} ms<br>energy=%{y:.6f} J"
                    "<br>class=%{customdata[0]}<br>N=%{customdata[1]}"
                    "<br>TDP=%{customdata[2]}W<br>cores=%{customdata[3]}"
                    "<br>carbon=%{customdata[4]:.6f} gCO2eq"
                    "<br>runtime_mode=%{customdata[5]}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[base_eval["runtime_ms"]],
                y=[base_eval["energy_j"]],
                mode="markers",
                name="Original",
                marker={"size": 12, "color": "#d62728", "symbol": "diamond"},
                hovertemplate=(
                    "Original<br>runtime=%{x:.3f} ms<br>energy=%{y:.6f} J"
                    f"<br>class={base_eval['algorithm_class']}<br>N={float(input_n)}"
                    f"<br>TDP={float(tdp)}W<br>cores={float(cores)}"
                    f"<br>carbon={calculate_emissions_gco2eq(base_eval['energy_j'], intensity):.6f} gCO2eq"
                    f"<br>runtime_mode={base_eval.get('runtime_mode', 'proxy')}"
                    "<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[best["eval"]["runtime_ms"]],
                y=[best["eval"]["energy_j"]],
                mode="markers",
                name="Optimized",
                marker={"size": 12, "color": "#2ca02c", "symbol": "star"},
                hovertemplate=(
                    "Optimized<br>runtime=%{x:.3f} ms<br>energy=%{y:.6f} J"
                    f"<br>class={best['eval']['algorithm_class']}<br>N={float(input_n)}"
                    f"<br>TDP={float(tdp)}W<br>cores={float(cores)}"
                    f"<br>carbon={calculate_emissions_gco2eq(best['eval']['energy_j'], intensity):.6f} gCO2eq"
                    f"<br>runtime_mode={best['eval'].get('runtime_mode', 'proxy')}"
                    "<extra></extra>"
                ),
            )
        )
        if len(frontier) > 1:
            fig.add_trace(
                go.Scatter(
                    x=frontier["runtime_ms"],
                    y=frontier["energy_j"],
                    mode="lines",
                    name="Pareto frontier",
                    line={"color": "#111111", "width": 2},
                )
            )

        fig.update_layout(
            xaxis_title="Execution time (ms)",
            yaxis_title="Energy (J)",
            template="plotly_white",
            legend={"orientation": "h", "y": 1.02, "x": 0.0},
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Runtime is measured using the runtime harness for original, optimized, and benchmark neighbors. "
            "When execution fails, runtime falls back to a proxy mode and is tagged in tooltips."
        )

        with st.expander("Graph energy model"):
            st.write(
                "The graph-based energy model is implemented in graph_energy_model.py and can be run independently. "
                "It builds Tree-sitter graphs, trains a pure-PyTorch GraphSAGE regressor, and compares it with the RandomForest baseline."
            )
            st.code("python tests/graph_energy_model_demo.py", language="bash")
            show_graph_model = st.checkbox("Show graph model / baseline predictions for the current snippet")
            if show_graph_model:
                try:
                    import graph_energy_model as p3
                except Exception:
                    p3 = None

                st.markdown("**Baseline RandomForest prediction**")
                try:
                    if os.path.exists("feature_engineered_rf_model.pkl"):
                        p2 = joblib.load("feature_engineered_rf_model.pkl")
                        fb = analyze_code_features(st.session_state.original_code_text, input_n=input_n, tdp=tdp, cores=cores)
                        vec = legacy_model_vector(fb)
                        p2_pred = float(p2.predict([vec])[0])
                        st.write(f"Baseline RF prediction: {p2_pred:.6f} J")
                    else:
                        st.info("No baseline RF model (feature_engineered_rf_model.pkl) found in project root.")
                except Exception as exc:
                    st.warning(f"Failed to compute baseline prediction: {exc}")

                st.markdown("**Graph energy model prediction**")
                if p3 is None:
                    st.info("Graph model module not importable. Run the graph model demo or install dependencies.")
                else:
                    # Attempt to load a saved GNN model if present
                    gnn_path = "graph_energy_model.pth"
                    if not os.path.exists(gnn_path):
                        st.info("No saved graph model found (graph_energy_model.pth). Run `graph_energy_model.run_graph_energy_experiment` to train one.")
                    else:
                        try:
                            device = p3.torch.device("cuda" if p3.torch.cuda.is_available() else "cpu")
                            # Build a single GraphExample for the snippet
                            builder = p3.ASTGraphBuilder()
                            example = builder.build(
                                code_text=st.session_state.original_code_text,
                                snippet_id="ui_snippet",
                                target=0.0,
                                input_n=input_n,
                                tdp=tdp,
                                cores=cores,
                                algorithm_class=infer_algorithm_class(st.session_state.original_code_text),
                            )
                            sample_graph = example
                            state = p3.torch.load(gnn_path, map_location=device)
                            if "node_type_embedding.weight" not in state:
                                raise ValueError("Saved graph model is missing node_type_embedding weights")
                            node_type_vocab_size = int(state["node_type_embedding.weight"].shape[0] - 2)
                            graph_feature_dim = int(sample_graph.graph_features.shape[0])
                            model = p3.ASTGNNRegressor(node_type_vocab_size=node_type_vocab_size, graph_feature_dim=graph_feature_dim)
                            model.load_state_dict(state)
                            model.to(device)
                            batch = p3.collate_graphs([example])
                            batch = {k: v.to(device) for k, v in batch.items()}
                            model.eval()
                            with p3.torch.no_grad():
                                pred = float(model(batch).cpu().item())
                            st.write(f"Graph model prediction: {pred:.6f} J")
                        except Exception as exc:
                            st.warning(f"Failed to run graph model: {exc}")


if __name__ == "__main__":
    main()
