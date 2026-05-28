import math
import os
import re
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

from project_ingestion import clone_git_repo, read_text_file, scan_project
from runtime_harness import measure_dotnet_project, measure_runtime
import load_env
from groq_client import generate_refactor
from feature_engineering import detect_language


def _sanitize_cpp_code(code: str) -> str:
    low = code.lower()
    need_vector = "vector<" in low and "#include <vector>" not in low
    need_iostream = "cout" in low and "#include <iostream>" not in low
    need_algorithm = "sort(" in low and "#include <algorithm>" not in low
    need_using_std = ("using namespace std" not in low) and (need_vector or need_iostream or need_algorithm)

    headers = []
    if need_iostream:
        headers.append("#include <iostream>")
    if need_vector:
        headers.append("#include <vector>")
    if need_algorithm:
        headers.append("#include <algorithm>")

    prefix = ""
    if headers:
        prefix = "\n".join(headers) + "\n"
    if need_using_std:
        prefix += "using namespace std;\n\n"

    if prefix and prefix.strip() not in code:
        return prefix + code
    return code
from log_config import get_logger

logger = get_logger(__name__)


st.set_page_config(
    page_title="EcoLogic : Energy-Aware Code Refactoring",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    defaults = {
        "workspace_mode": "Single file",
        "project_manifest": None,
        "project_source_label": "",
        "project_root": "",
        "selected_project_file": "",
        "project_runtime_profile": None,
        "code_text": (
            "def bubble_sort(arr):\n"
            "    n = len(arr)\n"
            "    for i in range(n):\n"
            "        for j in range(0, n - i - 1):\n"
            "            if arr[j] > arr[j + 1]:\n"
            "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
            "    return arr"
        ),
        "llm_output": "",
        "expanded_nodes": [""],
        "tree_page_offsets": {},
        "last_certificate_bytes": None,
        "last_certificate_filename": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _load_zip_to_temp(uploaded_file):
    temp_dir = Path(tempfile.mkdtemp(prefix="ecologic_upload_"))
    archive_path = temp_dir / uploaded_file.name
    archive_path.write_bytes(uploaded_file.getvalue())
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(temp_dir / "extracted")
    extracted_root = temp_dir / "extracted"
    entries = [path for path in extracted_root.iterdir() if path.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        return str(entries[0])
    return str(extracted_root)


def _scan_project_source(mode, folder_path, repo_url, uploaded_zip):
    if mode == "Local folder" and folder_path:
        return scan_project(folder_path), folder_path, folder_path

    if mode == "Git repo" and repo_url:
        cloned_root = clone_git_repo(repo_url)
        return scan_project(cloned_root), cloned_root, repo_url

    if mode == "ZIP upload" and uploaded_zip is not None:
        extracted_root = _load_zip_to_temp(uploaded_zip)
        return scan_project(extracted_root), extracted_root, uploaded_zip.name

    return None, "", ""


def _project_target_files(project_manifest):
    if not project_manifest:
        return []
    files = project_manifest.get("files", [])
    candidates = []
    for record in files:
        if record.get("language") in {"C#", "Python", "C++", "F#", "VB.NET"}:
            candidates.append(record.get("relative_path", ""))
    default_target = project_manifest.get("default_target") or {}
    default_path = default_target.get("relative_path", "")
    if default_path and default_path in candidates:
        candidates.remove(default_path)
        candidates.insert(0, default_path)
    return candidates


def _select_project_file(project_manifest, selected_relative_path):
    if not project_manifest or not selected_relative_path:
        return ""
    root_path = Path(project_manifest["root_path"])
    file_path = root_path / selected_relative_path
    if not file_path.exists():
        return ""
    return read_text_file(str(file_path))


def _choose_project_file(project_manifest, selected_relative_path):
    if not project_manifest or not selected_relative_path:
        return
    code_text = _select_project_file(project_manifest, selected_relative_path)
    if code_text:
        st.session_state.selected_project_file = selected_relative_path
        st.session_state.code_text = code_text


def _build_project_tree(files):
    tree = {"__files__": [], "__dirs__": {}}
    for record in files:
        relative_path = record.get("relative_path", "")
        if not relative_path:
            continue
        parts = [part for part in relative_path.split("/") if part]
        cursor = tree
        for part in parts[:-1]:
            dirs = cursor.setdefault("__dirs__", {})
            cursor = dirs.setdefault(part, {"__files__": [], "__dirs__": {}})
        cursor.setdefault("__files__", []).append(relative_path)
    return tree


def _render_tree_branch(node, prefix="", depth=0, max_files=120):
    entries = sorted(node.get("__dirs__", {}).items())
    file_names = sorted(node.get("__files__", []))

    for name, child in entries:
        path = f"{prefix}{name}"
        expanded = path in st.session_state.get("expanded_nodes", [])

        cols = st.columns([0.06, 0.94])
        btn_key = f"toggle_{path}"
        label = "−" if expanded else "+"
        if cols[0].button(label, key=btn_key, help=("Collapse" if expanded else "Expand")):
            expanded_nodes = list(st.session_state.get("expanded_nodes", []))
            if expanded:
                if path in expanded_nodes:
                    expanded_nodes.remove(path)
            else:
                expanded_nodes.append(path)
            st.session_state.expanded_nodes = expanded_nodes
            st.experimental_rerun()

        cols[1].markdown(f"**{name}/**")

        if expanded:
            _render_tree_branch(child, prefix=f"{path}/", depth=depth + 1, max_files=max_files)

    current_expanded = prefix in st.session_state.get("expanded_nodes", [])
    if prefix == "":
        current_expanded = True

    if file_names and current_expanded:
        page_size = 120 if max_files is None else max_files
        offsets = st.session_state.get("tree_page_offsets", {})
        start = int(offsets.get(prefix, 0))
        visible_files = file_names[start : start + page_size]
        st.caption(f"Files in {prefix or 'root'}")
        file_cols = st.columns(2)
        for index, relative_path in enumerate(visible_files):
            column = file_cols[index % 2]
            btn_key = f"project_tree_btn_{relative_path}"
            if column.button(relative_path.split("/")[-1], key=btn_key, help=relative_path):
                _choose_project_file(st.session_state.project_manifest, relative_path)
                st.rerun()

        if start + page_size < len(file_names):
            more_key = f"more_{prefix}"
            if st.button("Show more files...", key=more_key):
                offsets = dict(st.session_state.get("tree_page_offsets", {}))
                offsets[prefix] = start + page_size
                st.session_state.tree_page_offsets = offsets
                st.experimental_rerun()
        elif start > 0:
            back_key = f"back_{prefix}"
            if st.button("Show earlier files...", key=back_key):
                offsets = dict(st.session_state.get("tree_page_offsets", {}))
                offsets[prefix] = max(0, start - page_size)
                st.session_state.tree_page_offsets = offsets
                st.experimental_rerun()


def _run_project_profile(project_manifest):
    if not project_manifest:
        return None
    if project_manifest.get("project_type") == "dotnet":
        try:
            return measure_dotnet_project(project_manifest["root_path"])
        except Exception as exc:
            return {
                "mode": "error",
                "runtime_ms": None,
                "detail": {
                    "language": "C#",
                    "project_root": project_manifest.get("root_path"),
                    "reason": str(exc),
                },
            }
    try:
        lang_counts = project_manifest.get("language_counts", {}) or {}
        if lang_counts:
            top_lang = max(lang_counts.items(), key=lambda kv: kv[1])[0]
        else:
            top_lang = None

        if top_lang in {"Python", "C++"}:
            files = project_manifest.get("files", [])
            candidate = None
            for rec in files:
                if rec.get("language") == top_lang:
                    candidate = rec.get("relative_path")
                    break
            if candidate:
                code_text = _select_project_file(project_manifest, candidate)
                if code_text:
                    measured = measure_runtime(code_text, top_lang, infer_algorithm_class(code_text), 1000)
                    if isinstance(measured, dict) and measured.get("mode") == "measured":
                        return measured
        return {
            "mode": "error",
            "runtime_ms": None,
            "detail": {
                "language": top_lang or "unknown",
                "project_root": project_manifest.get("root_path"),
                "reason": "No measurable entrypoint found or measurement not possible on this host",
            },
        }
    except Exception as exc:
        return {
            "mode": "error",
            "runtime_ms": None,
            "detail": {
                "language": "mixed",
                "project_root": project_manifest.get("root_path"),
                "reason": str(exc),
            },
        }


def _render_project_summary(project_manifest, project_profile):
    if not project_manifest:
        return

    def build_tree_lines(paths):
        tree = {}
        for relative_path in sorted(paths):
            parts = [part for part in relative_path.split("/") if part]
            cursor = tree
            for part in parts:
                cursor = cursor.setdefault(part, {})

        lines = []

        def walk(node, prefix=""):
            items = list(node.items())
            for index, (name, child) in enumerate(items):
                is_last = index == len(items) - 1
                branch = "└── " if is_last else "├── "
                lines.append(f"{prefix}{branch}{name}")
                child_prefix = prefix + ("    " if is_last else "│   ")
                walk(child, child_prefix)

        walk(tree)
        return "\n".join(lines) if lines else "(empty project)"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Files", project_manifest.get("file_count", 0))
    col2.metric(".NET files", project_manifest.get("dotnet_file_count", 0))
    col3.metric("Project type", str(project_manifest.get("project_type", "unknown")).title())
    profile_label = project_profile["mode"] if project_profile else "pending"
    col4.metric("Profile", profile_label)

    summary_cols = st.columns(4)
    summary_cols[0].metric("Health score", project_manifest.get("project_health_score", 0))
    summary_cols[1].metric("Dependencies", project_manifest.get("dotnet_summary", {}).get("dependency_count", 0))
    summary_cols[2].metric("Entrypoints", project_manifest.get("dotnet_summary", {}).get("entrypoint_files", 0))
    summary_cols[3].metric("Total size", f"{project_manifest.get('total_bytes', 0) / 1024:.1f} KB")

    with st.expander("Project manifest", expanded=False):
        st.json(project_manifest)

    files = project_manifest.get("files", [])
    if files:
        with st.expander("Project tree", expanded=True):
            tree = _build_project_tree(files)
            _render_tree_branch(tree, max_files=60)

        language_counts = project_manifest.get("language_counts", {})
        language_labels = ", ".join(f"{name}: {count}" for name, count in sorted(language_counts.items()))
        st.caption(f"Languages detected: {language_labels}")

        largest_files = project_manifest.get("largest_files", [])
        if largest_files:
            top_rows = pd.DataFrame(largest_files[:5])
            st.dataframe(top_rows, width="stretch", hide_index=True)

    dotnet_summary = project_manifest.get("dotnet_summary", {})
    project_metadata = dotnet_summary.get("project_metadata", [])
    if project_metadata:
        with st.expander(".NET project metadata", expanded=True):
            st.dataframe(pd.DataFrame(project_metadata), width="stretch", hide_index=True)
            if len(project_metadata) > 1:
                st.caption("Multiple .NET project files detected. The first executable project is used for profiling.")

    if project_profile:
        with st.expander("Project profiling", expanded=False):
            st.json(project_profile)
        if project_profile.get("mode") == "error":
            st.error(project_profile.get("detail", {}).get("reason", "Project profiling failed"))


@st.cache_resource
def train_fallback_model(dataset):
    feature_rows = []
    target_values = []
    for _, row in dataset.iterrows():
        code_text = str(row.get("source_code", row.get("code", "")))
        root = parse_partial_ast(code_text)
        features = extract_features(
            root,
            row.get("input_scale_N", 10000),
            row.get("hardware_tdp", 45.0),
            row.get("hardware_cores", 4),
            9,
        )
        feature_rows.append(features)
        target_values.append(float(row.get("target_energy_joules", 1.0)))

    fallback_model = RandomForestRegressor(n_estimators=64, random_state=42, n_jobs=-1)
    fallback_model.fit(feature_rows, target_values)
    return fallback_model, "fallback_random_forest_trained_from_dataset"


@st.cache_resource
def load_model():
    model_files = [
        "phase1_model.pkl",
        "high_precision_energy_model.pkl",
        "energy_predictor_model.pkl",
    ]
    for filename in model_files:
        if os.path.exists(filename):
            try:
                model = joblib.load(filename)
                return model, filename
            except Exception as exc:
                logger.exception("Failed to load model file %s", filename)
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
            except Exception as exc:
                logger.exception("Failed to read dataset file %s", filename)
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
    fenced = re.findall(r"```(?:python|cpp|c\+\+|csharp|c#|cs)?\s*([\s\S]*?)```", llm_text, flags=re.IGNORECASE)
    if fenced:
        return fenced[0].strip()
    return llm_text.strip()


def _shorten_text(value, limit=240):
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _format_float(value, digits=3):
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "n/a"


def build_certificate_payload(
    *,
    original_code,
    best,
    base_eval,
    base_measured,
    measured_best,
    rounds,
    input_n,
    tdp,
    cores,
    intensity,
    model_name,
    dataset_name,
    elapsed_ms,
    project_manifest,
    selected_project_file,
):
    project_name = "Workspace"
    project_type = "single-file"
    if project_manifest:
        project_name = project_manifest.get("project_name") or project_manifest.get("root_path") or "Workspace"
        project_type = project_manifest.get("project_type", "unknown")

    measured_original_runtime_ms = base_measured.get("runtime_ms") if base_measured and base_measured.get("ok") else None
    measured_optimized_runtime_ms = measured_best.get("runtime_ms") if measured_best and measured_best.get("ok") else None
    measured_runtime_delta_ms = None
    if isinstance(measured_original_runtime_ms, (int, float)) and isinstance(measured_optimized_runtime_ms, (int, float)):
        measured_runtime_delta_ms = measured_original_runtime_ms - measured_optimized_runtime_ms

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_name": project_name,
        "project_type": project_type,
        "selected_file": selected_project_file or "n/a",
        "model_name": model_name or "unknown",
        "dataset_name": dataset_name or "unknown",
        "input_n": input_n,
        "tdp": tdp,
        "cores": cores,
        "intensity": intensity,
        "elapsed_ms": elapsed_ms,
        "rounds": len(rounds),
        "selection_reason": best.get("selection_reason", "unknown"),
        "original_algorithm": base_eval.get("algorithm_class", "unknown"),
        "best_algorithm": best.get("eval", {}).get("algorithm_class", "unknown"),
        "base_energy": base_eval.get("energy_j", 0.0),
        "best_energy": best.get("eval", {}).get("energy_j", 0.0),
        "base_runtime_ms": base_eval.get("runtime_ms", 0.0),
        "best_runtime_ms": best.get("eval", {}).get("runtime_ms", 0.0),
        "measured_original_runtime_ms": measured_original_runtime_ms,
        "measured_optimized_runtime_ms": measured_optimized_runtime_ms,
        "measured_runtime_delta_ms": measured_runtime_delta_ms,
        "original_code": original_code,
        "best_code": best.get("code", ""),
        "rounds_detail": rounds,
    }


def build_certificate_pdf(certificate):
    width = 595.28
    height = 841.89
    left = 42
    right = 42
    top = 48
    bottom = 44

    def pdf_escape(text):
        return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def text_line(x, y, text, size=10, font="F1"):
        return f"BT /{font} {size} Tf 1 0 0 1 {x:.2f} {y:.2f} Tm ({pdf_escape(text)}) Tj ET"

    def header_bar(title, subtitle):
        lines = [
            "0.0627 0.1961 0.2902 rg",
            f"0 {height - 58:.2f} {width:.2f} 58 re f",
            "1 1 1 rg",
            text_line(width / 2 - 145, height - 28, title, size=20),
            text_line(width / 2 - 185, height - 45, subtitle, size=9),
        ]
        return "\n".join(lines)

    def row(x, y, label, value, label_width=140, value_width=350, value_size=9):
        return "\n".join(
            [
                f"0.95 0.96 0.98 rg {x:.2f} {y - 14:.2f} {label_width:.2f} 16 re f",
                f"0.98 0.98 0.98 rg {x + label_width:.2f} {y - 14:.2f} {value_width:.2f} 16 re f",
                "0.82 0.85 0.88 RG 0.82 0.85 0.88 rg",
                f"{x:.2f} {y - 14:.2f} {label_width + value_width:.2f} 16 re S",
                text_line(x + 6, y - 3, label, size=8),
                text_line(x + label_width + 6, y - 3, value, size=value_size),
            ]
        )

    page1 = [header_bar("EcoLogic Optimization Certificate", "Closed-loop optimization summary and measured impact")]
    y = height - 82
    page1.append(text_line(left, y, f"Generated at: {certificate.get('generated_at', 'n/a')}", size=10))
    y -= 18
    page1.append(text_line(left, y, f"Project: {certificate.get('project_name', 'n/a')}", size=10))
    y -= 14
    page1.append(text_line(left, y, f"Project type: {certificate.get('project_type', 'n/a')} | Selected file: {certificate.get('selected_file', 'n/a')}", size=9))
    y -= 20
    page1.append(text_line(left, y, "Optimization summary", size=12))
    y -= 10
    page1.append(row(left, y, "Model", certificate.get("model_name", "n/a"), value_width=370))
    y -= 16
    page1.append(row(left, y, "Dataset", certificate.get("dataset_name", "n/a"), value_width=370))
    y -= 16
    page1.append(row(left, y, "Input scale N", str(certificate.get("input_n", "n/a")), value_width=370))
    y -= 16
    page1.append(row(left, y, "CPU cores / TDP", f"{certificate.get('cores', 'n/a')} / {certificate.get('tdp', 'n/a')}", value_width=370))
    y -= 16
    page1.append(row(left, y, "Carbon intensity", f"{_format_float(certificate.get('intensity'), 2)} gCO2eq/kWh", value_width=370))
    y -= 16
    page1.append(row(left, y, "Rounds", str(certificate.get("rounds", "n/a")), value_width=370))
    y -= 16
    page1.append(row(left, y, "Selection reason", certificate.get("selection_reason", "n/a"), value_width=370))
    y -= 22
    page1.append(text_line(left, y, "Performance comparison", size=12))
    y -= 10
    page1.append(row(left, y, "Metric", "Original", label_width=130, value_width=120))
    page1.append(row(left + 250, y, "Optimized", "", label_width=110, value_width=130))
    y -= 16
    page1.append(row(left, y, "Inferred algorithm", certificate.get("original_algorithm", "n/a"), label_width=130, value_width=120))
    page1.append(row(left + 250, y, "", certificate.get("best_algorithm", "n/a"), label_width=110, value_width=130))
    y -= 16
    page1.append(row(left, y, "Energy (J)", _format_float(certificate.get("base_energy"), 6), label_width=130, value_width=120))
    page1.append(row(left + 250, y, "", _format_float(certificate.get("best_energy"), 6), label_width=110, value_width=130))
    y -= 16
    page1.append(row(left, y, "Proxy runtime (ms)", _format_float(certificate.get("base_runtime_ms")), label_width=130, value_width=120))
    page1.append(row(left + 250, y, "", _format_float(certificate.get("best_runtime_ms")), label_width=110, value_width=130))
    y -= 16
    page1.append(row(left, y, "Measured runtime (ms)", _format_float(certificate.get("measured_original_runtime_ms")), label_width=130, value_width=120))
    page1.append(row(left + 250, y, "", _format_float(certificate.get("measured_optimized_runtime_ms")), label_width=110, value_width=130))
    y -= 16
    page1.append(row(left, y, "Measured delta (ms)", "", label_width=130, value_width=120))
    page1.append(row(left + 250, y, "", _format_float(certificate.get("measured_runtime_delta_ms")), label_width=110, value_width=130))

    page2 = [header_bar("EcoLogic Certificate - Supporting Details", "Code snapshot and reviewed rounds")]
    y2 = height - 82
    page2.append(text_line(left, y2, "Code snapshot", size=12))
    y2 -= 14
    for block_title, block_code in [
        ("Original snippet", certificate.get("original_code", "")),
        ("Optimized snippet", certificate.get("best_code", "")),
    ]:
        page2.append(text_line(left, y2, block_title, size=10))
        y2 -= 12
        for line in _shorten_text(block_code, 650).splitlines()[:10]:
            page2.append(text_line(left + 10, y2, _shorten_text(line, 110), size=7))
            y2 -= 9
        y2 -= 6

    page2.append(text_line(left, y2, "Rounds reviewed", size=12))
    y2 -= 14
    for round_entry in (certificate.get("rounds_detail", []) or [])[:3]:
        round_text = (
            f"Round {round_entry.get('round', '?')}: {round_entry.get('label', 'candidate')} | "
            f"energy={_format_float(round_entry.get('energy_j'), 6)} | runtime={_format_float(round_entry.get('runtime_ms'))} | "
            f"measured={_format_float(round_entry.get('measured_runtime_ms'))} | {round_entry.get('selection_reason', 'not selected')}"
        )
        for line in _shorten_text(round_text, 120).splitlines():
            page2.append(text_line(left + 8, y2, line, size=8))
            y2 -= 10

    page2.append(text_line(left, 60, "Generated by EcoLogic. This certificate is designed to stay within 1-2 pages.", size=8))

    def make_content_stream(lines):
        return ("\n".join(lines) + "\n").encode("utf-8")

    content1 = make_content_stream(page1)
    content2 = make_content_stream(page2)

    objects = []

    def add_object(content):
        objects.append(content)

    add_object("<< /Type /Catalog /Pages 2 0 R >>")
    add_object("<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>")
    add_object(
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.2f} {height:.2f}] /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>"
    )
    add_object(
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.2f} {height:.2f}] /Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>"
    )
    add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    add_object(f"<< /Length {len(content1)} >>\nstream\n".encode("utf-8") + content1 + b"endstream")
    add_object(f"<< /Length {len(content2)} >>\nstream\n".encode("utf-8") + content2 + b"endstream")

    pdf = bytearray()
    pdf.extend(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("utf-8"))
        if isinstance(obj, bytes):
            pdf.extend(obj)
            pdf.extend(b"\n")
        else:
            pdf.extend(obj.encode("utf-8"))
            pdf.extend(b"\n")
        pdf.extend(b"endobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("utf-8"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("utf-8"))
    pdf.extend(b"trailer\n")
    pdf.extend(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("utf-8"))
    pdf.extend(b"startxref\n")
    pdf.extend(f"{xref_start}\n".encode("utf-8"))
    pdf.extend(b"%%EOF")
    return bytes(pdf)


def infer_algorithm_class(code_text):
    text = code_text.lower()
    if "bubble_sort" in text or "bubblesort" in text or ("for" in text and "for" in text and any(tok in text for tok in ["swap", "temp =", "arr[j + 1]"])):
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


def fetch_electricity_maps_intensity(api_key, zone="IN-WE"):
    if not api_key:
        raise ValueError("Electricity Maps API key missing")
    url = "https://api.electricitymap.org/v3/carbon-intensity/latest"
    headers = {"auth-token": api_key}
    response = requests.get(url, headers=headers, params={"zone": zone}, timeout=8)
    response.raise_for_status()
    payload = response.json()

    possible_keys = ["carbonIntensity", "carbonIntensityAvg", "carbonIntensityForecast"]
    for key in possible_keys:
        if key in payload and payload[key] is not None:
            return float(payload[key])
    raise ValueError("Electricity Maps response missing carbon intensity")


def fetch_watttime_intensity(username, password, ba="MISO"):
    if not username or not password:
        raise ValueError("WattTime credentials missing")

    login_resp = requests.get(
        "https://api.watttime.org/login",
        auth=(username, password),
        timeout=8,
    )
    login_resp.raise_for_status()
    token = login_resp.json().get("token")
    if not token:
        raise ValueError("WattTime token not received")

    index_resp = requests.get(
        "https://api.watttime.org/index",
        headers={"Authorization": f"Bearer {token}"},
        params={"ba": ba},
        timeout=8,
    )
    index_resp.raise_for_status()

    data = index_resp.json()
    if isinstance(data, dict):
        for key in ["moer", "value"]:
            if key in data:
                return float(data[key])
    raise ValueError("WattTime response missing intensity")


@st.cache_data(ttl=600)
def get_live_carbon_intensity(provider, electricity_maps_key, watttime_user, watttime_password):
    if provider == "Electricity Maps":
        return {
            "intensity": fetch_electricity_maps_intensity(electricity_maps_key, zone="IN-WE"),
            "source": "Electricity Maps IN-WE",
        }
    if provider == "WattTime":
        return {
            "intensity": fetch_watttime_intensity(watttime_user, watttime_password),
            "source": "WattTime BA",
        }
    raise ValueError("Unsupported provider")


def make_refactor_candidates(original_code, llm_output):
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
    if "bubble_sort" in lower or ("bubble" in lower and "sort" in lower):
        # If this appears to be C++ source (includes, cout, std::), offer C++ heuristics
        is_cpp = any(tok in lower for tok in ["#include", "std::", "cout", "cin", "using namespace std"]) 
        is_csharp = any(tok in lower for tok in ["using system", "console.write", "console.writeline", "class program", "static void main("])
        if is_csharp:
            add_candidate(
                "Heuristic: Array.Sort (C#)",
                """using System;

class Program
{
    static void BubbleSort(int[] arr)
    {
        Array.Sort(arr);
    }

    static void Main()
    {
        int[] arr = { 5, 1, 4, 2, 8 };
        BubbleSort(arr);
        Console.Write("Sorted array: ");
        foreach (int x in arr) Console.Write(x + " ");
    }
}
""",
            )
            add_candidate(
                "Heuristic: early-exit bubble (C#)",
                """using System;

class Program
{
    static void BubbleSort(int[] arr)
    {
        int n = arr.Length;
        for (int i = 0; i < n - 1; i++)
        {
            bool swapped = false;
            for (int j = 0; j < n - i - 1; j++)
            {
                if (arr[j] > arr[j + 1])
                {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                    swapped = true;
                }
            }
            if (!swapped) break;
        }
    }

    static void Main()
    {
        int[] arr = { 5, 1, 4, 2, 8 };
        BubbleSort(arr);
        Console.Write("Sorted array: ");
        foreach (int x in arr) Console.Write(x + " ");
    }
}
""",
            )
        elif is_cpp:
            add_candidate(
                "Heuristic: early-exit bubble (C++)",
                """#include <algorithm>
#include <vector>
using namespace std;

void bubbleSortOptimized(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i) {
        bool swapped = false;
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}
""",
            )
            add_candidate(
                "Heuristic: std::sort (C++)",
                """#include <algorithm>
#include <vector>
using namespace std;

void sort_with_std(vector<int>& arr) {
    sort(arr.begin(), arr.end());
}
""",
            )
        else:
            add_candidate(
                "Heuristic: builtin sorted",
                "def optimized_sort(arr):\n    return sorted(arr)",
            )
    # Detect naive matrix multiplication patterns: nested loops and array indexing
    loop_count = lower.count("for ")
    has_array_index = any(tok in lower for tok in ["a[", "b[", "c[", "matrix[", "arr["])
    matmul_like = loop_count >= 2 and has_array_index
    if "matrix" in lower and "for" in lower or matmul_like:
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

    # Prefer deterministic C++ heuristics (std::sort) when present so users see
    # a concrete transformed candidate rather than only heuristic-similar originals.
    try:
        candidates = sorted(
            candidates,
            key=lambda c: (
                0 if any(tok in c["label"] for tok in ["std::sort", "Array.Sort"]) else 1,
                c["label"],
            ),
        )
    except Exception:
        pass
    return candidates


def evaluate_code(model, code_text, input_n, tdp, cores):
    expected_dim = int(getattr(model, "n_features_in_", 9))
    root = parse_partial_ast(code_text)
    features = extract_features(root, input_n, tdp, cores, expected_dim)
    if len(features) != expected_dim:
        raise ValueError(
            f"Feature length mismatch: got {len(features)} expected {expected_dim}"
        )
    pred = float(model.predict([features])[0])
    algo = infer_algorithm_class(code_text)
    runtime_ms = runtime_proxy(algo, input_n, cores)
    return {
        "energy_j": pred,
        "features": features,
        "algorithm_class": algo,
        "runtime_ms": runtime_ms,
    }


def _measure_runtime_safe(code_text, input_n, algorithm_class):
    language_name = detect_language(code_text)
    try:
        result = measure_runtime(code_text, language_name, algorithm_class, input_n)
        return {
            "ok": True,
            "runtime_ms": float(result.get("runtime_ms", 0.0)),
            "mode": result.get("mode", "unknown"),
            "detail": result.get("detail", {}),
            "language": language_name,
        }
    except Exception as exc:
        logger.exception("Measured runtime failed for %s code: %s", language_name, exc)
        return {
            "ok": False,
            "runtime_ms": None,
            "mode": "error",
            "detail": {"reason": str(exc)},
            "language": language_name,
        }


def run_closed_loop(model, original_code, llm_output, input_n, tdp, cores, max_rounds=3):
    base_eval = evaluate_code(model, original_code, input_n, tdp, cores)
    base_measured = _measure_runtime_safe(original_code, input_n, base_eval["algorithm_class"])
    best = {
        "label": "Original",
        "code": original_code,
        "eval": base_eval,
        "selection_reason": "baseline",
    }
    rounds = []

    candidates = make_refactor_candidates(original_code, llm_output)
    for idx, cand in enumerate(candidates[:max_rounds], start=1):
        round_entry = {
            "round": idx,
            "label": cand["label"],
            "status": "ok",
            "error": "",
            "energy_j": None,
            "runtime_ms": None,
            "delta_j": None,
            "measured_runtime_ms": None,
            "measured_delta_ms": None,
            "selection_reason": "",
            "code": cand["code"],
        }
        try:
            cand_eval = evaluate_code(model, cand["code"], input_n, tdp, cores)
            delta_j = base_eval["energy_j"] - cand_eval["energy_j"]
            round_entry["energy_j"] = cand_eval["energy_j"]
            round_entry["runtime_ms"] = cand_eval["runtime_ms"]
            round_entry["delta_j"] = delta_j

            measured = _measure_runtime_safe(cand["code"], input_n, cand_eval["algorithm_class"])
            if measured["ok"]:
                round_entry["measured_runtime_ms"] = measured["runtime_ms"]
                if base_measured["ok"]:
                    round_entry["measured_delta_ms"] = base_measured["runtime_ms"] - measured["runtime_ms"]

            # Selection policy:
            # 1) Better predicted energy always wins.
            # 2) If energies are close (within 2%), prefer measurably faster runtime when available.
            # 3) Keep original only when candidate is not better by either criterion.
            best_energy = best["eval"]["energy_j"]
            energy_improved = cand_eval["energy_j"] < best_energy
            energy_close = cand_eval["energy_j"] <= best_energy * 1.02
            measured_faster = False
            if measured["ok"] and base_measured["ok"]:
                measured_faster = measured["runtime_ms"] < base_measured["runtime_ms"]

            if energy_improved:
                best = {
                    "label": cand["label"],
                    "code": cand["code"],
                    "eval": cand_eval,
                    "selection_reason": "lower_predicted_energy",
                }
                round_entry["selection_reason"] = "selected: lower_predicted_energy"
            elif energy_close and measured_faster and cand["code"].strip() != original_code.strip():
                best = {
                    "label": cand["label"],
                    "code": cand["code"],
                    "eval": cand_eval,
                    "selection_reason": "measured_runtime_better_within_energy_tolerance",
                }
                round_entry["selection_reason"] = "selected: measured_runtime_better_within_energy_tolerance"
        except Exception as exc:
            round_entry["status"] = "rejected"
            round_entry["error"] = str(exc)
        rounds.append(round_entry)

    return base_eval, best, rounds


def main():
    init_session_state()

    model, model_name = load_model()
    dataset, dataset_name = load_dataset()

    st.title("EcoLogic : Energy-Aware Code Refactoring")
    st.caption("Project-aware refactoring, energy scoring, runtime profiling, and carbon translation")

    if dataset is None:
        st.error("No benchmark dataset found. Add eco_logic_synthetic_benchmark.csv.")
        st.stop()

    # If no saved model artifact is present, attempt to auto-train a fallback
    # model from the benchmark dataset. This removes the manual opt-in checkbox
    # to provide a smoother, industry-grade experience.
    if model is None:
        st.sidebar.info("No saved model artifact found; attempting auto-train from benchmark dataset...")
        try:
            model, model_name = train_fallback_model(dataset)
            st.sidebar.success("Fallback model trained and loaded.")
        except Exception as exc:
            logger.exception("Auto-training fallback model failed")
            st.sidebar.error(
                "No saved model artifact found. Upload a model file to the workspace."
            )

    st.sidebar.header("Workspace Intake")
    # Certificates panel: list generated certificates from workspace
    try:
        cert_dir = Path("certificates")
        cert_dir.mkdir(exist_ok=True)
        with st.sidebar.expander("Certificates", expanded=False):
            cert_files = sorted(cert_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
            if cert_files:
                for p in cert_files:
                    name = p.name
                    cols = st.columns([0.82, 0.18])
                    cols[0].write(name)
                    with open(p, "rb") as fh:
                        data = fh.read()
                    cols[1].download_button("DL", data=data, file_name=name, mime="application/pdf")
            else:
                st.write("No certificates yet. Run an optimization to generate one.")
            if st.button("Refresh certificates list"):
                st.experimental_rerun()
    except Exception:
        pass
    workspace_mode = st.sidebar.selectbox(
        "Input mode",
        ["Single file", "Local folder", "Git repo", "ZIP upload"],
        index=["Single file", "Local folder", "Git repo", "ZIP upload"].index(st.session_state.workspace_mode)
        if st.session_state.workspace_mode in ["Single file", "Local folder", "Git repo", "ZIP upload"]
        else 0,
    )
    st.session_state.workspace_mode = workspace_mode

    folder_path = ""
    repo_url = ""
    uploaded_zip = None
    if workspace_mode == "Local folder":
        folder_path = st.sidebar.text_input("Folder path", value=st.session_state.project_root or "")
    elif workspace_mode == "Git repo":
        repo_url = st.sidebar.text_input("Git repo URL", value=st.session_state.project_source_label or "")
    elif workspace_mode == "ZIP upload":
        uploaded_zip = st.sidebar.file_uploader("Upload a .zip of the project", type=["zip"])

    scan_clicked = st.sidebar.button("Scan workspace", type="primary")
    if scan_clicked:
        try:
            project_manifest, project_root, project_label = _scan_project_source(
                workspace_mode,
                folder_path,
                repo_url,
                uploaded_zip,
            )
            if project_manifest is None:
                st.sidebar.warning("Choose a folder, repo URL, or zip archive before scanning.")
            else:
                st.session_state.project_manifest = project_manifest
                st.session_state.project_root = project_root
                st.session_state.project_source_label = project_label
                st.session_state.project_runtime_profile = _run_project_profile(project_manifest)
                target_files = _project_target_files(project_manifest)
                if target_files:
                    st.session_state.selected_project_file = target_files[0]
                    loaded_code = _select_project_file(project_manifest, target_files[0])
                    if loaded_code:
                        st.session_state.code_text = loaded_code
                st.sidebar.success(f"Loaded {project_manifest.get('file_count', 0)} files")
        except Exception as exc:
            logger.exception("Workspace scan failed")
            st.sidebar.error(f"Workspace scan failed: {exc}")

    if st.session_state.project_manifest:
        st.sidebar.info(f"Workspace: {st.session_state.project_source_label}")
        st.sidebar.write(f"Detected type: {st.session_state.project_manifest.get('project_type', 'unknown')}")

    st.sidebar.header("Runtime Settings")
    input_n = st.sidebar.number_input("Input scale N", min_value=1, value=10000, step=1)
    tdp = st.sidebar.number_input("Hardware TDP (W)", min_value=1.0, value=45.0, step=1.0)
    cores = st.sidebar.number_input("CPU cores", min_value=1, value=8, step=1)
    max_rounds = st.sidebar.slider("Max agent rounds", min_value=1, max_value=5, value=3)
    prefer_heuristics = st.sidebar.checkbox("Prefer deterministic heuristics (force candidate)", value=False)

    st.sidebar.subheader("Carbon Telemetry")
    provider = st.sidebar.selectbox(
        "Provider",
        ["Fallback Constant", "Electricity Maps", "WattTime"],
        index=0,
    )
    fallback_intensity = st.sidebar.number_input(
        "Fallback intensity (gCO2eq/kWh)",
        min_value=1.0,
        value=714.0,
        step=1.0,
    )

    electricity_maps_key = st.sidebar.text_input(
        "Electricity Maps API key",
        value=os.getenv("ELECTRICITY_MAPS_API_KEY", ""),
        type="password",
    )
    watttime_user = st.sidebar.text_input(
        "WattTime username",
        value=os.getenv("WATTTIME_USERNAME", ""),
    )
    watttime_password = st.sidebar.text_input(
        "WattTime password",
        value=os.getenv("WATTTIME_PASSWORD", ""),
        type="password",
    )

    intensity = fallback_intensity
    intensity_source = "Fallback static intensity"
    if provider != "Fallback Constant":
        try:
            payload = get_live_carbon_intensity(
                provider,
                electricity_maps_key,
                watttime_user,
                watttime_password,
            )
            intensity = payload["intensity"]
            intensity_source = payload["source"]
        except Exception as exc:
            st.sidebar.warning(f"Live carbon fetch failed, using fallback: {exc}")

    st.sidebar.info(f"Carbon source: {intensity_source} ({intensity:.2f} gCO2eq/kWh)")
    st.sidebar.write(f"Model: {model_name}")
    st.sidebar.write(f"Dataset: {dataset_name}")

    project_manifest = st.session_state.project_manifest
    if project_manifest:
        _render_project_summary(project_manifest, st.session_state.project_runtime_profile)
        target_files = _project_target_files(project_manifest)
        if target_files:
            selected_index = 0
            if st.session_state.selected_project_file in target_files:
                selected_index = target_files.index(st.session_state.selected_project_file)
            selected_project_file = st.selectbox(
                "Target file to score",
                target_files,
                index=selected_index,
            )
            if selected_project_file != st.session_state.selected_project_file:
                st.session_state.selected_project_file = selected_project_file
                loaded_code = _select_project_file(project_manifest, selected_project_file)
                if loaded_code:
                    st.session_state.code_text = loaded_code

    col_left, col_right = st.columns(2)
    with col_left:
        st.text_area(
            "Source code under analysis",
            height=320,
            key="code_text",
            help="For project mode, this loads the selected file from the scanned repo, folder, or zip.",
        )
    with col_right:
        st.text_area(
            "LLM refactored output (raw or fenced code)",
            height=320,
            key="llm_output",
            help="Paste direct LLM output here. The app sanitizes fenced code automatically.",
        )

    original_code = st.session_state.code_text
    llm_output = st.session_state.llm_output

    if project_manifest and st.session_state.project_runtime_profile:
        runtime_profile = st.session_state.project_runtime_profile
        profile_cols = st.columns(3)
        profile_cols[0].metric("Project runtime", f"{runtime_profile['runtime_ms']:.1f} ms")
        profile_cols[1].metric("Profile mode", runtime_profile.get("mode", "unknown"))
        profile_cols[2].metric("Selected file", st.session_state.selected_project_file or "auto")

    run_clicked = st.button("Run closed-loop optimization", type="primary")

    if run_clicked:
        # If the user didn't paste an LLM output, call the LLM automatically.
        if not llm_output or not llm_output.strip():
            try:
                # ensure .env is loaded so GROQ_API_KEY is available
                try:
                    load_env.load()
                except Exception:
                    pass

                raw = None
                # 3 retries with basic backoff
                for attempt in range(3):
                    try:
                        lang = detect_language(original_code)
                        if lang == "C++":
                            prompt = (
                                "You are a senior C++ performance engineer. Input code is provided below. Return optimized code by replacing the algorithm implementation with a correct, compilable, and more efficient algorithmic implementation. "
                                "Prefer algorithmic improvements (e.g., std::sort for sorting, early-exit optimizations, reduce complexity) or use standard library algorithms when appropriate. "
                                "Include all necessary headers and `using namespace std;` if needed. Return ONLY the optimized C++ code inside a fenced code block (```cpp ... ```). Do not include any explanations.\n\n"
                                + original_code
                            )
                        elif lang == "C#":
                            prompt = (
                                "You are a senior C# performance engineer. Input code is provided below. Return optimized code by replacing the algorithm implementation with a correct, compilable, and more efficient implementation. "
                                "Prefer algorithmic improvements and library methods when appropriate (for sorting, `Array.Sort` is acceptable). "
                                "Return ONLY the optimized C# code inside a fenced code block (```csharp ... ```). Do not include any explanations.\n\n"
                                + original_code
                            )
                        else:
                            prompt = (
                                "You are a senior performance engineer. Input code is provided below. Return optimized code by replacing the algorithm implementation with a correct, runnable, and more efficient implementation. "
                                "Prefer builtin or vectorized operations (e.g., `sorted`, `numpy.dot`) and algorithmic improvements. Return ONLY the replacement Python code inside a fenced code block (```python ... ```). Do not include any explanations.\n\n"
                                + original_code
                            )
                        resp = generate_refactor(prompt, max_output_tokens=1200)
                        if resp and isinstance(resp, str) and resp.strip():
                            raw = resp
                            break
                    except Exception as exc:
                        logger.exception("LLM call attempt %d failed: %s", attempt + 1, exc)
                if raw:
                    st.session_state.llm_output = raw
                    llm_output = raw
                    logger.info("LLM raw output captured for UI")
                else:
                    logger.warning("LLM returned no output after retries")
            except Exception as exc:
                logger.exception("Automatic LLM invocation failed: %s", exc)

        with st.spinner("Running objective-driven loop..."):
            started = time.time()
            base_eval, best, rounds = run_closed_loop(
                model,
                original_code,
                llm_output,
                input_n,
                tdp,
                cores,
                max_rounds=max_rounds,
            )
            # If user prefers deterministic heuristics, pick the first heuristic candidate
            if prefer_heuristics and rounds:
                for r in rounds:
                    if r.get("label", "").lower().startswith("heuristic") and r.get("status") == "ok":
                        # replace best with this heuristic candidate
                        best = {"label": r.get("label"), "code": r.get("code", ""), "eval": {"energy_j": r.get("energy_j"), "runtime_ms": r.get("runtime_ms", 0), "algorithm_class": base_eval.get("algorithm_class")}}
                        break
            elapsed_ms = (time.time() - started) * 1000.0

        base_energy = base_eval["energy_j"]
        best_energy = best["eval"]["energy_j"]
        delta_energy = base_energy - best_energy
        pct = (delta_energy / base_energy * 100.0) if base_energy > 0 else 0.0

        base_carbon = calculate_emissions_gco2eq(base_energy, intensity)
        best_carbon = calculate_emissions_gco2eq(best_energy, intensity)

        # Try measured runtime on both original and selected best candidate.
        measured_base = _measure_runtime_safe(original_code, input_n, base_eval["algorithm_class"])
        measured_best = _measure_runtime_safe(best["code"], input_n, best["eval"]["algorithm_class"])

        measured_delta_ms = None
        measured_speedup = None
        if measured_base["ok"] and measured_best["ok"]:
            measured_delta_ms = measured_base["runtime_ms"] - measured_best["runtime_ms"]
            if measured_best["runtime_ms"] and measured_best["runtime_ms"] > 0:
                measured_speedup = measured_base["runtime_ms"] / measured_best["runtime_ms"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Energy", f"{base_energy:.6f} J", f"{best_energy:.6f} J best")
        m2.metric("Delta", f"{delta_energy:.6f} J", f"{pct:.2f}% reduction")
        m3.metric("Carbon", f"{base_carbon:.6f} gCO2eq", f"{best_carbon:.6f} gCO2eq")
        m4.metric("Loop runtime", f"{elapsed_ms:.1f} ms", f"{len(rounds)} rounds")

        analytics_cols = st.columns(3)
        analytics_cols[0].metric(
            "Proxy runtime delta",
            f"{(base_eval['runtime_ms'] - best['eval']['runtime_ms']):.3f} ms",
            f"{base_eval['runtime_ms']:.3f} -> {best['eval']['runtime_ms']:.3f} ms",
        )
        if measured_delta_ms is not None:
            speedup_text = f"{measured_speedup:.2f}x" if measured_speedup is not None else "n/a"
            analytics_cols[1].metric(
                "Measured runtime delta",
                f"{measured_delta_ms:.3f} ms",
                speedup_text,
            )
            analytics_cols[2].metric(
                "Measured runtime (orig -> best)",
                f"{measured_base['runtime_ms']:.3f} -> {measured_best['runtime_ms']:.3f} ms",
                measured_best.get("mode", "unknown"),
            )
        else:
            analytics_cols[1].metric("Measured runtime delta", "n/a", "measurement failed")
            analytics_cols[2].metric("Measured runtime (orig -> best)", "n/a", "see details below")

            certificate_payload = build_certificate_payload(
                original_code=original_code,
                best=best,
                base_eval=base_eval,
                base_measured=measured_base,
                measured_best=measured_best,
                rounds=rounds,
                input_n=input_n,
                tdp=tdp,
                cores=cores,
                intensity=intensity,
                model_name=model_name,
                dataset_name=dataset_name,
                elapsed_ms=elapsed_ms,
                project_manifest=project_manifest,
                selected_project_file=st.session_state.selected_project_file,
            )

            try:
                certificate_bytes = build_certificate_pdf(certificate_payload)
                safe_project_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(certificate_payload.get("project_name", "ecologic")))
                certificate_filename = f"ecologic_certificate_{safe_project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.session_state.last_certificate_bytes = certificate_bytes
                st.session_state.last_certificate_filename = certificate_filename
                # persist certificate to workspace folder so it's visible outside the app session
                try:
                    cert_dir = Path("certificates")
                    cert_dir.mkdir(exist_ok=True)
                    out_path = cert_dir / certificate_filename
                    with open(out_path, "wb") as f:
                        f.write(certificate_bytes)
                except Exception:
                    # non-fatal: keep UI download, but filesystem write may fail on read-only mounts
                    pass
                st.download_button(
                    "Download PDF certificate",
                    data=certificate_bytes,
                    file_name=certificate_filename,
                    mime="application/pdf",
                    help="Download a 1-2 page PDF summary of the optimization run.",
                )
            except Exception as exc:
                st.warning(f"PDF certificate unavailable: {exc}")

        st.subheader("Side-by-side optimization outcome")
        st.write(f"{base_energy:.6f} J -> {best_energy:.6f} J")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Original code**")
            st.code(original_code, language="python")
            st.caption(f"Inferred class: {base_eval['algorithm_class']}")
        with col_b:
            st.markdown(f"**Best candidate ({best['label']})**")
            st.code(best["code"], language="python")
            st.caption(f"Inferred class: {best['eval']['algorithm_class']}")
            st.caption(f"Selection reason: {best.get('selection_reason', 'unknown')}")

        st.subheader("Closed-loop rounds")
        rounds_df = pd.DataFrame(rounds)
        st.dataframe(rounds_df, width="stretch")

        with st.expander("Runtime measurement details", expanded=False):
            st.json(
                {
                    "original": measured_base,
                    "best_candidate": measured_best,
                }
            )

        st.subheader("Energy-Time Pareto Frontier")
        sampled = dataset.sample(min(500, len(dataset)), random_state=42).copy()
        sampled_points = sampled[["runtime_ms_proxy", "target_energy_joules", "algorithm_class"]].rename(
            columns={
                "runtime_ms_proxy": "runtime_ms",
                "target_energy_joules": "energy_j",
            }
        )
        sampled_points["label"] = "benchmark"

        custom_points = pd.DataFrame(
            [
                {
                    "runtime_ms": base_eval["runtime_ms"],
                    "energy_j": base_eval["energy_j"],
                    "algorithm_class": base_eval["algorithm_class"],
                    "label": "original",
                },
                {
                    "runtime_ms": best["eval"]["runtime_ms"],
                    "energy_j": best["eval"]["energy_j"],
                    "algorithm_class": best["eval"]["algorithm_class"],
                    "label": "optimized",
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
                hovertemplate="runtime=%{x:.3f} ms<br>energy=%{y:.6f} J<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[base_eval["runtime_ms"]],
                y=[base_eval["energy_j"]],
                mode="markers",
                name="Original",
                marker={"size": 12, "color": "#d62728", "symbol": "diamond"},
                hovertemplate="Original<br>runtime=%{x:.3f} ms<br>energy=%{y:.6f} J<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[best["eval"]["runtime_ms"]],
                y=[best["eval"]["energy_j"]],
                mode="markers",
                name="Optimized",
                marker={"size": 12, "color": "#2ca02c", "symbol": "star"},
                hovertemplate="Optimized<br>runtime=%{x:.3f} ms<br>energy=%{y:.6f} J<extra></extra>",
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
            xaxis_title="Execution time (ms, proxy)",
            yaxis_title="Energy (J)",
            template="plotly_white",
            legend={"orientation": "h", "y": 1.02, "x": 0.0},
            height=520,
        )
        st.plotly_chart(fig, width="stretch")

        st.caption(
            "Runtime axis currently uses a static proxy derived from algorithm class and input scale. "
            "Replace with measured execution time in Phase 1.5 for publication-grade results."
        )


if __name__ == "__main__":
    main()
