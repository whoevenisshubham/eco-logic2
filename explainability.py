from __future__ import annotations

from dataclasses import dataclass, field
import ast as py_ast
import html
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


LEGACY_FEATURE_NAMES = [
    "max_loop_depth",
    "busy_wait_score",
    "allocation_pressure",
    "stride_penalty",
    "sync_score",
    "vector_ops_count",
    "input_n",
    "tdp",
    "cores",
]


@dataclass
class VisualASTNode:
    node_type: str
    expression: str
    line_no: int = 0
    indentation: int = 0
    children: List["VisualASTNode"] = field(default_factory=list)


def detect_language(code_text: str) -> str:
    low = (code_text or "").lower()
    if "#include" in low or "std::" in low or "cout" in low or "using namespace" in low or "::" in low:
        return "C++"
    if "def " in low or "import " in low or "numpy" in low or "pandas" in low:
        return "Python"
    if "public static void main" in low or "system.out.println" in low:
        return "Java"
    return "Python"


def _escape_dot_label(text: str) -> str:
    text = (text or "").replace("\\", "\\\\").replace('"', '\\"')
    return text.replace("\n", "\\n")


def _truncate(text: str, max_len: int = 70) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _build_line_ast(code_text: str) -> VisualASTNode:
    lines = code_text.splitlines()
    root = VisualASTNode("Root", "Root", 0, -1)
    stack = [root]

    for idx, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
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

        node = VisualASTNode(node_type=node_type, expression=stripped, line_no=idx, indentation=indent)
        while len(stack) > 1 and stack[-1].indentation >= indent:
            stack.pop()
        stack[-1].children.append(node)
        if node_type in {"Loop", "Branch"}:
            stack.append(node)

    return root


def _build_python_ast(code_text: str) -> VisualASTNode:
    parsed = py_ast.parse(code_text)
    root = VisualASTNode("Module", "Module", 0, -1)

    def visit(node: py_ast.AST, parent: VisualASTNode, depth: int) -> None:
        node_type = type(node).__name__
        expr = getattr(node, "name", "") or getattr(node, "id", "") or node_type
        line_no = int(getattr(node, "lineno", 0) or 0)
        current = VisualASTNode(node_type=node_type, expression=str(expr), line_no=line_no, indentation=depth)
        parent.children.append(current)
        for child in py_ast.iter_child_nodes(node):
            visit(child, current, depth + 1)

    for child in py_ast.iter_child_nodes(parsed):
        visit(child, root, 0)
    return root


def build_ast_preview(code_text: str, language_name: Optional[str] = None) -> Dict[str, Any]:
    language_name = language_name or detect_language(code_text)
    try:
        if language_name == "Python":
            root = _build_python_ast(code_text)
        else:
            root = _build_line_ast(code_text)
    except Exception:
        root = _build_line_ast(code_text)

    node_count = 0
    max_depth = 0

    def walk(node: VisualASTNode, depth: int) -> None:
        nonlocal node_count, max_depth
        node_count += 1
        max_depth = max(max_depth, depth)
        for child in node.children:
            walk(child, depth + 1)

    walk(root, 0)
    return {
        "root": root,
        "language": language_name,
        "node_count": node_count,
        "max_depth": max_depth,
    }


def ast_to_dot(root: VisualASTNode) -> str:
    lines = [
        "digraph AST {",
        "rankdir=TB;",
        'node [shape=box, style="rounded,filled", fillcolor="#eef5ff", color="#4a6fa5", fontname="Arial"];',
    ]
    counter = 0

    def walk(node: VisualASTNode, parent_id: Optional[str] = None) -> None:
        nonlocal counter
        node_id = f"n{counter}"
        counter += 1
        label_parts = [node.node_type]
        if node.line_no > 0:
            label_parts.append(f"L{node.line_no}")
        expr = _truncate(node.expression, 90)
        if expr and expr != node.node_type:
            label_parts.append(expr)
        label = "\\n".join(label_parts)
        lines.append(f'{node_id} [label="{_escape_dot_label(label)}"];')
        if parent_id is not None:
            lines.append(f"{parent_id} -> {node_id};")
        for child in node.children:
            walk(child, node_id)

    walk(root)
    lines.append("}")
    return "\n".join(lines)


def _coerce_feature_values(feature_vector: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(feature_vector), dtype=float).reshape(-1)
    if arr.size != len(LEGACY_FEATURE_NAMES):
        raise ValueError(f"Expected {len(LEGACY_FEATURE_NAMES)} legacy features, got {arr.size}")
    return arr


def compute_shap_summary(
    model: Any,
    feature_vector: Sequence[float],
    feature_names: Sequence[str] = LEGACY_FEATURE_NAMES,
    baseline: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    values = _coerce_feature_values(feature_vector)
    feature_names = list(feature_names)

    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(values.reshape(1, -1))
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        arr = np.asarray(shap_values, dtype=float)
        if arr.ndim == 2:
            arr = arr[0]
        expected = explainer.expected_value
        if isinstance(expected, (list, tuple, np.ndarray)):
            expected_value = float(np.asarray(expected, dtype=float).mean())
        else:
            expected_value = float(expected)
        contributions = [
            {"feature": name, "value": float(val), "contribution": float(shap_val)}
            for name, val, shap_val in zip(feature_names, values.tolist(), arr.tolist())
        ]
        method = "shap.TreeExplainer"
    except Exception:
        importances = np.asarray(getattr(model, "feature_importances_", np.ones_like(values) / max(len(values), 1)), dtype=float)
        if importances.size != values.size:
            importances = np.resize(importances, values.size)
        if baseline is None:
            baseline_arr = np.zeros_like(values)
        else:
            baseline_arr = np.asarray(list(baseline), dtype=float).reshape(-1)
            if baseline_arr.size != values.size:
                baseline_arr = np.resize(baseline_arr, values.size)
        arr = (values - baseline_arr) * importances
        expected_value = float(np.mean(values))
        contributions = [
            {"feature": name, "value": float(val), "contribution": float(contrib)}
            for name, val, contrib in zip(feature_names, values.tolist(), arr.tolist())
        ]
        method = "approximate.feature_importance"

    contributions = sorted(contributions, key=lambda item: abs(item["contribution"]), reverse=True)
    predicted_value = None
    try:
        predicted_value = float(model.predict(values.reshape(1, -1))[0])
    except Exception:
        predicted_value = None

    return {
        "method": method,
        "expected_value": expected_value,
        "predicted_value": predicted_value,
        "feature_names": feature_names,
        "contributions": contributions,
        "top_contributions": contributions[: min(5, len(contributions))],
    }
