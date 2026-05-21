from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from tree_sitter_languages import get_parser
except Exception:  # pragma: no cover - optional dependency fallback
    get_parser = None


_DECISION_NODE_HINTS = (
    "if",
    "elif",
    "else_if",
    "while",
    "for",
    "case",
    "switch",
    "catch",
    "conditional",
    "ternary",
    "logical",
)

_LOOP_NODE_HINTS = (
    "for",
    "while",
    "do_statement",
    "range_for",
    "for_in_clause",
)

_BRANCH_NODE_HINTS = (
    "if",
    "switch",
    "case",
    "elif",
    "catch",
    "conditional",
)

_ALLOCATION_NODE_HINTS = (
    "new",
    "delete",
    "malloc",
    "calloc",
    "realloc",
    "make_unique",
    "make_shared",
    "push_back",
    "emplace_back",
    "append",
)

_VECTOR_OP_HINTS = (
    "matmul",
    "dot",
    "linalg",
    "transform",
    "accumulate",
    "inner_product",
)

_OPERATOR_TOKENS = {
    "+",
    "-",
    "*",
    "/",
    "%",
    "=",
    "==",
    "!=",
    "<",
    ">",
    "<=",
    ">=",
    "&&",
    "||",
    "!",
    "++",
    "--",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "<<",
    ">>",
    "&",
    "|",
    "^",
    "::",
    ".",
    "->",
    "[",
    "]",
    "(",
    ")",
    "?",
    ":",
}

_OPERAND_NODE_TYPES = {
    "identifier",
    "field_identifier",
    "number_literal",
    "integer_literal",
    "floating_point_literal",
    "string_literal",
    "character_literal",
    "true",
    "false",
    "null",
    "null_literal",
    "boolean_literal",
}


def detect_language(code_text: str) -> str:
    low = code_text.lower()
    if "#include" in low or "std::" in low or "cout" in low or "using namespace" in low or "::" in low:
        return "C++"
    if "def " in low or "import " in low or "numpy" in low or "pandas" in low:
        return "Python"
    if "public static void main" in low or "system.out.println" in low:
        return "Java"
    return "Python"


def _language_key(language_name: str) -> str:
    if language_name == "C++":
        return "cpp"
    if language_name == "Java":
        return "java"
    return "python"


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _safe_text(code_bytes: bytes, node: Any) -> str:
    try:
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _node_matches(node_type: str, hints: Sequence[str]) -> bool:
    low = node_type.lower()
    return any(hint in low for hint in hints)


def _extract_loop_variables(language_name: str, node_type: str, text: str) -> List[str]:
    text = _normalize_whitespace(text)
    if not text:
        return []
    if language_name == "Python":
        match = re.search(r"for\s+(.+?)\s+in\s+", text)
        if not match:
            return []
        raw = match.group(1)
        raw = raw.replace("[", " ").replace("]", " ")
        return [part.strip() for part in re.split(r"[,\s]+", raw) if part.strip() and part.strip() not in {"(", ")"}]
    if language_name == "C++":
        match = re.search(r"for\s*\(([^;]*);", text)
        if not match:
            return []
        init = match.group(1)
        candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", init)
        if not candidates:
            return []
        keywords = {
            "auto",
            "int",
            "size_t",
            "long",
            "short",
            "unsigned",
            "signed",
            "const",
            "static",
            "constexpr",
            "volatile",
            "mutable",
        }
        filtered = [candidate for candidate in candidates if candidate not in keywords]
        return filtered[-1:] if filtered else []
    return []


def _is_loop_node(node_type: str) -> bool:
    return _node_matches(node_type, _LOOP_NODE_HINTS)


def _is_branch_node(node_type: str) -> bool:
    return _node_matches(node_type, _BRANCH_NODE_HINTS)


def _is_allocation_node(node_type: str, text: str) -> bool:
    return _node_matches(node_type, _ALLOCATION_NODE_HINTS) or _node_matches(text, _ALLOCATION_NODE_HINTS)


def _is_vector_op(text: str) -> bool:
    return any(hint in text.lower() for hint in _VECTOR_OP_HINTS)


def _is_operator_token(token: str) -> bool:
    return token in _OPERATOR_TOKENS


def _collect_leaf_tokens(node: Any, code_bytes: bytes, tokens: List[Tuple[str, str]]) -> None:
    if getattr(node, "child_count", 0) == 0:
        node_type = getattr(node, "type", "")
        text = _safe_text(code_bytes, node)
        if node_type in _OPERAND_NODE_TYPES:
            tokens.append(("operand", text or node_type))
        elif _is_operator_token(text.strip()):
            tokens.append(("operator", text.strip()))
        elif text.strip() in {"+", "-", "*", "/", "%", "=", "==", "!=", "<", ">", "<=", ">=", "&&", "||", "!", "::", ".", "->", "?", ":", ","}:
            tokens.append(("operator", text.strip()))
        elif node_type in {"call", "call_expression", "function_call", "subscript_expression", "subscript", "field_expression", "binary_expression"} and text.strip():
            tokens.append(("operator", node_type))
        return

    for child in node.children:
        _collect_leaf_tokens(child, code_bytes, tokens)


def _traverse(node: Any, code_bytes: bytes, language_name: str, depth: int, state: Dict[str, Any]) -> None:
    node_type = getattr(node, "type", "")
    text = _normalize_whitespace(_safe_text(code_bytes, node))
    low_text = text.lower()

    state["node_count"] += 1
    state["max_branching_factor"] = max(state["max_branching_factor"], getattr(node, "child_count", 0))
    if getattr(node, "child_count", 0) > 0:
        state["branching_factor_total"] += getattr(node, "child_count", 0)
        state["branching_factor_nodes"] += 1

    if node_type in {"function_definition", "function_declaration", "method_definition", "lambda_expression"}:
        state["function_count"] += 1

    if _is_loop_node(node_type):
        state["loop_count"] += 1
        state["max_loop_depth"] = max(state["max_loop_depth"], depth + 1)
        vars_in_scope = _extract_loop_variables(language_name, node_type, text)
        next_loop_stack = state["loop_stack"] + [vars_in_scope[0] if vars_in_scope else None]
    else:
        next_loop_stack = state["loop_stack"]

    if _is_branch_node(node_type):
        state["branch_count"] += 1
        state["decision_count"] += 1

    if _is_allocation_node(node_type, low_text):
        state["allocation_count"] += 1
        if state["loop_stack"]:
            state["allocation_in_loops"] += 1

    if _is_vector_op(low_text):
        state["vector_ops_count"] += 1

    if node_type in {"subscript_expression", "subscript"}:
        score = _stride_penalty_for_access(text, state["loop_stack"])
        state["stride_penalty_total"] += score
        state["stride_penalty_samples"] += 1

    if _is_operator_token(text):
        state["operator_tokens"].append(text)

    if node_type in _OPERAND_NODE_TYPES and text:
        state["operand_tokens"].append(text)

    previous_stack = state["loop_stack"]
    state["loop_stack"] = next_loop_stack
    for child in getattr(node, "children", []):
        _traverse(child, code_bytes, language_name, depth + (1 if _is_loop_node(node_type) else 0), state)
    state["loop_stack"] = previous_stack


def _stride_penalty_for_access(access_text: str, loop_stack: Sequence[Optional[str]]) -> float:
    access_text = access_text.strip()
    if not access_text or not loop_stack:
        return 0.0
    loop_vars = [name for name in loop_stack if name]
    if not loop_vars:
        return 0.2

    hits: List[Tuple[int, str]] = []
    for var in loop_vars:
        match = re.search(rf"\b{re.escape(var)}\b", access_text)
        if match:
            hits.append((match.start(), var))

    if not hits:
        return 0.1

    positions = [pos for pos, _ in hits]
    if positions == sorted(positions):
        return 0.0 if len(hits) > 1 else 0.15
    if positions == sorted(positions, reverse=True):
        return 1.0 if len(hits) > 1 else 0.6
    return 0.5


def _fallback_feature_bundle(code_text: str, input_n: float, tdp: float, cores: float) -> Dict[str, Any]:
    low = code_text.lower()
    loop_count = low.count("for ") + low.count("while ")
    branch_count = low.count(" if ") + low.count("elif ") + low.count("else:") + low.count("switch")
    allocation_count = sum(low.count(token) for token in ["new ", "malloc", "calloc", "realloc", "append(", "push_back(", "emplace_back("])
    stride_penalty = 1.0 if any(pattern in low for pattern in ["[j][i]", "matrix[j][i]", "arr[j][i]"]) else 0.2 if "[" in low else 0.0
    cyclomatic_complexity = 1 + branch_count + loop_count
    halstead_volume = float(max(len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S", code_text)), 1))
    return {
        "language": detect_language(code_text),
        "parser_backend": "fallback-regex",
        "node_count": len(code_text.splitlines()),
        "function_count": max(code_text.count("def "), code_text.count("void ")),
        "loop_count": loop_count,
        "branch_count": branch_count,
        "decision_count": branch_count + loop_count,
        "cyclomatic_complexity": cyclomatic_complexity,
        "max_loop_depth": min(loop_count, 3),
        "avg_branching_factor": float(branch_count + loop_count + 1),
        "max_branching_factor": float(max(branch_count, loop_count, 1)),
        "total_operators": allocation_count + branch_count + loop_count,
        "distinct_operators": max(1, min(allocation_count + branch_count + loop_count, 8)),
        "total_operands": len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", code_text)),
        "distinct_operands": len(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", code_text))),
        "halstead_vocabulary": 1.0,
        "halstead_length": halstead_volume,
        "halstead_volume": halstead_volume * math.log2(max(2, halstead_volume)),
        "allocation_count": allocation_count,
        "allocation_in_loops": 1 if allocation_count and loop_count else 0,
        "memory_pressure": float(allocation_count),
        "stride_penalty": stride_penalty,
        "vector_ops_count": 1 if any(hint in low for hint in _VECTOR_OP_HINTS) else 0,
        "busy_wait_score": 1.0 if any(token in low for token in ["pass", "continue", "sleep"]) and loop_count else 0.0,
        "sync_score": 1.0 if any(token in low for token in ["lock", "mutex", "synchronized"]) else 0.0,
        "input_n": float(input_n),
        "tdp": float(tdp),
        "cores": float(cores),
        "parse_error": False,
    }


def analyze_code_features(code_text: str, input_n: float = 10000.0, tdp: float = 45.0, cores: float = 4.0, language_name: Optional[str] = None) -> Dict[str, Any]:
    language_name = language_name or detect_language(code_text)
    parser_backend = "fallback-regex"
    if get_parser is None:
        bundle = _fallback_feature_bundle(code_text, input_n, tdp, cores)
        bundle["parser_backend"] = parser_backend
        return bundle

    try:
        parser = get_parser(_language_key(language_name))
        code_bytes = code_text.encode("utf-8", errors="ignore")
        tree = parser.parse(code_bytes)
        root = tree.root_node
        state: Dict[str, Any] = {
            "node_count": 0,
            "function_count": 0,
            "loop_count": 0,
            "branch_count": 0,
            "decision_count": 0,
            "max_loop_depth": 0,
            "branching_factor_total": 0,
            "branching_factor_nodes": 0,
            "max_branching_factor": 0,
            "allocation_count": 0,
            "allocation_in_loops": 0,
            "stride_penalty_total": 0.0,
            "stride_penalty_samples": 0,
            "vector_ops_count": 0,
            "operator_tokens": [],
            "operand_tokens": [],
            "loop_stack": [],
        }
        _traverse(root, code_bytes, language_name, 0, state)

        tokens: List[Tuple[str, str]] = []
        _collect_leaf_tokens(root, code_bytes, tokens)
        operator_tokens = [token for kind, token in tokens if kind == "operator"]
        operand_tokens = [token for kind, token in tokens if kind == "operand"]
        distinct_operators = len(set(operator_tokens))
        distinct_operands = len(set(operand_tokens))
        total_operators = len(operator_tokens)
        total_operands = len(operand_tokens)
        vocabulary = distinct_operators + distinct_operands
        length = total_operators + total_operands
        halstead_volume = float(length * math.log2(max(vocabulary, 2)))
        avg_branching_factor = (
            state["branching_factor_total"] / state["branching_factor_nodes"]
            if state["branching_factor_nodes"]
            else 0.0
        )
        stride_penalty = (
            state["stride_penalty_total"] / state["stride_penalty_samples"]
            if state["stride_penalty_samples"]
            else 0.0
        )
        cyclomatic_complexity = max(1, 1 + state["decision_count"])
        memory_pressure = state["allocation_count"] / max(state["function_count"], 1)
        busy_wait_score = 1.0 if any(token in code_text.lower() for token in ["pass", "continue", "sleep"]) and state["loop_count"] else 0.0
        sync_score = 1.0 if any(token in code_text.lower() for token in ["lock", "mutex", "synchronized"]) else 0.0

        bundle = {
            "language": language_name,
            "parser_backend": f"tree-sitter:{_language_key(language_name)}",
            "node_count": state["node_count"],
            "function_count": state["function_count"],
            "loop_count": state["loop_count"],
            "branch_count": state["branch_count"],
            "decision_count": state["decision_count"],
            "cyclomatic_complexity": cyclomatic_complexity,
            "max_loop_depth": state["max_loop_depth"],
            "avg_branching_factor": avg_branching_factor,
            "max_branching_factor": state["max_branching_factor"],
            "total_operators": total_operators,
            "distinct_operators": distinct_operators,
            "total_operands": total_operands,
            "distinct_operands": distinct_operands,
            "halstead_vocabulary": vocabulary,
            "halstead_length": length,
            "halstead_volume": halstead_volume,
            "allocation_count": state["allocation_count"],
            "allocation_in_loops": state["allocation_in_loops"],
            "memory_pressure": memory_pressure,
            "stride_penalty": stride_penalty,
            "vector_ops_count": state["vector_ops_count"],
            "busy_wait_score": busy_wait_score,
            "sync_score": sync_score,
            "input_n": float(input_n),
            "tdp": float(tdp),
            "cores": float(cores),
            "parse_error": False,
        }
        return bundle
    except Exception:
        bundle = _fallback_feature_bundle(code_text, input_n, tdp, cores)
        bundle["parser_backend"] = f"fallback-regex:{language_name.lower()}"
        bundle["parse_error"] = True
        return bundle


def legacy_model_vector(bundle: Dict[str, Any]) -> List[float]:
    return [
        float(max(0.0, bundle.get("max_loop_depth", 0))),
        float(bundle.get("busy_wait_score", 0.0)),
        float(min(bundle.get("allocation_in_loops", 0) + bundle.get("memory_pressure", 0.0), 3.0)),
        float(max(0.0, min(bundle.get("stride_penalty", 0.0), 1.0))),
        float(max(0.0, min(bundle.get("sync_score", 0.0), 1.0))),
        float(max(0.0, min(bundle.get("vector_ops_count", 0.0), 3.0))),
        float(bundle.get("input_n", 10000.0)),
        float(bundle.get("tdp", 45.0)),
        float(bundle.get("cores", 4.0)),
    ]


def rich_feature_rows(bundle: Dict[str, Any]) -> List[Tuple[str, float]]:
    keys = [
        "max_loop_depth",
        "cyclomatic_complexity",
        "avg_branching_factor",
        "max_branching_factor",
        "allocation_count",
        "allocation_in_loops",
        "memory_pressure",
        "stride_penalty",
        "halstead_volume",
        "distinct_operators",
        "distinct_operands",
        "vector_ops_count",
        "busy_wait_score",
        "sync_score",
        "input_n",
        "tdp",
        "cores",
    ]
    rows: List[Tuple[str, float]] = []
    for key in keys:
        value = bundle.get(key, 0.0)
        if isinstance(value, (int, float)):
            rows.append((key, float(value)))
    return rows
