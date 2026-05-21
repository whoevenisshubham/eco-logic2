from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

from phase2_features import analyze_code_features, detect_language

try:
    from tree_sitter_languages import get_parser
except Exception:  # pragma: no cover - optional dependency fallback
    get_parser = None


EDGE_PARENT_CHILD = 0
EDGE_NEXT_TOKEN = 1
EDGE_DATA_FLOW = 2


def _seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _safe_text(code_bytes: bytes, node) -> str:
    try:
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _language_key(language_name: str) -> str:
    if language_name == "C++":
        return "cpp"
    if language_name == "Java":
        return "java"
    return "python"


def _is_identifier_node(node_type: str) -> bool:
    low = node_type.lower()
    return "identifier" in low or low.endswith("_name") or low in {"variable", "name"}


def _hash_token(token: str, bucket_size: int) -> int:
    if not token:
        return 0
    digest = hashlib.sha1(token.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16) % bucket_size


@dataclass
class GraphExample:
    snippet_id: int
    language: str
    algorithm_class: str
    target: float
    node_type_ids: torch.Tensor
    token_ids: torch.Tensor
    numeric_features: torch.Tensor
    edge_index: torch.Tensor
    edge_types: torch.Tensor
    graph_features: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.numeric_features.shape[0])


class ASTGraphBuilder:
    def __init__(self, token_buckets: int = 4096, include_data_flow: bool = True, include_cache_locality: bool = True):
        self.token_buckets = token_buckets
        self.include_data_flow = include_data_flow
        self.include_cache_locality = include_cache_locality
        self._parser_cache: Dict[str, object] = {}
        self._node_type_vocab: Dict[str, int] = {}

    def _get_parser(self, language_name: str):
        if get_parser is None:
            return None
        key = _language_key(language_name)
        if key not in self._parser_cache:
            self._parser_cache[key] = get_parser(key)
        return self._parser_cache[key]

    def _type_id(self, node_type: str) -> int:
        if node_type not in self._node_type_vocab:
            self._node_type_vocab[node_type] = len(self._node_type_vocab) + 1
        return self._node_type_vocab[node_type]

    def build(self, code_text: str, snippet_id: int, target: float, input_n: float, tdp: float, cores: float, algorithm_class: str, language_name: Optional[str] = None) -> GraphExample:
        language_name = language_name or detect_language(code_text)
        feature_bundle = analyze_code_features(code_text, input_n=input_n, tdp=tdp, cores=cores, language_name=language_name)

        parser = self._get_parser(language_name)
        if parser is None:
            return self._fallback_graph(code_text, snippet_id, target, input_n, tdp, cores, algorithm_class, language_name, feature_bundle)

        code_bytes = code_text.encode("utf-8", errors="ignore")
        try:
            tree = parser.parse(code_bytes)
            root = tree.root_node
        except Exception:
            return self._fallback_graph(code_text, snippet_id, target, input_n, tdp, cores, algorithm_class, language_name, feature_bundle)

        nodes: List[Dict[str, float]] = []
        edges: List[Tuple[int, int, int]] = []
        identifier_occurrences: Dict[str, List[int]] = {}
        preorder: List[int] = []

        def walk(node, depth: int, parent_idx: Optional[int], sibling_index: int) -> int:
            idx = len(nodes)
            node_type = getattr(node, "type", "unknown")
            text = _clean_text(_safe_text(code_bytes, node))
            is_leaf = 1.0 if getattr(node, "child_count", 0) == 0 else 0.0
            token_len = float(min(len(text), 64)) / 64.0
            type_id = float(self._type_id(node_type))
            token_id = float(_hash_token(text or node_type, self.token_buckets)) / float(self.token_buckets)
            line_norm = float(getattr(node, "start_point", (0, 0))[0] + 1) / max(code_text.count("\n") + 1, 1)
            pos_norm = float(getattr(node, "start_byte", 0)) / max(len(code_bytes), 1)
            subtree_size_placeholder = float(getattr(node, "child_count", 0) + 1)
            numeric = [
                type_id,
                token_id,
                float(depth) / 20.0,
                float(sibling_index) / 20.0,
                line_norm,
                pos_norm,
                subtree_size_placeholder / 20.0,
                is_leaf,
                token_len,
            ]
            nodes.append({
                "type_id": type_id,
                "token_id": token_id,
                "numeric": numeric,
                "text": text,
                "node_type": node_type,
            })
            preorder.append(idx)

            if parent_idx is not None:
                edges.append((parent_idx, idx, EDGE_PARENT_CHILD))
                edges.append((idx, parent_idx, EDGE_PARENT_CHILD))

            if _is_identifier_node(node_type):
                key = text or node_type
                identifier_occurrences.setdefault(key, []).append(idx)

            child_index = 0
            for child in getattr(node, "children", []):
                walk(child, depth + 1, idx, child_index)
                child_index += 1
            return idx

        walk(root, 0, None, 0)

        for src, dst in zip(preorder[:-1], preorder[1:]):
            edges.append((src, dst, EDGE_NEXT_TOKEN))
            edges.append((dst, src, EDGE_NEXT_TOKEN))

        if self.include_data_flow:
            for positions in identifier_occurrences.values():
                if len(positions) < 2:
                    continue
                for src, dst in zip(positions[:-1], positions[1:]):
                    edges.append((src, dst, EDGE_DATA_FLOW))
                    edges.append((dst, src, EDGE_DATA_FLOW))

        if not nodes:
            return self._fallback_graph(code_text, snippet_id, target, input_n, tdp, cores, algorithm_class, language_name, feature_bundle)

        node_type_ids = torch.tensor([int(node["type_id"]) for node in nodes], dtype=torch.long)
        token_ids = torch.tensor([int(round(node["token_id"] * self.token_buckets)) for node in nodes], dtype=torch.long)
        numeric_features = torch.tensor([node["numeric"] for node in nodes], dtype=torch.float32)
        if edges:
            edge_index = torch.tensor([[src for src, _, _ in edges], [dst for _, dst, _ in edges]], dtype=torch.long)
            edge_types = torch.tensor([edge_type for _, _, edge_type in edges], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_types = torch.zeros((0,), dtype=torch.long)

        graph_features = self._graph_features(feature_bundle)
        return GraphExample(
            snippet_id=snippet_id,
            language=language_name,
            algorithm_class=algorithm_class,
            target=float(target),
            node_type_ids=node_type_ids,
            token_ids=token_ids,
            numeric_features=numeric_features,
            edge_index=edge_index,
            edge_types=edge_types,
            graph_features=graph_features,
        )

    def _graph_features(self, feature_bundle: Dict[str, float]) -> torch.Tensor:
        stride_penalty = feature_bundle.get("stride_penalty", 0.0) if self.include_cache_locality else 0.0
        values = [
            feature_bundle.get("input_n", 0.0),
            feature_bundle.get("tdp", 0.0),
            feature_bundle.get("cores", 0.0),
            feature_bundle.get("cyclomatic_complexity", 0.0),
            stride_penalty,
            feature_bundle.get("allocation_count", 0.0),
            feature_bundle.get("max_loop_depth", 0.0),
            feature_bundle.get("memory_pressure", 0.0),
            feature_bundle.get("vector_ops_count", 0.0),
        ]
        return torch.tensor(values, dtype=torch.float32)

    def _fallback_graph(self, code_text: str, snippet_id: int, target: float, input_n: float, tdp: float, cores: float, algorithm_class: str, language_name: str, feature_bundle: Dict[str, float]) -> GraphExample:
        lines = [line for line in code_text.splitlines() if line.strip()]
        nodes: List[Dict[str, float]] = []
        edges: List[Tuple[int, int, int]] = []
        for idx, line in enumerate(lines or [code_text]):
            type_id = float(self._type_id("fallback_line"))
            token_id = float(_hash_token(line, self.token_buckets)) / float(self.token_buckets)
            nodes.append({
                "type_id": type_id,
                "token_id": token_id,
                "numeric": [
                    type_id,
                    token_id,
                    0.0,
                    float(idx) / 20.0,
                    float(idx + 1) / max(len(lines), 1),
                    0.0,
                    1.0,
                    1.0,
                    float(min(len(line), 64)) / 64.0,
                ],
                "text": line,
                "node_type": "fallback_line",
            })
            if idx > 0:
                edges.append((idx - 1, idx, EDGE_NEXT_TOKEN))
                edges.append((idx, idx - 1, EDGE_NEXT_TOKEN))

        node_type_ids = torch.tensor([int(node["type_id"]) for node in nodes], dtype=torch.long)
        token_ids = torch.tensor([int(round(node["token_id"] * self.token_buckets)) for node in nodes], dtype=torch.long)
        numeric_features = torch.tensor([node["numeric"] for node in nodes], dtype=torch.float32)
        if edges:
            edge_index = torch.tensor([[src for src, _, _ in edges], [dst for _, dst, _ in edges]], dtype=torch.long)
            edge_types = torch.tensor([edge_type for _, _, edge_type in edges], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_types = torch.zeros((0,), dtype=torch.long)

        graph_features = self._graph_features(feature_bundle)
        return GraphExample(
            snippet_id=snippet_id,
            language=language_name,
            algorithm_class=algorithm_class,
            target=float(target),
            node_type_ids=node_type_ids,
            token_ids=token_ids,
            numeric_features=numeric_features,
            edge_index=edge_index,
            edge_types=edge_types,
            graph_features=graph_features,
        )


class ASTGraphDataset(Dataset):
    def __init__(self, csv_path: str, limit: Optional[int] = None, include_data_flow: bool = True, include_cache_locality: bool = True, builder: Optional[ASTGraphBuilder] = None):
        self.csv_path = csv_path
        self.limit = limit
        self.include_data_flow = include_data_flow
        self.include_cache_locality = include_cache_locality
        self.df = self._load_dataframe(csv_path, limit)
        self.builder = builder or ASTGraphBuilder(include_data_flow=include_data_flow, include_cache_locality=include_cache_locality)
        self.examples = self._build_examples()

    def _load_dataframe(self, csv_path: str, limit: Optional[int]) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        if limit is not None:
            df = df.head(int(limit)).copy()
        if "snippet_id" not in df.columns:
            df = df.reset_index().rename(columns={"index": "snippet_id"})
        if "source_code" not in df.columns:
            raise ValueError("Dataset must contain a source_code column")
        if "target_energy_joules" not in df.columns:
            raise ValueError("Dataset must contain a target_energy_joules column")
        for col in ["input_scale_N", "hardware_tdp", "hardware_cores"]:
            if col not in df.columns:
                df[col] = 1.0
        if "algorithm_class" not in df.columns:
            df["algorithm_class"] = "unknown"
        df["source_code"] = df["source_code"].astype(str)
        df["algorithm_class"] = df["algorithm_class"].astype(str)
        return df

    def _build_examples(self) -> List[GraphExample]:
        examples: List[GraphExample] = []
        for _, row in self.df.iterrows():
            examples.append(
                self.builder.build(
                    code_text=row["source_code"],
                    snippet_id=str(row["snippet_id"]),
                    target=float(row["target_energy_joules"]),
                    input_n=float(row.get("input_scale_N", 1.0)),
                    tdp=float(row.get("hardware_tdp", 1.0)),
                    cores=float(row.get("hardware_cores", 1.0)),
                    algorithm_class=str(row.get("algorithm_class", "unknown")),
                    language_name=detect_language(str(row["source_code"])),
                )
            )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> GraphExample:
        return self.examples[idx]


def collate_graphs(batch: Sequence[GraphExample]) -> Dict[str, torch.Tensor]:
    node_type_ids = []
    token_ids = []
    numeric_features = []
    edge_indices = []
    edge_types = []
    batch_index = []
    graph_features = []
    targets = []
    snippet_ids = []
    graph_offsets = 0

    for graph_idx, example in enumerate(batch):
        node_type_ids.append(example.node_type_ids)
        token_ids.append(example.token_ids)
        numeric_features.append(example.numeric_features)
        graph_features.append(example.graph_features)
        targets.append(torch.tensor(example.target, dtype=torch.float32))
        snippet_ids.append(
            torch.tensor(
                [int(example.snippet_id)] if str(example.snippet_id).isdigit() else [_hash_token(str(example.snippet_id), 2**31 - 1)],
                dtype=torch.long,
            )
        )
        if example.edge_index.numel() > 0:
            edge_indices.append(example.edge_index + graph_offsets)
            edge_types.append(example.edge_types)
        batch_index.append(torch.full((example.num_nodes,), graph_idx, dtype=torch.long))
        graph_offsets += example.num_nodes

    merged_edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2, 0), dtype=torch.long)
    merged_edge_types = torch.cat(edge_types, dim=0) if edge_types else torch.zeros((0,), dtype=torch.long)
    return {
        "node_type_ids": torch.cat(node_type_ids, dim=0),
        "token_ids": torch.cat(token_ids, dim=0),
        "numeric_features": torch.cat(numeric_features, dim=0),
        "edge_index": merged_edge_index,
        "edge_types": merged_edge_types,
        "batch_index": torch.cat(batch_index, dim=0),
        "graph_features": torch.stack(graph_features, dim=0),
        "targets": torch.stack(targets, dim=0),
        "snippet_ids": torch.stack(snippet_ids, dim=0),
    }


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_proj = nn.Linear(in_dim, out_dim)
        self.neigh_proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            out = self.self_proj(x)
            return torch.relu(self.norm(out))

        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones(dst.shape[0], device=x.device, dtype=x.dtype))
        deg = deg.clamp_min(1.0).unsqueeze(-1)
        neigh = agg / deg
        out = self.self_proj(x) + self.neigh_proj(neigh)
        return torch.relu(self.norm(out))


class ASTGNNRegressor(nn.Module):
    def __init__(self, node_type_vocab_size: int, token_vocab_size: int = 4096, numeric_dim: int = 9, graph_feature_dim: int = 9, hidden_dim: int = 64, layers: int = 3, dropout: float = 0.15):
        super().__init__()
        self.node_type_embedding = nn.Embedding(node_type_vocab_size + 2, 16)
        self.token_embedding = nn.Embedding(token_vocab_size + 2, 16)
        self.numeric_proj = nn.Linear(numeric_dim, 32)
        input_dim = 16 + 16 + 32
        self.layers = nn.ModuleList()
        current_dim = input_dim
        for _ in range(layers):
            self.layers.append(GraphSAGELayer(current_dim, hidden_dim))
            current_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.graph_feature_proj = nn.Sequential(
            nn.Linear(graph_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 16, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_type_ids = batch["node_type_ids"]
        token_ids = batch["token_ids"]
        numeric_features = batch["numeric_features"]
        edge_index = batch["edge_index"]
        batch_index = batch["batch_index"]
        graph_features = batch["graph_features"]

        x = torch.cat(
            [
                self.node_type_embedding(node_type_ids),
                self.token_embedding(token_ids),
                self.numeric_proj(numeric_features),
            ],
            dim=-1,
        )
        for layer in self.layers:
            x = self.dropout(layer(x, edge_index))

        pooled = self._mean_pool(x, batch_index, graph_features.shape[0])
        graph_feat = self.graph_feature_proj(graph_features)
        fused = torch.cat([pooled, graph_feat], dim=-1)
        return self.head(fused).squeeze(-1)

    @staticmethod
    def _mean_pool(x: torch.Tensor, batch_index: torch.Tensor, graph_count: int) -> torch.Tensor:
        pooled = torch.zeros((graph_count, x.shape[1]), device=x.device, dtype=x.dtype)
        pooled.index_add_(0, batch_index, x)
        counts = torch.zeros((graph_count,), device=x.device, dtype=x.dtype)
        counts.index_add_(0, batch_index, torch.ones((x.shape[0],), device=x.device, dtype=x.dtype))
        return pooled / counts.clamp_min(1.0).unsqueeze(-1)


def split_by_snippet_id(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = df["snippet_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(splitter.split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    temp_groups = temp_df["snippet_id"].to_numpy()
    temp_ratio = val_ratio / max(1.0 - train_ratio, 1e-8)
    splitter2 = GroupShuffleSplit(n_splits=1, train_size=temp_ratio, random_state=seed + 1)
    val_idx, test_idx = next(splitter2.split(temp_df, groups=temp_groups))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def _evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds: List[float] = []
    targets: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(batch)
            preds.extend(output.detach().cpu().tolist())
            targets.extend(batch["targets"].detach().cpu().tolist())
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds) if len(set(targets)) > 1 else float("nan")
    return {"mse": float(mse), "mae": float(mae), "r2": float(r2)}


def _train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int = 12, lr: float = 1e-3, carbon_aware_objective: bool = False, carbon_weight_alpha: float = 0.35) -> Dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    patience = 3
    bad_epochs = 0

    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            output = model(batch)
            if carbon_aware_objective:
                # prioritize higher-TDP samples as a proxy for higher carbon sensitivity
                tdp = batch["graph_features"][:, 1].detach()
                tdp_norm = tdp / tdp.clamp_min(1.0).max()
                weights = 1.0 + carbon_weight_alpha * tdp_norm
                loss = ((output - batch["targets"]) ** 2 * weights).mean()
            else:
                loss = criterion(output, batch["targets"])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1

        val_metrics = _evaluate_model(model, val_loader, device)
        val_loss = val_metrics["mse"]
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return _evaluate_model(model, val_loader, device)


def _build_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, include_data_flow: bool, include_cache_locality: bool, limit_train: Optional[int] = None, limit_val: Optional[int] = None, limit_test: Optional[int] = None) -> Tuple[ASTGraphDataset, ASTGraphDataset, ASTGraphDataset, DataLoader, DataLoader, DataLoader]:
    train_ds = ASTGraphDataset(_materialize_csv(train_df, limit_train), include_data_flow=include_data_flow, include_cache_locality=include_cache_locality)
    val_ds = ASTGraphDataset(_materialize_csv(val_df, limit_val), include_data_flow=include_data_flow, include_cache_locality=include_cache_locality)
    test_ds = ASTGraphDataset(_materialize_csv(test_df, limit_test), include_data_flow=include_data_flow, include_cache_locality=include_cache_locality)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_graphs)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_graphs)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_graphs)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def _materialize_csv(df: pd.DataFrame, limit: Optional[int]) -> str:
    if limit is not None:
        df = df.head(int(limit)).copy()
    temp_path = Path(os.environ.get("TEMP", ".")) / f"phase3_split_{os.getpid()}_{random.randint(1000, 9999)}.csv"
    df.to_csv(temp_path, index=False)
    return str(temp_path)


def evaluate_baseline_phase1(model_path: str, test_df: pd.DataFrame) -> Dict[str, float]:
    if not Path(model_path).exists():
        return {"mse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    baseline = joblib.load(model_path)
    preds: List[float] = []
    targets: List[float] = []
    from phase2_features import legacy_model_vector

    for _, row in test_df.iterrows():
        bundle = analyze_code_features(
            str(row["source_code"]),
            input_n=float(row.get("input_scale_N", 1.0)),
            tdp=float(row.get("hardware_tdp", 1.0)),
            cores=float(row.get("hardware_cores", 1.0)),
            language_name=detect_language(str(row["source_code"])),
        )
        vec = legacy_model_vector(bundle)
        preds.append(float(baseline.predict([vec])[0]))
        targets.append(float(row["target_energy_joules"]))
    return {
        "mse": float(mean_squared_error(targets, preds)),
        "mae": float(mean_absolute_error(targets, preds)),
        "r2": float(r2_score(targets, preds) if len(set(targets)) > 1 else float("nan")),
    }


def run_phase3_experiment(csv_path: str, model_path: str = "phase1_model.pkl", epochs: int = 12, limit: Optional[int] = None, include_data_flow: bool = True, include_cache_locality: bool = True, batch_size: int = 8, seed: int = 42, save_path: Optional[str] = None, carbon_aware_objective: bool = False) -> Dict[str, Dict[str, float]]:
    _seed_everything(seed)
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(int(limit)).copy()
    if "snippet_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "snippet_id"})
    if "source_code" not in df.columns:
        raise ValueError("CSV must contain source_code")
    if "target_energy_joules" not in df.columns:
        raise ValueError("CSV must contain target_energy_joules")
    for col in ["input_scale_N", "hardware_tdp", "hardware_cores"]:
        if col not in df.columns:
            df[col] = 1.0
    if "algorithm_class" not in df.columns:
        df["algorithm_class"] = "unknown"

    train_df, val_df, test_df = split_by_snippet_id(df, seed=seed)
    train_path = _materialize_csv(train_df, None)
    val_path = _materialize_csv(val_df, None)
    test_path = _materialize_csv(test_df, None)

    try:
        shared_builder = ASTGraphBuilder(include_data_flow=include_data_flow, include_cache_locality=include_cache_locality)
        train_ds = ASTGraphDataset(train_path, include_data_flow=include_data_flow, include_cache_locality=include_cache_locality, builder=shared_builder)
        val_ds = ASTGraphDataset(val_path, include_data_flow=include_data_flow, include_cache_locality=include_cache_locality, builder=shared_builder)
        test_ds = ASTGraphDataset(test_path, include_data_flow=include_data_flow, include_cache_locality=include_cache_locality, builder=shared_builder)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_graphs)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)

        sample_graph = train_ds[0]
        model = ASTGNNRegressor(
            node_type_vocab_size=len(train_ds.builder._node_type_vocab),
            graph_feature_dim=sample_graph.graph_features.shape[0],
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_metrics = _train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=epochs,
            carbon_aware_objective=carbon_aware_objective,
        )
        # after training the best weights are loaded into `model` by `_train_model`
        if save_path:
            try:
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "meta": {
                            "node_type_vocab_size": len(train_ds.builder._node_type_vocab),
                            "graph_feature_dim": sample_graph.graph_features.shape[0],
                            "token_vocab_size": model.token_embedding.num_embeddings - 2,
                            "include_data_flow": include_data_flow,
                            "include_cache_locality": include_cache_locality,
                            "carbon_aware_objective": carbon_aware_objective,
                        },
                    },
                    save_path,
                )
            except Exception:
                pass
        test_metrics = _evaluate_model(model, test_loader, device)
        baseline_metrics = evaluate_baseline_phase1(model_path, test_df)
        return {
            "gnn_val": val_metrics,
            "gnn_test": test_metrics,
            "baseline_test": baseline_metrics,
            "split_sizes": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
            "include_data_flow": {"enabled": include_data_flow},
            "include_cache_locality": {"enabled": include_cache_locality},
            "carbon_aware_objective": {"enabled": carbon_aware_objective},
        }
    finally:
        for temp in [train_path, val_path, test_path]:
            try:
                Path(temp).unlink(missing_ok=True)
            except Exception:
                pass


def run_phase3_ablation_suite(csv_path: str, model_path: str = "phase1_model.pkl", epochs: int = 8, limit: Optional[int] = 240, batch_size: int = 8, seed: int = 42) -> Dict[str, Dict[str, Dict[str, float]]]:
    base = run_phase3_experiment(
        csv_path=csv_path,
        model_path=model_path,
        epochs=epochs,
        limit=limit,
        include_data_flow=True,
        include_cache_locality=True,
        batch_size=batch_size,
        seed=seed,
        carbon_aware_objective=False,
    )
    no_data_flow = run_phase3_experiment(
        csv_path=csv_path,
        model_path=model_path,
        epochs=epochs,
        limit=limit,
        include_data_flow=False,
        include_cache_locality=True,
        batch_size=batch_size,
        seed=seed,
        carbon_aware_objective=False,
    )
    no_cache = run_phase3_experiment(
        csv_path=csv_path,
        model_path=model_path,
        epochs=epochs,
        limit=limit,
        include_data_flow=True,
        include_cache_locality=False,
        batch_size=batch_size,
        seed=seed,
        carbon_aware_objective=False,
    )
    carbon_obj = run_phase3_experiment(
        csv_path=csv_path,
        model_path=model_path,
        epochs=epochs,
        limit=limit,
        include_data_flow=True,
        include_cache_locality=True,
        batch_size=batch_size,
        seed=seed,
        carbon_aware_objective=True,
    )
    return {
        "base": base,
        "without_data_flow": no_data_flow,
        "without_cache_locality": no_cache,
        "with_carbon_aware_objective": carbon_obj,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3 AST-GNN training and evaluation")
    parser.add_argument("--csv", "--data-file", dest="csv", default="eco_logic_synthetic_benchmark.csv")
    parser.add_argument("--model-path", default="phase1_model.pkl")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--save-path", default="phase3_model.pth")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-data-flow", action="store_true")
    parser.add_argument("--no-cache-locality", action="store_true")
    parser.add_argument("--carbon-aware-objective", action="store_true")
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.ablations:
        results = run_phase3_ablation_suite(
            csv_path=args.csv,
            model_path=args.model_path,
            epochs=args.epochs,
            limit=args.limit,
            batch_size=args.batch_size,
        )
        print(results)
        return

    results = run_phase3_experiment(
        csv_path=args.csv,
        model_path=args.model_path,
        epochs=args.epochs,
        limit=args.limit,
        include_data_flow=not args.no_data_flow,
        include_cache_locality=not args.no_cache_locality,
        batch_size=args.batch_size,
        save_path=args.save_path,
        carbon_aware_objective=args.carbon_aware_objective,
    )
    print(results)


if __name__ == "__main__":
    main()
