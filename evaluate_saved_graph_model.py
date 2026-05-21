from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

import graph_energy_model as p3


def _load_saved_model(model_path: str, model: torch.nn.Module, device: torch.device) -> None:
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)


def main(csv_path: str = "eco_logic_synthetic_benchmark.csv", saved_model: str = "graph_energy_model.pth", baseline_model: str = "feature_engineered_rf_model.pkl", seed: int = 42):
    df = pd.read_csv(csv_path)
    if "snippet_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "snippet_id"})
    train_df, val_df, test_df = p3.split_by_snippet_id(df, seed=seed)
    train_path = p3._materialize_csv(train_df, None)
    val_path = p3._materialize_csv(val_df, None)
    test_path = p3._materialize_csv(test_df, None)

    try:
        shared_builder = p3.ASTGraphBuilder()
        train_ds = p3.ASTGraphDataset(train_path, builder=shared_builder)
        val_ds = p3.ASTGraphDataset(val_path, builder=shared_builder)
        test_ds = p3.ASTGraphDataset(test_path, builder=shared_builder)

        batch_size = 8
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=p3.collate_graphs)

        # construct model with vocab sizes discovered by builder
        sample = train_ds[0]
        model = p3.ASTGNNRegressor(node_type_vocab_size=len(train_ds.builder._node_type_vocab), graph_feature_dim=sample.graph_features.shape[0])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not Path(saved_model).exists():
            print(f"Saved model not found: {saved_model}")
        else:
            _load_saved_model(saved_model, model, device)
            model.to(device)
            metrics = p3._evaluate_model(model, test_loader, device)
            print("Graph model saved metrics:")
            print(metrics)

        # baseline evaluation (feature_engineered_rf_model.pkl)
        if not Path(baseline_model).exists():
            print(f"Baseline model not found: {baseline_model}")
        else:
            baseline_metrics = p3.evaluate_baseline_model(baseline_model, test_df)
            print("Baseline metrics (feature_engineered_rf_model.pkl):")
            print(baseline_metrics)

    finally:
        for f in [train_path, val_path, test_path]:
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved graph energy model")
    parser.add_argument("--csv", "--data-file", dest="csv_path", default="eco_logic_synthetic_benchmark.csv")
    parser.add_argument("--saved-model", default="graph_energy_model.pth")
    parser.add_argument("--baseline-model", default="feature_engineered_rf_model.pkl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(csv_path=args.csv_path, saved_model=args.saved_model, baseline_model=args.baseline_model, seed=args.seed)
