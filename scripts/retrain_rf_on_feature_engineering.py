"""Retrain a RandomForest baseline using Phase-2 (tree-sitter) features.

Usage:
    python scripts/retrain_rf_on_feature_engineering.py --csv eco_logic_synthetic_benchmark.csv
"""
from __future__ import annotations

import argparse
import joblib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import feature_engineering


def build_dataset(df: pd.DataFrame):
    X = []
    y = []
    rows = []
    for _, row in df.iterrows():
        code = row.get("source_code") or row.get("source") or row.get("code")
        if not isinstance(code, str):
            continue
        input_n = float(row.get("input_scale_N", row.get("input_n", 10000)))
        tdp = float(row.get("hardware_tdp", row.get("tdp", 45.0)))
        cores = float(row.get("hardware_cores", row.get("cores", 4.0)))
        target = row.get("target_energy_joules")
        if target is None:
            continue
        try:
            bundle = feature_engineering.analyze_code_features(code, input_n=input_n, tdp=tdp, cores=cores)
            vec = feature_engineering.legacy_model_vector(bundle)
            X.append(vec)
            y.append(float(target))
            rows.append(row.get("snippet_id", ""))
        except Exception as exc:  # pragma: no cover - robustness for exploratory runs
            print("feature extraction failed for a row:", exc, file=sys.stderr)
            continue

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    return X, y, rows


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="eco_logic_synthetic_benchmark.csv")
    parser.add_argument("--out", default="feature_engineered_rf_model.pkl")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv if argv is not None else None)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        raise SystemExit(2)

    df = pd.read_csv(csv_path)
    X, y, ids = build_dataset(df)
    if len(X) == 0:
        print("No samples extracted from CSV; aborting.")
        raise SystemExit(2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    print(f"Training RF on {len(X_train)} samples, validating on {len(X_test)} samples")
    rf = RandomForestRegressor(n_estimators=200, random_state=args.random_state, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"R2: {r2:.4f}, MSE: {mse:.4f}")

    joblib.dump(rf, args.out)
    print(f"Saved model -> {args.out}")


if __name__ == "__main__":
    main()
