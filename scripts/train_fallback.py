from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Ensure repository root is on sys.path when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_engineering import analyze_code_features, legacy_model_vector


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
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
    out["target_energy_joules"] = pd.to_numeric(out["target_energy_joules"], errors="coerce").fillna(1.0)
    return out


def find_dataset() -> Path:
    candidates = ["eco_logic_synthetic_benchmark.csv", "eco_logic_ast_dataset.csv"]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


def build_feature_row(code_text: str, input_n: float, tdp: float, cores: float):
    bundle = analyze_code_features(code_text, input_n=input_n, tdp=tdp, cores=cores)
    vec = legacy_model_vector(bundle)
    return vec


def main():
    ds = find_dataset()
    if ds is None:
        print("No dataset found. Place eco_logic_synthetic_benchmark.csv in the workspace.")
        sys.exit(2)

    df = pd.read_csv(ds)
    df = normalize_dataset(df)

    X = []
    y = []
    for _, r in df.iterrows():
        code = str(r.get("source_code", r.get("code", "")))
        fv = build_feature_row(code, r.get("input_scale_N", 10000), r.get("hardware_tdp", 45.0), r.get("hardware_cores", 4))
        X.append(fv)
        y.append(float(r.get("target_energy_joules", 1.0)))

    print(f"Training fallback RandomForest on {len(X)} samples")
    model = RandomForestRegressor(n_estimators=64, random_state=42, n_jobs=-1)
    model.fit(X, y)
    out = Path("phase1_model.pkl")
    joblib.dump(model, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
