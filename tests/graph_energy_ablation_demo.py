import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph_energy_model import run_graph_energy_ablation_suite


def main():
    results = run_graph_energy_ablation_suite(
        csv_path=str(ROOT / "eco_logic_synthetic_benchmark.csv"),
        model_path=str(ROOT / "feature_engineered_rf_model.pkl"),
        epochs=3,
        limit=180,
        batch_size=8,
        seed=42,
    )
    print(json.dumps(results, indent=2))

    for key in ["base", "without_data_flow", "without_cache_locality", "with_carbon_aware_objective"]:
        assert key in results
        assert "gnn_test" in results[key]
        assert "r2" in results[key]["gnn_test"]

    print("Graph energy ablation suite checks passed.")


if __name__ == "__main__":
    main()
