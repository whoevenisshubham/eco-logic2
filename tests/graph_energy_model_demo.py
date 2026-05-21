import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph_energy_model import run_graph_energy_experiment


def main():
    results = run_graph_energy_experiment(
        csv_path=str(ROOT / "eco_logic_synthetic_benchmark.csv"),
        model_path=str(ROOT / "baseline_rf_model.pkl"),
        epochs=2,
        limit=90,
        include_data_flow=True,
        include_cache_locality=True,
        batch_size=8,
        seed=42,
    )
    print(json.dumps(results, indent=2))
    assert "gnn_test" in results
    assert "baseline_test" in results
    assert results["split_sizes"]["train"] > 0
    assert results["split_sizes"]["test"] > 0
    assert results["gnn_test"]["mse"] >= 0
    print("Graph energy model checks passed.")


if __name__ == "__main__":
    main()
