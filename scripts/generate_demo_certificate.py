import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import build_certificate_payload, build_certificate_pdf

payload = {
    "generated_at":"2026-05-28 10:00:00",
    "project_name":"Demo",
    "project_type":"single-file",
    "selected_file":"demo.py",
    "model_name":"phase1_model.pkl",
    "dataset_name":"eco_logic_synthetic_benchmark.csv",
    "input_n":100,"tdp":45,"cores":8,"intensity":714.0,
    "elapsed_ms":12.3,"rounds":2,"selection_reason":"selected",
    "original_algorithm":"bubble_sort","best_algorithm":"sorted",
    "base_energy":1.23,"best_energy":0.45,"base_runtime_ms":5.0,"best_runtime_ms":1.7,
    "measured_original_runtime_ms":5.5,"measured_optimized_runtime_ms":1.2,"measured_runtime_delta_ms":4.3,
    "original_code":"def bubble_sort(arr):\n    pass","best_code":"def optimized(arr):\n    return sorted(arr)",
    "rounds_detail":[{"round":1,"label":"heuristic","energy_j":0.45,"runtime_ms":1.7,"measured_runtime_ms":1.2,"selection_reason":"selected"}]
}

pdf = build_certificate_pdf(build_certificate_payload(
    original_code=payload['original_code'],
    best={"label":"Demo Best","code":payload['best_code'],"eval":{"energy_j":payload['best_energy'],"runtime_ms":payload['best_runtime_ms'],"algorithm_class":payload['best_algorithm']}},
    base_eval={"energy_j":payload['base_energy'],"runtime_ms":payload['base_runtime_ms'],"algorithm_class":payload['original_algorithm']},
    base_measured={"ok":True,"runtime_ms":payload['measured_original_runtime_ms']},
    measured_best={"ok":True,"runtime_ms":payload['measured_optimized_runtime_ms']},
    rounds=payload['rounds_detail'],
    input_n=payload['input_n'],
    tdp=payload['tdp'],
    cores=payload['cores'],
    intensity=payload['intensity'],
    model_name=payload['model_name'],
    dataset_name=payload['dataset_name'],
    elapsed_ms=payload['elapsed_ms'],
    project_manifest=None,
    selected_project_file=payload['selected_file'],
))

out_path = 'ecologic_certificate_demo.pdf'
with open(out_path, 'wb') as f:
    f.write(pdf)
print('WROTE', out_path)
