import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_engineering import analyze_code_features, legacy_model_vector


SNIPPETS = {
    "bubble_sort": """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
""".strip(),
    "quick_sort": """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + mid + quick_sort(right)
""".strip(),
    "matrix_mult_naive": """
def matrix_multiply(A, B, N):
    C = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C
""".strip(),
    "matrix_mult_optimized": """
import numpy as np

def matrix_mult_optimized(A, B):
    return np.dot(A, B)
""".strip(),
    "busy_wait_anomaly": """
def poll_until(flag):
    while not flag:
        pass
    return flag
""".strip(),
}


def validate_feature_vector_9(features):
    if len(features) != 9:
        raise AssertionError(f"Expected 9 features, got {len(features)}")


def predict_energy(model, code_text, input_n=1024.0, tdp=65.0, cores=8.0):
    bundle = analyze_code_features(code_text, input_n=input_n, tdp=tdp, cores=cores)
    vec = legacy_model_vector(bundle)
    validate_feature_vector_9(vec)
    pred = float(model.predict([vec])[0])
    return pred, bundle


def main():
    model_path = ROOT / "baseline_rf_model.pkl"
    model = joblib.load(model_path)

    for name, code in SNIPPETS.items():
        pred, bundle = predict_energy(model, code)
        print(f"[{name}] pred={pred:.6f}J backend={bundle.get('parser_backend')}")
        assert pred >= 0.0

    print("Deterministic 9-feature smoke checks passed.")


if __name__ == "__main__":
    main()
