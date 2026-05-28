from pathlib import Path
import sys
import joblib
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import make_refactor_candidates
from feature_engineering import analyze_code_features, legacy_model_vector


def load_model(path="phase1_model.pkl"):
    p = Path(path)
    if not p.exists():
        print(f"Model not found at {p.resolve()}")
        return None
    m = joblib.load(str(p))
    print(f"Loaded model: {p.name} type={type(m)}")
    try:
        print("n_features_in_:", getattr(m, "n_features_in_", None))
    except Exception:
        pass
    return m


def diag(code_text, model, input_n=10000, tdp=45.0, cores=4):
    bundle = analyze_code_features(code_text, input_n=input_n, tdp=tdp, cores=cores)
    vec = legacy_model_vector(bundle)
    print("features (len={})".format(len(vec)))
    print(json.dumps(vec, indent=2))
    if model is not None:
        try:
            pred = model.predict([vec])[0]
            print("model prediction:", float(pred))
        except Exception as exc:
            print("model.predict failed:", exc)


if __name__ == "__main__":
    model = load_model()

    original = '''def matmul_naive(A, B):
    n = len(A)
    C = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

if __name__ == "__main__":
    import random, time
    n = 120
    A = [[random.random() for _ in range(n)] for __ in range(n)]
    B = [[random.random() for _ in range(n)] for __ in range(n)]
    t0 = time.perf_counter()
    matmul_naive(A, B)
    print((time.perf_counter()-t0)*1000.0)
'''
    # Also test a C++ bubble sort sample for candidate generation
    cpp_sample = '''#include <iostream>
#include <vector>
using namespace std;

void bubbleSort(vector<int>& arr) {
    int n = arr.size();

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int main() {
    vector<int> arr = {5, 1, 4, 2, 8};

    bubbleSort(arr);

    cout << "Sorted array: ";
    for (int x : arr) {
        cout << x << " ";
    }

    return 0;
}
'''

    print('\n--- C++ Generated candidates ---')
    cands = make_refactor_candidates(cpp_sample, "")
    for i, c in enumerate(cands, start=1):
        print(f"\nCandidate {i}: {c['label']}")
        print(c['code'][:1000])
        diag(c['code'], model, input_n=1024, tdp=65.0, cores=4)

    print("\n--- Generated candidates ---")
    cands = make_refactor_candidates(original, "")
    for i, c in enumerate(cands, start=1):
        print(f"\nCandidate {i}: {c['label']}")
        print(c["code"][:1000])
        diag(c["code"], model, input_n=120, tdp=45.0, cores=4)
