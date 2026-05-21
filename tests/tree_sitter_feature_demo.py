import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_engineering import analyze_code_features, legacy_model_vector, rich_feature_rows


PY_BUBBLE = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr
""".strip()

PY_MATRIX_ROW_MAJOR = """
def sum_rows(matrix):
    total = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            total += matrix[i][j]
    return total
""".strip()

PY_MATRIX_COL_MAJOR = """
def sum_cols(matrix):
    total = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            total += matrix[j][i]
    return total
""".strip()

PY_BUSY_WAIT = """
def poll_until(flag):
    while not flag:
        pass
    return flag
""".strip()

CPP_BUBBLE = """
#include <vector>
#include <algorithm>
void cubicBubbleSort(std::vector<int>& arr) {
    int n = static_cast<int>(arr.size());
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }
}
""".strip()


def show(name, code, language=None):
    bundle = analyze_code_features(code, input_n=256, tdp=45.0, cores=8.0, language_name=language)
    print(f"[{name}] backend={bundle['parser_backend']} language={bundle['language']}")
    print("  legacy vector length:", len(legacy_model_vector(bundle)))
    print("  feature sample:", rich_feature_rows(bundle)[:6])
    return bundle


def main():
    bubble = show("python-bubble", PY_BUBBLE, "Python")
    row = show("python-row-major", PY_MATRIX_ROW_MAJOR, "Python")
    col = show("python-col-major", PY_MATRIX_COL_MAJOR, "Python")
    busy = show("python-busy", PY_BUSY_WAIT, "Python")
    cpp = show("cpp-bubble", CPP_BUBBLE, "C++")

    assert bubble["loop_count"] >= 2
    assert bubble["cyclomatic_complexity"] > 1
    assert busy["busy_wait_score"] > 0
    assert col["stride_penalty"] >= row["stride_penalty"]
    assert cpp["parser_backend"].startswith("tree-sitter") or cpp["parser_backend"].startswith("fallback")
    print("Tree-sitter feature checks passed.")


if __name__ == "__main__":
    main()
