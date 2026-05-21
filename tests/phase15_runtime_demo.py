import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_harness import measure_runtime


def main():
    python_bubble_sort = """
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

    py_result = measure_runtime(python_bubble_sort, "Python", "bubble_sort", 256)
    print("PYTHON:", py_result)

    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler:
        cpp_bubble_sort = """
#include <vector>
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
        cpp_result = measure_runtime(cpp_bubble_sort, "C++", "bubble_sort", 128)
        print("CPP:", cpp_result)
    else:
        print("CPP: skipped (no g++/clang++ found)")


if __name__ == "__main__":
    main()
