from __future__ import annotations

import json
import os
import re
from pathlib import Path

import joblib

import load_env
load_env.load()

from groq_client import generate_refactor
from phase2_features import analyze_code_features, legacy_model_vector
from runtime_harness import measure_runtime


ORIGINAL_CODE = r'''
// Inefficient Sort: Bubble Sort
// Time Complexity: O(n^2)

#include <iostream>
#include <vector>
using namespace std;

void bubbleSort(vector<int>& arr) {
    int n = arr.size();

    for(int i = 0; i < n - 1; ++i) {
        for(int j = 0; j < n - i - 1; ++j) {
            if(arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int main() {
    vector<int> arr = {5, 2, 8, 1, 3};

    bubbleSort(arr);

    for(int x : arr) {
        cout << x << " ";
    }

    return 0;
}
'''

PROMPT = (
    (
        "You are an expert C++ programmer. Refactor the following C++ function to reduce worst-case and typical CPU work (aim for algorithmic improvement). "
        "Do NOT replace the algorithm by calling a library routine that merely delegates to a built-in (for example: do not use `std::sort`, `qsort`, or similar). "
        "Return a complete, self-contained C++ implementation that improves the algorithm or technique (for example: remove extra nested loops, add early-exit, shrink loop bounds, sentinel optimizations). "
        "Preserve the original behavior and inputs/outputs where possible. "
        "Return ONLY valid, compilable C++ code inside a single fenced code block and do not include any explanations.\n\n"
    )
    + ORIGINAL_CODE
)

MODEL_PATH = Path("phase1_model.pkl")


def extract_fenced(code_text: str) -> str:
    m = re.search(r"```(?:cpp|c\+\+)?\s*([\s\S]*?)```", code_text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return code_text.strip()


def safe_predict(model, code_text):
    bundle = analyze_code_features(code_text, input_n=1024.0, tdp=65.0, cores=8.0)
    vec = legacy_model_vector(bundle)
    if len(vec) != 9:
        raise ValueError(f"legacy vector length {len(vec)} != 9")
    pred = float(model.predict([vec])[0])
    runtime = measure_runtime(code_text, "C++", "bubble_sort", 1024)
    return {
        "energy_j": pred,
        "runtime_ms": runtime["runtime_ms"],
        "runtime_mode": runtime.get("mode", "proxy"),
        "bundle": bundle,
    }


def main():
    model = joblib.load(MODEL_PATH)

    # heuristics
    heuristics = {
        "early_exit_bubble": '''#include <algorithm>
#include <vector>

void bubbleSortOptimized(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i) {
        bool swapped = false;
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}''',
    }

    candidates = []
    # original
    candidates.append(("original", ORIGINAL_CODE))

    # LLM
    try:
        out = generate_refactor(PROMPT)
        llm_code = extract_fenced(out)
        if llm_code and llm_code.strip() != "":
            candidates.append(("llm_refactor", llm_code))
    except Exception as e:
        print("LLM call failed:", e)

    # add heuristics
    for k, v in heuristics.items():
        candidates.append((k, v))

    results = []
    for label, code in candidates:
        try:
            res = safe_predict(model, code)
            results.append({"label": label, "energy_j": res["energy_j"], "runtime_ms": res["runtime_ms"], "runtime_mode": res["runtime_mode"]})
        except Exception as e:
            results.append({"label": label, "error": str(e)})

    # sort by energy
    successful = [r for r in results if "energy_j" in r]
    failed = [r for r in results if "error" in r]
    successful_sorted = sorted(successful, key=lambda x: x["energy_j"])

    out = {
        "successful": successful_sorted,
        "failed": failed,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
