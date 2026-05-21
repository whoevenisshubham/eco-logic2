import os
import math
import re
import shutil
import statistics
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path


def runtime_proxy(algorithm_class, n_val, cores):
    n = max(float(n_val), 1.0)
    core_factor = max(float(cores), 1.0)

    if algorithm_class == "bubble_sort":
        base = (n ** 2) / 2.2e7
    elif algorithm_class == "quick_sort":
        base = (n * math.log2(n + 1.0)) / 5.0e5
    elif algorithm_class == "matrix_mult_naive":
        cube = n ** 3
        base = cube / 2.8e11
    elif algorithm_class == "matrix_mult_optimized":
        base = (n ** 2.2) / 5.0e8
    elif algorithm_class == "busy_wait_anomaly":
        base = n / 6.0e3
    else:
        base = (n * math.log2(n + 1.0)) / 4.0e5

    runtime_ms = 1000.0 * (base / math.sqrt(core_factor))
    return float(max(runtime_ms, 0.01))


def choose_sample_size(algorithm_class, input_n):
    n = max(int(input_n), 1)
    if algorithm_class == "bubble_sort":
        return min(max(n, 32), 256)
    if algorithm_class == "quick_sort":
        return min(max(n, 64), 2048)
    if algorithm_class in {"matrix_mult_naive", "matrix_mult_optimized"}:
        return min(max(int(math.sqrt(n)), 8), 32)
    if algorithm_class == "busy_wait_anomaly":
        return min(max(n, 1), 5_000)
    return min(max(n, 32), 512)


def _unique_candidates(names):
    seen = set()
    ordered = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _extract_identifier_candidates(code_text, algorithm_class, language_name):
    candidates = []
    if algorithm_class == "bubble_sort":
        candidates.extend(["bubble_sort", "cubicBubbleSort", "optimized_sort", "sort"])
    elif algorithm_class == "quick_sort":
        candidates.extend(["quick_sort", "quickSort", "quicksort", "sort"])
    elif algorithm_class in {"matrix_mult_naive", "matrix_mult_optimized"}:
        candidates.extend(["matrix_multiply_opt", "matrix_mult", "matrix_multiply", "multiply"])
    else:
        candidates.extend(["solve", "optimized_code", "process", "main"])

    if language_name == "Python":
        pattern = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
    elif language_name == "C++":
        pattern = re.compile(
            r"(?:^|\n)\s*(?:template\s*<[^>]+>\s*)?(?:[\w:<>&*\s]+\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(([^\)]*)\)\s*(?:\{|;)",
            re.MULTILINE,
        )
    else:
        pattern = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)

    if language_name == "Python":
        candidates.extend(pattern.findall(code_text))
    elif language_name == "C++":
        for match in pattern.finditer(code_text):
            candidates.append(match.group(1))

    return _unique_candidates(candidates)


def _infer_cpp_arity(code_text, function_name):
    pattern = re.compile(
        rf"(?:^|\n)\s*(?:template\s*<[^>]+>\s*)?(?:[\w:<>&*\s]+\s+)?{re.escape(function_name)}\s*\(([^\)]*)\)\s*(?:\{{|;)",
        re.MULTILINE,
    )
    match = pattern.search(code_text)
    if not match:
        return None
    raw = match.group(1).strip()
    if not raw or raw == "void":
        return 0
    return len([part for part in raw.split(",") if part.strip()])


def _build_python_benchmark(code_text, algorithm_class, sample_n):
    candidate_names = _extract_identifier_candidates(code_text, algorithm_class, "Python")
    code_literal = repr(code_text)
    candidate_literal = repr(candidate_names)
    return textwrap.dedent(
        f"""
        import inspect
        import statistics
        import time

        namespace = {{}}
        code = {code_literal}
        exec(code, namespace, namespace)

        candidate_names = {candidate_literal}
        func = None
        for name in candidate_names:
            obj = namespace.get(name)
            if callable(obj):
                func = obj
                break

        if func is None:
            raise RuntimeError("No callable candidate found")

        param_count = len(inspect.signature(func).parameters)

        def build_inputs():
            if {algorithm_class!r} in {{"matrix_mult_naive", "matrix_mult_optimized"}}:
                size = {sample_n}
                left = [[(i + j) % 7 for j in range(size)] for i in range(size)]
                right = [[(i * j) % 5 for j in range(size)] for i in range(size)]
                return left, right
            data = list(range({sample_n}, 0, -1))
            return data

        def invoke_once():
            inputs = build_inputs()
            if {algorithm_class!r} == "quick_sort":
                if param_count >= 3:
                    return func(inputs, 0, len(inputs) - 1)
                return func(inputs)
            if {algorithm_class!r} in {{"matrix_mult_naive", "matrix_mult_optimized"}}:
                if param_count >= 2:
                    return func(*inputs)
                return func(inputs[0])
            return func(inputs)

        timings = []
        for _ in range(3):
            start = time.perf_counter()
            invoke_once()
            timings.append((time.perf_counter() - start) * 1000.0)

        print(statistics.median(timings))
        """
    ).strip()


def _build_cpp_benchmark(code_text, algorithm_class, sample_n):
    candidate_names = _extract_identifier_candidates(code_text, algorithm_class, "C++")
    code_text = code_text.strip()
    has_main = bool(re.search(r"\bmain\s*\(", code_text))
    if has_main:
        return code_text, candidate_names, False

    candidate_name = next(
        (
            name
            for name in candidate_names
            if re.search(rf"\b{re.escape(name)}\b", code_text)
        ),
        None,
    )
    if not candidate_name:
        raise RuntimeError("No callable candidate found")

    arity = _infer_cpp_arity(code_text, candidate_name)
    if arity is None:
        arity = 1 if algorithm_class != "quick_sort" else 3

    if algorithm_class in {"matrix_mult_naive", "matrix_mult_optimized"}:
        body = f"""
        int main() {{
            const int sample_n = {sample_n};
            std::vector<std::vector<int>> left(sample_n, std::vector<int>(sample_n, 1));
            std::vector<std::vector<int>> right(sample_n, std::vector<int>(sample_n, 2));
            auto start = std::chrono::steady_clock::now();
            {candidate_name}(left, right);
            auto end = std::chrono::steady_clock::now();
            std::cout << std::chrono::duration<double, std::milli>(end - start).count() << std::endl;
            return 0;
        }}
        """
    elif algorithm_class == "quick_sort" and arity >= 3:
        body = f"""
        int main() {{
            const int sample_n = {sample_n};
            std::vector<int> data(sample_n);
            for (int i = 0; i < sample_n; ++i) data[i] = sample_n - i;
            auto start = std::chrono::steady_clock::now();
            {candidate_name}(data, 0, sample_n - 1);
            auto end = std::chrono::steady_clock::now();
            std::cout << std::chrono::duration<double, std::milli>(end - start).count() << std::endl;
            return 0;
        }}
        """
    else:
        body = f"""
        int main() {{
            const int sample_n = {sample_n};
            std::vector<int> data(sample_n);
            for (int i = 0; i < sample_n; ++i) data[i] = sample_n - i;
            auto start = std::chrono::steady_clock::now();
            {candidate_name}(data);
            auto end = std::chrono::steady_clock::now();
            std::cout << std::chrono::duration<double, std::milli>(end - start).count() << std::endl;
            return 0;
        }}
        """

    wrapper = textwrap.dedent(
        f"""
        #include <chrono>
        #include <iostream>
        #include <vector>
        {code_text}
        {body}
        """
    ).strip()
    return wrapper, candidate_names, True


def measure_runtime(code_text, language_name, algorithm_class, input_n, timeout_s=8.0):
    sample_n = choose_sample_size(algorithm_class, input_n)
    detail = {
        "language": language_name,
        "algorithm_class": algorithm_class,
        "sample_n": sample_n,
    }

    try:
        if language_name == "Python":
            script = _build_python_benchmark(code_text, algorithm_class, sample_n)
            with tempfile.TemporaryDirectory() as temp_dir:
                script_path = Path(temp_dir) / "benchmark.py"
                script_path.write_text(script, encoding="utf-8")
                started = time.perf_counter()
                result = subprocess.run(
                    [shutil.which("python") or "python", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if result.returncode != 0:
                    raise RuntimeError((result.stderr or result.stdout or "Python benchmark failed").strip())
                measured_ms = float(result.stdout.strip().splitlines()[-1])
                detail.update({"mode": "measured", "tool": "python-subprocess", "run_ms": elapsed_ms})
                return {"runtime_ms": measured_ms, "mode": "measured", "detail": detail}

        if language_name == "C++":
            compiler = shutil.which("g++") or shutil.which("clang++")
            if not compiler:
                raise RuntimeError("No C++ compiler found (g++ or clang++ not installed)")
            source_text, candidate_names, has_main = _build_cpp_benchmark(code_text, algorithm_class, sample_n)
            detail["candidates"] = candidate_names
            detail["has_main"] = has_main
            with tempfile.TemporaryDirectory() as temp_dir:
                source_path = Path(temp_dir) / "benchmark.cpp"
                exe_path = Path(temp_dir) / ("benchmark.exe" if os.name == "nt" else "benchmark")
                source_path.write_text(source_text, encoding="utf-8")
                compile_result = subprocess.run(
                    [compiler, str(source_path), "-O2", "-std=c++17", "-o", str(exe_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
                if compile_result.returncode != 0:
                    raise RuntimeError((compile_result.stderr or compile_result.stdout or "C++ compile failed").strip())
                started = time.perf_counter()
                run_result = subprocess.run(
                    [str(exe_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if run_result.returncode != 0:
                    raise RuntimeError((run_result.stderr or run_result.stdout or "C++ benchmark failed").strip())
                output = run_result.stdout.strip().splitlines()[-1]
                measured_ms = float(output)
                detail.update({"mode": "measured", "tool": compiler, "run_ms": elapsed_ms})
                return {"runtime_ms": measured_ms, "mode": "measured", "detail": detail}

        raise RuntimeError(f"Unsupported language for execution measurement: {language_name}")
    except Exception as exc:
        fallback_ms = runtime_proxy(algorithm_class, input_n, 1)
        detail.update({"mode": "proxy", "reason": str(exc)})
        return {"runtime_ms": fallback_ms, "mode": "proxy", "detail": detail}