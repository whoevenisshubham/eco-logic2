import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from project_ingestion import scan_project
from runtime_harness import measure_runtime

print('Running integration tests...')
manifest = scan_project(str(root))
print('scan_project OK: files=', manifest.get('file_count'))

# Test Python measurement on a small snippet
code = '''def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
'''
try:
    res = measure_runtime(code, 'Python', 'bubble_sort', 64)
    print('measure_runtime OK:', res)
except Exception as e:
    print('measure_runtime FAILED:', e)
    raise

print('Integration tests completed.')
