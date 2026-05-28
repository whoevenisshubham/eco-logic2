import time
import sys
from pathlib import Path

# Ensure workspace root is on sys.path when running this script from scripts/ directory
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from project_ingestion import scan_project

path = str(root)
start = time.time()
manifest = scan_project(path)
elapsed = time.time() - start
print('scan_project elapsed(s):', elapsed)
print('file_count:', manifest.get('file_count'))
print('language_counts:', manifest.get('language_counts'))
