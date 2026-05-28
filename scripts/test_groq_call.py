import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from load_env import load
load()
from groq_client import generate_refactor, get_groq_api_key

print('GROQ_API_KEY present:', bool(get_groq_api_key()))

prompt = (
    "Refactor the following Python function to use NumPy for matrix multiplication. "
    "Return only the refactored function (no explanation).\n\n```python\n"
    "def matmul_naive(A,B):\n  n=len(A)\n  C=[[0]*n for _ in range(n)]\n  for i in range(n):\n    for j in range(n):\n      s=0\n      for k in range(n):\n        s+=A[i][k]*B[k][j]\n      C[i][j]=s\n  return C\n```")

try:
    out = generate_refactor(prompt, max_output_tokens=800)
    print('Groq response (raw):')
    print(out)
except Exception as e:
    print('Groq call failed:', repr(e))
