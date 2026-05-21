import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import load_env
from groq_client import generate_refactor

load_env.load()


def demo():
    prompt = (
        "Refactor the following Python function to be more energy-efficient and return only a fenced python code block:\n\n"
        "```python\n"
        "def bubble_sort(arr):\n"
        "    n = len(arr)\n"
        "    for i in range(n):\n"
        "        for j in range(0, n - i - 1):\n"
        "            if arr[j] > arr[j + 1]:\n"
        "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
        "    return arr\n"
        "```\n"
    )
    out = generate_refactor(prompt)
    print(out)


if __name__ == "__main__":
    demo()
