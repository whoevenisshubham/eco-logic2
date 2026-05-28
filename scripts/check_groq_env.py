import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from load_env import load
import os

loaded = load()
print('.env loaded by load_env:', loaded)
print('GROQ_API_KEY (repr):', repr(os.environ.get('GROQ_API_KEY')))
print('ELECTRICITY_MAPS_API_KEY (repr):', repr(os.environ.get('ELECTRICITY_MAPS_API_KEY')))
