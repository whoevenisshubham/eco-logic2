import os
import requests

KEY = os.environ.get("ELECTRICITY_MAPS_API_KEY")
if not KEY:
    print("ELECTRICITY_MAPS_API_KEY not set. Fill .env or export env var first.")
    raise SystemExit(1)

url = "https://api.electricitymap.org/v3/carbon-intensity/latest"
headers = {"auth-token": KEY}
params = {"zone": "IN-WE"}
print("Requesting", url, params)
resp = requests.get(url, headers=headers, params=params, timeout=10)
print("status", resp.status_code)
try:
    j = resp.json()
    print(j)
except Exception:
    print(resp.text)
