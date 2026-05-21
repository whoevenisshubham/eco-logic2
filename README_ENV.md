Usage: .env and helper scripts

1. Populate `.env` in the project root with your keys. Example:

ELECTRICITY_MAPS_API_KEY="your_key_here"
GROQ_API_KEY="..."

2. Recommended: add `.env` to your OS environment for long-running shells or CI using `setx` (Windows) or `export` (Linux/Mac). For quick local runs, load `.env` into the current session:

PowerShell:

```powershell
python -c "import load_env; load_env.load()"
```

Bash:

```bash
python -c 'import load_env; load_env.load()'
```

3. To test Electricity Maps connectivity run:

```bash
python tests/electricity_maps_check.py
```

4. Install optional helper package `python-dotenv` for robust .env parsing:

```bash
pip install -r requirements.txt
```

Security note: Never commit `.env` to source control. Rotate keys if they are exposed.
