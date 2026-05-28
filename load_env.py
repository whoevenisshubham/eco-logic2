from pathlib import Path
import os

try:
    from dotenv import load_dotenv
    _has_dotenv = True
except Exception as exc:
    from log_config import get_logger

    get_logger(__name__).warning("python-dotenv not available: %s", exc)
    _has_dotenv = False


def load(dotenv_path=None):
    """Load .env into environment. Returns True if loaded, False otherwise.
    Tries python-dotenv if available; otherwise falls back to a simple parser.
    """
    path = Path(dotenv_path) if dotenv_path else Path(".") / ".env"
    if not path.exists():
        return False
    if _has_dotenv:
        load_dotenv(dotenv_path=str(path))
        return True

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)
    return True


if __name__ == "__main__":
    ok = load()
    print("Loaded .env:" , ok)
    for key in [
        "ELECTRICITY_MAPS_API_KEY",
        "GROQ_API_KEY",
    ]:
        print(key, "=", os.environ.get(key))
