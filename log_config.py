import logging
import os
from logging.handlers import RotatingFileHandler

# Ensure logs directory exists
ROOT = os.path.dirname(__file__)
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_PATH = os.path.join(LOG_DIR, "ecologic.log")

_handler = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
_formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
_handler.setFormatter(_formatter)

_root = logging.getLogger()
if not _root.handlers:
    _root.setLevel(logging.INFO)
    _root.addHandler(_handler)


def get_logger(name: str):
    return logging.getLogger(name)
