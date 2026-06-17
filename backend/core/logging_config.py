"""
backend/core/logging_config.py
-------------------------------
Structured logging: JSON in production, readable text in development.
Rotating file handlers for api.log, predictions.log, errors.log.
"""
from __future__ import annotations
import json, logging, logging.handlers
from pathlib import Path
from backend.core.config import settings


class JSONFormatter(logging.Formatter):
    def format(self, record):
        obj = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
               "level": record.levelname, "logger": record.name,
               "msg": record.getMessage()}
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        for key in ("request_id","user_id","path","method","status","duration_ms"):
            if hasattr(record, key):
                obj[key] = getattr(record, key)
        return json.dumps(obj)


class TextFormatter(logging.Formatter):
    def __init__(self):
        super().__init__("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                         datefmt="%Y-%m-%dT%H:%M:%S")


def _rotating(path, level, fmt):
    path.parent.mkdir(parents=True, exist_ok=True)
    h = logging.handlers.RotatingFileHandler(
        path, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    h.setLevel(level); h.setFormatter(fmt)
    return h


def setup_logging():
    log_dir = settings.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    fmt = JSONFormatter() if settings.is_production else TextFormatter()

    root = logging.getLogger()
    root.setLevel(level); root.handlers.clear()
    ch = logging.StreamHandler(); ch.setLevel(level); ch.setFormatter(fmt)
    root.addHandler(ch)
    root.addHandler(_rotating(log_dir / "app.log", level, fmt))

    for name, fname in [("api.access","api.log"), ("api.prediction","predictions.log"),
                        ("api.error","errors.log")]:
        lg = logging.getLogger(name); lg.propagate = False
        err_level = logging.ERROR if "error" in name else logging.INFO
        lg.addHandler(_rotating(log_dir / fname, err_level, fmt)); lg.addHandler(ch)

    for lib in ("werkzeug","urllib3"):
        logging.getLogger(lib).setLevel(logging.WARNING)
