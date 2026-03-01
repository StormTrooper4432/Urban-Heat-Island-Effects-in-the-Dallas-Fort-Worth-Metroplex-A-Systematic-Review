import os
from datetime import datetime
from pathlib import Path
from typing import Iterable


def ensure_dirs(paths: Iterable[str]) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def file_exists(path: str) -> bool:
    return Path(path).exists()


def env_or_default(key: str, default: str) -> str:
    return os.environ.get(key, default)


def ts_print(*args, **kwargs) -> None:
    """Print with ISO-8601 timestamp prefix."""
    ts = datetime.now().isoformat(timespec="seconds")
    if "flush" not in kwargs:
        kwargs["flush"] = True
    print(f"[{ts}]", *args, **kwargs)
