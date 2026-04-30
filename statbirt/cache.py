from __future__ import annotations

from pathlib import Path
import pickle

from .config import CACHE_DIR


def cache_path(namespace: str, name: str) -> Path:
    path = CACHE_DIR / namespace
    path.mkdir(parents=True, exist_ok=True)
    return path / name


def load_pickle(path: str | Path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_pickle(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(value, f)

