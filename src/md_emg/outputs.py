from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

import joblib as _joblib
import numpy
import pandas as pd
import scipy
import sklearn
import threadpoolctl as _tpc
import yaml


def make_run_dir(root: str | Path, name: str) -> tuple[str, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    suffix = uuid.uuid4().hex[:4]
    run_id = f"{name}_{ts}_{os.getpid()}_{suffix}"
    d = Path(root) / run_id
    d.mkdir(parents=True, exist_ok=False)
    for sub in ["inner_cv", "models", "logs", "fold_outputs"]:
        (d / sub).mkdir()
    return run_id, d


def save_yaml(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def save_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def _git_commit(project_root: Path | None = None) -> str:
    """Return the current git commit hash, or 'unknown' if not in a git repo."""
    cwd = str(project_root) if project_root else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def environment_manifest(command: str, project_root: Path | None = None) -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "numpy": numpy.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": _joblib.__version__,
        "threadpoolctl": _tpc.__version__,
        "git_commit": _git_commit(project_root),
        "command": command,
    }


def save_table(df, path):
    """Save DataFrame as parquet when available; fall back to CSV if parquet engine is absent."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path
