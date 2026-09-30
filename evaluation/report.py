"""JSON reports. Metrics are null when a run did not execute."""

import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def machine_info() -> dict:
    memory_bytes = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory_bytes = int(line.split()[1]) * 1024
                break
    cuda = False
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": memory_bytes,
        "cuda_available": cuda,
    }


def write_report(directory: Path, prefix: str, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = directory / f"{prefix}_{stamp}.json"
    document = {
        "status": payload.get("status", "NOT_EXECUTED"),
        "reason": payload.get("reason"),
        "git_revision": payload.get("git_revision", "unknown"),
        "timestamp": datetime.now(UTC).isoformat(),
        "machine": payload.get("machine") or machine_info(),
        "python": platform.python_version(),
        "model_name": payload.get("model_name"),
        "configuration": payload.get("configuration") or {},
        "dataset_version": payload.get("dataset_version"),
        "methodology": payload.get("methodology"),
        "metrics": payload.get("metrics"),
    }
    reserved = set(document)
    for key, value in payload.items():
        if key not in reserved and key not in {"status", "reason", "metrics", "methodology", "configuration"}:
            document[key] = value
    path.write_text(json.dumps(document, indent=2, sort_keys=True, default=_json_default) + "\n")
    return path


def _json_default(value: object):
    """Encode values that the runners naturally produce."""
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")
