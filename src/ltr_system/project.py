"""Project paths, configuration validation, and atomic metadata writes."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

DEFAULTS_ROOT = Path(__file__).resolve().parent / "defaults"


def project_root() -> Path:
    """Resolve runtime storage without relying on the current shell directory."""
    if value := os.environ.get("LTR_PROJECT_ROOT"):
        return Path(value).expanduser().resolve()
    if value := os.environ.get("SCRATCH"):
        return (Path(value) / "learning-to-rank").resolve()
    return Path.cwd().resolve()


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--project-root", type=Path, default=project_root())
    result.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return result


def initialize(args: argparse.Namespace) -> Path:
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    return args.project_root.expanduser().resolve()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_revision(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("Model revision must be a full lowercase 40-character commit SHA.")
    return value


def encoder_config() -> dict[str, Any]:
    config = read_json(DEFAULTS_ROOT / "encoder.json")
    validate_revision(config["revision"])
    if not config["repo_id"] or config["dimension"] <= 0 or config["max_length"] <= 0:
        raise ValueError("Invalid encoder configuration.")
    return config


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Replace metadata atomically; do not leave partially written manifests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
        ) as stream:
            name = stream.name
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def configure_hub(root: Path, *, offline: bool) -> None:
    """Set cache policy before importing Hugging Face libraries."""
    os.environ["HF_HOME"] = str(root / "cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
