from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any


_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{1,127}$")
_BASE_MODEL_ENTRIES = ("model_index.json", "scheduler", "text_encoder", "tokenizer", "unet", "vae")


@dataclass(frozen=True, slots=True)
class HypirRuntimePaths:
    python: Path
    repository_root: Path
    base_model_root: Path
    weight: Path


def _path_from_config(config: dict[str, Any], key: str, environment_key: str) -> tuple[Path | None, str | None]:
    raw = config.get(key)
    if isinstance(raw, str) and raw.strip():
        return Path(raw).expanduser().resolve(), None
    environment_name = config.get(environment_key)
    if not isinstance(environment_name, str) or not _ENV_NAME.fullmatch(environment_name):
        return None, f"HYPIR {environment_key} must name an environment variable"
    environment_value = os.getenv(environment_name)
    if not environment_value:
        return None, f"Set {environment_name} to configure the optional HYPIR-SD2 runtime"
    return Path(environment_value).expanduser().resolve(), None


def resolve_hypir_runtime(config: dict[str, Any]) -> tuple[HypirRuntimePaths | None, str | None]:
    if config.get("allow_external_code") is not True:
        return None, "HYPIR-SD2 requires allow_external_code: true because it executes the official local PyTorch package"

    runtime_python, reason = _path_from_config(config, "runtime_python", "runtime_python_env")
    if reason:
        return None, reason
    repository_root, reason = _path_from_config(config, "repository_root", "repository_root_env")
    if reason:
        return None, reason
    base_model_root, reason = _path_from_config(config, "base_model_root", "base_model_root_env")
    if reason:
        return None, reason
    weight, reason = _path_from_config(config, "weight_path", "weight_path_env")
    if reason:
        return None, reason
    assert runtime_python is not None and repository_root is not None and base_model_root is not None and weight is not None

    if not runtime_python.is_file() or not os.access(runtime_python, os.X_OK):
        return None, f"HYPIR Python executable was not found or is not executable: {runtime_python}"
    if not (repository_root / "HYPIR" / "enhancer" / "sd2.py").is_file():
        return None, f"HYPIR repository root is invalid: {repository_root}"
    missing_entries = [entry for entry in _BASE_MODEL_ENTRIES if not (base_model_root / entry).exists()]
    if missing_entries:
        return None, f"Stable Diffusion 2.1 base model is incomplete; missing: {', '.join(missing_entries)}"
    if not weight.is_file() or weight.suffix.casefold() != ".pth":
        return None, f"HYPIR-SD2 LoRA weight was not found: {weight}"
    return HypirRuntimePaths(runtime_python, repository_root, base_model_root, weight), None
