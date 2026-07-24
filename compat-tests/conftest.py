"""Shared safeguards for source-interface compatibility tests."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPAT_ROOT = REPOSITORY_ROOT / "python"


def require_compat_module(module_name: str, issue_url: str) -> ModuleType:
    """Import only a compatibility module supplied by this repository."""

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            pytest.skip(f"{module_name} compatibility layer is pending: {issue_url}")
        raise

    module_file = getattr(module, "__file__", None)
    if module_file is None:
        pytest.fail(f"{module_name} does not expose a verifiable module path")

    allowed_root = Path(
        os.environ.get("THOULESS_COMPAT_ROOT", DEFAULT_COMPAT_ROOT)
    ).resolve()
    resolved_module = Path(module_file).resolve()
    try:
        resolved_module.relative_to(allowed_root)
    except ValueError:
        pytest.fail(
            f"{module_name} resolved to {resolved_module}, outside the Thouless "
            f"compatibility root {allowed_root}; the source implementation must "
            "not satisfy compatibility tests"
        )
    return module
