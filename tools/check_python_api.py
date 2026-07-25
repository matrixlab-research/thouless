#!/usr/bin/env python3
"""Validate source-package API inventories against compatibility modules."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 and older
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "spec" / "api"


def fail(message: str) -> None:
    print(f"API contract check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_attributes(
    owner: Any,
    names: list[str],
    description: str,
) -> None:
    missing = [name for name in names if not hasattr(owner, name)]
    if missing:
        fail(f"{description} is missing {missing}")


def check_manifest(path: Path) -> tuple[int, int]:
    with path.open("rb") as source:
        document = tomllib.load(source)
    if document.get("schema_version") != 1:
        fail(f"{path.name} has an unsupported schema version")
    upstream_path = ROOT / document.get("upstream_manifest", "")
    if not upstream_path.is_file():
        fail(f"{path.name} does not reference an upstream manifest")
    with upstream_path.open("rb") as source:
        upstream = tomllib.load(source)
    if document.get("source_commit") != upstream.get("commit"):
        fail(f"{path.name} does not match its pinned upstream commit")
    if document.get("gap_issue") != upstream.get("gap_issue"):
        fail(f"{path.name} does not match its upstream gap issue")
    modules = document.get("module", [])
    objects = document.get("object", [])
    if not modules:
        fail(f"{path.name} has no module inventory")

    for entry in modules:
        name = entry["name"]
        module = importlib.import_module(name)
        require_attributes(module, entry.get("symbols", []), name)
        expected_exports = entry.get("exports", [])
        actual_exports = getattr(module, "__all__", ())
        missing_exports = [
            symbol
            for symbol in expected_exports
            if symbol not in actual_exports
        ]
        if missing_exports:
            fail(f"{name}.__all__ is missing {missing_exports}")

    for entry in objects:
        module_name = entry["module"]
        object_name = entry["name"]
        module = importlib.import_module(module_name)
        if not hasattr(module, object_name):
            fail(f"{module_name} is missing object {object_name}")
        owner = getattr(module, object_name)
        require_attributes(
            owner,
            entry.get("members", []),
            f"{module_name}.{object_name}",
        )
    return len(modules), len(objects)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="*", type=Path)
    arguments = parser.parse_args()
    manifests = (
        [path.resolve() for path in arguments.manifests]
        if arguments.manifests
        else sorted(API_ROOT.glob("*.toml"))
    )
    if not manifests:
        fail("no API manifests found")

    module_count = 0
    object_count = 0
    for manifest in manifests:
        modules, objects = check_manifest(manifest)
        module_count += modules
        object_count += objects
    print(
        f"validated {len(manifests)} API manifests with "
        f"{module_count} modules and {object_count} objects"
    )


if __name__ == "__main__":
    main()
