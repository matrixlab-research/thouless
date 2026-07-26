#!/usr/bin/env python3
"""Validate the installed first-class Python API and its coverage matrix."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
API_PATH = ROOT / "spec" / "api" / "thouless-python.toml"
COVERAGE_PATH = ROOT / "spec" / "coverage" / "python-native.toml"
LANGUAGE_PATH = ROOT / "spec" / "api" / "thouless-native-languages.toml"
PYPROJECT_PATH = ROOT / "pyproject.toml"


def fail(message: str) -> None:
    print(f"Python native API check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def load(path: Path) -> dict[str, Any]:
    with path.open("rb") as source:
        return tomllib.load(source)


def main() -> None:
    api = load(API_PATH)
    coverage = load(COVERAGE_PATH)
    languages = load(LANGUAGE_PATH)
    pyproject = load(PYPROJECT_PATH)

    if api.get("schema_version") != 1:
        fail("unsupported API inventory schema")
    if coverage.get("schema_version") != 1:
        fail("unsupported coverage schema")
    if api.get("status") != "stable":
        fail("Python inventory is not marked stable")
    version = pyproject.get("project", {}).get("version")
    if api.get("contract_version") != version:
        fail("Python contract and package versions differ")

    for field in ("documentation", "typing_marker"):
        path = ROOT / api.get(field, "")
        if not path.is_file():
            fail(f"missing {field}: {path.relative_to(ROOT)}")

    module_count = 0
    symbol_count = 0
    for entry in api.get("module", []):
        module_name = entry.get("name")
        if not module_name:
            fail("a module inventory entry has no name")
        module = importlib.import_module(module_name)
        symbols = entry.get("symbols", [])
        if not symbols:
            fail(f"{module_name} has no public symbols")
        missing = [symbol for symbol in symbols if not hasattr(module, symbol)]
        if missing:
            fail(f"{module_name} is missing {missing}")
        exports = getattr(module, "__all__", ())
        missing_exports = [symbol for symbol in symbols if symbol not in exports]
        if missing_exports:
            fail(f"{module_name}.__all__ is missing {missing_exports}")
        module_count += 1
        symbol_count += len(symbols)

    root = importlib.import_module("thouless")
    if "_core" in root.__all__:
        fail("private _core leaked into thouless.__all__")

    target_ids = {
        row["id"]
        for row in languages.get("workflow", [])
    }
    rows = coverage.get("capability", [])
    coverage_ids = {row.get("id") for row in rows}
    if coverage_ids != target_ids:
        fail(
            "coverage/language workflow mismatch: "
            f"missing={sorted(target_ids - coverage_ids)}, "
            f"extra={sorted(coverage_ids - target_ids)}"
        )
    for row in rows:
        identifier = row["id"]
        if row.get("status") != "implemented":
            if not row.get("issue"):
                fail(f"{identifier} is incomplete without an issue")
            continue
        if not row.get("public_api"):
            fail(f"{identifier} has no public API")
        test = ROOT / row.get("test", "")
        if not test.is_file():
            fail(f"{identifier} references missing test {test.relative_to(ROOT)}")

    print(
        f"validated {module_count} Python modules, {symbol_count} public symbols, "
        f"and {len(rows)} workflow rows"
    )


if __name__ == "__main__":
    main()
