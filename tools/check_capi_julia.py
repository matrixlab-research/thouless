#!/usr/bin/env python3
"""Validate the stable C ABI and Julia workflow contracts."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    print(f"C ABI/Julia contract check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def load(path: Path) -> dict[str, Any]:
    with path.open("rb") as source:
        return tomllib.load(source)


def package_version(path: Path) -> str:
    return load(path).get("package", {}).get("version", "")


def main() -> None:
    languages = load(ROOT / "spec" / "api" / "thouless-native-languages.toml")
    capi = load(ROOT / "spec" / "api" / "thouless-capi.toml")
    julia = load(ROOT / "spec" / "api" / "Thouless-julia.toml")
    coverage = load(ROOT / "spec" / "coverage" / "julia-native.toml")
    cargo = load(ROOT / "crates" / "thouless-capi" / "Cargo.toml")
    project = load(ROOT / "julia" / "Thouless" / "Project.toml")

    for name, document in (
        ("language inventory", languages),
        ("C ABI inventory", capi),
        ("Julia inventory", julia),
        ("Julia coverage", coverage),
    ):
        if document.get("schema_version") != 1:
            fail(f"{name} has an unsupported schema")
    if languages.get("status") != "stable":
        fail("language inventory is not stable")
    if capi.get("status") != "stable" or julia.get("status") != "stable":
        fail("C ABI and Julia inventories must both be stable")
    if capi.get("package_version") != cargo.get("package", {}).get("version"):
        fail("C ABI inventory and crate versions differ")
    if julia.get("contract_version") != project.get("version"):
        fail("Julia inventory and project versions differ")
    if languages.get("contract_version") != project.get("version"):
        fail("language and Julia contract versions differ")

    target_ids = {row["id"] for row in languages.get("workflow", [])}
    capi_rows = capi.get("workflow", [])
    capi_ids = {row.get("id") for row in capi_rows}
    coverage_rows = coverage.get("capability", [])
    coverage_ids = {row.get("id") for row in coverage_rows}
    if capi_ids != target_ids:
        fail(f"C ABI workflow IDs differ: missing={sorted(target_ids - capi_ids)}")
    if coverage_ids != target_ids:
        fail(f"Julia workflow IDs differ: missing={sorted(target_ids - coverage_ids)}")

    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "crates" / "thouless-capi" / "src").glob("*.rs"))
    )
    exported = set(
        re.findall(
            r'pub(?: unsafe)? extern "C" fn\s+(thouless_[A-Za-z0-9_]+)',
            sources,
        )
    )
    declared = {
        symbol
        for row in capi_rows
        for symbol in row.get("symbols", [])
    }
    missing = declared - exported
    if missing:
        fail(f"C ABI inventory names missing Rust exports: {sorted(missing)}")
    for field in ("header", "documentation"):
        path = ROOT / capi.get(field, "")
        if not path.is_file():
            fail(f"missing C ABI {field}: {path.relative_to(ROOT)}")

    julia_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "julia" / "Thouless" / "src").glob("*.jl"))
    )
    for module in julia.get("module", []):
        for symbol in module.get("symbols", []):
            if not re.search(rf"\b{re.escape(symbol)}\b", julia_source):
                fail(f"{module['name']} inventory symbol is absent: {symbol}")
    for field in ("documentation", "project"):
        path = ROOT / julia.get(field, "")
        if not path.is_file():
            fail(f"missing Julia {field}: {path.relative_to(ROOT)}")
    for row in coverage_rows:
        if row.get("status") != "implemented":
            if not row.get("issue"):
                fail(f"{row['id']} is incomplete without an issue")
            continue
        if not row.get("public_api"):
            fail(f"{row['id']} has no Julia public API")
        test = ROOT / row.get("test", "")
        if not test.is_file():
            fail(f"{row['id']} references missing test {test.relative_to(ROOT)}")

    print(
        f"validated {len(capi_rows)} C ABI workflows, {len(exported)} exports, "
        f"{len(julia.get('module', []))} Julia modules, and "
        f"{len(coverage_rows)} Julia workflow rows"
    )


if __name__ == "__main__":
    main()
