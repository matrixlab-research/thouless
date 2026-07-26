#!/usr/bin/env python3
"""Validate the versioned Rust-native public API inventory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "spec" / "api" / "thouless-native.toml"
LANGUAGE_PATH = ROOT / "spec" / "api" / "thouless-native-languages.toml"
COVERAGE_PATH = ROOT / "spec" / "coverage" / "native.toml"
CARGO_PATH = ROOT / "Cargo.toml"


def fail(message: str) -> None:
    print(f"native API check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def load(path: Path) -> dict[str, Any]:
    with path.open("rb") as source:
        return tomllib.load(source)


def load_from_git(ref: str, path: Path) -> dict[str, Any] | None:
    relative = path.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "show", f"{ref}:{relative}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return tomllib.loads(result.stdout.decode("utf-8"))


def workflow_map(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = document.get("workflow", [])
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            fail("a workflow has no identifier")
        if identifier in mapped:
            fail(f"duplicate workflow {identifier}")
        mapped[identifier] = row
    return mapped


def stable_break(
    old: dict[str, Any],
    new: dict[str, Any],
) -> list[str]:
    breaks: list[str] = []
    old_rows = workflow_map(old)
    new_rows = workflow_map(new)
    for identifier, old_row in old_rows.items():
        new_row = new_rows.get(identifier)
        if new_row is None:
            breaks.append(f"removed workflow {identifier}")
            continue
        if old_row.get("module") != new_row.get("module"):
            breaks.append(f"changed module for {identifier}")
        old_symbols = set(old_row.get("symbols", []))
        new_symbols = set(new_row.get("symbols", []))
        for symbol in sorted(old_symbols - new_symbols):
            breaks.append(f"removed {identifier} symbol {symbol}")
    for field in ("precision", "ownership", "coordinates", "errors"):
        if old.get(field) != new.get(field):
            breaks.append(f"changed {field} policy")
    return breaks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-ref",
        help="git revision whose stable inventory is the compatibility baseline",
    )
    arguments = parser.parse_args()

    inventory = load(INVENTORY_PATH)
    languages = load(LANGUAGE_PATH)
    coverage = load(COVERAGE_PATH)
    cargo = load(CARGO_PATH)

    if inventory.get("schema_version") != 1:
        fail("unsupported inventory schema")
    if inventory.get("status") != "stable":
        fail("the Rust inventory is not marked stable")
    package_version = cargo.get("package", {}).get("version")
    contract_version = inventory.get("contract_version")
    if contract_version != package_version:
        fail(
            f"contract version {contract_version!r} does not match "
            f"crate version {package_version!r}"
        )

    for field in (
        "precision",
        "ownership",
        "coordinates",
        "errors",
        "public_api_test",
        "stability_document",
    ):
        if not inventory.get(field):
            fail(f"missing {field}")

    public_test = ROOT / inventory["public_api_test"]
    stability_document = ROOT / inventory["stability_document"]
    if not public_test.is_file():
        fail(f"public API test {public_test.relative_to(ROOT)} is missing")
    if not stability_document.is_file():
        fail(
            f"stability document {stability_document.relative_to(ROOT)} is missing"
        )
    public_test_text = public_test.read_text(encoding="utf-8")

    inventory_rows = workflow_map(inventory)
    language_rows = workflow_map(languages)
    coverage_ids = {
        row["id"]
        for row in coverage.get("capability", [])
        if row.get("id") != "validation.held_out"
    }
    expected = set(language_rows)
    if set(inventory_rows) != expected:
        fail(
            "inventory/language workflow mismatch: "
            f"missing={sorted(expected - set(inventory_rows))}, "
            f"extra={sorted(set(inventory_rows) - expected)}"
        )
    if set(inventory_rows) != coverage_ids:
        fail(
            "inventory/native coverage mismatch: "
            f"missing={sorted(coverage_ids - set(inventory_rows))}, "
            f"extra={sorted(set(inventory_rows) - coverage_ids)}"
        )

    validation_count = 0
    for identifier, row in inventory_rows.items():
        if not row.get("module"):
            fail(f"{identifier} has no stable module")
        symbols = row.get("symbols", [])
        if not symbols:
            fail(f"{identifier} has no stable symbols")
        for symbol in symbols:
            leaf = symbol.rsplit("::", 1)[-1]
            if leaf not in public_test_text:
                fail(f"{identifier} symbol {symbol} is absent from the compile contract")
        validations = row.get("validation", [])
        if not validations:
            fail(f"{identifier} has no direct validation")
        for validation in validations:
            path = ROOT / validation
            if not path.is_file():
                fail(f"{identifier} references missing validation {validation}")
            validation_count += 1

    if arguments.base_ref:
        baseline = load_from_git(arguments.base_ref, INVENTORY_PATH)
        if baseline and baseline.get("status") == "stable":
            breaks = stable_break(baseline, inventory)
            if breaks and baseline.get("contract_version") == contract_version:
                fail(
                    "stable breaking changes require a contract-version update: "
                    + "; ".join(breaks)
                )

    print(
        f"validated Rust contract {contract_version} with "
        f"{len(inventory_rows)} workflows and {validation_count} direct validations"
    )


if __name__ == "__main__":
    main()
