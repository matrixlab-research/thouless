#!/usr/bin/env python3
"""Validate coverage matrices and skipped-test issue links without network access."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 and older
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "spec" / "coverage"
UPSTREAM = ROOT / "spec" / "upstream"
NATIVE_LANGUAGE_CONTRACT = (
    ROOT / "spec" / "api" / "thouless-native-languages.toml"
)
NATIVE_LANGUAGE_DESIGN = ROOT / "docs" / "native-language-api-design.md"
AGENT_ENTRYPOINT = ROOT / "AGENTS.md"
AGENT_INSTRUCTION = ROOT / "instructions" / "scientific-software-reimplementation.md"
AGENT_INSTRUCTION_REFERENCE = "instructions/scientific-software-reimplementation.md"
ISSUE_PATTERN = re.compile(
    r"^https://github\.com/matrixlab-research/thouless/issues/[1-9][0-9]*$"
)
ISSUE_IN_TEXT_PATTERN = re.compile(
    r"https://github\.com/matrixlab-research/thouless/issues/[1-9][0-9]*"
)
VALID_STATUSES = {"implemented", "partial", "missing", "blocked"}
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def fail(message: str) -> None:
    print(f"contract check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for key in ("capability", "interface"):
        result.extend(document.get(key, []))
    return result


def check_matrix(path: Path) -> None:
    with path.open("rb") as source:
        document = tomllib.load(source)
    matrix_rows = rows(document)
    if not matrix_rows:
        fail(f"{path.relative_to(ROOT)} has no coverage rows")

    for index, row in enumerate(matrix_rows, start=1):
        status = row.get("status")
        if status not in VALID_STATUSES:
            fail(f"{path.name} row {index} has invalid status {status!r}")
        if status != "implemented" and not ISSUE_PATTERN.match(row.get("issue", "")):
            fail(f"{path.name} row {index} has no valid gap issue")

    if "source_package" in document:
        inventory_status = document.get("inventory_status")
        if inventory_status == "incomplete":
            if not ISSUE_PATTERN.match(document.get("inventory_issue", "")):
                fail(
                    f"{path.name} has an incomplete inventory without an issue"
                )
        elif inventory_status == "complete":
            api_manifest = document.get("api_manifest", "")
            if not api_manifest.startswith("spec/api/"):
                fail(f"{path.name} has no repository API manifest")
            if not (ROOT / api_manifest).is_file():
                fail(f"{path.name} references a missing API manifest")
        else:
            fail(
                f"{path.name} has invalid inventory status "
                f"{inventory_status!r}"
            )


def check_compatibility_test(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if not ISSUE_IN_TEXT_PATTERN.search(source):
        fail(f"{path.relative_to(ROOT)} does not link its gap issue")
    if "require_compat_module" not in source:
        fail(f"{path.relative_to(ROOT)} can bypass the compatibility import guard")


def check_upstream_manifest(path: Path) -> None:
    with path.open("rb") as source:
        document = tomllib.load(source)
    if not COMMIT_PATTERN.match(document.get("commit", "")):
        fail(f"{path.name} does not pin an exact source commit")
    if not ISSUE_PATTERN.match(document.get("gap_issue", "")):
        fail(f"{path.name} does not link its compatibility gap issue")
    expected_skips = document.get("expected_skip", [])
    declared_skips = document.get("strict_skips", 0)
    if declared_skips != len(expected_skips):
        fail(
            f"{path.name} declares {declared_skips} skips but "
            f"documents {len(expected_skips)}"
        )
    if "strict_passes" in document:
        if document["strict_passes"] + declared_skips != document["strict_tests"]:
            fail(f"{path.name} pass and skip counts do not equal strict_tests")
    for skip in expected_skips:
        if not skip.get("node") or not skip.get("reason"):
            fail(f"{path.name} has an under-specified expected skip")
        if not ISSUE_PATTERN.match(skip.get("issue", "")):
            fail(f"{path.name} has an expected skip without a valid issue")
    status = document.get("collection_status")
    if status == "complete":
        if document.get("collected_tests", 0) < 1:
            fail(f"{path.name} has no collected test count")
    elif status == "blocked":
        if not document.get("collection_blocker"):
            fail(f"{path.name} has a blocked collection without a reason")
    else:
        fail(f"{path.name} has invalid collection status {status!r}")


def check_agent_instructions() -> None:
    if not AGENT_ENTRYPOINT.is_file():
        fail("AGENTS.md is missing")
    if not AGENT_INSTRUCTION.is_file():
        fail(f"{AGENT_INSTRUCTION.relative_to(ROOT)} is missing")

    entrypoint = AGENT_ENTRYPOINT.read_text(encoding="utf-8")
    if AGENT_INSTRUCTION_REFERENCE not in entrypoint:
        fail("AGENTS.md does not point to the repository reimplementation instruction")

    instruction = AGENT_INSTRUCTION.read_text(encoding="utf-8")
    required_terms = (
        "Rust-native API",
        "PythTB 2.0",
        "Kwant 1.5",
        "GitHub Issues",
        "held-out",
        "Prohibit Fitting to Known Tests",
    )
    missing_terms = [term for term in required_terms if term not in instruction]
    if missing_terms:
        fail(f"agent instruction is missing required terms: {missing_terms}")


def check_native_language_contract() -> None:
    if not NATIVE_LANGUAGE_DESIGN.is_file():
        fail(f"{NATIVE_LANGUAGE_DESIGN.relative_to(ROOT)} is missing")
    if not NATIVE_LANGUAGE_CONTRACT.is_file():
        fail(f"{NATIVE_LANGUAGE_CONTRACT.relative_to(ROOT)} is missing")

    instruction = AGENT_INSTRUCTION.read_text(encoding="utf-8")
    for reference in (
        "../docs/native-language-api-design.md",
        "../spec/api/thouless-native-languages.toml",
    ):
        if reference not in instruction:
            fail(f"agent instruction does not reference {reference}")

    with NATIVE_LANGUAGE_CONTRACT.open("rb") as source:
        contract = tomllib.load(source)
    if contract.get("status") != "design":
        fail("native language contract must remain an explicit design")
    if not contract.get("contract_version"):
        fail("native language contract has no version")
    if set(contract.get("languages", {})) != {"rust", "python", "julia"}:
        fail("native language contract must name Rust, Python, and Julia")

    workflows = contract.get("workflow", [])
    identifiers = [workflow.get("id") for workflow in workflows]
    if len(identifiers) != len(set(identifiers)):
        fail("native language contract contains duplicate workflow identifiers")
    for index, workflow in enumerate(workflows, start=1):
        for language in ("rust", "python", "julia"):
            if not workflow.get(language):
                fail(
                    "native language contract workflow "
                    f"{index} has no {language} namespace"
                )

    with (COVERAGE / "native.toml").open("rb") as source:
        native = tomllib.load(source)
    expected = {
        capability["id"]
        for capability in native.get("capability", [])
        if capability["id"] != "validation.held_out"
    }
    actual = set(identifiers)
    if expected != actual:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        fail(
            "native language contract does not match native capabilities: "
            f"missing={missing}, unexpected={unexpected}"
        )


def main() -> None:
    check_agent_instructions()
    check_native_language_contract()
    matrices = sorted(COVERAGE.glob("*.toml"))
    if not matrices:
        fail("no coverage matrices found")
    for matrix in matrices:
        check_matrix(matrix)

    manifests = sorted(UPSTREAM.glob("*.toml"))
    if not manifests:
        fail("no upstream test manifests found")
    for manifest in manifests:
        check_upstream_manifest(manifest)

    tests = sorted((ROOT / "compat-tests").glob("test_*.py"))
    if not tests:
        fail("no compatibility smoke tests found")
    for test in tests:
        check_compatibility_test(test)

    print(
        "validated agent instructions, native language design, "
        f"{len(matrices)} coverage matrices, {len(manifests)} upstream "
        f"manifests, and {len(tests)} test modules"
    )


if __name__ == "__main__":
    main()
