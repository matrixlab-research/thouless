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
ISSUE_PATTERN = re.compile(
    r"^https://github\.com/matrixlab-research/thouless/issues/[1-9][0-9]*$"
)
ISSUE_IN_TEXT_PATTERN = re.compile(
    r"https://github\.com/matrixlab-research/thouless/issues/[1-9][0-9]*"
)
VALID_STATUSES = {"implemented", "partial", "missing", "blocked"}


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

    if document.get("inventory_status") == "incomplete":
        if not ISSUE_PATTERN.match(document.get("inventory_issue", "")):
            fail(f"{path.name} has an incomplete inventory without an issue")


def check_compatibility_test(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if not ISSUE_IN_TEXT_PATTERN.search(source):
        fail(f"{path.relative_to(ROOT)} does not link its gap issue")
    if "require_compat_module" not in source:
        fail(f"{path.relative_to(ROOT)} can bypass the compatibility import guard")


def main() -> None:
    matrices = sorted(COVERAGE.glob("*.toml"))
    if not matrices:
        fail("no coverage matrices found")
    for matrix in matrices:
        check_matrix(matrix)

    tests = sorted((ROOT / "compat-tests").glob("test_*.py"))
    if not tests:
        fail("no compatibility smoke tests found")
    for test in tests:
        check_compatibility_test(test)

    print(f"validated {len(matrices)} coverage matrices and {len(tests)} test modules")


if __name__ == "__main__":
    main()
