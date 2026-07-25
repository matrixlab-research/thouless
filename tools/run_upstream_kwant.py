#!/usr/bin/env python3
"""Run the pinned strict Kwant slice against the repository compatibility layer."""

from __future__ import annotations

import argparse
import subprocess
import sys
import types
from pathlib import Path

import pytest
import tomllib


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "spec" / "upstream" / "kwant.toml"


class ResultCounter:
    def __init__(self) -> None:
        self.passed = 0
        self.skipped = 0

    def pytest_runtest_logreport(self, report) -> None:
        if report.skipped and report.when in {"setup", "call"}:
            self.skipped += 1
        elif report.passed and report.when == "call":
            self.passed += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    checkout = args.checkout.resolve()

    with MANIFEST.open("rb") as source:
        manifest = tomllib.load(source)
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != manifest["commit"]:
        parser.error(f"Kwant checkout is {actual_commit}, expected {manifest['commit']}")

    # Lock the compatibility package before exposing only the upstream test
    # namespace. This prevents source implementation modules from satisfying
    # compatibility imports while still letting pytest use package-relative
    # test imports.
    import kwant  # noqa: F401

    tests_package = types.ModuleType("kwant.tests")
    tests_package.__path__ = [str(checkout / "kwant" / "tests")]
    sys.modules["kwant.tests"] = tests_package
    kwant.solvers.__path__ = []
    solver_tests_package = types.ModuleType("kwant.solvers.tests")
    solver_tests_package.__path__ = [
        str(checkout / "kwant" / "solvers" / "tests")
    ]
    sys.modules["kwant.solvers.tests"] = solver_tests_package
    kwant.physics.__path__ = []
    physics_tests_package = types.ModuleType("kwant.physics.tests")
    physics_tests_package.__path__ = [
        str(checkout / "kwant" / "physics" / "tests")
    ]
    sys.modules["kwant.physics.tests"] = physics_tests_package
    kwant.linalg.__path__ = []
    linalg_tests_package = types.ModuleType("kwant.linalg.tests")
    linalg_tests_package.__path__ = [
        str(checkout / "kwant" / "linalg" / "tests")
    ]
    sys.modules["kwant.linalg.tests"] = linalg_tests_package
    kwant.graph.__path__ = []
    graph_tests_package = types.ModuleType("kwant.graph.tests")
    graph_tests_package.__path__ = [
        str(checkout / "kwant" / "graph" / "tests")
    ]
    sys.modules["kwant.graph.tests"] = graph_tests_package

    test_nodes = [str(checkout / node) for node in manifest["strict_test_nodes"]]
    counter = ResultCounter()
    result = pytest.main(["-q", "-ra", *test_nodes], plugins=[counter])
    if result != pytest.ExitCode.OK:
        return int(result)
    expected_passes = manifest.get("strict_passes", manifest["strict_tests"])
    expected_skips = manifest.get("strict_skips", 0)
    if (counter.passed, counter.skipped) != (expected_passes, expected_skips):
        print(
            "strict Kwant outcome mismatch: "
            f"observed {counter.passed} passed and {counter.skipped} skipped; "
            f"expected {expected_passes} passed and {expected_skips} skipped",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
