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

    test_nodes = [str(checkout / node) for node in manifest["strict_test_nodes"]]
    return pytest.main(["-q", "-ra", *test_nodes])


if __name__ == "__main__":
    sys.exit(main())
