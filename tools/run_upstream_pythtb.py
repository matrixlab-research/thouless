#!/usr/bin/env python3
"""Run the pinned strict PythTB slice against the repository compatibility layer."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "spec" / "upstream" / "pythtb.toml"


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
        parser.error(
            f"PythTB checkout is {actual_commit}, expected {manifest['commit']}"
        )

    # Populate sys.modules before pytest adds the source checkout to sys.path.
    # This guarantees that source tests exercise the repository compatibility
    # package and never satisfy themselves with the original implementation.
    import pythtb  # noqa: F401

    # Some upstream example tests import shared helpers from tests.utils. Make
    # those test-only modules importable after locking the pythtb module above.
    sys.path.insert(0, str(checkout))

    test_paths = [str(checkout / path) for path in manifest["strict_test_files"]]
    return pytest.main(["-q", "-ra", *test_paths])


if __name__ == "__main__":
    sys.exit(main())
