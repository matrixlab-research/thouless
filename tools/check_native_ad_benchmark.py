#!/usr/bin/env python3
"""Validate native-AD accuracy, operation complexity, memory, and timing evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"native AD benchmark failed: {message}")


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: check_native_ad_benchmark.py REPORT.jsonl")
    path = Path(sys.argv[1])
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_name = {record["benchmark"]: record for record in records}
    expected = {"spectral_projector", "sparse_kpm"}
    if set(by_name) != expected:
        fail(f"expected {sorted(expected)}, found {sorted(by_name)}")

    for name, record in by_name.items():
        if record["maximum_relative_error"] >= 1.0e-6:
            fail(f"{name} gradient error is {record['maximum_relative_error']}")
        if record["speedup"] < 2.0:
            fail(f"{name} measured speedup is only {record['speedup']}")

    spectral = by_name["spectral_projector"]
    if spectral["finite_difference_eigensystems"] < 16 * spectral["native_eigensystems"]:
        fail("spectral reverse mode did not reduce eigensystem count by at least 16x")

    kpm = by_name["sparse_kpm"]
    if (
        kpm["finite_difference_operator_applications"]
        < 16 * kpm["native_operator_applications"]
    ):
        fail("KPM reverse mode did not reduce operator applications by at least 16x")
    if kpm["peak_stored_vectors"] >= kpm["full_tape_vectors"]:
        fail("KPM checkpointing did not reduce state-vector storage")

    print(
        "validated native AD: "
        f"spectral {spectral['speedup']:.2f}x, "
        f"KPM {kpm['speedup']:.2f}x, "
        f"errors <= {max(record['maximum_relative_error'] for record in records):.3e}"
    )


if __name__ == "__main__":
    main()
