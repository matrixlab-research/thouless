#!/usr/bin/env python3
"""Run realistic-size scientific paths and record time/allocation proxies."""

from __future__ import annotations

import json
import resource
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "nightly-metrics.json"

COMMANDS = {
    "kpm_large_sparse": [
        "cargo",
        "test",
        "--release",
        "large_sparse_recurrence_never_requires_dense_storage",
    ],
    "transport_large_sparse": [
        "cargo",
        "test",
        "--release",
        "sparse_open_system_scales_to_a_large_chain_without_dense_device_storage",
    ],
    "language_conformance": [
        "cargo",
        "run",
        "--release",
        "--quiet",
        "--example",
        "language_conformance",
    ],
}


def main() -> None:
    metrics: dict[str, dict[str, object]] = {}
    for name, command in COMMANDS.items():
        before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        started = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - started
        after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        metrics[name] = {
            "command": command,
            "elapsed_seconds": elapsed,
            "maximum_resident_set_kib": max(before, after),
            "return_code": result.returncode,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
        if result.returncode != 0:
            OUTPUT.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            raise SystemExit(result.returncode)
    OUTPUT.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"recorded {len(metrics)} nightly scientific metrics in {OUTPUT.name}")


if __name__ == "__main__":
    main()
