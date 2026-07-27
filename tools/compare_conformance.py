#!/usr/bin/env python3
"""Compare generated Rust, Python, and Julia physical-invariant reports."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


EXPECTED = {
    "ad_projector_value": (0.5 * (1.0 - 1.0 / math.sqrt(1.04)), 1.0e-12),
    "ad_projector_gradient": (0.1 / 1.04**1.5, 1.0e-12),
    "ssh_gap": (0.8, 1.0e-10),
    "ssh_polarization": (0.75, 1.0e-9),
    "chern_absolute": (1.0, 1.0e-9),
    "vacancy_states": (2.0, 0.0),
    "vacancy_observable_trace": (3.0, 1.0e-12),
    "ballistic_transmission": (1.0, 2.0e-6),
    "wilson_gauge_delta": (0.0, 1.0e-12),
    "invalid_shape_error": (1.0, 0.0),
}


def read(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        name, raw = line.split("=", 1)
        if name in EXPECTED:
            values[name] = float(raw)
    missing = set(EXPECTED) - values.keys()
    if missing:
        raise SystemExit(f"{path} is missing conformance metrics {sorted(missing)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs=3, type=Path)
    arguments = parser.parse_args()
    reports = [read(path) for path in arguments.reports]
    for name, (expected, tolerance) in EXPECTED.items():
        for path, report in zip(arguments.reports, reports, strict=True):
            if not math.isclose(report[name], expected, rel_tol=0.0, abs_tol=tolerance):
                raise SystemExit(
                    f"{path}: {name}={report[name]} differs from "
                    f"{expected} by more than {tolerance}"
                )
        spread = max(report[name] for report in reports) - min(
            report[name] for report in reports
        )
        if spread > tolerance:
            raise SystemExit(
                f"cross-language {name} spread {spread} exceeds {tolerance}"
            )
    print(f"validated {len(EXPECTED)} generated invariants across Rust, Python, and Julia")


if __name__ == "__main__":
    main()
