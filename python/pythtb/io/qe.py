"""Quantum ESPRESSO band-file readers."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np


BOHRTOANG = 0.52917721092


class QEParseError(RuntimeError):
    """Raised when a Quantum ESPRESSO bands file is missing or malformed."""


class QEConsistencyError(RuntimeError):
    """Raised when parsed band dimensions contradict the file header."""


_HEADER = re.compile(r"nbnd\s*=\s*(\d+).*?nks\s*=\s*(\d+)", re.I | re.S)


def _three_floats(line):
    try:
        values = [float(value) for value in line.split()]
    except ValueError:
        return None
    return values if len(values) == 3 else None


def read_bands_qe(root, prefix):
    """Return raw k markers, per-k energies, and header metadata."""
    path = Path(root).expanduser() / f"{prefix}_bands.dat"
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError as error:
        raise QEParseError(f"Missing QE bands file: {path}") from error
    metadata = {}
    header = _HEADER.search(text[:5000])
    if header:
        metadata = {
            "nbnd": int(header.group(1)),
            "nks": int(header.group(2)),
        }
    k_points = []
    energies = []
    current = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        marker = _three_floats(line)
        if marker is not None:
            if current:
                energies.append(current)
                current = []
            k_points.append(marker)
            continue
        try:
            current.extend(float(value) for value in line.split())
        except ValueError:
            continue
    if current:
        energies.append(current)
    if len(energies) != len(k_points):
        raise QEConsistencyError("energy rows do not match k-point markers")
    if "nks" in metadata and metadata["nks"] != len(k_points):
        raise QEConsistencyError("parsed k-point count contradicts nks")
    if "nbnd" in metadata and any(
        len(row) != metadata["nbnd"] for row in energies
    ):
        raise QEConsistencyError("parsed energy count contradicts nbnd")
    return np.asarray(k_points, dtype=float), energies, metadata


__all__ = [
    "BOHRTOANG",
    "QEConsistencyError",
    "QEParseError",
    "read_bands_qe",
]
