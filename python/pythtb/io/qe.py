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


def _qe_float(value):
    return float(value.replace("D", "E").replace("d", "e"))


def _numeric_row(line):
    try:
        return [_qe_float(value) for value in line.split()]
    except ValueError:
        return None


def _three_floats(line):
    values = _numeric_row(line)
    return values if values is not None and len(values) == 3 else None


def _read_header_shaped_rows(text, *, band_count, kpoint_count):
    numeric_rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        values = _numeric_row(line)
        if values is not None:
            numeric_rows.append(values)

    k_points = []
    energies = []
    cursor = 0
    for _ in range(kpoint_count):
        if cursor >= len(numeric_rows) or len(numeric_rows[cursor]) != 3:
            raise QEConsistencyError("expected a three-coordinate k-point marker")
        k_points.append(numeric_rows[cursor])
        cursor += 1

        row = []
        while len(row) < band_count:
            if cursor >= len(numeric_rows):
                raise QEConsistencyError("band-energy table is incomplete")
            values = numeric_rows[cursor]
            cursor += 1
            if len(row) + len(values) > band_count:
                raise QEConsistencyError(
                    "band-energy row contains more values than nbnd"
                )
            row.extend(values)
        energies.append(row)

    if cursor != len(numeric_rows):
        raise QEConsistencyError("numeric rows remain after the declared nks")
    return k_points, energies


def _read_headerless_rows(text):
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
        values = _numeric_row(line)
        if values is not None:
            current.extend(values)
    if current:
        energies.append(current)
    return k_points, energies


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
        k_points, energies = _read_header_shaped_rows(
            text,
            band_count=metadata["nbnd"],
            kpoint_count=metadata["nks"],
        )
    else:
        k_points, energies = _read_headerless_rows(text)
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
