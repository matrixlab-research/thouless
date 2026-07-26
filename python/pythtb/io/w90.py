"""Wannier90 text-file readers for the PythTB compatibility layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


BOHRTOANG = 0.52917721092


class W90ParseError(RuntimeError):
    """Raised when required Wannier90 text cannot be parsed."""


class W90ConsistencyError(RuntimeError):
    """Raised when independently parsed Wannier90 data disagree."""


@dataclass(frozen=True)
class HRBlock:
    """One real-space Hamiltonian block and its Wigner-Seitz degeneracy."""

    h: np.ndarray
    degeneracy: int


@dataclass(frozen=True)
class W90Dataset:
    """Parsed Wannier90 lattice, centers, Hamiltonian, path, and bands."""

    prefix: str
    root: Path
    lat_cart: np.ndarray
    centres_xyz: np.ndarray
    centres_red: np.ndarray
    num_wan: int
    ham_r: dict[tuple[int, int, int], HRBlock]
    kpath_nodes_red: np.ndarray | None = None
    kpath_labels: list[str] | None = None
    bands_k_red: np.ndarray | None = None
    bands_ene_ev: np.ndarray | None = None
    meta: dict | None = None


def _lines(path):
    path = Path(path)
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except FileNotFoundError as error:
        raise W90ParseError(f"Missing file: {path}") from error


def _block_marker(line, keyword, name):
    """Return text following a flexible Wannier90 block marker."""
    match = re.match(
        rf"^{re.escape(keyword)}\s*:?\s*{re.escape(name)}\b(.*)$",
        line,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return match.group(1).strip().lstrip(":").strip()


def _block(lines, name):
    result = []
    collecting = False
    for raw in lines:
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if not collecting:
            remainder = _block_marker(line, "begin", name)
            if remainder is None:
                continue
            collecting = True
            if remainder:
                result.append(remainder.replace(",", " "))
            continue
        if _block_marker(line, "end", name) is not None:
            return result
        result.append(line.replace(",", " "))
    return result


def read_win(root, prefix):
    """Return raw lines from ``prefix.win``."""
    return _lines(Path(root).expanduser() / f"{prefix}.win")


def parse_unit_cell_cart(win_lines):
    """Parse the three Cartesian primitive vectors in angstroms."""
    values = _block(win_lines, "unit_cell_cart")
    if not values:
        raise W90ParseError("unit_cell_cart block is missing")
    unit = values[0].lower()
    scale = BOHRTOANG if unit == "bohr" else 1.0
    if unit in ("bohr", "ang", "angstrom"):
        values = values[1:]
    if len(values) < 3:
        raise W90ParseError("unit_cell_cart requires three vectors")
    try:
        lattice = np.asarray(
            [[float(part) for part in row.split()[:3]] for row in values[:3]],
            dtype=float,
        )
    except ValueError as error:
        raise W90ParseError("unit_cell_cart contains nonnumeric data") from error
    if lattice.shape != (3, 3) or abs(np.linalg.det(lattice)) < 1e-14:
        raise W90ConsistencyError("unit_cell_cart must be a nonsingular 3x3 matrix")
    return scale * lattice


def read_centres(root, prefix, num_wan):
    """Read the first ``num_wan`` Wannier centers in Cartesian coordinates."""
    lines = _lines(Path(root).expanduser() / f"{prefix}_centres.xyz")
    if len(lines) < int(num_wan) + 2:
        raise W90ParseError("centres file is shorter than num_wan")
    result = []
    for line in lines[2 : 2 + int(num_wan)]:
        fields = line.split()
        if len(fields) < 4 or fields[0] != "X":
            raise W90ParseError("Wannier centers must be tagged with X")
        try:
            result.append([float(value) for value in fields[1:4]])
        except ValueError as error:
            raise W90ParseError("Wannier center is not numeric") from error
    return np.asarray(result, dtype=float)


def read_hr(root, prefix):
    """Return ``(num_wan, {R: HRBlock})`` from ``prefix_hr.dat``."""
    path = Path(root).expanduser() / f"{prefix}_hr.dat"
    lines = _lines(path)
    if len(lines) < 4:
        raise W90ParseError("_hr.dat header is incomplete")
    try:
        num_wan = int(lines[1].split()[0])
        shell_count = int(lines[2].split()[0])
    except (ValueError, IndexError) as error:
        raise W90ParseError("cannot read num_wan or shell count") from error
    if num_wan < 1 or shell_count < 1:
        raise W90ConsistencyError("num_wan and shell count must be positive")
    degeneracies = []
    cursor = 3
    while len(degeneracies) < shell_count and cursor < len(lines):
        try:
            degeneracies.extend(int(value) for value in lines[cursor].split())
        except ValueError as error:
            raise W90ParseError("invalid Wigner-Seitz degeneracy") from error
        cursor += 1
    if len(degeneracies) < shell_count:
        raise W90ParseError("degeneracy list is incomplete")

    ordered_vectors = []
    blocks = {}
    for line in lines[cursor:]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 7:
            raise W90ParseError("_hr.dat matrix rows require seven columns")
        try:
            vector = tuple(int(value) for value in fields[:3])
            row = int(fields[3]) - 1
            column = int(fields[4]) - 1
            value = complex(float(fields[5]), float(fields[6]))
        except ValueError as error:
            raise W90ParseError("invalid _hr.dat matrix row") from error
        if not 0 <= row < num_wan or not 0 <= column < num_wan:
            raise W90ConsistencyError("Wannier index is outside num_wan")
        if vector not in blocks:
            ordered_vectors.append(vector)
            blocks[vector] = np.zeros((num_wan, num_wan), dtype=complex)
        blocks[vector][row, column] += value
    if len(ordered_vectors) != shell_count:
        raise W90ConsistencyError(
            "Hamiltonian shell count does not match the degeneracy list"
        )
    return num_wan, {
        vector: HRBlock(blocks[vector], int(degeneracies[index]))
        for index, vector in enumerate(ordered_vectors)
    }


_LABEL = {
    "g": r"\Gamma",
    "gamma": r"\Gamma",
    "Γ": r"\Gamma",
    "delta": r"\Delta",
    "lambda": r"\Lambda",
    "sigma": r"\Sigma",
}


def _format_label(label):
    match = re.fullmatch(r"([^\d]+?)(\d+)?", label.strip())
    base, suffix = match.groups() if match else (label.strip(), None)
    formatted = _LABEL.get(base.lower(), base)
    if len(formatted) > 1 and not formatted.startswith("\\"):
        formatted = rf"\mathrm{{{formatted}}}"
    return (
        rf"${formatted}_{{{suffix}}}$"
        if suffix is not None
        else rf"${formatted}$"
    )


def read_kpoint_path(win_lines, *, latex=True):
    """Return path nodes and labels from a ``kpoint_path`` block."""
    values = _block(win_lines, "kpoint_path")
    if not values:
        return None, None
    nodes = []
    labels = []
    for line in values:
        fields = line.split()
        if len(fields) % 4:
            raise W90ParseError("kpoint_path entries require label and three coordinates")
        for start in range(0, len(fields), 4):
            label = fields[start]
            try:
                point = np.asarray(fields[start + 1 : start + 4], dtype=float)
            except ValueError as error:
                raise W90ParseError("kpoint_path coordinate is not numeric") from error
            if nodes and label == labels[-1][0] and np.allclose(point, nodes[-1]):
                continue
            nodes.append(point)
            labels.append((label, _format_label(label) if latex else label))
    return np.asarray(nodes, dtype=float), [formatted for _, formatted in labels]


def read_bands_w90(root, prefix, num_wan):
    """Read Wannier90 interpolated reduced k-points and band energies."""
    root = Path(root).expanduser()
    k_path = root / f"{prefix}_band.kpt"
    energy_path = root / f"{prefix}_band.dat"
    if not k_path.exists() or not energy_path.exists():
        raise W90ParseError("Wannier90 band files are missing")
    k_points = np.loadtxt(k_path, skiprows=1, ndmin=2)[:, :3]
    rows = np.loadtxt(energy_path, ndmin=2)
    if rows.shape[1] < 2:
        raise W90ParseError("_band.dat requires a coordinate and energy column")
    expected = int(num_wan) * len(k_points)
    if len(rows) != expected:
        raise W90ConsistencyError(
            f"expected {expected} band rows, found {len(rows)}"
        )
    energies = rows[:, 1].reshape(int(num_wan), len(k_points)).T
    return np.asarray(k_points, dtype=float), np.asarray(energies, dtype=float)


def load_w90_dataset(root, prefix):
    """Parse a self-consistent Wannier90 dataset."""
    root = Path(root).expanduser()
    win_lines = read_win(root, prefix)
    lattice = parse_unit_cell_cart(win_lines)
    num_wan, hamiltonian = read_hr(root, prefix)
    centers = read_centres(root, prefix, num_wan)
    reduced_centers = centers @ np.linalg.inv(lattice)
    nodes, labels = read_kpoint_path(win_lines)
    try:
        band_k, band_energies = read_bands_w90(root, prefix, num_wan)
    except W90ParseError:
        band_k = band_energies = None
    return W90Dataset(
        prefix=str(prefix),
        root=root,
        lat_cart=lattice,
        centres_xyz=centers,
        centres_red=reduced_centers,
        num_wan=num_wan,
        ham_r=hamiltonian,
        kpath_nodes_red=nodes,
        kpath_labels=labels,
        bands_k_red=band_k,
        bands_ene_ev=band_energies,
        meta={},
    )


__all__ = [
    "HRBlock",
    "W90Dataset",
    "load_w90_dataset",
    "parse_unit_cell_cart",
    "read_bands_w90",
    "read_centres",
    "read_hr",
    "read_kpoint_path",
    "read_win",
]
