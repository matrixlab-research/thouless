#!/usr/bin/env python3
"""Build a platform Julia/C ABI release archive from tested artifacts."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def library_name() -> str:
    if platform.system() == "Windows":
        return "thouless_capi.dll"
    if platform.system() == "Darwin":
        return "libthouless_capi.dylib"
    return "libthouless_capi.so"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("debug", "release"), default="release")
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    arguments = parser.parse_args()
    source_library = ROOT / "target" / arguments.profile / library_name()
    if not source_library.is_file():
        raise SystemExit(f"missing tested native library: {source_library}")
    system = platform.system().lower()
    machine = platform.machine().lower()
    archive_base = f"thouless-julia-0.1.0-{system}-{machine}"
    arguments.output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="thouless-julia-release-") as temporary:
        root = Path(temporary) / archive_base
        shutil.copytree(ROOT / "julia" / "Thouless", root / "Thouless")
        library_destination = root / "Thouless" / "deps" / "usr" / "lib"
        library_destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_library, library_destination / source_library.name)
        include = root / "include"
        include.mkdir()
        shutil.copy2(
            ROOT / "crates" / "thouless-capi" / "include" / "thouless.h",
            include / "thouless.h",
        )
        shutil.copy2(
            ROOT / "THIRD_PARTY_LICENSES.md",
            root / "THIRD_PARTY_LICENSES.md",
        )
        metadata = {
            "package_version": "0.1.0",
            "abi_version": "1.0",
            "system": platform.system(),
            "machine": platform.machine(),
        }
        (root / "artifact.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.make_archive(
            str(arguments.output / archive_base),
            "zip",
            root_dir=Path(temporary),
            base_dir=archive_base,
        )
    print(f"created {arguments.output / (archive_base + '.zip')}")


if __name__ == "__main__":
    main()
