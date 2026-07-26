#!/usr/bin/env python3
"""Install the built C ABI artifact into the Thouless.jl package layout."""

from __future__ import annotations

import argparse
import platform
import shutil
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
    arguments = parser.parse_args()
    source = ROOT / "target" / arguments.profile / library_name()
    if not source.is_file():
        raise SystemExit(f"missing built C ABI library: {source}")
    destination = (
        ROOT / "julia" / "Thouless" / "deps" / "usr" / "lib" / library_name()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"installed {source.relative_to(ROOT)} at {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
