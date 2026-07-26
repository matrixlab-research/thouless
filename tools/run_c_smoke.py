#!/usr/bin/env python3
"""Compile and run the public C header against the built dynamic library."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def default_library(profile: str) -> Path:
    suffix = {"Darwin": ".dylib", "Windows": ".dll"}.get(platform.system(), ".so")
    prefix = "" if platform.system() == "Windows" else "lib"
    return ROOT / "target" / profile / f"{prefix}thouless_capi{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("debug", "release"), default="debug")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--valgrind", action="store_true")
    arguments = parser.parse_args()
    library = (arguments.library or default_library(arguments.profile)).resolve()
    if not library.is_file():
        raise SystemExit(f"missing C ABI library: {library}")

    with tempfile.TemporaryDirectory(prefix="thouless-c-smoke-") as temporary:
        executable = Path(temporary) / ("smoke.exe" if os.name == "nt" else "smoke")
        include = ROOT / "crates" / "thouless-capi" / "include"
        source = ROOT / "crates" / "thouless-capi" / "tests" / "c_smoke.c"
        if platform.system() == "Windows":
            candidates = [
                library.with_suffix(".dll.lib"),
                library.with_suffix(".lib"),
                library.parent / "thouless_capi.dll.lib",
                library.parent / "thouless_capi.lib",
            ]
            import_library = next(
                (candidate for candidate in candidates if candidate.is_file()),
                None,
            )
            if import_library is None:
                raise SystemExit(f"missing import library beside {library}")
            command = [
                os.environ.get("CC", "cl"),
                "/nologo",
                "/W4",
                "/WX",
                f"/I{include}",
                str(source),
                str(import_library),
                f"/Fe:{executable}",
            ]
        else:
            command = [
                os.environ.get("CC", "cc"),
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                f"-I{include}",
                str(source),
                str(library),
                "-o",
                str(executable),
            ]
        subprocess.run(command, check=True)
        environment = os.environ.copy()
        if platform.system() == "Darwin":
            environment["DYLD_LIBRARY_PATH"] = str(library.parent)
        elif platform.system() == "Windows":
            environment["PATH"] = f"{library.parent}{os.pathsep}{environment['PATH']}"
        elif platform.system() != "Windows":
            environment["LD_LIBRARY_PATH"] = str(library.parent)
        run_command = [str(executable)]
        if arguments.valgrind:
            run_command = [
                "valgrind",
                "--error-exitcode=1",
                "--leak-check=full",
                "--show-leak-kinds=definite",
                str(executable),
            ]
        subprocess.run(run_command, check=True, env=environment)
    print(f"validated C header and dynamic library: {library}")


if __name__ == "__main__":
    main()
