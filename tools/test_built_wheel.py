#!/usr/bin/env python3
"""Install one built wheel into a clean environment and test public APIs."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def virtual_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-directory", type=Path, default=ROOT / "dist")
    parser.add_argument("--keep", type=Path)
    parser.add_argument("--extra")
    parser.add_argument("--skip-tests", action="store_true")
    arguments = parser.parse_args()
    wheels = sorted(arguments.wheel_directory.glob("thouless-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one wheel, found {[path.name for path in wheels]}")

    if arguments.keep:
        environment = arguments.keep.resolve()
        if environment.exists():
            shutil.rmtree(environment)
        environment.parent.mkdir(parents=True, exist_ok=True)
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="thouless-wheel-test-")
        environment = Path(cleanup.name)
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    python = virtual_python(environment)
    wheel_requirement = str(wheels[0])
    if arguments.extra:
        wheel_requirement = f"{wheel_requirement}[{arguments.extra}]"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            wheel_requirement,
            "pytest>=8.3,<9",
        ],
        check=True,
    )
    clean_environment = os.environ.copy()
    clean_environment.pop("PYTHONPATH", None)
    if not arguments.skip_tests:
        subprocess.run(
            [str(python), str(ROOT / "tools" / "check_python_native_api.py")],
            cwd=ROOT,
            env=clean_environment,
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "pytest",
                "-q",
                str(ROOT / "python-tests" / "native"),
            ],
            cwd=ROOT,
            env=clean_environment,
            check=True,
        )
        print(f"validated clean installation of {wheels[0].name}")
    else:
        print(f"installed clean environment from {wheels[0].name}")
    if cleanup is not None:
        cleanup.cleanup()


if __name__ == "__main__":
    main()
