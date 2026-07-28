#!/usr/bin/env python3
"""Build and assemble the Rust, Python, and Julia documentation sites."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run one documentation command from the repository root."""
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def replace_tree(source: Path, destination: Path) -> None:
    """Copy a generated documentation tree into the combined site."""
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def main() -> None:
    """Build all native references and assemble the documentation portal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=ROOT / "target" / "site",
        help="destination for the combined static site",
    )
    arguments = parser.parse_args()
    site_dir = arguments.site_dir.resolve()
    python_build = ROOT / "target" / "docs-python"
    rust_build = ROOT / "target" / "docs-rust"

    command_env = os.environ.copy()
    # Some local Conda and desktop environments export a locale name that the
    # subprocess Python runtime does not provide. The portable C locale is
    # sufficient because every documentation source is explicitly UTF-8.
    command_env["LC_ALL"] = "C"
    command_env["LANG"] = "C"
    rustdoc_env = command_env.copy()
    rustdoc_env["CARGO_TARGET_DIR"] = str(rust_build)
    rustdoc_env["RUSTDOCFLAGS"] = " ".join(
        filter(None, [rustdoc_env.get("RUSTDOCFLAGS", ""), "-D warnings"])
    )
    run(
        ["cargo", "doc", "--package", "thouless", "--all-features", "--no-deps"],
        env=rustdoc_env,
    )
    run(
        [
            "sphinx-build",
            "-W",
            "--keep-going",
            "-b",
            "html",
            "docs/python",
            str(python_build),
        ],
        env=command_env,
    )
    run(
        [
            "julia",
            "--startup-file=no",
            "--project=julia/Thouless/docs",
            "julia/Thouless/docs/make.jl",
        ],
        env=command_env,
    )
    run(
        ["mkdocs", "build", "--strict", "--site-dir", str(site_dir)],
        env=command_env,
    )

    replace_tree(rust_build / "doc", site_dir / "rust")
    replace_tree(python_build, site_dir / "python")
    replace_tree(
        ROOT / "julia" / "Thouless" / "docs" / "build",
        site_dir / "julia",
    )
    print(f"Combined documentation: {site_dir}", flush=True)


if __name__ == "__main__":
    main()
