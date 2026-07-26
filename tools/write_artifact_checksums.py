#!/usr/bin/env python3
"""Write portable SHA-256 checksums for release artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("SHA256SUMS"))
    arguments = parser.parse_args()
    artifacts = sorted(
        path
        for path in arguments.directory.rglob("*")
        if path.is_file() and path.resolve() != arguments.output.resolve()
    )
    lines = []
    for path in artifacts:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(arguments.directory).as_posix()}")
    arguments.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote checksums for {len(artifacts)} artifacts")


if __name__ == "__main__":
    main()
