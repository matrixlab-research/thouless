#!/usr/bin/env python3
"""Require docstrings on every locally defined public Python API symbol."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "python" / "thouless"


def exported_names(tree: ast.Module, path: Path) -> list[str]:
    """Extract a literal module ``__all__`` declaration."""
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in statement.targets
        ):
            continue
        try:
            values = ast.literal_eval(statement.value)
        except (ValueError, TypeError, SyntaxError) as error:
            raise ValueError(f"{path}: __all__ must be a literal sequence") from error
        if not isinstance(values, (list, tuple)) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(f"{path}: __all__ must contain only strings")
        return list(values)
    raise ValueError(f"{path}: public modules must define __all__")


def local_definitions(
    statements: list[ast.stmt],
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    """Index top-level functions and classes by binding name."""
    return {
        statement.name: statement
        for statement in statements
        if isinstance(
            statement,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        )
    }


def public_class_members(node: ast.ClassDef) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return public methods and property accessors declared on a class."""
    return [
        statement
        for statement in node.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not statement.name.startswith("_")
    ]


def main() -> None:
    """Report all missing public docstrings and fail if any are found."""
    missing: list[str] = []
    for path in sorted(PACKAGE.glob("*.py")):
        if path.stem.startswith("_") and path.stem != "__init__":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_name = f"thouless.{path.stem}"
        if ast.get_docstring(tree) is None:
            missing.append(module_name)
        definitions = local_definitions(tree.body)
        for name in exported_names(tree, path):
            node = definitions.get(name)
            if node is None:
                # Re-exported modules, errors, and model types are documented at
                # their defining binding.
                continue
            if ast.get_docstring(node) is None:
                missing.append(f"{module_name}.{name}")
            if isinstance(node, ast.ClassDef):
                for member in public_class_members(node):
                    if ast.get_docstring(member) is None:
                        missing.append(f"{module_name}.{name}.{member.name}")

    if missing:
        print("Missing public Python docstrings:")
        for name in missing:
            print(f"- {name}")
        raise SystemExit(1)
    print("Python public API docstrings: complete")


if __name__ == "__main__":
    main()
