"""The echo-window path must import on the chirpsounder2 station's Python 3.8."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

MODULES = ("echo_window", "io_chirp", "io_lfs", "spectro", "calibrate", "paths",
           "loader", "io_digisonde", "stations")

BUILTINS = {"list", "dict", "tuple", "set", "frozenset", "type"}


def source(name):
    root = Path(__file__).resolve().parent.parent
    return (root / "muf" / f"{name}.py").read_text(encoding="utf-8")


def runtime(tree):
    """Nodes evaluated when the module runs, so not annotations."""
    hidden = set()
    for node in ast.walk(tree):
        notes = []
        if isinstance(node, (ast.AnnAssign, ast.arg)) and node.annotation:
            notes.append(node.annotation)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.returns:
            notes.append(node.returns)
        for note in notes:
            hidden.update(id(inner) for inner in ast.walk(note))
    return [node for node in ast.walk(tree) if id(node) not in hidden]


@pytest.mark.parametrize("name", MODULES)
def test_module_parses_as_python_38(name):
    try:
        ast.parse(source(name), feature_version=(3, 8))
    except SyntaxError as exc:
        pytest.fail(f"muf/{name}.py does not parse as Python 3.8: {exc}")


@pytest.mark.parametrize("name", MODULES)
def test_module_subscripts_no_builtin_at_runtime(name):
    found = [node.lineno for node in runtime(ast.parse(source(name)))
             if isinstance(node, ast.Subscript)
             and getattr(node.value, "id", None) in BUILTINS]
    assert found == [], (f"muf/{name}.py subscripts a builtin at runtime, "
                         f"which Python 3.8 cannot, at lines {found}")


@pytest.mark.parametrize("name", MODULES)
def test_module_unions_no_types_at_runtime(name):
    found = [node.lineno for node in runtime(ast.parse(source(name)))
             if isinstance(node, ast.Call)
             and getattr(node.func, "id", None) == "isinstance"
             and len(node.args) == 2 and isinstance(node.args[1], ast.BinOp)]
    assert found == [], (f"muf/{name}.py passes an X | Y union to isinstance, "
                         f"which Python 3.8 cannot, at lines {found}")


@pytest.mark.parametrize("name", MODULES)
def test_module_postpones_its_annotations(name):
    assert "from __future__ import annotations" in source(name), \
        f"muf/{name}.py evaluates its annotations, which breaks on Python 3.8"
