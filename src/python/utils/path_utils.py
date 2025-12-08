"""Utilities for resolving repository paths in a consistent way.

The helper discovers the project root (directory containing ``setup.py`` or
``.git``) starting from this file's location. This works both when the package
is installed in editable mode and when it is imported directly from the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable


def _discover_project_root() -> Path:
    """Return the repository root by walking parents until a marker is found."""
    start = Path(__file__).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "setup.py").exists() or (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Could not locate project root (missing setup.py/.git)")


# Public constant for convenience across scripts.
PROJECT_ROOT: Path = _discover_project_root()


def get_project_root() -> Path:
    """Return the repository root as a ``Path``."""
    return PROJECT_ROOT


def get_python_src() -> Path:
    """Path to ``src/python`` where the package modules live."""
    return PROJECT_ROOT / "src" / "python"


def get_scripts_src() -> Path:
    """Path to ``src/scripts`` for standalone utility scripts."""
    return PROJECT_ROOT / "src" / "scripts"


def ensure_repo_paths_on_sys_path(extra: Iterable[Path] | None = None) -> None:
    """Ensure key repo paths are on ``sys.path`` for script-style entry points.

    The base paths added are ``src/python`` and ``src/scripts``; callers can
    extend this list by passing additional Paths via ``extra``.
    """
    to_add = [get_python_src(), get_scripts_src()]
    if extra:
        to_add.extend(extra)

    for path in to_add:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
