"""Nested output-workspace names.

A workspace is a folder below Maestro's output root (``save_path``).  The
name ``"default"`` is the output root itself; every other workspace name is a
relative path of one or more folder names joined with ``/``, for example
``"ClientA"`` or ``"ClientA/Shoot-1"``.

Two validation levels exist:

* lenient (browsing an existing folder): any safe folder name, so legacy
  folders created outside Maestro (for example with spaces) stay viewable;
* strict (creating, renaming, moving, activating): every folder name uses
  letters, numbers, hyphens and underscores, as workspace creation always has.

Folders whose names start with ``_`` or ``.`` hold Maestro's own data
(Director assets, Editor projects, trash) and are never workspaces.  The
top-level ``Worlds`` folder belongs to the Worlds feature: it is listed, but
its children are feature data, and it cannot be renamed, moved or nested into.
"""

from __future__ import annotations

import os
import re
from typing import Iterable

DEFAULT_WORKSPACE = "default"
SEPARATOR = "/"
MAX_WORKSPACE_DEPTH = 8
MAX_WORKSPACE_NAME_LENGTH = 240
MAX_SEGMENT_LENGTH = 120
RESERVED_TOP_LEVEL = frozenset({"Worlds"})

_STRICT_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_WINDOWS_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


class WorkspaceNameError(ValueError):
    """A workspace name that cannot address a folder below the output root."""


def is_hidden_folder_name(name: str) -> bool:
    """Maestro-owned folders (``_director_assets``, ``.maestro_editor``...)."""
    return name.startswith(("_", "."))


def split_workspace_name(value: object, *, strict: bool = False) -> tuple[str, ...]:
    """Return the folder names of a workspace; ``()`` is the output root.

    Back-slashes are accepted as separators so a Windows-style name resolves
    to the same folder, but the canonical form always uses ``/``.
    """
    text = "" if value is None else str(value).strip()
    if not text or text == DEFAULT_WORKSPACE:
        return ()
    if len(text) > MAX_WORKSPACE_NAME_LENGTH:
        raise WorkspaceNameError("Workspace name is too long.")
    parts = tuple(text.replace("\\", SEPARATOR).split(SEPARATOR))
    if len(parts) > MAX_WORKSPACE_DEPTH:
        raise WorkspaceNameError(
            f"Workspaces can be nested at most {MAX_WORKSPACE_DEPTH} folders deep."
        )
    for part in parts:
        if part in {"", ".", ".."} or len(part) > MAX_SEGMENT_LENGTH:
            raise WorkspaceNameError("Invalid workspace name.")
        if ":" in part or any(ord(character) < 32 or ord(character) == 127 for character in part):
            raise WorkspaceNameError("Invalid workspace name.")
        if strict:
            if not _STRICT_SEGMENT.match(part):
                raise WorkspaceNameError(
                    "Invalid workspace name. Use letters, numbers, hyphens, underscores."
                )
            if part.casefold() == DEFAULT_WORKSPACE or part.upper() in _WINDOWS_DEVICE_NAMES:
                raise WorkspaceNameError(f'"{part}" is reserved and cannot be a folder name.')
    return parts


def join_workspace_name(parts: Iterable[str]) -> str:
    parts = tuple(parts)
    return SEPARATOR.join(parts) if parts else DEFAULT_WORKSPACE


def normalize_workspace_name(value: object, *, strict: bool = False) -> str:
    """Canonical workspace name (``default`` or ``a/b``)."""
    return join_workspace_name(split_workspace_name(value, strict=strict))


def workspace_parent(name: object) -> str:
    """Parent workspace (``default`` for a top-level folder)."""
    parts = split_workspace_name(name)
    return join_workspace_name(parts[:-1]) if parts else DEFAULT_WORKSPACE


def workspace_leaf(name: object) -> str:
    parts = split_workspace_name(name)
    return parts[-1] if parts else DEFAULT_WORKSPACE


def workspace_depth(name: object) -> int:
    return len(split_workspace_name(name))


def _fold(part: str) -> str:
    # Output folders live on the host filesystem; Windows and macOS treat
    # "Clients" and "clients" as the same folder.
    return os.path.normcase(part) if os.name == "nt" else part


def is_within_workspace(name: object, ancestor: object) -> bool:
    """True when ``name`` is ``ancestor`` or a folder somewhere below it.

    Every workspace is within ``default``.  Invalid names are never within.
    """
    try:
        parts = split_workspace_name(name)
        ancestor_parts = split_workspace_name(ancestor)
    except WorkspaceNameError:
        return False
    if len(parts) < len(ancestor_parts):
        return False
    return all(_fold(a) == _fold(b) for a, b in zip(parts, ancestor_parts))


def rebase_workspace_name(name: object, old: object, new: object) -> str | None:
    """Rename ``old`` (and everything below it) to ``new``.

    Returns the rebased canonical name, or ``None`` when ``name`` is not
    within ``old``.
    """
    if not is_within_workspace(name, old):
        return None
    parts = split_workspace_name(name)
    old_parts = split_workspace_name(old)
    return join_workspace_name(split_workspace_name(new) + parts[len(old_parts):])


def workspace_directory(root: str, name: object, *, strict: bool = False) -> str:
    """Folder path for a workspace (not resolved, not created)."""
    parts = split_workspace_name(name, strict=strict)
    return os.path.join(root, *parts) if parts else root


def _is_inside(path: str, root: str) -> bool:
    try:
        candidate = os.path.normcase(os.path.realpath(path))
        boundary = os.path.normcase(os.path.realpath(root))
        return os.path.commonpath([candidate, boundary]) == boundary
    except (OSError, ValueError):
        return False


def scan_workspace_tree(
    root: str,
    *,
    count_files: bool = True,
    max_depth: int = MAX_WORKSPACE_DEPTH,
) -> list[dict]:
    """Every workspace below ``root`` in depth-first order (root excluded).

    Each entry is ``{"name", "path", "file_count"}`` where ``file_count`` is
    the number of non-hidden files directly inside the folder.  Children of
    hidden folders and of the reserved ``Worlds`` folder are skipped, and the
    walk never follows a link or junction out of ``root`` or into a loop.
    """
    results: list[dict] = []
    if not os.path.isdir(root):
        return results
    visited: set[str] = set()

    def _visit(directory: str, parts: tuple[str, ...]) -> None:
        try:
            real = os.path.normcase(os.path.realpath(directory))
        except OSError:
            return
        if real in visited:
            return
        visited.add(real)
        try:
            with os.scandir(directory) as iterator:
                entries = list(iterator)
        except OSError:
            return
        child_names = []
        for entry in entries:
            name = entry.name
            if is_hidden_folder_name(name):
                continue
            try:
                if entry.is_dir():
                    child_names.append(name)
            except OSError:
                continue
        for name in sorted(child_names):
            child_parts = parts + (name,)
            child = os.path.join(directory, name)
            # Listing a folder that a junction points elsewhere matches the
            # old one-level behaviour; descending through it does not.
            if len(child_parts) > 1 and not _is_inside(child, root):
                continue
            entry_info = {
                "name": join_workspace_name(child_parts),
                "path": child,
                "file_count": _file_count(child) if count_files else 0,
            }
            results.append(entry_info)
            reserved = len(child_parts) == 1 and name in RESERVED_TOP_LEVEL
            if not reserved and len(child_parts) < max_depth and _is_inside(child, root):
                _visit(child, child_parts)

    _visit(root, ())
    return results


def _file_count(path: str) -> int:
    """Non-hidden files directly inside a folder (scandir answers is_file()
    from the enumeration data on Windows, so this needs no per-file stat)."""
    try:
        with os.scandir(path) as entries:
            return sum(1 for entry in entries if not entry.name.startswith(".") and entry.is_file())
    except OSError:
        return 0


def iter_workspace_dirs(root: str) -> list[tuple[str, str]]:
    """``(name, path)`` for the output root and every nested workspace."""
    return [(DEFAULT_WORKSPACE, root)] + [
        (entry["name"], entry["path"])
        for entry in scan_workspace_tree(root, count_files=False)
    ]
