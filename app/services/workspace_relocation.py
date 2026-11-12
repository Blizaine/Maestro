"""Keep saved references valid after a workspace folder is renamed or moved.

Maestro records where media lives in several JSON files: generation sidecars
(``*.meta.json``), Director pipeline states, the Director queue and Editor
projects.  They hold workspace names (``"workspace": "ClientA"``), absolute or
output-relative file paths, and gallery URLs (``/api/v1/file/x.mp4?workspace=
ClientA``).  After the folder moves on disk, :class:`WorkspaceRebase` rewrites
only the values that point into the moved folder and leaves everything else
byte-for-byte alone (files without a match are never rewritten).
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from services.workspace_paths import (
    iter_workspace_dirs,
    rebase_workspace_name,
    workspace_leaf,
)

EDITOR_PROJECT_DIR = ".maestro_editor"
PIPELINE_FILE_PREFIX = "_director_pipeline_"
_MAX_PATH_LENGTH = 4096


class WorkspaceRebase:
    """Map references from ``old_name``/``old_dir`` to ``new_name``/``new_dir``.

    ``aliases`` are further ``(old_dir, new_dir)`` spellings of the same move,
    for example the configured path when the output folder is a link and
    ``old_dir`` is its resolved target.  Saved paths may use either.
    """

    def __init__(self, old_name: str, new_name: str, old_dir: str, new_dir: str,
                 *, aliases: Iterable[tuple[str, str]] = (), cwd: str | None = None):
        self.old_name = old_name
        self.new_name = new_name
        self.cwd = os.path.abspath(cwd or os.getcwd())
        self.old_dir = os.path.normpath(os.path.abspath(old_dir))
        self.new_dir = os.path.normpath(os.path.abspath(new_dir))
        self._pairs: list[tuple[str, str, str]] = []
        for old, new in ((old_dir, new_dir), *aliases):
            old = os.path.normpath(os.path.abspath(old))
            key = os.path.normcase(old)
            if all(key != existing for existing, _old, _new in self._pairs):
                self._pairs.append((key, old, os.path.normpath(os.path.abspath(new))))
        self.marker = workspace_leaf(old_name).casefold()

    # -- single values -------------------------------------------------
    def workspace(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        return rebase_workspace_name(value, self.old_name, self.new_name)

    def path(self, value: str) -> str | None:
        if not value or len(value) > _MAX_PATH_LENGTH or ("/" not in value and "\\" not in value):
            return None
        if "://" in value or value.startswith(("data:", "/api/")):
            return None
        try:
            absolute = value if os.path.isabs(value) else os.path.join(self.cwd, value)
            normalized = os.path.normpath(absolute)
        except (TypeError, ValueError):
            return None
        key = os.path.normcase(normalized)
        for old_key, old_dir, new_dir in self._pairs:
            if key == old_key:
                target = new_dir
                break
            if key.startswith(old_key + os.sep):
                target = os.path.join(new_dir, normalized[len(old_dir) + 1:])
                break
        else:
            return None
        if not os.path.isabs(value):
            try:
                target = os.path.relpath(target, self.cwd)
            except ValueError:
                pass
        if "/" in value and "\\" not in value:
            target = target.replace("\\", "/")
        return target

    def url(self, value: str) -> str | None:
        """Rebase the ``workspace`` query parameter of a Maestro file URL."""
        if not value.startswith("/api/") or "workspace=" not in value:
            return None
        try:
            parts = urlsplit(value)
            query = parse_qsl(parts.query, keep_blank_values=True)
        except ValueError:
            return None
        changed = False
        rebased_query = []
        for key, item in query:
            if key == "workspace":
                rebased = rebase_workspace_name(item, self.old_name, self.new_name)
                if rebased is not None and rebased != item:
                    item = rebased
                    changed = True
            rebased_query.append((key, item))
        if not changed:
            return None
        return urlunsplit(parts._replace(query=urlencode(rebased_query, quote_via=quote, safe="")))

    # -- structures ----------------------------------------------------
    def rebase(self, value: Any, key: str | None = None) -> tuple[Any, bool]:
        """Return ``(value, changed)`` with every moved reference rewritten."""
        if isinstance(value, dict):
            changed = False
            result = {}
            for item_key, item in value.items():
                rebased, item_changed = self.rebase(item, item_key if isinstance(item_key, str) else None)
                result[item_key] = rebased
                changed = changed or item_changed
            return (result, True) if changed else (value, False)
        if isinstance(value, list):
            changed = False
            result = []
            for item in value:
                rebased, item_changed = self.rebase(item, key)
                result.append(rebased)
                changed = changed or item_changed
            return (result, True) if changed else (value, False)
        if isinstance(value, str):
            if key is not None and key.lower().endswith("workspace"):
                rebased = self.workspace(value)
                if rebased is not None and rebased != value:
                    return rebased, True
            rebased = self.url(value) or self.path(value)
            if rebased is not None and rebased != value:
                return rebased, True
        return value, False

    def rebase_in_place(self, mapping: dict) -> bool:
        """Rebase a live dict (an in-memory job or pipeline) without copying it."""
        rebased, changed = self.rebase(mapping)
        if changed:
            mapping.clear()
            mapping.update(rebased)
        return changed


def _detect_indent(text: str) -> int | None:
    lines = text.splitlines()
    if len(lines) < 2:
        return None
    stripped = lines[1].lstrip(" ")
    indent = len(lines[1]) - len(stripped)
    return indent or None


def rewrite_json_file(path: str, rebase: WorkspaceRebase) -> bool:
    """Rewrite one JSON file when it references the moved folder.

    The file's modification time is preserved: the gallery and several
    caches order and fingerprint media by mtime.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except (OSError, UnicodeError):
        return False
    if rebase.marker not in text.casefold():
        return False
    try:
        data = json.loads(text)
    except ValueError:
        return False
    rebased, changed = rebase.rebase(data)
    if not changed:
        return False
    temporary = f"{path}.{os.getpid()}.relocate.tmp"
    try:
        stat = os.stat(path)
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(rebased, handle, indent=_detect_indent(text), ensure_ascii=False)
        os.replace(temporary, path)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    except OSError as error:
        print(f"[Workspace] Could not update references in {path}: {error}")
        try:
            os.remove(temporary)
        except OSError:
            pass
        return False
    return True


def _is_reference_file(directory: str, name: str) -> bool:
    if not name.endswith(".json"):
        return False
    if name.endswith(".meta.json") or name.startswith(PIPELINE_FILE_PREFIX):
        return True
    return os.path.basename(directory) == EDITOR_PROJECT_DIR


def _moved_reference_files(new_dir: str) -> Iterable[str]:
    for directory, folders, files in os.walk(new_dir):
        # Editor preview caches are keyed by their source fingerprint and
        # rebuild themselves; trash folders are about to be deleted.
        folders[:] = [
            folder for folder in folders
            if not folder.startswith(".trash_")
            and not (os.path.basename(directory) == EDITOR_PROJECT_DIR and folder == "media_cache")
        ]
        for name in files:
            if _is_reference_file(directory, name):
                yield os.path.join(directory, name)


def _editor_project_files(save_root: str, skip_under: str) -> Iterable[str]:
    """Editor projects in other workspaces can use media from the moved folder."""
    skip_key = os.path.normcase(os.path.normpath(os.path.abspath(skip_under)))
    for _name, directory in iter_workspace_dirs(save_root):
        key = os.path.normcase(os.path.normpath(os.path.abspath(directory)))
        if key == skip_key or key.startswith(skip_key + os.sep):
            continue
        project_dir = os.path.join(directory, EDITOR_PROJECT_DIR)
        try:
            names = os.listdir(project_dir)
        except OSError:
            continue
        for name in names:
            if name.endswith(".json"):
                yield os.path.join(project_dir, name)


def rewrite_workspace_references(save_root: str, rebase: WorkspaceRebase) -> dict:
    """Update every saved JSON reference after the folder has moved."""
    updated = 0
    for path in _moved_reference_files(rebase.new_dir):
        updated += rewrite_json_file(path, rebase)
    for path in _editor_project_files(save_root, rebase.new_dir):
        updated += rewrite_json_file(path, rebase)
    return {"files_updated": updated}


__all__ = [
    "WorkspaceRebase",
    "rewrite_json_file",
    "rewrite_workspace_references",
]
