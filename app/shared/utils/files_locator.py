from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, List, Optional, Union

default_checkpoints_paths = ["ckpts", "."]

_checkpoints_paths = default_checkpoints_paths

# Snapshot the app directory at import time. is_external_root must not
# depend on os.getcwd() at call time: some code paths os.chdir() during
# generation (e.g. the YuE codec builder), which would misclassify roots
# for the duration.
_APP_DIR = os.path.abspath(os.getcwd())

def set_checkpoints_paths(checkpoints_paths):
    global _checkpoints_paths
    _checkpoints_paths = [path.strip() for path in checkpoints_paths if len(path.strip()) > 0 ]
    if len(checkpoints_paths) == 0:
        _checkpoints_paths = default_checkpoints_paths

def is_external_root(root):
    """True for a checkpoints path that lives OUTSIDE the app's own directory
    (e.g. another Pinokio app's ckpts folder, linked via the "model folders"
    setting). External roots are searched for reads but are never chosen as
    a write/delete target — Maestro must not modify another app's install.

    Classification uses the RESOLVED path, so a hand-edited relative entry
    like "../wan.git2/app/ckpts" counts as external, while "ckpts" and "."
    stay internal. Comparison is case-normalized for Windows.
    """
    root_n = os.path.normcase(os.path.abspath(root))
    app_n = os.path.normcase(_APP_DIR)
    try:
        return os.path.commonpath([root_n, app_n]) != app_n
    except ValueError:
        # Different drives on Windows — definitely external.
        return True


def get_checkpoints_paths():
    """The current search roots, primary (write target) first."""
    return list(_checkpoints_paths)


def describe_file_source(path):
    """Describe which configured model root supplies ``path``.

    The model loader normally needs only the resolved filename, but large
    shared installations are much easier to troubleshoot when the terminal
    also says whether a component came from Maestro or a linked app.  Keep
    this helper read-only and path based: it never changes search order or
    assumes that a linked folder is writable.
    """
    if not path:
        return {
            "path": None,
            "kind": "missing",
            "installation": None,
            "root": None,
            "root_index": None,
        }

    resolved = os.path.abspath(os.fspath(path))
    resolved_norm = os.path.normcase(resolved)
    for index, root in enumerate(_checkpoints_paths):
        root_abs = os.path.abspath(root)
        root_norm = os.path.normcase(root_abs)
        try:
            if os.path.commonpath([resolved_norm, root_norm]) != root_norm:
                continue
        except ValueError:
            continue

        # Linked model folders conventionally point at
        # <pinokio>/api/<installation>/app/ckpts.  Preserve a useful label
        # for custom roots as well instead of returning an empty source.
        root_parent = os.path.dirname(root_abs)
        if os.path.basename(root_parent).lower() == "app":
            installation = os.path.basename(os.path.dirname(root_parent))
        else:
            installation = os.path.basename(root_parent) or os.path.basename(root_abs)
        linked = index > 0 and is_external_root(root)
        return {
            "path": resolved,
            "kind": "linked" if linked else "primary",
            "installation": installation,
            "root": root_abs,
            "root_index": index,
        }

    return {
        "path": resolved,
        "kind": "unmanaged",
        "installation": None,
        "root": None,
        "root_index": None,
    }


def is_protected_path(path):
    """True when path lives inside a LINKED install — under a non-primary
    external root's parent app folder (covering its ckpts AND its sibling
    loras). Anything protected must never be deleted or overwritten by
    Maestro. The primary root (index 0) is exempt even when external: it is
    the user-chosen, Maestro-managed download target."""
    if not path:
        return False
    p = os.path.normcase(os.path.abspath(path))
    for root in _checkpoints_paths[1:]:
        if not is_external_root(root):
            continue
        app_of_root = os.path.normcase(os.path.dirname(os.path.abspath(root)))
        try:
            if os.path.commonpath([p, app_of_root]) == app_of_root:
                return True
        except ValueError:
            continue
    return False

def _normalize_force_path(force_path):
    if force_path is not None and isinstance(force_path, list) and len(force_path):
        force_path = force_path[0]
    if force_path is None:
        return None
    force_path = os.fspath(force_path).strip()
    if len(force_path) == 0:
        return None
    normalized = os.path.normpath(force_path)
    return None if normalized in ("", ".") else normalized

def get_download_location(file_name = None, force_path= None):
    if file_name is not None and os.path.isabs(file_name): return file_name
    if force_path is not None and isinstance(force_path, list) and len(force_path): force_path = force_path[0]
    if file_name is not None:
        if force_path is None:
            return os.path.join(_checkpoints_paths[0], file_name)
        else:
            return os.path.join(_checkpoints_paths[0], force_path, file_name)
    else:
        if force_path is None:
            return _checkpoints_paths[0]
        else:
            return os.path.join(_checkpoints_paths[0])

def get_smart_download_root(force_path = None):
    force_path = _normalize_force_path(force_path)
    if force_path is None:
        return _checkpoints_paths[0]
    if os.path.isabs(force_path):
        return force_path
    for folder in _checkpoints_paths:
        # Never resume/extend a folder inside an external (linked) root:
        # reads find its files fine, but new downloads must land in a
        # writable root so linked installs stay read-only. The FIRST entry
        # is exempt — it is the user-chosen download target even when it
        # points outside the app (upstream supports absolute primaries
        # like D:/models).
        if folder != _checkpoints_paths[0] and is_external_root(folder):
            continue
        candidate = os.path.join(folder, force_path)
        if os.path.isdir(candidate):
            return folder
    return _checkpoints_paths[0]

def get_smart_download_location(file_name = None, force_path = None):
    if file_name is not None and os.path.isabs(file_name):
        return file_name
    force_path = _normalize_force_path(force_path)
    if force_path is None:
        return get_download_location(file_name)
    if os.path.isabs(force_path):
        return force_path if file_name is None else os.path.join(force_path, file_name)
    root = get_smart_download_root(force_path)
    base_path = os.path.join(root, force_path)
    return base_path if file_name is None else os.path.join(base_path, file_name)

def locate_folder(folder_name, error_if_none = True, required_files = None):
    """Find folder_name across the search roots (primary first).

    required_files: optional list of filenames that must exist INSIDE the
    folder for a root to qualify. Guards against sparse-folder shadowing
    (issue #17): a folder created in the primary root by one download
    (e.g. just the text-encoder weight) would otherwise hide the complete
    folder a linked root provides. When no root satisfies the requirement,
    falls back to the first existing folder (the pre-parameter behavior)
    so callers still get their historical best-effort answer.
    """
    searched_locations = []
    if os.path.isabs(folder_name):
        if os.path.isdir(folder_name): return folder_name
        searched_locations.append(folder_name)
    else:
        first_existing = None
        for folder in _checkpoints_paths:
            path = os.path.join(folder, folder_name)
            if os.path.isdir(path):
                if not required_files:
                    return path
                if all(os.path.isfile(os.path.join(path, f)) for f in required_files):
                    return path
                if first_existing is None:
                    first_existing = path
            searched_locations.append(os.path.abspath(path))
        if first_existing is not None:
            return first_existing
    if error_if_none: raise Exception(f"Unable to locate folder '{folder_name}', tried {searched_locations}")
    return None


def locate_file(file_name, create_path_if_none = False, error_if_none = True, extra_paths = None):
    if file_name.startswith("http"):
        file_name = os.path.basename(file_name)
    searched_locations = []
    if os.path.isabs(file_name):
        if os.path.isfile(file_name): return file_name
        searched_locations.append(file_name)
    else:
        for folder in _checkpoints_paths + ([] if extra_paths is None else extra_paths):
            path = os.path.join(folder, file_name)
            if os.path.isfile(path):
                return path
            searched_locations.append(os.path.abspath(path))
    
    if create_path_if_none:
        return get_download_location(file_name)
    if error_if_none: raise Exception(f"Unable to locate file '{file_name}', tried {searched_locations}")
    return None

