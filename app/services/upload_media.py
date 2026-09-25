"""Constrained deletion of gallery uploads, without touching output folders."""
from __future__ import annotations

import os
from urllib.parse import unquote, urlsplit

MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff",
                    ".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v",
                    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


def upload_path(root, name):
    if (not isinstance(name, str) or not name or name.startswith(".")
            or any(char in name for char in "/\\:\x00")
            or os.path.splitext(name)[1].lower() not in MEDIA_EXTENSIONS):
        raise ValueError("Invalid upload filename")
    root = os.path.realpath(root)
    if os.path.islink(os.path.join(root, name)):
        raise ValueError("Linked uploads cannot be deleted here")
    path = os.path.realpath(os.path.join(root, name))
    if os.path.normcase(os.path.dirname(path)) != os.path.normcase(root):
        raise ValueError("Upload must be inside the uploads folder")
    return path


def references_upload(payload, path):
    """Recognize stored filenames, full paths, nested refs and upload URLs."""
    if isinstance(payload, dict):
        return any(references_upload(value, path) for value in list(payload.values()))
    if isinstance(payload, (list, tuple)):
        return any(references_upload(value, path) for value in payload)
    if not isinstance(payload, str):
        return False
    name = os.path.basename(path)
    value = unquote(payload).replace("\\", "/")
    if value.casefold() in {name.casefold(), ("uploads/" + name).casefold()}:
        return True
    if value.startswith(("http://", "https://", "/api/")):
        parsed = urlsplit(value)
        return (parsed.path.casefold() == ("/api/v1/uploads/" + name).casefold()
                or (parsed.path.casefold() == ("/api/v1/file/" + name).casefold()
                    and "workspace=__uploads__" in parsed.query))
    if os.path.isabs(value):
        return os.path.normcase(os.path.realpath(value)) == os.path.normcase(path)
    return False


def delete_upload_file(root, name, *, safe_delete):
    path = upload_path(root, name)
    if not os.path.exists(path):
        return {"deleted": name}
    if not os.path.isfile(path):
        raise ValueError("Upload is not a media file")
    result = safe_delete(path)
    if not result.get("deleted") and result.get("reason") != "not_found":
        raise PermissionError("This upload is still in use. Close its playback and try again.")
    # Do not follow a sidecar link outside uploads, even when media is valid.
    sidecar = os.path.splitext(path)[0] + ".meta.json"
    if (os.path.isfile(sidecar) and not os.path.islink(sidecar)
            and os.path.dirname(os.path.realpath(sidecar)) == os.path.realpath(root)):
        safe_delete(sidecar)
    return {"deleted": name, "deferred": bool(result.get("deferred"))}
