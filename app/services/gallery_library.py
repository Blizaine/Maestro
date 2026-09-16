"""Bounded, folder-aware gallery queries without loading media into memory."""

import base64
import hashlib
import json
import os
import re
import threading
from urllib.parse import quote

from .search_index import SearchIndex

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".gif"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS | {".png", ".jpg", ".jpeg", ".webp"}
_SEED_GROUP = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}h\d{2}m\d{2}s_(seed\d+_.+)\.(mp4|webm|mkv)$", re.I)


class GalleryLibrary:
    def __init__(self):
        self._lock = threading.RLock()
        # Cache compact searchable text and listing fields, never full prompts,
        # images or decoded media. Stat signatures also catch late sidecars.
        self._cache = {}

    def _metadata(self, path):
        sidecar = os.path.splitext(path)[0] + ".meta.json"
        try:
            stat = os.stat(sidecar)
            stamp = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            stamp = None
        previous = self._cache.get(path)
        if previous and previous[0] == stamp:
            return previous[1]
        meta = {}
        if stamp is not None:
            try:
                with open(sidecar, encoding="utf-8") as stream:
                    meta = json.load(stream)
                if not isinstance(meta, dict):
                    meta = {}
            except (OSError, ValueError):
                # A writer may still be publishing this sidecar. Retry next scan.
                stamp = None
        params = meta.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        mci = params.get("multi_clip_info")
        if isinstance(mci, dict):
            try:
                mci = dict(mci, index=int(mci.get("index", 0)), total=int(mci.get("total", 0)))
            except (ValueError, TypeError):
                mci = None
        compact = {
            "mode": meta.get("generation_mode"),
            "edit_sub_mode": params.get("edit_sub_mode"),
            "multi_clip_info": mci,
            "metadata_ready": stamp is not None,
            "metadata_updated_at": stamp[0] / 1e9 if stamp else None,
            "tokens": SearchIndex.searchable_tokens(os.path.basename(path), meta),
        }
        self._cache[path] = (stamp, compact)
        return compact

    def list(self, folders, *, favorites=None, limit=100, offset=0, search="",
             media_filter="all", favorites_only=False, multiclip_only=False, cursor=""):
        """Folders are validated (workspace, absolute directory) pairs.

        Cursor pagination uses the last item's sort key, so new files at the
        front and deletes between requests cannot skip older matches. A scope
        digest prevents accidental reuse after switching filters or folders.
        """
        if media_filter not in {"all", "images", "videos", "audio", "avatars", "favorites", "multiclip"}:
            raise ValueError("Invalid media filter")
        favorites_only = favorites_only or media_filter == "favorites"
        multiclip_only = multiclip_only or media_filter == "multiclip"
        scope = hashlib.sha256(json.dumps([folders, search, media_filter, favorites_only, multiclip_only],
                                          sort_keys=True).encode()).hexdigest()[:20]
        after = None
        if cursor:
            try:
                if len(cursor) > 8192:
                    raise ValueError()
                decoded = json.loads(base64.urlsafe_b64decode(cursor.encode()))
                if decoded[0] != scope or len(decoded[1]) != 3:
                    raise ValueError()
                after = (float(decoded[1][0]), str(decoded[1][1]), str(decoded[1][2]))
            except (ValueError, TypeError, IndexError, KeyError):
                raise ValueError("Invalid gallery cursor; refresh the listing") from None
        query_tokens = set(SearchIndex._tokenize(search))
        results = []
        with self._lock:
            scanned = set()
            roots = {os.path.realpath(directory) for _, directory in folders}
            for workspace, directory in folders:
                directory = os.path.realpath(directory)
                folder_items = []
                groups = {}
                favs = favorites(workspace) if favorites and workspace != "__uploads__" else set()
                try:
                    with os.scandir(directory) as entries:
                        for entry in entries:
                            ext = os.path.splitext(entry.name)[1].lower()
                            if entry.name.startswith(".") or ext not in MEDIA_EXTENSIONS:
                                continue
                            try:
                                if not entry.is_file(follow_symlinks=False):
                                    continue
                                stat = entry.stat(follow_symlinks=False)
                            except OSError:
                                continue
                            path = entry.path
                            scanned.add(path)
                            meta = self._metadata(path)
                            mci = meta.get("multi_clip_info")
                            if isinstance(mci, dict) and mci.get("group_id"):
                                gid = str(mci["group_id"])
                                index, total = int(mci.get("index", 0)), int(mci.get("total", 0))
                                previous = groups.get(gid, (-1, total))
                                groups[gid] = (max(previous[0], index), total)
                            folder_items.append((entry.name, path, ext, stat, meta))
                except OSError:
                    continue
                visible = []
                for name, path, ext, stat, meta in folder_items:
                    mci = meta.get("multi_clip_info")
                    if isinstance(mci, dict) and mci.get("group_id"):
                        highest, total = groups[str(mci["group_id"])]
                        if highest >= total - 1 or int(mci.get("index", 0)) < highest:
                            continue
                    kind = "video" if ext in VIDEO_EXTENSIONS else "audio" if ext in AUDIO_EXTENSIONS else "image"
                    item = {key: meta[key] for key in ("mode", "edit_sub_mode", "metadata_ready", "metadata_updated_at")}
                    item.update(name=name, workspace=workspace, path=path,
                                id=f"{workspace}/{name}", type=kind, size=stat.st_size,
                                created_at=stat.st_mtime, favorite=name in favs,
                                url=f"/api/v1/file/{quote(name, safe='')}?workspace={quote(workspace, safe='')}")
                    visible.append((item, meta["tokens"]))
                multi_names = {item["name"] for item, _ in visible
                               if item["type"] == "video" and "multiclip" in item["name"].lower()}
                seed_groups = {}
                if multiclip_only:
                    for item, _ in visible:
                        match = _SEED_GROUP.match(item["name"])
                        if match and "_tmp." not in item["name"]:
                            seed_groups.setdefault(match[1], []).append(item)
                    for members in seed_groups.values():
                        if len(members) > 1 and max(m["created_at"] for m in members) - min(m["created_at"] for m in members) < 7200:
                            multi_names.add(max(members, key=lambda m: (m["size"], m["created_at"]))["name"])
                for item, tokens in visible:
                    if favorites_only and not item["favorite"]:
                        continue
                    if multiclip_only and item["name"] not in multi_names:
                        continue
                    if media_filter in ("images", "videos", "audio") and item["type"] != {"images": "image", "videos": "video", "audio": "audio"}[media_filter]:
                        continue
                    if media_filter == "avatars" and not (item["edit_sub_mode"] or item["mode"] == "avatar"):
                        continue
                    if search.strip() and not (query_tokens and query_tokens <= tokens) and search.casefold() not in item["name"].casefold():
                        continue
                    results.append(item)
            # Keep other folder caches warm; remove stale files in scanned roots.
            for path in list(self._cache):
                if os.path.dirname(path) in roots and path not in scanned:
                    del self._cache[path]
        key = lambda item: (item["created_at"], item["workspace"], item["name"])
        results.sort(key=key, reverse=True)
        total = len(results)
        if after is not None:
            results = [item for item in results if key(item) < after]
        elif offset > 0:
            results = results[offset:]
        page = results[:min(limit, 500)] if limit > 0 else results
        next_cursor = None
        if page and len(page) < len(results):
            next_cursor = base64.urlsafe_b64encode(json.dumps([scope, key(page[-1])]).encode()).decode()
        return {"outputs": page, "total": total, "next_cursor": next_cursor}


gallery_library = GalleryLibrary()
