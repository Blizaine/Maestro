"""
In-memory search index for the media gallery.

Indexes prompt text, model type, generation mode, and filename from .meta.json
sidecars. Built lazily on first search, then updated incrementally as new files
appear. Typical build time: <1s for 5000 files.

The index is a simple inverted token → set-of-filenames map. Searches split the
query into tokens and intersect their result sets (AND logic). This gives instant
results for multi-word queries across thousands of files.
"""

import os
import json
import time
import threading
from typing import Optional


class SearchIndex:
    def __init__(self):
        self._index: dict[str, set[str]] = {}
        self._indexed_files: set[str] = set()
        self._workspace: str = ""
        self._lock = threading.Lock()
        self._built = False
        self._last_build_time: float = 0

    def search(self, query: str, workspace_dir: str) -> set[str]:
        """Search for files matching ALL tokens in the query.

        Returns a set of matching filenames (not full paths).
        """
        if not query.strip():
            return set()

        with self._lock:
            if workspace_dir != self._workspace or not self._built:
                self._full_rebuild(workspace_dir)
            else:
                self._incremental_update(workspace_dir)

        tokens = self._tokenize(query)
        if not tokens:
            return set()

        with self._lock:
            result: Optional[set[str]] = None
            for token in tokens:
                matches = self._index.get(token, set())
                if result is None:
                    result = set(matches)
                else:
                    result &= matches
                if not result:
                    return set()
            return result or set()

    def invalidate(self):
        """Force a full rebuild on next search."""
        with self._lock:
            self._built = False

    def remove_file(self, filename: str):
        """Remove a file from the index (e.g., after deletion)."""
        with self._lock:
            self._indexed_files.discard(filename)
            for token_set in self._index.values():
                token_set.discard(filename)

    def _full_rebuild(self, workspace_dir: str):
        """Build the entire index from scratch."""
        t0 = time.time()
        self._index.clear()
        self._indexed_files.clear()
        self._workspace = workspace_dir

        if not os.path.isdir(workspace_dir):
            self._built = True
            return

        count = 0
        for name in os.listdir(workspace_dir):
            if not name.endswith(".meta.json"):
                continue
            media_name = name[:-10]  # strip .meta.json
            meta_path = os.path.join(workspace_dir, name)
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._index_file(media_name, meta)
                count += 1
            except Exception:
                pass

        self._built = True
        self._last_build_time = time.time()
        elapsed = time.time() - t0
        print(f"[SearchIndex] Built index: {count} files in {elapsed:.2f}s, {len(self._index)} tokens")

    def _incremental_update(self, workspace_dir: str):
        """Index any new files that appeared since last build."""
        if not os.path.isdir(workspace_dir):
            return

        new_count = 0
        for name in os.listdir(workspace_dir):
            if not name.endswith(".meta.json"):
                continue
            media_name = name[:-10]
            if media_name in self._indexed_files:
                continue
            meta_path = os.path.join(workspace_dir, name)
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._index_file(media_name, meta)
                new_count += 1
            except Exception:
                pass

        if new_count > 0:
            print(f"[SearchIndex] Incremental update: +{new_count} files")

    def _index_file(self, media_name: str, meta: dict):
        """Add a single file's searchable content to the index."""
        self._indexed_files.add(media_name)

        # Collect all searchable text
        searchable_parts = [media_name]

        params = meta.get("params", {})
        if isinstance(params, dict):
            prompt = params.get("prompt", "")
            if prompt:
                searchable_parts.append(str(prompt))
            neg = params.get("negative_prompt", "")
            if neg:
                searchable_parts.append(str(neg))
            model = params.get("model_type", "")
            if model:
                searchable_parts.append(str(model))
            # Window prompts (multi-window scenes)
            for wp in params.get("window_prompts", []) or []:
                if wp:
                    searchable_parts.append(str(wp))

        mode = meta.get("generation_mode", "")
        if mode:
            searchable_parts.append(str(mode))

        # Tokenize all parts and add to inverted index
        full_text = " ".join(searchable_parts)
        for token in self._tokenize(full_text):
            if token not in self._index:
                self._index[token] = set()
            self._index[token].add(media_name)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into lowercase tokens for indexing/querying.

        Keeps tokens >= 2 chars. Splits on whitespace and common punctuation.
        """
        import re
        tokens = re.split(r'[\s,._\-/\\()\[\]{}:;!?"+]+', text.lower())
        return [t for t in tokens if len(t) >= 2]


# Singleton instance
_search_index = SearchIndex()


def get_search_index() -> SearchIndex:
    return _search_index
