"""
In-memory search index for the media gallery.

Indexes prompts plus useful generation details from .meta.json sidecars: model
and workflow aliases, resolution, active LoRAs, and acceleration settings. Built
lazily on first search, then updated incrementally as new files appear. Typical
build time: <1s for 5000 files.

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
            for key in (
                "prompt",
                "_tts_original_prompt",
                "_h3_original_prompt",
                "negative_prompt",
            ):
                self._append_searchable(searchable_parts, params.get(key))

            resolution = str(params.get("resolution") or "").strip()
            self._append_searchable(searchable_parts, resolution)
            self._append_resolution_aliases(searchable_parts, resolution)

            model = str(params.get("model_type") or "").strip()
            self._append_searchable(searchable_parts, model)
            self._append_model_aliases(searchable_parts, model)

            # Effective prompts used by each multi-window renderer. Older
            # sidecars used ``window_prompts``; H3 and LTX now retain their
            # model-specific compiled arrays beside the original concept.
            for key in ("window_prompts", "h3_window_prompts", "ltx_window_prompts"):
                self._append_searchable(searchable_parts, params.get(key))

            active_loras = params.get("activated_loras") or []
            if isinstance(active_loras, str):
                active_loras = [active_loras]
            if isinstance(active_loras, (list, tuple)):
                for lora in active_loras:
                    self._append_searchable(searchable_parts, lora)

            # Acceleration labels are deliberately conditional. Maestro stores
            # the default H3 Turbo preset even while Turbo is disabled, so
            # indexing that field unconditionally would make a search for
            # "turbo" return ordinary full-step generations.
            turbo_enabled = params.get("minimax_h3_turbo_mode") is True
            lora_text = " ".join(str(item) for item in active_loras).lower()
            turbo_preset = str(params.get("minimax_h3_turbo_preset") or "")
            if turbo_enabled or "turbo" in lora_text or "acc-" in lora_text:
                searchable_parts.append("turbo accelerated acceleration")
                self._append_searchable(searchable_parts, turbo_preset)
            if "pdd" in turbo_preset.lower() or "pdd" in lora_text:
                searchable_parts.append("pdd turbo acceleration")

            if str(params.get("skip_steps_cache_type") or "").lower() == "first_block":
                searchable_parts.append("first block cache first_block")

            override_attention = str(params.get("override_attention") or "").lower()
            if override_attention == "sol":
                searchable_parts.append("sol engine attention optimization")

            # A few high-value scalar details make exact searches such as
            # "40 steps" and "seed 1234" useful without indexing every private
            # implementation field in the sidecar.
            steps = params.get("num_inference_steps")
            if steps is not None:
                searchable_parts.append(f"{steps} steps")
            seed = params.get("seed")
            if seed is not None:
                searchable_parts.append(f"seed {seed}")

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
    def _append_searchable(parts: list[str], value) -> None:
        """Append scalar or nested textual metadata without indexing nulls."""
        if value is None or value == "":
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                SearchIndex._append_searchable(parts, item)
            return
        if isinstance(value, dict):
            for item in value.values():
                SearchIndex._append_searchable(parts, item)
            return
        parts.append(str(value))

    @staticmethod
    def _append_model_aliases(parts: list[str], model: str) -> None:
        """Add the human-facing names users are likely to type in Search."""
        slug = model.lower()
        if "minimax_h3" in slug:
            parts.append("minimax h3")
            if "ref2va" in slug:
                parts.append("omni reference ref2va")
            else:
                parts.append("first last fl2va")
            if "pruned" in slug:
                parts.append("pruned")
            else:
                parts.append("full")
        elif "ltx2_5" in slug or "ltx-2.5" in slug:
            parts.append("ltx 2.5")
        elif "ltx2" in slug:
            parts.append("ltx 2.3")

    @staticmethod
    def _append_resolution_aliases(parts: list[str], resolution: str) -> None:
        """Index exact dimensions plus Maestro's familiar preset labels."""
        import re

        match = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*", resolution)
        if not match:
            return
        width, height = (int(match.group(1)), int(match.group(2)))
        short_edge = min(width, height)
        # H3/LTX latent-compatible dimensions are close to, but not always
        # identical to, the UI preset name (e.g. 1280x704 is shown as 720p).
        presets = ((512, "480p"), (560, "540p"), (720, "720p"), (800, "768p"), (1120, "1080p"))
        label = min(presets, key=lambda item: abs(short_edge - item[0]))[1]
        parts.extend((label, f"{width}p", f"{height}p"))

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
