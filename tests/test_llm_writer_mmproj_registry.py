"""Contracts for the registered Director/prompt-writer vision projectors."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP = os.path.join(_ROOT, "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from services import llm_service


PROJECTORS = {
    "mradermacher/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-i1-GGUF": (
        "mradermacher/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-GGUF",
        "Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated.mmproj-Q8_0.gguf",
        "mmproj-Q8_0.gguf",
        0.624,
    ),
    "Nesuwka/gemma-4-E2B-it-heretic-ara-Q4_K_M-GGUF": (
        "ggml-org/gemma-4-E2B-it-GGUF",
        "mmproj-gemma-4-E2B-it-BF16.gguf",
        "mmproj-gemma-4-e2b-it-f16.gguf",
        0.987,
    ),
    "paperscarecrow/Gemma-4-31B-it-abliterated-gguf": (
        "ggml-org/gemma-4-31B-it-GGUF",
        "mmproj-gemma-4-31B-it-BF16.gguf",
        "mmproj-gemma-4-31B-it-f16.gguf",
        1.2,
    ),
}


class TestWriterMMProjRegistry(unittest.TestCase):
    def test_registered_projectors_use_verified_model_specific_repositories(self):
        for repo_id, (projector_repo, filename, alias, size_gb) in PROJECTORS.items():
            with self.subTest(repo_id=repo_id):
                entry = llm_service.MODEL_REGISTRY[repo_id]
                self.assertEqual(entry["mmproj_repo"], projector_repo)
                self.assertEqual(entry["mmproj_file"], filename)
                self.assertEqual(entry["mmproj_cache_aliases"], [alias])
                self.assertAlmostEqual(entry["mmproj_gb"], size_gb, places=3)

    def test_existing_legacy_cache_alias_is_reused_without_download(self):
        for repo_id, (projector_repo, _filename, alias, _size_gb) in PROJECTORS.items():
            with self.subTest(repo_id=repo_id), tempfile.TemporaryDirectory() as cache_dir:
                cached_path = os.path.join(cache_dir, alias)
                with open(cached_path, "wb") as cached_projector:
                    cached_projector.write(b"cached GGUF placeholder")

                with mock.patch.object(llm_service, "_download_gguf") as download:
                    resolved_path, resolved_repo = llm_service._resolve_mmproj(
                        repo_id,
                        llm_service.MODEL_REGISTRY[repo_id],
                        cache_dir,
                    )

                self.assertEqual(resolved_path, os.path.normpath(cached_path))
                self.assertEqual(resolved_repo, projector_repo)
                download.assert_not_called()

    def test_cache_miss_download_uses_the_registered_repository_and_filename(self):
        for repo_id, (projector_repo, filename, _alias, _size_gb) in PROJECTORS.items():
            with self.subTest(repo_id=repo_id), tempfile.TemporaryDirectory() as cache_dir:
                downloaded_path = os.path.join(cache_dir, filename)
                with mock.patch.object(
                    llm_service, "_download_gguf", return_value=downloaded_path
                ) as download:
                    resolved_path, resolved_repo = llm_service._resolve_mmproj(
                        repo_id,
                        llm_service.MODEL_REGISTRY[repo_id],
                        cache_dir,
                    )

                self.assertEqual(resolved_path, os.path.normpath(downloaded_path))
                self.assertEqual(resolved_repo, projector_repo)
                download.assert_called_once_with(projector_repo, filename, cache_dir)

    def test_empty_legacy_alias_does_not_hide_the_replacement_download(self):
        for repo_id, (projector_repo, filename, alias, _size_gb) in PROJECTORS.items():
            with self.subTest(repo_id=repo_id), tempfile.TemporaryDirectory() as cache_dir:
                with open(os.path.join(cache_dir, alias), "wb"):
                    pass
                downloaded_path = os.path.join(cache_dir, filename)
                with mock.patch.object(llm_service, "_download_gguf", return_value=downloaded_path) as download:
                    resolved_path, _ = llm_service._resolve_mmproj(
                        repo_id, llm_service.MODEL_REGISTRY[repo_id], cache_dir
                    )
                self.assertEqual(resolved_path, os.path.normpath(downloaded_path))
                download.assert_called_once_with(projector_repo, filename, cache_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
