"""Regression tests for Maestro's cached llama.cpp runtime."""

from __future__ import annotations

import os
import io
import json
import sys
import tempfile
import unittest
import zipfile
from unittest import mock


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP = os.path.join(_ROOT, "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from services import llm_service


class TestLlamaBuildMetadata(unittest.TestCase):
    def test_positive_build_is_parsed(self):
        self.assertEqual(
            llm_service._positive_llama_build(
                "version: 10488 (012345678)\nbuilt with Clang"
            ),
            10488,
        )

    def test_prefixed_build_is_parsed(self):
        self.assertEqual(
            llm_service._positive_llama_build("version: b10488"),
            10488,
        )

    def test_zero_build_means_unknown(self):
        self.assertIsNone(
            llm_service._positive_llama_build("version: 0 (unknown)")
        )

    def test_release_tag_build(self):
        self.assertEqual(llm_service._llama_release_build("b9632"), 9632)
        self.assertIsNone(llm_service._llama_release_build("latest"))

    def test_semver_pointer_is_not_treated_as_a_binary_release(self):
        release = {
            "tag_name": "v0.2.0",
            "assets": [
                {
                    "name": "nightly-tag.txt",
                    "browser_download_url": "https://example.test/nightly-tag.txt",
                }
            ],
        }
        self.assertFalse(
            llm_service._llama_release_has_assets(
                release,
                [("llama-", "bin-win-cuda-12.4-x64.zip")],
            )
        )
        self.assertEqual(
            llm_service._llama_nightly_pointer_url(release),
            "https://example.test/nightly-tag.txt",
        )

    def test_binary_nightly_requires_every_requested_asset(self):
        release = {
            "tag_name": "b10566",
            "assets": [
                {
                    "name": "llama-b10566-bin-win-cuda-12.4-x64.zip",
                    "browser_download_url": "https://example.test/llama.zip",
                },
                {
                    "name": "cudart-llama-bin-win-cuda-12.4-x64.zip",
                    "browser_download_url": "https://example.test/cudart.zip",
                },
            ],
        }
        self.assertTrue(
            llm_service._llama_release_has_assets(
                release,
                [
                    ("llama-", "bin-win-cuda-12.4-x64.zip"),
                    ("cudart-", "bin-win-cuda-12.4-x64.zip"),
                ],
            )
        )
        self.assertFalse(
            llm_service._llama_release_has_assets(
                {**release, "assets": release["assets"][:1]},
                [
                    ("llama-", "bin-win-cuda-12.4-x64.zip"),
                    ("cudart-", "bin-win-cuda-12.4-x64.zip"),
                ],
            )
        )


class TestLlamaRuntimeReceipt(unittest.TestCase):
    def test_receipt_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            llm_service._write_llama_runtime_receipt(
                directory,
                tag="b10488",
                build=10488,
            )
            receipt = llm_service._read_llama_runtime_receipt(directory)
            self.assertEqual(receipt["schema_version"], 1)
            self.assertEqual(receipt["release_tag"], "b10488")
            self.assertEqual(receipt["build"], 10488)

    def test_corrupt_receipt_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(
                llm_service._llama_runtime_receipt_path(directory),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write("not-json")
            self.assertEqual(
                llm_service._read_llama_runtime_receipt(directory),
                {},
            )


class TestLlamaRuntimeIdempotency(unittest.TestCase):
    @staticmethod
    def _write_complete_runtime(directory: str) -> None:
        executable = "llama-server.exe" if sys.platform.startswith("win") else "llama-server"
        with open(os.path.join(directory, executable), "wb") as handle:
            handle.write(b"installed")
        if sys.platform.startswith("win"):
            for filename in llm_service._WINDOWS_LLAMA_CUDA_FILES:
                with open(os.path.join(directory, filename), "wb") as handle:
                    handle.write(b"installed")

    def test_zero_or_unreadable_build_does_not_redownload(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_complete_runtime(directory)
            with (
                mock.patch.object(
                    llm_service,
                    "_llama_server_build",
                    return_value=None,
                ),
                mock.patch(
                    "urllib.request.urlopen",
                    side_effect=AssertionError("network should not be used"),
                ),
            ):
                llm_service._ensure_llama_server(directory)

    def test_receipt_keeps_metadata_less_release_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_complete_runtime(directory)
            llm_service._write_llama_runtime_receipt(
                directory,
                tag="b10488",
                build=10488,
            )
            with (
                mock.patch.object(
                    llm_service,
                    "_llama_server_build",
                    return_value=None,
                ),
                mock.patch(
                    "urllib.request.urlopen",
                    side_effect=AssertionError("network should not be used"),
                ),
            ):
                llm_service._ensure_llama_server(directory)


class TestLlamaRuntimeReleaseResolution(unittest.TestCase):
    class _Response(io.BytesIO):
        def __init__(self, payload: bytes):
            super().__init__(payload)
            self.headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.close()
            return False

    @staticmethod
    def _json_response(payload: dict):
        return TestLlamaRuntimeReleaseResolution._Response(
            json.dumps(payload).encode("utf-8")
        )

    @staticmethod
    def _zip_response(files: dict[str, bytes]):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, payload in files.items():
                archive.writestr(name, payload)
        return TestLlamaRuntimeReleaseResolution._Response(buffer.getvalue())

    def test_stable_pointer_resolves_the_binary_nightly(self):
        latest = {
            "tag_name": "v0.2.0",
            "assets": [
                {
                    "name": "nightly-tag.txt",
                    "browser_download_url": "https://example.test/nightly-tag.txt",
                }
            ],
        }
        nightly = {
            "tag_name": "b10566",
            "assets": [
                {
                    "name": "llama-b10566-bin-win-cuda-12.4-x64.zip",
                    "browser_download_url": "https://example.test/llama.zip",
                },
                {
                    "name": "cudart-llama-bin-win-cuda-12.4-x64.zip",
                    "browser_download_url": "https://example.test/cudart.zip",
                },
            ],
        }
        requested_urls = []
        responses = iter(
            [
                self._json_response(latest),
                self._Response(b"b10566\n"),
                self._json_response(nightly),
                self._zip_response({"build/bin/llama-server.exe": b"server"}),
                self._zip_response(
                    {
                        f"build/bin/{name}": b"cuda"
                        for name in llm_service._WINDOWS_LLAMA_CUDA_FILES
                    }
                ),
            ]
        )

        def fake_urlopen(request, timeout=None):
            requested_urls.append(getattr(request, "full_url", str(request)))
            return next(responses)

        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(sys, "platform", "win32"),
                mock.patch.object(
                    llm_service,
                    "_llama_server_build",
                    return_value=10566,
                ),
                mock.patch("urllib.request.urlopen", side_effect=fake_urlopen),
            ):
                llm_service._ensure_llama_server(directory)

            self.assertTrue(os.path.isfile(os.path.join(directory, "llama-server.exe")))
            for filename in llm_service._WINDOWS_LLAMA_CUDA_FILES:
                self.assertTrue(os.path.isfile(os.path.join(directory, filename)))
            receipt = llm_service._read_llama_runtime_receipt(directory)
            self.assertEqual(receipt["release_tag"], "b10566")
            self.assertFalse(
                any("v0.2.0/llama-v0.2.0" in url for url in requested_urls)
            )

if __name__ == "__main__":
    unittest.main(verbosity=2)
