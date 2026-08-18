"""Regression tests for Maestro's cached llama.cpp runtime."""

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
