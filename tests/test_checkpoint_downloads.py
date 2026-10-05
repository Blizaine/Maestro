"""CPU-only tests for named checkpoint source selection and downloads."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from app.shared.checkpoint_downloads import (
    CheckpointDownloadError,
    download_named_checkpoint,
    find_named_checkpoint_source,
)


_ROOT = Path(__file__).resolve().parents[1]


def _safetensors_bytes(payload: bytes = b"test weights") -> bytes:
    header = {
        "weight": {
            "dtype": "U8",
            "shape": [len(payload)],
            "data_offsets": [0, len(payload)],
        }
    }
    encoded_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return len(encoded_header).to_bytes(8, "little") + encoded_header + payload


class FakeResponse:
    def __init__(self, body: bytes = b"", status_code: int = 200):
        self.body = body
        self.status_code = status_code
        self.closed = False

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self.body), max(1, chunk_size // 4)):
            yield self.body[start : start + max(1, chunk_size // 4)]

    def close(self):
        self.closed = True


def _source(url: str, content: bytes, **overrides) -> dict:
    source = {
        "url": url,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }
    source.update(overrides)
    return source


class TestCheckpointDownloads(unittest.TestCase):
    def test_named_download_reaches_maestros_live_download_banner(self):
        import tqdm
        from app.services import safe_download

        content = _safetensors_bytes(b"x" * (1024 * 1024))
        states = []
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(tqdm.tqdm, "__init__", tqdm.tqdm.__init__), \
                patch.object(tqdm.tqdm, "update", tqdm.tqdm.update), \
                patch.object(tqdm.tqdm, "close", tqdm.tqdm.close), \
                patch.object(tqdm.tqdm, "_maestro_progress_patched", False, create=True):
            safe_download._install_tqdm_hook()
            with patch("requests.get", return_value=FakeResponse(content)):
                download_named_checkpoint(
                    _source("https://civitai.com/api/download/models/101?fileId=11", content),
                    Path(directory) / "banner-checkpoint.safetensors",
                    progress_hook=lambda *_: states.extend(safe_download.get_active_downloads()),
                )
            tracked = [state for state in states if state["filename"] == "banner-checkpoint.safetensors"]
            self.assertTrue(tracked)
            self.assertEqual(tracked[-1]["downloaded_bytes"], len(content))
            self.assertEqual(tracked[-1]["total_bytes"], len(content))
            self.assertNotIn("token", tracked[-1]["file_id"])
            self.assertEqual(safe_download.get_active_downloads(), [])

    def test_named_source_selection_is_exact_and_variants_are_independent(self):
        high_name = "dasiwa_high.safetensors"
        low_name = "dasiwa_low.safetensors"
        high_source = {"url": "https://civitai.com/api/download/models/101?fileId=11"}
        low_source = {"url": "https://civitai.com/api/download/models/202?fileId=22"}
        high_model = {
            "URLs": [high_name],
            "download_sources": {high_name: high_source},
        }
        low_model = {
            "URLs": [low_name],
            "download_sources": {low_name: low_source},
        }

        self.assertIs(find_named_checkpoint_source(high_model, high_name), high_source)
        self.assertIs(find_named_checkpoint_source(low_model, low_name), low_source)
        self.assertIsNone(find_named_checkpoint_source(high_model, low_name))
        self.assertIsNone(
            find_named_checkpoint_source(
                high_model, "https://civitai.com/api/download/models/101?fileId=11",
            )
        )
        self.assertIsNone(find_named_checkpoint_source(low_model, "../dasiwa_low.safetensors"))

    def test_download_uses_named_target_auth_headers_and_progress_hook(self):
        content = _safetensors_bytes()
        response = FakeResponse(content)
        progress = []
        with tempfile.TemporaryDirectory() as directory:
            target_dir = Path(directory) / "nested" / "checkpoint-dir"
            target = target_dir / "dasiwa_high.safetensors"
            source = _source(
                "https://civitai.com/api/download/models/101?fileId=11",
                content,
            )
            with patch("requests.get", return_value=response) as get:
                result = download_named_checkpoint(
                    source,
                    target,
                    civitai_api_key="secret-api-key",
                    progress_hook=lambda *args: progress.append(args),
                )

            self.assertEqual(result, str(target))
            self.assertEqual(target.read_bytes(), content)
            self.assertEqual([path.name for path in target_dir.iterdir()], [target.name])
            self.assertTrue(response.closed)
            request_url = get.call_args.args[0]
            query = parse_qs(urlsplit(request_url).query)
            self.assertEqual(query["fileId"], ["11"])
            self.assertEqual(query["token"], ["secret-api-key"])
            headers = get.call_args.kwargs["headers"]
            self.assertEqual(headers["Authorization"], "Bearer secret-api-key")
            self.assertEqual(headers["Accept-Encoding"], "gzip, deflate")
            self.assertIn("Chrome/121.0.0.0", headers["User-Agent"])
            self.assertTrue(progress)
            self.assertEqual(progress[-1], (1, len(content), len(content)))

    def test_existing_token_is_not_duplicated_or_overridden(self):
        content = _safetensors_bytes()
        response = FakeResponse(content)
        with tempfile.TemporaryDirectory() as directory:
            source = _source(
                "https://civitai.red/api/download/models/101?token=source-token&fileId=11",
                content,
            )
            with patch("requests.get", return_value=response) as get:
                download_named_checkpoint(
                    source,
                    Path(directory) / "checkpoint.safetensors",
                    civitai_api_key="configured-key",
                )
            request_url = get.call_args.args[0]
            self.assertEqual(parse_qs(urlsplit(request_url).query)["token"], ["source-token"])
            self.assertNotIn("Authorization", get.call_args.kwargs["headers"])

    def test_configured_key_is_not_sent_to_other_hosts(self):
        content = _safetensors_bytes()
        response = FakeResponse(content)
        with tempfile.TemporaryDirectory() as directory:
            source = _source("https://downloads.example.test/checkpoint", content)
            with patch("requests.get", return_value=response) as get:
                download_named_checkpoint(
                    source,
                    Path(directory) / "checkpoint.safetensors",
                    civitai_api_key="secret-api-key",
                )
            request_url = get.call_args.args[0]
            self.assertNotIn("secret-api-key", request_url)
            self.assertNotIn("token", parse_qs(urlsplit(request_url).query))
            self.assertNotIn("Authorization", get.call_args.kwargs["headers"])

    def test_failed_digest_preserves_existing_target_and_cleans_temp(self):
        content = _safetensors_bytes()
        response = FakeResponse(content)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint.safetensors"
            target.write_bytes(b"existing checkpoint")
            source = _source(
                "https://civitai.com/api/download/models/101?fileId=11",
                content,
                sha256="0" * 64,
            )
            with patch("requests.get", return_value=response):
                with self.assertRaisesRegex(CheckpointDownloadError, "SHA-256"):
                    download_named_checkpoint(source, target)
            self.assertEqual(target.read_bytes(), b"existing checkpoint")
            self.assertEqual([path.name for path in Path(directory).iterdir()], [target.name])

    def test_html_auth_page_is_rejected_even_when_size_and_hash_match(self):
        content = b"<!doctype html><html><title>Access denied</title></html>"
        response = FakeResponse(content)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint.safetensors"
            target.write_bytes(b"existing checkpoint")
            with patch("requests.get", return_value=response):
                with self.assertRaisesRegex(CheckpointDownloadError, "safetensors header"):
                    download_named_checkpoint(
                        _source("https://civitai.com/api/download/models/101", content),
                        target,
                    )
            self.assertEqual(target.read_bytes(), b"existing checkpoint")
            self.assertEqual([path.name for path in Path(directory).iterdir()], [target.name])

    def test_civitai_auth_errors_are_actionable_and_redact_credentials(self):
        response = FakeResponse(b"secret-api-key", status_code=403)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint.safetensors"
            target.write_bytes(b"existing checkpoint")
            source = _source(
                "https://civitai.com/api/download/models/101?fileId=11",
                b"placeholder",
            )
            with patch("requests.get", return_value=response):
                with self.assertRaises(CheckpointDownloadError) as raised:
                    download_named_checkpoint(
                        source,
                        target,
                        civitai_api_key="secret-api-key",
                    )
            message = str(raised.exception)
            self.assertIn("Settings > Services > Civitai API key", message)
            self.assertIn("authorized access to the official creator file", message)
            self.assertNotIn("secret-api-key", message)
            self.assertEqual(target.read_bytes(), b"existing checkpoint")
            self.assertEqual([path.name for path in Path(directory).iterdir()], [target.name])

    def test_missing_key_is_not_injected_and_auth_failure_stays_actionable(self):
        response = FakeResponse(status_code=401)
        with tempfile.TemporaryDirectory() as directory:
            source = _source(
                "https://civitai.com/api/download/models/101?fileId=11",
                b"placeholder",
            )
            with patch("requests.get", return_value=response) as get:
                with self.assertRaisesRegex(CheckpointDownloadError, "Settings > Services"):
                    download_named_checkpoint(
                        source,
                        Path(directory) / "checkpoint.safetensors",
                    )
            request_url = get.call_args.args[0]
            self.assertNotIn("token", parse_qs(urlsplit(request_url).query))
            self.assertNotIn("Authorization", get.call_args.kwargs["headers"])

    def test_request_error_does_not_echo_tokenized_url(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _source(
                "https://civitai.com/api/download/models/101?fileId=11",
                b"placeholder",
            )
            with patch("requests.get", side_effect=RuntimeError("failed with URL token=secret-api-key")):
                with self.assertRaises(CheckpointDownloadError) as raised:
                    download_named_checkpoint(
                        source,
                        Path(directory) / "checkpoint.safetensors",
                        civitai_api_key="secret-api-key",
                    )
            self.assertNotIn("secret-api-key", str(raised.exception))
            self.assertNotIn("token=", str(raised.exception))

    def test_target_filename_cannot_escape_the_passed_directory(self):
        content = _safetensors_bytes()
        with tempfile.TemporaryDirectory() as directory:
            source = _source("https://civitai.com/api/download/models/101", content)
            for target in (
                Path(directory) / ".." / "outside.safetensors",
                Path(directory) / "CON.safetensors",
                Path(directory) / "has?.safetensors",
            ):
                with self.subTest(target=target), patch("requests.get") as get:
                    with self.assertRaises(CheckpointDownloadError):
                        download_named_checkpoint(source, target)
                    get.assert_not_called()

    def test_wgp_download_models_uses_named_source_outside_legacy_cleanup(self):
        wgp_path = _ROOT / "app" / "wgp.py"
        source = wgp_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(wgp_path))
        download_models = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "download_models"
        )
        module = ast.Module(body=[download_models], type_ignores=[])
        ast.fix_missing_locations(module)

        filename = "dasiwa_high.safetensors"
        source_def = {
            "url": "https://civitai.com/api/download/models/101?fileId=11",
            "sha256": "a" * 64,
            "size_bytes": 123,
        }
        model_def = {
            "URLs": [filename],
            "download_sources": {filename: source_def},
        }
        helper_calls = []
        progress_hook = object()

        with tempfile.TemporaryDirectory() as directory:
            target_path = str(Path(directory) / filename)
            shared_module = ModuleType("shared")
            shared_module.__path__ = []
            shared_utils_module = ModuleType("shared.utils")
            shared_utils_module.__path__ = []
            checkpoint_downloads = ModuleType("shared.checkpoint_downloads")
            checkpoint_downloads.find_named_checkpoint_source = find_named_checkpoint_source
            download_utils = ModuleType("shared.utils.download")
            download_utils.create_progress_hook = lambda _filename: progress_hook

            def record_helper(*args, **kwargs):
                helper_calls.append((args, kwargs))

            checkpoint_downloads.download_named_checkpoint = record_helper
            namespace = {
                "os": os,
                "get_base_model_type": lambda _model_type: "test_architecture",
                "get_runtime_model_def": lambda _model_type: model_def,
                "model_types_handlers": {"test_architecture": object()},
                "get_compatible_local_model_filename": lambda *args, **kwargs: None,
                "fl": SimpleNamespace(
                    get_smart_download_location=lambda clean_name, force_path=None: target_path,
                ),
                "server_config": {"services": {"civitai_api_key": "test-api-key"}},
            }
            with patch.dict(
                "sys.modules",
                {
                    "shared": shared_module,
                    "shared.utils": shared_utils_module,
                    "shared.checkpoint_downloads": checkpoint_downloads,
                    "shared.utils.download": download_utils,
                },
            ):
                exec(compile(module, str(wgp_path), "exec"), namespace)
                namespace["download_models"](filename, "dasiwa_high", file_type=2)
                self.assertEqual(len(helper_calls), 1)
                args, kwargs = helper_calls[0]
                self.assertIs(args[0], source_def)
                self.assertEqual(args[1], target_path)
                self.assertEqual(kwargs["civitai_api_key"], "test-api-key")
                self.assertIs(kwargs["progress_hook"], progress_hook)

                Path(target_path).write_bytes(b"existing checkpoint")

                def fail_helper(*args, **kwargs):
                    helper_calls.append((args, kwargs))
                    raise RuntimeError("expected download failure")

                checkpoint_downloads.download_named_checkpoint = fail_helper
                with self.assertRaisesRegex(RuntimeError, "expected download failure"):
                    namespace["download_models"](filename, "dasiwa_high", file_type=2)
                self.assertEqual(Path(target_path).read_bytes(), b"existing checkpoint")

        self.assertEqual(len(helper_calls), 2)


if __name__ == "__main__":
    unittest.main()
