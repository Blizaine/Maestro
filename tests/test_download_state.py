"""Model-free regressions for CivitAI/HuggingFace download state wiring."""
from __future__ import annotations

import ast
import json
import os
from pathlib import PureWindowsPath
import struct
import tempfile
import threading
import time
import unittest
import uuid
import zipfile


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LAUNCH_PATH = os.path.join(_ROOT, "app", "launch.py")


def _parse_launch() -> tuple[ast.Module, str]:
    with open(_LAUNCH_PATH, "r", encoding="utf-8") as handle:
        source = handle.read()
    return ast.parse(source, filename="app/launch.py"), source


def _function(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} not found")


def _source_segment(tree: ast.AST, source: str, name: str) -> str:
    segment = ast.get_source_segment(source, _function(tree, name))
    if segment is None:
        raise AssertionError(f"Could not read source for {name!r}")
    return segment


def _load_helpers(tree: ast.Module, names: tuple[str, ...]) -> dict:
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "os": os,
        "PureWindowsPath": PureWindowsPath,
        "time": time,
        "threading": threading,
        "uuid": uuid,
        "_civitai_downloads": {},
        "_civitai_download_lock": threading.Lock(),
        "_download_target_reservations": {},
        "_lora_guide_scans": {},
        "_lora_guide_scan_lock": threading.Lock(),
    }
    exec(compile(module, "app/launch.py", "exec"), namespace)
    return namespace


class TestDownloadState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree, cls.source = _parse_launch()

    def test_common_record_and_serialization_contract(self):
        helpers = _load_helpers(
            self.tree,
            (
                "_new_download_record",
                "_safe_download_number",
                "_serialize_download_record",
            ),
        )
        record = helpers["_new_download_record"](
            "hf_example",
            "example.safetensors",
            _internal=object(),
        )
        required = {
            "id", "filename", "status", "progress", "bytes_downloaded",
            "bytes_total", "error", "started_at", "completed_at",
        }
        self.assertTrue(required <= record.keys())
        self.assertEqual(record["progress"], 0)
        self.assertIsNotNone(record["started_at"])
        self.assertIsNone(record["completed_at"])

        public = helpers["_serialize_download_record"](
            {
                "progress": 999,
                "bytes_downloaded": "12",
                "bytes_total": object(),
                "error": RuntimeError("network failed"),
                "warnings": "not-a-list",
                "_internal": object(),
            },
            fallback_id="legacy-entry",
        )
        self.assertTrue(required <= public.keys())
        self.assertEqual(public["id"], "legacy-entry")
        self.assertEqual(public["progress"], 100)
        self.assertEqual(public["bytes_downloaded"], 12)
        self.assertEqual(public["bytes_total"], 0)
        self.assertEqual(public["error"], "network failed")
        self.assertNotIn("_internal", public)
        json.dumps(public)

    def test_terminal_helpers_set_completed_at_under_registry_lock(self):
        helpers = _load_helpers(
            self.tree,
            (
                "_new_download_record",
                "_update_download_record",
                "_complete_download_record",
                "_fail_download_record",
            ),
        )
        registry = helpers["_civitai_downloads"]
        registry["done"] = helpers["_new_download_record"]("done", "done.bin")
        helpers["_complete_download_record"]("done")
        self.assertEqual(registry["done"]["status"], "completed")
        self.assertEqual(registry["done"]["progress"], 100)
        self.assertIsNotNone(registry["done"]["completed_at"])

        registry["failed"] = helpers["_new_download_record"]("failed", "failed.bin")
        helpers["_fail_download_record"]("failed", ValueError("bad payload"))
        self.assertEqual(registry["failed"]["status"], "failed")
        self.assertEqual(registry["failed"]["error"], "bad payload")
        self.assertIsNotNone(registry["failed"]["completed_at"])

    def test_hf_progress_uses_zero_to_one_hundred_scale(self):
        helpers = _load_helpers(
            self.tree,
            (
                "_response_content_length",
                "_download_progress_percent",
                "_require_complete_download",
            ),
        )
        response_total = helpers["_response_content_length"]
        self.assertEqual(response_total({"content-length": "200"}), 200)
        self.assertEqual(
            response_total({
                "content-length": "200", "content-encoding": "identity",
            }),
            200,
        )
        self.assertEqual(
            response_total({
                "content-length": "80", "content-encoding": "gzip",
            }),
            0,
        )
        self.assertEqual(
            response_total({
                "content-length": "80", "content-encoding": "br",
            }),
            0,
        )
        self.assertEqual(
            response_total({
                "Content-Length": "80", "Content-Encoding": "gzip",
            }),
            0,
        )
        progress = helpers["_download_progress_percent"]
        self.assertEqual(progress(0, 200), 0)
        self.assertEqual(progress(50, 200), 25)
        self.assertEqual(progress(200, 200), 100)
        self.assertEqual(progress(300, 200), 100)
        self.assertEqual(progress(50, 0), 0)
        require_complete = helpers["_require_complete_download"]
        require_complete(200, 200)
        require_complete(50, 0)
        with self.assertRaisesRegex(IOError, "received 199 of 200 bytes"):
            require_complete(199, 200)

        hf_import = _function(self.tree, "hf_import_lora")
        hf_worker = _function(hf_import, "_do_download")
        calls = {
            node.func.id
            for node in ast.walk(hf_worker)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("_download_progress_percent", calls)
        self.assertIn("_response_content_length", calls)

    def test_path_components_reject_traversal_and_endpoint_wiring(self):
        helpers = _load_helpers(self.tree, ("_is_safe_path_component",))
        is_safe = helpers["_is_safe_path_component"]
        for value in ("model.safetensors", "flux2_klein_9b", "name-with dots"):
            with self.subTest(value=value):
                self.assertTrue(is_safe(value))
        for value in (
            "", ".", "..", "../escape", "folder/file", "folder\\file",
            "bad\x00name", "trailing.", "trailing ", "C:\\escape",
            "file:stream", "bad?.safetensors", "CON.safetensors", "LPT9.txt",
            "CON .safetensors", "CONIN$", "CONOUT$",
            "COM¹.txt", "COM².txt", "COM³.txt",
            "LPT¹.txt", "LPT².txt", "LPT³.txt",
        ):
            with self.subTest(value=value):
                self.assertFalse(is_safe(value))

        civitai_endpoint = _source_segment(
            self.tree, self.source, "civitai_download",
        )
        self.assertIn("_is_safe_path_component(filename)", civitai_endpoint)
        self.assertIn("_is_safe_path_component(target_arch)", civitai_endpoint)
        self.assertIn("_safe_join(lora_root, target_arch)", civitai_endpoint)
        hf_endpoint = _source_segment(
            self.tree, self.source, "hf_import_lora",
        )
        self.assertIn(
            "_is_safe_path_component(target_dir_override)", hf_endpoint,
        )
        self.assertIn("_is_safe_path_component(disk_filename)", hf_endpoint)

        civitai_worker = _source_segment(
            self.tree, self.source, "_run_civitai_download",
        )
        self.assertIn("os.path.basename", civitai_worker)
        self.assertIn("_is_safe_path_component(remote_filename)", civitai_worker)

    def test_target_reservation_is_normalized_and_single_flight(self):
        helpers = _load_helpers(
            self.tree,
            (
                "_normalize_download_target",
                "_reserve_download_target",
                "_release_download_target",
            ),
        )
        reserve = helpers["_reserve_download_target"]
        release = helpers["_release_download_target"]
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "models", "model.safetensors")
            equivalent = os.path.join(
                directory, "models", "..", "models", "model.safetensors",
            )
            reserved = reserve("first", target)
            self.assertIsNotNone(reserved)
            self.assertIsNone(reserve("second", equivalent))
            release("not-the-owner", target)
            self.assertIsNone(reserve("second", target))
            release("first", equivalent)
            self.assertIsNotNone(reserve("second", target))
            release("second", target)

    def test_guide_scans_use_a_separate_registry(self):
        status_source = _source_segment(
            self.tree, self.source, "scan_status",
        )
        start_source = _source_segment(
            self.tree, self.source, "scan_and_generate_guides",
        )
        self.assertIn("_lora_guide_scans", status_source)
        for helper_name in (
            "_register_lora_guide_scan",
            "_update_lora_guide_scan",
            "_append_lora_guide_scan_result",
        ):
            self.assertIn(helper_name, start_source)
        self.assertNotIn("_civitai_downloads", status_source)
        self.assertNotIn("_civitai_downloads", start_source)

    def test_guide_scan_helpers_lock_updates_and_appends(self):
        helpers = _load_helpers(
            self.tree,
            (
                "_register_lora_guide_scan",
                "_update_lora_guide_scan",
                "_append_lora_guide_scan_result",
            ),
        )
        register = helpers["_register_lora_guide_scan"]
        update = helpers["_update_lora_guide_scan"]
        append = helpers["_append_lora_guide_scan_result"]
        registry = helpers["_lora_guide_scans"]
        register("scan", {"status": "running", "results": []})

        threads = [
            threading.Thread(
                target=append,
                args=("scan", {"result": index}),
            )
            for index in range(40)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        update("scan", status="complete", current=40)

        self.assertEqual(registry["scan"]["status"], "complete")
        self.assertEqual(registry["scan"]["current"], 40)
        self.assertEqual(len(registry["scan"]["results"]), 40)
        for helper_name in (
            "_register_lora_guide_scan",
            "_update_lora_guide_scan",
            "_append_lora_guide_scan_result",
        ):
            helper_source = _source_segment(
                self.tree, self.source, helper_name,
            )
            self.assertIn("with _lora_guide_scan_lock", helper_source)

    def test_status_feed_serializes_instead_of_indexing_internal_records(self):
        status_source = _source_segment(
            self.tree, self.source, "civitai_downloads_status",
        )
        self.assertIn("_serialize_download_record", status_source)
        for required_key in (
            'dl["id"]', 'dl["filename"]', 'dl["bytes_downloaded"]',
        ):
            self.assertNotIn(required_key, status_source)

    def test_completion_follows_provider_postprocessing(self):
        civitai = _source_segment(
            self.tree, self.source, "_run_civitai_download",
        )
        civitai_complete = civitai.rindex("_complete_download_record")
        self.assertGreater(
            civitai_complete, civitai.index("_register_checkpoint_finetune"),
        )
        self.assertGreater(
            civitai_complete, civitai.index("_generate_and_save_checkpoint_guide"),
        )
        self.assertGreater(
            civitai_complete, civitai.index("_generate_and_save_lora_guide"),
        )

        hf_import = _function(self.tree, "hf_import_lora")
        hf_worker = _function(hf_import, "_do_download")
        hf_source = ast.get_source_segment(self.source, hf_worker)
        self.assertIsNotNone(hf_source)
        hf_complete = hf_source.rindex("_complete_download_record")
        self.assertGreater(hf_complete, hf_source.index("for i, media_url"))
        self.assertGreater(
            hf_complete, hf_source.index("_generate_and_save_lora_guide"),
        )

    def test_civitai_archive_members_are_validated_and_published_atomically(self):
        helpers = _load_helpers(
            self.tree,
            (
                "_safe_join",
                "_is_safe_path_component",
                "_normalize_download_target",
                "_reserve_download_target",
                "_release_download_target",
                "_validate_safetensors_payload",
                "_zip_member_target",
                "_copy_zip_member_to_partial",
                "_extract_civitai_archive",
            ),
        )
        reserve = helpers["_reserve_download_target"]
        release = helpers["_release_download_target"]
        extract = helpers["_extract_civitai_archive"]
        member_target = helpers["_zip_member_target"]
        valid_payload = (
            struct.pack("<Q", 2)
            + b"{}"
            + b"\0" * (100 * 1024)
        )

        with tempfile.TemporaryDirectory() as directory:
            target_dir = os.path.join(directory, "loras")
            os.makedirs(target_dir)
            archive = os.path.join(target_dir, "bundle.zip")
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("nested/model.safetensors", valid_payload)

            reserved_targets = {reserve("download", archive)}
            partial_paths = set()
            try:
                extracted = extract(
                    archive,
                    target_dir,
                    "download",
                    reserved_targets,
                    partial_paths,
                )
                self.assertEqual(
                    extracted, [os.path.join(target_dir, "model.safetensors")],
                )
                self.assertTrue(os.path.isfile(extracted[0]))
                self.assertEqual(os.path.getsize(extracted[0]), len(valid_payload))
                self.assertFalse(partial_paths)
                self.assertTrue(os.path.isfile(archive))
            finally:
                for partial_path in tuple(partial_paths):
                    if os.path.isfile(partial_path):
                        os.remove(partial_path)
                for target in tuple(reserved_targets):
                    release("download", target)

            self.assertTrue(
                member_target(target_dir, "nested/good.txt", flatten=False).endswith(
                    os.path.join("nested", "good.txt")
                )
            )
            for unsafe_member in (
                "../escape.txt", "nested/../escape.txt", "nested/CONIN$",
            ):
                with self.subTest(member=unsafe_member):
                    with self.assertRaisesRegex(ValueError, "unsafe archive member"):
                        member_target(target_dir, unsafe_member, flatten=False)

    def test_civitai_archive_failure_keeps_archive_and_rejects_duplicates(self):
        helpers = _load_helpers(
            self.tree,
            (
                "_safe_join",
                "_is_safe_path_component",
                "_normalize_download_target",
                "_reserve_download_target",
                "_release_download_target",
                "_validate_safetensors_payload",
                "_zip_member_target",
                "_copy_zip_member_to_partial",
                "_extract_civitai_archive",
            ),
        )
        reserve = helpers["_reserve_download_target"]
        release = helpers["_release_download_target"]
        extract = helpers["_extract_civitai_archive"]
        valid_payload = (
            struct.pack("<Q", 2)
            + b"{}"
            + b"\0" * (100 * 1024)
        )

        with tempfile.TemporaryDirectory() as directory:
            target_dir = os.path.join(directory, "loras")
            os.makedirs(target_dir)
            archive = os.path.join(target_dir, "duplicate.zip")
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("one/model.safetensors", valid_payload)
                handle.writestr("two/model.safetensors", valid_payload)

            reserved_targets = {reserve("download", archive)}
            partial_paths = set()
            try:
                with self.assertRaisesRegex(ValueError, "duplicate target"):
                    extract(
                        archive,
                        target_dir,
                        "download",
                        reserved_targets,
                        partial_paths,
                    )
                self.assertTrue(os.path.isfile(archive))
                self.assertFalse(
                    os.path.exists(os.path.join(target_dir, "model.safetensors"))
                )
            finally:
                for partial_path in tuple(partial_paths):
                    if os.path.isfile(partial_path):
                        os.remove(partial_path)
                for target in tuple(reserved_targets):
                    release("download", target)

            invalid_archive = os.path.join(target_dir, "invalid.zip")
            with zipfile.ZipFile(invalid_archive, "w") as handle:
                handle.writestr("bad.safetensors", b"not safetensors")
            reserved_targets = {reserve("invalid", invalid_archive)}
            partial_paths = set()
            try:
                with self.assertRaisesRegex(
                    RuntimeError, "Invalid safetensors archive member",
                ):
                    extract(
                        invalid_archive,
                        target_dir,
                        "invalid",
                        reserved_targets,
                        partial_paths,
                    )
                self.assertTrue(os.path.isfile(invalid_archive))
            finally:
                for partial_path in tuple(partial_paths):
                    if os.path.isfile(partial_path):
                        os.remove(partial_path)
                for target in tuple(reserved_targets):
                    release("invalid", target)

            short_archive = os.path.join(target_dir, "short.zip")
            with zipfile.ZipFile(short_archive, "w") as handle:
                handle.writestr("payload.bin", b"expected payload")

            def short_copy(_zip_file, _member, partial_path):
                with open(partial_path, "wb") as handle:
                    handle.write(b"x")
                return 1

            helpers["_copy_zip_member_to_partial"] = short_copy
            reserved_targets = {reserve("short", short_archive)}
            partial_paths = set()
            try:
                with self.assertRaisesRegex(IOError, "Incomplete archive member"):
                    extract(
                        short_archive,
                        target_dir,
                        "short",
                        reserved_targets,
                        partial_paths,
                    )
                self.assertTrue(os.path.isfile(short_archive))
            finally:
                for partial_path in tuple(partial_paths):
                    if os.path.isfile(partial_path):
                        os.remove(partial_path)
                for target in tuple(reserved_targets):
                    release("short", target)

    def test_provider_streams_publish_atomically_and_clean_partials(self):
        civitai = _source_segment(
            self.tree, self.source, "_run_civitai_download",
        )
        hf_import = _function(self.tree, "hf_import_lora")
        hf_worker = _function(hf_import, "_do_download")
        hf_source = ast.get_source_segment(self.source, hf_worker)
        self.assertIsNotNone(hf_source)

        for provider_source in (civitai, hf_source):
            self.assertIn("uuid.uuid4().hex", provider_source)
            self.assertIn(".part", provider_source)
            self.assertIn("_response_content_length", provider_source)
            self.assertIn("os.replace(partial_path, save_path)", provider_source)
            reserve_at = provider_source.index("_reserve_download_target")
            open_at = provider_source.index('open(partial_path, "wb")')
            size_check_at = provider_source.index("_require_complete_download")
            replace_at = provider_source.index(
                "os.replace(partial_path, save_path)",
            )
            release_at = provider_source.index("_release_download_target")
            self.assertLess(reserve_at, open_at)
            self.assertLess(size_check_at, replace_at)
            self.assertGreater(release_at, replace_at)

        self.assertIn("os.remove(cleanup_path)", civitai)
        self.assertNotIn("ZIP extraction failed", civitai)
        self.assertIn("_extract_civitai_archive", civitai)
        self.assertIn("_validate_safetensors_payload(partial_path)", civitai)
        self.assertGreater(
            hf_source.index("os.remove(partial_path)"),
            hf_source.index("os.replace(partial_path, save_path)"),
        )

        archive_source = _source_segment(
            self.tree, self.source, "_extract_civitai_archive",
        )
        for wiring in (
            "_zip_member_target",
            "_reserve_download_target",
            "duplicate target",
            "uuid.uuid4().hex",
            "copied != member.file_size",
            "_validate_safetensors_payload(partial_path)",
            "os.replace(partial_path, final_path)",
        ):
            self.assertIn(wiring, archive_source)

        hf_import_source = _source_segment(
            self.tree, self.source, "hf_import_lora",
        )
        self.assertIn('dl_id = f"hf_{uuid.uuid4().hex}"', hf_import_source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
