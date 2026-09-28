"""LoRA presentation names and creator guides never change model identity."""
from __future__ import annotations

import ast
import asyncio
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from services import lora_metadata as metadata


ORBIT = {
    "source": "huggingface", "name": "MiniMax-H3-360-Orbit-LoRA",
    "repoId": "creator/MiniMax-H3-360-Orbit-LoRA", "trainedWords": [],
    "description": "# MiniMax-H3 360° Orbit LoRA\nA frozen instant.\n\n"
                   "## Prompt\nOnly the camera moves in a continuous 360 orbit.\n"
                   "Keep every person motionless.\n\n## Training\nInternal training logs.",
}
FILENAME = "minimax_h3_flf2v_lora_v1.safetensors"


def launch_functions(names, namespace=None):
    tree = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    scope = {"os": os, "json": json, "Request": object, "PureWindowsPath": PureWindowsPath,
             **(namespace or {})}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "app/launch.py", "exec"), scope)
    return scope


class LoRaMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names = self.root / "settings/names.json"
        self.name_patch = patch.object(metadata, "_NAMES_PATH", self.names)
        self.name_patch.start()
        self.addCleanup(self.name_patch.stop)

    def test_existing_hf_filename_gets_model_card_title_without_migration(self):
        before = json.dumps(ORBIT)
        result = metadata.lora_name_fields(FILENAME, ORBIT, "minimax_h3")
        self.assertEqual(result["display_name"], "MiniMax-H3 360° Orbit LoRA")
        self.assertIsNone(result["display_name_override"])
        self.assertEqual(json.dumps(ORBIT), before)
        self.assertFalse(self.names.exists())

    def test_alias_persists_and_reset_restores_publisher_name(self):
        metadata.set_lora_display_name(FILENAME, ORBIT, "minimax_h3", "Frozen orbit")
        self.assertEqual(metadata.lora_name_fields(FILENAME, ORBIT, "minimax_h3")["display_name"], "Frozen orbit")
        self.assertEqual(metadata.lora_name_fields(FILENAME, ORBIT, "minimax_h3")["suggested_name"], "MiniMax-H3 360° Orbit LoRA")
        metadata.set_lora_display_name(FILENAME, ORBIT, "minimax_h3", None)
        self.assertEqual(metadata.lora_name_fields(FILENAME, ORBIT, "minimax_h3")["display_name"], "MiniMax-H3 360° Orbit LoRA")

    def test_civitai_alias_survives_version_and_filename_changes(self):
        metadata.set_lora_display_name("v1.safetensors", {"modelId": "123", "versionId": 4}, "ltx2", "Smooth camera")
        result = metadata.lora_name_fields("v2.safetensors", {"modelId": 123, "versionId": 8, "name": "Official v2"}, "ltx2")
        self.assertEqual(result["display_name"], "Smooth camera")
        self.assertEqual(result["suggested_name"], "Official v2")

    def test_same_publisher_title_keeps_installed_versions_and_variants_distinct(self):
        info = {"modelId": 123, "name": "Example adapter"}
        files = ["Adapter_H3-V1.safetensors", "Adapter_H3-V4.safetensors",
                 "Adapter_H3-V4-ref2va.safetensors"]
        labels = [metadata.lora_name_fields(name, info, "minimax_h3") for name in files]
        self.assertEqual([item["display_name"] for item in labels], ["Example adapter"] * 3)
        self.assertEqual([item["version_label"] for item in labels], ["V1", "V4", "V4 · ref2va"])
        self.assertFalse(self.names.exists(), "Existing files need no migration or metadata rescan")
        metadata.set_lora_display_name(files[0], info, "minimax_h3", "My adapter")
        renamed = [metadata.lora_name_fields(name, info, "minimax_h3") for name in files]
        self.assertEqual([item["display_name"] for item in renamed], ["My adapter"] * 3)
        self.assertEqual([item["version_label"] for item in renamed], ["V1", "V4", "V4 · ref2va"])

    def test_release_titles_and_filename_variants_are_combined_without_duplicates(self):
        cases = [
            ("adapter_v1.2_fp16.safetensors", {"versionName": "Release 1.2"}, "Release 1.2 · fp16"),
            ("adapter_V4-ref2va.safetensors", {"versionName": "V4 ref2va"}, "V4 ref2va"),
            ("weights.safetensors", {"versionName": "Turbo Beta"}, "Turbo Beta"),
            ("minimax_h3_flf2v_lora_v1.safetensors", ORBIT, "V1"),
            ("adapter_fp16.safetensors", {}, None),
            ("Qwen3_H3_ref2va.safetensors", {}, None),
        ]
        for filename, info, expected in cases:
            with self.subTest(filename=filename):
                self.assertEqual(metadata.lora_version_label(filename, info), expected)

    def test_update_check_backfills_the_installed_release_title_not_latest(self):
        tree = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
        helper = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "_backfill_published_at")
        namespace = {"json": json}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "app/launch.py", "exec"), namespace)
        first = self.root / "old.civitai.json"
        second = self.root / "new.civitai.json"
        first.write_text(json.dumps({"modelId": 123, "versionId": 1, "publishedAt": "saved date"}), encoding="utf-8")
        second.write_text(json.dumps({"modelId": 123, "versionId": 4}), encoding="utf-8")
        namespace["_backfill_published_at"]([str(first), str(second)], {"modelVersions": [
            {"id": 4, "name": "V4", "publishedAt": "new date"},
            {"id": 1, "name": "V1", "publishedAt": "old date"},
        ]})
        self.assertEqual(json.loads(first.read_text()), {
            "modelId": 123, "versionId": 1, "publishedAt": "saved date", "versionName": "V1"})
        self.assertEqual(json.loads(second.read_text()), {
            "modelId": 123, "versionId": 4, "publishedAt": "new date", "versionName": "V4"})

    def test_local_names_do_not_collide_between_model_folders(self):
        metadata.set_lora_display_name("adapter.safetensors", {}, "qwen21", "Portrait detail")
        other = metadata.lora_name_fields("adapter.safetensors", {}, "minimax_h3")
        self.assertIsNone(other["display_name_override"])

    def test_legacy_scan_date_cannot_make_old_weights_new(self):
        weights = self.root / "adapter.safetensors"
        weights.write_bytes(b"old weights")
        original = "2026-08-11T22:46:17Z"
        epoch = datetime.fromisoformat(original.replace("Z", "+00:00")).timestamp()
        os.utime(weights, (epoch, epoch))
        scanned = {"downloadedAt": "2026-09-28T13:00:00Z"}
        expected = {"released_at": None, "downloaded_at": original}
        self.assertEqual(metadata.lora_date_fields(weights, scanned), expected)
        self.assertEqual(scanned["downloadedAt"], "2026-09-28T13:00:00Z")
        # Alias/guide metadata updates must not influence either displayed date.
        metadata.set_lora_display_name(weights.name, scanned, "h3", "Camera")
        weights.with_suffix(".guide.md").write_text("Updated guidance", encoding="utf-8")
        self.assertEqual(metadata.lora_date_fields(weights, scanned), expected)

    def test_release_and_explicit_download_dates_are_preserved(self):
        weights = self.root / "adapter.safetensors"
        weights.write_bytes(b"copied file with upstream timestamp")
        os.utime(weights, (1700000000, 1700000000))
        saved = {"publishedAt": "2026-08-01T10:30:00+02:00",
                 "downloadedAt": "2026-09-01T12:00:00Z", "downloadedAtSource": "download"}
        self.assertEqual(metadata.lora_date_fields(weights, saved), {
            "released_at": "2026-08-01T08:30:00Z", "downloaded_at": "2026-09-01T12:00:00Z"})
        # Moving/copying the weights later also keeps an older recorded download.
        saved.pop("downloadedAtSource")
        os.utime(weights, None)
        self.assertEqual(metadata.lora_date_fields(weights, saved)["downloaded_at"], "2026-09-01T12:00:00Z")

    def test_invalid_dates_fall_back_without_making_up_today(self):
        weights = self.root / "missing.safetensors"
        result = metadata.lora_date_fields(weights, {"publishedAt": "bad", "downloadedAt": 123})
        self.assertEqual(result, {"released_at": None, "downloaded_at": None})
        weights.write_bytes(b"weights")
        os.utime(weights, (1700000000, 1700000000))
        result = metadata.lora_date_fields(weights, {"publishedAt": "bad", "downloadedAt": "bad"})
        self.assertIsNone(result["released_at"])
        self.assertEqual(result["downloaded_at"], "2023-11-14T22:13:20Z")

    def test_hash_scan_records_release_and_weight_dates_not_scan_time(self):
        weights = self.root / "adapter.safetensors"
        weights.write_bytes(b"test weights")
        os.utime(weights, (1700000000, 1700000000))
        # Exercise the real background worker without requests, LLMs or threads.
        tree = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
        parent = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                      and n.name == "scan_and_generate_guides")
        worker = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == "_run_scan")
        response = SimpleNamespace(status_code=200, json=lambda: {
            "id": 1, "name": "V1", "publishedAt": "2023-11-01T12:00:00Z", "model": {"name": "Camera"}})
        guide_writer = Mock(return_value={"guide": "Usage"})
        events = Mock()
        scope = {"os": os, "json": json, "time": time, "hashlib": hashlib,
                 "wgp": SimpleNamespace(server_config={}), "requests": SimpleNamespace(get=Mock(return_value=response)),
                 "CIVITAI_USER_AGENT": "test", "CIVITAI_BASE_URL": "https://example.invalid",
                 "scan_id": "test", "to_process": [{"path": str(weights), "filename": weights.name,
                     "write_base": str(weights.with_suffix("")), "has_sidecar": False}],
                 "_update_lora_guide_scan": Mock(), "_append_lora_guide_scan_result": events,
                 "_ensure_llm_loaded": Mock(), "_generate_and_save_lora_guide": guide_writer}
        exec(compile(ast.Module(body=[worker], type_ignores=[]), "app/launch.py", "exec"), scope)
        scope["_run_scan"]()
        saved = json.loads(weights.with_suffix(".civitai.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["publishedAt"], "2023-11-01T12:00:00Z")
        self.assertEqual(saved["downloadedAt"], "2023-11-14T22:13:20Z")
        self.assertEqual(saved["downloadedAtSource"], "weight_file")
        self.assertNotEqual(saved["metadataScannedAt"], saved["downloadedAt"])
        guide_writer.assert_called_once()
        self.assertNotIn("error", repr(events.call_args_list))

    def test_reject_invalid_alias_without_losing_saved_name(self):
        metadata.set_lora_display_name(FILENAME, ORBIT, "minimax_h3", "Saved name")
        for value in (12, "x" * 121, "name\nextra", "name\x00"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                metadata.set_lora_display_name(FILENAME, ORBIT, "minimax_h3", value)
        self.assertEqual(metadata.lora_name_fields(FILENAME, ORBIT, "minimax_h3")["display_name"], "Saved name")

    def test_guide_can_supply_name_when_no_publisher_metadata(self):
        result = metadata.lora_name_fields("weights_v1.safetensors", {}, "minimax_h3", "# Gentle camera push\nUsage.")
        self.assertEqual(result["display_name"], "Gentle camera push")

    def test_creator_notes_preserved_without_inventing_trigger(self):
        guide = metadata.source_lora_guide(ORBIT, FILENAME)
        self.assertIn("Only the camera moves in a continuous 360 orbit", guide)
        self.assertIn("Keep every person motionless", guide)
        self.assertNotIn("Internal training logs", guide)
        self.assertNotIn("Creator-declared trigger words", guide)

    def test_creator_trigger_tokens_are_exact(self):
        guide = metadata.source_lora_guide({"name": "Camera", "trainedWords": ["c4m_token", "<orbit>"],
                                            "description": "<p>Keep the subject still.</p><script>bad()</script>"})
        self.assertIn("c4m_token, <orbit>", guide)
        self.assertIn("Keep the subject still.", guide)
        self.assertNotIn("bad()", guide)

    def test_import_guide_needs_no_llm_and_keeps_version_tracking(self):
        weights = self.root / FILENAME
        weights.write_bytes(b"unchanged model bytes")
        sidecar = weights.with_suffix(".civitai.json")
        info = {**ORBIT, "modelId": 123, "versionId": 42, "recommendedWeights": {"default": 1.0}}
        sidecar.write_text(json.dumps(info), encoding="utf-8")
        namespace = launch_functions({"_generate_and_save_lora_guide"})
        # No _ensure_llm_loaded / _build_lora_context globals or llm_service module
        # are present: import-time guide creation must not need the GPU service.
        with patch.dict(sys.modules, {"services.llm_service": None}):
            result = namespace["_generate_and_save_lora_guide"](str(weights), info, FILENAME, source_only=True)
        self.assertIn("continuous 360 orbit", result["guide"])
        self.assertEqual(weights.read_bytes(), b"unchanged model bytes")
        after = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertEqual(after["versionId"], 42)
        self.assertEqual(after["recommendedWeights"], {"default": 1.0})
        self.assertEqual(after["guideSource"], "creator")

    def test_automatic_guide_does_not_overwrite_an_existing_edited_guide(self):
        weights = self.root / FILENAME
        guide = weights.with_suffix(".guide.md")
        guide.write_text("My carefully edited guide.", encoding="utf-8")
        result = launch_functions({"_generate_and_save_lora_guide"})["_generate_and_save_lora_guide"](
            str(weights), ORBIT, FILENAME, source_only=True)
        self.assertEqual(result["guide"], "My carefully edited guide.")
        self.assertEqual(guide.read_text(encoding="utf-8"), result["guide"])


class HttpError(Exception):
    def __init__(self, status_code, detail):
        self.status_code, self.detail = status_code, detail


class DisplayNameRouteTests(unittest.TestCase):
    def setUp(self):
        LoRaMetadataTests.setUp(self)
        self.library = self.root / "loras"
        self.primary = self.library / "minimax_h3"
        self.primary.mkdir(parents=True)
        self.linked = self.root / "linked"
        (self.linked / "minimax_h3").mkdir(parents=True)
        self.file = self.linked / "minimax_h3" / FILENAME
        self.file.write_bytes(b"weights in linked library")
        self.sidecar = self.primary / Path(FILENAME).with_suffix(".civitai.json")
        self.sidecar.write_text(json.dumps(ORBIT), encoding="utf-8")
        self.ns = launch_functions({"update_lora_display_name", "_safe_join", "_is_safe_path_component", "_compute_lora_id"}, {
            "HTTPException": HttpError,
            "_resolve_lora_root": lambda: str(self.library),
            "_linked_lora_roots": lambda: [str(self.linked)],
            "wgp": SimpleNamespace(get_model_def=lambda model: {} if model == "h3" else None,
                get_lora_dir=lambda _: str(self.primary),
                resolve_lora_path=lambda _, filename: str(self.file) if filename == FILENAME else None),
        })

    def request(self, **body):
        async def json_body():
            return body
        return asyncio.run(self.ns["update_lora_display_name"](SimpleNamespace(json=json_body)))

    def test_linked_model_rename_roundtrips_across_selector_and_library(self):
        before = self.sidecar.read_bytes()
        result = self.request(filename=FILENAME, display_name="Camera orbit", model_type="h3")
        self.assertEqual(result["filename"], FILENAME)
        self.assertEqual(result["lora_id"], "local:" + FILENAME)
        self.assertEqual(result["display_name"], "Camera orbit")
        reset = self.request(filename=FILENAME, display_name=None, directory="minimax_h3")
        self.assertEqual(reset["display_name"], "MiniMax-H3 360° Orbit LoRA")
        self.assertEqual(self.sidecar.read_bytes(), before)
        self.assertEqual(self.file.read_bytes(), b"weights in linked library")
        self.assertEqual(list((self.linked / "minimax_h3").iterdir()), [self.file])

    def test_invalid_paths_missing_files_and_ambiguous_requests_fail(self):
        for payload, code in [
            ({"filename": "../escape.safetensors", "display_name": "Bad", "directory": "minimax_h3"}, 400),
            ({"filename": FILENAME, "display_name": "Bad", "directory": "../linked/minimax_h3"}, 400),
            ({"filename": FILENAME, "model_type": "h3"}, 400),
            ({"filename": FILENAME, "display_name": "Bad", "model_type": "h3", "directory": "minimax_h3"}, 400),
            ({"filename": "missing.safetensors", "display_name": "Bad", "directory": "minimax_h3"}, 404),
        ]:
            with self.subTest(payload=payload), self.assertRaises(HttpError) as error:
                self.request(**payload)
            self.assertEqual(error.exception.status_code, code)
        self.assertFalse(self.names.exists())


class EnhancementLoRaWiringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        (folder / FILENAME).write_bytes(b"weights")
        (folder / Path(FILENAME).with_suffix(".civitai.json")).write_text(json.dumps(ORBIT), encoding="utf-8")
        (folder / Path(FILENAME).with_suffix(".guide.md")).write_text("A frozen instant. Only the camera moves in a continuous 360 orbit.", encoding="utf-8")
        self.wgp = SimpleNamespace(server_config={"services": {}}, get_model_def=lambda _: {},
                                   get_lora_search_dirs=lambda _: [str(folder)])
        self.ns = launch_functions({"_active_lora_hint", "_llm_enhance_prompt_payload"}, {
            "wgp": self.wgp, "_ensure_llm_loaded": Mock(), "HTTPException": HttpError,
            "enhancement_settings": lambda value: value, "enhancement_warnings": lambda: [],
            "_PUBLIC_LLM_PROVIDERS": {"openai", "anthropic"},
        })

    def test_h3_receives_saved_motion_guide_while_source_stays_unchanged(self):
        from services import llm_service
        with patch.object(llm_service, "enhance_prompt", return_value="Valid draft") as writer:
            result = asyncio.run(self.ns["_llm_enhance_prompt_payload"]({
                "prompt": ".", "model_type": "minimax_h3_fused_turbo", "mode": "video",
                "activated_loras": [FILENAME], "duration_seconds": 5.88,
            }))
        self.assertEqual(result["original"], ".")
        self.assertEqual(writer.call_args.kwargs["prompt"], ".")
        self.assertIn("continuous 360 orbit", writer.call_args.kwargs["lora_system_hint"])
        self.assertEqual(writer.call_args.kwargs["duration_seconds"], 5.88)

    def test_disabled_lora_does_not_influence_enhancement(self):
        from services import llm_service
        with patch.object(llm_service, "enhance_prompt", return_value="Plain draft") as writer:
            asyncio.run(self.ns["_llm_enhance_prompt_payload"]({"prompt": ".", "model_type": "image", "mode": "image"}))
        self.assertEqual(writer.call_args.kwargs["lora_system_hint"], "")

    def test_legacy_enhancer_receives_same_guide_without_switching_provider(self):
        self.wgp.server_config["enhancer_enabled"] = 1
        legacy = AsyncMock(return_value={"original": ".", "enhanced": "Legacy draft"})
        self.ns["_enhance_with_wangp"] = legacy
        asyncio.run(self.ns["_llm_enhance_prompt_payload"]({
            "prompt": ".", "model_type": "image", "mode": "image", "activated_loras": [FILENAME],
        }))
        self.assertIn("continuous 360 orbit", legacy.call_args.kwargs["lora_system_hint"])
        self.ns["_ensure_llm_loaded"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
