"""Speech model_mode handling and handler-managed (Chatterbox) pre-downloads."""
from __future__ import annotations

import ast
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

from models.TTS import chatterbox_handler, qwen3_handler  # noqa: E402

LAUNCH = APP / "launch.py"
STORE = ROOT / "ui" / "src" / "stores" / "useStore.ts"
AUDIO_SECTION = ROOT / "ui" / "src" / "components" / "Sidebar" / "AudioModeSection.tsx"


def launch_functions(names, namespace):
    tree = ast.parse(LAUNCH.read_text(encoding="utf-8"), filename=str(LAUNCH))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    missing = set(names) - {node.name for node in selected}
    if missing:
        raise AssertionError(f"launch.py is missing {sorted(missing)}")
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(LAUNCH), "exec"), namespace)
    return namespace


class SpeechModelModeSettingsTests(unittest.TestCase):
    """API requests merge onto primary_settings (model_mode=None), and the
    React UI used to carry YuE 2's numeric mode into Qwen3 Base."""

    def fixed(self, handler, base_model_type, model_mode):
        settings = {"model_mode": model_mode}
        handler.family_handler.fix_settings(base_model_type, 2.6, {}, settings)
        return settings["model_mode"]

    def test_qwen3_replaces_missing_or_numeric_mode_with_default(self):
        speaker = qwen3_handler.get_qwen3_speakers("qwen3_tts_customvoice")[0]
        cases = {
            "qwen3_tts_customvoice": speaker,
            "qwen3_tts_voicedesign": "auto",
            "qwen3_tts_base": "auto",
        }
        for base_model_type, default in cases.items():
            for stale in (None, 2, 0, "", "  "):
                with self.subTest(model=base_model_type, model_mode=stale):
                    self.assertEqual(self.fixed(qwen3_handler, base_model_type, stale), default)

    def test_qwen3_keeps_explicit_language_and_speaker(self):
        self.assertEqual(self.fixed(qwen3_handler, "qwen3_tts_base", "english"), "english")
        self.assertEqual(self.fixed(qwen3_handler, "qwen3_tts_customvoice", "serena"), "serena")

    def test_customvoice_no_longer_rejects_a_request_without_speaker(self):
        settings = {"model_mode": None}
        qwen3_handler.family_handler.fix_settings("qwen3_tts_customvoice", 2.6, {}, settings)
        error = qwen3_handler.family_handler.validate_generative_prompt(
            "qwen3_tts_customvoice", {}, settings, "Hello there.")
        self.assertIsNone(error)

    def test_chatterbox_replaces_missing_or_numeric_language(self):
        for stale in (None, 2, ""):
            with self.subTest(model_mode=stale):
                self.assertEqual(self.fixed(chatterbox_handler, "chatterbox", stale), "en")
        self.assertEqual(self.fixed(chatterbox_handler, "chatterbox", "fr"), "fr")


class HandlerManagedDownloadTests(unittest.TestCase):
    """Chatterbox declares no URLs; its weights come from query_model_files()."""

    FILE_DEFS = [chatterbox_handler.family_handler.query_model_files(None, "chatterbox", {})]

    def helpers(self, root, model_def=None):
        def locate_file(name, error_if_none=True):
            path = os.path.join(root, name)
            return path if os.path.isfile(path) else None

        def locate_folder(name, error_if_none=True):
            path = os.path.join(root, name)
            return path if os.path.isdir(path) else None

        handler = types.SimpleNamespace(query_model_files=lambda compute, base, md: self.FILE_DEFS[0])
        wgp = types.SimpleNamespace(
            get_model_def=lambda _model_type: model_def if model_def is not None else {"URLs": []},
            get_base_model_type=lambda model_type: model_type,
            get_runtime_model_def=lambda _model_type: {},
            get_model_recursive_prop=lambda *_args, **_kwargs: [],
            resolve_lora_path=lambda *_args, **_kwargs: "",
            model_types_handlers={"chatterbox": handler},
            fl=types.SimpleNamespace(locate_file=locate_file, locate_folder=locate_folder),
        )
        return launch_functions(
            {"_model_weight_groups", "_handler_model_files", "_handler_files_downloaded",
             "_check_model_downloaded"},
            {"os": os, "wgp": wgp, "_variant_group_downloaded": lambda *_args, **_kwargs: True},
        )

    def test_handler_files_decide_readiness_when_no_urls(self):
        with tempfile.TemporaryDirectory() as root:
            api = self.helpers(root)
            self.assertEqual(api["_handler_model_files"]("chatterbox"), self.FILE_DEFS)
            self.assertFalse(api["_check_model_downloaded"]("chatterbox"))
            # wgp.process_files_def writes <root>/<target>/<source>/<file>.
            folder = Path(root, "chatterbox")
            folder.mkdir()
            *first, last = self.FILE_DEFS[0]["fileList"][0]
            for name in first:
                (folder / name).write_bytes(b"x")
            self.assertFalse(api["_check_model_downloaded"]("chatterbox"))
            (folder / last).write_bytes(b"x")
            self.assertTrue(api["_check_model_downloaded"]("chatterbox"))

    def test_url_models_still_use_weight_groups(self):
        with tempfile.TemporaryDirectory() as root:
            api = self.helpers(root, model_def={"URLs": ["https://example.test/model.safetensors"]})
            self.assertTrue(api["_check_model_downloaded"]("chatterbox"))

    def test_predownload_runs_handler_file_list_when_no_urls(self):
        calls = []
        wgp = types.SimpleNamespace(
            get_model_def=lambda _model_type: {"URLs": []},
            transformer_quantization="int8",
            transformer_dtype_policy="",
            text_encoder_quantization="int8",
            get_transformer_dtype=lambda *_args: None,
            get_model_filename=lambda **_kwargs: "",
            get_model_recursive_prop=lambda *_args, **_kwargs: None if _args[1] == "text_encoder_URLs" else [],
            download_models=lambda *args, **kwargs: calls.append(args),
        )
        api = launch_functions({"_download_model_files"}, {
            "os": os, "wgp": wgp, "check_download_cancelled": lambda: None,
            "_check_model_downloaded": lambda _model_type: True,
        })
        api["_download_model_files"]("chatterbox")
        self.assertEqual(calls, [("", "chatterbox", 0, -1)])


class SpeechModeUiContractTests(unittest.TestCase):
    def test_model_switch_resets_model_owned_mode(self):
        source = STORE.read_text(encoding="utf-8")
        start = source.index("function _applyModelDefaults(")
        body = source[start:source.index("\n}\n", start)]
        self.assertIn("overrides.model_mode = (d as Record<string, unknown>).model_mode ?? undefined", body)

    def test_speech_section_offers_language_and_speaker_choices(self):
        source = AUDIO_SECTION.read_text(encoding="utf-8")
        self.assertIn("modelOptions.model_modes", source)
        self.assertIn("setParam('model_mode', speechModes.default)", source)
        self.assertIn("aria-label={speechModes.label}", source)


if __name__ == "__main__":
    unittest.main()
