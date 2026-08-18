"""Behavioral regressions for Director text and prompt integrity."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


_ROOT = Path(__file__).resolve().parents[1]
_APP = _ROOT / "app"
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from services.text_integrity import repair_payload, repair_text
from services.director.planners.base import BasePlanner
from services import llm_service
from services.director_pipeline import _write_pipeline_json_unlocked, load_pipeline_state


class _Planner(BasePlanner):
    skill_type = "integrity_test"

    def plan(self, **kwargs):  # pragma: no cover - abstract interface only
        return kwargs


class TestUnicodeIntegrity(unittest.TestCase):
    def test_repairs_utf8_decoded_as_windows_codepage(self):
        self.assertEqual(repair_text("WÃ¶rter"), "Wörter")
        self.assertEqual(repair_text("WÃƒÂ¶rter"), "Wörter")
        self.assertEqual(repair_text("Youâ€™re ready"), "You’re ready")
        self.assertEqual(
            repair_payload({"shots": [{"prompt": "GrÃ¼ÃŸe aus KÃ¶ln"}]}),
            {"shots": [{"prompt": "Grüße aus Köln"}]},
        )

    def test_preserves_valid_international_text(self):
        value = "François heißt Björk — Ελληνικά — 日本語 — 你好"
        self.assertEqual(repair_text(value), value)

    def test_shared_planner_repairs_input_and_nested_llm_output(self):
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return '[{"title":"WÃ¶rter", "nested":{"line":"GrÃ¼ÃŸe"}}]'

        planner = _Planner(llm_generate=generate)
        result = planner._call_llm_json(
            "Plane WÃ¶rter",
            "Bewahre GrÃ¼ÃŸe",
            streaming=False,
            thinking_budget=0,
        )

        self.assertEqual(calls[0]["prompt"], "Plane Wörter")
        self.assertEqual(calls[0]["system_prompt"], "Bewahre Grüße")
        self.assertEqual(result[0]["title"], "Wörter")
        self.assertEqual(result[0]["nested"]["line"], "Grüße")

    def test_director_state_is_persisted_as_utf8_without_mojibake(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.json"
            _write_pipeline_json_unlocked(
                str(path),
                {"scene_description": "WÃ¶rter", "clips": []},
            )
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["scene_description"], "Wörter")

    def test_legacy_saved_director_state_is_repaired_when_loaded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "_director_pipeline_demo.json"
            path.write_text(
                json.dumps({
                    "pipeline_id": "demo",
                    "status": "completed",
                    "scene_description": "GrÃ¼ÃŸe aus KÃ¶ln",
                    "clips": [],
                    "output_files": [],
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            loaded = load_pipeline_state(temp_dir, "demo")
            persisted = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["scene_description"], "Grüße aus Köln")
        self.assertEqual(persisted["scene_description"], "Grüße aus Köln")


class TestH3PromptIntegrity(unittest.TestCase):
    def test_requested_language_is_part_of_dialogue_contract(self):
        prompt = 'A woman speaks perfect French and says, "Bonjour, comment ça va ?"'
        compiled = llm_service._compile_h3_explicit_dialogue(prompt)
        requirement = llm_service._build_h3_dialogue_requirement(prompt, 8)

        self.assertEqual(llm_service._detect_h3_dialogue_language(prompt), "French")
        self.assertIn("<d>[French] Bonjour, comment ça va ?</d>", compiled)
        self.assertIn("<d>[French] Bonjour, comment ça va ?</d>", requirement)
        self.assertFalse(
            llm_service._h3_dialogue_contract_satisfied(
                prompt,
                "Woman (S1): <d>[English] Bonjour, comment ça va ?</d>",
            )
        )
        self.assertTrue(
            llm_service._h3_dialogue_contract_satisfied(
                prompt,
                "Woman (S1): <d>[French] Bonjour, comment ça va ?</d>",
            )
        )

    def test_h3_enhancer_repairs_language_and_grounds_attached_frame(self):
        prompt = 'A woman speaks perfect French and says, "Bonjour, comment ça va ?"'
        weak = (
            "integrated_multimodal_description: [Shot 1] A woman speaks to camera. "
            "From 0.00 to 1.50 seconds, her mouth stays closed and there is no human voice. "
            "Woman (S1): <d>[English] Bonjour, comment ça va ?</d> "
            "From 4.50 to 8.00 seconds, she remains silent with her mouth closed.\n\n"
            "overall_soundscape: Quiet room tone with no other voices.\n\n"
            "non_diegetic_music: N/A"
        )
        visual_anchor = (
            "A woman with shoulder-length auburn hair wears a navy wool jacket and stands "
            "screen-left in a medium shot inside a sunlit kitchen, with white cabinets and a "
            "wooden table behind her; warm window light casts soft shadows across the room."
        )
        calls = []

        def fake_generate(**kwargs):
            calls.append(kwargs)
            if "visual continuity observer" in kwargs.get("system_prompt", "").casefold():
                return visual_anchor
            return weak

        with (
            mock.patch.object(llm_service, "generate", side_effect=fake_generate),
            mock.patch(
                "services.enhance_guides.get_enhance_guide",
                return_value="Write the required MiniMax H3 three-field prompt.",
            ),
        ):
            enhanced = llm_service.enhance_prompt(
                prompt,
                mode="video",
                model_type="minimax_h3",
                image_paths=["frame.png"],
                duration_seconds=8,
            )

        self.assertIn("<d>[French] Bonjour, comment ça va ?</d>", enhanced)
        self.assertNotIn("<d>[English] Bonjour, comment ça va ?</d>", enhanced)
        self.assertEqual(enhanced.count("Bonjour, comment ça va ?"), 1)
        self.assertIn("Attached-frame visual evidence", enhanced)
        self.assertIn("shoulder-length auburn hair", enhanced)
        self.assertEqual(len(calls), 3)  # initial rewrite, one repair, one vision grounding pass
        self.assertEqual(calls[-1]["image_paths"], ["frame.png"])

    def test_optional_visual_grounding_failure_preserves_valid_main_result(self):
        weak = (
            "integrated_multimodal_description: [Shot 1] A person waves.\n"
            "overall_soundscape: Room tone.\n"
            "non_diegetic_music: N/A"
        )

        def unavailable(**_kwargs):
            raise RuntimeError("vision backend unavailable")

        self.assertEqual(
            llm_service._ensure_h3_visual_grounding(
                weak,
                "A person waves.",
                ["frame.png"],
                generate_fn=unavailable,
            ),
            weak,
        )


class TestGemmaDiagnostics(unittest.TestCase):
    def test_template_compat_notice_is_not_reported_as_crash_cause(self):
        previous = list(llm_service._server_log)
        try:
            llm_service._server_log.clear()
            warning = (
                "common_chat_try_specialized_template: detected an outdated gemma4 chat "
                "template, applying compatibility workarounds."
            )
            llm_service._server_log.extend([warning, "fatal: CUDA allocation failed"])
            tail = llm_service._server_log_tail(20)
            self.assertNotIn("outdated gemma4", tail)
            self.assertIn("fatal: CUDA allocation failed", tail)
            self.assertTrue(llm_service._is_benign_gemma_template_warning(warning))
        finally:
            llm_service._server_log.clear()
            llm_service._server_log.extend(previous)


if __name__ == "__main__":
    unittest.main()
