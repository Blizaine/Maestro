"""CPU-only tests for animated title planning and its interactive API wiring."""
import ast
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from functools import wraps
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.animated_title_planner import (  # noqa: E402
    MAX_DRAFT_TOKENS,
    TITLE_PLAN_JSON_SCHEMA,
    NATIVE_FRAME_LENGTHS,
    normalize_title_draft,
    plan_animated_title,
    validate_title_request,
)


class TestAnimatedTitlePlanner(unittest.TestCase):
    def test_preserves_ordered_international_quoted_copy(self):
        brief = 'First “京都の朝”, then “Café solaire” in that order.'
        raw = "```json\n" + json.dumps({
            "phrases": [
                {"text": "京都の朝", "animation": "Ink stroke reveal"},
                {"text": "Café solaire", "animation": "Soft glow"},
            ],
            "sequence": True,
            "look": "Hand painted anime typography",
            "direction": "Centered, then drift upward",
        }, ensure_ascii=False) + "\n```"

        result = normalize_title_draft(raw, brief=brief)

        self.assertTrue(result["plan"]["sequence"])
        self.assertEqual(
            [phrase["text"] for phrase in result["plan"]["phrases"]],
            ["京都の朝", "Café solaire"],
        )
        self.assertFalse(result["review_required"])
        self.assertEqual(result["raw_draft"], raw)

    def test_quote_mismatch_keeps_draft_words_and_requests_review(self):
        raw = json.dumps({"title": "Good morning, Kyoto", "motion": "Fade in"})

        result = normalize_title_draft(raw, brief='The words must be “おはよう京都”.')

        self.assertEqual(result["plan"]["phrases"][0]["text"], "Good morning, Kyoto")
        self.assertTrue(result["review_required"])
        self.assertTrue(any("quoted phrases" in warning for warning in result["warnings"]))

    def test_missing_look_uses_generic_look_and_scopes_fire_hint_per_sequence_phrase(self):
        fire_hint = (
            "Sparks trace and reveal the title with fiery edges. Hold the complete lettering clearly, "
            "then let the embers fade and clear the frame."
        )
        raw = json.dumps({
            "phrases": [{"text": "Première"}, {"text": "Aube"}],
            "sequence": True,
            "direction": "Show each phrase in order, then clear the frame.",
        }, ensure_ascii=False)

        result = normalize_title_draft(
            raw,
            brief='Show “Première”, then “Aube”.',
            style_hint=fire_hint,
        )

        self.assertEqual(result["plan"]["look"], "Polished, legible animated typography.")
        self.assertNotIn("title", result["plan"]["look"].lower())
        self.assertEqual([phrase["text"] for phrase in result["plan"]["phrases"]], ["Première", "Aube"])
        for phrase in result["plan"]["phrases"]:
            self.assertIn("current phrase", phrase["animation"])
            self.assertIn("this phrase", phrase["animation"])
            self.assertNotIn("complete title", phrase["animation"].lower())
        self.assertFalse(result["review_required"])

    def test_quoted_copy_additions_order_case_and_punctuation_need_review(self):
        cases = (
            ('Animate “HELLO”.', [{"text": "HELLO WORLD"}], False),
            ('Animate “HELLO”.', [{"text": "hello"}], False),
            ('Animate “HELLO!”.', [{"text": "HELLO"}], False),
            ('First “A”, then “B”.', [{"text": "B"}, {"text": "A"}], True),
        )
        for brief, phrases, sequence in cases:
            with self.subTest(brief=brief, phrases=phrases):
                raw = json.dumps({"phrases": phrases, "sequence": sequence})
                result = normalize_title_draft(raw, brief=brief)
                self.assertIsNotNone(result["plan"])
                self.assertEqual(result["plan"]["phrases"][0]["text"], phrases[0]["text"])
                self.assertTrue(result["review_required"])

    def test_phrase_weights_must_be_positive_and_finite(self):
        for weight in (0, -1, float("inf"), float("nan")):
            with self.subTest(weight=weight):
                result = normalize_title_draft(
                    json.dumps({"text": "Title", "weight": weight}),
                    brief="A title",
                )
                phrase = result["plan"]["phrases"][0]
                self.assertEqual(phrase["weight"], 1.0)
                self.assertTrue(any("positive finite weight" in warning for warning in result["warnings"]))

    def test_display_field_limits_return_raw_draft_without_truncation(self):
        cases = (
            {"look": "L" * 1201},
            {"direction": "D" * 1201},
            {"phrases": [{"text": "Title", "animation": "A" * 801}]},
        )
        for override in cases:
            with self.subTest(field=next(iter(override))):
                payload = {"phrases": [{"text": "Title", "animation": "Fade"}],
                           "look": "Clear lettering", "direction": "Centered", **override}
                raw = json.dumps(payload)
                result = normalize_title_draft(raw, brief="A title")
                self.assertIsNone(result["plan"])
                self.assertTrue(result["review_required"])
                self.assertEqual(result["raw_draft"], raw)
                self.assertIn("exceeded", result["warnings"][0])

    def test_sequence_limit_includes_newline_separator_used_by_editor(self):
        raw = json.dumps({"phrases": [{"text": "A" * 80}, {"text": "B" * 80}]})

        result = normalize_title_draft(raw, brief="A sequence")

        self.assertIsNone(result["plan"])
        self.assertIn("160 total title characters", result["warnings"][0])

    def test_single_title_aliases_and_missing_optional_fields_use_defaults(self):
        result = normalize_title_draft(
            json.dumps({"title": "René & 朝", "motion": "Wipe on"}),
            brief="A title in French and Japanese.",
            style_hint="Flat anime lettering",
        )

        self.assertEqual(result["plan"]["phrases"], [{
            "text": "René & 朝",
            "animation": "Wipe on",
            "weight": 1.0,
        }])
        self.assertEqual(result["plan"]["look"], "Polished, legible animated typography.")
        self.assertEqual(result["plan"]["direction"], "Keep the title legible and inside the frame.")
        self.assertFalse(result["plan"]["sequence"])
        self.assertFalse(result["review_required"])

    def test_items_and_beat_motion_aliases_infer_sequence_order(self):
        result = normalize_title_draft(json.dumps({
            "items": [
                {"beat": "First light", "motion": "Reveal from the left"},
                {"title": "New day", "motion": "Settle below"},
            ],
            "sequence": False,
        }), brief="A title that changes from First light to New day.")

        self.assertEqual([item["text"] for item in result["plan"]["phrases"]], ["First light", "New day"])
        self.assertTrue(result["plan"]["sequence"])
        self.assertTrue(result["review_required"])
        self.assertTrue(result["warnings"])

    def test_invalid_json_returns_raw_draft_for_review(self):
        raw = '{"phrases":[{"text":"unfinished"}'

        result = normalize_title_draft(raw, brief="A title")

        self.assertIsNone(result["plan"])
        self.assertTrue(result["review_required"])
        self.assertEqual(result["raw_draft"], raw)
        self.assertIn("complete JSON", result["warnings"][0])

    def test_oversized_title_is_never_truncated(self):
        title = "日" * 161
        raw = json.dumps({"text": title}, ensure_ascii=False)

        result = normalize_title_draft(raw, brief="Use a long title")

        self.assertIsNone(result["plan"])
        self.assertIn("no title text was truncated", result["warnings"][0])
        self.assertEqual(result["raw_draft"], raw)
        self.assertIn(title, result["raw_draft"])

    def test_oversized_sequence_phrase_is_not_truncated(self):
        title = "a" * 81
        raw = json.dumps({"phrases": [{"text": title}, {"text": "Last"}]})

        result = normalize_title_draft(raw, brief="Two title beats")

        self.assertIsNone(result["plan"])
        self.assertIn("sequence phrase exceeded 80", result["warnings"][0])
        self.assertEqual(result["raw_draft"], raw)

    def test_oversized_raw_draft_is_preserved_for_review(self):
        raw = "{" + (" " * 40000)

        result = normalize_title_draft(raw, brief="A title")

        self.assertIsNone(result["plan"])
        self.assertIn("oversized", result["warnings"][0])
        self.assertEqual(result["raw_draft"], raw)

    def test_writer_is_called_once_with_bounded_json_guidance(self):
        calls = []

        def writer(**kwargs):
            calls.append(kwargs)
            return json.dumps({"title": "Sunrise", "motion": "Reveal"})

        result = plan_animated_title(
            "An anime title for sunrise.",
            "hand-drawn cel animation",
            158,
            "black",
            writer,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["max_new_tokens"], MAX_DRAFT_TOKENS)
        self.assertEqual(calls[0]["json_schema"], TITLE_PLAN_JSON_SCHEMA)
        self.assertEqual(TITLE_PLAN_JSON_SCHEMA["required"], ["phrases", "look", "direction", "sequence"])
        self.assertFalse(TITLE_PLAN_JSON_SCHEMA["additionalProperties"])
        self.assertEqual(
            TITLE_PLAN_JSON_SCHEMA["properties"]["phrases"]["items"]["required"],
            ["text", "animation", "weight"],
        )
        self.assertFalse(calls[0]["enable_thinking"])
        self.assertIn("hand-drawn cel animation", calls[0]["prompt"])
        self.assertIn("158 frames", calls[0]["prompt"])
        self.assertIn("24 fps", calls[0]["prompt"])
        self.assertIn("6.583 seconds", calls[0]["prompt"])
        self.assertIn("solid black", calls[0]["prompt"])
        self.assertTrue(result["plan"])

    def test_invalid_native_duration_is_rejected_without_writer_call(self):
        writer = Mock(return_value='{"title":"Sunrise"}')

        for frames in (123, 125, 346, True, "141"):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                plan_animated_title("A title", "", frames, "chroma", writer)

        self.assertEqual(NATIVE_FRAME_LENGTHS[0], 124)
        self.assertEqual(NATIVE_FRAME_LENGTHS[-1], 345)
        self.assertEqual(len(NATIVE_FRAME_LENGTHS), 14)
        writer.assert_not_called()

    def test_request_limits_and_background_are_validated(self):
        validate_title_request("x" * 4000, "y" * 1600, 345, "chroma")
        for args in (
            ("x" * 4001, "", 124, "black"),
            ("brief", "y" * 1601, 124, "black"),
            ("brief", "", 124, "green"),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_title_request(*args)


class TestAnimatedTitleApiWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT / "app" / "launch.py"
        selected = []
        for node in ast.parse(source.read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
                "_interactive_enhancement_slot",
                "llm_plan_animated_title",
            }:
                node.decorator_list = []
                selected.append(node)
        cls.code = compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec")

    def test_endpoint_uses_enhancer_model_and_interactive_guard(self):
        from services.studio_enhancement import captured_settings, current_settings, enhancement_context

        with tempfile.TemporaryDirectory() as directory:
            config = {
                "llm_provider": "remote",
                "llm_model_id": "default-writer",
                "llm_remote_url": "https://writer.invalid/v1",
                "enhance_llm_model_id": "configured-title-writer",
                "enhance_llm_device": "cpu",
            }
            wgp = SimpleNamespace(
                server_config={"services": config},
                wan_model=None,
                offloadobj=None,
                release_model=Mock(),
            )
            loader = Mock()
            guard = Mock()
            namespace = {
                "wraps": wraps,
                "Request": object,
                "asyncio": asyncio,
                "Path": Path,
                "uuid": uuid,
                "time": time,
                "HTTPException": HTTPException,
                "_app_dir": directory,
                "_gen_lock": threading.Lock(),
                "_guard_interactive_llm_against_generation": guard,
                "_enhancement_settings_snapshot": lambda: captured_settings(config),
                "enhancement_context": enhancement_context,
                "enhancement_settings": current_settings,
                "wgp": wgp,
                "_ensure_llm_loaded": loader,
            }
            exec(self.code, namespace)
            endpoint = namespace["_interactive_enhancement_slot"](namespace["llm_plan_animated_title"])
            request_body = {
                "brief": "An animated title that reads ‘À bientôt’.",
                "style_hint": "Anime lettering",
                "frames": 141,
                "background": "black",
            }

            class Request:
                async def json(self):
                    return request_body

            raw = json.dumps({"text": "À bientôt", "motion": "Fade up"}, ensure_ascii=False)
            from services import llm_service
            with patch.object(llm_service, "generate", return_value=raw) as writer:
                result = asyncio.run(endpoint(Request()))

            guard.assert_called_once_with()
            loader.assert_called_once_with(
                model_id="configured-title-writer",
                device="cpu",
            )
            writer.assert_called_once()
            self.assertEqual(writer.call_args.kwargs["max_new_tokens"], MAX_DRAFT_TOKENS)
            self.assertEqual(writer.call_args.kwargs["json_schema"], TITLE_PLAN_JSON_SCHEMA)
            self.assertEqual(result["plan"]["phrases"][0]["text"], "À bientôt")
            self.assertTrue(result["draft_history_id"])
            self.assertEqual(result["raw_draft"], raw)

    def test_endpoint_rejects_invalid_duration_before_loading_writer(self):
        namespace = {
            "wraps": wraps,
            "Request": object,
            "asyncio": asyncio,
            "Path": Path,
            "uuid": uuid,
            "time": time,
            "HTTPException": HTTPException,
            "_app_dir": tempfile.gettempdir(),
            "_gen_lock": threading.Lock(),
            "_guard_interactive_llm_against_generation": lambda: None,
            "_enhancement_settings_snapshot": lambda: {},
            "enhancement_context": __import__("services.studio_enhancement", fromlist=["enhancement_context"]).enhancement_context,
            "wgp": SimpleNamespace(wan_model=None, offloadobj=None, server_config={"services": {}}),
            "enhancement_settings": lambda value: value,
            "_ensure_llm_loaded": Mock(),
        }
        exec(self.code, namespace)
        endpoint = namespace["_interactive_enhancement_slot"](namespace["llm_plan_animated_title"])

        class Request:
            async def json(self):
                return {"brief": "A title", "style_hint": "", "frames": 125, "background": "black"}

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(endpoint(Request()))

        self.assertEqual(raised.exception.status_code, 400)
        namespace["_ensure_llm_loaded"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
