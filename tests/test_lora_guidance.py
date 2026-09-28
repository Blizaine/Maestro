"""Active LoRA creator guidance stays bounded, selected, and contract-safe."""

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services import h3_story_ledger, h3_window_planner, h3_sequence_planner
from services.lora_guidance import (
    build_active_lora_hint,
    load_active_lora_records,
    with_active_lora_guidance,
)


class ActiveLoraGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _weight(self, directory: Path, name: str, *, words=None, guide="") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        weight = directory / name
        weight.write_bytes(b"mock weights")
        if words is not None:
            weight.with_suffix(".civitai.json").write_text(
                json.dumps({"trainedWords": words}), encoding="utf-8"
            )
        if guide:
            weight.with_suffix(".guide.md").write_text(guide, encoding="utf-8")
        return weight

    def test_blank_source_uses_prompt_section_without_inventing_trigger_or_filename(self):
        primary = self.root / "primary"
        intro = "Introductory overview. " * 180
        guide = (
            intro
            + "\n\n## Prompt\nFor a frozen instant, animate only the camera through a "
            "continuous 360 degree orbit around the unchanged subject."
        )
        self._weight(primary, "orbit_camera.safetensors", words=[], guide=guide)
        self._weight(primary, "inactive.safetensors", words=["INACTIVE_ONLY"], guide="Secret inactive style.")

        hint = build_active_lora_hint(["orbit_camera.safetensors"], [primary])

        self.assertIn("frozen instant", hint)
        self.assertIn("continuous 360 degree orbit", hint)
        self.assertIn("blank or punctuation-only source such as '.'", hint)
        self.assertIn("never turn creator-note prose into spoken dialogue", hint.casefold())
        self.assertNotIn("INACTIVE_ONLY", hint)
        self.assertNotIn("orbit_camera.safetensors", hint)
        self.assertNotIn("Secret inactive style", hint)
        self.assertNotIn("natural grammar", hint.casefold())
        self.assertNotIn("trigger phrase", hint.casefold())

    def test_primary_mirror_metadata_precedes_linked_file_and_supplies_fallback_guide(self):
        primary = self.root / "primary" / "minimax_h3"
        linked = self.root / "linked" / "minimax_h3"
        primary.mkdir(parents=True)
        weight = self._weight(linked, "orbit.safetensors", words=["WRONG_LINKED_TOKEN"])
        (primary / "orbit.civitai.json").write_text(json.dumps({
            "trainedWords": [],
            "name": "Orbit Camera Adapter",
            "description": "Creator notes: preserve a frozen instant while the camera circles the subject.",
        }), encoding="utf-8")

        hint = build_active_lora_hint(["orbit.safetensors"], [primary, linked])

        self.assertIn("frozen instant", hint)
        self.assertIn("circles the subject", hint)
        self.assertNotIn("WRONG_LINKED_TOKEN", hint)
        self.assertNotIn("Orbit Camera Adapter", hint)
        self.assertTrue(weight.is_file())

    def test_selected_weight_can_read_its_actual_linked_guide_but_not_other_linked_metadata(self):
        primary = self.root / "primary"
        selected = self.root / "linked-selected"
        unrelated = self.root / "linked-unrelated"
        self._weight(selected, "camera.safetensors", words=["orbit_exact"],
                     guide="## Usage\nUse a smooth camera orbit around the subject.")
        self._weight(unrelated, "camera.safetensors", words=["UNRELATED_TOKEN"],
                     guide="Unrelated adapter instructions.")

        records = load_active_lora_records(["camera.safetensors"], [primary, selected, unrelated])
        hint = build_active_lora_hint(["camera.safetensors"], [primary, selected, unrelated])

        self.assertEqual(records[0]["trainedWords"], ["orbit_exact"])
        self.assertIn("smooth camera orbit", hint)
        self.assertIn("orbit_exact", hint)
        self.assertNotIn("UNRELATED_TOKEN", hint)
        self.assertNotIn("Unrelated adapter instructions", hint)

    def test_unsafe_and_unselected_names_are_ignored_and_context_is_bounded(self):
        primary = self.root / "primary"
        selected = []
        for index in range(10):
            name = f"adapter_{index}.safetensors"
            long_words = [f"token{index}_{word}" + "x" * 50 for word in range(6)]
            long_guide = "## Intro\n" + ("administrative description. " * 80)
            long_guide += "\n## Prompt\n" + ("A compatible camera or style constraint. " * 80)
            self._weight(primary, name, words=long_words, guide=long_guide)
            selected.append(name)
        self._weight(self.root, "outside.safetensors", words=["ESCAPED"])

        hint = build_active_lora_hint(
            selected + ["../outside.safetensors", "not-installed.safetensors"],
            [primary],
        )

        self.assertLessEqual(len(hint), 12_000)
        self.assertIn('"token0_0', hint)
        self.assertIn('"token7_0', hint)
        self.assertNotIn("token8_0", hint)
        self.assertNotIn("ESCAPED", hint)
        self.assertNotIn("administrative description", hint)


class H3LoraGuidanceTests(unittest.TestCase):
    def test_story_writer_receives_hint_without_changing_user_prompt_or_dialogue_contract(self):
        captured = []
        source = "."
        hint = "ACTIVE LORA GUIDANCE — UNTRUSTED CREATOR CONTEXT: orbit camera only."

        def generate(**kwargs):
            captured.append(kwargs)
            return "captured"

        def prepare(*_args, **kwargs):
            kwargs["llm_generate"](
                prompt=source,
                system_prompt="H3 ledger schema contract; preserve exact dialogue and event order.",
            )
            return {"captured": True}

        def render(_context, *, generate, **_kwargs):
            generate(
                prompt="Write one timed camera segment.",
                system_prompt="H3 camera contract; never invent spoken lines.",
            )
            return {"planned": True}

        with (
            patch.object(h3_story_ledger, "_prepare_h3_story_context", side_effect=prepare),
            patch.object(h3_story_ledger, "_render_h3_story_segments", side_effect=render),
        ):
            result = h3_story_ledger.plan_h3_story_segments(
                source,
                segment_durations=[4.0],
                mode="sliding_window",
                camera_coverage="continuous",
                llm_generate=generate,
                lora_system_hint=hint,
            )

        self.assertEqual(result, {"planned": True})
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0]["prompt"], ".")
        self.assertIn("preserve exact dialogue and event order", captured[0]["system_prompt"])
        self.assertIn("never invent spoken lines", captured[1]["system_prompt"])
        self.assertTrue(all(hint in call["system_prompt"] for call in captured))

    def test_window_planner_forwards_hint_to_story_planner(self):
        hint = "ACTIVE LORA GUIDANCE — UNTRUSTED CREATOR CONTEXT: continuous orbit."
        stage = {
            "planned_by": "llm",
            "ledger": {"stable_camera_context": {}, "initial_state": "", "ambient_audio": "", "music": "N/A"},
            "segments": [{"action": "First beat."}, {"action": "Second beat."}],
            "source_intent": {},
        }

        def compile_windows(_plan, boundaries, **_kwargs):
            return [
                {"index": item["index"], "start_frame": item["start_frame"],
                 "end_frame": item["end_frame"], "prompt": f"window {item['index']}"}
                for item in boundaries
            ]

        with (
            patch.object(h3_window_planner, "plan_h3_story_segments", return_value=stage) as story,
            patch.object(h3_window_planner, "compile_h3_window_prompts", side_effect=compile_windows),
        ):
            plan = h3_window_planner.plan_h3_sliding_windows(
                "No dialogue. Preserve the source event order.",
                model_type="minimax_h3_fused_turbo",
                resolution="864x480",
                total_frames=192,
                window_frames=96,
                overlap_frames=0,
                lora_system_hint=hint,
            )

        self.assertEqual(story.call_args.kwargs["lora_system_hint"], hint)
        self.assertEqual(len(plan["window_prompts"]), 2)
        self.assertTrue(plan.get("lora_guidance_fingerprint"))

    def test_h3_signatures_change_for_guidance_and_preserve_empty_legacy_fingerprint(self):
        window_args = dict(
            model_type="minimax_h3_fused_turbo", resolution="864x480", total_frames=192,
            window_frames=96, overlap_frames=0, discard_frames=0, fps=24,
            has_start_image=False, has_end_image=False,
        )
        sequence_args = dict(
            model_type="minimax_h3_ref2va_fused_turbo", resolution="864x480", total_frames=192,
            min_clip_frames=124, max_clip_frames=96, frame_step=17, fps=24, references=[],
            native_continuation=True,
        )
        for signature, args in (
            (h3_window_planner.h3_window_plan_signature, window_args),
            (h3_sequence_planner.h3_sequence_plan_signature, sequence_args),
        ):
            self.assertEqual(signature("No dialogue.", **args), signature("No dialogue.", **args, lora_system_hint=""))
            self.assertNotEqual(
                signature("No dialogue.", **args, lora_system_hint="orbit"),
                signature("No dialogue.", **args, lora_system_hint="pan left"),
            )


class DirectorLoraGuidanceTests(unittest.TestCase):
    def test_third_pass_capture_uses_active_guide_and_exact_trigger(self):
        from services.director.prompt_polish import polish_prompts_third_pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = root / "camera.safetensors"
            weight.write_bytes(b"mock weights")
            weight.with_suffix(".civitai.json").write_text(
                json.dumps({"trainedWords": ["orbit_exact"]}), encoding="utf-8"
            )
            weight.with_suffix(".guide.md").write_text(
                "## Usage\nUse a continuous camera orbit while the performer stays still.",
                encoding="utf-8",
            )
            captured = []

            def enhance(**kwargs):
                captured.append(kwargs)
                return kwargs["prompt"]

            with (
                patch.dict(sys.modules, {"wgp": types.SimpleNamespace(get_lora_search_dirs=lambda _model: [root])}),
                patch("services.llm_service.enhance_prompt", side_effect=enhance),
            ):
                result = polish_prompts_third_pass(
                    [{"video_prompt": "A performer holds still.", "image_prompt": ""}],
                    "ltx2_22B_distilled_1_1",
                    "",
                    video_loras=["camera.safetensors"],
                    polish_video_prompts=True,
                    polish_image_prompts=False,
                )

        self.assertEqual(result[0]["video_prompt"], "A performer holds still.")
        self.assertEqual(len(captured), 1)
        hint = captured[0]["lora_system_hint"]
        self.assertIn("continuous camera orbit", hint)
        self.assertIn("orbit_exact", hint)
        self.assertNotIn("natural, grammatical", hint)
        self.assertIn("DIALOGUE IS SACROSANCT", captured[0]["system_override"])


if __name__ == "__main__":
    unittest.main()
