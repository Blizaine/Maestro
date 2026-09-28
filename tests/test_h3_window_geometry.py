"""CPU-only regressions for issue #160: a rolling total is not a native pass."""

import ast
import asyncio
import copy
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.minimax_h3.minimax_h3_handler import (
    apply_h3_window_memory_policy,
    family_handler,
    normalize_h3_window_geometry,
)
from services.h3_window_planner import (
    compute_h3_window_boundaries,
    reviewed_h3_window_plan_matches,
    reviewed_h3_window_plan_mismatch,
)


MODEL = {"architecture": "minimax_h3", "frames_maximum": 345}


def request(**changes):
    return {
        "model_type": "minimax_h3_fused_turbo",
        "resolution": "864x480",
        "video_length": 336,
        "sliding_window_size": 124,
        "sliding_window_overlap": 18,
        "minimax_h3_multi_window": True,
        "sliding_window_memory_override": True,
        **changes,
    }


def spans(inputs):
    return compute_h3_window_boundaries(
        inputs["video_length"], inputs["sliding_window_size"], fps=24,
        overlap_frames=inputs["sliding_window_overlap"],
        discard_frames=inputs.get("sliding_window_discard_last_frames", 0),
    )


class H3WindowGeometryTests(unittest.TestCase):
    def test_three_short_windows_stay_three_through_runtime_and_saved_settings(self):
        settings = request()
        normalize_h3_window_geometry(settings, MODEL)
        expected = spans(settings)
        self.assertEqual([(b["start_frame"], b["end_frame"]) for b in expected],
                         [(0, 124), (124, 230), (230, 336)])
        with patch("torch.cuda.is_available", return_value=False):
            self.assertIsNone(family_handler.validate_generative_settings("minimax_h3", MODEL, settings))
        self.assertEqual(spans(settings), expected)
        family_handler.fix_settings("minimax_h3", 2.58, MODEL, settings)
        self.assertEqual(settings["video_length"], 336)
        self.assertEqual(spans(settings), expected)

    def test_native_passes_still_snap_but_joined_totals_do_not(self):
        for total, window, enabled, expected in (
            (120, 124, False, 124), (336, 345, False, 345),
            (336, 124, False, 124), (120, 124, True, 124),
            (230, 124, True, 230), (336, 124, True, 336),
            (442, 124, True, 442), (720, 345, True, 720),
        ):
            with self.subTest(total=total, window=window, enabled=enabled):
                settings = request(video_length=total, sliding_window_size=window,
                                   minimax_h3_multi_window=enabled)
                normalize_h3_window_geometry(settings, MODEL)
                self.assertEqual(settings["video_length"], expected)
                normalized = copy.deepcopy(settings)
                normalize_h3_window_geometry(settings, MODEL)
                self.assertEqual(settings, normalized)

    def test_overlap_and_clean_tail_use_runtime_rules(self):
        for raw, expected in ((0, 1), (17, 18), (18, 18), (200, 103)):
            with self.subTest(overlap=raw):
                settings = request(sliding_window_overlap=raw,
                                   custom_settings={"h3_long_sequence_clean_tail": True})
                normalize_h3_window_geometry(settings, MODEL)
                self.assertEqual(settings["sliding_window_overlap"], expected)
                self.assertEqual(settings["sliding_window_discard_last_frames"], 17)
                before = spans(settings)
                with patch("torch.cuda.is_available", return_value=False):
                    self.assertIsNone(family_handler.validate_generative_settings("minimax_h3", MODEL, settings))
                self.assertEqual(spans(settings), before)

        clean_tail_spans = compute_h3_window_boundaries(
            213, 124, overlap_frames=18, discard_frames=17,
        )
        self.assertEqual([(b["start_frame"], b["end_frame"]) for b in clean_tail_spans],
                         [(0, 107), (107, 196), (196, 213)])

    def test_outpaint_exact_timeline_is_preserved(self):
        settings = request(video_length=230, sliding_window_size=345,
                           video_guide_outpainting="1,0,0,0")
        normalize_h3_window_geometry(settings, MODEL)
        self.assertEqual(settings["video_length"], 230)

    def test_other_h3_workflows_keep_their_own_timing_rules(self):
        for flag in ("omni_reference", "audio_only", "minimax_h3_viggle"):
            with self.subTest(workflow=flag):
                settings = request(video_length=57, sliding_window_overlap=0)
                before = copy.deepcopy(settings)
                normalize_h3_window_geometry(settings, {**MODEL, flag: True})
                self.assertEqual(settings, before)

    def test_memory_limited_window_is_chosen_before_rounding_the_total(self):
        settings = request(sliding_window_size=345, sliding_window_memory_override=False,
                           resolution="1792x768")
        planned = copy.deepcopy(settings)
        apply_h3_window_memory_policy(planned, MODEL, {"gpu_vram_gb": 12})
        normalize_h3_window_geometry(planned, MODEL)
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties", return_value=types.SimpleNamespace(total_memory=12 * 1024 ** 3)):
            self.assertIsNone(family_handler.validate_generative_settings("minimax_h3", MODEL, settings))
        self.assertEqual(settings["video_length"], 336)
        self.assertEqual(spans(settings), spans(planned))

    def test_reviewed_prompts_survive_and_mismatches_explain_the_change(self):
        settings = request()
        normalize_h3_window_geometry(settings, MODEL)
        boundaries = spans(settings)
        prompts = ['Alice says "Wait."', 'Bob replies "Okay."', "They leave."]
        plan = {
            "model_type": settings["model_type"], "resolution": settings["resolution"],
            "source_prompt": "A short conversation.", "window_frames": 124,
            "windows": [{**b, "prompt": p} for b, p in zip(boundaries, prompts)],
        }
        args = dict(source_prompt=plan["source_prompt"], model_type=plan["model_type"],
                    resolution=plan["resolution"], window_frames=124, boundaries=boundaries)
        self.assertTrue(reviewed_h3_window_plan_matches(plan, prompts, **args))
        rounded = {**settings, "video_length": 345}
        self.assertIn("require 4 windows", reviewed_h3_window_plan_mismatch(
            plan, prompts, **{**args, "boundaries": spans(rounded)}))
        self.assertEqual(reviewed_h3_window_plan_mismatch(
            plan, prompts, **{**args, "source_prompt": "A changed story."}), "the source prompt changed")
        self.assertIn("invalid", reviewed_h3_window_plan_mismatch(
            {**plan, "window_frames": "bad"}, prompts, **args))
        self.assertEqual([w["prompt"] for w in plan["windows"]], prompts)

    def test_planning_endpoint_passes_effective_geometry_to_the_writer(self):
        # Execute the real endpoint without importing/starting the application
        # or invoking a writer. This catches drift in request serialization.
        path = ROOT / "app" / "launch.py"
        node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.AsyncFunctionDef) and n.name == "llm_plan_h3_windows")
        node.decorator_list = []
        namespace = dict(
            Request=object, asyncio=asyncio, os=os,
            wgp=types.SimpleNamespace(get_model_def=lambda _: MODEL, server_config={}),
            _get_cached_hardware=lambda: {}, _ensure_llm_loaded=lambda: None,
            enhancement_settings=lambda _: {}, _PUBLIC_LLM_PROVIDERS=(),
            _h3_injected_keyframes_from_body=lambda _: [], _active_lora_hint=lambda *_: "",
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        async def body():
            return {"prompt": "A silent walk.", "model_type": "minimax_h3_fused_turbo",
                    "total_frames": 336, "window_frames": 124, "overlap_frames": 17,
                    "custom_settings": {"h3_long_sequence_clean_tail": True}}
        with patch.dict(sys.modules, {"services.llm_service": types.ModuleType("services.llm_service")}), \
             patch("services.h3_window_planner.plan_h3_sliding_windows", return_value={}) as writer:
            result = asyncio.run(namespace["llm_plan_h3_windows"](types.SimpleNamespace(json=body)))
        self.assertEqual(result["effective_window_frames"], 124)
        self.assertEqual(writer.call_args.kwargs["total_frames"], 336)
        self.assertEqual(writer.call_args.kwargs["overlap_frames"], 18)
        self.assertEqual(writer.call_args.kwargs["discard_frames"], 17)


if __name__ == "__main__":
    unittest.main()
