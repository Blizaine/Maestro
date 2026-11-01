"""CPU regressions for independent Director clips routed to Pika H3 Flex."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.director_video_strategy import (  # noqa: E402
    advance_director_multiclip_audio_offset,
    apply_h3_clip_drive_audio,
    pika_flex_local_window_start_frame,
    should_continue_multi_clip_task,
)


LAUNCH = APP / "launch.py"
WGP = APP / "wgp.py"
H3_MAIN = APP / "models" / "minimax_h3" / "minimax_h3_main.py"


class PikaDirectorMulticlipTests(unittest.TestCase):
    def test_independent_reference_shots_keep_source_offsets_without_history(self):
        # Native durations can include conditioning tails. Source audio time
        # advances by the committed output duration after each clip is trimmed.
        native_frames = [124, 141, 158]
        trim_frames = [0, 4, 6]
        offsets = []
        cumulative_offset = 0
        for index, (frames, trim) in enumerate(zip(native_frames, trim_frames)):
            offsets.append(cumulative_offset)
            self.assertFalse(
                should_continue_multi_clip_task(
                    clip_index=index,
                    has_start=False,
                    bounded_director=True,
                    per_clip_continuations=[False, False, False],
                ),
                "independent bounded Director cuts must not carry generated history",
            )
            cumulative_offset = advance_director_multiclip_audio_offset(
                cumulative_offset,
                frames,
                trim,
            )

        self.assertEqual(offsets, [0, 124, 261])
        self.assertEqual(cumulative_offset, 413)
        self.assertEqual(
            offsets,
            [0, 124, 124 + (141 - 4)],
            "each source-audio slice starts after the prior clip's output trim",
        )

    def test_explicit_same_scene_continuation_still_routes(self):
        self.assertTrue(
            should_continue_multi_clip_task(
                clip_index=1,
                has_start=False,
                bounded_director=True,
                per_clip_continuations=[False, True],
            )
        )
        self.assertFalse(
            should_continue_multi_clip_task(
                clip_index=1,
                has_start=True,
                bounded_director=True,
                per_clip_continuations=[False, True],
            ),
            "a fresh start image already anchors an independent task",
        )

    def test_exact_drive_mode_survives_global_and_per_clip_audio_routes(self):
        source_audio = {
            "audio_prompt_type": "AD",
            "audio_guide": "full-song.wav",
            "audio_frame_offset": 124,
        }
        self.assertFalse(apply_h3_clip_drive_audio(source_audio, None, None))
        self.assertEqual(source_audio["audio_prompt_type"], "AD")
        self.assertEqual(source_audio["audio_guide"], "full-song.wav")
        self.assertEqual(source_audio["audio_frame_offset"], 124)

        clip_audio = {"audio_prompt_type": "A", "audio_guide": "old-audio.wav"}
        self.assertTrue(
            apply_h3_clip_drive_audio(clip_audio, "scene-two.wav", 2)
        )
        self.assertEqual(clip_audio["audio_prompt_type"], "AD")
        self.assertEqual(clip_audio["audio_guide"], "scene-two.wav")
        self.assertEqual(clip_audio["minimax_h3_exact_drive_audio_ordinal"], 2)

    def test_pika_cursor_is_local_and_requires_a_real_nonnegative_frame(self):
        self.assertEqual(pika_flex_local_window_start_frame(0), 0)
        self.assertEqual(pika_flex_local_window_start_frame(106), 106)
        for invalid in (-1, 1.5, float("nan"), True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                pika_flex_local_window_start_frame(invalid)

    def test_launch_multiclip_branch_uses_the_tested_route_helpers(self):
        tree = ast.parse(LAUNCH.read_text(encoding="utf-8"))
        generation = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_generation"
        )
        branch = next(
            node for node in ast.walk(generation)
            if isinstance(node, ast.If)
            and ast.unparse(node.test)
            == "raw_params.get('multi_prompts_gen_type') == 3"
        )
        calls = {
            node.func.id
            for node in ast.walk(branch)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertTrue({
            "should_continue_multi_clip_task",
            "advance_director_multiclip_audio_offset",
            "apply_h3_clip_drive_audio",
        }.issubset(calls))
        audio_offset_assignment = next(
            node for node in ast.walk(branch)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "clip_params"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "audio_frame_offset"
                for target in node.targets
            )
        )
        self.assertEqual(ast.unparse(audio_offset_assignment.value), "clip_audio_frame_offset")

    def test_wgp_sends_local_pika_cursor_without_changing_reference_timeline(self):
        tree = ast.parse(WGP.read_text(encoding="utf-8"))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "call_with_sticky_interrupt"
        ]
        call = next(
            node for node in calls
            if any(keyword.arg == "window_start_frame_no" for keyword in node.keywords)
        )
        global_origin = next(
            keyword.value for keyword in call.keywords
            if keyword.arg == "window_start_frame_no"
        )
        self.assertIn("reference_start_frame", ast.unparse(global_origin))

        pika_options = next(
            node.value for node in call.keywords
            if node.arg is None
            and "pika_local_window_start_frame_no" in ast.unparse(node.value)
        )
        self.assertIn("pika_local_window_start_frame_no", ast.unparse(pika_options))
        local_origin_assignment = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and any(
                isinstance(target, ast.Name)
                and target.id == "pika_local_window_start_frame_no"
                for target in node.targets
            )
        )
        self.assertIsInstance(local_origin_assignment.value, ast.Call)
        self.assertEqual(
            ast.unparse(local_origin_assignment.value.func),
            "pika_flex_local_window_start_frame",
        )
        self.assertEqual(
            ast.unparse(local_origin_assignment.value.args[0]),
            "aligned_window_start_frame",
        )

        # Independent tasks use local zero while their source soundtrack and
        # ordered references still begin at their distinct source-time offsets.
        source_offsets = [0, 124]
        self.assertEqual(
            [pika_flex_local_window_start_frame(0) for _ in source_offsets],
            [0, 0],
        )
        self.assertEqual(source_offsets, [0, 124])

        # A real continuation window keeps its positive local origin even when
        # the global reference/source origin has advanced further.
        self.assertEqual(pika_flex_local_window_start_frame(106), 106)

    def test_pika_validation_uses_local_cursor_but_ref2va_keeps_global_slice_origin(self):
        tree = ast.parse(H3_MAIN.read_text(encoding="utf-8"))
        model_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3Model"
        )
        generate = next(
            node for node in model_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "generate"
        )
        generate_source = ast.unparse(generate)
        self.assertIn(
            "_kwargs.get('pika_local_window_start_frame_no', window_start_frame_no)",
            generate_source,
        )
        self.assertIn("timeline_start_frame=window_start_frame_no", generate_source)


if __name__ == "__main__":
    unittest.main()
