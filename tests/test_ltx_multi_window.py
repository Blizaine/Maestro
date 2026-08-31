import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.ltx_window_planner import (  # noqa: E402
    compute_ltx_window_count,
    deterministic_ltx_window_prompts,
    extract_ltx_global_invariants,
    parse_ltx_window_prompts,
    reinforce_ltx_window_invariants,
    requests_open_ended_ltx_motion,
)
from services.llm_service import _build_enhance_user_prompt  # noqa: E402


class LtxWindowPlannerTests(unittest.TestCase):
    def test_window_count_matches_wangp_stride(self):
        self.assertEqual(
            3,
            compute_ltx_window_count(
                600,
                241,
                overlap_frames=9,
                discard_frames=8,
            ),
        )
        self.assertEqual(
            1,
            compute_ltx_window_count(
                241,
                241,
                overlap_frames=9,
                discard_frames=8,
            ),
        )

    def test_58_second_timeline_uses_three_max_length_ltx_windows(self):
        self.assertEqual(
            3,
            compute_ltx_window_count(
                58 * 25,
                501,
                overlap_frames=9,
                discard_frames=8,
            ),
        )

    def test_invalid_stride_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "greater than"):
            compute_ltx_window_count(
                300,
                17,
                overlap_frames=9,
                discard_frames=8,
            )

    def test_manual_lines_are_exact(self):
        self.assertEqual(
            ["first beat", "second beat", "final beat"],
            parse_ltx_window_prompts(
                "first beat\nsecond beat\nfinal beat",
                expected_count=3,
            ),
        )
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            parse_ltx_window_prompts(
                "first beat\nfinal beat",
                expected_count=3,
            )

    def test_ai_paragraphs_are_collapsed_to_one_line_per_window(self):
        result = parse_ltx_window_prompts(
            "First paragraph has\na wrapped line.\n\nSecond paragraph continues.",
            expected_count=2,
        )
        self.assertEqual(
            [
                "First paragraph has a wrapped line.",
                "Second paragraph continues.",
            ],
            result,
        )

    def test_fallback_advances_middle_and_final_windows(self):
        prompts = deterministic_ltx_window_prompts(
            "A woman enters. She finds a key. She opens the vault.",
            3,
        )
        self.assertEqual(3, len(prompts))
        self.assertIn("begin this beat", prompts[0])
        self.assertIn("advance only this middle beat", prompts[1])
        self.assertIn("complete the requested final beat", prompts[2])

    def test_global_sequence_rules_are_repeated_in_every_native_pass(self):
        source = (
            "an amateur handheld recording of a very fast movement falling "
            "through never ending rooms, never stopping. different 90s style "
            "rooms, all indoors. Seamless oner with no clear cuts, flowing "
            "through an ever rising sweeping path inside a very large torus."
        )
        invariants = extract_ltx_global_invariants(source)
        self.assertIn("an amateur handheld recording", invariants)
        self.assertIn("very fast movement and pacing", invariants)
        self.assertIn("continuous falling motion", invariants)
        self.assertIn(
            "nonstop motion that never slows, settles, stops, or resolves",
            invariants,
        )
        self.assertIn("90s visual design", invariants)
        self.assertIn("every environment remaining indoors", invariants)
        self.assertIn(
            "one seamless continuous take with no cuts or invisible resets",
            invariants,
        )
        self.assertIn(
            "an ever-rising sweeping path inside a vast torus",
            invariants,
        )

        prompts = reinforce_ltx_window_invariants(
            [
                "Fall through a floral living room into a tiled passage.",
                "Fall through an indoor pool toward a concrete opening.",
                "Fall through a parking garage and onward beyond the frame.",
            ],
            source,
        )
        self.assertEqual(3, len(prompts))
        for prompt in prompts:
            for invariant in invariants:
                self.assertIn(invariant, prompt)
            self.assertTrue(
                prompt.startswith("Throughout this complete window, preserve")
            )
        self.assertNotIn("indoor pool", prompts[0])
        self.assertNotIn("floral living room", prompts[1])
        self.assertEqual(
            prompts,
            reinforce_ltx_window_invariants(prompts, source),
        )

    def test_a_local_period_room_is_not_forced_across_the_whole_sequence(self):
        invariants = extract_ltx_global_invariants(
            "Start in a 1990s living room, then pass into a futuristic lab."
        )
        self.assertNotIn("1990s visual design", invariants)

    def test_vocal_performance_contract_reaches_every_ltx_window(self):
        source = (
            "A rapper rides a horse and raps directly to the supplied song."
        )
        prompts = reinforce_ltx_window_invariants(
            [
                "The rapper begins the verse while riding toward camera.",
                "The rapper continues through the chorus on the same road.",
            ],
            source,
        )
        self.assertEqual(2, len(prompts))
        for prompt in prompts:
            self.assertIn("lip-syncing each vocal syllable", prompt)
            self.assertIn("supplied source soundtrack", prompt)

    def test_explicit_camera_and_speed_changes_are_not_forced_globally(self):
        invariants = extract_ltx_global_invariants(
            "Begin with handheld footage during a very fast fall, then stop "
            "and switch to a locked-off static camera for a slow-motion walk."
        )
        self.assertNotIn("handheld camera movement", invariants)
        self.assertNotIn("a locked-off static camera", invariants)
        self.assertNotIn("very fast movement and pacing", invariants)
        self.assertNotIn("continuous falling motion", invariants)

    def test_endless_fallback_never_invents_a_resolved_finale(self):
        source = (
            "A handheld camera keeps falling through endless rooms, nonstop, "
            "never stopping."
        )
        self.assertTrue(requests_open_ended_ltx_motion(source))
        prompts = deterministic_ltx_window_prompts(source, 3)
        self.assertNotIn("complete the requested final beat", prompts[-1])
        self.assertIn("without slowing, settling, stopping, or resolving", prompts[-1])

    def test_ltx_enhance_request_explains_the_isolated_window_runtime(self):
        request = _build_enhance_user_prompt(
            "A continuous journey through many rooms.",
            "video",
            120,
            6,
            20,
            "ltx2_25",
        )
        self.assertIn("LTX STANDALONE-WINDOW CONTRACT", request)
        self.assertIn("short audiovisual overlap", request)
        self.assertIn("never any previous prompt text", request)
        self.assertIn("EVERY paragraph", request)
        self.assertIn("must not slow, settle, stop, or resolve", request)


class LtxMultiWindowWiringTests(unittest.TestCase):
    def test_every_ltx_video_handler_declares_sequence_controls(self):
        for relative in (
            "app/models/ltx_video/ltxv_handler.py",
            "app/models/ltx2/ltx2_handler.py",
            "app/models/ltx25/ltx25_handler.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn('"multi_window_sequence_controls"', source, relative)

    def test_ltx23_repairs_a_visible_soundtrack_missing_its_hidden_mode(self):
        handler = (
            ROOT / "app/models/ltx2/ltx2_handler.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"infer_audio_prompt_from_guide": True', handler)

    def test_api_and_backend_publish_the_sequence_contract(self):
        launch = (APP / "launch.py").read_text(encoding="utf-8")
        self.assertIn('"multi_window_sequence_controls"', launch)
        self.assertIn('body.get("ltx_multi_window") is True', launch)
        self.assertIn('body["multi_prompts_gen_type"] = 1', launch)
        self.assertIn('response["ltx_window_plan"]', launch)
        self.assertIn("needs_ltx_window_plan", launch)
        self.assertIn("reinforce_ltx_window_invariants", launch)

    def test_studio_exposes_toggle_modes_counts_and_manual_validation(self):
        controls = (
            ROOT
            / "ui/src/components/Sidebar/H3MultiWindowControls.tsx"
        ).read_text(encoding="utf-8")
        duration = (
            ROOT / "ui/src/components/Sidebar/DurationSlider.tsx"
        ).read_text(encoding="utf-8")
        prompt = (
            ROOT / "ui/src/components/Sidebar/PromptInput.tsx"
        ).read_text(encoding="utf-8")
        store = (ROOT / "ui/src/stores/useStore.ts").read_text(encoding="utf-8")
        self.assertIn("multi_window_sequence_controls", controls)
        self.assertIn("ltx_multi_window", controls)
        self.assertIn("ltx_window_prompt_mode", controls)
        self.assertIn("AI-planned window prompts", duration)
        self.assertIn("LTX long-form Auto follows Duration", duration)
        self.assertNotIn("if (isLtx && ltxMultiWindow) return", duration)
        self.assertIn("usesLtxManualPrompts", prompt)
        self.assertIn("Manual LTX sequence needs exactly", store)

    def test_ltx_guide_distributes_actions_chronologically(self):
        guide = (
            APP / "services/llm_guides/enhance/ltx2_video.md"
        ).read_text(encoding="utf-8")
        normalized_guide = " ".join(guide.split())
        self.assertIn("Distribute the requested story chronologically", normalized_guide)
        self.assertIn("never repeat", normalized_guide)
        self.assertIn("native audio", normalized_guide)
        self.assertIn("COMPLETE STANDALONE generation prompt", normalized_guide)
        self.assertIn("short audiovisual overlap", normalized_guide)
        self.assertIn("Repeat EVERY applicable global invariant", normalized_guide)
        self.assertIn("must not resolve", normalized_guide)
        self.assertIn("one seamless continuous take", normalized_guide)


if __name__ == "__main__":
    unittest.main()
