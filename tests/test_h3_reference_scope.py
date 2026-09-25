import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.h3_reference_scope import prepare_scoped_reference_context  # noqa: E402
from services.h3_prompt_budget import H3PromptBudgetError  # noqa: E402
from services.h3_sequence_planner import (  # noqa: E402
    _image_reference_roles,
    plan_h3_reference_sequence,
)
from services.h3_story_ledger import _prepare_h3_story_context  # noqa: E402


class H3ReferenceScopeTests(unittest.TestCase):
    def _roles(self, first_intent="scene", second_intent="style"):
        return [
            {
                "path": r"C:\refs\hall.png",
                "image_index": 1,
                "image_intent": first_intent,
                "role": "the archive hall",
            },
            {
                "path": r"C:\refs\palette.png",
                "image_index": 2,
                "image_intent": second_intent,
                "role": "the cool palette",
            },
        ]

    def _response(self, roles, summaries=None):
        summaries = summaries or {
            1: "A high stone hall with arched windows and cool side light.",
            2: "Muted blue-gray palette with soft grain and low-contrast shadows.",
        }
        return json.dumps({
            f"image_{row['image_index']}": {
                "image_index": row["image_index"],
                "image_intent": row["image_intent"],
                "role": row["role"],
                "visual_context": summaries[row["image_index"]],
            }
            for row in roles
        })

    def test_scoped_summary_preserves_bindings_and_withholds_pixels_from_story(self):
        roles = self._roles(first_intent="composition", second_intent="scene")
        paths = [row["path"] for row in roles]
        contract = (
            "<Picture 1> is a composition reference for the archive hall.\n"
            "<Picture 2> defines the archive location."
        )
        calls = []

        def generate(prompt, **kwargs):
            calls.append((prompt, kwargs))
            return self._response(roles, {
                1: "A long stone hall runs between two rows of columns toward a raised dais.",
                2: "Muted blue-gray palette with soft grain and low-contrast shadows.",
            })

        result = prepare_scoped_reference_context(
            generate,
            "A courier crosses the hall.",
            contract,
            paths,
            roles,
        )

        self.assertTrue(result["scoped"])
        self.assertIsNone(result["image_paths"])
        self.assertIsNone(result["warning"])
        self.assertTrue(result["context"].startswith(contract))
        self.assertIn("Scoped visual observations from non-identity image references", result["context"])
        self.assertIn("A long stone hall runs between two rows of columns", result["context"])
        self.assertIn("<Picture 1> (composition; role: the archive hall)", result["context"])
        self.assertIn("<Picture 2> (scene; role: the cool palette)", result["context"])
        self.assertEqual(len(calls), 1)
        prompt, kwargs = calls[0]
        self.assertIn("A courier crosses the hall.", prompt)
        self.assertEqual(kwargs["image_paths"], paths)
        self.assertEqual(kwargs["enable_thinking"], False)
        self.assertEqual(kwargs["thinking_budget"], 0)
        self.assertLessEqual(kwargs["max_new_tokens"], 1200)
        schema = kwargs["json_schema"]
        self.assertEqual(schema["additionalProperties"], False)
        self.assertEqual(
            schema["properties"]["image_1"]["properties"]["image_intent"]["enum"],
            ["composition"],
        )
        self.assertEqual(
            schema["properties"]["image_2"]["properties"]["role"]["enum"],
            ["the cool palette"],
        )
        self.assertEqual(paths, [row["path"] for row in roles])

    def test_identity_unsupported_unknown_and_unaligned_inputs_keep_existing_path(self):
        path = r"C:\refs\hall.png"
        cases = [
            (self._roles(first_intent="identity", second_intent="scene"), [path]),
            (self._roles(first_intent="object", second_intent="scene"), [path]),
            (self._roles(first_intent="unknown", second_intent="scene"), [path]),
            ([], [path]),
            (self._roles(), [r"C:\refs\unmapped.png"]),
        ]
        for roles, paths in cases:
            with self.subTest(roles=roles, paths=paths):
                generate = unittest.mock.Mock(side_effect=AssertionError("must not summarize"))
                result = prepare_scoped_reference_context(
                    generate,
                    "Nora enters.",
                    "canonical reference contract",
                    paths,
                    roles,
                )
                self.assertFalse(result["scoped"])
                self.assertEqual(result["image_paths"], paths)
                self.assertEqual(result["context"], "canonical reference contract")
                self.assertIsNone(result["warning"])
                self.assertEqual(generate.call_count, 0)

        no_images = prepare_scoped_reference_context(
            unittest.mock.Mock(side_effect=AssertionError("must not summarize")),
            "Nora enters.",
            "contract",
            None,
            self._roles(),
        )
        self.assertFalse(no_images["scoped"])
        self.assertIsNone(no_images["image_paths"])

    def test_summary_failure_fails_closed_and_reports_visible_warning(self):
        roles = self._roles()
        paths = [row["path"] for row in roles]
        contract = "<Picture 1> is a scene reference."
        responses = [
            RuntimeError("local model unavailable"),
            "not json",
            self._response(roles, {
                1: "A person in a blue robe stands in the hall.",
                2: "Muted blue-gray palette with soft grain and shadows.",
            }),
            self._response(roles, {
                1: "A seated figure in a pale tunic stands beneath the arch.",
                2: "Muted blue-gray palette with soft grain and shadows.",
            }),
            self._response(roles, {
                1: "A high stone hall with arched windows.",
                2: "",
            }),
            json.dumps({
                "image_1": {
                    "image_index": 2,
                    "image_intent": "style",
                    "role": "the cool palette",
                    "visual_context": "swapped binding",
                },
                "image_2": {
                    "image_index": 1,
                    "image_intent": "scene",
                    "role": "the archive hall",
                    "visual_context": "swapped binding",
                },
            }),
        ]
        for response in responses:
            with self.subTest(response=response):
                def generate(_prompt, _response=response, **_kwargs):
                    if isinstance(_response, Exception):
                        raise _response
                    return _response

                result = prepare_scoped_reference_context(
                    generate,
                    "A courier crosses the hall.",
                    contract,
                    paths,
                    roles,
                )
                self.assertTrue(result["scoped"])
                self.assertIsNone(result["image_paths"])
                self.assertTrue(result["warning"])
                self.assertTrue(result["context"].startswith(contract))
                self.assertIn("no image pixels or inferred image facts", result["context"])
                self.assertNotIn("blue robe", result["context"])
                self.assertNotIn("seated figure", result["context"])
                self.assertNotIn("pale tunic", result["context"])
                self.assertTrue(any("_failed:" in item for item in result["diagnostics"]))

    def test_cancellation_and_budget_exceptions_propagate(self):
        roles = self._roles()
        for exception in (
            InterruptedError("cancelled"),
            H3PromptBudgetError("budget exhausted"),
        ):
            with self.subTest(exception=type(exception).__name__):
                def generate(_prompt, _exception=exception, **_kwargs):
                    raise _exception

                with self.assertRaises(type(exception)):
                    prepare_scoped_reference_context(
                        generate,
                        "A courier crosses the hall.",
                        "canonical contract",
                        [row["path"] for row in roles],
                        roles,
                    )

    def test_summary_schema_keeps_manifest_indices_when_path_order_differs(self):
        roles = [
            {
                "path": r"C:\refs\second.png",
                "image_index": 2,
                "image_intent": "style",
                "role": "palette",
            },
            {
                "path": r"C:\refs\first.png",
                "image_index": 1,
                "image_intent": "scene",
                "role": "hall",
            },
        ]
        seen = {}

        def generate(_prompt, **kwargs):
            seen.update(kwargs)
            ordered = sorted(roles, key=lambda row: row["image_index"])
            return self._response(ordered)

        result = prepare_scoped_reference_context(
            generate,
            "Nora crosses the hall.",
            "contract",
            [roles[0]["path"], roles[1]["path"]],
            roles,
        )
        self.assertTrue(result["scoped"])
        self.assertEqual(
            seen["image_paths"],
            [r"C:\refs\first.png", r"C:\refs\second.png"],
        )
        self.assertIn("<Picture 1> (scene; role: hall)", result["context"])
        self.assertIn("<Picture 2> (style; role: palette)", result["context"])

    def test_summary_budget_scales_for_multiple_long_fixed_roles(self):
        roles = [
            {
                "path": rf"C:\refs\section_{index}.png",
                "image_index": index,
                "image_intent": "scene",
                "role": f"archive section {index} " + ("north gallery " * 40),
            }
            for index in range(1, 4)
        ]
        seen = {}

        def generate(_prompt, **kwargs):
            seen.update(kwargs)
            return self._response(roles, {
                index: "Stone walls and high windows with cool natural light."
                for index in range(1, 4)
            })

        result = prepare_scoped_reference_context(
            generate,
            "A courier crosses the archive.",
            "canonical contract",
            [row["path"] for row in roles],
            roles,
        )
        self.assertTrue(result["scoped"])
        self.assertGreaterEqual(seen["max_new_tokens"], 1080)
        self.assertLessEqual(seen["max_new_tokens"], 3000)
        self.assertEqual(len(seen["json_schema"]["required"]), 3)

    def test_sequence_planner_passes_normalized_picture_roles_to_story_planner(self):
        references = [
            {
                "type": "audio",
                "path": r"C:\refs\guide.wav",
                "audio_intent": "style",
                "role": "soft room tone",
            },
            {
                "type": "image",
                "path": r"C:\refs\hall.png",
                "image_intent": "scene",
                "role": "archive hall",
            },
            {
                "type": "image",
                "path": r"C:\refs\palette.png",
                "image_intent": "composition",
                "role": "framing study",
            },
        ]
        paths = [r"C:\refs\hall.png", r"C:\refs\palette.png"]
        self.assertEqual(
            _image_reference_roles(references, paths),
            [
                {
                    "path": paths[0], "image_index": 1,
                    "image_intent": "scene", "role": "archive hall",
                },
                {
                    "path": paths[1], "image_index": 2,
                    "image_intent": "composition", "role": "framing study",
                },
            ],
        )
        captured = {}

        def story_planner(_prompt, **kwargs):
            captured.update(kwargs)
            return {
                "planned_by": "test",
                "ledger": {
                    "subject_continuity": "",
                    "setting_continuity": "",
                    "visual_continuity": "",
                    "ambient_audio": "",
                    "music": "N/A",
                },
                "segments": [],
            }

        with (
            patch("services.h3_sequence_planner.plan_h3_story_segments", side_effect=story_planner),
            patch(
                "services.h3_sequence_planner.compile_h3_reference_sequence_prompts",
                side_effect=lambda _plan, clips, **_kwargs: [
                    {"clip": index + 1, "prompt": f"Window {index + 1}"}
                    for index, _clip in enumerate(clips)
                ],
            ),
        ):
            plan_h3_reference_sequence(
                "A courier crosses the archive hall.",
                model_type="test",
                resolution="512x512",
                total_frames=500,
                references=references,
                min_clip_frames=124,
                max_clip_frames=250,
                frame_step=1,
                fps=24,
                image_paths=paths,
            )
        self.assertEqual(captured["image_paths"], paths)
        self.assertEqual(
            captured["image_reference_roles"],
            _image_reference_roles(references, paths),
        )

    def test_story_ledger_scopes_images_before_story_writer_and_propagates_warning(self):
        roles = [{
            "path": r"C:\refs\hall.png",
            "image_index": 1,
            "image_intent": "scene",
            "role": "archive hall",
        }]
        paths = [roles[0]["path"]]
        contract = "<Picture 1> defines the archive hall as a scene reference."

        for summary_response in (
            self._response(roles, {1: "A stone archive hall with cool window light."}),
            "not json",
        ):
            with self.subTest(summary_valid=summary_response != "not json"):
                calls = []

                def generate(*args, **kwargs):
                    calls.append((args, kwargs))
                    if args:
                        return summary_response
                    raise RuntimeError("stop after observing the story-planner request")

                context = _prepare_h3_story_context(
                    "Nora enters the archive and studies a map.",
                    segment_durations=[8.0],
                    mode="reference_sequence",
                    camera_coverage="single_shot",
                    reference_context=contract,
                    expect_dialogue=False,
                    planning_style="faithful",
                    image_paths=paths,
                    image_reference_roles=roles,
                    has_start_image=False,
                    nsfw=False,
                    llm_generate=generate,
                )

                self.assertEqual(len(calls), 2)
                vision_args, vision_kwargs = calls[0]
                self.assertTrue(vision_args)
                self.assertEqual(vision_kwargs["image_paths"], paths)
                story_args, story_kwargs = calls[1]
                self.assertFalse(story_args)
                self.assertIsNone(story_kwargs["image_paths"])
                self.assertIn(contract, story_kwargs["prompt"])
                self.assertEqual(context["image_paths"], None)
                if summary_response == "not json":
                    self.assertIn("could not be summarized safely", story_kwargs["prompt"])
                    self.assertTrue(any(
                        "Scoped visual context could not be summarized safely" in warning
                        for warning in context["planning_warnings"]
                    ))
                    self.assertTrue(any(
                        "nonidentity_reference_scope_failed" in diagnostic
                        for diagnostic in context["planning_diagnostics"]
                    ))
                else:
                    self.assertIn("A stone archive hall with cool window light", story_kwargs["prompt"])
                    self.assertFalse(any(
                        "Scoped visual context could not be summarized safely" in warning
                        for warning in context["planning_warnings"]
                    ))

    def test_start_frame_skips_scope_and_keeps_original_image_on_story_call(self):
        roles = [{
            "path": r"C:\refs\opening.png",
            "image_index": 1,
            "image_intent": "scene",
            "role": "archive hall",
        }]
        paths = [roles[0]["path"]]
        contract = "<Picture 1> is the supplied opening frame."
        calls = []

        def generate(*args, **kwargs):
            calls.append((args, kwargs))
            raise RuntimeError("stop after observing the story-planner request")

        context = _prepare_h3_story_context(
            "Nora enters the archive and studies a map.",
            segment_durations=[8.0],
            mode="sliding_window",
            camera_coverage="single_shot",
            reference_context=contract,
            expect_dialogue=False,
            planning_style="faithful",
            image_paths=paths,
            image_reference_roles=roles,
            has_start_image=True,
            nsfw=False,
            llm_generate=generate,
        )
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0][0])
        self.assertEqual(calls[0][1]["image_paths"], paths)
        self.assertIn(contract, calls[0][1]["prompt"])
        self.assertTrue(context["start_frame_supplied"])
        self.assertEqual(context["image_paths"], paths)


if __name__ == "__main__":
    unittest.main()
