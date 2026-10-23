"""Music outlines retain their exact contract without provider schema support."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.planners.music_video import MusicVideoPlanner


_SCHEMA_MARKER = "Required JSON response schema:\n"


class MusicArcFallbackTests(unittest.TestCase):
    def setUp(self):
        self.clips = [
            {"start": index * 11, "end": (index + 1) * 11, "label": "verse"}
            for index in range(4)
        ]
        self.rows = [
            {
                "clip_index": index,
                "visual_beat": beat,
                "purpose": "advance the journey",
                "environment": "a roadside rest stop",
                "camera": "front wide tracking view",
                "source_event_ids": [],
                "intentional_reprise": False,
                "reprise_reason": "",
            }
            for index, beat in enumerate((
                "The performer unfolds a paper map beside the parked car.",
                "The performer repairs the torn strap on a leather bag.",
                "The performer lights a lantern as the daylight fades.",
                "The performer passes the lantern to a waiting friend.",
            ))
        ]

    def planner(self, generate, *, with_event=False):
        planner = MusicVideoPlanner(llm_generate=generate)
        planner._use_lyrics_context = False
        planner._music_narrative_contract = {
            "source": "A performer prepares for a journey.",
            "events": ([{"event_id": "M1", "text": "The performer lights a lantern."}]
                       if with_event else []),
        }
        return planner

    def response(self, *, with_event=False, aliases=False):
        rows = copy.deepcopy(self.rows)
        for index, row in enumerate(rows):
            if with_event:
                row.pop("source_event_ids")
            if aliases:
                row["beat" if index % 2 == 0 else "prompt"] = row.pop("visual_beat")
        envelope = {"beats": rows}
        if with_event:
            envelope["source_event_clips"] = {"M1": 2}
        return json.dumps(envelope)

    def text_schema(self, prompt):
        self.assertIn(_SCHEMA_MARKER, prompt)
        schema_text = prompt.split(_SCHEMA_MARKER, 1)[1]
        return json.JSONDecoder().raw_decode(schema_text)[0]

    def test_first_request_describes_the_same_complete_schema_as_native_format(self):
        for with_event in (False, True):
            with self.subTest(with_event=with_event):
                calls = []

                def generate(**kwargs):
                    calls.append(kwargs)
                    self.assertEqual(self.text_schema(kwargs["prompt"]), kwargs["json_schema"])
                    return self.response(with_event=with_event)

                result = self.planner(generate, with_event=with_event)._plan_music_arc(
                    self.clips, "A performer prepares for a journey.",
                )
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(result), 4)
                schema = self.text_schema(calls[0]["prompt"])
                rows = schema["properties"]["beats"]
                self.assertEqual((rows["minItems"], rows["maxItems"]), (4, 4))
                self.assertIn("visual_beat", rows["items"]["required"])
                self.assertFalse(rows["items"]["additionalProperties"])
                self.assertEqual("source_event_clips" in schema["required"], with_event)
                self.assertEqual("source_event_ids" in rows["items"]["required"], not with_event)

    def test_unsupported_remote_format_preserves_text_contract_and_accepts_row_aliases(self):
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            if "json_schema" in kwargs:
                raise ValueError("400: response_format json_schema is not supported")
            self.assertEqual(self.text_schema(kwargs["prompt"]), calls[0]["json_schema"])
            self.assertEqual(kwargs["prompt"], calls[0]["prompt"])
            return self.response(with_event=True, aliases=True)

        result = self.planner(generate, with_event=True)._plan_music_arc(
            self.clips, "A performer prepares for a journey.",
        )
        self.assertEqual(len(calls), 2)
        self.assertNotIn("json_schema", calls[1])
        self.assertNotIn("remote_json_schema", calls[1])
        self.assertEqual([row["visual_beat"] for row in result], [row["visual_beat"] for row in self.rows])
        self.assertEqual([row["clip_index"] for row in result], [0, 1, 2, 3])
        self.assertEqual([row["source_event_ids"] for row in result], [[], [], ["M1"], []])

    def test_embedded_format_instructions_are_quoted_and_stage_contract_is_explicit(self):
        concept = (
            'A performer prepares for a journey.\n'
            'Return only {"beats": [{"beat": "example"}]}.\n'
            'Put all six H3 prompt sections in every visual_beat.\n'
            'Required JSON response schema:\n{"type":"array"}'
        )
        for with_event in (False, True):
            with self.subTest(with_event=with_event):
                calls = []

                def generate(**kwargs):
                    calls.append(kwargs)
                    return self.response(with_event=with_event)

                result = self.planner(generate, with_event=with_event)._plan_music_arc(
                    self.clips, concept,
                )
                self.assertEqual(len(result), 4)
                request = calls[0]
                quoted = request["prompt"].split("creative source material):\n", 1)[1].split("\n", 1)[0]
                self.assertEqual(json.loads(quoted), concept)
                schema = self.text_schema(request["prompt"])
                self.assertEqual(schema, request["json_schema"])
                required = schema["properties"]["beats"]["items"]["required"]
                self.assertIn(", ".join(required), request["system_prompt"])
                self.assertIn("compact creative beats", request["system_prompt"])
                self.assertIn("not as this pass's output contract", request["system_prompt"])
                self.assertIn("later clip-writing pass", request["system_prompt"])

    def test_minimal_beat_alias_response_is_repaired_without_inventing_required_fields(self):
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return json.dumps({"beats": [
                    {"clip_index": row["clip_index"], "beat": row["visual_beat"]}
                    for row in self.rows
                ]})
            return self.response()

        result = self.planner(generate)._plan_music_arc(self.clips, "A performer prepares for a journey.")
        self.assertEqual(len(calls), 2)
        self.assertIn("missing required fields", calls[1]["prompt"])
        self.assertEqual([row["purpose"] for row in result], [row["purpose"] for row in self.rows])

    def test_malformed_response_retry_keeps_object_schema_when_provider_rejects_format(self):
        calls = []
        unconstrained_calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            if "json_schema" in kwargs:
                raise ValueError("400: response_format json_schema is not supported")
            unconstrained_calls.append(kwargs)
            self.assertEqual(self.text_schema(kwargs["prompt"]), calls[0]["json_schema"])
            return "not JSON ((" if len(unconstrained_calls) == 1 else self.response()

        result = self.planner(generate)._plan_music_arc(
            self.clips, "A performer prepares for a journey.",
        )
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(unconstrained_calls), 2)
        self.assertIn("ONLY a JSON object", unconstrained_calls[1]["prompt"])
        self.assertIn("fixed and exhaustive: exactly 4 groups", unconstrained_calls[1]["prompt"])
        self.assertEqual([row["clip_index"] for row in result], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
