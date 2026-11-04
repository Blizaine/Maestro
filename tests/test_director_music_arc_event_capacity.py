"""The music-arc source ledger has a total and a per-row bound."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.music_arc import (
    MAX_MUSIC_ARC_EVENTS,
    MAX_MUSIC_ARC_EVENTS_PER_ROW,
    arc_allocation,
    creative_plan_schema,
    normalize_music_arc,
)


def _arc_row(index: int, event_ids: list[str]) -> dict:
    return {
        "clip_index": index,
        "visual_beat": f"The courier studies group {index + 1} in detail.",
        "purpose": "advance the visual progression",
        "environment": "the same location",
        "camera": "locked medium-wide frame",
        "source_event_ids": event_ids,
        "intentional_reprise": False,
        "reprise_reason": "",
    }


def _contract(event_count: int) -> dict:
    return {
        "events": [
            {"event_id": f"M{index + 1}", "text": f"source action {index + 1}"}
            for index in range(event_count)
        ]
    }


class MusicArcEventCapacityTests(unittest.TestCase):
    def test_forty_eight_total_events_are_allowed_and_forty_nine_are_rejected(self):
        self.assertEqual(MAX_MUSIC_ARC_EVENTS, 48)
        allowed = _contract(48)
        rows = [
            _arc_row(index, [f"M{index + 1}"])
            for index in range(48)
        ]
        normalized = normalize_music_arc(rows, [{} for _ in rows], allowed)
        self.assertEqual(
            [event["event_id"] for event in arc_allocation(normalized)],
            [f"M{index}" for index in range(1, 49)],
        )

        over_limit = _contract(49)
        over_rows = [
            _arc_row(index, [f"M{index + 1}"])
            for index in range(49)
        ]
        with self.assertRaisesRegex(ValueError, "exceeds 48 source events"):
            normalize_music_arc(over_rows, [{} for _ in over_rows], over_limit)

    def test_per_row_event_budget_stays_sixteen(self):
        self.assertEqual(MAX_MUSIC_ARC_EVENTS_PER_ROW, 16)
        source_contract = _contract(17)
        row = _arc_row(0, [f"M{index}" for index in range(1, 18)])
        with self.assertRaisesRegex(ValueError, "at most 16 IDs"):
            normalize_music_arc([row], [{}], source_contract)

        schema = creative_plan_schema()
        self.assertEqual(
            schema["items"]["properties"]["source_event_ids"]["maxItems"],
            16,
        )


if __name__ == "__main__":
    unittest.main()
