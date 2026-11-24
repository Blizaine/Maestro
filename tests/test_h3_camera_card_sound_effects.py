"""Omitted sound_effects on camera cards is an empty string, not a rejection."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.h3_story_ledger import _expand_camera_event_cards  # noqa: E402
from services.h3_source_camera import build_source_camera_contract  # noqa: E402


def _card(**updates):
    card = {
        "transition": "hard cut",
        "framing": "A clear medium view",
        "camera": "Hold the established axis",
        "action": "Mina catches the rope.",
    }
    card.update(updates)
    return card


def _expand(event, *, beat=None):
    return _expand_camera_event_cards(
        {"event_cards": {"event_1": event}},
        assigned_beats=[beat or {"dialogue_ids": []}],
        segment_number=1, duration=4.0,
    )


def test_phase_cards_without_sound_effects_are_accepted():
    result = _expand({"phases": [_card(), _card(action="She ties it off.")]})
    assert [s["sound_effects"] for s in result["shots"]] == ["", ""]
    assert result["shots"][1]["action"] == "She ties it off."


def test_present_sound_effects_is_kept():
    result = _expand({"phases": [_card(sound_effects="Rope snaps")]})
    assert result["shots"][0]["sound_effects"] == "Rope snaps"


@pytest.mark.parametrize("missing", ["action", "camera", "framing", "transition"])
def test_other_missing_fields_still_rejected(missing):
    card = _card()
    del card[missing]
    with pytest.raises(ValueError, match="needs a complete camera/action card"):
        _expand({"phases": [card]})


def test_extra_key_still_rejected():
    with pytest.raises(ValueError, match="needs a complete camera/action card"):
        _expand({"phases": [_card(surprise="x")]})


def test_flat_card_without_phases_wrapper_still_rejected():
    with pytest.raises(ValueError, match=r"needs exactly these cards: phases\."):
        _expand(_card())


def test_opening_card_without_sound_effects_is_accepted():
    opening = {
        "transition": "continue supplied frame",
        "framing": "The supplied frame exact opening composition",
        "camera": "Hold the frame",
        "action": "Mina lowers her hands.",
        "recovery": "",
    }
    beat = {"dialogue_ids": [], "_start_frame_continuation": True}
    result = _expand({"opening": opening, "phases": [_card()]}, beat=beat)
    assert [s["sound_effects"] for s in result["shots"]] == ["", ""]


# --- SourceCameraContract.bind ------------------------------------------

def _contract(**beat_updates):
    beat = {
        "beat_id": "B1", "source_event_ids": ["E1"], "dialogue_ids": [],
        "description": "source-bound action", **beat_updates,
    }
    event = {"event_id": "E1", "text": "Mara grips the latch and pulls it down."}
    contract = build_source_camera_contract([beat], source_events=[event])
    assert contract is not None
    return contract


def _bind_card(schema):
    props = schema["properties"]
    return {
        "framing": props["framing"].get("const", "a readable medium view"),
        "camera": "the camera follows the assigned movement",
        "transition": props["transition"].get("const", "a motivated cut"),
        "sound_effects": "Natural synchronized effects",
    }


def _draft(contract, segment=1):
    cards = {}
    for key, ev in contract.schema(segment)["properties"]["event_cards"]["properties"].items():
        cards[key] = {pk: _bind_card(ps) for pk, ps in ev["properties"].items()}
    return {
        "segment": segment, "title": "t", "coverage": "c", "pacing": "p",
        "event_cards": cards,
    }


def test_bind_accepts_phase_without_sound_effects():
    contract = _contract()
    draft = _draft(contract)
    del draft["event_cards"]["event_1"]["phase_1"]["sound_effects"]
    bound = contract.bind(draft)
    assert bound["event_cards"]["event_1"]["phases"][0]["sound_effects"] == ""


def test_bind_accepts_opening_without_sound_effects():
    contract = _contract(_start_frame_continuation=True)
    draft = _draft(contract, segment=2)
    del draft["event_cards"]["event_1"]["opening"]["sound_effects"]
    bound = contract.bind(draft)
    assert bound["event_cards"]["event_1"]["opening"]["sound_effects"] == ""


def test_bind_still_rejects_missing_camera():
    contract = _contract()
    draft = _draft(contract)
    del draft["event_cards"]["event_1"]["phase_1"]["sound_effects"]
    del draft["event_cards"]["event_1"]["phase_1"]["camera"]
    with pytest.raises(ValueError, match=r"event_cards\.event_1: phase_1"):
        contract.bind(draft)


def test_bind_still_rejects_extra_key():
    contract = _contract()
    draft = _draft(contract)
    del draft["event_cards"]["event_1"]["phase_1"]["sound_effects"]
    draft["event_cards"]["event_1"]["phase_1"]["surprise"] = "x"
    with pytest.raises(ValueError, match=r"event_cards\.event_1: phase_1"):
        contract.bind(draft)
