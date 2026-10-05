from __future__ import annotations

import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

_candidate_app = Path(__file__).resolve().parents[1] / "app"
_repository_app = Path(__file__).resolve().parents[4] / "app"
sys.path[:0] = [str(_repository_app), str(_candidate_app)]
import services  # noqa: E402

if str(_candidate_app / "services") not in services.__path__:
    services.__path__.insert(0, str(_candidate_app / "services"))

from services.h3_source_camera import build_source_camera_contract


def _beat(event_ids, **updates):
    beat = {
        "beat_id": "B1",
        "source_event_ids": list(event_ids),
        "dialogue_ids": [],
        "description": "source-bound action",
    }
    beat.update(updates)
    return beat


def _event(event_id="E1", text="Mara grips the latch and pulls it down.", **updates):
    event = {"event_id": event_id, "text": text}
    event.update(updates)
    return event


def _camera_card(schema, *, camera="the camera follows the assigned movement"):
    properties = schema["properties"]
    return {
        "framing": properties["framing"].get("const", "a readable medium view"),
        "camera": camera,
        "transition": properties["transition"].get("const", "a motivated cut"),
        "sound_effects": "Natural synchronized effects",
    }


def _draft(contract, *, segment=1):
    schema = contract.schema(segment)
    event_properties = schema["properties"]["event_cards"]["properties"]
    event_cards = {}
    for event_key, event_schema in event_properties.items():
        event = {}
        for phase_key, phase_schema in event_schema["properties"].items():
            event[phase_key] = _camera_card(phase_schema)
        event_cards[event_key] = event
    return {
        "segment": segment,
        "title": "Source-bound action",
        "coverage": "coherent coverage",
        "pacing": "natural real-time pacing",
        "event_cards": event_cards,
    }


def _words(text):
    return re.findall(r"[\w’'-]+", text.casefold())


def test_source_action_is_compiler_owned_and_rejects_shoulder_injection():
    source = "Mara drives the padded edge into Theo's elbow, keeping contact as his arm folds toward his ribs."
    beat = _beat(["E1"])
    contract = build_source_camera_contract([beat], source_events=[_event(text=source)])

    assert contract is not None
    draft = _draft(contract)
    bound = contract.bind(draft)
    action = bound["event_cards"]["event_1"]["phases"][0]["action"]
    assert "elbow" in action
    assert "shoulder" not in action

    injected = _draft(contract)
    injected["event_cards"]["event_1"]["phase_1"]["action"] = "Mara drives the padded edge into Theo's shoulder."
    with pytest.raises(ValueError, match=r"event_cards\.event_1: phase_1"):
        contract.bind(injected)


def test_complete_source_action_text_is_retained_once_across_phases():
    sentences = [
        "Mara braces her left foot against the lower rail and clamps both hands around the latch.",
        "She pulls the latch downward with her full weight while keeping the gate from swinging.",
        "The metal arm drops into its slot and the gate remains closed against the frame.",
        "Mara releases the latch only after the arm stops moving and the frame holds firm.",
        "She keeps her palm flat on the gate while checking that the latch stays seated.",
    ]
    source = " ".join(sentences)
    beat = _beat(["E1"])
    contract = build_source_camera_contract([beat], source_events=[_event(text=source)])
    assert contract is not None

    bound = contract.bind(_draft(contract))
    actions = [
        phase["action"]
        for phase in bound["event_cards"]["event_1"]["phases"]
    ]
    assert 2 <= len(actions) <= 4
    assert Counter(_words(" ".join(actions))) == Counter(_words(source))
    assert _words(" ".join(actions)) == _words(source)


def test_source_clause_over_ninety_words_is_not_cut_or_truncated():
    source = (
        "Mara keeps her left palm pressed against the steel gate while "
        + " ".join(f"marker{index}" for index in range(115))
        + " and holds the latch closed."
    )
    beat = _beat(["E1"])
    contract = build_source_camera_contract([beat], source_events=[_event(text=source)])
    assert contract is not None

    bound = contract.bind(_draft(contract))
    actions = [item["action"] for item in bound["event_cards"]["event_1"]["phases"]]
    assert len(actions) == 1
    assert _words(actions[0]) == _words(source)
    assert len(_words(actions[0])) > 90


def test_source_line_label_and_clock_are_removed_by_ledger_helpers():
    source = "MARA: At 4.25 seconds, Mara grips the latch and pulls it down."
    contract = build_source_camera_contract(
        [_beat(["E1"])], source_events=[_event(text=source)],
    )
    assert contract is not None
    bound = contract.bind(_draft(contract))
    action = bound["event_cards"]["event_1"]["phases"][0]["action"]
    assert action == "Mara grips the latch and pulls it down"
    assert "MARA:" not in action
    assert "4.25 seconds" not in action


def test_missing_foreign_phase_and_extra_writer_fields_fail_closed():
    beat = _beat(["E1"])
    contract = build_source_camera_contract([beat], source_events=[_event()])
    assert contract is not None

    missing_event = _draft(contract)
    missing_event["event_cards"] = {}
    with pytest.raises(ValueError, match="event_cards"):
        contract.bind(missing_event)

    missing_phase = _draft(contract)
    missing_phase["event_cards"]["event_1"] = {}
    with pytest.raises(ValueError, match=r"event_cards\.event_1:"):
        contract.bind(missing_phase)

    extra_field = _draft(contract)
    extra_field["event_cards"]["event_1"]["phase_1"]["action"] = "invented action"
    with pytest.raises(ValueError, match=r"event_cards\.event_1: phase_1"):
        contract.bind(extra_field)

    assert build_source_camera_contract(
        [_beat([])], source_events=[_event()],
    ) is None
    assert build_source_camera_contract(
        [_beat(["E9"])], source_events=[_event()],
    ) is None


def test_supplied_frame_opening_is_fixed_and_recovery_is_empty():
    source = "Mara lowers her hands from the visible guard and steps toward the open doorway."
    beat = _beat(["E1"], _start_frame_continuation=True)
    contract = build_source_camera_contract([beat], source_events=[_event(text=source)])
    assert contract is not None

    schema = contract.schema(2)
    opening_schema = schema["properties"]["event_cards"]["properties"]["event_1"]["properties"]["opening"]
    assert opening_schema["properties"]["framing"]["const"] == "The supplied frame's exact opening composition"
    assert opening_schema["properties"]["transition"]["const"] == "continue supplied frame"
    assert "recovery" not in opening_schema["properties"]

    bound = contract.bind(_draft(contract, segment=2))
    opening = bound["event_cards"]["event_1"]["opening"]
    assert opening["framing"] == "The supplied frame's exact opening composition"
    assert opening["transition"] == "continue supplied frame"
    assert opening["recovery"] == ""
    assert "action" not in _draft(contract)["event_cards"]["event_1"]["opening"]


def test_pure_optics_join_adjacent_action_without_an_invented_hold():
    source = (
        "The camera slowly pulls back from the steel table. "
        "Mara lifts the brass key from the table and closes her left hand around its head."
    )
    beat = _beat(["E1"])
    contract = build_source_camera_contract([beat], source_events=[_event(text=source)])
    assert contract is not None

    bound = contract.bind(_draft(contract))
    phase = bound["event_cards"]["event_1"]["phases"][0]
    assert "camera slowly pulls back" in phase["camera"].casefold()
    assert "camera slowly pulls back" in phase["action"].casefold()
    assert "Mara lifts the brass key" in phase["action"]
    assert "hold" not in phase["action"].casefold()
    assert len(bound["event_cards"]["event_1"]["phases"]) == 1


def test_mixed_lens_clause_keeps_every_physical_fact_in_source_action():
    source = (
        "Character B drives forward. Lens rapidly pushes toward him, showing him "
        "shatter ruins like teleporting, rip through smoke, killing back to the "
        "battlefield at terrifying speed. Character A turns toward the impact."
    )
    beat = _beat(["E1"])
    contract = build_source_camera_contract(
        [beat], source_events=[_event(text=source)],
        cast_names=["Character A", "Character B"],
    )
    assert contract is not None

    bound = contract.bind(_draft(contract))
    phases = bound["event_cards"]["event_1"]["phases"]
    action = " ".join(item["action"] for item in phases)
    assert _words(action) == _words(source)
    assert action.casefold().count("shatter ruins") == 1
    assert action.casefold().count("rip through smoke") == 1
    assert action.casefold().count("killing back") == 1
    assert all("shatter ruins" not in item["camera"].casefold() for item in phases)


def test_lens_sentence_after_semicolon_stays_with_physical_phase():
    source = (
        "Mara braces both feet against the lower rail; the camera slowly pushes in "
        "from behind her. Then Mara pulls the latch downward until it seats."
    )
    contract = build_source_camera_contract(
        [_beat(["E1"])], source_events=[_event(text=source)],
        cast_names=["Mara"],
    )
    assert contract is not None
    bound = contract.bind(_draft(contract))
    phases = bound["event_cards"]["event_1"]["phases"]
    assert len(phases) == 2
    assert _words(" ".join(item["action"] for item in phases)) == _words(source)
    assert "pushes in" in phases[-1]["action"]
    assert "pushes in" in phases[-1]["camera"]


def test_safe_midparagraph_optical_sentence_folds_into_neighboring_action():
    source = (
        "Mara drives the padded edge into Theo's elbow. The lens cuts to the "
        "cracked stone wall. Theo drops to one knee."
    )
    contract = build_source_camera_contract(
        [_beat(["E1"])], source_events=[_event(text=source)],
        cast_names=["Mara", "Theo"],
    )
    assert contract is not None
    bound = contract.bind(_draft(contract))
    phases = bound["event_cards"]["event_1"]["phases"]
    assert len(phases) == 2
    assert all("lens cuts to the cracked stone wall" not in phase["action"].casefold()
               or "Theo drops" in phase["action"] for phase in phases)
    assert _words(" ".join(item["action"] for item in phases)) == _words(source)
    assert "lens cuts to the cracked stone wall" in phases[-1]["camera"].casefold()


def test_consecutive_optical_units_keep_authored_order_when_folded_forward():
    source = (
        "The lens slowly pans left across the wooden room. The camera pushes "
        "in from beside the red bench. Mara grips the brass key with her right hand."
    )
    contract = build_source_camera_contract(
        [_beat(["E1"])], source_events=[_event(text=source)],
        cast_names=["Mara"],
    )
    assert contract is not None
    bound = contract.bind(_draft(contract))
    phases = bound["event_cards"]["event_1"]["phases"]
    assert len(phases) == 1
    action = phases[0]["action"]
    assert _words(action) == _words(source)
    assert action.index("pans left") < action.index("pushes in") < action.index("grips")
    camera = phases[0]["camera"]
    assert camera.index("pans left") < camera.index("pushes in")


def test_proof_invalidates_changed_action_or_beat_binding_and_is_not_json():
    source = "Mara grips the latch and pulls it down until the metal arm seats in the slot."
    beat = _beat(["E1"])
    events = [_event(text=source)]
    contract = build_source_camera_contract([beat], source_events=events)
    assert contract is not None

    bound = contract.bind(_draft(contract))
    proof = bound["_source_camera_proof"]
    source_map = {"E1": source}
    assert proof.covers_beat(bound, beat, source_map)
    with pytest.raises(TypeError):
        json.dumps(bound)

    changed_action = copy.deepcopy(bound)
    changed_action["event_cards"]["event_1"]["phases"][0]["action"] += " toward the shoulder"
    assert not proof.covers_beat(changed_action, beat, source_map)
    changed_action_case = copy.deepcopy(bound)
    changed_action_case["event_cards"]["event_1"]["phases"][0]["action"] = (
        changed_action_case["event_cards"]["event_1"]["phases"][0]["action"].lower()
    )
    assert not proof.covers_beat(changed_action_case, beat, source_map)

    changed_binding = dict(beat, source_event_ids=["E2"])
    assert not proof.covers_beat(bound, changed_binding, source_map)
    changed_canonical_beat = dict(beat, _canonical_action="Mara drives the latch into the shoulder.")
    assert not proof.covers_beat(bound, changed_canonical_beat, source_map)
    audio_beat = dict(beat, _audio_driven=True)
    assert not proof.covers_beat(bound, audio_beat, source_map)
    changed_source = {"E1": source.replace("latch", "shoulder")}
    assert not proof.covers_beat(bound, beat, changed_source)
    source_with_changed_whitespace = {"E1": source + " "}
    assert not proof.covers_beat(bound, beat, source_with_changed_whitespace)
    source_with_dialogue = {
        "E1": {"event_id": "E1", "text": source, "dialogue_ids": ["D1"]},
    }
    assert not proof.covers_beat(bound, beat, source_with_dialogue)


def test_canonicalized_opening_proof_checks_frame_constants_and_occurrence_flags():
    source = "Mara lowers her hands from the visible guard. She steps toward the open doorway."
    beat = _beat(["E1"], _start_frame_continuation=True)
    contract = build_source_camera_contract([beat], source_events=[_event(text=source)])
    assert contract is not None
    bound = contract.bind(_draft(contract))
    opening = bound["event_cards"]["event_1"]["opening"]
    later = bound["event_cards"]["event_1"]["phases"][0]
    segment = {
        "_source_camera_proof": bound["_source_camera_proof"],
        "shots": [
            {
                "beat_ids": ["B1"], "action": opening["action"],
                "framing": opening["framing"], "transition": opening["transition"],
                "camera": opening["camera"],
            },
            {
                "beat_ids": ["B1"], "action": later["action"],
                "framing": later["framing"], "transition": later["transition"],
                "camera": later["camera"],
            },
        ],
    }
    proof = bound["_source_camera_proof"]
    assert proof.covers_beat(segment, beat, {"E1": source})

    wrong_frame = copy.deepcopy(segment)
    wrong_frame["shots"][0]["framing"] = "a different composition"
    assert not proof.covers_beat(wrong_frame, beat, {"E1": source})

    wrong_transition = copy.deepcopy(segment)
    wrong_transition["shots"][0]["transition"] = "hard cut"
    assert not proof.covers_beat(wrong_transition, beat, {"E1": source})

    occurrence_beat = dict(beat, source_occurrence=2)
    assert not proof.covers_beat(segment, occurrence_beat, {"E1": source})
    assert not proof.covers_beat(segment, beat, [{"event_id": "E1", "text": source, "occurrence": 2}])


@pytest.mark.parametrize(
    "beat,source",
    [
        (_beat(["E1"], dialogue_ids=["D1"]), "Mara grips the latch."),
        (_beat(["E1"], generated=True), "Mara grips the latch."),
        (_beat(["E1"], _source_enabler_groups=[]), "Mara grips the latch."),
        (_beat(["E1"], _spaced_recurrence={"occurrence": 1}), "Mara grips the latch."),
        (_beat(["E1"], source_occurrence=2), "Mara grips the latch."),
        (_beat(["E1"]), "Mara says the latch is ready."),
    ],
)
def test_speaking_generated_enabler_and_recurrence_beats_are_ineligible(beat, source):
    assert build_source_camera_contract(
        [beat], source_events=[_event(text=source)],
    ) is None


def test_unknown_source_ids_and_unknown_coverage_are_ineligible():
    assert build_source_camera_contract(
        [_beat(["E404"])], source_events=[_event()],
    ) is None
    assert build_source_camera_contract(
        [_beat(["E1"])], source_events=[_event()], camera_coverage="locked_pov",
    ) is None
    assert build_source_camera_contract(
        [_beat(["E1"])], source_events=[_event(dialogue_ids=["D1"])],
    ) is None
    assert build_source_camera_contract(
        [_beat(["E1"])], source_events=[_event(text="The camera slowly pulls back from the table.")],
    ) is None
    assert build_source_camera_contract(
        [_beat(["E1"])],
        source_events=[
            _event("E1", "Mara grips the latch.", requirement_kind="final_state"),
            _event("E1", "Mara pulls the latch down."),
        ],
    ) is None


def test_continuous_coverage_uses_reframes_for_the_same_compiled_actions():
    beat = _beat(["E1"])
    contract = build_source_camera_contract(
        [beat], source_events=[_event()], camera_coverage="continuous",
    )
    assert contract is not None
    schema = contract.schema(1)
    transition = schema["properties"]["event_cards"]["properties"]["event_1"]["properties"]["phase_1"]["properties"]["transition"]
    assert transition["const"] == "continuous reframe"
    assert contract.fallback_event("event_1")["phase_1"]["transition"] == "continuous reframe"
    invalid = _draft(contract)
    invalid["event_cards"]["event_1"]["phase_1"]["transition"] = "hard cut"
    with pytest.raises(ValueError, match=r"event_cards\.event_1: phase_1"):
        contract.bind(invalid)


def test_prompt_events_are_json_serializable_and_recovery_is_event_scoped():
    beats = [_beat(["E1"]), dict(_beat(["E2"]), beat_id="B2")]
    events = [
        _event("E1", "Mara grips the latch and pulls it down."),
        _event("E2", "The latch seats in the slot and the gate stays shut."),
    ]
    contract = build_source_camera_contract(beats, source_events=events)
    assert contract is not None
    prompt_events = contract.prompt_events()
    assert isinstance(prompt_events, list)
    assert [item["event_card"] for item in prompt_events] == ["event_1", "event_2"]
    json.dumps(prompt_events)
    assert set(contract.fallback_event("event_2")) == {"phase_1"}
    assert contract.fallback_event("event_9") is None
