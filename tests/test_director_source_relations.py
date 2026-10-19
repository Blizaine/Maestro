"""Regressions for source-actor support/contact relation validation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.physical_continuity import source_physical_state_issues
from services.director.source_contract import source_contract_issues

CAST = [
    {"character_id": "char_kitten", "speaker_name": "Kitten"},
    {"character_id": "char_dragon", "speaker_name": "Dragon"},
    {"character_id": "char_wolf", "speaker_name": "Wolf"},
]


def fact(subject_id: str, predicate: str, value: str) -> dict[str, str]:
    return {"subject_id": subject_id, "predicate": predicate, "value": value}


def row(
    opening: list[dict[str, str]],
    closing: list[dict[str, str]],
    transitions: list[dict[str, Any]] | None = None,
    *,
    sidecar: list[str | None] | None = None,
    cast: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "shot_id": "relation-case",
        "source_event_ids": ["E1"],
        "subjects_on_screen": cast if cast is not None else CAST,
        "physical_state": {
            "opening": opening,
            "transitions": transitions or [],
            "closing": closing,
        },
    }
    if sidecar is not None:
        result["_director_transition_event_ids"] = sidecar
    return result


def issues(source: str, plan_row: dict[str, Any]) -> list[str]:
    return source_physical_state_issues(
        [plan_row], [[{"event_id": "E1", "text": source}]],
    )


def support(subject: str, target: str) -> dict[str, str]:
    return fact(subject, "support", target)


def contact(subject: str, target: str) -> dict[str, str]:
    return fact(subject, "contact", target)


def test_reversed_ride_and_unrelated_owner_are_not_treated_as_rider_support():
    source = "Kitten rides Dragon."
    reversed_relation = row(
        [support("char_dragon", "char_kitten")],
        [support("char_dragon", "char_kitten")],
    )
    unrelated_owner = row(
        [support("char_wolf", "char_dragon")],
        [support("char_wolf", "char_dragon")],
    )

    for plan_row in (reversed_relation, unrelated_owner):
        found = issues(source, plan_row)
        assert any("source actor 'char_kitten' must own" in item for item in found), found


def test_wrong_support_target_is_rejected_but_correct_kitten_dragon_ride_passes():
    source = "Kitten rides Dragon."
    wrong_target = row(
        [support("char_kitten", "astride char_wolf")],
        [support("char_kitten", "astride char_wolf")],
    )
    correct_target = row(
        [support("char_kitten", "astride char_dragon")],
        [support("char_kitten", "astride char_dragon")],
    )

    found = issues(source, wrong_target)
    assert any("target does not match" in item for item in found), found
    assert issues(source, correct_target) == []


def test_contact_target_is_bound_to_named_source_actor_and_other_actor():
    source = "Kitten pats Dragon's snout."
    transition = {
        "subject_id": "char_kitten",
        "predicate": "contact",
        "from_value": "none",
        "to_value": "touching char_dragon snout",
        "action": "Kitten pats Dragon's snout",
    }
    correct = row(
        [contact("char_kitten", "none")],
        [contact("char_kitten", "touching char_dragon snout")],
        [transition],
        sidecar=["E1"],
    )
    wrong_target = row(
        [contact("char_kitten", "none")],
        [contact("char_kitten", "touching char_wolf snout")],
        [{**transition, "to_value": "touching char_wolf snout"}],
        sidecar=["E1"],
    )
    unrelated_owner = row(
        [contact("char_wolf", "none")],
        [contact("char_wolf", "touching char_dragon snout")],
        [{**transition, "subject_id": "char_wolf"}],
        sidecar=["E1"],
    )

    assert issues(source, correct) == []
    wrong = issues(source, wrong_target)
    assert any("target does not match" in item for item in wrong), wrong
    owner = issues(source, unrelated_owner)
    assert any("source actor 'char_kitten' must own" in item for item in owner), owner

    missing_transition = row(
        [contact("char_kitten", "none")],
        [contact("char_kitten", "touching char_dragon snout")],
    )
    missing = issues(source, missing_transition)
    assert any("target does not match" in item for item in missing), missing


def test_ambiguous_unresolvable_and_negated_source_relations_are_not_inferred():
    filled = [support("char_wolf", "char_dragon")]
    ambiguous = row(
        filled,
        filled,
    )
    ambiguous_source = "Kitten and Wolf ride Dragon together."
    assert issues(ambiguous_source, ambiguous) == []

    unresolvable_source = "The guest rides Dragon."
    unresolvable = row(filled, filled)
    assert issues(unresolvable_source, unresolvable) == []

    unresolved_target_source = "Kitten rides it."
    unresolved_target = row(
        [support("char_kitten", "char_wolf")],
        [support("char_kitten", "char_wolf")],
    )
    assert issues(unresolved_target_source, unresolved_target) == []

    negated = "Kitten does not ride Dragon."
    assert issues(negated, unresolved_target) == []


def test_boarding_may_open_grounded_when_same_source_event_proves_transition():
    source = "Kitten climbs aboard Dragon."
    transition = {
        "subject_id": "char_kitten",
        "predicate": "support",
        "from_value": "standing on ground",
        "to_value": "astride char_dragon",
        "action": "climbs aboard Dragon",
    }
    valid = row(
        [support("char_kitten", "standing on ground")],
        [support("char_kitten", "astride char_dragon")],
        [transition],
        sidecar=["E1"],
    )
    assert issues(source, valid) == []

    wrong_owner_transition = row(
        [support("char_kitten", "ground")],
        [support("char_kitten", "astride char_dragon")],
        [{**transition, "subject_id": "char_wolf"}],
        sidecar=["E1"],
    )
    owner_issues = issues(source, wrong_owner_transition)
    assert any("target does not match" in item for item in owner_issues), owner_issues

    wrong_target_transition = row(
        [support("char_kitten", "ground")],
        [support("char_kitten", "astride char_wolf")],
        [{**transition, "to_value": "astride char_wolf"}],
        sidecar=["E1"],
    )
    target_issues = issues(source, wrong_target_transition)
    assert any("target does not match" in item for item in target_issues), target_issues

    wrong_event_transition = row(
        [support("char_kitten", "ground")],
        [support("char_kitten", "astride char_dragon")],
        [transition],
        sidecar=["E2"],
    )
    event_issues = issues(source, wrong_event_transition)
    assert any("target does not match" in item for item in event_issues), event_issues


def test_ordinary_transition_cannot_replace_missing_opening_owner_fact():
    source = "Kitten rides Dragon."
    ordinary_transition = {
        "subject_id": "char_kitten",
        "predicate": "support",
        "from_value": "ground",
        "to_value": "astride char_dragon",
        "action": "Kitten rides Dragon",
    }
    plan_row = row(
        [support("char_wolf", "standing nearby")],
        [support("char_kitten", "astride char_dragon")],
        [ordinary_transition],
        sidecar=["E1"],
    )
    found = issues(source, plan_row)
    assert any("source actor 'char_kitten' must own" in item for item in found), found


def test_already_seated_ride_requires_target_relation_at_both_endpoints():
    source = "Kitten rides Dragon."
    already_seated = row(
        [support("char_kitten", "astride char_dragon")],
        [support("char_kitten", "astride char_dragon")],
    )
    assert issues(source, already_seated) == []

    grounded_opening = row(
        [support("char_kitten", "ground")],
        [support("char_kitten", "astride char_dragon")],
    )
    found = issues(source, grounded_opening)
    assert any("target does not match" in item for item in found), found


def test_multi_event_ride_then_dismount_checks_owned_trajectory_without_guessing_endpoint():
    source_events = [
        {"event_id": "E1", "text": "Kitten rides Dragon."},
        {"event_id": "E2", "text": "Kitten dismounts Dragon onto the ground."},
    ]
    plan_row = row(
        [support("char_kitten", "astride char_dragon")],
        [support("char_kitten", "on ground")],
        [{
            "subject_id": "char_kitten",
            "predicate": "support",
            "from_value": "astride char_dragon",
            "to_value": "on ground",
            "action": "Kitten dismounts Dragon onto the ground",
        }],
        sidecar=["E2"],
    )
    found = source_physical_state_issues([plan_row], [source_events])
    assert found == [], found


def test_single_event_slide_down_names_departed_support_not_closing_destination():
    source = "Kitten slides down Dragon's shoulder onto the ground."
    plan_row = row(
        [support("char_kitten", "char_dragon shoulder")],
        [support("char_kitten", "on ground")],
        [{
            "subject_id": "char_kitten",
            "predicate": "support",
            "from_value": "char_dragon shoulder",
            "to_value": "on ground",
            "action": source,
        }],
        sidecar=["E1"],
    )
    assert issues(source, plan_row) == []
    plan_row["physical_state"]["opening"][0]["value"] = "char_wolf shoulder"
    found = issues(source, plan_row)
    assert any("target does not match" in issue for issue in found), found


def test_wrong_owner_dismount_cannot_supply_missing_rider_relation():
    source_events = [
        {"event_id": "E1", "text": "Kitten rides Dragon."},
        {"event_id": "E2", "text": "Wolf dismounts Dragon onto the ground."},
    ]
    plan_row = row(
        [support("char_wolf", "astride char_dragon")],
        [support("char_wolf", "on ground")],
        [{
            "subject_id": "char_wolf",
            "predicate": "support",
            "from_value": "astride char_dragon",
            "to_value": "on ground",
            "action": "Wolf dismounts Dragon onto the ground",
        }],
        sidecar=["E2"],
    )
    found = source_physical_state_issues([plan_row], [source_events])
    assert any("source actor 'char_kitten' must own" in item for item in found), found


def test_explicitly_established_contact_keeps_exact_event_owner_and_target():
    source = "Maya grips the saddle strap."
    maya_cast = [{"character_id": "maya", "speaker_name": "Maya"}]
    established = row(
        [],
        [contact("maya", "saddle strap")],
        [{
            "subject_id": "maya",
            "predicate": "contact",
            "operation": "establish",
            "event_id": "E1",
            "from_value": None,
            "to_value": "saddle strap",
            "action": "grips the saddle strap",
        }],
        cast=maya_cast,
    )
    assert issues(source, established) == []


def test_source_contract_integration_uses_owner_target_guard():
    source = "Kitten rides Dragon."
    reversed_relation = row(
        [support("char_dragon", "char_kitten")],
        [support("char_dragon", "char_kitten")],
    )
    contract = {"events": [{"event_id": "E1", "text": source}]}
    found = source_contract_issues([reversed_relation], contract)
    assert any("source actor 'char_kitten' must own" in item for item in found), found


def test_untyped_legacy_source_contract_remains_unchanged():
    contract = {"events": [{"event_id": "E1", "text": "Kitten rides Dragon."}]}
    assert source_contract_issues(
        [{"scene_goal": "A brief connecting shot"}], contract, require_ids=False,
    ) == []
