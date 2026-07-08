"""Unit tests for the JSON-grammar gating in BasePlanner._call_llm_json.

Contract under test (no llama-server involved — stub generate fns):
  1. thinking-off call (budget resolves to 0): schema attached on the
     FIRST attempt.
  2. thinking-on call (explicit budget > 0): first attempt unconstrained,
     schema attached on the parse-failure retry (with thinking off).
  3. no schema given: retry still gets the generic array-of-objects grammar.
  4. a gen_fn that rejects json_schema (TypeError, like an old server or
     odd provider wrapper) degrades to the historical unconstrained call.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from services.director.planners.base import BasePlanner, _GENERIC_ARRAY_SCHEMA  # noqa: E402


class _Planner(BasePlanner):
    skill_type = "test"

    def plan(self, **kwargs):  # pragma: no cover - abstract filler
        raise NotImplementedError


SCHEMA = {"type": "array", "items": {"type": "object"}, "maxItems": 3}
VALID = '[{"scene_goal": "x"}]'


def _planner_with(responses):
    """Planner whose streaming fn pops canned responses and records kwargs."""
    calls = []

    def fake_gen(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    return _Planner(llm_generate=fake_gen, llm_generate_streaming=fake_gen), calls


def test_schema_on_first_attempt_when_thinking_off():
    planner, calls = _planner_with([VALID])
    out = planner._call_llm_json("u", "s", thinking_budget=0, json_schema=SCHEMA)
    assert out == [{"scene_goal": "x"}]
    assert len(calls) == 1
    assert calls[0]["json_schema"] == SCHEMA
    assert calls[0]["enable_thinking"] is False


def test_thinking_on_first_attempt_unconstrained_retry_constrained():
    planner, calls = _planner_with(["not json at all ((", VALID])
    out = planner._call_llm_json("u", "s", thinking_budget=4096, json_schema=SCHEMA)
    assert out == [{"scene_goal": "x"}]
    assert len(calls) == 2
    assert "json_schema" not in calls[0]          # thinking-on path untouched
    assert calls[1]["json_schema"] == SCHEMA      # retry is grammar-locked
    assert calls[1]["enable_thinking"] is False
    assert calls[1]["thinking_budget"] == 0


def test_retry_gets_generic_grammar_without_schema():
    planner, calls = _planner_with(["garbage ((", VALID])
    out = planner._call_llm_json("u", "s", thinking_budget=4096)
    assert out == [{"scene_goal": "x"}]
    assert calls[1]["json_schema"] == _GENERIC_ARRAY_SCHEMA


def test_degrades_when_gen_fn_rejects_schema():
    calls = []
    responses = [VALID]

    def legacy_gen(**kwargs):
        if "json_schema" in kwargs:
            raise TypeError("unexpected keyword argument 'json_schema'")
        calls.append(kwargs)
        return responses.pop(0)

    planner = _Planner(llm_generate=legacy_gen, llm_generate_streaming=legacy_gen)
    out = planner._call_llm_json("u", "s", thinking_budget=0, json_schema=SCHEMA)
    assert out == [{"scene_goal": "x"}]
    assert len(calls) == 1
    assert "json_schema" not in calls[0]          # constrained call degraded cleanly


def test_degraded_retry_matches_historical_call():
    responses = ["garbage ((", VALID]
    calls = []

    def legacy_gen(**kwargs):
        if "json_schema" in kwargs:
            raise TypeError("unexpected keyword argument 'json_schema'")
        calls.append(kwargs)
        return responses.pop(0)

    planner = _Planner(llm_generate=legacy_gen, llm_generate_streaming=legacy_gen)
    out = planner._call_llm_json("u", "s", thinking_budget=4096)
    assert out == [{"scene_goal": "x"}]
    # degraded retry restores the historical thinking budget and drops
    # the grammar-only kwargs entirely
    assert len(calls) == 2
    assert calls[1]["thinking_budget"] == 2048
    assert "enable_thinking" not in calls[1]


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
