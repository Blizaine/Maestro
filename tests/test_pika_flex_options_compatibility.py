"""Execute actual option/validation helpers without importing the GPU server."""
from __future__ import annotations

import ast
from pathlib import Path
import sys

import pytest

# ruff: noqa: E402 -- app imports follow the repository path bootstrap.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.minimax_h3.pika_flex import validate_pika_flex_request


def turbo_option():
    tree = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_minimax_h3_turbo_option")
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {}
    exec(compile(module, "launch.py", "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize("reference", [False, True])
def test_pika_options_do_not_offer_incompatible_turbo(reference):
    definition = {"architecture": "minimax_h3", "minimax_h3_pika_flex": True,
                  "omni_reference": reference}
    assert turbo_option()(definition) is None


@pytest.mark.parametrize("reference", [False, True])
def test_standard_h3_retains_managed_turbo(reference):
    definition = {"architecture": "minimax_h3", "omni_reference": reference}
    option = turbo_option()(definition)
    assert option is not None and option["presets"]


@pytest.mark.parametrize("attention", ["sla", "sol", "sage2", "sdpa"])
def test_first_block_rejection_is_actionable_and_attention_remains_valid(attention):
    request = {"override_attention": attention, "skip_steps_cache_type": "first_block"}
    with pytest.raises(ValueError, match="Disable First Block") as error:
        validate_pika_flex_request(request)
    assert "additional LoRAs" not in str(error.value)
    request["skip_steps_cache_type"] = ""
    validate_pika_flex_request(request)


@pytest.mark.parametrize("payload", [
    {"activated_loras": ["extra.safetensors"]},
    {"loras_selected": ["extra.safetensors"]},
    {"minimax_h3_turbo_mode": True},
    {"minimax_h3_pdd": True},
])
def test_real_extra_loras_and_accelerators_still_rejected(payload):
    with pytest.raises(ValueError, match="additional LoRAs or accelerators"):
        validate_pika_flex_request(payload)
