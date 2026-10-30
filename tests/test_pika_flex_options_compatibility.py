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


@pytest.mark.parametrize("selection_key", ["activated_loras", "loras_selected"])
def test_extra_loras_require_experimental_opt_in(selection_key):
    with pytest.raises(ValueError, match="Enable minimax_h3_experimental_loras"):
        validate_pika_flex_request({selection_key: ["extra.safetensors"]})


@pytest.mark.parametrize("selection_key", ["activated_loras", "loras_selected"])
@pytest.mark.parametrize("enabled", [False, None, 1, "true"])
def test_extra_loras_require_literal_true(selection_key, enabled):
    with pytest.raises(ValueError, match="Enable minimax_h3_experimental_loras"):
        validate_pika_flex_request({
            selection_key: ["extra.safetensors"],
            "minimax_h3_experimental_loras": enabled,
        })


@pytest.mark.parametrize("selection_key", ["activated_loras", "loras_selected"])
def test_ordinary_loras_allowed_with_explicit_opt_in(selection_key):
    validate_pika_flex_request({
        selection_key: ["extra.safetensors"],
        "minimax_h3_experimental_loras": True,
    })


@pytest.mark.parametrize("accelerator", ["minimax_h3_turbo_mode", "minimax_h3_pdd"])
@pytest.mark.parametrize("experimental_loras", [False, True])
def test_accelerators_rejected_even_with_lora_opt_in(accelerator, experimental_loras):
    with pytest.raises(ValueError, match="additional LoRAs or accelerators"):
        validate_pika_flex_request({
            accelerator: True,
            "minimax_h3_experimental_loras": experimental_loras,
        })
