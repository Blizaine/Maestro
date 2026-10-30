"""Pika H3 Flex defaults and attention routing remain separate from its sampler."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import pytest
# ruff: noqa: E402 -- repository app path bootstrap.

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

from models.minimax_h3.pika_flex import normalize_pika_flex_attention, validate_pika_flex_request
from models.minimax_h3.pika_flex_assets import pika_flex_capabilities, pika_flex_defaults


@pytest.mark.parametrize("mode", ["sage2", "sdpa", "sol", "sla"])
def test_requested_backend_survives_fixed_recipe_defaults(mode):
    for reference in (False, True):
        settings = pika_flex_defaults(reference, attention=mode)
        assert settings["override_attention"] == mode
        assert settings["num_inference_steps"] == 6
        assert settings["guidance_scale"] == 1
        assert settings["flow_shift"] == 12
        assert settings["audio_flow_shift"] == 3
        assert not settings["skip_steps_cache_type"]
        validate_pika_flex_request(settings)


@pytest.mark.parametrize("mode", [None, "", "auto", "default", "  SLA  "])
def test_missing_attention_records_sla_intention(mode):
    assert normalize_pika_flex_attention(mode) == "sla"


@pytest.mark.parametrize("reference", [False, True])
def test_api_defaults_and_native_capabilities_align_on_sla(reference):
    settings = pika_flex_defaults(reference)
    capabilities = pika_flex_capabilities(reference)
    model_type = "minimax_h3_ref2va_pika_flex" if reference else "minimax_h3_pika_flex"
    model_defaults = json.loads(
        (ROOT / "app" / "defaults" / f"{model_type}.json").read_text(encoding="utf-8")
    )

    assert settings["override_attention"] == "sla"
    assert model_defaults["override_attention"] == settings["override_attention"]
    assert model_defaults["model"]["sla_attention_default"] is True
    assert capabilities["sla_attention_default"] is True
    assert capabilities["sla_attention"] is True


def test_unsupported_attention_and_first_block_cache_stay_rejected():
    with pytest.raises(ValueError, match="supports Sage2"):
        validate_pika_flex_request({"override_attention": "veda"})
    with pytest.raises(ValueError, match="First Block.*step-skipping caches"):
        validate_pika_flex_request({"override_attention": "sla", "skip_steps_cache_type": "first_block"})


def _load_wgp_attention_helpers():
    source = (ROOT / "app" / "wgp.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    resolver = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "resolve_h3_attention_override"
    )
    selector = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "attn"
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "auto"
        and 'model_def.get("sol_attention"' in ast.get_source_segment(source, node)
    )
    select = ast.FunctionDef(
        name="select_attention",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="attn"),
                ast.arg(arg="model_def"),
                ast.arg(arg="override_attention_modes_supported"),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[selector, ast.Return(value=ast.Name(id="attn", ctx=ast.Load()))],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[resolver, select], type_ignores=[]))
    return compile(module, str(ROOT / "app" / "wgp.py"), "exec")


def test_native_missing_override_uses_sla_and_explicit_dense_modes_win():
    namespace = {"get_overridden_attention": lambda _model_type: None}
    exec(_load_wgp_attention_helpers(), namespace)
    pika = {"minimax_h3_pika_flex": True, "sla_attention_default": True}

    resolve = namespace["resolve_h3_attention_override"]
    assert resolve("pika", "", pika) == "sla"
    assert resolve("pika", None, pika) == "sla"
    assert resolve("pika", "auto", pika) == "sla"
    assert resolve("pika", "DEFAULT", pika) == "sla"
    assert resolve("pika", "  Sage2  ", pika) == "sage2"
    assert resolve("pika", "sdpa", pika) == "sdpa"
    assert resolve("ordinary", "", {}) is None

    namespace["get_overridden_attention"] = lambda _model_type: "sdpa"
    assert resolve("pika", "", pika) == "sdpa"
    assert resolve("pika", "auto", pika) == "sdpa"


@pytest.mark.parametrize("safe_dense", ["sage2", "sdpa"])
def test_native_unavailable_sla_default_falls_back_before_h3_dispatch(safe_dense):
    attention = ModuleType("shared.attention")
    attention.get_default_attention_mode = lambda: safe_dense
    attention.get_sla_attention_status = lambda: {
        "supported": False,
        "reason": "Triton unavailable",
    }
    attention.get_sol_attention_status = lambda: {"reason": "Triton unavailable"}
    messages = []
    namespace = {
        "get_auto_attention": lambda: safe_dense,
        "send_cmd": lambda *message: messages.append(message),
        "print": lambda *_args: None,
    }
    exec(_load_wgp_attention_helpers(), namespace)

    with patch.dict(sys.modules, {"shared.attention": attention}):
        result = namespace["select_attention"](
            "sla",
            {"sla_attention": True, "minimax_h3_pika_flex": True},
            ["sage2", "sdpa"],
        )
        explicit_sdpa = namespace["select_attention"](
            "sdpa",
            {"sla_attention": True, "minimax_h3_pika_flex": True},
            ["sage2", "sdpa"],
        )

    assert result == safe_dense
    assert explicit_sdpa == "sdpa"
    assert ("exit",) not in messages
    assert any(f"safe dense {safe_dense} backend" in message[1].lower() for message in messages)


def test_supported_sla_default_stays_sparse():
    namespace = {
        "get_auto_attention": lambda: "sage2",
        "send_cmd": lambda *_args: None,
        "print": lambda *_args: None,
    }
    exec(_load_wgp_attention_helpers(), namespace)
    result = namespace["select_attention"](
        "sla",
        {"sla_attention": True, "minimax_h3_pika_flex": True},
        ["sage2", "sdpa", "sla"],
    )
    assert result == "sla"
