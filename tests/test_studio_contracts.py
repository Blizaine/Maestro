"""Bounded Studio request/model-option contracts."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _return_dict(function: ast.FunctionDef) -> ast.Dict:
    return next(
        node.value for node in ast.walk(function)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    )


def test_wan_expert_defaults_come_from_resolved_model_settings():
    tree = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
    endpoint = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_model_options"
    )
    values = {
        key.value: value
        for key, value in zip(_return_dict(endpoint).keys, _return_dict(endpoint).values)
        if isinstance(key, ast.Constant)
    }

    namespace = {
        "_ui_defaults": {
            "guidance_phases": 2,
            "guidance2_scale": 1.0,
            "switch_threshold": 900,
        },
        "md": {
            "guidance_max_phases": 3,
            "guidance_phases": 1,
            "guidance2_scale": 5.0,
            "switch_threshold": 500,
        },
    }
    resolved = {
        name: eval(compile(ast.Expression(values[name]), "model-options", "eval"), namespace)
        for name in (
            "default_guidance_phases",
            "default_guidance2_scale",
            "default_switch_threshold",
        )
    }
    assert resolved == {
        "default_guidance_phases": 2,
        "default_guidance2_scale": 1.0,
        "default_switch_threshold": 900,
    }
    assert "guidance_max_phases" in values, "Capability maximum remains a separate response field"


def test_typed_reference_drive_is_routed_before_flex_request_validation():
    source = (ROOT / "app/launch.py").read_text(encoding="utf-8")
    prepare = source[source.index("async def _prepare_generation_submission") :]

    assert prepare.index("split_exact_drive_audio_reference") < prepare.index(
        "validate_pika_flex_single_pass_request(body, _generation_model_def)"
    )
    assert 'body["audio_prompt_type"] = "AD"' in prepare
    assert 'body["audio_guide"] = typed_drive_path' in prepare
    assert 'body.pop("audio_guide", None)' in prepare
    assert '== "AD"' in prepare, "The explicit legacy exact-drive API remains accepted"
