"""Source contracts for the Worlds-to-WanGP integration boundary.

WanGP imports the UI, CUDA, and memory-management stack at module import time,
so these tests inspect its syntax tree instead of importing the application.
They pin the ordering and argument handoff that the injected Worlds callbacks
depend on.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_ast(relative_path: str) -> tuple[str, ast.Module]:
    source = (ROOT / relative_path).read_text(encoding="utf-8-sig")
    return source, ast.parse(source, filename=relative_path)


def _function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"Could not find function {name!r}")


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def test_world_sidefile_is_captured_before_dict_unwrap_and_persisted_after_save():
    _source, tree = _read_ast("app/wgp.py")
    generate_video = _function(tree, "generate_video")
    nodes = list(ast.walk(generate_video))

    sidefile_captures = [
        node for node in nodes
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "model_side_files" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and _call_name(node.value) == "get"
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "samples"
        and node.value.args
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value == "side_files"
    ]
    unwraps = [
        node for node in nodes
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "samples" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and _call_name(node.value) == "get"
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "samples"
        and node.value.args
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value == "x"
    ]
    persistence = [
        node for node in nodes
        if isinstance(node, ast.Call)
        and _call_name(node) == "persist_world_sidecar"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id == "model_side_files"
    ]
    saves = [
        node for node in nodes
        if isinstance(node, ast.Call) and _call_name(node) == "save_video"
    ]

    assert len(sidefile_captures) == 1
    assert len(unwraps) == 1
    assert len(persistence) == 1
    assert saves, "Expected the primary media save to remain in generate_video"
    assert sidefile_captures[0].lineno < unwraps[0].lineno
    assert min(node.lineno for node in saves) < persistence[0].lineno


def test_world_runtime_acknowledgement_and_generation_settings_reach_model_calls():
    _source, tree = _read_ast("app/wgp.py")
    generate_video = _function(tree, "generate_video")
    calls = [node for node in ast.walk(generate_video) if isinstance(node, ast.Call)]

    runtime_context = [node for node in calls if _call_name(node) == "world_runtime_context"]
    generation_context = [node for node in calls if _call_name(node) == "world_generation_context"]
    model_loads = [node for node in calls if _call_name(node) == "load_models"]
    sticky_calls = [node for node in calls if _call_name(node) == "call_with_sticky_interrupt"]
    model_kwargs_copies = [
        node for node in ast.walk(generate_video)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "model_load_kwargs" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and _call_name(node.value) == "dict"
        and node.value.args
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == "model_kwargs"
    ]

    assert len(runtime_context) == 1
    assert len(generation_context) == 1
    assert model_loads
    assert sticky_calls
    assert model_kwargs_copies
    assert runtime_context[0].lineno < model_loads[0].lineno
    assert model_kwargs_copies[0].lineno < model_loads[0].lineno
    assert any(
        keyword.arg is None
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "model_load_kwargs"
        for node in model_loads
        for keyword in node.keywords
    ), "The explicit Worlds acknowledgement must flow into load_models"
    assert generation_context[0].lineno < min(node.lineno for node in sticky_calls)
    assert any(
        keyword.arg is None
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "world_generate_kwargs"
        for node in sticky_calls
        for keyword in node.keywords
    ), "The validated world_output mode must flow into model.generate"


def test_world_jobs_disable_preview_and_register_as_a_normal_model_family():
    _wgp_source, wgp_tree = _read_ast("app/wgp.py")
    handler_source, _handler_tree = _read_ast("app/models/hy_nomas/hy_nomas_handler.py")
    generate_video = _function(wgp_tree, "generate_video")

    preview_guards = [
        node for node in ast.walk(generate_video)
        if isinstance(node, ast.If)
        and any(
            isinstance(part, ast.Compare)
            and isinstance(part.left, ast.Call)
            and _call_name(part.left) == "get"
            and isinstance(part.left.func, ast.Attribute)
            and isinstance(part.left.func.value, ast.Name)
            and part.left.func.value.id == "model_def"
            and part.left.args
            and isinstance(part.left.args[0], ast.Constant)
            and part.left.args[0].value == "supports_latent_preview"
            and any(isinstance(operator, ast.Is) for operator in part.ops)
            and any(isinstance(comparator, ast.Constant) and comparator.value is False for comparator in part.comparators)
            for part in ast.walk(node.test)
        )
        and any(
            isinstance(statement, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "generation_preview_mode" for target in statement.targets)
            and isinstance(statement.value, ast.Constant)
            and statement.value.value == "off"
            for statement in node.body
        )
    ]
    assert preview_guards, "Models that declare supports_latent_preview=False must select preview mode off"

    family_registration = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "family_handlers"
        and node.func.attr == "append"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "models.hy_nomas.hy_nomas_handler"
        for node in ast.walk(wgp_tree)
    )
    assert family_registration, "Hy-Nomas must remain registered through the standard family-handler path"

    handler_source_lines = handler_source.splitlines()
    assert any(
        '"supports_latent_preview": False' in line
        for line in handler_source_lines
    ), "Hy-Nomas must declare preview support as disabled"
