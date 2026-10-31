from __future__ import annotations

from pathlib import Path

import services.agent_access.contracts as contracts
from services.agent_access.contracts import load_contract_index


def _field_map(hints: dict) -> dict[str, dict]:
    return {field["name"]: field for field in hints["fields"]}


def test_source_hints_cover_studio_director_and_editor_handler_shapes(tmp_path: Path):
    source = '''
@api.post("/api/v1/generate")
async def generate(request: Request):
    body = await request.json()
    if _generation_request_uses_serial_auto_planner(body):
        return _enqueue_deferred_generation_preparation(body)
    return await _prepare_generation_submission(body)

async def _prepare_generation_submission(body: dict):
    if not body.get("model_type"):
        raise ValueError("model_type is required")
    prompt = body.get("prompt")
    mode = body.get("generation_mode")
    duration = int(body.get("video_length", 81))
    return prompt, mode, duration

@api.post("/api/v1/director/pipeline/start")
async def director_pipeline_start(request: Request):
    body = _strip_private_director_fields(await request.json())
    return start_pipeline(body)

def _strip_private_director_fields(payload):
    return payload

@api.put("/api/v1/editor/projects/{project_id}")
async def editor_project_update(project_id: str, request: Request):
    body = await request.json()
    workspace = body.get("workspace") or _get_active_workspace()
    project = body.get("project") if isinstance(body.get("project"), dict) else body
    project = dict(project)
    project["id"] = project_id
    return save_editor_project(workspace, project)

@api.put("/api/v1/director/queue/{entry_id}")
async def director_queue_update(entry_id: str, request: Request):
    body = await request.json()
    params = body.get("params") if isinstance(body, dict) else None
    return update_director_queue_entry(entry_id, params)

@api.put("/api/v1/director/pipelines/{pid}/clips/{clip_index}/tag")
async def director_tag(pid: str, clip_index: int, request: Request):
    body = await request.json()
    return update_clip_tag(pid, clip_index, body.get("tag"))
'''
    (tmp_path / "launch.py").write_text(source, encoding="utf-8")
    index = load_contract_index(tmp_path)

    studio = _field_map(index.describe("POST", "/api/v1/generate"))
    assert {"model_type", "prompt", "generation_mode", "video_length"} <= set(studio)
    assert studio["video_length"] == {"name": "video_length", "type": "integer", "required": False, "default": 81}
    assert studio["model_type"]["required"] is False  # guarded with .get() before an error

    director_start = index.describe("POST", "/api/v1/director/pipeline/start")
    assert director_start["status"] == "available"
    assert director_start["fields"] == []  # native planner is outside the permitted same-file trace
    queue_update = _field_map(index.describe("PUT", "/api/v1/director/queue/{entry_id}"))
    assert "params" in queue_update
    clip_tag = _field_map(index.describe("PUT", "/api/v1/director/pipelines/{pid}/clips/{clip_index}/tag"))
    assert "tag" in clip_tag

    editor = _field_map(index.describe("PUT", "/api/v1/editor/projects/{project_id}"))
    assert {"workspace", "project"} <= set(editor)
    assert editor["project"]["type"] == "object"


def test_hints_follow_bounded_same_file_helpers_and_mark_truncation(tmp_path: Path):
    source = '''
@api.post("/api/v1/deep")
async def route(request: Request):
    body = await request.json()
    return _helper_1(body)

def _helper_1(value):
    return _helper_2(value)

def _helper_2(value):
    return _helper_3(value)

def _helper_3(value):
    return _helper_4(value)

def _helper_4(value):
    return _helper_5(value)

def _helper_5(value):
    visible = value.get("visible_at_depth_five", 4)
    return _helper_6(value)

def _helper_6(value):
    hidden = value.get("beyond_depth_limit")
    return hidden
'''
    (tmp_path / "launch.py").write_text(source, encoding="utf-8")
    hints = load_contract_index(tmp_path).describe("POST", "/api/v1/deep")
    fields = _field_map(hints)
    assert "visible_at_depth_five" in fields
    assert "beyond_depth_limit" not in fields
    assert hints["truncated"] is True


def test_helper_function_count_is_bounded(tmp_path: Path):
    helpers = "\n".join(
        f"def _helper_{index}(body):\n    return body.get('field_{index}')"
        for index in range(20)
    )
    calls = "\n".join(f"    _helper_{index}(body)" for index in range(20))
    source = f'''
@api.post("/api/v1/branching")
async def route(request: Request):
    body = await request.json()
{calls}
{helpers}
'''
    (tmp_path / "launch.py").write_text(source, encoding="utf-8")
    hints = load_contract_index(tmp_path).describe("POST", "/api/v1/branching")
    assert hints["truncated"] is True
    assert len(hints["fields"]) <= 96
    assert len(hints["fields"]) < 20


def test_helper_call_count_is_bounded(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(contracts, "_MAX_HELPER_FUNCTIONS", 200)
    helpers = "\n".join(
        f"def _call_helper_{index}(body):\n    return body.get('call_field_{index}')"
        for index in range(100)
    )
    calls = "\n".join(f"    _call_helper_{index}(body)" for index in range(100))
    source = f'''
@api.post("/api/v1/call-bound")
async def route(request: Request):
    body = await request.json()
{calls}
{helpers}
'''
    (tmp_path / "launch.py").write_text(source, encoding="utf-8")
    hints = load_contract_index(tmp_path).describe("POST", "/api/v1/call-bound")
    assert hints["truncated"] is True
    assert len(hints["fields"]) <= contracts._MAX_HELPER_CALLS


def test_source_byte_and_ast_node_limits_disable_hints(tmp_path: Path, monkeypatch):
    launch = tmp_path / "launch.py"
    launch.write_text(
        '@api.post("/api/v1/limited")\nasync def route(request: Request):\n    body = await request.json()\n    return body.get("name")\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(contracts, "_MAX_SOURCE_BYTES", 8)
    assert load_contract_index(tmp_path).status == "unavailable"

    monkeypatch.setattr(contracts, "_MAX_SOURCE_BYTES", 16 * 1024 * 1024)
    monkeypatch.setattr(contracts, "_MAX_AST_NODES", 5)
    contracts._CACHE.clear()
    assert load_contract_index(tmp_path).status == "unavailable"


def test_direct_request_json_expression_can_seed_a_same_file_helper(tmp_path: Path):
    source = '''
@api.post("/api/v1/direct-json")
async def route(request: Request):
    return _accept(await request.json())

def _accept(payload):
    return payload.get("direct_field", 3)
'''
    (tmp_path / "launch.py").write_text(source, encoding="utf-8")
    hints = load_contract_index(tmp_path).describe("POST", "/api/v1/direct-json")
    assert _field_map(hints)["direct_field"] == {
        "name": "direct_field", "type": "integer", "required": False, "default": 3,
    }


def test_director_start_traces_only_the_allowlisted_external_helper(tmp_path: Path):
    launch = '''
@api.post("/api/v1/director/pipeline/start")
async def director_pipeline_start(request: Request):
    body = _strip_private_director_fields(await request.json())
    return await asyncio.to_thread(start_pipeline, body)

def _strip_private_director_fields(payload):
    return payload
'''
    director = '''
raise RuntimeError("this AST source must not be executed")

def start_pipeline(params: dict):
    params.pop("workspace", None)
    params.pop("_director_planning_checkpoint", None)
    _validate_director_models(params)
    profile = _create_director_video_execution_profile(params)
    prepared_plans = params.get("prepared_clip_plans")
    prepared_timeline = params.get("prepared_planned_clips")
    auto_mode = params.get("auto_mode", True)
    video_model = params.get("video_model")
    return profile, prepared_plans, prepared_timeline, auto_mode, video_model

def _validate_director_models(params):
    if not params.get("video_model"):
        raise ValueError("video_model required")
    return params.get("image_model")

def _create_director_video_execution_profile(params):
    return {"quality": params.get("video_quality", "balanced")}
'''
    (tmp_path / "launch.py").write_text(launch, encoding="utf-8")
    services = tmp_path / "services"
    services.mkdir()
    (services / "director_pipeline.py").write_text(director, encoding="utf-8")
    (services / "unrelated_private.py").write_text(
        'def start_pipeline(params):\n    return params.get("not_allowlisted")\n',
        encoding="utf-8",
    )

    hints = load_contract_index(tmp_path).describe("POST", "/api/v1/director/pipeline/start")
    fields = _field_map(hints)
    assert hints["status"] == "available"
    assert hints["sources"] == ["launch.py", "services/director_pipeline.py"]
    assert {
        "workspace",
        "prepared_clip_plans",
        "prepared_planned_clips",
        "auto_mode",
        "video_model",
        "image_model",
        "video_quality",
    } <= set(fields)
    assert fields["auto_mode"]["type"] == "boolean"
    assert fields["auto_mode"]["default"] is True
    assert "type" not in fields["workspace"]  # a null default is not an accepted-type contract
    assert "_director_planning_checkpoint" not in fields
    assert "not_allowlisted" not in repr(hints)


def test_llm_derived_json_is_not_treated_as_request_body_and_sensitive_defaults_are_hidden(tmp_path: Path):
    secret = "this-source-secret-must-never-be-returned"
    long_prompt = "This source prompt is longer than the safe default limit and must not be returned."
    source = f'''
API_TOKEN = {secret!r}

@api.post("/api/v1/plan")
async def plan(request: Request):
    body = await request.json()
    user_prompt = body.get("prompt", {long_prompt!r})
    label = body.get("label", "Storyboard")
    credential = body.get("api_token", {secret!r})
    generated_json = json.loads(await llm.complete(user_prompt))
    derived_only = generated_json.get("planner_internal_field")
    return user_prompt, label, credential, derived_only
'''
    (tmp_path / "launch.py").write_text(source, encoding="utf-8")
    # A nearby private file is deliberately irrelevant; the parser only opens launch.py.
    (tmp_path / "private.json").write_text('{"private_field":"do not read"}', encoding="utf-8")
    hints = load_contract_index(tmp_path).describe("POST", "/api/v1/plan")
    fields = _field_map(hints)
    assert set(fields) == {"prompt", "label"}
    assert fields["label"]["default"] == "Storyboard"
    assert fields["prompt"]["type"] == "string"
    assert "default" not in fields["prompt"]
    rendered = repr(hints)
    assert secret not in rendered
    assert long_prompt not in rendered
    assert "planner_internal_field" not in rendered


def test_required_is_only_a_literal_subscript_without_a_detected_fallback(tmp_path: Path):
    source = '''
@api.post("/api/v1/required")
async def route(request: Request):
    body = await request.json()
    required_value = body["required_value"]
    optional_value = body["optional_value"]
    if "guarded_value" in body:
        guarded = body["guarded_value"]
    if body.get("fallback_value"):
        fallback = body["fallback_value"]
    return required_value, optional_value, guarded, fallback
'''
    (tmp_path / "launch.py").write_text(source, encoding="utf-8")
    fields = _field_map(load_contract_index(tmp_path).describe("POST", "/api/v1/required"))
    assert fields["required_value"]["required"] is True
    assert fields["optional_value"]["required"] is True
    assert fields["guarded_value"]["required"] is False
    assert fields["fallback_value"]["required"] is False


def test_parser_does_not_execute_source_and_missing_source_degrades_gracefully(tmp_path: Path):
    source = '''
raise RuntimeError("launch source must not be executed")
@api.post("/api/v1/safe")
async def route(request: Request):
    body = await request.json()
    return body.get("name", "Untitled")
'''
    (tmp_path / "launch.py").write_text(source, encoding="utf-8")
    hints = load_contract_index(tmp_path).describe("POST", "/api/v1/safe")
    assert hints["status"] == "available"
    assert _field_map(hints)["name"]["default"] == "Untitled"

    missing = load_contract_index(tmp_path / "missing")
    assert missing.describe("POST", "/api/v1/safe")["status"] == "unavailable"


def test_source_index_cache_is_keyed_by_launch_hash(tmp_path: Path):
    launch = tmp_path / "launch.py"
    launch.write_text(
        '@api.post("/api/v1/cache")\nasync def route(request: Request):\n    body = await request.json()\n    return body.get("first")\n',
        encoding="utf-8",
    )
    first = load_contract_index(tmp_path)
    assert load_contract_index(tmp_path) is first
    launch.write_text(
        '@api.post("/api/v1/cache")\nasync def route(request: Request):\n    body = await request.json()\n    return body.get("second")\n',
        encoding="utf-8",
    )
    second = load_contract_index(tmp_path)
    assert second is not first
    assert "second" in _field_map(second.describe("POST", "/api/v1/cache"))
