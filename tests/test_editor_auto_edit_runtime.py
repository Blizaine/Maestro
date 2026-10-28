"""No inference/network: provider privacy and Editor planner resource ownership."""
import ast
import asyncio
from contextlib import nullcontext
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest
from services.editor_auto_edit_runtime import editor_auto_edit_capabilities, editor_auto_edit_writer


class RouteError(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail


def fake_writer(provider="local", loaded=True, remote=""):
    calls = []
    llm = SimpleNamespace(_remote_url=remote, calls=calls,
        get_status=lambda: {"loaded": loaded, "provider": provider, "model_id": "installed-writer"},
        keep_loaded=nullcontext,
        generate=lambda *args, **kwargs: calls.append((args, kwargs)) or '{"summary":"Review","actions":[],"missing_media":[]}',
    )
    return llm


@pytest.mark.parametrize("provider,url,allowed", [
    ("local", "", True), ("remote", "http://192.168.1.3:1234/v1", True),
    ("remote", "http://localhost:1234", True), ("remote", "https://host.tail123.ts.net", True),
    ("remote", "https://hosted.example.com/v1", False), ("openai", "", False),
    ("anthropic", "", False),
])
def test_writer_boundary(provider, url, allowed):
    llm = fake_writer(provider, remote=url)
    assert editor_auto_edit_capabilities(llm)["writer_available"] is allowed
    assert llm.calls == []
    assert editor_auto_edit_capabilities(llm)["director_available"] is True


def test_writer_uses_loaded_model_and_detects_changed_identity():
    llm = fake_writer()
    write = editor_auto_edit_writer(llm)
    write(ledger={"assets": []}, request={"brief": "Keep illustrative style"}, schema={"type": "object"})
    args, kwargs = llm.calls[0]
    assert "Keep illustrative style" in args[0]
    assert kwargs["enable_thinking"] is False and kwargs["max_new_tokens"] == 6000
    assert kwargs["json_schema"] == {"type": "object"}
    llm.get_status = lambda: {"loaded": True, "provider": "local", "model_id": "different"}
    with pytest.raises(ValueError, match="writer changed"):
        write(ledger={}, request={}, schema={})
    assert len(llm.calls) == 1


def test_applied_plan_audit_survives_project_save_and_is_bounded():
    from services.editor_projects import create_editor_project, normalize_editor_project, EditorProjectError
    project = create_editor_project()
    project["auto_edit_history"] = [{"proposal_id": "proposal-0001", "action_ids": ["a1"],
        "summary": "Shorter opening", "base_revision": 1, "applied_at": 123}]
    assert normalize_editor_project(project)["auto_edit_history"] == project["auto_edit_history"]
    project["auto_edit_history"] *= 21
    with pytest.raises(EditorProjectError, match="at most 20"):
        normalize_editor_project(project)


def route_fixture(monkeypatch, tmp_path, *, existing=False, loaded=True):
    import services
    import services.editor_auto_edit as service
    llm = fake_writer(loaded=loaded)
    monkeypatch.setitem(sys.modules, "services.llm_service", llm)
    monkeypatch.setattr(services, "llm_service", llm, raising=False)
    project = {"id": "project-0001", "workspace": "Fixture", "revision": 4}
    calls = []
    if existing:
        monkeypatch.setattr(service, "get_editor_auto_edit_proposal", lambda *_: {"status": "ready"})
    else:
        def missing(*_):
            raise service.EditorAutoEditError("not found")
        monkeypatch.setattr(service, "get_editor_auto_edit_proposal", missing)
    monkeypatch.setattr(service, "create_editor_auto_edit_proposal", lambda *args, **kwargs: calls.append(kwargs) or {"status": "ready"})
    lock = threading.Lock()
    def guard():
        if lock.locked():
            raise RouteError(409, "busy")
    tree = ast.parse((Path(__file__).parents[1] / "app/launch.py").read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "editor_auto_edit_plan")
    node.decorator_list = []
    for arg in node.args.args:
        arg.annotation = None
    scope = {"asyncio": asyncio, "HTTPException": RouteError, "load_editor_project": lambda *_: project,
             "_editor_save_root": lambda: str(tmp_path), "_get_active_workspace": lambda: "Fixture",
             "_editor_auto_edit_directory": lambda: tmp_path, "_gen_lock": lock,
             "_guard_interactive_llm_against_generation": guard}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "editor-route", "exec"), scope)
    body = {"workspace": "Fixture", "project_id": project["id"], "revision": 4,
            "proposal_id": "intent-0001", "mode": "polish", "brief": "Keep the quiet opening", "client_signature": "signature"}
    async def submit():
        async def json_body():
            return body
        return await scope["editor_auto_edit_plan"](SimpleNamespace(json=json_body))
    return submit, body, calls, lock, service


def test_saved_revision_and_busy_generation_guards(monkeypatch, tmp_path):
    submit, body, calls, lock, _ = route_fixture(monkeypatch, tmp_path)
    body["revision"] = 3
    with pytest.raises(RouteError) as error:
        asyncio.run(submit())
    assert error.value.status_code == 409 and calls == []
    body["revision"] = 4
    lock.acquire()
    try:
        with pytest.raises(RouteError) as error:
            asyncio.run(submit())
        assert error.value.status_code == 409 and calls == []
    finally:
        lock.release()


def test_existing_receipt_and_director_need_no_writer(monkeypatch, tmp_path):
    submit, body, calls, _, _ = route_fixture(monkeypatch, tmp_path, existing=True, loaded=False)
    assert asyncio.run(submit())["status"] == "ready"
    assert "writer" not in calls[0]
    submit, body, calls, _, _ = route_fixture(monkeypatch, tmp_path, loaded=False)
    body["mode"] = "director"
    assert asyncio.run(submit())["status"] == "ready" and "writer" not in calls[0]


def test_disconnected_browser_keeps_lease_until_result_persisted(monkeypatch, tmp_path):
    submit, _, calls, lock, service = route_fixture(monkeypatch, tmp_path)
    started, finish = threading.Event(), threading.Event()
    def delayed(*args, **kwargs):
        started.set()
        assert finish.wait(5)
        calls.append("persisted")
        return {"status": "ready"}
    monkeypatch.setattr(service, "create_editor_auto_edit_proposal", delayed)
    async def run():
        task = asyncio.create_task(submit())
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(.005)
        assert started.is_set() and lock.locked()
        task.cancel()
        await asyncio.sleep(.01)
        assert lock.locked(), "A client disconnect cannot release resources still used by the planner"
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not lock.locked() and calls == ["persisted"]
    asyncio.run(run())
