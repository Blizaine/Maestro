from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import importlib.util
import unittest

if importlib.util.find_spec("mcp") is None:
    raise unittest.SkipTest("MCP protocol tests run in the separate Agent Access environment")

import pytest
from starlette.testclient import TestClient

from services.agent_access.mcp_server import create_mcp_server


class BackendError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.status_code = status


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.spec: dict[str, Any] = {
            "openapi": "3.0.3",
            "components": {
                "schemas": {
                    "Generation": {
                        "type": "object",
                        "required": ["model_type", "prompt"],
                        "properties": {
                            "model_type": {"type": "string", "enum": ["test_video"]},
                            "prompt": {"type": "string", "minLength": 1},
                            "asset": {"type": "string"},
                        },
                        "additionalProperties": False,
                    }
                }
            },
            "paths": {
                "/api/v1/system-stats": {
                    "get": {
                        "summary": "Live workstation status",
                        "responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}},
                    }
                },
                "/api/v1/models": {
                    "get": {
                        "summary": "List available models",
                        "responses": {"200": {"content": {"application/json": {"schema": {"type": "array"}}}}},
                    }
                },
                "/api/v1/model-options/{model_type}": {
                    "get": {
                        "summary": "Read model capabilities",
                        "parameters": [
                            {"name": "model_type", "in": "path", "required": True, "schema": {"type": "string"}}
                        ],
                        "responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}},
                    }
                },
                "/api/v1/defaults/{model_type}": {
                    "get": {
                        "summary": "Read model defaults",
                        "parameters": [
                            {"name": "model_type", "in": "path", "required": True, "schema": {"type": "string"}}
                        ],
                        "responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}},
                    }
                },
                "/api/v1/generate": {
                    "post": {
                        "summary": "Start Studio generation",
                        "requestBody": {
                            "required": True,
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Generation"}}},
                        },
                        "responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}},
                    }
                },
                "/api/v1/director/pipeline/start": {
                    "post": {
                        "summary": "Start a Director pipeline",
                        "responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}},
                    }
                },
                "/api/v1/editor/projects/{project_id}": {
                    "put": {
                        "summary": "Update an Editor project",
                        "parameters": [
                            {"name": "project_id", "in": "path", "required": True, "schema": {"type": "string"}}
                        ],
                        "responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}},
                    }
                },
                "/api/v1/status/{job_id}": {
                    "get": {
                        "summary": "Read job status",
                        "parameters": [
                            {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}
                        ],
                        "responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}},
                    }
                },
                "/api/v1/outputs/{name}": {
                    "delete": {
                        "summary": "Delete output",
                        "parameters": [
                            {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
                        ],
                        "responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}},
                    }
                },
                "/api/v1/characters/import": {
                    "post": {
                        "summary": "Import character media",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "multipart/form-data": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["file"],
                                        "properties": {
                                            "file": {"type": "string", "format": "binary"},
                                            "name": {"type": "string", "minLength": 1},
                                        },
                                        "additionalProperties": False,
                                    }
                                }
                            },
                        },
                        "responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}},
                    }
                },
                "/api/v1/recipes/{rid}/thumbnail": {
                    "get": {
                        "summary": "Download recipe thumbnail",
                        "parameters": [
                            {"name": "rid", "in": "path", "required": True, "schema": {"type": "string"}}
                        ],
                        "responses": {"200": {"content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}}},
                    }
                },
                "/api/v1/file/{filename:path}": {
                    "get": {"responses": {"200": {"content": {"application/octet-stream": {"schema": {"type": "string"}}}}}}
                },
                "/api/v1/uploads/{filename}": {
                    "get": {"responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}}}
                },
                "/api/v1/agent-access/clients": {
                    "get": {"responses": {"200": {"content": {"application/json": {"schema": {"type": "array"}}}}}}
                },
            },
        }

    async def openapi(self) -> dict[str, Any]:
        return self.spec

    async def request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        json: Any = None,
        idempotency_key: str | None = None,
        request_kind: str = "json",
        response_kind: str = "json",
        file_fields: tuple[str, ...] = (),
    ) -> Any:
        self.calls.append((method, path, {
            "query": query,
            "json": json,
            "idempotency_key": idempotency_key,
            "request_kind": request_kind,
            "response_kind": response_kind,
            "file_fields": file_fields,
        }))
        if path == "/api/v1/status/missing":
            raise BackendError(404, "job not found")
        if path == "/api/v1/status/job-1":
            return {"job_id": "job-1", "state": "running"}
        if path == "/api/v1/models":
            return {"families": [], "models": [{"model_type": "test_video", "name": "Test Video"}]}
        if path == "/api/v1/model-options/test_video":
            return {"model_type": "test_video", "fps": 24, "frames_maximum": 345, "audio_only": False}
        if path == "/api/v1/defaults/test_video":
            return {"model_type": "test_video", "num_inference_steps": 4}
        if path == "/api/v1/system-stats":
            return {"model": {"loaded": False}, "gpu": {"available": True}}
        if path == "/api/v1/agent-access/jobs/missing":
            return {"job_id": "missing", "state": "uncertain", "source": "durable_receipt"}
        if path == "/api/v1/recipes/coast%20one/thumbnail":
            return {"transfer_id": "transfer-1", "filename": "thumbnail.png", "size": 12, "sha256": "a" * 64}
        return {"method": method, "path": path, "body": json, "idempotency_key": idempotency_key}

    async def peers(self) -> dict[str, Any]:
        return {"local": {"name": "Maestro"}, "peers": []}

    async def lifecycle(self, action: str, request_id: str | None = None) -> dict[str, Any]:
        self.calls.append(("LIFECYCLE", action, {"request_id": request_id}))
        return {"action": action, "request_id": request_id, "status": "accepted"}

    async def upload_begin(self, filename: str, size: int, sha256: str) -> dict[str, Any]:
        self.calls.append(("UPLOAD_BEGIN", filename, {"size": size, "sha256": sha256}))
        return {"upload_id": "upload-1", "received_bytes": 0}

    async def upload_status(self, upload_id: str) -> dict[str, Any]:
        return {"upload_id": upload_id, "received_bytes": 10}

    async def upload_chunk(self, upload_id: str, offset: int, data_base64: str, request_id: str) -> dict[str, Any]:
        self.calls.append(("UPLOAD_CHUNK", upload_id, {"offset": offset, "data_base64": data_base64, "request_id": request_id}))
        return {"upload_id": upload_id, "received_bytes": offset + 5, "request_id": request_id}

    async def upload_complete(self, upload_id: str, request_id: str) -> dict[str, Any]:
        return {"upload_id": upload_id, "request_id": request_id, "asset_id": "asset:00000000-0000-0000-0000-000000000001"}

    async def get_submission(self, request_id: str) -> dict[str, Any]:
        self.calls.append(("SUBMISSION", request_id, {}))
        return {"request_id": request_id, "state": "complete", "result": {"job_id": "job-1"}}

    async def fetch_media(self, filename: str, workspace: str | None = None) -> dict[str, Any]:
        self.calls.append(("FETCH_MEDIA", filename, {"workspace": workspace}))
        return {"transfer_id": "transfer-1", "filename": filename, "workspace": workspace}

    async def read_transfer_chunk(
        self, transfer_id: str, offset: int = 0, max_bytes: int = 65_536,
    ) -> dict[str, Any]:
        self.calls.append(("TRANSFER_CHUNK", transfer_id, {"offset": offset, "max_bytes": max_bytes}))
        return {"transfer_id": transfer_id, "offset": offset, "data_base64": "aGVsbG8=", "eof": True}


def _wire_headers(version: str, method: str, name: str | None = None) -> dict[str, str]:
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "MCP-Protocol-Version": version,
        "Mcp-Method": method,
    }
    if name is not None:
        headers["Mcp-Name"] = name
    return headers


def _legacy_post(client: TestClient, method: str, params: dict[str, Any], request_id: int) -> dict[str, Any]:
    response = client.post(
        "/mcp",
        headers=_wire_headers("2025-11-25", method),
        json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
    )
    assert response.status_code == 200, response.text
    assert response.headers.get("Mcp-Session-Id") is None
    return response.json()


def _modern_post(
    client: TestClient,
    method: str,
    params: dict[str, Any],
    request_id: int,
    *,
    name: str | None = None,
) -> dict[str, Any]:
    modern_params = {
        **params,
        "_meta": {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientCapabilities": {},
            "io.modelcontextprotocol/clientInfo": {"name": "agent-access-test", "version": "1"},
        },
    }
    response = client.post(
        "/mcp",
        headers=_wire_headers("2026-07-28", method, name),
        json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": modern_params},
    )
    assert response.status_code == 200, response.text
    assert response.headers.get("Mcp-Session-Id") is None
    return response.json()


def _tool_result(response: dict[str, Any]) -> Any:
    assert "error" not in response, response
    return response["result"]


def test_server_has_bounded_tools_resources_prompts_and_short_instructions():
    from services.agent_access.guides import SERVER_INSTRUCTIONS

    backend = FakeBackend()
    server = create_mcp_server(backend, Path("."), "4.2.0")
    assert len(SERVER_INSTRUCTIONS) <= 512
    assert "credential" not in SERVER_INSTRUCTIONS.casefold()
    app = server.streamable_http_app(json_response=True, stateless_http=True)

    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        legacy = _legacy_post(
            client,
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "legacy-client", "version": "1"},
            },
            1,
        )
        assert _tool_result(legacy)["protocolVersion"] == "2025-11-25"

        legacy_tools = _tool_result(_legacy_post(client, "tools/list", {}, 2))["tools"]
        tool_names = {tool["name"] for tool in legacy_tools}
        assert {
            "get_status",
            "list_models",
            "get_model_options",
            "get_model_defaults",
            "help",
            "list_operations",
            "describe_operation",
            "execute_operation",
            "prepare_generation",
            "get_job",
            "upload_asset_begin",
            "upload_asset_status",
            "upload_asset_chunk",
            "upload_asset_complete",
            "get_submission",
            "fetch_media",
            "read_transfer_chunk",
            "list_workstations",
            "workstation_lifecycle",
        } <= tool_names
        assert "/api/v1/agent-access" not in " ".join(item["description"] for item in legacy_tools)

        resources = _tool_result(_legacy_post(client, "resources/list", {}, 3))["resources"]
        assert {resource["uri"] for resource in resources} >= {
            "maestro://guide/overview",
            "maestro://guide/studio",
            "maestro://guide/director",
            "maestro://guide/editor",
            "maestro://guide/models",
            "maestro://guide/assets",
        }
        assert all(resource.get("annotations", {}).get("priority") == 0.65 for resource in resources)
        resource = _tool_result(
            _legacy_post(client, "resources/read", {"uri": "maestro://guide/editor"}, 31)
        )
        assert resource["contents"][0]["text"].startswith("# Editor projects")
        prompts = _tool_result(_legacy_post(client, "prompts/list", {}, 4))["prompts"]
        assert {prompt["name"] for prompt in prompts} >= {
            "plan_studio_generation",
            "review_director_pipeline",
            "revise_editor_project",
        }
        prompt = _tool_result(
            _legacy_post(
                client,
                "prompts/get",
                {"name": "plan_studio_generation", "arguments": {"request": "A coast at dawn", "model_type": "test_video"}},
                41,
            )
        )
        assert "A coast at dawn" in prompt["messages"][0]["content"]["text"]

        help_result = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "help", "arguments": {"topic": "fish"}},
                5,
            )
        )
        assert help_result["content"][0]["text"].startswith("# Studio generation")

        # The same stateless ASGI app serves the SDK's 2026 per-request era.
        discovery = _tool_result(_modern_post(client, "server/discover", {}, 6))
        assert "2026-07-28" in discovery["supportedVersions"]
        modern_tools = _tool_result(_modern_post(client, "tools/list", {}, 7))["tools"]
        assert {tool["name"] for tool in modern_tools} >= tool_names
        modern_call = _tool_result(
            _modern_post(
                client,
                "tools/call",
                {"name": "list_models", "arguments": {}},
                8,
                name="list_models",
            )
        )
        assert modern_call["structuredContent"]["models"][0]["model_type"] == "test_video"


def test_tools_discover_and_execute_only_registered_schema_checked_routes():
    backend = FakeBackend()
    server = create_mcp_server(backend, Path("."), "4.2.0")
    app = server.streamable_http_app(json_response=True, stateless_http=True)

    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        tools = _tool_result(_legacy_post(client, "tools/list", {}, 1))["tools"]
        tool_by_name = {tool["name"]: tool for tool in tools}
        assert tool_by_name["execute_operation"]["annotations"]["destructiveHint"] is True
        assert tool_by_name["get_status"]["annotations"]["readOnlyHint"] is True
        assert tool_by_name["prepare_generation"]["annotations"]["idempotentHint"] is True

        status = _tool_result(
            _legacy_post(client, "tools/call", {"name": "get_status", "arguments": {}}, 2)
        )
        assert status["structuredContent"]["model"]["loaded"] is False
        models = _tool_result(
            _legacy_post(client, "tools/call", {"name": "list_models", "arguments": {}}, 3)
        )
        assert models["structuredContent"]["models"][0]["model_type"] == "test_video"
        capabilities = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "get_model_options", "arguments": {"model_type": "test_video"}},
                4,
            )
        )
        assert capabilities["structuredContent"]["fps"] == 24
        defaults = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "get_model_defaults", "arguments": {"model_type": "test_video"}},
                5,
            )
        )
        assert defaults["structuredContent"]["num_inference_steps"] == 4

        prepared = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {
                    "name": "prepare_generation",
                    "arguments": {"body": {"model_type": "test_video", "prompt": "A quiet shore"}, "request_id": "prepare-1"},
                },
                51,
            )
        )
        assert prepared["structuredContent"]["idempotency_key"] == "prepare-1"

        operations = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "list_operations", "arguments": {"domain": "studio", "limit": 30}},
                6,
            )
        )["structuredContent"]
        generate = next(item for item in operations["operations"] if item["path"] == "/api/v1/generate")
        assert generate["scope"] == "generate"

        assets = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "list_operations", "arguments": {"domain": "assets", "limit": 100}},
                61,
            )
        )["structuredContent"]["operations"]
        import_operation = next(item for item in assets if item["path"] == "/api/v1/characters/import")
        assert import_operation["request_kind"] == "multipart"
        assert import_operation["file_fields"] == ["file"]
        thumbnail = next(item for item in assets if item["path"] == "/api/v1/recipes/{rid}/thumbnail")
        assert thumbnail["response_kind"] == "binary"

        describe = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "describe_operation", "arguments": {"operation_id": generate["operation_id"]}},
                7,
            )
        )["structuredContent"]
        assert describe["request_body"]["required"] is True
        assert describe["request_body"]["content"]["application/json"]["properties"]["prompt"]["type"] == "string"

        missing_key = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {
                    "name": "execute_operation",
                    "arguments": {
                        "operation_id": generate["operation_id"],
                        "body": {"model_type": "test_video", "prompt": "A coast at dawn"},
                    },
                },
                8,
            )
        )
        assert missing_key["isError"] is True
        assert "request_id" in missing_key["content"][0]["text"]

        submitted = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {
                    "name": "execute_operation",
                    "arguments": {
                        "operation_id": generate["operation_id"],
                        "body": {"model_type": "test_video", "prompt": "A coast at dawn"},
                        "request_id": "render-1",
                    },
                },
                9,
            )
        )
        assert submitted["structuredContent"]["result"]["idempotency_key"] == "render-1"
        assert backend.calls[-1][0:2] == ("POST", "/api/v1/generate")
        assert backend.calls[-1][2]["request_kind"] == "json"
        assert backend.calls[-1][2]["response_kind"] == "json"

        imported = _tool_result(
            _modern_post(
                client,
                "tools/call",
                {
                    "name": "execute_operation",
                    "arguments": {
                        "operation_id": import_operation["operation_id"],
                        "body": {"file": "asset:00000000-0000-0000-0000-000000000001", "name": "Coast"},
                        "request_id": "import-1",
                    },
                },
                91,
                name="execute_operation",
            )
        )
        assert imported["structuredContent"]["result"]["body"]["name"] == "Coast"
        assert backend.calls[-1][2]["request_kind"] == "multipart"
        assert backend.calls[-1][2]["file_fields"] == ("file",)

        downloaded = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {
                    "name": "execute_operation",
                    "arguments": {
                        "operation_id": thumbnail["operation_id"],
                        "path_params": {"rid": "coast one"},
                    },
                },
                92,
            )
        )
        assert downloaded["structuredContent"]["result"]["transfer_id"] == "transfer-1"
        assert backend.calls[-1][2]["response_kind"] == "binary"
        assert backend.calls[-1][2]["request_kind"] == "none"

        native_mutations = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "list_operations", "arguments": {"domain": "director", "limit": 100}},
                93,
            )
        )["structuredContent"]["operations"]
        start_pipeline = next(item for item in native_mutations if item["path"] == "/api/v1/director/pipeline/start")
        described = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "describe_operation", "arguments": {"operation_id": start_pipeline["operation_id"]}},
                94,
            )
        )["structuredContent"]
        assert described["request_body"]["schema_available"] is False
        assert "native" in described["request_body"]["validation"]
        assert described["request_body"]["field_hints"]["status"] == "unavailable"
        _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {
                    "name": "execute_operation",
                    "arguments": {
                        "operation_id": start_pipeline["operation_id"],
                        "body": {"title": "A coast at dawn", "clips": [{"prompt": "Wave breaks"}]},
                        "request_id": "director-start-1",
                    },
                },
                95,
            )
        )
        assert backend.calls[-1][2]["request_kind"] == "json"
        assert backend.calls[-1][2]["json"]["clips"][0]["prompt"] == "Wave breaks"

        editor_mutations = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "list_operations", "arguments": {"domain": "editor", "limit": 100}},
                96,
            )
        )["structuredContent"]["operations"]
        update_project = next(item for item in editor_mutations if item["path"] == "/api/v1/editor/projects/{project_id}")
        _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {
                    "name": "execute_operation",
                    "arguments": {
                        "operation_id": update_project["operation_id"],
                        "path_params": {"project_id": "project-1"},
                        "body": {"project": {"name": "Coast", "tracks": []}},
                        "request_id": "editor-update-1",
                    },
                },
                97,
            )
        )
        assert backend.calls[-1][2]["request_kind"] == "json"
        assert backend.calls[-1][2]["json"]["project"]["name"] == "Coast"

        invalid_body = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {
                    "name": "execute_operation",
                    "arguments": {
                        "operation_id": generate["operation_id"],
                        "body": {"model_type": "unknown", "prompt": "A coast at dawn"},
                        "request_id": "render-2",
                    },
                },
                10,
            )
        )
        assert invalid_body["isError"] is True
        assert "documented choices" in invalid_body["content"][0]["text"]

        unknown = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "execute_operation", "arguments": {"operation_id": "get_anything"}},
                11,
            )
        )
        assert unknown["isError"] is True
        assert "Unknown operation_id" in unknown["content"][0]["text"]


def test_jobs_assets_lifecycle_and_path_validation():
    backend = FakeBackend()
    server = create_mcp_server(backend, Path("."), "4.2.0")
    app = server.streamable_http_app(json_response=True, stateless_http=True)

    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        direct = _tool_result(
            _legacy_post(client, "tools/call", {"name": "get_job", "arguments": {"job_id": "job-1"}}, 1)
        )
        assert direct["structuredContent"]["state"] == "running"
        fallback = _tool_result(
            _legacy_post(client, "tools/call", {"name": "get_job", "arguments": {"job_id": "missing"}}, 2)
        )
        assert fallback["structuredContent"]["source"] == "durable_receipt"
        assert backend.calls[-1][1] == "/api/v1/agent-access/jobs/missing"

        submission = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "get_submission", "arguments": {"request_id": "render-1"}},
                21,
            )
        )
        assert submission["structuredContent"]["request_id"] == "render-1"
        assert backend.calls[-1][0] == "SUBMISSION"

        fetched = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "fetch_media", "arguments": {"filename": "coast.mp4", "workspace": "default"}},
                22,
            )
        )
        assert fetched["structuredContent"]["transfer_id"] == "transfer-1"
        chunk = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "read_transfer_chunk", "arguments": {"transfer_id": "transfer-1", "max_bytes": 5}},
                23,
            )
        )
        assert chunk["structuredContent"]["data_base64"] == "aGVsbG8="
        traversal = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "fetch_media", "arguments": {"filename": "../outside.mp4"}},
                24,
            )
        )
        assert traversal["isError"] is True
        assert "basename" in traversal["content"][0]["text"]

        invalid_name = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "upload_asset_begin", "arguments": {"filename": "../private.mp4", "size": 10, "sha256": "a" * 64}},
                3,
            )
        )
        assert invalid_name["isError"] is True
        assert "basename" in invalid_name["content"][0]["text"]

        begun = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "upload_asset_begin", "arguments": {"filename": "scene.mp4", "size": 10, "sha256": "A" * 64}},
                4,
            )
        )
        assert begun["structuredContent"]["upload_id"] == "upload-1"
        completed = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "upload_asset_complete", "arguments": {"upload_id": "upload-1", "request_id": "upload-done"}},
                5,
            )
        )
        assert completed["structuredContent"]["asset_id"].startswith("asset:")

        bad_chunk = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {
                    "name": "upload_asset_chunk",
                    "arguments": {"upload_id": "upload-1", "offset": 0, "data_base64": "not-base64!", "request_id": "chunk-0"},
                },
                51,
            )
        )
        assert bad_chunk["isError"] is True
        assert "valid base64" in bad_chunk["content"][0]["text"]

        chunk = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {
                    "name": "upload_asset_chunk",
                    "arguments": {"upload_id": "upload-1", "offset": 0, "data_base64": "aGVsbG8=", "request_id": "chunk-0"},
                },
                52,
            )
        )
        assert chunk["structuredContent"]["received_bytes"] == 5
        assert backend.calls[-1][2]["request_id"] == "chunk-0"

        lifecycle = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "workstation_lifecycle", "arguments": {"action": "restart", "request_id": "restart-1"}},
                6,
            )
        )
        assert lifecycle["structuredContent"]["request_id"] == "restart-1"

        options_operation = next(
            operation
            for operation in build_catalog_for_test(backend.spec).values()
            if operation.path == "/api/v1/model-options/{model_type}"
        )
        with pytest.raises(ValueError, match="safe path segment"):
            validate_inputs_for_test(backend.spec, options_operation, {"model_type": "../../x"})


def test_describe_operation_includes_installed_launch_ast_hints(tmp_path: Path):
    (tmp_path / "launch.py").write_text(
        '''
@api.post("/api/v1/director/pipeline/start")
async def director_pipeline_start(request: Request):
    body = await request.json()
    return await asyncio.to_thread(start_pipeline, body)

@api.put("/api/v1/editor/projects/{project_id}")
async def editor_project_update(project_id: str, request: Request):
    body = await request.json()
    workspace = body.get("workspace") or _get_active_workspace()
    project = body.get("project") if isinstance(body.get("project"), dict) else body
    return save_editor_project(workspace, project)
''',
        encoding="utf-8",
    )
    services = tmp_path / "services"
    services.mkdir()
    (services / "director_pipeline.py").write_text(
        '''
def start_pipeline(params):
    video_model = params.get("video_model")
    auto_mode = params.get("auto_mode", True)
    return video_model, auto_mode
''',
        encoding="utf-8",
    )
    backend = FakeBackend()
    server = create_mcp_server(backend, tmp_path, "4.2.0")
    app = server.streamable_http_app(json_response=True, stateless_http=True)

    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        _tool_result(
            _legacy_post(
                client,
                "initialize",
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "contract-hints-test", "version": "1"},
                },
                200,
            )
        )
        operations = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "list_operations", "arguments": {"domain": "editor", "limit": 100}},
                201,
            )
        )["structuredContent"]["operations"]
        operation = next(item for item in operations if item["path"] == "/api/v1/editor/projects/{project_id}")
        description = _tool_result(
            _legacy_post(
                client,
                "tools/call",
                {"name": "describe_operation", "arguments": {"operation_id": operation["operation_id"]}},
                202,
            )
        )["structuredContent"]
    hints = description["request_body"]["field_hints"]
    assert hints["status"] == "available"
    assert hints["validation"] == "Hints only; native validation is authoritative."
    assert {field["name"] for field in hints["fields"]} >= {"workspace", "project"}

    director_server = create_mcp_server(backend, tmp_path, "4.2.0")
    director_app = director_server.streamable_http_app(json_response=True, stateless_http=True)
    with TestClient(director_app, base_url="http://127.0.0.1:8765") as director_client:
        _tool_result(
            _legacy_post(
                director_client,
                "initialize",
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "director-hints-test", "version": "1"},
                },
                203,
            )
        )
        director_ops = _tool_result(
            _legacy_post(
                director_client,
                "tools/call",
                {"name": "list_operations", "arguments": {"domain": "director", "limit": 100}},
                204,
            )
        )["structuredContent"]["operations"]
        director_start = next(item for item in director_ops if item["path"] == "/api/v1/director/pipeline/start")
        director = _tool_result(
            _legacy_post(
                director_client,
                "tools/call",
                {"name": "describe_operation", "arguments": {"operation_id": director_start["operation_id"]}},
                205,
            )
        )["structuredContent"]["request_body"]["field_hints"]
    assert director["sources"] == ["launch.py", "services/director_pipeline.py"]
    assert {field["name"] for field in director["fields"]} >= {"video_model", "auto_mode"}


def test_status_adds_only_stable_identity_when_backend_exposes_it():
    class IdentityBackend(FakeBackend):
        async def identity(self) -> dict[str, str]:
            return {
                "installation_id": "install-123",
                "workstation_id": "workstation-456",
                "access_token": "must-not-be-returned",
            }

    backend = IdentityBackend()
    server = create_mcp_server(backend, Path("."), "4.2.0")
    app = server.streamable_http_app(json_response=True, stateless_http=True)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        _tool_result(
            _legacy_post(
                client,
                "initialize",
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "identity-test", "version": "1"},
                },
                300,
            )
        )
        status = _tool_result(
            _legacy_post(client, "tools/call", {"name": "get_status", "arguments": {}}, 301)
        )["structuredContent"]
    assert status["model"]["loaded"] is False
    assert status["agent_access"]["identity"] == {
        "installation_id": "install-123",
        "workstation_id": "workstation-456",
    }
    assert "must-not-be-returned" not in repr(status)


def build_catalog_for_test(openapi: dict[str, Any]):
    from services.agent_access.catalog import build_catalog

    return build_catalog(openapi)


def validate_inputs_for_test(openapi: dict[str, Any], operation: Any, path_params: dict[str, Any]):
    from services.agent_access.catalog import validate_operation_inputs

    return validate_operation_inputs(openapi, operation, path_params, None, None)
