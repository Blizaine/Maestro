from __future__ import annotations

import pytest

from services.agent_access.catalog import (
    SchemaValidationError,
    build_catalog,
    domain_counts,
    operation_description,
    required_scope,
    validate_operation_inputs,
)


def test_required_scope_is_conservative_and_domain_aware():
    assert required_scope("GET", "/api/v1/models") == "observe"
    assert required_scope("GET", "/api/v1/system-stats") == "observe"
    assert required_scope("POST", "/api/v1/generate") == "generate"
    assert required_scope("GET", "/api/v1/director/pipeline/abc") == "observe"
    assert required_scope("POST", "/api/v1/director/pipeline/abc/resume") == "generate"
    assert required_scope("GET", "/api/v1/director/queue") == "observe"
    assert required_scope("GET", "/api/v1/director/pipeline/abc/llm-drafts") == "observe"
    assert required_scope("GET", "/api/v1/editor/projects/abc") == "observe"
    assert required_scope("GET", "/api/v1/editor/projects/abc/revisions") == "observe"
    assert required_scope("GET", "/api/v1/outputs") == "observe"
    assert required_scope("GET", "/api/v1/workspaces") == "observe"
    assert required_scope("GET", "/api/v1/characters/abc/media/portrait") == "observe"
    assert required_scope("GET", "/api/v1/characters/remote-files") == "manage"
    assert required_scope("GET", "/api/v1/loras/installed") == "manage"
    assert required_scope("GET", "/api/v1/loras/scan-status/scan-1") == "observe"
    assert required_scope("GET", "/api/v1/model-folders/scan") == "manage"
    assert required_scope("GET", "/api/v1/storage/usage") == "manage"
    assert required_scope("GET", "/api/v1/models/wan/debug") == "manage"
    assert required_scope("POST", "/api/v1/models/checkpoint/download") == "manage"
    assert required_scope("GET", "/api/v1/system-config") == "admin"
    assert required_scope("GET", "/api/v1/services-config") == "admin"
    assert required_scope("POST", "/api/v1/remote-access/tailscale/enable") == "admin"
    assert required_scope("GET", "/api/v1/agent-access/grants") == "admin"
    assert required_scope("POST", "/api/v1/agent-access/prepare") == "generate"
    assert required_scope("GET", "/api/v1/agent-access/jobs/job-1") == "observe"
    assert required_scope("GET", "/api/v1/agent-access/submissions") == "observe"
    assert required_scope("GET", "/api/v1/agent-access/peers") == "observe"
    assert required_scope("GET", "/api/v1/agent-access/coverage") == "observe"
    assert required_scope("GET", "/api/v1/agent-access/guide/studio") == "observe"
    assert required_scope("GET", "/api/v1/agent-access/peers/unknown") == "admin"
    assert required_scope("GET", "/api/v1/agent-access/jobs") == "admin"
    assert required_scope("POST", "/api/v1/agent-access/assets/uploads") == "manage"
    assert required_scope("GET", "/api/v1/agent-access/assets/uploads/upload-1") == "observe"
    assert required_scope("PUT", "/api/v1/agent-access/assets/uploads/upload-1") == "manage"
    assert required_scope("PATCH", "/api/v1/agent-access/assets/uploads/upload-1") == "admin"
    assert required_scope("POST", "/api/v1/agent-access/assets/uploads/upload-1/complete") == "manage"
    assert required_scope("POST", "/api/v1/agent-access/lifecycle") == "admin"
    assert required_scope("GET", "/api/v1/civitai/search") == "observe"
    assert required_scope("GET", "/api/v1/civitai/downloads") == "observe"
    assert required_scope("POST", "/api/v1/civitai/download") == "manage"
    assert required_scope("GET", "/api/v1/checkpoints/installed") == "observe"
    assert required_scope("POST", "/api/v1/media/trim") == "manage"
    assert required_scope("POST", "/api/v1/new/unknown-write") == "admin"


def test_catalog_is_built_from_registered_json_api_routes_only():
    openapi = {
        "openapi": "3.0.3",
        "paths": {
            "/api/v1/models": {
                "get": {
                    "summary": "List models",
                    "responses": {"200": {"content": {"application/json": {"schema": {"type": "array"}}}}},
                }
            },
            "/api/v1/director/pipeline/{pid}/resume": {
                "post": {
                    "summary": "Resume pipeline",
                    "responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}},
                }
            },
            "/api/v1/editor/projects/{project_id}/revisions": {
                "get": {
                    "summary": "List revisions",
                    "responses": {"200": {"content": {"application/json": {"schema": {"type": "array"}}}}},
                }
            },
            "/api/v1/system-config": {
                "get": {
                    "summary": "Read system config",
                    "responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}},
                }
            },
            "/api/v1/file/{filename:path}": {
                "get": {"responses": {"200": {"content": {"application/octet-stream": {"schema": {"type": "string"}}}}}},
            },
            "/api/v1/uploads/{filename}": {
                "get": {"responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}}},
            },
            "/api/v1/agent-access/grants": {
                "get": {"responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}}},
            },
            "/api/v1/upload-audio": {
                "post": {"responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}}},
            },
            "/api/v1/loras/preview/{filename}": {
                "get": {"responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}}},
            },
            "/api/v1/recipes/{rid}/thumbnail": {
                "get": {"responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}}},
            },
            "/api/v1/characters/{character_id}/media/{slot}": {
                "get": {"responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}}},
            },
            "/api/v1/director/pipelines/{pid}/thumbnail": {
                "get": {"responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}}},
            },
            "/api/v1/jobs/{job_id}/preview/{revision}": {
                "get": {"responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}}},
            },
            "/api/v1/face-refiner/analyses/{analysis_id}/faces/{track_id}": {
                "get": {"responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}}},
            },
            "/api/v1/editor/media/cache/{preview_id}/{kind}": {
                "get": {"responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}}},
            },
            "/classic/api/v1/models": {
                "get": {"responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}}},
            },
            "/api/v1/image": {
                "post": {"requestBody": {"content": {"multipart/form-data": {"schema": {"type": "object"}}}}, "responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}}},
            },
        },
    }

    catalog = build_catalog(openapi)
    assert len(catalog) == 13
    assert {operation.domain for operation in catalog.values()} == {"models", "director", "editor", "settings", "assets", "studio"}
    assert {operation.scope for operation in catalog.values()} >= {"observe", "generate", "manage", "admin"}
    assert not any(
        operation.path.startswith(("/api/v1/file/", "/api/v1/uploads/"))
        or "/agent-access/" in operation.path
        for operation in catalog.values()
    )
    assert domain_counts(catalog) == {
        "studio": 4,
        "director": 2,
        "editor": 2,
        "models": 2,
        "assets": 2,
        "settings": 1,
        "other": 0,
    }


def test_operation_schema_validation_quotes_path_params_and_rejects_traversal():
    openapi = {
        "openapi": "3.0.3",
        "components": {
            "schemas": {
                "Render": {
                    "type": "object",
                    "required": ["model_type", "prompt"],
                    "properties": {
                        "model_type": {"type": "string", "enum": ["wan", "qwen"]},
                        "prompt": {"type": "string", "minLength": 1},
                        "options": {"type": "object", "additionalProperties": True},
                    },
                    "additionalProperties": False,
                }
            }
        },
        "paths": {
            "/api/v1/models/{model_type}/guide": {
                "post": {
                    "parameters": [
                        {"name": "model_type", "in": "path", "required": True, "schema": {"type": "string"}},
                        {"name": "force", "in": "query", "required": False, "schema": {"type": "boolean"}},
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Render"}}},
                    },
                    "responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}},
                }
            }
        },
    }
    catalog = build_catalog(openapi)
    operation = next(iter(catalog.values()))

    route, query, body = validate_operation_inputs(
        openapi,
        operation,
        {"model_type": "wan test"},
        {"force": True},
        {"model_type": "wan", "prompt": "A quiet forest", "options": {"seed": 3}},
    )
    assert route == "/api/v1/models/wan%20test/guide"
    assert query == {"force": True}
    assert body["prompt"] == "A quiet forest"

    for unsafe in ("../secret", "one/two", "one\\two", ""):
        with pytest.raises(SchemaValidationError, match="safe path segment"):
            validate_operation_inputs(openapi, operation, {"model_type": unsafe}, {}, {"model_type": "wan", "prompt": "x"})

    with pytest.raises(SchemaValidationError, match="missing required field"):
        validate_operation_inputs(openapi, operation, {"model_type": "wan"}, {}, {"model_type": "wan"})
    with pytest.raises(SchemaValidationError, match="documented choices"):
        validate_operation_inputs(openapi, operation, {"model_type": "wan"}, {}, {"model_type": "flux", "prompt": "x"})
    with pytest.raises(SchemaValidationError, match="undocumented field"):
        validate_operation_inputs(openapi, operation, {"model_type": "wan"}, {}, {"model_type": "wan", "prompt": "x", "oops": 1})
    with pytest.raises(SchemaValidationError, match="undocumented field"):
        validate_operation_inputs(openapi, operation, {"model_type": "wan"}, {"bad": "query"}, {"model_type": "wan", "prompt": "x"})


def test_operation_description_hides_credential_fields_but_keeps_schema_help():
    openapi = {
        "paths": {
            "/api/v1/services-config": {
                "put": {
                    "summary": "Update services",
                    "description": "Update the selected services.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "enabled": {"type": "boolean"},
                                        "api_key": {"type": "string"},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {"200": {"content": {"application/json": {"schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}}}}}},
                }
            }
        }
    }
    operation = next(iter(build_catalog(openapi).values()))
    description = operation_description(operation, openapi)
    schema = description["request_body"]["content"]["application/json"]
    assert schema["properties"] == {"enabled": {"type": "boolean"}}
    assert description["scope"] == "admin"
    assert description["responses"]["200"]["content"]["application/json"]["properties"]["ok"]["type"] == "boolean"


def test_binary_and_multipart_operations_remain_named_schema_checked_exports():
    openapi = {
        "openapi": "3.1.0",
        "paths": {
            "/api/v1/characters/import": {
                "post": {
                    "summary": "Import a character package",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["file"],
                                    "properties": {
                                        "file": {"type": "string", "format": "binary"},
                                        "metadata_file": {"type": "string", "contentMediaType": "application/json"},
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
            "/api/v1/loras/preview/{filename}": {
                "get": {
                    "parameters": [{"name": "filename", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}},
                }
            },
            "/api/v1/exports/{export_id}": {
                "get": {
                    "parameters": [{"name": "export_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "responses": {"200": {"content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}}}},
                }
            },
            "/api/v1/file/{filename:path}": {
                "get": {"responses": {"200": {"content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}}}
            },
            "/api/v1/uploads/{filename}": {
                "get": {"responses": {"200": {"content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}}}
            },
        },
    }
    catalog = build_catalog(openapi)
    assert len(catalog) == 3
    importer = next(operation for operation in catalog.values() if operation.path == "/api/v1/characters/import")
    preview = next(operation for operation in catalog.values() if operation.path.startswith("/api/v1/loras/preview/"))
    export = next(operation for operation in catalog.values() if operation.path == "/api/v1/exports/{export_id}")

    assert importer.request_kind == "multipart"
    assert importer.request_media_type == "multipart/form-data"
    assert importer.file_fields == ("file",)
    assert importer.response_kind == "json"
    description = operation_description(importer, openapi)
    assert description["request_body"]["kind"] == "multipart"
    assert description["request_body"]["file_fields"] == ["file"]
    assert description["request_body"]["content"]["multipart/form-data"]["properties"]["file"]["format"] == "binary"
    assert preview.response_kind == "binary"  # known native binary response is mislabeled in older schemas
    assert export.response_kind == "binary"  # media type/schema provides a second detector
    assert operation_description(preview, openapi)["responses"]["200"]["content"] == {}

    route, query, body = validate_operation_inputs(
        openapi,
        importer,
        None,
        None,
        {"file": "asset:00000000-0000-0000-0000-000000000001", "name": "character"},
    )
    assert route == "/api/v1/characters/import"
    assert query is None
    assert body["name"] == "character"
    with pytest.raises(SchemaValidationError, match="owned asset"):
        validate_operation_inputs(
            openapi,
            importer,
            None,
            None,
            {"file": "C:/Users/person/character.zip", "name": "character"},
        )


def test_request_json_mutations_without_openapi_body_use_native_validation():
    openapi = {
        "openapi": "3.1.0",
        "paths": {
            "/api/v1/generate": {
                "post": {"summary": "Submit generation", "responses": {"200": {"description": "Accepted"}}}
            },
            "/api/v1/director/pipeline/start": {
                "post": {"summary": "Start Director pipeline", "responses": {"200": {"description": "Accepted"}}}
            },
            "/api/v1/editor/projects/{project_id}": {
                "put": {
                    "summary": "Update Editor project",
                    "parameters": [{"name": "project_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "responses": {"200": {"description": "Saved"}},
                }
            },
            "/api/v1/director/pipeline/{pid}": {
                "get": {
                    "parameters": [{"name": "pid", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}},
                }
            },
        },
    }
    catalog = build_catalog(openapi)
    examples = {
        "/api/v1/generate": {"model_type": "test_video", "prompt": "A quiet shore", "video_length": 81},
        "/api/v1/director/pipeline/start": {"title": "A coast at dawn", "clips": [{"prompt": "Wave breaks"}]},
        "/api/v1/editor/projects/{project_id}": {"project": {"name": "Coast", "tracks": []}},
    }
    for path, body in examples.items():
        operation = next(item for item in catalog.values() if item.path == path)
        assert operation.request_kind == "json"
        assert operation.request_schema_available is False
        params = {"project_id": "project-1"} if "{project_id}" in path else None
        route, query, validated_body = validate_operation_inputs(openapi, operation, params, None, body)
        assert route == path.replace("{project_id}", "project-1")
        assert query is None
        assert validated_body == body
        description = operation_description(operation, openapi)
        assert description["request_schema_available"] is False
        assert description["request_body"]["kind"] == "json"
        assert description["request_body"]["schema_available"] is False
        assert description["request_body"]["validation"] == "native; OpenAPI body schema unavailable"
        schema = description["request_body"]["content"]["application/json"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is True

    read_status = next(item for item in catalog.values() if item.path == "/api/v1/director/pipeline/{pid}")
    assert read_status.request_kind == "none"
    assert read_status.request_schema_available is True


def test_new_owner_and_dedicated_agent_access_routes_stay_out_of_generic_catalogue():
    openapi = {
        "openapi": "3.1.0",
        "paths": {
            "/api/v1/agent-access/owner/login": {"post": {"responses": {"200": {}}}},
            "/api/v1/agent-access/owner/code": {"post": {"responses": {"200": {}}}},
            "/api/v1/agent-access/settings": {"get": {"responses": {"200": {}}}},
            "/api/v1/agent-access/grants": {"get": {"responses": {"200": {}}}},
            "/api/v1/agent-access/audit": {"get": {"responses": {"200": {}}}},
            "/api/v1/agent-access/jobs/{job_id}": {"get": {"responses": {"200": {}}}},
            "/api/v1/agent-access/submissions": {"get": {"responses": {"200": {}}}},
            "/api/v1/agent-access/assets/uploads": {"post": {"responses": {"200": {}}}},
            "/api/v1/agent-access/coverage": {"get": {"responses": {"200": {}}}},
            "/api/v1/system-stats": {"get": {"responses": {"200": {}}}},
        },
    }
    catalog = build_catalog(openapi)
    assert {operation.path for operation in catalog.values()} == {"/api/v1/system-stats"}
