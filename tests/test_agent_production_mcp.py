from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import importlib.util
import unittest

if importlib.util.find_spec("mcp") is None:
    raise unittest.SkipTest("MCP protocol tests run in the separate Agent Access environment")

from mcp.types import AudioContent, CallToolResult, ImageContent, TextContent
from services.agent_access.mcp_server import create_mcp_server


PRODUCTION_TOOLS = {
    "finish_film",
    "inspect_take",
    "record_take_review",
    "create_polish_budget",
    "get_polish_budget",
    "polish_take",
}


class ProductionBackend:
    def __init__(self, image_base64: str, audio_base64: str) -> None:
        self.image_base64 = image_base64
        self.audio_base64 = audio_base64
        self.calls: list[tuple[str, str, Any, str | None]] = []

    async def request(
        self,
        method: str,
        path: str,
        *,
        query: Any = None,
        json: Any = None,
        idempotency_key: str | None = None,
        request_kind: str = "json",
        response_kind: str = "json",
        file_fields: tuple[str, ...] = (),
    ) -> Any:
        self.calls.append((method, path, json, idempotency_key))
        assert path == "/api/v1/agent-access/production/inspect"
        return {
            "asset": {"filename": "take.mp4"},
            "contact_sheet_base64": self.image_base64,
            "audio_preview_base64": self.audio_base64,
            "probe": {"duration": 1.0, "width": 16, "height": 16},
        }


def test_production_tools_register_and_inspection_returns_native_media_content():
    async def exercise() -> None:
        image_base64 = base64.b64encode(b"\xff\xd8\xff\xd9").decode("ascii")
        audio_base64 = base64.b64encode(b"RIFF\x04\x00\x00\x00WAVE").decode("ascii")
        backend = ProductionBackend(image_base64, audio_base64)
        app_root = Path(__file__).resolve().parents[1] / "app"
        server = create_mcp_server(backend, app_root, "4.2.0")

        tools = {tool.name: tool for tool in await server.list_tools()}
        assert PRODUCTION_TOOLS <= tools.keys()
        assert all(tools[name].description for name in PRODUCTION_TOOLS)

        result = await server.call_tool(
            "inspect_take", {"body": {"asset": "asset:take-1", "include_audio": True}}
        )

        assert isinstance(result, CallToolResult)
        assert isinstance(result.content[0], TextContent)
        assert isinstance(result.content[1], ImageContent)
        assert isinstance(result.content[2], AudioContent)
        assert result.content[1].data == image_base64
        assert result.content[1].mime_type == "image/jpeg"
        assert result.content[2].data == audio_base64
        assert result.content[2].mime_type == "audio/wav"
        assert result.structured_content == {
            "asset": {"filename": "take.mp4"},
            "probe": {"duration": 1.0, "width": 16, "height": 16},
        }

        wire_result = result.model_dump(mode="json", by_alias=True)
        assert [item["type"] for item in wire_result["content"]] == ["text", "image", "audio"]
        assert wire_result["content"][1]["mimeType"] == "image/jpeg"
        assert wire_result["content"][2]["mimeType"] == "audio/wav"
        assert len(backend.calls) == 1
        method, path, body, request_id = backend.calls[0]
        assert (method, path, body) == (
            "POST",
            "/api/v1/agent-access/production/inspect",
            {"asset": "asset:take-1", "include_audio": True},
        )
        assert request_id and request_id.startswith("inspect-")

    asyncio.run(exercise())
