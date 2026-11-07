"""MCP server for bounded Maestro Agent Access tools and guides."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
from collections.abc import Awaitable, Mapping
from pathlib import Path
import re
import uuid
from typing import Any, Protocol
from urllib.parse import quote

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import Annotations, ToolAnnotations, CallToolResult, TextContent, ImageContent, AudioContent

from .catalog import (
    DOMAIN_ORDER,
    Operation,
    SchemaValidationError,
    build_catalog,
    domain_counts,
    normalize_domain,
    operation_description,
    operation_summary,
    validate_operation_inputs,
)
from .contracts import ContractIndex, load_contract_index
from .guides import GUIDES, SERVER_INSTRUCTIONS, get_guide


_MAX_UPLOAD_CHUNK_BYTES = 1_048_576
_MAX_UPLOAD_CHUNK_BASE64 = 4 * ((_MAX_UPLOAD_CHUNK_BYTES + 2) // 3)


class Backend(Protocol):
    """Narrow authenticated backend surface owned by the HTTP controller."""

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
    ) -> Any: ...

    async def openapi(self) -> dict[str, Any]: ...

    async def peers(self) -> dict[str, Any]: ...

    async def lifecycle(self, action: str, request_id: str | None = None) -> dict[str, Any]: ...

    async def upload_begin(self, filename: str, size: int, sha256: str) -> dict[str, Any]: ...

    async def upload_status(self, upload_id: str) -> dict[str, Any]: ...

    async def upload_chunk(
        self,
        upload_id: str,
        offset: int,
        data_base64: str,
        request_id: str,
    ) -> dict[str, Any]: ...

    async def upload_complete(self, upload_id: str, request_id: str) -> dict[str, Any]: ...

    async def get_submission(self, request_id: str) -> dict[str, Any]: ...

    async def fetch_media(self, filename: str, workspace: str | None = None) -> dict[str, Any]: ...

    async def read_transfer_chunk(
        self, transfer_id: str, offset: int = 0, max_bytes: int = 65_536,
    ) -> dict[str, Any]: ...


class _OpenApiCatalog:
    """Lazily build one immutable catalogue from the running backend schema."""

    def __init__(self, backend: Backend, app_root: Path):
        self.backend = backend
        self.app_root = Path(app_root)
        self._lock = asyncio.Lock()
        self._openapi: Mapping[str, Any] | None = None
        self._catalog: dict[str, Operation] | None = None
        self._contracts: ContractIndex | None = None
        self._revision = None

    async def get(self) -> tuple[Mapping[str, Any], dict[str, Operation]]:
        revision = self.backend.catalog_revision() if callable(getattr(self.backend, "catalog_revision", None)) else None
        if revision != self._revision:
            self._openapi = self._catalog = self._contracts = None
        if self._openapi is not None and self._catalog is not None:
            return self._openapi, self._catalog
        async with self._lock:
            if self._openapi is None or self._catalog is None:
                spec = await self.backend.openapi()
                if not isinstance(spec, Mapping):
                    raise RuntimeError("Maestro returned an invalid OpenAPI document")
                catalog = build_catalog(spec)
                contracts = load_contract_index(self.app_root)
                self._openapi = spec
                self._catalog = catalog
                self._contracts = contracts
                self._revision = revision
        return self._openapi, self._catalog

    def request_field_hints(self, operation: Operation) -> dict[str, Any]:
        if operation.request_schema_available:
            return {
                "source": "launch.py",
                "status": "not_needed",
                "validation": "OpenAPI schema is authoritative.",
                "fields": [],
                "truncated": False,
            }
        index = self._contracts or ContractIndex.unavailable()
        return index.describe(operation.method, operation.path)


def _request_id(value: str | None, *, name: str = "request_id") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{name} must be a non-empty string for this mutation")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ToolError(f"{name} must be valid UTF-8") from error
    if len(encoded) > 256:
        raise ToolError(f"{name} is too long")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ToolError(f"{name} contains a control character")
    return value


def _validated_segment(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ToolError(f"{name} must be a string or number")
    text = str(value)
    if not text or text in {".", ".."} or any(char in text for char in ("/", "\\", "\x00")):
        raise ToolError(f"{name} must be a single safe identifier")
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise ToolError(f"{name} contains a control character")
    return text


def _safe_segment(value: Any, name: str) -> str:
    text = _validated_segment(value, name)
    return quote(text, safe="")


def _status_code(error: Exception) -> int | None:
    for name in ("status", "status_code", "http_status"):
        value = getattr(error, name, None)
        if isinstance(value, int):
            return value
    return None


async def _safe_backend(awaitable: Awaitable[Any]) -> Any:
    """Pass documented backend errors to the model without leaking internals."""

    try:
        return await awaitable
    except ToolError:
        raise
    except Exception as error:
        if _status_code(error) is not None:
            raise ToolError(str(error)) from error
        raise


def _resource(server: MCPServer, uri: str, title: str, text: str) -> None:
    topic = uri.rsplit("/", 1)[-1]

    def read_guide() -> str:
        return text

    server.resource(
        uri,
        name=f"maestro_{topic}_guide",
        title=title,
        description=f"Bundled Maestro user guide: {title}.",
        mime_type="text/markdown",
        annotations=Annotations(audience=["assistant", "user"], priority=0.65),
    )(read_guide)


def create_mcp_server(
    backend: Backend, app_root: Path, version: str, *, server_class: type[MCPServer] = MCPServer,
) -> MCPServer:
    """Create the tool server; the parent owns HTTP auth, middleware and lifespan.

    The caller should mount ``server.streamable_http_app(json_response=True,
    stateless_http=True)`` beneath its authenticated ``/mcp`` route.
    Contract hints parse public ``app_root/launch.py`` and the fixed Director
    helper ``app_root/services/director_pipeline.py`` with AST; guides are
    bundled and never read private/development files from disk.
    """

    server = server_class(
        name="Maestro Agent Access",
        title="Maestro Agent Access",
        description="Scoped control of a paired Maestro workstation.",
        instructions=SERVER_INSTRUCTIONS,
        version=str(version),
        warn_on_duplicate_resources=False,
    )
    catalog_store = _OpenApiCatalog(backend, app_root)

    _resource(server, "maestro://guide/overview", "Agent Access overview", GUIDES["overview"])
    for topic, title in (
        ("studio", "Studio generation"),
        ("director", "Director workflows"),
        ("editor", "Editor projects"),
        ("filmmaking", "Title cards, finishing and take review"),
        ("models", "Models and performance"),
        ("assets", "Assets and workstations"),
        ("settings", "Settings and workstation controls"),
    ):
        _resource(server, f"maestro://guide/{topic}", title, GUIDES[topic])

    @server.tool(
        name="get_status",
        title="Get Maestro status",
        description=(
            "Read live Maestro workstation and hardware status. When the backend exposes stable installation identity, "
            "it is returned under agent_access. Does not start a model or job."
        ),
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    )
    async def get_status() -> dict[str, Any]:
        """Read the current workstation and loaded-model status."""
        status = await _safe_backend(backend.request("GET", "/api/v1/system-stats"))
        identity_method = getattr(backend, "identity", None)
        if not callable(identity_method):
            return status
        identity = await _safe_backend(identity_method())
        if not isinstance(identity, Mapping):
            return status
        stable_identity = {
            name: value.strip()
            for name in ("installation_id", "workstation_id")
            if isinstance((value := identity.get(name)), str) and value.strip() and len(value) <= 128
        }
        if not stable_identity or not isinstance(status, Mapping):
            return status
        return {**status, "agent_access": {"identity": stable_identity}}

    @server.tool(
        name="list_models",
        title="List installed models",
        description="List models registered by the live Maestro installation.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    )
    async def list_models() -> dict[str, Any]:
        """List the available Maestro model choices."""
        return await _safe_backend(backend.request("GET", "/api/v1/models"))

    @server.tool(
        name="get_model_options",
        title="Get model capabilities",
        description="Read one model's live supported modes, reference roles, timing limits and options.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    )
    async def get_model_options(model_type: str) -> dict[str, Any]:
        """Read the live capabilities for one model type."""
        model = _safe_segment(model_type, "model_type")
        return await _safe_backend(backend.request("GET", f"/api/v1/model-options/{model}"))

    @server.tool(
        name="get_model_defaults",
        title="Get model defaults",
        description="Read the installation's current default settings for one model.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    )
    async def get_model_defaults(model_type: str) -> dict[str, Any]:
        """Read defaults for the requested model type."""
        model = _safe_segment(model_type, "model_type")
        return await _safe_backend(backend.request("GET", f"/api/v1/defaults/{model}"))

    @server.tool(
        name="help",
        title="Read a Maestro guide",
        description="Return a bundled workflow guide. Use this when the client does not surface MCP resources.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    )
    async def help(topic: str = "overview") -> str:
        """Read a guide topic: overview, studio, director, editor, models, assets, or settings."""
        return get_guide(topic)

    @server.tool(
        name="list_operations",
        title="List Maestro operations",
        description=(
            "List named operations registered by the live Maestro OpenAPI schema. "
            "Domains: studio, director, editor, models, assets, settings, other, or all. "
            "Use offset and limit to page through a large domain."
        ),
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    )
    async def list_operations(
        domain: str = "all",
        offset: int = 0,
        limit: int = 40,
        search: str | None = None,
    ) -> dict[str, Any]:
        """Discover callable API operation IDs from the running OpenAPI document."""
        openapi, catalog = await catalog_store.get()
        del openapi
        selected_domain = normalize_domain(domain)
        if selected_domain is None and str(domain).strip().casefold() not in {"", "all", "*"}:
            raise ToolError(f"Unknown domain {domain!r}; choose one of: {', '.join(DOMAIN_ORDER)}")
        if isinstance(offset, bool) or offset < 0:
            raise ToolError("offset must be a non-negative integer")
        if isinstance(limit, bool) or limit < 1 or limit > 100:
            raise ToolError("limit must be between 1 and 100")
        term = (search or "").strip().casefold()
        matching = [
            operation
            for operation in catalog.values()
            if (selected_domain is None or operation.domain == selected_domain)
            and (
                not term
                or term in operation.operation_id.casefold()
                or term in operation.path.casefold()
                or term in operation.summary.casefold()
            )
        ]
        matching.sort(key=lambda operation: (operation.domain, operation.path, operation.method))
        return {
            "domains": domain_counts(catalog),
            "domain": selected_domain or "all",
            "count": len(matching),
            "offset": offset,
            "limit": limit,
            "operations": [operation_summary(operation) for operation in matching[offset: offset + limit]],
        }

    @server.tool(
        name="describe_operation",
        title="Describe a Maestro operation",
        description=(
            "Show one operation's exact path, scope, parameters and live OpenAPI schema. "
            "When OpenAPI omits a native JSON body schema, include bounded, non-enforcing field hints from launch.py."
        ),
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    )
    async def describe_operation(operation_id: str) -> dict[str, Any]:
        """Describe an operation ID returned by list_operations."""
        _openapi, catalog = await catalog_store.get()
        operation = catalog.get(operation_id)
        if operation is None:
            raise ToolError("Unknown operation_id. Call list_operations to choose a registered operation.")
        return operation_description(
            operation,
            _openapi,
            request_field_hints=catalog_store.request_field_hints(operation),
        )

    @server.tool(
        name="execute_operation",
        title="Execute a named Maestro operation",
        description=(
            "Call a named operation using path_params, query and body exactly as described. Multipart file fields "
            "accept only registered asset:<UUID> references. Binary responses become private transfer records "
            "read with read_transfer_chunk or the transfer URL. Every mutation needs a stable request_id; use "
            "get_submission after an ambiguous response and never submit it again under a new ID."
        ),
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False),
    )
    async def execute_operation(
        operation_id: str,
        path_params: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
        body: Any = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute one named JSON or multipart operation and return JSON or transfer metadata."""
        openapi, catalog = await catalog_store.get()
        operation = catalog.get(operation_id)
        if operation is None:
            raise ToolError("Unknown operation_id. Call list_operations to choose a registered operation.")
        try:
            route, safe_query, safe_body = validate_operation_inputs(
                openapi,
                operation,
                path_params,
                query,
                body,
            )
        except SchemaValidationError as error:
            raise ToolError(str(error)) from error
        key = None
        if operation.method not in {"GET", "HEAD", "OPTIONS"}:
            key = _request_id(request_id)
        result = await _safe_backend(
            backend.request(
                operation.method,
                route,
                query=safe_query,
                json=safe_body,
                idempotency_key=key,
                request_kind=operation.request_kind,
                response_kind=operation.response_kind,
                file_fields=operation.file_fields,
            )
        )
        return {"result": result}

    @server.tool(
        name="get_creation_defaults",
        title="Read workflow creation defaults",
        description="Read explicit agent defaults and compatible installed models for Frames, References, images, songs, speech and sound effects. Does not write settings or load models.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    )
    async def get_creation_defaults() -> dict[str, Any]:
        return await _safe_backend(backend.request("GET", "/api/v1/agent-access/creative-defaults"))

    @server.tool(
        name="get_creation_context",
        title="Get native guides and creation timeline",
        description=(
            "Start agent-led creative work here. Body: workflow (video.frames, video.references, image.generate, "
            "image.edit, audio.music, audio.speech, audio.sfx, title.card, director.story, director.music_video), brief, optional "
            "model_type/duration_seconds/resolution/params/references/writer_instructions/voice_count. References "
            "use stable IDs plus role/type and an uploaded asset in native params. Resolves compatible installed "
            "defaults and returns the actual model guides, references and exact native window timeline. No local "
            "writer, downloads or render. You author the requested native prompt/script/lyrics/Director clips. "
            "title.card is exact-text CPU titling; read help(filmmaking) for cards, finishing and bounded polishing."
        ),
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=False, openWorldHint=False),
    )
    async def get_creation_context(body: dict[str, Any]) -> dict[str, Any]:
        return await _safe_backend(backend.request(
            "POST", "/api/v1/agent-access/creative-context", json=body,
            idempotency_key="context-" + uuid.uuid4().hex,
        ))

    @server.tool(
        name="prepare_authored_generation",
        title="Review an agent-authored creation plan",
        description=(
            "Validate your own creative writing against a server context without starting a render or invoking "
            "Maestro's local writer. Supply context_id and plan with the same context_id: prompt (plus style for "
            "songs) for single output, exact windows from the context for sliding generation, or clips for Director. "
            "Preserve reference IDs and continuity. Returns prepared_id, exact prepared settings and reasons for "
            "any rejected draft. Use submit_authored_generation separately when ready."
        ),
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False),
    )
    async def prepare_authored_generation(context_id: str, plan: dict[str, Any], request_id: str) -> dict[str, Any]:
        return await _safe_backend(backend.request(
            "POST", "/api/v1/agent-access/prepare-authored",
            json={"context_id": context_id, "plan": plan}, idempotency_key=_request_id(request_id),
        ))

    @server.tool(
        name="submit_authored_generation",
        title="Start a reviewed agent-authored render",
        description=(
            "Submit a prepared_id returned by prepare_authored_generation. Starts the Studio render or Director "
            "pipeline using that exact reviewed server revision without local prompt enhancement. Keep the same "
            "request_id after a timeout and recover with get_submission; never submit a replacement ID blindly."
        ),
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False),
    )
    async def submit_authored_generation(prepared_id: str, request_id: str) -> dict[str, Any]:
        return await _safe_backend(backend.request(
            "POST", "/api/v1/agent-access/submit-authored",
            json={"prepared_id": prepared_id}, idempotency_key=_request_id(request_id),
        ))

    @server.tool(
        name="get_authored_draft",
        title="Read a rejected authored draft",
        description="Read this agent's saved draft and actionable validation reasons; no writing or rendering.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    )
    async def get_authored_draft(draft_id: str) -> dict[str, Any]:
        return await _safe_backend(backend.request("GET", f"/api/v1/agent-access/authoring-drafts/{_safe_segment(draft_id, 'draft_id')}"))

    @server.tool(name="finish_film", title="Assemble and finish a film", description="Create a NEW editable Editor project from ordered clips/title-card assets, narration, music and SFX. Body: name/workspace/canvas/clips[{asset,duration?,source_in?,include_audio?,gain_db?,fade_in?,fade_out?}], narration/music/sfx[{asset,start?,duration?,gain_db?,fade_in?,fade_out?}], speech_ranges?, duck_music?, export?. Media uses asset IDs or {filename,workspace}. Defaults music -18dB, SFX -9dB; ducks music only under placed narration/reviewed speech ranges. export=false saves without rendering; true queues native software export. Read help(filmmaking).", annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
    async def finish_film(body: dict[str, Any], request_id: str) -> dict[str, Any]:
        return await _safe_backend(backend.request("POST", "/api/v1/agent-access/production/finish", json=body, idempotency_key=_request_id(request_id)))

    @server.tool(name="inspect_take", title="Inspect actual rendered media", description="CPU-only inspection of an existing asset. Body: asset (uploaded ID or {filename,workspace}), expected_duration?, include_audio?. Returns probe/software-decode evidence plus an actual contact-sheet image and optional audio preview. Stills do not establish continuous motion or voice ownership. Creative status stays not_assessed until YOU inspect and record observations. No model call or generation.", annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=False, openWorldHint=False))
    async def inspect_take(body: dict[str, Any]) -> CallToolResult:
        packet = await _safe_backend(backend.request("POST", "/api/v1/agent-access/production/inspect", json=body, idempotency_key="inspect-" + uuid.uuid4().hex))
        sheet = packet.pop("contact_sheet_base64", None)
        audio = packet.pop("audio_preview_base64", None)
        content = [TextContent(type="text", text=json.dumps(packet, ensure_ascii=False))]
        if sheet:
            content.append(ImageContent(type="image", data=sheet, mimeType="image/jpeg"))
        if audio:
            content.append(AudioContent(type="audio", data=audio, mimeType="audio/wav"))
        return CallToolResult(content=content, structuredContent=packet)

    @server.tool(name="record_take_review", title="Save an agent's creative take review", description="Record what YOU observed in an inspect_take packet: body {review_id,review:{verdict:keep|needs_work|reject,summary,observations:[{category,severity:low|medium|high|critical,text}]}}. Use concrete observations; skipped/unknown checks are not passes. Advisory metadata never blocks the whole film and never submits a render.", annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
    async def record_take_review(body: dict[str, Any], request_id: str) -> dict[str, Any]:
        return await _safe_backend(backend.request("POST", "/api/v1/agent-access/production/reviews", json=body, idempotency_key=_request_id(request_id)))

    @server.tool(name="create_polish_budget", title="Set explicit retake limits", description="Create an agent-owned durable budget with explicit max_renders (1..12) and max_retakes_per_clip (1..3). Obtain user authorization for extra renders before using it; creating a budget submits no work. Every attempted/uncertain request consumes a slot. The budget cannot be reset by reopening it.", annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
    async def create_polish_budget(max_renders: int, max_retakes_per_clip: int, request_id: str) -> dict[str, Any]:
        return await _safe_backend(backend.request("POST", "/api/v1/agent-access/production/budgets", json={"max_renders": max_renders, "max_retakes_per_clip": max_retakes_per_clip}, idempotency_key=_request_id(request_id)))

    @server.tool(name="get_polish_budget", title="Read retake budget and receipts", description="Read this agent's durable polish budget, consumed attempts and owned receipts. Reconcile an uncertain request through these receipts and the original pipeline; never blindly submit another request ID.", annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False))
    async def get_polish_budget(budget_id: str) -> dict[str, Any]:
        return await _safe_backend(backend.request("GET", "/api/v1/agent-access/production/budgets/" + _safe_segment(budget_id, "budget_id")))

    @server.tool(name="polish_take", title="Revise one reviewed Director take", description="Submit ONE explicitly authorized selected-take video rerun. Body: budget_id, review_id (recorded needs_work/reject), pipeline_id, clip_index (zero-based), optional revised prompt. Binds source fingerprint and persists intent before native generation. A changed take or busy/uncertain machine is deferred. Returns an owned job ID; poll get_job. Preserve request_id after timeouts. Rejoin the Director cut explicitly after accepting the replacement.", annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
    async def polish_take(body: dict[str, Any], request_id: str) -> dict[str, Any]:
        return await _safe_backend(backend.request("POST", "/api/v1/agent-access/production/polish", json={**body, "request_id": _request_id(request_id)}, idempotency_key=_request_id(request_id)))

    @server.tool(
        name="prepare_generation",
        title="Prepare a Studio generation",
        description=(
            "Prepare a Studio request through Maestro's native flow; multi-window planning may use the configured writer. "
            "Preparation does not enqueue a render. Prefer the agent-authored context/prepare/submit flow; "
            "supply a stable request_id and inspect the returned preparation before proceeding."
        ),
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False),
    )
    async def prepare_generation(body: dict[str, Any], request_id: str) -> dict[str, Any]:
        """Prepare native settings; does not enqueue. May invoke the configured local writer."""
        key = _request_id(request_id)
        return await _safe_backend(
            backend.request(
                "POST",
                "/api/v1/agent-access/prepare",
                json=body,
                idempotency_key=key,
            )
        )

    @server.tool(
        name="get_job",
        title="Get a Maestro job",
        description="Poll the normal job status; if it is absent there, read this principal's durable Agent Access receipt.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    )
    async def get_job(job_id: str) -> dict[str, Any]:
        """Read a Maestro job or the durable Agent Access job snapshot."""
        job = _safe_segment(job_id, "job_id")
        try:
            return await backend.request("GET", f"/api/v1/status/{job}")
        except Exception as error:
            if _status_code(error) != 404:
                if _status_code(error) is not None:
                    raise ToolError(str(error)) from error
                raise
        return await _safe_backend(backend.request("GET", f"/api/v1/agent-access/jobs/{job}"))

    @server.tool(
        name="get_submission",
        title="Recover a submission receipt",
        description=(
            "Look up the durable receipt for a request_id, even if Maestro is offline. Use the exact same request_id "
            "after a timeout or lost tool response; pending and uncertain receipts are never dispatched again."
        ),
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    )
    async def get_submission(request_id: str) -> dict[str, Any]:
        """Read a principal-owned submission receipt by its original request ID."""
        return await _safe_backend(backend.get_submission(_request_id(request_id)))

    @server.tool(
        name="fetch_media",
        title="Fetch a Maestro media file",
        description=(
            "Fetch one visible Maestro output or media item by basename and optional workspace. The file is stored "
            "as a private transfer record; read it in bounded chunks or use the returned authenticated URL."
        ),
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    )
    async def fetch_media(filename: str, workspace: str | None = None) -> dict[str, Any]:
        """Fetch one safe basename without accepting filesystem paths."""
        if not isinstance(filename, str) or not filename.strip():
            raise ToolError("filename must be a non-empty basename")
        name = filename.strip()
        if name in {".", ".."} or any(char in name for char in ("/", "\\", ":", "%", "\x00")):
            raise ToolError("filename must be a basename without path separators")
        if any(ord(char) < 32 or ord(char) == 127 for char in name):
            raise ToolError("filename contains a control character")
        if workspace is not None:
            workspace = _validated_segment(workspace, "workspace")
        return await _safe_backend(backend.fetch_media(name, workspace))

    @server.tool(
        name="read_transfer_chunk",
        title="Read a transfer chunk",
        description=(
            "Read a private binary transfer record returned by execute_operation or fetch_media. Use the returned "
            "offset for the next call; each base64 result contains at most 1 MiB of decoded data."
        ),
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    )
    async def read_transfer_chunk(
        transfer_id: str,
        offset: int = 0,
        max_bytes: int = 65_536,
    ) -> dict[str, Any]:
        """Read one bounded chunk from an owned transfer artifact."""
        identifier = _validated_segment(transfer_id, "transfer_id")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ToolError("offset must be a non-negative byte count")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1 or max_bytes > _MAX_UPLOAD_CHUNK_BYTES:
            raise ToolError("max_bytes must be between 1 byte and 1 MiB")
        return await _safe_backend(backend.read_transfer_chunk(identifier, offset, max_bytes))

    @server.tool(
        name="upload_asset_begin",
        title="Begin an asset upload",
        description="Create a resumable upload record using a filename, byte size and SHA-256 digest.",
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False),
    )
    async def upload_asset_begin(filename: str, size: int, sha256: str) -> dict[str, Any]:
        """Begin a media upload without sending workstation-local paths."""
        name = str(filename).strip()
        if not name or name in {".", ".."} or any(char in name for char in ("/", "\\", "\x00")):
            raise ToolError("filename must be a basename without path separators")
        if any(ord(char) < 32 or ord(char) == 127 for char in name):
            raise ToolError("filename contains a control character")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ToolError("size must be a non-negative byte count")
        if not isinstance(sha256, str):
            raise ToolError("sha256 must contain exactly 64 hexadecimal characters")
        digest = sha256.strip().casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ToolError("sha256 must contain exactly 64 hexadecimal characters")
        return await _safe_backend(backend.upload_begin(name, size, digest))

    @server.tool(
        name="upload_asset_status",
        title="Check an asset upload",
        description="Read transfer progress for an upload ID returned by upload_asset_begin.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    )
    async def upload_asset_status(upload_id: str) -> dict[str, Any]:
        """Read resumable upload state using its opaque upload ID."""
        identifier = _validated_segment(upload_id, "upload_id")
        return await _safe_backend(backend.upload_status(identifier))

    @server.tool(
        name="upload_asset_chunk",
        title="Send an asset upload chunk",
        description=(
            "Send one base64 transfer chunk. Decoded content is capped at 1 MiB per call; "
            "send sequentially by byte offset and reuse request_id for an exact retry."
        ),
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False),
    )
    async def upload_asset_chunk(
        upload_id: str,
        offset: int,
        data_base64: str,
        request_id: str,
    ) -> dict[str, Any]:
        """Send one bounded chunk in a resumable upload."""
        identifier = _validated_segment(upload_id, "upload_id")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ToolError("offset must be a non-negative byte count")
        if not isinstance(data_base64, str) or not data_base64:
            raise ToolError("data_base64 must contain a base64-encoded chunk")
        if len(data_base64) > _MAX_UPLOAD_CHUNK_BASE64:
            raise ToolError("decoded upload chunk must not exceed 1 MiB")
        try:
            decoded = base64.b64decode(data_base64, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ToolError("data_base64 is not valid base64") from error
        if not decoded:
            raise ToolError("upload chunk must not be empty")
        if len(decoded) > _MAX_UPLOAD_CHUNK_BYTES:
            raise ToolError("decoded upload chunk must not exceed 1 MiB")
        del decoded
        return await _safe_backend(
            backend.upload_chunk(
                identifier,
                offset,
                data_base64,
                _request_id(request_id),
            )
        )

    @server.tool(
        name="upload_asset_complete",
        title="Complete an asset upload",
        description="Verify and register an uploaded asset, using a stable request_id for completion.",
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False),
    )
    async def upload_asset_complete(upload_id: str, request_id: str) -> dict[str, Any]:
        """Complete a verified upload and obtain its portable asset ID."""
        identifier = _validated_segment(upload_id, "upload_id")
        return await _safe_backend(backend.upload_complete(identifier, _request_id(request_id)))

    @server.tool(
        name="list_workstations",
        title="List paired workstations",
        description="Read discovered Maestro workstation peers. No job or lifecycle action is started.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    )
    async def list_workstations() -> dict[str, Any]:
        """List workstation peers known to Agent Access."""
        return await _safe_backend(backend.peers())

    @server.tool(
        name="workstation_lifecycle",
        title="Control Maestro lifecycle",
        description="Request a supported Maestro lifecycle action. Confirm the action with the user first.",
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False),
    )
    async def workstation_lifecycle(action: str, request_id: str | None = None) -> dict[str, Any]:
        """Control the Maestro application through its supported launcher integration."""
        normalized = action.strip().casefold() if isinstance(action, str) else ""
        if not normalized:
            raise ToolError("action must be a non-empty string")
        if len(normalized) > 64 or any(ord(char) < 32 or ord(char) == 127 for char in normalized):
            raise ToolError("action is invalid")
        read_action = normalized in {"status", "inspect", "list"}
        key = _request_id(request_id) if request_id is not None or not read_action else None
        return await _safe_backend(backend.lifecycle(normalized, key))

    @server.prompt(
        name="plan_studio_generation",
        title="Plan a Studio generation",
        description="Prepare a request using this installation's model options, references and timing limits.",
    )
    def plan_studio_generation(request: str, model_type: str | None = None) -> str:
        """Create a model-aware plan without starting generation."""
        model_hint = f"The requested model is {model_type}." if model_type else "Resolve compatible workstation defaults with get_creation_context."
        return (
            f"The user's request is: {request}\n\n{model_hint} Read `maestro://guide/studio`, then call "
            "get_creation_context for the requested workflow. You write the native prompts using its guides, "
            "timeline and explicit reference roles. Review with prepare_authored_generation, then render the exact "
            "prepared_id using submit_authored_generation and a stable request_id. Respect existing user "
            "authorization; ask only for additional downloads or work outside that scope. Do not call a local writer."
        )

    @server.prompt(
        name="review_director_pipeline",
        title="Review a Director pipeline",
        description="Inspect existing Director work and choose a safe next action.",
    )
    def review_director_pipeline(pipeline_id: str, requested_change: str = "") -> str:
        """Review a pipeline without repeating a start request."""
        return (
            f"Inspect Director pipeline {pipeline_id} and its current status before acting. The requested change is: "
            f"{requested_change or 'review and report status'}. Read `maestro://guide/director`. Distinguish a paused "
            "pipeline (`continue`) from a crashed pipeline (`resume`); preserve completed takes and use `stop` only "
            "when asked. Do not re-submit a start request when the result is ambiguous."
        )

    @server.prompt(
        name="revise_editor_project",
        title="Revise an Editor project",
        description="Inspect project revisions and export capabilities before editing.",
    )
    def revise_editor_project(project_id: str, requested_change: str) -> str:
        """Plan a non-destructive Editor revision."""
        return (
            f"Inspect Editor project {project_id}, its current revision, and the requested export capabilities. "
            f"The user's requested change is: {requested_change}\n\nRead `maestro://guide/editor`. Preserve original media "
            "and unrelated human-authored edits. Review proposed operations, confirm the target revision before a restore, "
            "and treat export as a separate action that creates a new output."
        )

    # Returning the instance leaves transport setup, shared auth and startup
    # resource management with the trusted outer controller.
    return server


__all__ = ["Backend", "create_mcp_server"]
