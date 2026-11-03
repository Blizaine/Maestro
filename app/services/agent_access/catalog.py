"""Bounded, OpenAPI-backed catalogue for Agent Access API operations.

The scope classifier in this module intentionally uses only the Python
standard library. The model runtime can import it without loading the CPU
control-plane dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote, urlsplit


HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete"})
READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
DOMAIN_ORDER = ("studio", "director", "editor", "models", "assets", "settings", "other")
_PRIVATE_PROPERTY = re.compile(r"(?:api[_-]?key|authorization|password|secret|token|credential)", re.I)
_ASSET_REFERENCE = re.compile(
    r"^(?:asset|maestro-asset):[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_.-]*)(?::[^}]+)?\}")
_BINARY_RESPONSE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^/api/v1/loras/preview/\{[^/]+\}$",
        r"^/api/v1/recipes/\{[^/]+\}/thumbnail$",
        r"^/api/v1/characters/\{[^/]+\}/media/\{[^/]+\}$",
        r"^/api/v1/characters/\{[^/]+\}/export-file$",
        r"^/api/v1/characters/\{[^/]+\}/images\.zip$",
        r"^/api/v1/director/pipelines/\{[^/]+\}/thumbnail$",
        r"^/api/v1/jobs/\{[^/]+\}/preview/\{[^/]+\}$",
        r"^/api/v1/face-refiner/analyses/\{[^/]+\}/faces/\{[^/]+\}$",
        r"^/api/v1/editor/media/cache/\{[^/]+\}/\{[^/]+\}$",
        r"^/api/v1/music-training/projects/\{[^/]+\}/preparation/\{[^/]+\}/audio$",
        r"^/api/v1/music-training/projects/\{[^/]+\}/recordings/\{[^/]+\}$",
        r"^/api/v1/music-training/projects/\{[^/]+\}/auditions/\{[^/]+\}$",
        r"^/api/v1/music-styles/\{[^/]+\}/export$",
        r"^/api/v1/worlds/\{[^/]+\}/save$",
        r"^/api/v1/worlds/\{[^/]+\}/record$",
        r"^/api/v1/worlds/\{[^/]+\}/capture$",
    )
)
_WORD = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Operation:
    """One callable route copied from the running Maestro OpenAPI document."""

    operation_id: str
    method: str
    path: str
    domain: str
    scope: str
    summary: str
    description: str
    parameters: tuple[Mapping[str, Any], ...]
    request_body: Mapping[str, Any] | None
    responses: Mapping[str, Any]
    request_kind: str = "none"
    response_kind: str = "json"
    request_media_type: str | None = None
    file_fields: tuple[str, ...] = ()
    request_schema_available: bool = True


class SchemaValidationError(ValueError):
    """A safe, compact validation error for an operation's OpenAPI schema."""


def required_scope(method: str, path: str) -> str:
    """Return the least Agent Access scope that can call an API route.

    Unknown mutating routes intentionally require ``admin``. This function is
    safe to import from the model backend: it performs no I/O and imports no
    third-party modules.
    """

    verb = str(method or "").upper()
    raw_path = str(path or "")
    try:
        route = urlsplit(raw_path).path or "/"
    except (TypeError, ValueError):
        route = raw_path
    route = re.sub(r"/{2,}", "/", route).rstrip("/") or "/"
    lowered = route.casefold()
    segments = tuple(segment for segment in lowered.split("/") if segment)
    read = verb in READ_METHODS

    # ComfyUI is an optional, separately managed execution engine. Reads are
    # observable (backend job ownership checks still apply); generation and
    # receipt recovery use exact workflow/submission routes. Model inspection
    # and the connection probe can access local files or contact another
    # service, so they require manage. Workflow/configuration writes and every
    # unknown ComfyUI write remain admin.
    if len(segments) >= 3 and segments[:3] == ("api", "v1", "comfyui"):
        if read:
            return "observe"
        if (
            verb == "POST"
            and len(segments) == 6
            and segments[3] == "workflows"
            and segments[4]
            and segments[5] == "run"
        ):
            return "generate"
        if (
            verb == "POST"
            and len(segments) == 7
            and segments[3:5] == ("jobs", "by-submission")
            and (segments[5] == "{uuid}" or re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                segments[5],
            ))
            and segments[6] in {"resume", "abandon"}
        ):
            return "generate"
        if verb == "POST" and (
            (len(segments) == 5 and segments[3:5] == ("connection", "test"))
            or (len(segments) == 5 and segments[3] == "models" and segments[4] in {"discover", "scan", "link"})
        ):
            return "manage"
        return "admin"

    # Agent Access owns these endpoints. The two data-plane routes used by the
    # dedicated MCP tools have narrow scopes; auth/owner routes always require
    # admin and remain excluded from generic operation dispatch.
    if len(segments) >= 3 and segments[:2] == ("api", "v1") and segments[2] == "agent-access":
        if len(segments) == 4 and segments[3] in {"creative-context", "creative-defaults"} and (read or segments[3] == "creative-context" and verb == "POST"):
            return "observe"
        if len(segments) == 4 and segments[3] in {"prepare-authored", "submit-authored"} and verb == "POST":
            return "generate"
        if len(segments) == 5 and segments[3] == "authoring-drafts" and read:
            return "observe"
        if len(segments) == 4 and segments[3] == "prepare" and verb == "POST":
            return "generate"
        if len(segments) == 5 and segments[3] == "jobs" and read:
            return "observe"
        if len(segments) == 4 and segments[3] == "submissions" and read:
            return "observe"
        if len(segments) >= 4 and read and (
            (segments[3] in {"peers", "coverage"} and len(segments) == 4)
            or (segments[3] == "guide" and len(segments) == 5)
        ):
            return "observe"
        if len(segments) >= 5 and segments[3:5] == ("assets", "uploads"):
            if len(segments) == 5 and verb == "POST":
                return "manage"
            if len(segments) == 6 and verb == "GET":
                return "observe"
            if len(segments) == 6 and verb == "PUT":
                return "manage"
            if len(segments) == 7 and segments[6] in {"complete", "import"} and verb == "POST":
                return "manage"
        return "admin"

    # Server configuration, network exposure, service control, model reload,
    # and unknown writes are owner-level operations.
    admin_prefixes = (
        "/api/v1/system-config",
        "/api/v1/services-config",
        "/api/v1/remote-access",
        "/api/v1/notifications/push",
        "/api/v1/model-visibility",
        "/api/v1/h3-window-overrides",
        "/api/v1/studio-preferences",
        "/api/v1/system-detect/apply",
        "/api/v1/system-detect/reset-learning",
        "/api/v1/system/release-model",
        "/api/v1/lifecycle",
        "/api/v1/llm/load",
        "/api/v1/llm/unload",
        "/api/v1/llm/remote",
    )
    if any(lowered == prefix or lowered.startswith(prefix + "/") for prefix in admin_prefixes):
        return "admin"
    if lowered in {"/api/v1/system-stats", "/api/v1/system/preflight", "/api/v1/system-detect"} and read:
        return "observe"

    # Director status, saved state and queue inspection are read-only. Native
    # job/pipeline ownership guards still limit which work an agent can inspect.
    if read and (lowered == "/api/v1/director" or lowered.startswith("/api/v1/director/")):
        return "observe"

    # Browsing registries and download status is read-only discovery. The
    # operations that scan local LoRA/model folders or start downloads stay in
    # the manage scope below.
    if read and lowered.startswith("/api/v1/civitai/"):
        return "observe"
    if read and lowered == "/api/v1/checkpoints/installed":
        return "observe"
    if read and lowered == "/api/v1/models/downloads/status":
        return "observe"
    if read and (
        lowered in {"/api/v1/downloads/active", "/api/v1/downloads", "/api/v1/active-downloads"}
    ):
        return "observe"

    # These reads inspect local model files, perform filesystem scans, or
    # fetch a remote Hugging Face listing. Keep them in manage even though the
    # HTTP method is GET. A status poll for an existing LoRA scan is cheap.
    if read and (
        lowered == "/api/v1/loras/scan-status"
        or lowered.startswith("/api/v1/loras/scan-status/")
    ):
        return "observe"
    if read and (
        lowered == "/api/v1/loras"
        or lowered.startswith("/api/v1/loras/")
        or lowered == "/api/v1/model-folders/scan"
        or lowered == "/api/v1/storage/usage"
        or lowered == "/api/v1/storage/duplicates"
        or lowered == "/api/v1/characters/remote-files"
        or re.match(r"^/api/v1/models/[^/]+/debug$", lowered)
    ):
        return "manage"

    # Writes in these namespaces can mutate assets, projects, exports or model
    # libraries. Their ordinary reads stay observe; expensive filesystem scans
    # and debug reads were held at manage above.
    manage_prefixes = (
        "/api/v1/loras",
        "/api/v1/huggingface",
        "/api/v1/model-folders",
        "/api/v1/active-downloads",
        "/api/v1/media",
        "/api/v1/uploads",
        "/api/v1/upload",
        "/api/v1/outputs",
        "/api/v1/file",
        "/api/v1/media-info",
        "/api/v1/thumbnail",
        "/api/v1/characters",
        "/api/v1/character-transfers",
        "/api/v1/recipes",
        "/api/v1/presets",
        "/api/v1/favorites",
        "/api/v1/editor",
        "/api/v1/worlds",
        "/api/v1/music-training",
        "/api/v1/music-styles",
        "/api/v1/storage",
        "/api/v1/workspaces",
    )
    if any(lowered == prefix or lowered.startswith(prefix + "/") for prefix in manage_prefixes):
        return "observe" if read else "manage"
    if lowered.startswith("/api/v1/checkpoints/"):
        return "manage"
    if lowered.startswith("/api/v1/civitai/"):
        return "manage"
    if lowered.startswith("/api/v1/downloads/"):
        return "manage"
    if re.match(r"^/api/v1/models/[^/]+/(?:download|download/cancel|debug)$", lowered):
        return "manage"
    if lowered in {"/api/v1/models/reload", "/api/v1/models/downloads/status"}:
        return "manage"

    # Rendering, enhancement, and Director planning/control consume the
    # generation grant, including their related status and continuation APIs.
    generation_prefixes = (
        "/api/v1/director",
        "/api/v1/generate",
        "/api/v1/retake",
        "/api/v1/extract-frames",
        "/api/v1/edit-anything",
        "/api/v1/repaint",
        "/api/v1/recast",
        "/api/v1/outpaint",
        "/api/v1/blend",
        "/api/v1/segment",
        "/api/v1/inpaint",
        "/api/v1/face-refiner",
        "/api/v1/media-flow",
        "/api/v1/tools",
        "/api/v1/audio",
        "/api/v1/upload-audio",
        "/api/v1/jobs",
        "/api/v1/cancel",
    )
    if any(lowered == prefix or lowered.startswith(prefix + "/") for prefix in generation_prefixes):
        if lowered.startswith("/api/v1/jobs") or lowered.startswith("/api/v1/cancel"):
            return "generate" if not read else "observe"
        return "generate"
    if lowered.startswith("/api/v1/llm/"):
        if any(part in lowered for part in ("/generate", "/write-song", "/plan-", "/enhance-", "/describe-image")):
            return "generate"
        if read:
            return "observe"
        return "admin"

    # Remaining GETs are discovery/read operations. Any unknown write is
    # intentionally elevated instead of being guessed into a narrower scope.
    if read:
        return "observe"
    return "admin"


def operation_domain(path: str) -> str:
    """Group a route under the six user-facing Agent Access catalogue areas."""

    lowered = str(path or "").casefold()
    if lowered == "/api/v1/director" or lowered.startswith("/api/v1/director/"):
        return "director"
    if lowered == "/api/v1/editor" or lowered.startswith("/api/v1/editor/"):
        return "editor"
    if any(
        lowered == prefix or lowered.startswith(prefix + "/")
        for prefix in (
            "/api/v1/models",
            "/api/v1/model-visibility",
            "/api/v1/model-folders",
            "/api/v1/downloads",
            "/api/v1/active-downloads",
            "/api/v1/model-options",
            "/api/v1/defaults",
            "/api/v1/resolutions",
            "/api/v1/loras",
            "/api/v1/checkpoints",
            "/api/v1/civitai",
            "/api/v1/huggingface",
            "/api/v1/presets",
        )
    ):
        return "models"
    if any(
        lowered == prefix or lowered.startswith(prefix + "/")
        for prefix in (
            "/api/v1/uploads",
            "/api/v1/upload",
            "/api/v1/outputs",
            "/api/v1/file",
            "/api/v1/media",
            "/api/v1/thumbnail",
            "/api/v1/characters",
            "/api/v1/character-transfers",
            "/api/v1/recipes",
            "/api/v1/favorites",
            "/api/v1/worlds",
            "/api/v1/music-training",
            "/api/v1/music-styles",
            "/api/v1/storage",
            "/api/v1/media",
        )
    ):
        return "assets"
    if any(
        lowered == prefix or lowered.startswith(prefix + "/")
        for prefix in (
            "/api/v1/system",
            "/api/v1/system-config",
            "/api/v1/services-config",
            "/api/v1/services",
            "/api/v1/remote-access",
            "/api/v1/notifications",
            "/api/v1/workspaces",
            "/api/v1/studio-preferences",
            "/api/v1/h3-window-overrides",
        )
    ):
        return "settings"
    if lowered in {"/api/v1/system-stats", "/api/v1/system-detect"}:
        return "settings"
    if lowered.startswith("/api/v1/"):
        return "studio"
    return "other"


def _operation_identifier(method: str, path: str) -> str:
    pieces: list[str] = [method.lower()]
    for segment in path.split("/"):
        if not segment or segment in {"api", "v1"}:
            continue
        match = _PLACEHOLDER.fullmatch(segment)
        if match:
            pieces.extend(("by", match.group(1)))
        else:
            pieces.append(segment)
    slug = _WORD.sub("_", "_".join(pieces).casefold()).strip("_") or "operation"
    return "api." + slug


def _path_has_unsafe_capture(path: str) -> bool:
    # Routes that capture arbitrary path tails can act as file proxies. They
    # are deliberately served through explicit asset tools instead.
    if re.search(r"\{[^}]*:(?:path|str)\}", path, re.I):
        return True
    if "*" in path:
        return True
    return False


def _is_json_media_type(media_type: Any) -> bool:
    media = str(media_type).casefold()
    return media == "application/json" or media.endswith("+json")


def _request_description(operation: Mapping[str, Any]) -> tuple[str, str | None] | None:
    """Return a supported request kind/media type, or None for unsupported bodies."""
    request_body = operation.get("requestBody")
    if request_body is None:
        return "none", None
    if not isinstance(request_body, Mapping):
        return None
    content = request_body.get("content")
    if not isinstance(content, Mapping):
        return None
    for media in content:
        if _is_json_media_type(media):
            return "json", str(media)
    for media in content:
        if str(media).casefold() == "multipart/form-data":
            return "multipart", str(media)
    return None


def _success_media_types(operation: Mapping[str, Any]) -> tuple[str, ...]:
    responses = operation.get("responses")
    if not isinstance(responses, Mapping):
        return ()
    media: list[str] = []
    for status, response in responses.items():
        code = str(status).casefold()
        if code != "default" and not code.startswith("2"):
            continue
        if not isinstance(response, Mapping):
            continue
        content = response.get("content")
        if isinstance(content, Mapping):
            media.extend(str(value) for value in content)
    return tuple(dict.fromkeys(media))


def _response_kind(path: str, operation: Mapping[str, Any]) -> str:
    if any(pattern.fullmatch(path) for pattern in _BINARY_RESPONSE_PATTERNS):
        return "binary"
    media = _success_media_types(operation)
    if any(not _is_json_media_type(value) for value in media):
        return "binary"
    if media:
        return "json"
    responses = operation.get("responses")
    if isinstance(responses, Mapping) and any(
        str(status).casefold() == "204"
        and isinstance(response, Mapping)
        and not response.get("content")
        for status, response in responses.items()
    ):
        return "none"
    # A response with no declared content is common in older JSON endpoints.
    return "json"


def _supported_request_body(operation: Mapping[str, Any]) -> bool:
    return _request_description(operation) is not None


_NATIVE_BODY_SCHEMA = {
    "type": "object",
    "description": "Maestro validates this JSON body; OpenAPI does not publish its field schema.",
    "additionalProperties": True,
}


def _request_has_schema(request_body: Mapping[str, Any] | None, media_type: str | None) -> bool:
    if not isinstance(request_body, Mapping) or not isinstance(media_type, str):
        return False
    content = request_body.get("content")
    if not isinstance(content, Mapping):
        return False
    record = next(
        (value for name, value in content.items() if str(name).casefold() == media_type.casefold()),
        None,
    )
    return isinstance(record, Mapping) and "schema" in record


def _native_validated_request_body(raw: Any) -> dict[str, Any]:
    """Advertise an optional object body when a mutation hides Request.json()."""
    description = "Maestro validates this JSON body; OpenAPI does not publish its field schema."
    required = False
    if isinstance(raw, Mapping):
        required = raw.get("required") is True
        if isinstance(raw.get("description"), str) and raw["description"].strip():
            description = (raw["description"].strip() + " " + description)[:1000]
    return {
        "required": required,
        "description": description,
        "content": {"application/json": {"schema": _NATIVE_BODY_SCHEMA}},
    }


def _merge_parameters(path_item: Mapping[str, Any], operation: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    merged: dict[tuple[str, str], Mapping[str, Any]] = {}
    for source in (path_item.get("parameters", ()), operation.get("parameters", ())):
        if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
            continue
        for parameter in source:
            if isinstance(parameter, Mapping) and isinstance(parameter.get("name"), str):
                key = (str(parameter.get("in", "")), parameter["name"])
                merged[key] = parameter
    return tuple(merged[key] for key in sorted(merged))


def _schema_is_file(
    schema: Any,
    openapi: Mapping[str, Any],
    *,
    depth: int = 0,
    active_refs: frozenset[str] = frozenset(),
) -> bool:
    if depth > 16 or not isinstance(schema, Mapping):
        return False
    reference = schema.get("$ref")
    if isinstance(reference, str) and reference not in active_refs:
        resolved = _resolve_ref(schema, openapi)
        if isinstance(resolved, Mapping) and _schema_is_file(
            resolved,
            openapi,
            depth=depth + 1,
            active_refs=active_refs | {reference},
        ):
            return True
    media = str(schema.get("contentMediaType", "")).casefold()
    if schema.get("format") == "binary" or schema.get("format") == "byte":
        return True
    if media and not _is_json_media_type(media) and not media.startswith("text/"):
        return True
    items = schema.get("items")
    if items is not None and _schema_is_file(items, openapi, depth=depth + 1, active_refs=active_refs):
        return True
    for composition in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(composition)
        if isinstance(branches, Sequence) and not isinstance(branches, (str, bytes)):
            if any(_schema_is_file(branch, openapi, depth=depth + 1, active_refs=active_refs) for branch in branches[:16]):
                return True
    return False


def _multipart_file_fields(
    openapi: Mapping[str, Any],
    request_body: Mapping[str, Any],
) -> tuple[str, ...]:
    content = request_body.get("content")
    multipart = next(
        (
            value
            for media, value in content.items()
            if str(media).casefold() == "multipart/form-data" and isinstance(value, Mapping)
        ),
        None,
    ) if isinstance(content, Mapping) else None
    schema = multipart.get("schema") if isinstance(multipart, Mapping) else None
    if not isinstance(schema, Mapping):
        return ()
    resolved = _resolve_ref(schema, openapi) if isinstance(schema.get("$ref"), str) else schema
    if not isinstance(resolved, Mapping):
        return ()
    properties = resolved.get("properties")
    if not isinstance(properties, Mapping):
        return ()
    return tuple(
        sorted(
            str(name)
            for name, property_schema in properties.items()
            if isinstance(name, str) and _schema_is_file(property_schema, openapi)
        )
    )


def build_catalog(openapi: Mapping[str, Any]) -> dict[str, Operation]:
    """Build callable operation IDs from registered JSON/multipart API routes.

    Binary responses remain named operations and are captured into private
    transfer records. Agent auth routes, classic UI routes, dedicated asset
    upload paths, arbitrary file proxies, and arbitrary path captures remain
    excluded by construction. Mutations whose FastAPI ``Request.json()`` body
    is absent from OpenAPI accept an optional generic JSON object and delegate
    validation to the native handler; their missing field schema is explicit.
    """

    paths = openapi.get("paths") if isinstance(openapi, Mapping) else None
    if not isinstance(paths, Mapping):
        return {}
    entries: list[tuple[str, Operation]] = []
    for path, path_item in paths.items():
        if not isinstance(path, str) or not path.startswith("/api/v1/"):
            continue
        lowered = path.casefold()
        if "/agent-access" in lowered or "/classic" in lowered:
            continue
        if _path_has_unsafe_capture(path):
            continue
        # These file transfer routes deliberately use the asset transfer API,
        # which maps an opaque upload ID to an installation-local file.
        if lowered == "/api/v1/upload" or lowered.startswith("/api/v1/uploads/"):
            continue
        if any(fragment in lowered for fragment in ("/api/v1/file/", "/api/v1/media-info/", "/api/v1/thumbnail/")):
            continue
        if not isinstance(path_item, Mapping):
            continue
        for method, raw_operation in path_item.items():
            method_lower = str(method).lower()
            if method_lower not in HTTP_METHODS or not isinstance(raw_operation, Mapping):
                continue
            if raw_operation.get("deprecated") is True:
                continue
            request_description = _request_description(raw_operation)
            if request_description is None:
                continue
            request_kind, request_media_type = request_description
            request_body = raw_operation.get("requestBody")
            if not isinstance(request_body, Mapping):
                request_body = None
            request_schema_available = (
                request_kind == "none"
                or _request_has_schema(request_body, request_media_type)
            )
            if request_kind == "none" and method_lower in {"post", "put", "patch", "delete"}:
                # FastAPI's Request.json() handlers are absent from OpenAPI's
                # body schema. Expose their native JSON path without inventing
                # field constraints; the backend remains the validator.
                request_kind = "json"
                request_media_type = "application/json"
                request_body = _native_validated_request_body(None)
                request_schema_available = False
            elif request_kind == "json" and not request_schema_available:
                request_body = _native_validated_request_body(request_body)
            operation_id = _operation_identifier(method_lower, path)
            description = raw_operation.get("description")
            summary = raw_operation.get("summary") or raw_operation.get("operationId") or f"{method_lower.upper()} {path}"
            if not isinstance(description, str):
                description = ""
            if not isinstance(summary, str):
                summary = f"{method_lower.upper()} {path}"
            entry = Operation(
                operation_id=operation_id,
                method=method_lower.upper(),
                path=path,
                domain=operation_domain(path),
                scope=required_scope(method_lower, path),
                summary=summary.strip()[:240],
                description=description.strip()[:4_000],
                parameters=_merge_parameters(path_item, raw_operation),
                request_body=request_body,
                responses=raw_operation.get("responses", {}) if isinstance(raw_operation.get("responses"), Mapping) else {},
                request_kind=request_kind,
                response_kind=_response_kind(path, raw_operation),
                request_media_type=request_media_type,
                file_fields=_multipart_file_fields(openapi, request_body) if request_kind == "multipart" and request_body else (),
                request_schema_available=request_schema_available,
            )
            entries.append((operation_id, entry))

    catalog: dict[str, Operation] = {}
    for operation_id, entry in sorted(entries, key=lambda pair: (pair[1].path, pair[1].method)):
        final_id = operation_id
        if final_id in catalog:
            # Hash the complete route to make collisions stable across runs.
            import hashlib

            digest = hashlib.sha256(f"{entry.method} {entry.path}".encode("utf-8")).hexdigest()[:8]
            final_id = f"{operation_id}.{digest}"
        catalog[final_id] = Operation(
            operation_id=final_id,
            method=entry.method,
            path=entry.path,
            domain=entry.domain,
            scope=entry.scope,
            summary=entry.summary,
            description=entry.description,
            parameters=entry.parameters,
            request_body=entry.request_body,
            responses=entry.responses,
            request_kind=entry.request_kind,
            response_kind=entry.response_kind,
            request_media_type=entry.request_media_type,
            file_fields=entry.file_fields,
            request_schema_available=entry.request_schema_available,
        )
    return catalog


def _public_schema(
    schema: Any,
    openapi: Mapping[str, Any] | None = None,
    depth: int = 0,
    active_refs: frozenset[str] = frozenset(),
) -> Any:
    """Return a compact JSON-schema view without examples or secret fields."""

    if depth > 5:
        return {"type": "object", "description": "Nested schema omitted beyond the guide depth limit."}
    if isinstance(schema, bool):
        return schema
    if not isinstance(schema, Mapping):
        return {}
    reference = schema.get("$ref")
    if isinstance(reference, str) and openapi is not None and reference not in active_refs:
        resolved = _resolve_ref(schema, openapi)
        if isinstance(resolved, Mapping):
            expanded = _public_schema(resolved, openapi, depth + 1, active_refs | {reference})
            if isinstance(expanded, dict):
                return {"$ref": reference, **expanded}
            return {"$ref": reference, "schema": expanded}
    public: dict[str, Any] = {}
    for key in ("$ref", "type", "title", "description", "format", "contentMediaType", "contentEncoding", "enum", "default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength", "pattern", "minItems", "maxItems", "minProperties", "maxProperties", "nullable", "readOnly", "writeOnly"):
        if key in schema:
            value = schema[key]
            if key == "description" and isinstance(value, str):
                value = value[:600]
            if key in {"enum", "default"} and isinstance(value, str) and len(value) > 160:
                value = value[:157] + "..."
            public[key] = value
    required = schema.get("required")
    if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
        public["required"] = [name for name in required if isinstance(name, str) and not _PRIVATE_PROPERTY.search(name)]
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        safe_properties = {
            str(name): _public_schema(value, openapi, depth + 1, active_refs)
            for name, value in properties.items()
            if isinstance(name, str) and not _PRIVATE_PROPERTY.search(name)
        }
        if safe_properties:
            public["properties"] = safe_properties
    if "items" in schema:
        public["items"] = _public_schema(schema["items"], openapi, depth + 1, active_refs)
    for key in ("oneOf", "anyOf", "allOf"):
        variants = schema.get(key)
        if isinstance(variants, Sequence) and not isinstance(variants, (str, bytes)):
            public[key] = [_public_schema(item, openapi, depth + 1, active_refs) for item in variants[:8]]
    additional = schema.get("additionalProperties")
    if isinstance(additional, bool):
        public["additionalProperties"] = additional
    elif isinstance(additional, Mapping):
        public["additionalProperties"] = _public_schema(additional, openapi, depth + 1, active_refs)
    return public


def operation_summary(operation: Operation) -> dict[str, Any]:
    """Compact list entry for operation discovery."""

    body = operation.request_body or {}
    return {
        "operation_id": operation.operation_id,
        "method": operation.method,
        "path": operation.path,
        "domain": operation.domain,
        "scope": operation.scope,
        "summary": operation.summary,
        "request_kind": operation.request_kind,
        "response_kind": operation.response_kind,
        "has_body": bool(body),
        "body_required": bool(body.get("required", False)),
        "request_schema_available": operation.request_schema_available,
        "file_fields": list(operation.file_fields),
    }


def operation_description(
    operation: Operation,
    openapi: Mapping[str, Any] | None = None,
    *,
    request_field_hints: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the operation's caller-facing description and useful schemas."""

    parameters: list[dict[str, Any]] = []
    for parameter in operation.parameters:
        schema = parameter.get("schema") if isinstance(parameter.get("schema"), Mapping) else {}
        name = parameter.get("name")
        if not isinstance(name, str) or _PRIVATE_PROPERTY.search(name):
            continue
        parameters.append(
            {
                "name": name,
                "in": str(parameter.get("in", "query")),
                "required": bool(parameter.get("required", False)),
                "description": str(parameter.get("description", ""))[:600],
                "schema": _public_schema(schema, openapi),
            }
        )
    request_body: dict[str, Any] | None = None
    if operation.request_body:
        content = operation.request_body.get("content")
        schemas: dict[str, Any] = {}
        if isinstance(content, Mapping):
            for media, record in content.items():
                if not isinstance(record, Mapping) or not isinstance(record.get("schema"), Mapping):
                    continue
                if _is_json_media_type(media) or str(media).casefold() == "multipart/form-data":
                    schemas[str(media)] = _public_schema(record["schema"], openapi)
        request_body = {
            "kind": operation.request_kind,
            "required": bool(operation.request_body.get("required", False)),
            "description": str(operation.request_body.get("description", ""))[:1000],
            "content": schemas,
            "file_fields": list(operation.file_fields),
            "schema_available": operation.request_schema_available,
            "validation": "schema" if operation.request_schema_available else "native; OpenAPI body schema unavailable",
        }
        if not operation.request_schema_available and isinstance(request_field_hints, Mapping):
            request_body["field_hints"] = dict(request_field_hints)
    responses: dict[str, Any] = {}
    for status, response in operation.responses.items():
        if not isinstance(response, Mapping):
            continue
        content = response.get("content")
        schemas: dict[str, Any] = {}
        is_success = str(status).casefold() == "default" or str(status).startswith("2")
        if isinstance(content, Mapping):
            for media, record in content.items():
                if not isinstance(record, Mapping) or not isinstance(record.get("schema"), Mapping):
                    continue
                if operation.response_kind == "binary" and is_success:
                    continue
                if _is_json_media_type(media):
                    schemas[str(media)] = _public_schema(record["schema"], openapi)
        responses[str(status)] = {
            "description": str(response.get("description", ""))[:600],
            "kind": operation.response_kind if is_success else "json",
            "content": schemas,
        }
    return {
        **operation_summary(operation),
        "response_kind": operation.response_kind,
        "description": operation.description or operation.summary,
        "parameters": parameters,
        "request_body": request_body,
        "responses": responses,
    }


def domain_counts(catalog: Mapping[str, Operation]) -> dict[str, int]:
    """Return all requested catalogue groups, including empty groups."""

    counts = {domain: 0 for domain in DOMAIN_ORDER}
    for operation in catalog.values():
        counts[operation.domain] = counts.get(operation.domain, 0) + 1
    return counts


def normalize_domain(value: str | None) -> str | None:
    """Normalize a human-facing domain name; ``None`` means no filter."""

    if value is None or not str(value).strip() or str(value).strip().casefold() in {"all", "*"}:
        return None
    token = str(value).strip().casefold()
    aliases = {"studio": "studio", "models": "models", "model": "models", "assets": "assets", "asset": "assets", "settings": "settings", "setting": "settings", "director": "director", "editor": "editor", "other": "other"}
    return aliases.get(token)


def _resolve_ref(schema: Mapping[str, Any], openapi: Mapping[str, Any]) -> Mapping[str, Any] | bool:
    reference = schema.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/"):
        return schema
    node: Any = openapi
    try:
        for part in reference[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            node = node[part]
    except (KeyError, IndexError, TypeError):
        raise SchemaValidationError("OpenAPI schema contains an unresolved local reference") from None
    if not isinstance(node, (Mapping, bool)):
        raise SchemaValidationError("OpenAPI schema reference is not a JSON schema")
    return node


def validate_json_schema(
    value: Any,
    schema: Any,
    openapi: Mapping[str, Any],
    *,
    label: str = "value",
    _depth: int = 0,
    _active_refs: frozenset[str] = frozenset(),
) -> None:
    """Validate common OpenAPI 3 JSON Schema keywords using stdlib only.

    Maestro's endpoints use object, array, scalar, enum and local ``$ref``
    schemas. This validator covers those constraints and composition keywords
    without importing a schema package into the model runtime.
    """

    if _depth > 64:
        raise SchemaValidationError(f"{label} exceeds the supported schema depth")
    if schema is True or schema is None:
        return
    if schema is False:
        raise SchemaValidationError(f"{label} is not accepted by this operation")
    if not isinstance(schema, Mapping):
        return

    reference = schema.get("$ref")
    if isinstance(reference, str):
        if reference in _active_refs:
            return
        resolved = _resolve_ref(schema, openapi)
        if isinstance(resolved, Mapping):
            validate_json_schema(
                value,
                resolved,
                openapi,
                label=label,
                _depth=_depth + 1,
                _active_refs=_active_refs | {reference},
            )
        return

    for composition in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(composition)
        if not isinstance(branches, Sequence) or isinstance(branches, (str, bytes)):
            continue
        results: list[bool] = []
        errors: list[str] = []
        for branch in branches:
            try:
                validate_json_schema(value, branch, openapi, label=label, _depth=_depth + 1, _active_refs=_active_refs)
                results.append(True)
            except SchemaValidationError as error:
                results.append(False)
                errors.append(str(error))
        if composition == "allOf" and not all(results):
            raise SchemaValidationError(errors[0] if errors else f"{label} does not match all required schemas")
        if composition == "anyOf" and not any(results):
            raise SchemaValidationError(f"{label} does not match an allowed schema")
        if composition == "oneOf" and sum(results) != 1:
            raise SchemaValidationError(f"{label} must match exactly one allowed schema")

    nullable = schema.get("nullable") is True
    declared_type = schema.get("type")
    types = declared_type if isinstance(declared_type, Sequence) and not isinstance(declared_type, (str, bytes)) else [declared_type]
    if value is None and nullable:
        return
    if declared_type is not None and not any(_matches_schema_type(value, kind) for kind in types):
        type_names = ", ".join(str(kind) for kind in types)
        raise SchemaValidationError(f"{label} must be {type_names}")

    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(f"{label} must be one of the documented choices")

    if isinstance(value, Mapping):
        required = schema.get("required", ())
        if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
            missing = [name for name in required if name not in value]
            if missing:
                raise SchemaValidationError(f"{label} is missing required field(s): {', '.join(map(str, missing))}")
        properties = schema.get("properties", {})
        properties = properties if isinstance(properties, Mapping) else {}
        minimum = schema.get("minProperties")
        maximum = schema.get("maxProperties")
        if isinstance(minimum, int) and len(value) < minimum:
            raise SchemaValidationError(f"{label} must contain at least {minimum} fields")
        if isinstance(maximum, int) and len(value) > maximum:
            raise SchemaValidationError(f"{label} must contain at most {maximum} fields")
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            child_label = f"{label}.{key}"
            if key in properties:
                validate_json_schema(item, properties[key], openapi, label=child_label, _depth=_depth + 1, _active_refs=_active_refs)
            elif additional is False:
                raise SchemaValidationError(f"{label} contains an undocumented field: {key}")
            elif isinstance(additional, Mapping):
                validate_json_schema(item, additional, openapi, label=child_label, _depth=_depth + 1, _active_refs=_active_refs)
        return

    if isinstance(value, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            raise SchemaValidationError(f"{label} must contain at least {minimum} items")
        if isinstance(maximum, int) and len(value) > maximum:
            raise SchemaValidationError(f"{label} must contain at most {maximum} items")
        items = schema.get("items")
        if items is not None:
            for index, item in enumerate(value):
                validate_json_schema(item, items, openapi, label=f"{label}[{index}]", _depth=_depth + 1, _active_refs=_active_refs)
        return

    if isinstance(value, str):
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if isinstance(minimum, int) and len(value) < minimum:
            raise SchemaValidationError(f"{label} must be at least {minimum} characters")
        if isinstance(maximum, int) and len(value) > maximum:
            raise SchemaValidationError(f"{label} must be at most {maximum} characters")
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                matched = re.search(pattern, value)
            except re.error:
                matched = True  # An invalid advertised regex should not open access.
            if not matched:
                raise SchemaValidationError(f"{label} does not match the required format")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise SchemaValidationError(f"{label} must be finite")
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            raise SchemaValidationError(f"{label} must be at least {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            raise SchemaValidationError(f"{label} must be at most {maximum}")
        exclusive_min = schema.get("exclusiveMinimum")
        exclusive_max = schema.get("exclusiveMaximum")
        if isinstance(exclusive_min, (int, float)) and value <= exclusive_min:
            raise SchemaValidationError(f"{label} must be greater than {exclusive_min}")
        if isinstance(exclusive_max, (int, float)) and value >= exclusive_max:
            raise SchemaValidationError(f"{label} must be less than {exclusive_max}")


def _matches_schema_type(value: Any, schema_type: Any) -> bool:
    if schema_type in {None, ""}:
        return True
    if schema_type == "null":
        return value is None
    if schema_type == "object":
        return isinstance(value, Mapping)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def _parameter_schema(parameter: Mapping[str, Any]) -> Mapping[str, Any]:
    schema = parameter.get("schema")
    return schema if isinstance(schema, Mapping) else {}


def validate_operation_inputs(
    openapi: Mapping[str, Any],
    operation: Operation,
    path_params: Mapping[str, Any] | None,
    query: Mapping[str, Any] | None,
    body: Any,
) -> tuple[str, dict[str, Any] | None, Any]:
    """Validate path/query/body values and safely fill a registered route.

    Path parameters are URL-encoded as single path segments. They cannot be
    used to introduce a slash, a backslash, a dot traversal segment, or a NUL.
    """

    provided_path = dict(path_params or {})
    provided_query = dict(query or {})
    template_names = _PLACEHOLDER.findall(operation.path)
    if set(provided_path) != set(template_names):
        missing = sorted(set(template_names) - set(provided_path))
        extra = sorted(set(provided_path) - set(template_names))
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unknown " + ", ".join(extra))
        raise SchemaValidationError("Path parameters must match this operation: " + "; ".join(detail))

    parameter_map = {(str(item.get("in", "")), str(item.get("name", ""))): item for item in operation.parameters}
    path_schema_params = {name: parameter_map.get(("path", name), {}) for name in template_names}
    route = operation.path
    for name in template_names:
        value = provided_path[name]
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise SchemaValidationError(f"Path parameter {name} must be a string or number")
        text = str(value)
        if not text or text in {".", ".."} or any(ch in text for ch in ("/", "\\", "\x00")):
            raise SchemaValidationError(f"Path parameter {name} must be a single safe path segment")
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
            raise SchemaValidationError(f"Path parameter {name} contains a control character")
        parameter = path_schema_params[name]
        if parameter:
            validate_json_schema(value, _parameter_schema(parameter), openapi, label=f"path_params.{name}")
        route = re.sub(r"\{" + re.escape(name) + r"(?::[^}]+)?\}", quote(text, safe=""), route, count=1)

    query_params = {name: item for (location, name), item in parameter_map.items() if location == "query"}
    for name, parameter in query_params.items():
        if name not in provided_query and parameter.get("required") is True:
            raise SchemaValidationError(f"query is missing required field: {name}")
    unknown_query = sorted(set(provided_query) - set(query_params))
    if unknown_query:
        raise SchemaValidationError("Query contains undocumented field(s): " + ", ".join(map(str, unknown_query)))
    for name, value in provided_query.items():
        validate_json_schema(value, _parameter_schema(query_params[name]), openapi, label=f"query.{name}")

    request_body = operation.request_body
    if request_body is None:
        if body is not None:
            raise SchemaValidationError("This operation does not accept a request body")
        return route, provided_query or None, None

    content = request_body.get("content")
    body_schema: Any = None
    desired_media = operation.request_media_type
    if isinstance(content, Mapping):
        for media_type, record in content.items():
            media_matches = str(media_type).casefold() == str(desired_media or "").casefold()
            if media_matches and isinstance(record, Mapping):
                body_schema = record.get("schema")
                break
    if body is None and request_body.get("required") is True:
        kind_name = "multipart" if operation.request_kind == "multipart" else "JSON"
        raise SchemaValidationError(f"This operation requires a {kind_name} body")
    if body is not None and body_schema is None:
        raise SchemaValidationError("This operation does not publish a request body schema")
    if body is not None:
        validate_json_schema(body, body_schema, openapi, label="body")
        if operation.request_kind == "multipart":
            if not isinstance(body, Mapping):
                raise SchemaValidationError("Multipart body must be an object of documented form fields")
            for field_name in operation.file_fields:
                if field_name not in body or body[field_name] is None:
                    continue
                values = body[field_name] if isinstance(body[field_name], list) else [body[field_name]]
                for value in values:
                    if value is None:
                        continue
                    if not isinstance(value, str) or not _ASSET_REFERENCE.fullmatch(value):
                        raise SchemaValidationError(
                            f"Multipart file field {field_name} must use an owned asset:<UUID> reference"
                        )
            for field_name, value in body.items():
                if field_name in operation.file_fields:
                    continue
                candidates = value if isinstance(value, list) else [value]
                if any(isinstance(item, str) and item.startswith(("asset:", "maestro-asset:")) for item in candidates):
                    raise SchemaValidationError(
                        "Multipart asset references are only accepted in documented file fields"
                    )
    return route, provided_query or None, body


__all__ = [
    "DOMAIN_ORDER",
    "Operation",
    "SchemaValidationError",
    "build_catalog",
    "domain_counts",
    "normalize_domain",
    "operation_description",
    "operation_domain",
    "operation_summary",
    "required_scope",
    "validate_json_schema",
    "validate_operation_inputs",
]
