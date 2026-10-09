#!/usr/bin/env python3
"""Run a bounded, serial WanGP H3 Pruned benchmark through MCP v2."""
from __future__ import annotations

import argparse
from copy import deepcopy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

EXPECTED_MODEL_TYPE = "minimax_h3_fl2va_pruned"
UNSUPPORTED_H3_SETTINGS = frozenset({"generation_mode", "guidance_scale", "negative_prompt"})
REQUIRED_H3_DEFAULTS = ("video_length", "num_inference_steps", "flow_shift")
MAX_MANIFEST_BYTES = 1_000_000
MAX_CASES = 12
MAX_REPEATS = 3
MAX_TOTAL_RUNS = 24
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_EVENTS = 500
MCP_PROTOCOL_VERSION = "2025-03-26"
DEFAULT_TIMEOUT = 1800.0
DEFAULT_REQUEST_TIMEOUT = 20.0
DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_CANCEL_GRACE = 30.0


class BenchmarkError(Exception):
    """Invalid manifest, unsupported MCP contract, or benchmark error."""


class MCPError(BenchmarkError):
    """HTTP, JSON-RPC, or WanGP tool error."""


class MCPToolError(MCPError):
    pass


class BusyQueueError(BenchmarkError):
    def __init__(self, queue: dict[str, Any]):
        self.queue = queue
        counts = queue_counts(queue)
        super().__init__(
            f"WanGP queue is busy (total={counts['total_count']}, "
            f"queued={counts['queued_count']}, running={counts['running_count']}); "
            "refusing to submit benchmark jobs"
        )


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def validate_base_url(value: str) -> tuple[str, str]:
    try:
        parsed = urlsplit(value.strip())
        _ = parsed.port
    except (AttributeError, ValueError) as error:
        raise BenchmarkError(f"Invalid MCP base URL: {value!r}") from error
    if parsed.scheme.lower() not in {"http", "https"}:
        raise BenchmarkError("MCP base URL must use http or https")
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise BenchmarkError("MCP base URL must have a host and no embedded credentials")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/", "/mcp"}:
        raise BenchmarkError("MCP base URL must be an origin, optionally ending in /mcp")
    origin = urlunsplit((parsed.scheme.lower(), parsed.netloc, "", "", ""))
    return origin, origin + "/mcp"


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{label} must be a JSON object")
    return value


def _validate_request_settings(settings: dict[str, Any], label: str) -> None:
    unsupported = sorted(UNSUPPORTED_H3_SETTINGS.intersection(settings))
    if unsupported:
        raise BenchmarkError(
            f"{label} includes unsupported H3 setting(s): {', '.join(unsupported)}; "
            "H3 uses implicit video generation and CFG 1.0"
        )
    controls = sorted({"wait", "timeout_s", "event_limit"}.intersection(settings))
    if controls:
        raise BenchmarkError(f"{label} contains MCP control field(s): {', '.join(controls)}")


def validate_h3_defaults(manifest: dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
    """Check that this server declares the H3 fields the baseline overrides."""
    defaults = details.get("defaults")
    if not isinstance(defaults, dict):
        raise BenchmarkError("WanGP H3 model defaults did not return a JSON object")
    missing = [field for field in REQUIRED_H3_DEFAULTS if field not in defaults]
    if missing:
        raise BenchmarkError(
            "WanGP H3 defaults omit required baseline field(s): " + ", ".join(missing)
        )
    runs = manifest.get("runs", [])
    requested = set().union(*(
        run.get("settings", {}).keys()
        for run in runs if isinstance(run, dict) and isinstance(run.get("settings"), dict)
    )) if runs else set()
    invalid_types = []
    for field in REQUIRED_H3_DEFAULTS:
        default = defaults[field]
        expected_type = type(default)
        for run in runs:
            settings = run.get("settings", {}) if isinstance(run, dict) else {}
            if field not in settings:
                continue
            value = settings[field]
            # JSON has distinct booleans; permit int/float interchange for numeric settings.
            is_number_pair = (
                isinstance(default, (int, float)) and not isinstance(default, bool)
                and isinstance(value, (int, float)) and not isinstance(value, bool)
            )
            if not is_number_pair and type(value) is not expected_type:
                invalid_types.append({
                    "field": field,
                    "expected_type": expected_type.__name__,
                    "actual_type": type(value).__name__,
                })
                break
    if invalid_types:
        summary = ", ".join(
            f"{item['field']} expects {item['expected_type']}, got {item['actual_type']}"
            for item in invalid_types
        )
        raise BenchmarkError("WanGP H3 baseline setting type mismatch: " + summary)
    return {
        "status": "valid",
        "required_model_defaults": list(REQUIRED_H3_DEFAULTS),
        "declared_model_defaults": sorted(defaults),
        "requested_fields_absent_from_model_defaults": sorted(requested - defaults.keys()),
        "absent_defaults_note": (
            "Model selection, prompt, resolution/seed, per-model config selectors, and "
            "API-declared task controls may be supplied outside the pristine H3 defaults. "
            "This check requires only the H3 generation fields overridden by this baseline."
        ),
    }


def normalize_manifest(raw: Any) -> dict[str, Any]:
    """Validate and expand root settings plus named case overrides."""
    value = _object(raw, "manifest")
    extra = set(value) - {"name", "settings", "request", "cases"}
    if extra:
        raise BenchmarkError(f"Unsupported manifest fields: {', '.join(sorted(extra))}")
    name = str(value.get("name", "wan2gp-h3-benchmark")).strip()
    if not name or len(name) > 100:
        raise BenchmarkError("manifest name must contain 1 to 100 characters")
    common = _object(value.get("settings", value.get("request", {})), "manifest settings")
    _validate_request_settings(common, "manifest settings")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise BenchmarkError("manifest cases must be a non-empty list")
    if len(cases) > MAX_CASES:
        raise BenchmarkError(f"manifest has too many cases (maximum {MAX_CASES})")

    runs: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(cases):
        case = _object(item, f"cases[{index}]")
        extra = set(case) - {"name", "settings", "request", "repeats"}
        if extra:
            raise BenchmarkError(f"cases[{index}] has unsupported fields: {', '.join(sorted(extra))}")
        case_name = str(case.get("name", "")).strip()
        if not case_name or len(case_name) > 100:
            raise BenchmarkError(f"cases[{index}].name must contain 1 to 100 characters")
        if case_name in names:
            raise BenchmarkError(f"case names must be unique; duplicate: {case_name!r}")
        names.add(case_name)
        overrides = _object(case.get("settings", case.get("request", {})), f"cases[{index}].settings")
        _validate_request_settings(overrides, f"cases[{index}].settings")
        settings = deepcopy(common)
        settings.update(deepcopy(overrides))
        if settings.get("model_type") != EXPECTED_MODEL_TYPE:
            raise BenchmarkError(f"cases[{index}] must use model_type={EXPECTED_MODEL_TYPE!r}")
        if not isinstance(settings.get("prompt"), str) or not settings["prompt"].strip():
            raise BenchmarkError(f"cases[{index}] needs a non-empty prompt")
        repeats = case.get("repeats", 1)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= MAX_REPEATS:
            raise BenchmarkError(f"cases[{index}].repeats must be between 1 and {MAX_REPEATS}")
        runs.extend(
            {"name": case_name, "repeat": repeat, "settings": deepcopy(settings)}
            for repeat in range(1, repeats + 1)
        )
    if len(runs) > MAX_TOTAL_RUNS:
        raise BenchmarkError(f"expanded manifest exceeds {MAX_TOTAL_RUNS} total runs")
    return {"name": name, "settings": deepcopy(common), "runs": runs}


def load_manifest(path: str | os.PathLike[str]) -> tuple[dict[str, Any], str]:
    manifest_path = Path(path).expanduser()
    try:
        raw_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise BenchmarkError(f"Could not read manifest {manifest_path}: {error}") from error
    if len(raw_bytes) > MAX_MANIFEST_BYTES:
        raise BenchmarkError(f"manifest exceeds {MAX_MANIFEST_BYTES} bytes")
    try:
        raw = json.loads(raw_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"Manifest is not valid UTF-8 JSON: {error}") from error
    return normalize_manifest(raw), hashlib.sha256(raw_bytes).hexdigest()


def _sse_messages(raw: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    data_lines: list[str] = []
    for line in raw.splitlines() + [""]:
        if not line:
            if data_lines:
                data = "\n".join(data_lines)
                data_lines.clear()
                if data.strip() and data.strip() != "[DONE]":
                    try:
                        value = json.loads(data)
                    except json.JSONDecodeError as error:
                        raise MCPError(f"Invalid JSON in MCP event stream: {error}") from error
                    if isinstance(value, dict):
                        result.append(value)
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    return result


class StreamableHTTPMCP:
    """Minimal Streamable HTTP transport implemented with urllib."""

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        headers: dict[str, str] | None = None,
        opener: Callable[..., Any] | None = None,
    ):
        self.origin, self.endpoint = validate_base_url(base_url)
        if not 1 <= request_timeout <= 120:
            raise BenchmarkError("request timeout must be between 1 and 120 seconds")
        self.request_timeout = float(request_timeout)
        self.headers = dict(headers or {})
        self.opener = opener or urlopen
        self.session_id: str | None = None
        self.protocol_version: str | None = None
        self.request_id = 0
        self.initialized = False

    def _post(self, payload: dict[str, Any], timeout: float | None = None) -> tuple[bytes, str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self.headers,
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.protocol_version:
            headers["MCP-Protocol-Version"] = self.protocol_version
        request = Request(self.endpoint, data=body, headers=headers, method="POST")
        try:
            response = self.opener(request, timeout=timeout or self.request_timeout)
            try:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                content_type = response.headers.get("Content-Type", "")
                response_headers = response.headers
            finally:
                close = getattr(response, "close", None)
                if close:
                    close()
        except HTTPError as error:
            detail = error.read(4096).decode("utf-8", errors="replace").strip()
            raise MCPError(f"POST {self.endpoint} failed ({error.code}): {detail or error.reason}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise MCPError(f"POST {self.endpoint} failed: {getattr(error, 'reason', error)}") from error
        if len(raw) > MAX_RESPONSE_BYTES:
            raise MCPError(f"MCP response exceeded {MAX_RESPONSE_BYTES} bytes")
        return raw, content_type, response_headers

    def _decode(self, raw: bytes, content_type: str, request_id: int | None) -> dict[str, Any] | None:
        if not raw.strip():
            return None
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise MCPError("MCP response was not UTF-8") from error
        if "text/event-stream" in content_type.lower():
            messages = _sse_messages(text)
        else:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                messages = _sse_messages(text)
            else:
                messages = parsed if isinstance(parsed, list) else [parsed]
        for message in messages:
            if isinstance(message, dict) and (request_id is None or message.get("id") == request_id):
                return message
        raise MCPError("MCP response had no matching JSON-RPC message")

    def _rpc(self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        self.request_id += 1
        request_id = self.request_id
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        raw, content_type, response_headers = self._post(payload, timeout)
        self.session_id = response_headers.get("Mcp-Session-Id") or self.session_id
        message = self._decode(raw, content_type, request_id)
        if message is None:
            raise MCPError(f"MCP returned an empty response to {method}")
        if "error" in message:
            error = message["error"]
            detail = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            raise MCPError(f"MCP {method} failed: {detail}")
        return message.get("result")

    def _notify(self, method: str) -> None:
        raw, content_type, _ = self._post({"jsonrpc": "2.0", "method": method})
        if raw.strip():
            self._decode(raw, content_type, None)

    def initialize(self) -> None:
        result = self._rpc("initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "Maestro WanGP benchmark", "version": "1.0"},
        })
        if not isinstance(result, dict):
            raise MCPError("MCP initialize returned an invalid result")
        self.protocol_version = result.get("protocolVersion") or MCP_PROTOCOL_VERSION
        self._notify("notifications/initialized")
        self.initialized = True

    def list_tools(self) -> list[dict[str, Any]]:
        if not self.initialized:
            raise MCPError("Initialize MCP before listing tools")
        cursor: str | None = None
        result: list[dict[str, Any]] = []
        while True:
            page = self._rpc("tools/list", {"cursor": cursor} if cursor else {})
            if not isinstance(page, dict) or not isinstance(page.get("tools"), list):
                raise MCPError("MCP tools/list returned an invalid result")
            result.extend(tool for tool in page["tools"] if isinstance(tool, dict))
            cursor = page.get("nextCursor")
            if not cursor:
                return result

    def call_tool(self, name: str, arguments: dict[str, Any], timeout: float | None = None) -> Any:
        if not self.initialized:
            raise MCPError("Initialize MCP before calling tools")
        result = self._rpc("tools/call", {"name": name, "arguments": arguments}, timeout)
        if not isinstance(result, dict):
            raise MCPError(f"MCP tool {name} returned an invalid result")
        if result.get("isError"):
            content = result.get("content", [])
            blocks = content if isinstance(content, list) else []
            detail = "\n".join(
                str(block.get("text", ""))
                for block in blocks
                if isinstance(block, dict) and block.get("type") == "text"
            )
            raise MCPToolError(f"WanGP tool {name} failed: {detail or 'unspecified tool error'}")
        if "structuredContent" in result:
            return result["structuredContent"]
        content = result.get("content", [])
        values = []
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text", ""))
                try:
                    values.append(json.loads(text))
                except json.JSONDecodeError:
                    values.append(text)
        return values[0] if len(values) == 1 else values


def queue_counts(queue: dict[str, Any]) -> dict[str, int]:
    sources = [queue] + [
        queue[key] for key in ("metadata", "snapshot", "summary")
        if isinstance(queue.get(key), dict)
    ]

    def get_int(key: str) -> int | None:
        for source in sources:
            value = source.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return max(0, value)
        return None

    queued = get_int("queued_count") or 0
    running = get_int("running_count") or 0
    total = get_int("total_count")
    if total is None:
        total = get_int("count")
    if total is None:
        items = queue.get("tasks", queue.get("items", []))
        total = len(items) if isinstance(items, list) else queued + running
    return {
        "total_count": max(total, queued + running),
        "queued_count": queued,
        "running_count": running,
    }


class WanGPClient:
    def __init__(self, transport: StreamableHTTPMCP):
        self.transport = transport
        self.tools: dict[str, dict[str, Any]] = {}

    def initialize(self) -> None:
        self.transport.initialize()
        self.tools = {
            tool["name"]: tool for tool in self.transport.list_tools()
            if isinstance(tool.get("name"), str)
        }
        required = {"wangp_generate", "wangp_session", "wangp_model", "wangp_toolbox"}
        missing = sorted(required - self.tools.keys())
        if missing:
            raise BenchmarkError("WanGP MCP v2 tools missing: " + ", ".join(missing))
        schema = self.tools["wangp_generate"].get("inputSchema", {})
        wait_schema = schema.get("properties", {}).get("wait", {}) if isinstance(schema, dict) else {}
        if not isinstance(wait_schema, dict) or wait_schema.get("const") is True:
            raise BenchmarkError("MCP async is disabled; start the v2 server with --mcp-async")

    def _call(self, name: str, args: dict[str, Any], timeout: float | None = None) -> Any:
        if name not in self.tools:
            raise BenchmarkError(f"WanGP MCP tool unavailable: {name}")
        return self.transport.call_tool(name, args, timeout)

    def queue_state(self, timeout: float | None = None) -> dict[str, Any]:
        value = self._call("wangp_session", {
            "action": "list_queue", "arguments": {},
        }, timeout)
        return _object(value, "queue result")

    def model_details(self, model_type: str) -> dict[str, Any]:
        details: dict[str, Any] = {"model_type": model_type}
        for action, key in (("defaults", "defaults"), ("capabilities", "schema")):
            details[key] = self._call("wangp_model", {
                "model_type": model_type, "action": action, "arguments": {},
            })
        try:
            details["accelerator_profiles"] = self._call("wangp_model", {
                "model_type": model_type, "action": "profiles", "arguments": {},
            })
        except MCPError as error:
            details["accelerator_profiles_error"] = str(error)
        return details

    def submit(self, settings: dict[str, Any], event_limit: int, timeout: float) -> dict[str, Any]:
        value = self._call("wangp_generate", {
            "settings": settings, "wait": False, "event_limit": event_limit,
        }, timeout)
        return _object(value, "generation result")

    def get_job(self, job_id: str, event_limit: int, timeout: float) -> dict[str, Any]:
        value = self._call("wangp_session", {
            "action": "get_job", "arguments": {"job_id": job_id, "event_limit": event_limit},
        }, timeout)
        return _object(value, "get_job result")

    def cancel_job(self, job_id: str, event_limit: int, timeout: float) -> dict[str, Any]:
        value = self._call("wangp_session", {
            "action": "cancel_job", "arguments": {"job_id": job_id, "event_limit": event_limit},
        }, timeout)
        return _object(value, "cancel_job result")

    def output_settings(self, media_id: str, timeout: float) -> dict[str, Any]:
        value = self._call("wangp_toolbox", {
            "action": "media_settings", "arguments": {"media_id": media_id},
        }, timeout)
        return _object(value, "media_settings result")


def _event_key(event: Any) -> str:
    try:
        text = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        text = repr(event)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


_LIVE_PROGRESS_FIELDS = frozenset({
    "kind", "status", "state", "phase", "stage", "step", "current_step",
    "total_steps", "current", "total", "frame", "frames", "percent",
    "progress", "progress_pct", "progress_percent", "elapsed", "elapsed_seconds",
})


def _live_progress_fields(value: Any) -> dict[str, Any]:
    # Keep small structured values; omit free-form event text and settings.
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for field in _LIVE_PROGRESS_FIELDS:
        item = value.get(field)
        if isinstance(item, str):
            result[field] = item[:80]
        elif isinstance(item, (int, float, bool)):
            result[field] = item
    return result


def _server_time(value: Any) -> str | None:
    try:
        return dt.datetime.fromtimestamp(float(value), dt.timezone.utc).isoformat(timespec="milliseconds")
    except (TypeError, ValueError, OverflowError, OSError):
        return None


class BenchmarkRunner:
    def __init__(
        self,
        client: WanGPClient,
        manifest: dict[str, Any],
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        cancel_grace_seconds: float = DEFAULT_CANCEL_GRACE,
        event_limit: int = MAX_EVENTS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        if not 1 <= timeout_seconds <= 86400:
            raise BenchmarkError("job timeout must be between 1 second and 24 hours")
        if not 0.1 <= poll_interval <= 30:
            raise BenchmarkError("poll interval must be between 0.1 and 30 seconds")
        if not 0 <= cancel_grace_seconds <= 120:
            raise BenchmarkError("cancel grace must be between 0 and 120 seconds")
        if not 1 <= event_limit <= MAX_EVENTS:
            raise BenchmarkError(f"event limit must be between 1 and {MAX_EVENTS}")
        self.client = client
        self.manifest = manifest if "runs" in manifest else normalize_manifest(manifest)
        self.timeout_seconds = float(timeout_seconds)
        self.poll_interval = float(poll_interval)
        self.cancel_grace = float(cancel_grace_seconds)
        self.event_limit = event_limit
        self.clock = clock
        self.sleep = sleep
        self.progress_callback = progress_callback
        self.progress_callback_warning: str | None = None

    def _queue_or_raise(self) -> dict[str, Any]:
        queue = self.client.queue_state(self.client.transport.request_timeout)
        if queue_counts(queue)["total_count"]:
            raise BusyQueueError(queue)
        return queue

    @staticmethod
    def _events(record: dict[str, Any], snapshot: dict[str, Any]) -> None:
        events = snapshot.get("events", [])
        if not isinstance(events, list):
            return
        seen = record.setdefault("_seen_events", set())
        stored = record.setdefault("progress_events", [])
        for event in events:
            key = _event_key(event)
            if key not in seen:
                seen.add(key)
                stored.append(event)

    def _emit_progress(
        self,
        record: dict[str, Any],
        snapshot: dict[str, Any],
        job_id: str,
        started: float,
    ) -> None:
        if self.progress_callback is None or self.progress_callback_warning is not None:
            return

        progress: dict[str, Any] = {}
        progress.update(_live_progress_fields(snapshot.get("progress")))
        progress.update(_live_progress_fields(snapshot))
        for event in reversed(record.get("progress_events", [])):
            if isinstance(event, dict) and event.get("kind") in {
                "started", "progress", "preview", "status", "completed", "error"
            }:
                progress.update(_live_progress_fields(event))
                progress.update(_live_progress_fields(event.get("data")))
                break

        cancellation = record.get("timeout_cancel")
        if not isinstance(cancellation, dict):
            cancel_state = {
                "status": "not_requested", "requested": False, "confirmed": False,
            }
        else:
            requested = bool(cancellation.get("requested"))
            confirmed = bool(cancellation.get("confirmed"))
            state = (
                "confirmed" if confirmed
                else "request_failed" if cancellation.get("error")
                else "monitoring_failed" if cancellation.get("monitor_error")
                else "requested" if requested
                else "not_requested"
            )
            cancel_state = {
                "status": state, "requested": requested, "confirmed": confirmed,
            }

        live_snapshot = {
            "schema_version": 1,
            "benchmark": self.manifest["name"],
            "case": record["name"],
            "repeat": record["repeat"],
            "job_id": job_id,
            "submitted_at": record.get("submitted_at"),
            "elapsed_seconds": round(max(0.0, self.clock() - started), 3),
            "done": bool(snapshot.get("done")),
            "status": record.get("status"),
            "cancel_state": cancel_state,
            "progress": progress,
        }
        try:
            self.progress_callback(live_snapshot)
        except Exception as error:
            self.progress_callback_warning = f"{type(error).__name__}: {error}"
            self.progress_callback = None

    def _capture_result(self, record: dict[str, Any], snapshot: dict[str, Any]) -> None:
        result = snapshot.get("result")
        record["job_snapshot"] = snapshot
        record["result"] = result if isinstance(result, dict) else None
        if not isinstance(result, dict):
            record["generated_files"] = []
            record["errors"] = [{"message": "Job completed without a result object"}]
            record["status"] = "failed"
            return
        record["generated_files"] = result.get("generated_files", []) if isinstance(result.get("generated_files"), list) else []
        record["errors"] = result.get("errors", []) if isinstance(result.get("errors"), list) else []
        record["gallery_items"] = result.get("gallery_items", []) if isinstance(result.get("gallery_items"), list) else []
        record["status"] = "completed" if result.get("success") else "failed"

    @staticmethod
    def _compare_settings(expected: dict[str, Any], actual: Any) -> dict[str, Any]:
        if not isinstance(actual, dict):
            return {"status": "unavailable", "matched_fields": [], "mismatches": [], "unreported_fields": sorted(expected)}
        matched = []
        mismatches = []
        unreported = []
        for key, expected_value in sorted(expected.items()):
            if key not in actual:
                unreported.append(key)
            elif actual[key] == expected_value:
                matched.append(key)
            else:
                mismatches.append({
                    "field": key, "expected": expected_value, "actual": actual[key],
                })
        status = "mismatch" if mismatches else "incomplete" if unreported else "matched"
        return {
            "status": status,
            "matched_fields": matched,
            "mismatches": mismatches,
            "unreported_fields": unreported,
        }

    def _read_output_settings(self, record: dict[str, Any]) -> None:
        outputs = []
        for item in record.get("gallery_items", []):
            if not isinstance(item, dict) or not isinstance(item.get("media_id"), str):
                continue
            try:
                settings = self.client.output_settings(
                    item["media_id"], self.client.transport.request_timeout
                )
                outputs.append({
                    "media_id": item["media_id"],
                    "filename": item.get("filename"),
                    "response": settings,
                    "settings_comparison": self._compare_settings(
                        record["requested_settings"], settings.get("settings")
                    ),
                })
            except BenchmarkError as error:
                outputs.append({
                    "media_id": item["media_id"],
                    "filename": item.get("filename"),
                    "error": str(error),
                })
        record["actual_output_settings"] = outputs

    def _run_case(self, case: dict[str, Any], queue_before: dict[str, Any]) -> dict[str, Any]:
        started = self.clock()
        record: dict[str, Any] = {
            "name": case["name"],
            "repeat": case["repeat"],
            "status": "submitting",
            "submitted_at": utc_now(),
            "requested_settings": deepcopy(case["settings"]),
            "queue_before": queue_before,
            "progress_events": [],
            "errors": [],
        }
        try:
            snapshot = self.client.submit(
                case["settings"], self.event_limit,
                min(self.client.transport.request_timeout, self.timeout_seconds),
            )
        except MCPToolError as error:
            record.update(status="submission_error", submission_error=str(error))
            record["elapsed_seconds"] = round(self.clock() - started, 3)
            return record
        except MCPError as error:
            record.update(
                status="submission_unknown",
                submission_error=str(error),
                note="Submission outcome is ambiguous; it was not retried.",
            )
            record["elapsed_seconds"] = round(self.clock() - started, 3)
            return record
        job_id = snapshot.get("job_id")
        if not isinstance(job_id, str) or not job_id.strip():
            record.update(
                status="submission_unknown",
                submission_error="No job_id returned; refusing to retry an ambiguous submission",
                initial_response=snapshot,
            )
            record["elapsed_seconds"] = round(self.clock() - started, 3)
            return record
        job_id = job_id.strip()
        record["job_id"] = job_id
        record["status"] = "finalizing" if snapshot.get("done") else "running"
        self._events(record, snapshot)
        record["job_snapshot"] = snapshot
        self._emit_progress(record, snapshot, job_id, started)
        deadline = started + self.timeout_seconds
        while not snapshot.get("done"):
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            try:
                snapshot = self.client.get_job(
                    job_id, self.event_limit,
                    max(0.1, min(self.client.transport.request_timeout, remaining)),
                )
            except MCPError as error:
                record.update(status="poll_error", poll_error=str(error), job_snapshot=snapshot)
                record["elapsed_seconds"] = round(self.clock() - started, 3)
                self._emit_progress(record, snapshot, job_id, started)
                record.pop("_seen_events", None)
                return record
            if snapshot.get("job_id") not in {None, job_id}:
                record.update(status="poll_error", poll_error="WanGP returned a different job_id", job_snapshot=snapshot)
                record["elapsed_seconds"] = round(self.clock() - started, 3)
                self._emit_progress(record, {"done": False}, job_id, started)
                record.pop("_seen_events", None)
                return record
            self._events(record, snapshot)
            record["job_snapshot"] = snapshot
            record["status"] = "finalizing" if snapshot.get("done") else "running"
            self._emit_progress(record, snapshot, job_id, started)
            if snapshot.get("done"):
                break
            remaining = deadline - self.clock()
            if remaining > 0:
                self.sleep(min(self.poll_interval, remaining))

        if not snapshot.get("done"):
            record.update(
                status="timed_out",
                timeout_seconds=self.timeout_seconds,
                timeout_cancel={"requested": False, "confirmed": False},
            )
            self._emit_progress(record, snapshot, job_id, started)
            try:
                snapshot = self.client.cancel_job(
                    job_id, self.event_limit, self.client.transport.request_timeout
                )
                record["timeout_cancel"]["requested"] = True
                self._events(record, snapshot)
                if snapshot.get("done"):
                    record["timeout_cancel"]["confirmed"] = True
                self._emit_progress(record, snapshot, job_id, started)
            except MCPError as error:
                record["timeout_cancel"]["error"] = str(error)
                self._emit_progress(record, snapshot, job_id, started)
            grace_deadline = self.clock() + self.cancel_grace
            while not snapshot.get("done") and self.clock() < grace_deadline:
                remaining = grace_deadline - self.clock()
                try:
                    snapshot = self.client.get_job(
                        job_id, self.event_limit,
                        max(0.1, min(self.client.transport.request_timeout, remaining)),
                    )
                except MCPError as error:
                    record["timeout_cancel"]["monitor_error"] = str(error)
                    break
                self._events(record, snapshot)
                if snapshot.get("done"):
                    record["timeout_cancel"]["confirmed"] = True
                self._emit_progress(record, snapshot, job_id, started)
                if snapshot.get("done"):
                    break
                if remaining > 0:
                    self.sleep(min(self.poll_interval, remaining))
            record["job_snapshot"] = snapshot
            record["elapsed_seconds"] = round(self.clock() - started, 3)
            record.pop("_seen_events", None)
            self._emit_progress(record, snapshot, job_id, started)
            return record

        self._capture_result(record, snapshot)
        record["server_created_at"] = _server_time(snapshot.get("created_at"))
        record["server_updated_at"] = _server_time(snapshot.get("updated_at"))
        record["completed_at"] = utc_now()
        record["elapsed_seconds"] = round(self.clock() - started, 3)
        record["progress"] = [
            event for event in record["progress_events"]
            if isinstance(event, dict) and event.get("kind") in {
                "started", "progress", "status", "completed", "error"
            }
        ]
        record.pop("_seen_events", None)
        self._read_output_settings(record)
        self._emit_progress(record, snapshot, job_id, started)
        return record

    def run(self, manifest_sha256: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": 1,
            "benchmark": self.manifest["name"],
            "model_type": EXPECTED_MODEL_TYPE,
            "target": {
                "mcp_origin": self.client.transport.origin,
                "mcp_endpoint": self.client.transport.endpoint,
            },
            "manifest_sha256": manifest_sha256,
            "started_at": utc_now(),
            "status": "running",
            "queue_preflight": None,
            "model_discovery": None,
            "runs": [],
            "errors": [],
        }
        try:
            self.client.initialize()
            result["queue_preflight"] = self._queue_or_raise()
            result["model_discovery"] = self.client.model_details(EXPECTED_MODEL_TYPE)
            result["model_discovery"]["baseline_settings_validation"] = validate_h3_defaults(
                self.manifest, result["model_discovery"]
            )
            for case in self.manifest["runs"]:
                try:
                    queue = self._queue_or_raise()
                except BusyQueueError as error:
                    result["status"] = "refused_busy_queue"
                    result["busy_queue"] = error.queue
                    result["errors"].append(str(error))
                    break
                run = self._run_case(case, queue)
                result["runs"].append(run)
                if run["status"] in {"timed_out", "submission_unknown", "poll_error"}:
                    result["status"] = run["status"]
                    break
                if run["status"] == "submission_error":
                    result["status"] = "failed"
                    result["errors"].append(run.get("submission_error", "submission failed"))
                    break
            else:
                result["status"] = (
                    "completed" if all(run.get("status") == "completed" for run in result["runs"])
                    else "completed_with_errors"
                )
        except BusyQueueError as error:
            result["status"] = "refused_busy_queue"
            result["queue_preflight"] = error.queue
            result["errors"].append(str(error))
        except BenchmarkError as error:
            result["status"] = "failed"
            result["errors"].append(str(error))
        except Exception as error:
            result["status"] = "failed"
            result["errors"].append(f"{type(error).__name__}: {error}")
        result["ended_at"] = utc_now()
        return result


def write_result(path: str | os.PathLike[str], result: dict[str, Any]) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise BenchmarkError(f"Could not write report {output}: {error}") from error
    return output


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="MCP origin or URL ending in /mcp")
    parser.add_argument("--manifest", required=True, help="JSON benchmark manifest")
    parser.add_argument("--output", help="Report JSON path; defaults to a timestamped filename")
    parser.add_argument("--progress-file", help="Live progress JSON path; defaults to <output>.progress.json")
    parser.add_argument("--timeout", type=_positive_float, default=DEFAULT_TIMEOUT, help="Per-job timeout, at most 86400 seconds")
    parser.add_argument("--request-timeout", type=_positive_float, default=DEFAULT_REQUEST_TIMEOUT, help="HTTP timeout from 1 to 120 seconds")
    parser.add_argument("--poll-interval", type=_positive_float, default=DEFAULT_POLL_INTERVAL, help="Poll interval from 0.1 to 30 seconds")
    parser.add_argument("--cancel-grace", type=float, default=DEFAULT_CANCEL_GRACE, help="Seconds to confirm cancellation after timeout")
    args = parser.parse_args(argv)
    try:
        manifest, digest = load_manifest(args.manifest)
        client = WanGPClient(StreamableHTTPMCP(args.base_url, request_timeout=args.request_timeout))
        output_path = Path(args.output).expanduser() if args.output else (
            Path.cwd() / f"wangp-benchmark-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        progress_path = Path(args.progress_file).expanduser() if args.progress_file else (
            output_path.with_name(output_path.name + ".progress.json")
        )
        if output_path.resolve() == progress_path.resolve():
            raise BenchmarkError("--progress-file must differ from --output")
        runner = BenchmarkRunner(
            client,
            manifest,
            timeout_seconds=args.timeout,
            poll_interval=args.poll_interval,
            cancel_grace_seconds=args.cancel_grace,
            progress_callback=lambda snapshot: write_result(progress_path, snapshot),
        )
        result = runner.run(digest)
        output = write_result(output_path, result)
        if runner.progress_callback_warning is not None:
            print(
                "Warning: live progress snapshots could not be saved; benchmark monitoring "
                f"continued without further snapshots ({runner.progress_callback_warning})",
                file=sys.stderr,
            )
    except BenchmarkError as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        return 2
    print(f"WanGP benchmark {result['status']}; report: {output}")
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
