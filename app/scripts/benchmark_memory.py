#!/usr/bin/env python3
"""Run a small, serial Maestro generation matrix and record memory telemetry.

The script intentionally uses only Python's standard library. It talks to the
local Maestro HTTP API, saves a restore snapshot before changing settings, and
keeps that snapshot if it cannot prove every submitted job has stopped.

Matrix format::

    {
      "request": {"generation_mode": "video", "model_type": "..."},
      "cases": [
        {"name": "default", "settings": {}, "request": {},
         "repeats": 2, "reload": true}
      ]
    }

Case request fields shallowly override the common request. When repeats is 2,
reload=true releases the model before the first (cold) repeat; the second repeat
keeps the resident model warm.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as _datetime
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:42012"
API_TIMEOUT_SECONDS = 120.0
CANCEL_GRACE_SECONDS = 120.0
MAX_CASES = 20
MAX_TOTAL_RUNS = 40
MAX_LOG_CAPTURE_BYTES = 4 * 1024 * 1024
MAX_LOG_EXCERPT_CHARS = 160_000
MAX_EVIDENCE_LINES = 120
ACTIVE_STATUSES = {"queued", "running"}
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def _safe_print(
    *values: Any,
    sep: str = " ",
    end: str = "\n",
    file: Any = None,
    flush: bool = False,
) -> None:
    """Print without letting a legacy Windows console encoding abort a run."""
    stream = sys.stdout if file is None else file
    text = sep.join(str(value) for value in values) + end
    encoding = getattr(stream, "encoding", None)
    if encoding:
        try:
            # Keep characters supported by the active console code page and
            # replace only unsupported ones (for example, emoji on cp1252).
            text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        except LookupError:
            text = text.encode("ascii", errors="replace").decode("ascii")
    try:
        stream.write(text)
    except UnicodeEncodeError:
        # A wrapped/custom stream may not expose its real encoding. This final
        # fallback still preserves benchmark progress instead of aborting it.
        stream.write(text.encode("ascii", errors="replace").decode("ascii"))
    if flush:
        stream.flush()


# The write endpoint permits these system-config fields. Keeping the runner's
# allowlist aligned with that API prevents accidental changes to unrelated
# settings and makes the restore snapshot exact.
MEMORY_SETTING_KEYS = {
    "vram_allocator", "ram_allocator", "smart_memory_pinning", "read_ahead",
    "perc_reserved_mem_max", "attention_head_split", "int8_kernels",
    *(
        f"{kind}_{suffix}"
        for kind in ("video", "image", "audio")
        for suffix in ("preload_mode", "preload_in_VRAM")
    ),
}
SYSTEM_SETTING_KEYS = MEMORY_SETTING_KEYS | {
    "attention_mode", "transformer_quantization", "vae_config", "compile",
    "video_profile", "image_profile", "audio_profile",
    "video_output_codec", "image_output_codec", "enhancer_enabled",
    "prompt_enhancer_quantization", "vram_safety_coefficient",
    "generation_preview",
}
REQUEST_REPORT_KEYS = {
    "model_type", "generation_mode", "video_length", "num_inference_steps",
    "resolution", "width", "height", "image_mode", "override_profile",
    "override_attention", "seed", "guidance_scale", "activated_loras",
    "loras_multipliers", "minimax_h3_reference_detail",
    "sliding_window_size", "sliding_window_overlap",
    "sliding_window_discard_last_frames", "minimax_h3_multi_window",
    "sliding_window_memory_override", "minimax_h3_reference_sequence",
    "minimax_h3_text_encoder",
    "settings_version", "workspace",
}

OUTPUT_VALIDATION_PARAM_KEYS = (
    "video_length", "num_inference_steps", "resolution", "width", "height",
    "sliding_window_size", "sliding_window_overlap",
    "sliding_window_discard_last_frames", "minimax_h3_multi_window",
    "sliding_window_memory_override", "minimax_h3_reference_sequence",
    "minimax_h3_text_encoder",
    "settings_version",
)
OUTPUT_MEDIA_INFO_KEYS = (
    "frames", "width", "height", "fps", "duration_seconds", "format", "codec",
)

EVIDENCE_PATTERNS = {
    "h3_perf": re.compile(r"\bH3\s+Perf\b", re.IGNORECASE),
    "attention": re.compile(
        r"attention|sage(?:attention)?|\bSDPA\b|\bSLA\b|head.?split",
        re.IGNORECASE,
    ),
    "int8": re.compile(r"\bINT8\b|int8_kernels|\bTriton\b|\bKitchen\b", re.IGNORECASE),
    "mmgp_budget": re.compile(r"\bMMGP\b|residency|\bbudget\b", re.IGNORECASE),
    "preload": re.compile(r"preload|pre-load|prefetch", re.IGNORECASE),
    "shuttle": re.compile(r"shuttl|weight transfer|offload transfer", re.IGNORECASE),
    "pinning": re.compile(r"pinning|pinned memory", re.IGNORECASE),
}


class BenchmarkError(Exception):
    """A validation or execution error that is safe to show to the operator."""


class ApiError(BenchmarkError):
    def __init__(self, method: str, url: str, message: str, status_code: int | None = None):
        self.method = method
        self.url = url
        self.status_code = status_code
        super().__init__(f"{method} {url} failed" + (f" ({status_code})" if status_code else "") + f": {message}")


class BusyServerError(BenchmarkError):
    pass


class ForeignJobError(BenchmarkError):
    pass


class ApiClient:
    def __init__(self, base_url: str, timeout_seconds: float = API_TIMEOUT_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def request_json(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        url = self.base_url + path
        payload = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        request = Request(url, data=payload, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except HTTPError as error:
            detail = error.read(4096).decode("utf-8", errors="replace").strip()
            raise ApiError(method, path, detail or error.reason, error.code) from error
        except (URLError, TimeoutError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise ApiError(method, path, str(reason)) from error
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ApiError(method, path, "response was not valid JSON") from error

    def get(self, path: str) -> Any:
        return self.request_json("GET", path)

    def post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        return self.request_json("POST", path, body)

    def put(self, path: str, body: dict[str, Any]) -> Any:
        return self.request_json("PUT", path, body)


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="milliseconds")


def validate_base_url(value: str) -> str:
    """Accept only HTTP(S) API roots addressed through a loopback host."""
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port  # Accessing .port also validates its numeric range.
        hostname = (parsed.hostname or "").lower().rstrip(".")
    except (AttributeError, ValueError) as error:
        raise BenchmarkError(f"Invalid base URL: {value!r}") from error
    if parsed.scheme not in {"http", "https"}:
        raise BenchmarkError("Base URL must use http or https")
    if not hostname or parsed.username is not None or parsed.password is not None:
        raise BenchmarkError("Base URL must contain a host and no credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise BenchmarkError("Base URL must be the API origin without a path, query, or fragment")
    is_loopback = hostname == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        raise BenchmarkError("Base URL must point to localhost or a loopback IP address")
    netloc = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None:
        netloc += f":{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, "", "", ""))


def load_matrix(value: str) -> dict[str, Any]:
    try:
        if value.lstrip().startswith(("{", "[")):
            raw = value
        else:
            candidate = Path(value).expanduser()
            raw = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
        matrix = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"Could not read matrix JSON: {error}") from error
    if not isinstance(matrix, dict):
        raise BenchmarkError("Matrix must be a JSON object")
    request = matrix.get("request")
    cases = matrix.get("cases")
    if not isinstance(request, dict):
        raise BenchmarkError("Matrix field 'request' must be an object")
    if not isinstance(cases, list) or not cases:
        raise BenchmarkError("Matrix field 'cases' must be a non-empty array")
    if len(cases) > MAX_CASES:
        raise BenchmarkError(f"Matrix has too many cases (maximum {MAX_CASES})")
    names: set[str] = set()
    total_runs = 0
    normalized_cases = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise BenchmarkError(f"Case {index + 1} must be an object")
        name = case.get("name")
        if not isinstance(name, str) or not name.strip():
            raise BenchmarkError(f"Case {index + 1} needs a non-empty name")
        normalized_name = name.strip()
        if normalized_name in names:
            raise BenchmarkError(f"Case names must be unique after trimming whitespace: {name!r}")
        names.add(normalized_name)
        settings = case.get("settings", {})
        overrides = case.get("request", {})
        repeats = case.get("repeats", 1)
        reload_model = case.get("reload", False)
        if not isinstance(settings, dict):
            raise BenchmarkError(f"Case {name!r} settings must be an object")
        invalid_settings = sorted(set(settings) - SYSTEM_SETTING_KEYS)
        if invalid_settings:
            raise BenchmarkError(
                f"Case {name!r} has unsupported system setting(s): {', '.join(invalid_settings)}"
            )
        if not isinstance(overrides, dict):
            raise BenchmarkError(f"Case {name!r} request overrides must be an object")
        if type(repeats) is not int or repeats not in (1, 2):
            raise BenchmarkError(f"Case {name!r} repeats must be 1 or 2")
        if type(reload_model) is not bool:
            raise BenchmarkError(f"Case {name!r} reload must be true or false")
        total_runs += repeats
        normalized_cases.append({
            "name": normalized_name,
            "settings": dict(settings),
            "request": dict(overrides),
            "repeats": repeats,
            "reload": reload_model,
        })
    if total_runs > MAX_TOTAL_RUNS:
        raise BenchmarkError(f"Matrix has too many full generations (maximum {MAX_TOTAL_RUNS})")
    return {"request": dict(request), "cases": normalized_cases}


def validate_runtime_limits(timeout_seconds: float, poll_seconds: float) -> None:
    if not (1 <= timeout_seconds <= 86_400):
        raise BenchmarkError("--timeout-seconds must be between 1 and 86400")
    if not (0.2 <= poll_seconds <= 60):
        raise BenchmarkError("--poll-seconds must be between 0.2 and 60")


def resolve_output_dir(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise BenchmarkError(f"Output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise BenchmarkError(f"Output directory must be empty and unique: {path}")
    return path


def _write_json_exclusive(path: Path, data: dict[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise BenchmarkError(f"Refusing to overwrite existing backup: {path}") from error


def _atomic_json_write(path: Path, data: dict[str, Any]) -> None:
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


class ResultStore:
    CSV_FIELDS = (
        "case_name", "repeat", "reload_before_run", "job_id", "status",
        "job_terminal_status", "timed_out", "oom", "wall_time_seconds",
        "denoise_time_seconds", "other_phase_seconds", "peak_physical_vram_gb",
        "peak_ram_used_gb", "settings_used", "auto_performance_during_run",
        "request_parameters", "output_files", "error", "phase_durations_seconds",
        "log_evidence", "benchmark_eligible", "output_validation",
    )

    def __init__(self, output_dir: Path, data: dict[str, Any]):
        self.output_dir = output_dir
        self.data = data
        self.json_path = output_dir / "results.json"
        self.csv_path = output_dir / "results.csv"
        self._csv_handle = None
        self._csv_writer = None

    def start(self) -> None:
        self._csv_handle = self.csv_path.open("x", encoding="utf-8", newline="")
        self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=self.CSV_FIELDS)
        self._csv_writer.writeheader()
        self._csv_handle.flush()
        self.save()

    def save(self) -> None:
        _atomic_json_write(self.json_path, self.data)

    def append_run(self, run: dict[str, Any]) -> None:
        self.data["runs"].append(run)
        row = {
            key: run.get(key, "") for key in self.CSV_FIELDS
        }
        for key in (
            "settings_used", "request_parameters", "output_files",
            "phase_durations_seconds", "log_evidence", "output_validation",
        ):
            if row[key] != "":
                row[key] = json.dumps(row[key], ensure_ascii=False, separators=(",", ":"))
        assert self._csv_writer is not None and self._csv_handle is not None
        self._csv_writer.writerow(row)
        self._csv_handle.flush()
        self.save()

    def close(self) -> None:
        if self._csv_handle is not None:
            self._csv_handle.close()
            self._csv_handle = None


class LogCursor:
    """Capture a per-run log delta from either an append log or a terminal snapshot.

    Pinokio's ``latest`` log can be rewritten from a rendered terminal buffer.
    The cursor keeps the complete initial snapshot and suppresses its lines if
    the file is rewritten or its scrollback slides. Ordinary append logs retain
    byte-accurate capture, including a line that repeats earlier text.
    """

    _ANSI_ESCAPE = re.compile(
        r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))"
    )

    def __init__(self, path: Path | None):
        self.path = path
        self.offset = 0
        self.identity: tuple[int, int] | None = None
        self.last_bytes = b""
        self.baseline_lines: list[str] = []
        self.baseline_flat = ""
        self.previous_snapshot_lines: list[str] = []
        self.seen_snapshot_lines: set[str] = set()
        self.captured_lines: list[str] = []
        self.append_pending = ""
        self.discard_initial_partial = False
        self.byte_count = 0
        self.truncated = False
        self.capture_limited = False
        self.rewritten_or_sliding_snapshot = False
        self.snapshot_mode = False
        self.scope_ambiguous = False
        self.ambiguity_reasons: list[str] = []
        self.capture_mode = "unavailable" if path is None else "append"
        self.capture_modes_seen: list[str] = []
        if path is not None:
            try:
                raw = path.read_bytes()
                info = path.stat()
                self.last_bytes = raw
                self.offset = len(raw)
                self.identity = (info.st_dev, info.st_ino)
                self.baseline_lines = self._normalized_lines(raw)
                self.baseline_flat = " ".join(self.baseline_lines)
                self.previous_snapshot_lines = list(self.baseline_lines)
                self.seen_snapshot_lines = set(self.baseline_lines)
                self.discard_initial_partial = bool(raw and not raw.endswith((b"\n", b"\r")))
                self.capture_modes_seen.append("baseline")
            except FileNotFoundError:
                # A log created later has a known-empty baseline.
                pass
            except OSError:
                self._mark_ambiguous("could_not_read_start_of_run_log_snapshot")

    @classmethod
    def _normalize_line(cls, line: str) -> str:
        line = cls._ANSI_ESCAPE.sub("", line)
        line = line.replace("\x00", "").replace("\t", " ")
        line = "".join(char for char in line if char >= " " or char == "\x1b")
        return " ".join(line.split())

    @classmethod
    def _normalized_lines(cls, raw: bytes) -> list[str]:
        decoded = raw.decode("utf-8", errors="replace")
        decoded = decoded.replace("\r\n", "\n").replace("\r", "\n")
        return [line for raw_line in decoded.split("\n") if (line := cls._normalize_line(raw_line))]

    def _mark_ambiguous(self, reason: str) -> None:
        self.scope_ambiguous = True
        if reason not in self.ambiguity_reasons:
            self.ambiguity_reasons.append(reason)

    def _record_line(self, line: str) -> None:
        if not line:
            return
        encoded_size = len(line.encode("utf-8", errors="replace")) + 1
        if self.byte_count + encoded_size > MAX_LOG_CAPTURE_BYTES:
            self.capture_limited = True
            return
        self.captured_lines.append(line)
        self.byte_count += encoded_size

    @staticmethod
    def _contains_control_rewrite(data: bytes) -> bool:
        """CR progress updates/ANSI cursor motion indicate a rendered buffer."""
        without_crlf = data.replace(b"\r\n", b"")
        return b"\r" in without_crlf or b"\x1b[" in data

    @staticmethod
    def _snapshot_has_exact_overlap(previous: list[str], current: list[str]) -> bool:
        if not previous or not current:
            return not previous
        if len(current) >= len(previous) and current[:len(previous)] == previous:
            return True
        maximum = min(len(previous), len(current))
        for size in range(maximum, 0, -1):
            if previous[-size:] == current[:size]:
                return True
        return False

    def _capture_append(self, data: bytes) -> None:
        decoded = data.decode("utf-8", errors="replace")
        decoded = decoded.replace("\r\n", "\n")
        self.append_pending += decoded
        pieces = self.append_pending.split("\n")
        self.append_pending = pieces.pop() if pieces else ""
        for raw_line in pieces:
            line = self._normalize_line(raw_line)
            if self.discard_initial_partial:
                # The old terminal buffer ended in this same unterminated
                # line. Its continuation cannot be separated from old text.
                self.discard_initial_partial = False
                if line:
                    self._mark_ambiguous("append_completed_a_pre_run_partial_line")
                continue
            self._record_line(line)

    def _capture_snapshot(self, raw: bytes) -> None:
        current_lines = self._normalized_lines(raw)
        previous_lines = self.previous_snapshot_lines
        previous_counts = Counter(previous_lines)
        current_counts = Counter(current_lines)
        self.rewritten_or_sliding_snapshot = True
        if not self._snapshot_has_exact_overlap(previous_lines, current_lines):
            self._mark_ambiguous("rewritten_snapshot_has_no_exact_line_overlap")
        if raw and not raw.endswith((b"\n", b"\r")):
            self._mark_ambiguous("snapshot_ends_with_an_unterminated_line")
        recordable_lines = current_lines
        if raw and not raw.endswith((b"\n", b"\r")) and recordable_lines:
            # The final screen row may be mid-write. Keep it in the alignment
            # history, but do not turn a partial row into evidence.
            recordable_lines = recordable_lines[:-1]
        current_set = set(current_lines)
        for line in recordable_lines:
            was_seen = line in self.seen_snapshot_lines
            matches_baseline = line in self.baseline_lines
            is_evidence = any(pattern.search(line) for pattern in EVIDENCE_PATTERNS.values())
            reflowed_baseline_fragment = (
                line in self.baseline_flat and (len(line) >= 12 or is_evidence)
            )
            if matches_baseline or reflowed_baseline_fragment:
                if (
                    is_evidence and current_counts[line] > previous_counts[line]
                ):
                    self._mark_ambiguous("pre_run_evidence_text_reappeared_in_snapshot")
                continue
            if was_seen:
                continue
            self._record_line(line)
        self.seen_snapshot_lines.update(current_set)
        self.previous_snapshot_lines = current_lines
        self.append_pending = ""

    def read_new(self) -> None:
        if self.path is None:
            return
        try:
            info = self.path.stat()
            identity = (info.st_dev, info.st_ino)
            current = self.path.read_bytes()
            identity_changed = identity != self.identity
            size_shrank = len(current) < len(self.last_bytes)
            if (self.identity is not None and identity_changed) or size_shrank:
                self.truncated = True
            append = (
                not self.snapshot_mode
                and not identity_changed
                and not size_shrank
                and current.startswith(self.last_bytes)
            )
            if append:
                delta = current[len(self.last_bytes):]
                repeated_snapshot = bool(
                    len(self.baseline_lines) >= 2
                    and self.last_bytes
                    and delta.startswith(self.last_bytes)
                )
                if self._contains_control_rewrite(delta) or repeated_snapshot:
                    append = False
            if append:
                if current != self.last_bytes:
                    self.capture_mode = "append"
                    if self.capture_mode not in self.capture_modes_seen:
                        self.capture_modes_seen.append(self.capture_mode)
                    self._capture_append(delta)
                self.identity = identity
                self.last_bytes = current
                self.offset = len(current)
                return

            if current != self.last_bytes or identity_changed:
                self.snapshot_mode = True
                self.capture_mode = "snapshot_delta"
                if self.capture_mode not in self.capture_modes_seen:
                    self.capture_modes_seen.append(self.capture_mode)
                if self.append_pending:
                    self._mark_ambiguous("rewrite_followed_an_incomplete_append_line")
                    self.append_pending = ""
                self._capture_snapshot(current)
            self.identity = identity
            self.last_bytes = current
            self.offset = len(current)
        except FileNotFoundError:
            self.truncated = True
            self._mark_ambiguous("log_file_disappeared_during_run")
            return
        except OSError:
            self._mark_ambiguous("log_read_failed_during_run")
            return

    def text(self) -> str:
        if self.append_pending:
            self._mark_ambiguous("run_ended_with_an_incomplete_log_line")
        return "\n".join(self.captured_lines)

    def metadata(self) -> dict[str, Any]:
        return {
            "file": str(self.path) if self.path else None,
            "bytes_captured": self.byte_count,
            "capture_mode": self.capture_mode,
            "capture_modes_seen": list(self.capture_modes_seen),
            "latest_file_truncated_or_rotated": self.truncated,
            "rewritten_or_sliding_snapshot": self.rewritten_or_sliding_snapshot,
            "capture_limit_reached": self.capture_limited,
            "scope_ambiguous": self.scope_ambiguous,
            "ambiguity_reasons": list(self.ambiguity_reasons),
        }


def make_log_excerpt(text: str, max_chars: int = MAX_LOG_EXCERPT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[log excerpt shortened]...\n" + text[-half:]


def parse_log_evidence(text: str, limit: int = MAX_EVIDENCE_LINES) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for category, pattern in EVIDENCE_PATTERNS.items():
            if pattern.search(stripped):
                key = (category, stripped)
                if key not in seen:
                    seen.add(key)
                    evidence.append({"category": category, "line": stripped[:2000]})
        if len(evidence) >= limit:
            break
    return evidence


def _phase_label(status: dict[str, Any]) -> str:
    phase = status.get("phase")
    if isinstance(phase, str) and phase.strip():
        return phase.strip()
    message = status.get("message")
    if isinstance(message, str) and message.strip():
        return f"message: {message.strip()}"
    return "unreported"


def _looks_like_denoising(status: dict[str, Any], phase_label: str) -> bool:
    if str(status.get("status") or "").lower() in TERMINAL_STATUSES:
        return False
    phase = str(status.get("phase") or "").strip()
    message = str(status.get("message") or "").strip()
    explicit = phase or message
    text = explicit.lower()
    if any(token in text for token in (
        "decod", "encod", "load", "sav", "prepar", "finaliz", "upscal",
        "offload", "preview", "post-process", "pre-generation",
    )):
        return False
    if any(token in text for token in ("denois", "diffusion", "sampling", "generat")):
        return True
    # A non-empty API phase/message is stronger evidence than a step counter:
    # several post-processing phases retain the final denoising step number.
    if explicit:
        return False
    try:
        return int(status.get("total_steps") or 0) > 0 and int(status.get("step") or 0) > 0
    except (TypeError, ValueError, OverflowError):
        return False


def aggregate_timings(samples: list[dict[str, Any]], wall_time_seconds: float) -> dict[str, Any]:
    """Attribute each observed poll interval to its reported phase/status."""
    wall = max(0.0, float(wall_time_seconds))
    phase_seconds: dict[str, float] = {}
    denoise = 0.0
    ordered = sorted(samples, key=lambda sample: float(sample.get("elapsed_seconds", 0.0)))
    for index, sample in enumerate(ordered):
        try:
            start = max(0.0, float(sample.get("elapsed_seconds", 0.0)))
        except (TypeError, ValueError):
            start = 0.0
        next_start = wall
        if index + 1 < len(ordered):
            try:
                next_start = float(ordered[index + 1].get("elapsed_seconds", wall))
            except (TypeError, ValueError):
                next_start = wall
        duration = max(0.0, min(wall, next_start) - min(wall, start))
        status = sample.get("status_detail") if isinstance(sample.get("status_detail"), dict) else {}
        label = _phase_label(status)
        phase_seconds[label] = phase_seconds.get(label, 0.0) + duration
        if _looks_like_denoising(status, label):
            denoise += duration
    phase_seconds = {key: round(value, 3) for key, value in phase_seconds.items()}
    denoise = min(wall, denoise)
    return {
        "wall_time_seconds": round(wall, 3),
        "denoise_time_seconds": round(denoise, 3),
        "other_phase_seconds": round(max(0.0, wall - denoise), 3),
        "phase_durations_seconds": phase_seconds,
        "timing_method": "monotonic polling intervals attributed to the API phase/status observed at each poll",
    }


def _safe_status_fields(status: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "job_id", "status", "phase", "message", "progress", "step",
        "total_steps", "error", "oom_info", "output_files", "kind",
    )
    return {key: status[key] for key in keep if key in status}


def _request_summary(request: dict[str, Any]) -> dict[str, Any]:
    """Keep useful generation controls while omitting prompts and media paths."""
    return {key: request[key] for key in sorted(REQUEST_REPORT_KEYS) if key in request}


def _same_report_value(requested: Any, effective: Any) -> bool:
    """Compare JSON controls without treating booleans as numeric values."""
    if isinstance(requested, bool) or isinstance(effective, bool):
        return type(requested) is type(effective) and requested == effective
    if isinstance(requested, (int, float)) and isinstance(effective, (int, float)):
        return float(requested) == float(effective)
    return requested == effective


def _output_metadata_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    params = metadata.get("params")
    media_info = metadata.get("media_info")
    timing = metadata.get("multi_window_timing")
    if isinstance(timing, dict):
        timing_summary = {
            key: timing[key]
            for key in ("window_count", "completed_windows", "scene_duration_seconds")
            if key in timing
        }
    elif timing is None:
        timing_summary = None
    else:
        timing_summary = {"malformed_type": type(timing).__name__}
    return {
        "params": _request_summary(params) if isinstance(params, dict) else {},
        "media_info": {
            key: media_info[key]
            for key in OUTPUT_MEDIA_INFO_KEYS
            if isinstance(media_info, dict) and key in media_info
        },
        "multi_window_timing": timing_summary,
    }


def _numeric_resolution(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", value)
    return (int(match.group(1)), int(match.group(2))) if match else None


def verify_completed_outputs(
    api: ApiClient,
    request: dict[str, Any],
    output_files: Any,
) -> dict[str, Any]:
    """Check completed media against effective generation params and metadata.

    Metadata lookup failures do not change the generation's terminal status.
    They make the measurement unverified so callers cannot treat it as a valid
    performance observation.
    """
    result: dict[str, Any] = {
        "status": "unverified",
        "benchmark_eligible": False,
        "requested_parameters": _request_summary(request),
        "outputs": [],
        "mismatches": [],
        "warnings": [],
    }
    if not isinstance(output_files, list) or not output_files:
        result["warnings"].append("Completed job returned no output files to verify.")
        return result

    workspace = request.get("workspace", "")
    if workspace is None:
        workspace = ""
    if not isinstance(workspace, str):
        result["warnings"].append(
            "The request workspace is not a string, so output metadata could not be queried."
        )
        return result

    metadata_seen = 0
    metadata_unavailable = False
    multi_window_requested = request.get("minimax_h3_multi_window") is True
    sequence_window_count: int | None = None
    sequence_final_verified = False
    for item in output_files:
        filename = item if isinstance(item, str) else None
        if not filename:
            metadata_unavailable = True
            result["warnings"].append(
                f"Output entry has an unsupported shape ({type(item).__name__}); metadata was not checked."
            )
            continue

        metadata_path = (
            f"/api/v1/outputs/{quote(filename, safe='')}/metadata"
            f"?workspace={quote(workspace, safe='')}"
        )
        try:
            metadata = api.get(metadata_path)
        except Exception as error:
            metadata_unavailable = True
            result["warnings"].append(
                f"Could not retrieve metadata for {filename!r}: {error}"
            )
            continue
        if not isinstance(metadata, dict):
            metadata_unavailable = True
            result["warnings"].append(
                f"Metadata for {filename!r} had an unexpected response shape."
            )
            continue

        metadata_seen += 1
        summary = _output_metadata_summary(metadata)
        output_record = {"filename": filename, **summary}
        result["outputs"].append(output_record)
        params = metadata.get("params")
        media_info = metadata.get("media_info")
        mismatch_start = len(result["mismatches"])
        warning_start = len(result["warnings"])
        sequence_status = "not_applicable"
        if multi_window_requested:
            timing = metadata.get("multi_window_timing")
            timing_error = None
            if not isinstance(timing, dict):
                timing_error = "missing" if timing is None else "malformed"
            else:
                window_count = timing.get("window_count")
                completed_windows = timing.get("completed_windows")
                if type(window_count) is not int or window_count < 1:
                    timing_error = "malformed"
                elif (
                    type(completed_windows) is not int
                    or completed_windows < 1
                    or completed_windows > window_count
                ):
                    timing_error = "malformed"
                else:
                    duration = timing.get("scene_duration_seconds")
                    invalid_duration = False
                    if duration is not None:
                        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
                            invalid_duration = True
                        else:
                            try:
                                invalid_duration = not math.isfinite(duration) or duration <= 0
                            except OverflowError:
                                invalid_duration = True
                    if invalid_duration:
                        timing_error = "malformed"
                    elif sequence_window_count is not None and window_count != sequence_window_count:
                        timing_error = "inconsistent"
                    else:
                        sequence_window_count = window_count
                        sequence_status = (
                            "final" if completed_windows == window_count else "intermediate"
                        )
            if timing_error is not None:
                metadata_unavailable = True
                sequence_status = "unverified"
                result["warnings"].append(
                    f"Metadata for {filename!r} has {timing_error} multi-window completion timing."
                )
        output_record["sequence_status"] = sequence_status
        if not isinstance(params, dict) and not isinstance(media_info, dict):
            metadata_unavailable = True
            result["warnings"].append(
                f"Metadata for {filename!r} contained no effective parameters or media information."
            )

        requested_dimensions = _numeric_resolution(request.get("resolution"))
        for key in OUTPUT_VALIDATION_PARAM_KEYS:
            if key not in request:
                continue
            # A symbolic setting such as auto_720p is retained in the report,
            # but has no single expected pixel size to compare with the file.
            if key == "resolution" and requested_dimensions is None:
                continue
            if key in {"video_length", "width", "height", "resolution"}:
                if isinstance(params, dict) and key in params and not _same_report_value(
                    request[key], params[key]
                ):
                    result["mismatches"].append({
                        "field": key,
                        "requested": request[key],
                        "effective": params[key],
                        "source": "metadata.params",
                        "filename": filename,
                    })
                continue
            if not isinstance(params, dict) or key not in params:
                metadata_unavailable = True
                result["warnings"].append(
                    f"Metadata for {filename!r} is missing requested workload parameter {key!r}."
                )
                continue
            if not _same_report_value(request[key], params[key]):
                result["mismatches"].append({
                    "field": key,
                    "requested": request[key],
                    "effective": params[key],
                    "source": "metadata.params",
                    "filename": filename,
                })

        expected_dimensions: dict[str, Any] = {}
        if requested_dimensions is not None:
            expected_dimensions.update(width=requested_dimensions[0], height=requested_dimensions[1])
        for key in ("width", "height"):
            if key in request:
                expected_dimensions[key] = request[key]
        for key, expected in expected_dimensions.items():
            actual = media_info.get(key) if isinstance(media_info, dict) else None
            if actual is None:
                metadata_unavailable = True
                result["warnings"].append(
                    f"Metadata for {filename!r} has no actual {key}; requested geometry cannot be verified."
                )
            elif not _same_report_value(expected, actual):
                result["mismatches"].append({
                    "field": "resolution",
                    "dimension": key,
                    "requested": expected,
                    "effective": actual,
                    "source": "media_info",
                    "filename": filename,
                })

        requested_frames = request.get("video_length")
        output_frames = media_info.get("frames") if isinstance(media_info, dict) else None
        effective_frames = params.get("video_length") if isinstance(params, dict) else None
        if requested_frames is not None:
            if output_frames is None:
                metadata_unavailable = True
                result["warnings"].append(
                    f"Metadata for {filename!r} has no actual frame count; requested video length cannot be verified."
                )
            elif (
                sequence_status != "intermediate"
                and not _same_report_value(requested_frames, output_frames)
            ):
                result["mismatches"].append({
                    "field": "video_length",
                    "requested": requested_frames,
                    "effective": output_frames,
                    "source": "media_info.frames",
                    "filename": filename,
                    "frame_delta": (
                        output_frames - requested_frames
                        if isinstance(output_frames, (int, float))
                        and isinstance(requested_frames, (int, float))
                        else None
                    ),
                })
        if (
            output_frames is not None and effective_frames is not None
            and sequence_status != "intermediate"
            and not _same_report_value(effective_frames, output_frames)
        ):
            result["mismatches"].append({
                "field": "video_length",
                "requested": effective_frames,
                "effective": output_frames,
                "source": "metadata.params_vs_media_info",
                "filename": filename,
            })

        local_mismatches = result["mismatches"][mismatch_start:]
        local_warnings = result["warnings"][warning_start:]
        if local_mismatches:
            output_record["verification_status"] = "workload_mismatch"
        elif local_warnings or sequence_status == "unverified":
            output_record["verification_status"] = "unverified"
        elif sequence_status == "intermediate":
            output_record["verification_status"] = "intermediate"
        else:
            output_record["verification_status"] = "verified"
        if (
            multi_window_requested
            and sequence_status == "final"
            and output_record["verification_status"] == "verified"
        ):
            sequence_final_verified = True

    if multi_window_requested and not sequence_final_verified:
        metadata_unavailable = True
        result["warnings"].append(
            "No verified final output completed all saved multi-window sequence windows."
        )

    if result["mismatches"]:
        result["status"] = "workload_mismatch"
    elif metadata_unavailable or metadata_seen == 0:
        result["status"] = "unverified"
    else:
        result["status"] = "verified"
        result["benchmark_eligible"] = True
    return result


def _active_jobs(jobs_response: Any) -> list[dict[str, Any]]:
    if isinstance(jobs_response, dict):
        jobs = jobs_response.get("jobs", jobs_response.get("active", []))
    else:
        jobs = jobs_response
    if not isinstance(jobs, list):
        raise BenchmarkError("GET /api/v1/jobs returned an unexpected response")
    return [
        job for job in jobs
        if isinstance(job, dict) and str(job.get("status", "")).lower() in ACTIVE_STATUSES
    ]


def _job_id(job: dict[str, Any]) -> str | None:
    value = job.get("job_id", job.get("id"))
    return str(value) if value is not None else None


def assert_idle(api: ApiClient) -> list[dict[str, Any]]:
    active = _active_jobs(api.get("/api/v1/jobs"))
    if active:
        rendered = ", ".join(
            f"{_job_id(job) or '?'}:{job.get('status')}" for job in active
        )
        raise BusyServerError(f"Maestro has queued or running job(s): {rendered}")
    return active


def _sample_stats(api: ApiClient) -> dict[str, Any]:
    response = api.get("/api/v1/system-stats")
    if not isinstance(response, dict):
        raise BenchmarkError("GET /api/v1/system-stats returned an unexpected response")
    return response


def _sample_entry(
    api: ApiClient,
    status: dict[str, Any],
    started_monotonic: float,
    sample_kind: str = "poll",
) -> dict[str, Any]:
    return {
        "elapsed_seconds": round(max(0.0, time.monotonic() - started_monotonic), 3),
        "sampled_at_utc": utc_now(),
        "sample_kind": sample_kind,
        "status_detail": _safe_status_fields(status),
        "system_stats": _sample_stats(api),
    }


def _transition_view(status: dict[str, Any], elapsed: float) -> dict[str, Any]:
    return {
        "elapsed_seconds": round(elapsed, 3),
        "status": status.get("status"),
        "phase": status.get("phase") or "",
        "message": status.get("message") or "",
        "progress": status.get("progress"),
        "step": status.get("step"),
        "total_steps": status.get("total_steps"),
    }


def _is_oom(status: dict[str, Any]) -> bool:
    if status.get("oom_info"):
        return True
    error = str(status.get("error") or "").lower()
    message = str(status.get("message") or "").lower()
    return any(marker in error or marker in message for marker in (
        "out of memory", "cuda oom", "cuda out of memory", "memory allocation failed",
    ))


def _peak_metric(samples: Iterable[dict[str, Any]], section: str, key: str) -> float | None:
    values = []
    for sample in samples:
        stats = sample.get("system_stats", {})
        metrics = stats.get(section, {}) if isinstance(stats, dict) else {}
        if section == "gpu" and metrics.get("available") is not True:
            continue
        value = metrics.get(key) if isinstance(metrics, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return round(max(values), 3) if values else None


def _matrix_log_path(value: str | None, repo_root: Path) -> Path | None:
    if value:
        return Path(value).expanduser().resolve()
    for relative in (
        "logs/api/start.js/latest",
        "logs/api/start_sol.js/latest",
        "logs/api/start_classic.js/latest",
    ):
        candidate = (repo_root / relative).resolve()
        if candidate.is_file():
            return candidate
    return None


class BenchmarkRunner:
    def __init__(
        self,
        api: ApiClient,
        matrix: dict[str, Any],
        output_dir: Path,
        timeout_seconds: float,
        poll_seconds: float,
        log_path: Path | None,
        *,
        progress: bool = True,
    ):
        self.api = api
        self.matrix = matrix
        self.output_dir = output_dir
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self.log_path = log_path
        self.progress = progress
        self.backup_path = output_dir / "settings-backup.json"
        self.backup: dict[str, Any] | None = None
        self.store: ResultStore | None = None
        self.active_own_job_id: str | None = None
        self.defer_restore = False
        self.current_run: dict[str, Any] | None = None
        self._last_progress_at = 0.0

    def _say(self, message: str) -> None:
        if self.progress:
            _safe_print(message, flush=True)

    def _record_status(self, status: str, message: str | None = None) -> None:
        if self.store is None:
            return
        self.store.data["status"] = status
        if message:
            self.store.data["message"] = message
        self.store.data["updated_at"] = utc_now()
        self.store.save()

    def _snapshot_settings(self) -> dict[str, Any]:
        changed_keys = sorted({
            key
            for case in self.matrix["cases"]
            for key in case["settings"]
        })
        system = self.api.get("/api/v1/system-config")
        services = self.api.get("/api/v1/services-config")
        if not isinstance(system, dict) or not isinstance(services, dict):
            raise BenchmarkError("Maestro returned an unexpected settings response")
        missing = [key for key in changed_keys if key not in system]
        if missing:
            raise BenchmarkError(
                "Cannot snapshot setting(s) missing from /api/v1/system-config: "
                + ", ".join(missing)
            )
        auto_performance = services.get("auto_performance")
        if type(auto_performance) is not bool:
            raise BenchmarkError("services-config did not return a boolean auto_performance value")
        for allocator_key in ("vram_allocator", "ram_allocator"):
            allocator_choices = {
                case["settings"][allocator_key]
                for case in self.matrix["cases"]
                if allocator_key in case["settings"]
            }
            if allocator_choices and any(value != system.get(allocator_key) for value in allocator_choices):
                raise BenchmarkError(
                    f"This runner cannot switch {allocator_key} in a live process; "
                    "select the installed allocator before starting Maestro."
                )
        backup = {
            "schema_version": 1,
            "base_url": self.api.base_url,
            "created_at": utc_now(),
            "system_config": {key: system[key] for key in changed_keys},
            "services": {"auto_performance": auto_performance},
        }
        _write_json_exclusive(self.backup_path, backup)
        self.backup = backup
        return backup

    def _put_if_changed(self, path: str, desired: dict[str, Any], get_path: str) -> None:
        if not desired:
            return
        assert_idle(self.api)
        current = self.api.get(get_path)
        if not isinstance(current, dict):
            raise BenchmarkError(f"GET {get_path} returned an unexpected response")
        delta = {key: value for key, value in desired.items() if current.get(key) != value}
        if delta:
            assert_idle(self.api)
            self.api.put(path, delta)

    def _apply_case_settings(self, case: dict[str, Any]) -> None:
        assert self.backup is not None
        original = self.backup["system_config"]
        desired = dict(original)
        desired.update(case["settings"])
        self._put_if_changed(
            "/api/v1/system-config", desired, "/api/v1/system-config"
        )

    def _prepare_output(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=False)
        elif any(self.output_dir.iterdir()):
            raise BenchmarkError(f"Output directory must be empty and unique: {self.output_dir}")

    def _new_results(self) -> ResultStore:
        data = {
            "schema_version": 1,
            "status": "preparing",
            "base_url": self.api.base_url,
            "started_at": utc_now(),
            "timeout_seconds_per_run": self.timeout_seconds,
            "poll_seconds": self.poll_seconds,
            "measurement": {
                "polling_resolution_seconds": self.poll_seconds,
                "vram_metric": "NVML physical GPU 0 device-wide vram_used_gb from /api/v1/system-stats; includes all processes",
                "ram_metric": "system ram.used_gb from /api/v1/system-stats",
                "torch_allocator_stats_used": False,
                "notes": [
                    "Peak memory is the highest observed telemetry sample; short spikes between polls may be missed.",
                    "Phase durations attribute intervals to the API phase/status most recently observed at the configured poll cadence.",
                    "Only runs with output_validation.status=verified and benchmark_eligible=true are valid performance observations.",
                ],
            },
            "log_file": str(self.log_path) if self.log_path else None,
            "settings_backup": self.backup_path.name,
            "case_plan": [
                {
                    "name": case["name"],
                    "repeats": case["repeats"],
                    "reload": case["reload"],
                    "settings": case["settings"],
                    "request_parameters": _request_summary(
                        {**self.matrix["request"], **case["request"]}
                    ),
                }
                for case in self.matrix["cases"]
            ],
            "restoration": {"status": "pending"},
            "runs": [],
        }
        store = ResultStore(self.output_dir, data)
        store.start()
        self.store = store
        return store

    def _check_foreign_active_jobs(self, own_job_id: str) -> list[dict[str, Any]]:
        active = _active_jobs(self.api.get("/api/v1/jobs"))
        foreign = [job for job in active if _job_id(job) != own_job_id]
        if foreign:
            rendered = ", ".join(
                f"{_job_id(job) or '?'}:{job.get('status')}" for job in foreign
            )
            raise ForeignJobError(f"Detected foreign queued/running job(s): {rendered}")
        return active

    def _append_status_sample(
        self,
        status: dict[str, Any],
        started_monotonic: float,
        samples: list[dict[str, Any]],
        transitions: list[dict[str, Any]],
        cursor: LogCursor,
        sample_kind: str = "poll",
    ) -> None:
        sample = _sample_entry(self.api, status, started_monotonic, sample_kind)
        samples.append(sample)
        transition = _transition_view(
            sample["status_detail"], sample["elapsed_seconds"]
        )
        if not transitions or any(
            transitions[-1].get(key) != transition.get(key)
            for key in ("status", "phase", "message", "progress", "step", "total_steps")
        ):
            transitions.append(transition)
        cursor.read_new()
        now = time.monotonic()
        if now - self._last_progress_at >= 15:
            self._last_progress_at = now
            self._say(
                f"PROGRESS {self.current_run['case_name']}#{self.current_run['repeat']}: "
                f"{status.get('status')} phase={status.get('phase') or status.get('message') or '…'} "
                f"progress={status.get('progress', 0)}% elapsed={sample['elapsed_seconds']:.0f}s"
            )

    def _fetch_status(self, job_id: str) -> dict[str, Any]:
        path = f"/api/v1/status/{quote(job_id, safe='')}"
        status = self.api.get(path)
        if not isinstance(status, dict):
            raise BenchmarkError(f"GET {path} returned an unexpected response")
        return status

    def _cancel_owned_job(
        self,
        job_id: str,
        reason: str,
        *,
        started_monotonic: float | None = None,
        samples: list[dict[str, Any]] | None = None,
        transitions: list[dict[str, Any]] | None = None,
        cursor: LogCursor | None = None,
    ) -> dict[str, Any] | None:
        """Cancel only this runner's job, then wait for a terminal API status."""
        status_path = f"/api/v1/status/{quote(job_id, safe='')}"
        cancel_path = f"/api/v1/cancel/{quote(job_id, safe='')}"

        def capture(status: Any) -> None:
            if not (
                isinstance(status, dict) and started_monotonic is not None
                and samples is not None and transitions is not None and cursor is not None
            ):
                return
            try:
                self._append_status_sample(
                    status, started_monotonic, samples, transitions, cursor,
                    sample_kind="cancel_wait",
                )
            except Exception:
                # Cancellation and terminal confirmation take priority over a
                # transient telemetry/log read failure.
                cursor.read_new()

        try:
            current = self.api.get(status_path)
        except Exception:
            current = None
        if isinstance(current, dict) and str(current.get("status", "")).lower() in TERMINAL_STATUSES:
            capture(current)
            return current
        try:
            self.api.post(cancel_path)
        except Exception as error:
            self._say(f"CANCEL {job_id}: request failed ({error}); waiting for terminal status")
        self._say(f"CANCEL {job_id}: requested ({reason}); waiting for terminal status")
        deadline = time.monotonic() + min(CANCEL_GRACE_SECONDS, max(30.0, self.timeout_seconds))
        while time.monotonic() < deadline:
            try:
                status = self.api.get(status_path)
                capture(status)
                if isinstance(status, dict) and str(status.get("status", "")).lower() in TERMINAL_STATUSES:
                    return status
            except Exception:
                pass
            time.sleep(min(self.poll_seconds, max(0.0, deadline - time.monotonic())))
        return None

    def _run_one(self, case: dict[str, Any], repeat: int, cold_reload: bool) -> dict[str, Any]:
        run: dict[str, Any] = {
            "case_name": case["name"],
            "repeat": repeat,
            "reload_before_run": cold_reload,
            "status": "starting",
            "started_at": utc_now(),
            "samples": [],
            "status_transitions": [],
            "output_files": [],
            "oom": False,
            "benchmark_eligible": False,
        }
        self.current_run = run
        cursor = LogCursor(self.log_path)
        started_monotonic = time.monotonic()
        terminal_status: dict[str, Any] | None = None
        settled = False
        self.active_own_job_id = None
        merged_request = dict(self.matrix["request"])
        merged_request.update(case["request"])
        effective_settings = dict(self.backup["system_config"]) if self.backup else {}
        effective_settings.update(case["settings"])
        run["settings_used"] = effective_settings
        run["auto_performance_during_run"] = False
        run["request_parameters"] = _request_summary(merged_request)
        try:
            assert_idle(self.api)
            if cold_reload:
                assert_idle(self.api)
                self.api.post("/api/v1/system/release-model")
            # Baseline telemetry is sampled after any requested model release.
            baseline_status = {"status": "idle", "phase": "", "message": "pre-generation"}
            run["samples"].append(_sample_entry(
                self.api, baseline_status, started_monotonic, "pre_generation"
            ))
            cursor.read_new()
            assert_idle(self.api)
            self._say(f"START {case['name']}#{repeat} {'cold' if cold_reload else 'warm'}")
            response = self.api.post("/api/v1/generate", merged_request)
            if not isinstance(response, dict) or not response.get("job_id"):
                raise BenchmarkError("POST /api/v1/generate did not return a job_id")
            job_id = str(response["job_id"])
            run["job_id"] = job_id
            self.active_own_job_id = job_id
            run["submission_status"] = response.get("status")
            deadline = started_monotonic + self.timeout_seconds
            while True:
                status = self._fetch_status(job_id)
                self._append_status_sample(
                    status, started_monotonic, run["samples"],
                    run["status_transitions"], cursor,
                )
                run["job_terminal_status"] = str(status.get("status", "")).lower()
                if run["job_terminal_status"] in TERMINAL_STATUSES:
                    terminal_status = status
                    settled = True
                    break
                self._check_foreign_active_jobs(job_id)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    run["timed_out"] = True
                    run["status"] = "timed_out"
                    terminal_status = self._cancel_owned_job(
                        job_id, "per-run timeout",
                        started_monotonic=started_monotonic,
                        samples=run["samples"],
                        transitions=run["status_transitions"],
                        cursor=cursor,
                    )
                    if terminal_status is None:
                        self.defer_restore = True
                        run["restoration_deferred"] = True
                    else:
                        settled = True
                        run["job_terminal_status"] = str(terminal_status.get("status", "")).lower()
                        run["final_status"] = _safe_status_fields(terminal_status)
                    break
                time.sleep(min(self.poll_seconds, remaining))
            cursor.read_new()
            ended_monotonic = time.monotonic()
            wall_time = max(0.0, ended_monotonic - started_monotonic)
            run.update(aggregate_timings(run["samples"], wall_time))
            run["finished_at"] = utc_now()
            run["oom"] = bool(terminal_status and _is_oom(terminal_status))
            run["error"] = terminal_status.get("error") if terminal_status else None
            run["output_files"] = terminal_status.get("output_files", []) if terminal_status else []
            if terminal_status is not None and "final_status" not in run:
                run["final_status"] = _safe_status_fields(terminal_status)
            if run["status"] != "timed_out":
                run["status"] = run["job_terminal_status"] or "unknown"
            if run["job_terminal_status"] == "completed":
                output_validation = verify_completed_outputs(
                    self.api, merged_request, run["output_files"]
                )
                run["output_validation"] = output_validation
                run["benchmark_eligible"] = output_validation["benchmark_eligible"]
                if output_validation["status"] != "verified":
                    run["status"] = output_validation["status"]
            run["peak_physical_vram_gb"] = _peak_metric(
                run["samples"], "gpu", "vram_used_gb"
            )
            run["peak_ram_used_gb"] = _peak_metric(
                run["samples"], "ram", "used_gb"
            )
            log_text = cursor.text()
            run["log_excerpt"] = make_log_excerpt(log_text)
            run["log_evidence"] = parse_log_evidence(log_text)
            run["log_capture"] = cursor.metadata()
            if self.store:
                self.store.append_run(run)
            self._say(
                f"DONE {case['name']}#{repeat}: {run['status']} "
                f"wall={run['wall_time_seconds']:.1f}s "
                f"denoise={run['denoise_time_seconds']:.1f}s "
                f"peak_vram={run['peak_physical_vram_gb']} GiB"
            )
        except KeyboardInterrupt:
            if self.active_own_job_id:
                terminal_status = self._cancel_owned_job(
                    self.active_own_job_id, "operator interrupt",
                    started_monotonic=started_monotonic,
                    samples=run["samples"],
                    transitions=run["status_transitions"],
                    cursor=cursor,
                )
                if terminal_status is None:
                    self.defer_restore = True
                else:
                    settled = True
                    run["final_status"] = _safe_status_fields(terminal_status)
            raise
        except ForeignJobError as error:
            run["status"] = "aborted_foreign_job"
            run["error"] = str(error)
            if self.active_own_job_id:
                terminal_status = self._cancel_owned_job(
                    self.active_own_job_id, "foreign job detected",
                    started_monotonic=started_monotonic,
                    samples=run["samples"],
                    transitions=run["status_transitions"],
                    cursor=cursor,
                )
                if terminal_status is None:
                    self.defer_restore = True
                    run["restoration_deferred"] = True
                else:
                    settled = True
                    run["job_terminal_status"] = str(terminal_status.get("status", "")).lower()
                    run["final_status"] = _safe_status_fields(terminal_status)
            if self.store and not any(
                item.get("case_name") == run["case_name"] and item.get("repeat") == repeat
                for item in self.store.data["runs"]
            ):
                run["finished_at"] = utc_now()
                run["samples"] = run.get("samples", [])
                log_text = cursor.text()
                run["log_excerpt"] = make_log_excerpt(log_text)
                run["log_evidence"] = parse_log_evidence(log_text)
                run["log_capture"] = cursor.metadata()
                self.store.append_run(run)
            raise
        except Exception as error:
            run["status"] = "error"
            run["error"] = str(error)
            run["finished_at"] = utc_now()
            if self.active_own_job_id and not settled:
                terminal_status = self._cancel_owned_job(
                    self.active_own_job_id, "runner error",
                    started_monotonic=started_monotonic,
                    samples=run["samples"],
                    transitions=run["status_transitions"],
                    cursor=cursor,
                )
                if terminal_status is None:
                    self.defer_restore = True
                    run["restoration_deferred"] = True
                else:
                    settled = True
                    run["job_terminal_status"] = str(terminal_status.get("status", "")).lower()
                    run["final_status"] = _safe_status_fields(terminal_status)
            log_text = cursor.text()
            run["log_excerpt"] = make_log_excerpt(log_text)
            run["log_evidence"] = parse_log_evidence(log_text)
            run["log_capture"] = cursor.metadata()
            if self.store and not any(
                item.get("case_name") == run["case_name"] and item.get("repeat") == repeat
                for item in self.store.data["runs"]
            ):
                self.store.append_run(run)
            raise
        finally:
            if settled:
                self.active_own_job_id = None
            self.current_run = None
        if run.get("timed_out"):
            raise BenchmarkError(
                f"Run {case['name']}#{repeat} exceeded {self.timeout_seconds:g}s"
                + ("; cancellation did not reach terminal state, so settings restoration is deferred" if self.defer_restore else "; its job was cancelled and settled")
            )
        if run.get("status") == "aborted_foreign_job":
            raise ForeignJobError(run.get("error", "Foreign job detected"))
        return run

    def _restore(self, backup: dict[str, Any]) -> list[str]:
        errors = []
        system_settings = backup.get("system_config", {})
        services = backup.get("services", {})
        if not isinstance(system_settings, dict) or not isinstance(services, dict):
            raise BenchmarkError("Settings backup has an invalid shape")
        if system_settings:
            try:
                self._put_if_changed(
                    "/api/v1/system-config", system_settings,
                    "/api/v1/system-config",
                )
            except Exception as error:
                errors.append(f"system-config restoration failed: {error}")
        if type(services.get("auto_performance")) is bool:
            try:
                self._put_if_changed(
                    "/api/v1/services-config",
                    {"auto_performance": services["auto_performance"]},
                    "/api/v1/services-config",
                )
            except Exception as error:
                errors.append(f"auto_performance restoration failed: {error}")
        else:
            errors.append("settings backup has no boolean services.auto_performance value")
        return errors

    def execute(self) -> dict[str, Any]:
        store: ResultStore | None = None
        fatal_error: Exception | None = None
        try:
            assert_idle(self.api)
            self._prepare_output()
            self._snapshot_settings()
            store = self._new_results()
            self._record_status("running")
            # Keep the per-case memory settings authoritative for the matrix.
            # The snapshot lets the finally block restore the user's original
            # auto-tune choice even if a later operation fails.
            assert_idle(self.api)
            self.api.put("/api/v1/services-config", {"auto_performance": False})
            for case in self.matrix["cases"]:
                assert_idle(self.api)
                self._apply_case_settings(case)
                for repeat in range(1, case["repeats"] + 1):
                    assert_idle(self.api)
                    cold_reload = bool(case["reload"] and repeat == 1)
                    self._run_one(case, repeat, cold_reload)
            failed_runs = [
                run for run in store.data["runs"]
                if run.get("job_terminal_status") not in (None, "completed")
                or run.get("status") in {
                    "error", "unknown", "workload_mismatch", "unverified",
                }
            ]
            validation_counts = Counter(
                (run.get("output_validation") or {}).get("status", "not_applicable")
                for run in store.data["runs"]
            )
            if validation_counts["workload_mismatch"]:
                validation_status = "workload_mismatch"
            elif validation_counts["unverified"]:
                validation_status = "unverified"
            elif validation_counts["verified"] == len(store.data["runs"]):
                validation_status = "verified"
            else:
                validation_status = "not_applicable"
            store.data["benchmark_validation"] = {
                "status": validation_status,
                "eligible_runs": sum(
                    bool(run.get("benchmark_eligible")) for run in store.data["runs"]
                ),
                "workload_mismatch_runs": validation_counts["workload_mismatch"],
                "unverified_runs": validation_counts["unverified"],
                "not_applicable_runs": validation_counts["not_applicable"],
            }
            self._record_status("completed_with_failures" if failed_runs else "completed")
        except KeyboardInterrupt:
            fatal_error = BenchmarkError("Interrupted by operator")
            self._record_status("interrupted", str(fatal_error))
        except Exception as error:
            fatal_error = error
            self._record_status("failed", str(error))
        finally:
            if self.backup is not None and not self.defer_restore:
                try:
                    errors = self._restore(self.backup)
                except Exception as error:
                    errors = [str(error)]
                if errors:
                    self.defer_restore = True
                    if store:
                        store.data["restoration"] = {
                            "status": "restoration_failed",
                            "errors": errors,
                            "backup_file": self.backup_path.name,
                            "restore_with": "--restore-only",
                        }
                        store.data["status"] = "restoration_failed"
                        store.data["finished_at"] = utc_now()
                        store.save()
                    self._say("RESTORE FAILED: " + "; ".join(errors))
                else:
                    if store:
                        store.data["restoration"] = {
                            "status": "restored",
                            "restored_at": utc_now(),
                            "backup_file": self.backup_path.name,
                        }
                        store.data["finished_at"] = utc_now()
                        store.save()
                    self._say("RESTORE complete")
            elif self.backup is not None:
                reason = "An owned job may still be active; use --restore-only after it becomes terminal."
                if store:
                    store.data["restoration"] = {
                        "status": "restoration_failed",
                        "reason": reason,
                        "backup_file": self.backup_path.name,
                        "restore_with": "--restore-only",
                    }
                    store.data["status"] = "restoration_failed"
                    store.data["finished_at"] = utc_now()
                    store.save()
                self._say("RESTORE deferred: " + reason)
            elif store:
                store.data["restoration"] = {
                    "status": "not_needed",
                    "reason": "No settings backup or mutations were created.",
                }
                store.data["finished_at"] = utc_now()
                store.save()
            if store:
                store.close()

        if store is None:
            if fatal_error is not None:
                raise BenchmarkError(str(fatal_error)) from fatal_error
            raise BenchmarkError("Benchmark setup failed before results were initialized")
        result = store.data
        if fatal_error is not None:
            result["error"] = str(fatal_error)
        if result.get("status") == "restoration_failed":
            raise BenchmarkError(
                f"Settings restoration needs recovery; backup remains at {self.backup_path}"
            ) from fatal_error
        if fatal_error is not None:
            raise BenchmarkError(str(fatal_error)) from fatal_error
        return result


def restore_only(api: ApiClient, output_dir: Path) -> dict[str, Any]:
    backup_path = output_dir / "settings-backup.json"
    try:
        backup = json.loads(backup_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"Could not read settings backup {backup_path}: {error}") from error
    if not isinstance(backup, dict) or backup.get("schema_version") != 1:
        raise BenchmarkError("Settings backup has an unsupported schema")
    assert_idle(api)
    runner = object.__new__(BenchmarkRunner)
    runner.api = api
    errors = BenchmarkRunner._restore(runner, backup)
    status = "restored" if not errors else "restoration_failed"
    result = {"status": status, "restored_at": utc_now(), "backup_file": str(backup_path)}
    marker = output_dir / "restore-result.json"
    if not marker.exists():
        _write_json_exclusive(marker, result)
    else:
        result["marker_note"] = "Existing restore-result.json was preserved."
    if errors:
        result["errors"] = errors
        raise BenchmarkError("; ".join(errors))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help=f"Local Maestro API origin (default {DEFAULT_BASE_URL})")
    parser.add_argument("--matrix", help="Matrix JSON file path or inline JSON")
    parser.add_argument("--output-dir", required=True, help="New or empty directory for backup and results")
    parser.add_argument("--log-file", help="Maestro app log to sample by byte offset (default: Pinokio start.js latest log)")
    parser.add_argument("--timeout-seconds", type=float, default=1800, help="Per-generation timeout (default: 1800)")
    parser.add_argument("--poll-seconds", type=float, default=2, help="System/job polling cadence (default: 2)")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without API calls or file writes")
    parser.add_argument("--restore-only", action="store_true", help="Restore settings from output-dir/settings-backup.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_dir = Path(args.output_dir).expanduser().resolve()
        if args.restore_only:
            if args.dry_run:
                raise BenchmarkError("--dry-run and --restore-only cannot be combined")
            backup_path = output_dir / "settings-backup.json"
            try:
                backup = json.loads(backup_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise BenchmarkError(f"Could not read settings backup {backup_path}: {error}") from error
            saved_base_url = backup.get("base_url") if isinstance(backup, dict) else None
            chosen_base = args.base_url or saved_base_url or DEFAULT_BASE_URL
            base_url = validate_base_url(chosen_base)
            if saved_base_url and args.base_url and validate_base_url(saved_base_url) != base_url:
                raise BenchmarkError("--base-url does not match the base URL in the saved settings backup")
            result = restore_only(ApiClient(base_url), output_dir)
            _safe_print(json.dumps(result, indent=2), flush=True)
            return 0
        if not args.matrix:
            raise BenchmarkError("--matrix is required unless --restore-only is selected")
        base_url = validate_base_url(args.base_url or DEFAULT_BASE_URL)
        validate_runtime_limits(args.timeout_seconds, args.poll_seconds)
        matrix = load_matrix(args.matrix)
        if args.dry_run:
            resolve_output_dir(args.output_dir)
            _safe_print(json.dumps({
                "status": "dry_run",
                "base_url": base_url,
                "output_dir": str(output_dir),
                "total_full_generations": sum(case["repeats"] for case in matrix["cases"]),
                "cases": [
                    {"name": case["name"], "repeats": case["repeats"], "reload": case["reload"], "settings": case["settings"]}
                    for case in matrix["cases"]
                ],
                "note": "No API calls, settings changes, or output files were made.",
            }, indent=2), flush=True)
            return 0
        output_dir = resolve_output_dir(args.output_dir)
        repo_root = Path(__file__).resolve().parents[2]
        log_path = _matrix_log_path(args.log_file, repo_root)
        runner = BenchmarkRunner(
            ApiClient(base_url), matrix, output_dir,
            args.timeout_seconds, args.poll_seconds, log_path,
        )
        result = runner.execute()
        _safe_print(json.dumps({
            "status": result["status"],
            "output_dir": str(output_dir),
            "runs": len(result["runs"]),
            "benchmark_validation": result.get("benchmark_validation"),
            "restoration": result["restoration"],
        }, indent=2), flush=True)
        return 0
    except BenchmarkError as error:
        _safe_print(f"benchmark_memory: {error}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        _safe_print("benchmark_memory: interrupted", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
