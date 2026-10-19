"""Run a bounded, single-machine Maestro acceptance bench.

Every HTTP request is loopback-only. Prompt preparation is delegated to the
existing promptbench runner; video measurements are delegated to the existing
memory benchmark runner, which snapshots and restores settings.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable
import ctypes
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener
import zipfile


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
from scripts import benchmark_memory as memory_bench  # noqa: E402 - app-local imports after sys.path setup
CASES_FILE = Path(__file__).with_name("cases.json")
DEFAULT_WORKSPACE = APP / "outputs" / "Test-Bench"
MODEL_TYPE = "minimax_h3_fused_turbo"
PROMPT_WRITER = "gemma"
PROMPT_CASE_TIMEOUT = 300
PROMPT_MAX_CALLS = 12
PROMPT_MAX_ATTEMPTS = 2
NIGHTLY_LIMIT_SECONDS = 40 * 60
LOG_TAIL_LIMIT = 2 * 1024 * 1024
BUNDLE_LIMIT = 8 * 1024 * 1024
JOBS_RESPONSE_LIMIT = 8 * 1024 * 1024
GIT_PATH_OUTPUT_LIMIT = 512 * 1024
GIT_FINGERPRINT_MAX_FILES = 256
GIT_FINGERPRINT_MAX_FILE_BYTES = 2 * 1024 * 1024
GIT_FINGERPRINT_MAX_BYTES = 8 * 1024 * 1024
GIT_SOURCE_SUFFIXES = {
    ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".css", ".html",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cu", ".cuh", ".sh", ".bat",
    ".ps1", ".rs", ".go", ".java", ".cs",
}
GIT_EXCLUDED_PARTS = {
    ".git", ".venv", "venv", "env", "outputs", "output", "ckpt", "ckpts",
    "checkpoint", "checkpoints", "weights", "cache", "caches",
    "settings", "secrets", "secret", "credentials", "logs", "node_modules",
    "site-packages", "__pycache__", "dist", "build", "temp", "tmp",
}
GIT_EXCLUDED_NAMES = {".env", "settings.py", "secrets.py", "credentials.py", "tokens.py"}
SENSITIVE_SOURCE_ASSIGNMENT = re.compile(
    rb"(?im)(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|private[_-]?key)"
    rb"\s*[:=]\s*['\"][^'\"\r\n]{12,}['\"]|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)
PACKAGE_NAMES = (
    "torch", "torchvision", "torchaudio", "triton", "transformers",
    "diffusers", "accelerate", "numpy", "fastapi", "uvicorn", "safetensors",
)
SETTING_KEYS = (
    "video_profile", "video_preload_mode", "video_preload_in_VRAM",
    "int8_kernels", "read_ahead",
)
SAFE_SYSTEM_KEYS = (
    "app_version", "video_profile", "video_preload_mode", "video_preload_in_VRAM",
    "int8_kernels", "read_ahead", "smart_memory_pinning", "perc_reserved_mem_max",
    "attention_head_split", "vram_allocator", "attention_mode",
    "transformer_quantization", "vae_config", "compile",
)
SAFE_HARDWARE_KEYS = (
    "cuda_available", "gpu_name", "gpu_vram_gb", "gpu_capability",
    "driver_version", "nvidia_cuda", "ram_gb", "platform", "machine",
    "ram_tier", "vram_tier", "torch_version", "runtime_version",
    "supports_mmgp_allocator",
)
SENSITIVE_LOG_LINE = re.compile(
    r"(?im)^.*(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|"
    r"authorization|\bbearer\b|\bpassword\b|\bsecret\b).*$"
)
SENSITIVE_QUERY = re.compile(
    r"(?i)([?&](?:api[_-]?key|access_token|refresh_token|token|secret|password)=)[^&\s]+"
)
BEARER_VALUE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
LOCAL_PATH = re.compile(
    r"(?i)(?:[A-Z]:\\(?:[^\s\"'<>|]+)|\\\\[^\s\"'<>|]+|"
    r"/(?:home|Users|mnt|tmp|var/tmp)/[^\s\"'<>|]+)"
)
LANGUAGE_TAG = re.compile(r"^\s*\[[A-Za-z][A-Za-z0-9_-]{0,15}\]\s*")


class HarnessError(Exception):
    """A safe, operator-facing harness error."""


class DeferredError(HarnessError):
    """A required server, idle, asset, or reconciliation precondition is absent."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def make_run_id(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    return current.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalized_text(raw: bytes) -> bytes:
    text = raw.decode("utf-8-sig")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _digest_named_source_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name.casefold()):
        name = path.name.encode("utf-8")
        payload = _normalized_text(path.read_bytes())
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def code_digest() -> str:
    return _digest_named_source_files(Path(__file__).parent.glob("*.py"))


def _canonical_json_digest(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(canonical.encode("utf-8"))


def suite_digest(path: Path | None = None) -> str:
    return _canonical_json_digest(path or CASES_FILE)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return default


def validate_base_url(value: str) -> str:
    """Normalize an HTTP(S) URL only when its host is a loopback address."""
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
        host = (parsed.hostname or "").lower().rstrip(".")
    except (AttributeError, ValueError) as error:
        raise HarnessError("Invalid Maestro base URL") from error
    if parsed.scheme.lower() not in {"http", "https"} or not host:
        raise HarnessError("Maestro base URL must use http or https and include a loopback host")
    if parsed.username is not None or parsed.password is not None:
        raise HarnessError("Maestro base URL must not contain credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise HarnessError("Maestro base URL must be an origin without path, query, or fragment")
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise HarnessError("Maestro base URL must point to localhost or a loopback IP")
        except ValueError as error:
            raise HarnessError("Maestro base URL must point to localhost or a loopback IP") from error
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc += f":{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, "", "", ""))


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = validate_base_url(base_url)
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}))

    def get(self, path: str):
        request = Request(self.base_url + path, headers={"Accept": "application/json"}, method="GET")
        response_limit = JOBS_RESPONSE_LIMIT if path == "/api/v1/jobs" else 2 * 1024 * 1024
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read(response_limit + 1)
        except HTTPError as error:
            raise DeferredError(f"Local API GET {path} returned HTTP {error.code}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise DeferredError(f"Local Maestro API is unavailable at {self.base_url}") from error
        if len(raw) > response_limit:
            cap_mib = response_limit // (1024 * 1024)
            raise DeferredError(f"Local API response {path} exceeded its {cap_mib} MiB bounded response cap")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DeferredError(f"Local API GET {path} did not return JSON") from error
        if not isinstance(value, dict):
            raise DeferredError(f"Local API GET {path} returned an unexpected response")
        return value


def admission(api) -> dict:
    """Read only safe counts/statuses; never retain user jobs or API error text."""
    try:
        jobs = api.get("/api/v1/jobs")
        stream = api.get("/api/v1/llm/stream-status")
        downloads = api.get("/api/v1/models/downloads/status")
    except Exception as error:
        if isinstance(error, DeferredError):
            raise
        raise DeferredError("Could not establish local job and download admission state") from error
    if not isinstance(jobs.get("jobs"), list) or type(stream.get("done")) is not bool:
        raise DeferredError("Maestro returned an incomplete busy/idle response")
    active_jobs = sum(
        isinstance(job, dict) and job.get("status") in {"queued", "held", "running"}
        for job in jobs["jobs"]
    )
    active_downloads = 0
    states = downloads.get("downloads", {})
    if isinstance(states, dict):
        active_downloads = sum(
            isinstance(item, dict) and str(item.get("status", "")).lower() in {"queued", "running", "downloading"}
            for item in states.values()
        )
    reasons = []
    if active_jobs:
        reasons.append(f"{active_jobs} queued/running job(s)")
    if stream.get("done") is not True:
        reasons.append("an LLM stream is active")
    if active_downloads:
        reasons.append(f"{active_downloads} model download(s) active")
    return {
        "busy": bool(reasons), "reasons": reasons,
        "active_jobs": active_jobs, "llm_idle": stream.get("done") is True,
        "active_downloads": active_downloads,
    }


def _safe_stats(value: dict) -> dict:
    cpu, ram, gpu = value.get("cpu") or {}, value.get("ram") or {}, value.get("gpu") or {}
    return {
        "cpu": {key: cpu[key] for key in ("percent",) if key in cpu},
        "ram": {key: ram[key] for key in ("percent", "used_gb", "total_gb") if key in ram},
        "gpu": {key: gpu[key] for key in (
            "available", "name", "gpu_name", "percent", "vram_used_gb", "vram_total_gb", "vram_percent",
        ) if key in gpu},
    }


def _safe_detect(value: dict) -> dict:
    hardware = value.get("hardware") or {}
    recommended = value.get("recommended") or {}
    return {
        "hardware": {key: hardware[key] for key in SAFE_HARDWARE_KEYS if key in hardware},
        "recommended": {key: recommended[key] for key in SAFE_SYSTEM_KEYS if key in recommended},
    }


def server_snapshot(api) -> dict:
    """Collect allowlisted read-only API evidence; do not persist full config/job payloads."""
    state = admission(api)
    system = api.get("/api/v1/system-config")
    services = api.get("/api/v1/services-config")
    stats = api.get("/api/v1/system-stats")
    models = api.get("/api/v1/models")
    detect = api.get("/api/v1/system-detect")
    model_rows = models.get("models", [])
    model = next((item for item in model_rows if isinstance(item, dict) and item.get("model_type") == MODEL_TYPE), None)
    selected_model = None if model is None else {
        "model_type": MODEL_TYPE,
        "architecture": model.get("architecture"),
        "is_downloaded": bool(model.get("is_downloaded")),
        "identity_method": "Maestro model catalog readiness; checkpoint files are not hashed",
    }
    return {
        "admission": state,
        "system": {key: system[key] for key in SAFE_SYSTEM_KEYS if key in system},
        "services": {"auto_performance": services.get("auto_performance")},
        "stats": _safe_stats(stats),
        "h3_model": selected_model,
        "system_detect": _safe_detect(detect),
    }


def _package_diagnostics() -> dict:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
        except Exception:
            versions[name] = "unavailable"
    torch_cuda = "unavailable"
    try:
        import torch  # Read version metadata; no tensors or CUDA context are created.
        torch_cuda = getattr(getattr(torch, "version", None), "cuda", None)
    except Exception:
        pass
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": sys.platform,
        "torch_cuda_build": torch_cuda,
        "packages": versions,
    }


def _run_bounded(command: list[str], timeout: int = 8, max_output: int = 16_384,
                 *, strip_output: bool = True) -> dict:
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "timed_out", "timeout_seconds": timeout, "truncated": False}
    except OSError:
        return {"status": "unavailable", "truncated": False}
    raw = result.stdout or b""
    output = raw[:max_output].decode("utf-8", "replace")
    if strip_output:
        output = output.strip()
    return {"status": "available" if result.returncode == 0 else "error", "output": output,
            "truncated": len(raw) > max_output}


def _is_fingerprint_source_path(relative: str) -> bool:
    normalized = relative.replace("\\", "/")
    parts = normalized.split("/")
    if len(parts) < 2 or parts[0].casefold() != "app" or any(part in {"", ".", ".."} for part in parts):
        return False
    folded = [part.casefold() for part in parts]
    filename = folded[-1]
    if any(part in GIT_EXCLUDED_PARTS for part in folded) or filename in GIT_EXCLUDED_NAMES:
        return False
    if filename == ".env" or filename.startswith(".env.") or filename.endswith((".pem", ".key")):
        return False
    return Path(filename).suffix in GIT_SOURCE_SUFFIXES


def _dirty_source_fingerprint(root: Path, paths: Iterable[str]) -> dict:
    """Hash only bounded, allowlisted changed source contents; never store source text."""
    unique = sorted(set(paths), key=lambda item: item.casefold())
    digest = hashlib.sha256()
    hashed = bytes_hashed = excluded = skipped = 0
    partial = len(unique) > GIT_FINGERPRINT_MAX_FILES
    for relative in unique[:GIT_FINGERPRINT_MAX_FILES]:
        if not _is_fingerprint_source_path(relative):
            excluded += 1
            continue
        parts = relative.replace("\\", "/").split("/")
        app_root = (root / "app").resolve()
        try:
            path = (root.joinpath(*parts)).resolve(strict=False)
            path.relative_to(app_root)
        except (OSError, ValueError):
            excluded += 1
            continue
        if not path.is_file():
            name = relative.replace("\\", "/").encode("utf-8", "surrogateescape")
            digest.update(len(name).to_bytes(4, "big"))
            digest.update(name)
            digest.update(b"\0deleted\0")
            hashed += 1
            continue
        try:
            if path.stat().st_size > GIT_FINGERPRINT_MAX_FILE_BYTES:
                skipped += 1
                partial = True
                continue
            raw = path.read_bytes()
        except OSError:
            skipped += 1
            partial = True
            continue
        if SENSITIVE_SOURCE_ASSIGNMENT.search(raw):
            skipped += 1
            partial = True
            continue
        try:
            payload = _normalized_text(raw)
        except UnicodeDecodeError:
            skipped += 1
            partial = True
            continue
        if bytes_hashed + len(payload) > GIT_FINGERPRINT_MAX_BYTES:
            skipped += 1
            partial = True
            break
        name = relative.replace("\\", "/").encode("utf-8", "surrogateescape")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        hashed += 1
        bytes_hashed += len(payload)
    return {
        "sha256": digest.hexdigest(), "status": "partial" if partial else "complete",
        "candidate_paths": len(unique), "files_hashed": hashed, "bytes_hashed": bytes_hashed,
        "excluded_paths": excluded, "skipped_paths": skipped,
        "scope": "changed tracked and untracked allowlisted app source; normalized UTF-8 worktree text only",
    }


def _git_diagnostics() -> dict:
    head = _run_bounded(["git", "rev-parse", "HEAD"], timeout=5, max_output=256)
    tracked = _run_bounded(
        ["git", "diff", "--name-only", "-z", "HEAD", "--", "app"],
        timeout=8, max_output=GIT_PATH_OUTPUT_LIMIT, strip_output=False,
    )
    untracked = _run_bounded(
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", "app"],
        timeout=8, max_output=GIT_PATH_OUTPUT_LIMIT, strip_output=False,
    )
    head_value = head.get("output") if head.get("status") == "available" else None
    listings = (tracked, untracked)
    if any(item.get("status") != "available" for item in listings):
        fingerprint = {"sha256": None, "status": "unavailable", "candidate_paths": 0,
                      "files_hashed": 0, "bytes_hashed": 0, "excluded_paths": 0,
                      "skipped_paths": 0, "scope": "changed tracked and untracked allowlisted app source"}
        list_status = next(item.get("status", "unavailable") for item in listings if item.get("status") != "available")
    elif any(item.get("truncated") for item in listings):
        fingerprint = {"sha256": None, "status": "truncated", "candidate_paths": 0,
                      "files_hashed": 0, "bytes_hashed": 0, "excluded_paths": 0,
                      "skipped_paths": 0, "scope": "changed tracked and untracked allowlisted app source"}
        list_status = "truncated"
    else:
        paths = []
        for item in listings:
            output = item.get("output") or ""
            paths.extend(path for path in output.split("\0") if path)
        fingerprint = _dirty_source_fingerprint(ROOT, paths)
        list_status = "available"
    return {
        "head": head_value,
        "head_status": head.get("status"),
        "diff_sha256": fingerprint.get("sha256"),
        "diff_digest_scope": (
            "SHA-256 of app-relative paths and normalized UTF-8 contents of changed tracked/untracked source files "
            "(including app/models implementation source); "
            "excludes settings, outputs, checkpoint/weight/cache/env/secret paths and non-source files; "
            f"caps {GIT_FINGERPRINT_MAX_FILES} files, {GIT_FINGERPRINT_MAX_FILE_BYTES} bytes/file, "
            f"{GIT_FINGERPRINT_MAX_BYTES} bytes total"
        ),
        "diff_status": list_status if list_status != "available" else fingerprint.get("status"),
        "diff_files_hashed": fingerprint.get("files_hashed"),
        "diff_bytes_hashed": fingerprint.get("bytes_hashed"),
        "diff_excluded_paths": fingerprint.get("excluded_paths"),
        "diff_skipped_paths": fingerprint.get("skipped_paths"),
        "diff_candidate_paths": fingerprint.get("candidate_paths"),
    }


def _windows_metadata() -> dict:
    if sys.platform != "win32":
        return {"status": "not_applicable", "platform": sys.platform}
    command = (
        "$identity=[Security.Principal.WindowsIdentity]::GetCurrent(); "
        "$principal=New-Object Security.Principal.WindowsPrincipal($identity); "
        "$svc=Get-Service -Name sshd -ErrorAction SilentlyContinue; "
        "$status=if($null -eq $svc){'not_installed'}else{$svc.Status.ToString()}; "
        "[pscustomobject]@{account_name=$identity.Name;computer_name=$env:COMPUTERNAME;"
        "admin_elevated=$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator);"
        "sshd_service_status=$status}|ConvertTo-Json -Compress"
    )
    result = _run_bounded(["powershell", "-NoProfile", "-NonInteractive", "-Command", command], timeout=8, max_output=2048)
    if result.get("status") != "available":
        return {"status": result.get("status", "unavailable")}
    try:
        value = json.loads(result["output"])
    except json.JSONDecodeError:
        return {"status": "invalid_response"}
    if not isinstance(value, dict):
        return {"status": "invalid_response"}
    return {"status": "available", **{
        key: value.get(key) for key in (
            "account_name", "computer_name", "admin_elevated", "sshd_service_status",
        )
    }}


def diagnostics() -> dict:
    nvidia = _run_bounded([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader",
    ], timeout=8, max_output=2048)
    return {
        "git": _git_diagnostics(),
        "runtime": _package_diagnostics(),
        "nvidia_smi": nvidia,
        "windows": _windows_metadata(),
    }


def redact_log(value: str) -> str:
    value = SENSITIVE_LOG_LINE.sub("[redacted sensitive log line]", value)
    value = SENSITIVE_QUERY.sub(r"\1<redacted>", value)
    value = BEARER_VALUE.sub("Bearer <redacted>", value)
    return LOCAL_PATH.sub("<local path redacted>", value)


def _redact_json(value, *, parent_key: str = ""):
    """Sanitize serialized diagnostics while preserving useful run evidence."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if any(word in normalized for word in ("password", "secret", "token", "api_key", "authorization")):
                result[key] = "<redacted>"
            elif normalized in {"path", "model_path", "checkpoint_path", "source_path", "asset_path"}:
                result[key] = "<local path redacted>"
            else:
                result[key] = _redact_json(item, parent_key=normalized)
        return result
    if isinstance(value, list):
        if parent_key == "output_files":
            return [Path(str(item)).name for item in value if isinstance(item, (str, Path))]
        return [_redact_json(item, parent_key=parent_key) for item in value]
    if isinstance(value, str):
        return redact_log(value)
    return value


def tail_file(path: Path, limit: int = LOG_TAIL_LIMIT) -> bytes:
    if limit < 0:
        raise ValueError("tail limit must be non-negative")
    with path.open("rb") as source:
        source.seek(0, os.SEEK_END)
        size = source.tell()
        source.seek(max(0, size - limit), os.SEEK_SET)
        return source.read(limit)


def _log_candidates() -> dict[str, tuple[Path, ...]]:
    return {
        "start.log": (
            ROOT / "logs" / "api" / "start.js" / "latest",
            ROOT / "logs" / "api" / "start_sol.js" / "latest",
            ROOT / "logs" / "api" / "start_classic.js" / "latest",
        ),
        "llama.log": (
            APP / "logs" / "llama.log", APP / "logs" / "llama" / "latest",
            ROOT / "logs" / "llama.log", ROOT / "logs" / "llama" / "latest",
            ROOT / "logs" / "api" / "llama" / "latest",
        ),
    }


def _tail_logs() -> dict[str, bytes]:
    result = {}
    for name, candidates in _log_candidates().items():
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            continue
        try:
            raw = tail_file(path)
        except OSError:
            continue
        result[name] = redact_log(raw.decode("utf-8", "replace")).encode("utf-8")[-LOG_TAIL_LIMIT:]
    return result


def _assert_render_preconditions(snapshot: dict) -> None:
    model = snapshot.get("h3_model")
    if not model:
        raise DeferredError("The fused H3 model is not present in Maestro's model catalog")
    if model.get("is_downloaded") is not True:
        raise DeferredError("Required H3 model components are not all installed; install them in Maestro before benchmarking")
    hardware = (snapshot.get("system_detect") or {}).get("hardware") or {}
    gpu = (snapshot.get("stats") or {}).get("gpu") or {}
    cuda_available = hardware.get("cuda_available", gpu.get("available"))
    if cuda_available is not True:
        raise DeferredError("A CUDA-capable GPU is not available; render benchmarking was not submitted")
    settings = snapshot.get("system") or {}
    if "video_profile" not in settings:
        raise DeferredError("Maestro did not report the current video profile; no profile was guessed")


def _memory_settings(current: dict) -> dict:
    return {key: current[key] for key in SETTING_KEYS if key in current}


def build_matrix(snapshot: dict, *, compare: bool) -> dict:
    _assert_render_preconditions(snapshot)
    case = json.loads(CASES_FILE.read_text(encoding="utf-8"))["render_case"]
    current = snapshot["system"]
    current_settings = _memory_settings(current)
    profile = current["video_profile"]
    request = dict(case["request"])
    request["override_profile"] = profile
    if not compare:
        return {
            "request": request,
            "cases": [{
                "name": case["name"], "settings": current_settings,
                "request": {"override_profile": profile},
                "repeats": 2, "reload": True,
            }],
            "selection": "current hardware settings; first run cold, second run warm",
        }
    recommended = ((snapshot.get("system_detect") or {}).get("recommended") or {})
    if "video_profile" not in recommended:
        raise DeferredError("Maestro did not return a hardware-based video profile recommendation")
    recommended_profile = recommended["video_profile"]
    # Compare profiles while preserving every other current memory/runtime
    # setting. The recommendation endpoint also suggests allocator, pinning,
    # and other flags; the bench intentionally does not apply those changes.
    recommended_settings = dict(current_settings)
    recommended_settings["video_profile"] = recommended_profile
    return {
        "request": request,
        "cases": [
            {
                "name": "current-settings", "settings": current_settings,
                "request": {"override_profile": profile}, "repeats": 1, "reload": True,
            },
            {
                "name": "hardware-recommended", "settings": recommended_settings,
                "request": {"override_profile": recommended_profile}, "repeats": 1, "reload": True,
            },
        ],
        "selection": "one cold sample each: current profile versus Maestro's read-only recommendation; warm comparison is untested",
        "identical_settings": current_settings == recommended_settings,
    }


class StopRequested(memory_bench.BenchmarkError):
    pass


class ControlledBenchmarkRunner(memory_bench.BenchmarkRunner):
    """Add admission and STOP checks between bounded owned render jobs."""
    def __init__(self, *args, before_run, stop_requested, **kwargs):
        super().__init__(*args, **kwargs)
        self.before_run = before_run
        self.stop_requested = stop_requested
        self.halt_reason = None

    def _run_one(self, case, repeat, cold_reload):
        if self.stop_requested():
            self.halt_reason = "STOP requested; no new render case was submitted"
            raise StopRequested(self.halt_reason)
        try:
            state = self.before_run()
        except DeferredError as error:
            self.halt_reason = str(error)
            raise StopRequested(self.halt_reason) from error
        if state.get("busy"):
            self.halt_reason = "User work became active; no new render case was submitted"
            raise StopRequested(self.halt_reason)
        return super()._run_one(case, repeat, cold_reload)


def _stop_present(workspace: Path, output: Path) -> bool:
    return (workspace / "STOP").exists() or (output / "STOP").exists()


def _prompt_admission_client_class(api):
    # promptbench owns the POST lifecycle and uncertain-attempt receipt. This
    # adapter only adds the same safe busy/download check before each case.
    from promptbench import runner as prompt_runner

    class GuardedClient(prompt_runner.Client):
        def busy(self):
            return admission(api)["busy"]

    return GuardedClient


def _run_prompt_stage(api, output: Path, base_url: str, timeout_budget: int = 600, stop_requested=lambda: False) -> dict:
    from promptbench import runner as prompt_runner

    suite = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    prompt_output = output / "prompt"
    prompt_output.mkdir(parents=True, exist_ok=False)
    suite_path = prompt_output / "suite.json"
    write_json(suite_path, {"version": 1, "cases": suite["prompt_cases"]})
    args = argparse.Namespace(
        base_url=base_url, suite=str(suite_path), output=str(prompt_output), case=[],
        split="development", writer=[PROMPT_WRITER], candidate="testbench-baseline",
        experiment="baseline", repetitions=1, limit=PROMPT_MAX_ATTEMPTS,
        max_seconds=max(PROMPT_CASE_TIMEOUT, min(600, timeout_budget)),
        case_timeout=PROMPT_CASE_TIMEOUT, max_calls=PROMPT_MAX_CALLS, resume=False,
    )

    result_holder = {"code": None, "error": None}
    original = prompt_runner.Client
    prompt_runner.Client = _prompt_admission_client_class(api)

    def execute():
        try:
            result_holder["code"] = prompt_runner.run(args)
        except Exception as error:  # Promptbench records its partial/uncertain attempt first.
            result_holder["error"] = error

    worker = threading.Thread(target=execute, name="maestro-testbench-prompt", daemon=True)
    try:
        worker.start()
        while worker.is_alive():
            if stop_requested():
                (prompt_output / "STOP").touch(exist_ok=True)
            worker.join(timeout=0.25)
    finally:
        prompt_runner.Client = original

    prompt_manifest = read_json(prompt_output / "manifest.json", {})
    attempts = []
    for path in sorted((prompt_output / "attempts").glob("*.json")) if (prompt_output / "attempts").is_dir() else []:
        record = read_json(path, {})
        result = record.get("result") or {}
        assessment = record.get("assessment") or {}
        prepared = result.get("prepared") or {}
        plan = prepared.get("h3_window_plan") or {}
        warning_values = [
            *(assessment.get("warnings") or []), *(prepared.get("enhancement_warnings") or []),
            *(plan.get("planning_warnings") or []), *(result.get("warnings") or []),
        ]
        warnings = list(dict.fromkeys(str(item) for item in warning_values if str(item).strip()))
        normalized_warnings = [item.casefold() for item in warnings]
        source_fallback = any(
            "source-based draft" in item or "source based draft" in item
            or "did not produce a valid h3 frame prompt" in item
            for item in normalized_warnings
        )
        planner_fallback = "fallback" in str(plan.get("planned_by") or "").casefold()
        warned_fallback = any("fallback" in item for item in normalized_warnings)
        fallback_type = "source_based_draft" if source_fallback else (
            "native_fallback" if planner_fallback or warned_fallback else None
        )
        fallback = fallback_type is not None
        speech_parts = [
            LANGUAGE_TAG.sub("", str(item[1]))
            for item in assessment.get("spoken_spans") or []
            if isinstance(item, (list, tuple)) and len(item) > 1
        ]
        speech = " ".join(speech_parts)
        word_count = len(re.findall(r"\b[\w’'-]+\b", speech, flags=re.UNICODE))
        case = next((item for item in suite["prompt_cases"] if item["id"] == record.get("case_id")), None)
        target = case.get("dialogue_word_target") if case else None
        attempts.append({
            "case_id": record.get("case_id"), "status": record.get("status"),
            "calls": assessment.get("calls", 0), "warning_count": len(warnings),
            "native_fallback": fallback, "fallback_type": fallback_type,
            "enhanced_success": record.get("status") == "complete" and not fallback,
            "dialogue_words": word_count if target else None,
            "dialogue_target": target,
            "dialogue_target_met": bool(target and target["minimum"] <= word_count <= target["maximum"]) if target else None,
        })
    if any(item.get("status") in {"uncertain", "running"} for item in attempts):
        stage_status = "uncertain"
    elif prompt_manifest.get("status") in {
        "busy", "user work pending", "configured attempt/time limit reached",
        "STOP file requested; no new case submitted",
    } or stop_requested():
        stage_status = "deferred"
    elif result_holder["error"] is not None:
        message = str(result_holder["error"])
        if "not installed" in message.lower() or "asset" in message.lower():
            stage_status = "deferred"
        else:
            stage_status = "failed"
    elif any(item.get("status") != "complete" for item in attempts) or len(attempts) < len(suite["prompt_cases"]):
        stage_status = "completed_with_warnings"
    elif any(item.get("native_fallback") or item.get("warning_count") or item.get("dialogue_target_met") is False for item in attempts):
        stage_status = "completed_with_warnings"
    else:
        stage_status = "completed_with_warnings"  # Mechanical checks do not certify creative/rendered quality.
    return {
        "status": stage_status,
        "runner_status": prompt_manifest.get("status"),
        "attempts": attempts,
        "attempt_count": len(attempts),
        "max_attempts": PROMPT_MAX_ATTEMPTS,
        "case_timeout_seconds": PROMPT_CASE_TIMEOUT,
        "max_calls_per_case": PROMPT_MAX_CALLS,
        "writer": PROMPT_WRITER,
        "enhanced_success_count": sum(bool(item.get("enhanced_success")) for item in attempts),
        "native_fallback_count": sum(bool(item.get("native_fallback")) for item in attempts),
        "source_based_fallback_count": sum(item.get("fallback_type") == "source_based_draft" for item in attempts),
        "rendered_quality": "not tested by prompt mode; manual visual review required",
        "error": redact_log(str(result_holder["error"])) if result_holder["error"] else None,
        "child_report": "prompt/report.html" if (prompt_output / "report.html").is_file() else None,
    }


def _run_render_stage(api, snapshot: dict, output: Path, *, compare: bool,
                      timeout_seconds: int, before_run, stop_requested) -> dict:
    _assert_render_preconditions(snapshot)
    matrix = build_matrix(snapshot, compare=compare)
    matrix_path = output / ("compare-matrix.json" if compare else "render-matrix.json")
    write_json(matrix_path, matrix)
    render_output = output / ("compare" if compare else "render")
    log_path = memory_bench._matrix_log_path(None, ROOT)
    runner = ControlledBenchmarkRunner(
        memory_bench.ApiClient(api.base_url), matrix, render_output,
        timeout_seconds, 2, log_path,
        before_run=before_run, stop_requested=stop_requested,
        progress=False,
    )
    try:
        result = runner.execute()
    except memory_bench.BenchmarkError as error:
        result = read_json(render_output / "results.json", {})
        restoration = result.get("restoration") if isinstance(result, dict) else None
        if isinstance(restoration, dict) and restoration.get("status") in {"pending", "restoration_failed"}:
            status = "uncertain"
        elif runner.halt_reason:
            status = "deferred"
        else:
            status = "failed"
        return {
            "status": status, "error": redact_log(str(error)),
            "runner_status": result.get("status") if isinstance(result, dict) else None,
            "restoration": restoration,
            "runs": result.get("runs", []) if isinstance(result, dict) else [],
            "run_count": len(result.get("runs", [])) if isinstance(result, dict) else 0,
            "selection": matrix.get("selection"),
            "identical_settings": matrix.get("identical_settings", False),
            "manual_visual_quality": "not tested",
        }
    restoration = result.get("restoration") or {}
    if restoration.get("status") in {"pending", "restoration_failed"}:
        stage_status = "uncertain"
    elif restoration.get("status") != "restored":
        stage_status = "failed"
    elif result.get("status") == "completed_with_failures":
        stage_status = "failed"
    else:
        # Timings are useful, but they do not certify rendered visual quality.
        stage_status = "completed_with_warnings"
    return {
        "status": stage_status,
        "runner_status": result.get("status"),
        "restoration": restoration,
        "run_count": len(result.get("runs", [])),
        "runs": result.get("runs", []),
        "selection": matrix.get("selection"),
        "identical_settings": matrix.get("identical_settings", False),
        "manual_visual_quality": "not tested; operator review is required",
        "child_results": ("compare/results.json" if compare else "render/results.json"),
    }


def _html_report(manifest: dict) -> str:
    def esc(value):
        return html.escape(str(value), quote=True)

    rows = []
    for name, value in manifest.get("stages", {}).items():
        rows.append(
            "<tr><th>" + esc(name) + "</th><td>" + esc(value.get("status")) + "</td><td>"
            + esc(value.get("message") or value.get("error") or "") + "</td></tr>"
        )
        result = value.get("result") or {}
        if name == "prompt" and result.get("attempts"):
            for attempt in result["attempts"]:
                rows.append(
                    "<tr><th>" + esc(attempt.get("case_id")) + "</th><td>" + esc(attempt.get("status"))
                    + "</td><td>calls=" + esc(attempt.get("calls")) + "; warnings="
                    + esc(attempt.get("warning_count")) + "; native fallback="
                    + esc(attempt.get("native_fallback")) + " (" + esc(attempt.get("fallback_type"))
                    + "); enhanced success=" + esc(attempt.get("enhanced_success")) + "; dialogue words="
                    + esc(attempt.get("dialogue_words")) + "; target met="
                    + esc(attempt.get("dialogue_target_met")) + "</td></tr>"
                )
        if name in {"render", "compare"}:
            rows.append(
                "<tr><th>visual review</th><td>not tested</td><td>" + esc(result.get("manual_visual_quality", "")) + "</td></tr>"
            )
            for run in result.get("runs") or []:
                output_names = []
                for item in run.get("output_files") or []:
                    candidate = (item.get("filename") or item.get("name")) if isinstance(item, dict) else item
                    if isinstance(candidate, str) and candidate:
                        filename = Path(candidate.replace("\\", "/")).name
                        output_names.append(
                            '<a href="/api/v1/file/' + esc(quote(filename, safe=""))
                            + '?workspace=Test-Bench">' + esc(filename) + "</a>"
                        )
                links = ", ".join(output_names) if output_names else "none reported"
                rows.append(
                    "<tr><th>" + esc(run.get("case_name")) + " #" + esc(run.get("repeat")) + "</th><td>"
                    + esc(run.get("status")) + "</td><td>cold/warm=" + esc("cold" if run.get("reload_before_run") else "warm")
                    + "; wall=" + esc(run.get("wall_time_seconds")) + "s; denoise="
                    + esc(run.get("denoise_time_seconds")) + "s; peak VRAM="
                    + esc(run.get("peak_physical_vram_gb")) + " GiB; peak RAM="
                    + esc(run.get("peak_ram_used_gb")) + " GiB; profile/settings="
                    + esc(json.dumps(run.get("settings_used") or {}, sort_keys=True))
                    + "; output=" + links + "</td></tr>"
                )
    diagnostics = manifest.get("diagnostics") or {}
    runtime = diagnostics.get("runtime") or {}
    windows = diagnostics.get("windows") or {}
    model = (manifest.get("server") or {}).get("h3_model") or {}
    info = {
        "run_id": manifest.get("run_id"), "mode": manifest.get("mode"),
        "status": manifest.get("status"), "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"), "H3 model": model,
        "Python": runtime.get("python"), "PyTorch CUDA build": runtime.get("torch_cuda_build"),
        "Windows": windows,
    }
    return (
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\">"
        "<title>Maestro Test Bench</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;background:#161616;color:#eee}"
        "table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:10px;border-bottom:1px solid #444}pre{white-space:pre-wrap;background:#222;padding:14px}a{color:#8cf}</style><body>"
        "<h1>Maestro Test Bench</h1><p>" + esc(manifest.get("status")) + " · " + esc(manifest.get("mode")) + " · "
        + esc(manifest.get("run_id")) + "</p><p>Prompt outcomes report success, native fallbacks and warnings separately. Rendered visual quality was not manually reviewed.</p>"
        "<h2>Stages</h2><table><thead><tr><th>Stage</th><th>Status</th><th>Evidence</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table><h2>Runtime and readiness</h2><pre>"
        + esc(json.dumps(info, indent=2, ensure_ascii=False)) + "</pre><p>Code digest: "
        + esc(manifest.get("code_digest")) + "</p><p>Suite digest: " + esc(manifest.get("suite_digest"))
        + "</p><p><a href=\"" + esc(manifest.get("artifact_filename")) + "\">Download diagnostic bundle</a></p></body></html>"
    )


def _bundle_files(run_dir: Path) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for relative in (
        "manifest.json", "summary.json", "report.html", "diagnostics.json",
        "prompt/manifest.json", "prompt/summary.json",
        "render/results.json", "render/settings-backup.json", "compare/results.json", "compare/settings-backup.json",
        "prompt/suite.json", "render-matrix.json", "compare-matrix.json",
    ):
        path = run_dir / relative
        if path.is_file():
            paths.append((relative, path))
    attempts = run_dir / "prompt" / "attempts"
    if attempts.is_dir():
        paths.extend((f"prompt/attempts/{path.name}", path) for path in sorted(attempts.glob("*.json")))
    return paths


def create_bundle(run_dir: Path, bundle_path: Path, diagnostics_value: dict) -> None:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    included = 0
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(CASES_FILE, "cases.json")
        included += CASES_FILE.stat().st_size
        for name, path in _bundle_files(run_dir):
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if len(raw) > LOG_TAIL_LIMIT:
                continue
            if path.suffix.lower() == ".json":
                try:
                    value = _redact_json(json.loads(raw.decode("utf-8")))
                    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            else:
                payload = redact_log(raw.decode("utf-8", "replace")).encode("utf-8")
            if len(payload) > LOG_TAIL_LIMIT or included + len(payload) > BUNDLE_LIMIT - (2 * LOG_TAIL_LIMIT):
                continue
            archive.writestr(name, payload)
            included += len(payload)
        # Logs have fixed allowlisted sources, last-2-MiB tails and credential redaction.
        for name, payload in _tail_logs().items():
            payload = redact_log(payload.decode("utf-8", "replace")).encode("utf-8")
            if len(payload) > LOG_TAIL_LIMIT or included + len(payload) > BUNDLE_LIMIT:
                continue
            archive.writestr("logs/" + name, payload)
            included += len(payload)
        if included > BUNDLE_LIMIT:
            raise HarnessError("Diagnostic bundle exceeded the configured size limit")


def _alive(pid: int, *, platform_name: str | None = None, kernel32=None) -> bool:
    if pid <= 0:
        return False
    if (platform_name or sys.platform) == "win32":
        # os.kill(pid, 0) has process-control semantics on Windows; query the
        # process handle instead and fail closed if its state cannot be read.
        try:
            kernel = kernel32
            if kernel is None:
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.OpenProcess.restype = ctypes.c_void_p
                kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
                kernel.GetExitCodeProcess.restype = ctypes.c_int
                kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
                kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
            handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        except Exception:
            return True
        if not handle:
            try:
                error = ctypes.get_last_error()
            except AttributeError:
                error = 5
            return error != 87  # ERROR_INVALID_PARAMETER means the PID is gone.
        try:
            exit_code = ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
    except OSError as error:
        return getattr(error, "winerror", None) == 5 or error.errno == 1


def _lock_payload(path: Path) -> dict | None:
    return read_json(path, None) if path.is_file() else None


def _acquire_lock(workspace: Path, latest: Path) -> tuple[Path | None, dict | None]:
    lock = workspace / ".testbench.lock"
    prior = read_json(latest, {}) or {}
    reconcile_statuses = {"running", "uncertain", "interrupted", "restoration_pending"}
    if lock.exists():
        state = _lock_payload(lock)
        pid = state.get("pid") if isinstance(state, dict) else None
        if not isinstance(pid, int) or pid <= 0:
            raise DeferredError("A Test Bench lock has no verifiable process ID; reconcile it before retrying")
        if isinstance(pid, int) and _alive(pid):
            raise HarnessError("Another Test Bench process is active; inspect its current receipt before starting a second run")
        if prior.get("status") in reconcile_statuses:
            raise DeferredError(
                f"Previous Test Bench run {prior.get('run_id', 'unknown')} needs reconciliation; no new cases were submitted"
            )
        try:
            lock.unlink()
        except OSError as error:
            raise DeferredError("A stale Test Bench lock could not be cleared safely") from error
    if prior.get("status") in reconcile_statuses:
        if prior.get("status") == "running":
            prior.update(status="uncertain", finished_at=utc_now(),
                         message="Previous process ended without a final receipt; reconcile its prompt/render state before retrying.")
            write_json(latest, prior)
        raise DeferredError(
            f"Previous Test Bench run {prior.get('run_id', 'unknown')} needs reconciliation; no new cases were submitted"
        )
    try:
        with lock.open("x", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "created_at": utc_now()}, handle)
    except FileExistsError as error:
        raise HarnessError("Another Test Bench process acquired the run lock") from error
    return lock, prior


def _run_id_from_path(output: Path) -> str:
    return output.name if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", output.name) else make_run_id()


class TestBench:
    def __init__(self, mode: str, base_url: str | None = None, output: str | Path | None = None,
                 *, workspace: Path | None = None, api_factory=ApiClient):
        if mode not in {"smoke", "prompt", "render", "nightly", "compare"}:
            raise HarnessError("Unknown test mode")
        self.mode = mode
        self.raw_base_url = base_url or os.environ.get("MAESTRO_TEST_BASE_URL")
        self.workspace = Path(workspace or DEFAULT_WORKSPACE).resolve()
        self.run_id = _run_id_from_path(Path(output).expanduser().resolve()) if output else make_run_id()
        self.output = Path(output).expanduser().resolve() if output else self.workspace / self.run_id
        self.api_factory = api_factory
        self.api = None
        self.base_url = None
        self.started_at = utc_now()
        self.started_monotonic = time.monotonic()
        self.deadline = self.started_monotonic + NIGHTLY_LIMIT_SECONDS if mode == "nightly" else None
        self.manifest = {
            "schema_version": 1, "run_id": self.run_id, "mode": mode,
            "status": "running", "started_at": self.started_at, "finished_at": None,
            "code_digest": code_digest(), "suite_digest": suite_digest(),
            "case_ids": [item["id"] for item in json.loads(CASES_FILE.read_text(encoding="utf-8"))["prompt_cases"]],
            "suite_file": "cases.json", "stages": {}, "server": {}, "diagnostics": {},
            "artifact_filename": self.run_id + ".zip", "report_path": self.run_id + ".html",
            "quality_note": "Manual visual quality review was not performed.",
        }
        self.lock = None
        self.output_created = False

    def _stop(self) -> bool:
        return _stop_present(self.workspace, self.output)

    def _publish(self) -> None:
        self.manifest["elapsed_seconds"] = round(time.monotonic() - self.started_monotonic, 2)
        write_json(self.output / "manifest.json", self.manifest)
        write_json(self.output / "summary.json", {
            "run_id": self.run_id, "mode": self.mode, "status": self.manifest["status"],
            "stages": self.manifest["stages"], "finished_at": self.manifest.get("finished_at"),
        })
        write_json(self.workspace / "testbench-latest.json", {
            "schema_version": 1, "run_id": self.run_id, "status": self.manifest["status"],
            "mode": self.mode, "started_at": self.manifest["started_at"],
            "finished_at": self.manifest.get("finished_at"), "code_digest": self.manifest["code_digest"],
            "suite_digest": self.manifest["suite_digest"],
            "artifact_filename": self.manifest["artifact_filename"],
            "report_path": self.manifest["report_path"],
            "message": self.manifest.get("message"),
        })

    def _set_stage(self, name: str, status: str, **fields) -> None:
        self.manifest["stages"][name] = {"status": status, **fields}
        self._publish()

    def _deadline_budget(self) -> int:
        if self.deadline is None:
            return 600
        return max(0, int(self.deadline - time.monotonic()))

    def _state_for_stage(self, name: str) -> dict | None:
        if self._stop():
            self._set_stage(name, "deferred", message="STOP file present; no new case was submitted")
            return None
        try:
            state = admission(self.api)
        except DeferredError as error:
            self._set_stage(name, "deferred", message=str(error))
            return None
        if state["busy"]:
            self._set_stage(name, "deferred", message="User work is active: " + ", ".join(state["reasons"]), admission=state)
            return None
        return state

    def _smoke(self) -> None:
        self._set_stage("smoke", "running", message="Reading idle state and host diagnostics")
        try:
            self.manifest["server"] = server_snapshot(self.api)
            state = self.manifest["server"]["admission"]
            status = "deferred" if state["busy"] else "completed"
            message = ("User activity was observed; later cases were deferred" if state["busy"]
                       else "Read-only APIs and bounded host diagnostics completed")
            self._set_stage("smoke", status, message=message, admission=state)
        except DeferredError as error:
            self._set_stage("smoke", "deferred", message=str(error))

    def _stage_snapshot(self, name: str) -> dict | None:
        state = self._state_for_stage(name)
        if state is None:
            return None
        try:
            snapshot = server_snapshot(self.api)
        except DeferredError as error:
            self._set_stage(name, "deferred", message=str(error), admission=state)
            return None
        self.manifest["server"] = snapshot
        latest_state = snapshot.get("admission") or {}
        if latest_state.get("busy"):
            self._set_stage(name, "deferred", message="User work became active during the read-only preflight",
                            admission=latest_state)
            return None
        return snapshot

    def _prompt(self) -> None:
        budget = self._deadline_budget()
        if budget < PROMPT_CASE_TIMEOUT:
            self._set_stage("prompt", "deferred", message="Nightly time budget is too short for another complete prompt case")
            return
        self._set_stage("prompt", "running", message="Starting bounded Gemma prompt cases")
        try:
            result = _run_prompt_stage(
                self.api, self.output, self.base_url,
                timeout_budget=min(600, budget), stop_requested=self._stop,
            )
        except DeferredError as error:
            self._set_stage("prompt", "deferred", message=str(error))
            return
        self._set_stage("prompt", result["status"], result=result,
                        message=result.get("error") or result.get("runner_status"))

    def _render(self, *, compare: bool) -> None:
        name = "compare" if compare else "render"
        snapshot = self._stage_snapshot(name)
        if snapshot is None:
            return
        try:
            _assert_render_preconditions(snapshot)
            self._set_stage(name, "running", message="Starting bounded H3 render benchmark")
            budget = self._deadline_budget()
            if compare:
                run_timeout = min(900, max(60, budget // 2)) if self.deadline else 900
            elif self.deadline:
                # Reserve room for both cold/warm runs, cancellation cleanup and report output.
                run_timeout = min(720, max(60, (budget - 300) // 2))
            else:
                run_timeout = 900
            if self.deadline and budget < 420:
                raise DeferredError("Nightly time budget is too short for a bounded render case")
            result = _run_render_stage(
                self.api, snapshot, self.output, compare=compare,
                timeout_seconds=run_timeout,
                before_run=lambda: admission(self.api), stop_requested=self._stop,
            )
            status = result["status"]
        except DeferredError as error:
            result, status = None, "deferred"
            self._set_stage(name, status, message=str(error))
            return
        except Exception as error:
            result, status = {"error": redact_log(str(error))}, "failed"
        self._set_stage(name, status, result=result, message=result.get("error") or result.get("runner_status"))

    def _overall_status(self) -> tuple[str, str | None]:
        stages = list(self.manifest["stages"].values())
        if any(stage["status"] in {"uncertain", "interrupted"} for stage in stages):
            return "uncertain", "An in-flight operation has an uncertain outcome; reconcile before rerunning."
        if any(((stage.get("result") or {}).get("restoration") or {}).get("status")
               in {"pending", "restoration_failed"} for stage in stages):
            return "uncertain", "Settings restoration is unresolved; reconcile it before rerunning."
        if any(stage["status"] == "failed" for stage in stages):
            return "failed", "One or more requested stages failed; inspect the report and bundle."
        if any(stage["status"] == "deferred" for stage in stages):
            return "deferred", "A prerequisite, STOP request, or user activity deferred remaining cases."
        if any(stage["status"] == "completed_with_warnings" for stage in stages):
            return "completed_with_warnings", "Some cases need review; native fallbacks and manual visual quality are reported separately."
        return "completed", None

    def execute(self) -> int:
        self.workspace.mkdir(parents=True, exist_ok=True)
        latest = self.workspace / "testbench-latest.json"
        try:
            self.lock, _ = _acquire_lock(self.workspace, latest)
        except DeferredError as error:
            prior = read_json(latest, {}) or {}
            if prior.get("status") == "running":
                prior.update(status="uncertain", finished_at=utc_now(), message=str(error))
                write_json(latest, prior)
            print(str(error), file=sys.stderr)
            return 4
        except HarnessError as error:
            print(str(error), file=sys.stderr)
            return 1
        try:
            if self.output == self.workspace or self.output.exists():
                raise HarnessError("Output must be a new, unique directory outside the Test-Bench workspace root")
            if (self.workspace / self.manifest["artifact_filename"]).exists() or (self.workspace / self.manifest["report_path"]).exists():
                raise HarnessError("Run ID collides with an existing Test-Bench artifact; choose a new output name")
            self.output.mkdir(parents=True, exist_ok=False)
            self.output_created = True
            self.manifest["diagnostics"] = diagnostics()
            write_json(self.output / "diagnostics.json", self.manifest["diagnostics"])
            self._publish()
            if not self.raw_base_url:
                self._set_stage("preflight", "deferred", message="Supply --base-url or MAESTRO_TEST_BASE_URL with the local Maestro URL")
            else:
                try:
                    self.base_url = validate_base_url(self.raw_base_url)
                    self.api = self.api_factory(self.base_url)
                except HarnessError as error:
                    self._set_stage("preflight", "failed", message=str(error))
                if self.api is not None:
                    if self.mode in {"smoke", "nightly"}:
                        self._smoke()
                    stop_chain = any(
                        stage.get("status") in {"deferred", "failed", "uncertain", "interrupted"}
                        for stage in self.manifest["stages"].values()
                    )
                    if not stop_chain and self.mode in {"prompt", "nightly"}:
                        snapshot = self._stage_snapshot("prompt")
                        if snapshot is not None and not (snapshot.get("admission") or {}).get("busy"):
                            self._prompt()
                        elif snapshot is not None:
                            self._set_stage("prompt", "deferred", message="User work became active before prompt submission")
                    stop_chain = any(
                        stage.get("status") in {"deferred", "failed", "uncertain", "interrupted"}
                        for stage in self.manifest["stages"].values()
                    )
                    if not stop_chain and self.mode in {"render", "nightly"}:
                        self._render(compare=False)
                    elif not stop_chain and self.mode == "compare":
                        self._render(compare=True)
            status, message = self._overall_status()
            self.manifest.update(status=status, message=message, finished_at=utc_now())
            self._publish()
            report = _html_report(self.manifest)
            (self.output / "report.html").write_text(report, encoding="utf-8")
            report_path = self.workspace / self.manifest["report_path"]
            report_path.write_text(report, encoding="utf-8")
            write_json(self.output / "manifest.json", self.manifest)
            write_json(self.output / "summary.json", {
                "run_id": self.run_id, "mode": self.mode, "status": status,
                "stages": self.manifest["stages"], "finished_at": self.manifest["finished_at"],
            })
            create_bundle(self.output, self.workspace / self.manifest["artifact_filename"], self.manifest["diagnostics"])
            self._publish()
            print(f"Test Bench {self.run_id}: {status}; report {report_path.name}; bundle {self.manifest['artifact_filename']}")
            return 0 if status in {"completed", "completed_with_warnings"} else (4 if status == "uncertain" else 2 if status == "deferred" else 1)
        except KeyboardInterrupt:
            self.manifest.update(status="interrupted", message="Interrupted; inspect any running attempt before retrying", finished_at=utc_now())
            self._write_terminal_artifacts()
            return 4
        except HarnessError as error:
            self.manifest.update(status="failed", message=redact_log(str(error)), finished_at=utc_now())
            self._write_terminal_artifacts()
            print(str(error), file=sys.stderr)
            return 1
        except Exception as error:
            self.manifest.update(status="failed", message=redact_log(str(error)), finished_at=utc_now())
            self._write_terminal_artifacts()
            print("Test Bench failed; inspect its local report.", file=sys.stderr)
            return 1
        finally:
            if self.lock is not None:
                try:
                    self.lock.unlink(missing_ok=True)
                except OSError:
                    pass

    def _write_terminal_artifacts(self) -> None:
        if not self.output_created:
            return
        try:
            self._publish()
            report = _html_report(self.manifest)
            (self.output / "report.html").write_text(report, encoding="utf-8")
            (self.workspace / self.manifest["report_path"]).write_text(report, encoding="utf-8")
            write_json(self.output / "manifest.json", self.manifest)
            write_json(self.output / "summary.json", {
                "run_id": self.run_id, "mode": self.mode, "status": self.manifest["status"],
                "stages": self.manifest["stages"], "finished_at": self.manifest["finished_at"],
            })
            create_bundle(self.output, self.workspace / self.manifest["artifact_filename"], self.manifest["diagnostics"])
            self._publish()
        except Exception:
            # Keep the latest receipt even if a secondary artifact write fails.
            try:
                write_json(self.workspace / "testbench-latest.json", {
                    "schema_version": 1, "run_id": self.run_id,
                    "status": self.manifest["status"], "mode": self.mode,
                    "started_at": self.manifest["started_at"],
                    "finished_at": self.manifest.get("finished_at"),
                    "code_digest": self.manifest["code_digest"], "suite_digest": self.manifest["suite_digest"],
                    "artifact_filename": self.manifest["artifact_filename"],
                    "report_path": self.manifest["report_path"],
                    "message": self.manifest.get("message"),
                })
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded local Maestro test bench.")
    parser.add_argument("--base-url", help="Loopback Maestro API origin (or MAESTRO_TEST_BASE_URL)")
    parser.add_argument("--mode", choices=("smoke", "prompt", "render", "nightly", "compare"), default="smoke")
    parser.add_argument("--output", help="New local run directory; defaults to app/outputs/Test-Bench/<run_id>")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return TestBench(args.mode, args.base_url, args.output).execute()
    except HarnessError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
