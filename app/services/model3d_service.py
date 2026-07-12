"""Native Hunyuan3D job manager.

The Hunyuan runtime lives inside Maestro but uses an isolated Python
environment.  Each job runs in a short-lived worker process, so CUDA state and
VRAM are fully released when generation finishes.  This is deliberately
separate from Maestro's audio/video environment: the official Hunyuan3D 2.0
and 2.1 stacks require older diffusers/transformers builds.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any


SERVICE_DIR = Path(__file__).resolve().parent / "hunyuan3d"
ENV_DIR = SERVICE_DIR / "env"
INSTALL_MARKER = ENV_DIR / ".maestro_hunyuan3d_v1.installed"
WORKER_PATH = SERVICE_DIR / "worker.py"
VENDOR_DIR = SERVICE_DIR / "vendor"
JOBS_DIR = Path(__file__).resolve().parents[1] / "ckpts" / "model3d" / "jobs"
HF_CACHE_DIR = Path(__file__).resolve().parents[1] / "ckpts" / "model3d" / "huggingface"

MODEL3D_EXTENSIONS = {"glb", "obj", "ply", "stl"}


MODELS: list[dict[str, Any]] = [
    {
        "id": "hunyuan3d-2mini-turbo",
        "label": "Hunyuan3D 2 Mini Turbo",
        "engine": "v2",
        "repo": "tencent/Hunyuan3D-2mini",
        "subfolder": "hunyuan3d-dit-v2-mini-turbo",
        "parameters": "0.6B",
        "multiview": False,
        "turbo": True,
        "supports_text": True,
        "recommended_vram_gb": 6,
        "description": "Fastest geometry model; best when sharing the GPU with other Maestro workloads.",
    },
    {
        "id": "hunyuan3d-2mini-fast",
        "label": "Hunyuan3D 2 Mini Fast",
        "engine": "v2",
        "repo": "tencent/Hunyuan3D-2mini",
        "subfolder": "hunyuan3d-dit-v2-mini-fast",
        "parameters": "0.6B",
        "multiview": False,
        "turbo": False,
        "supports_text": True,
        "recommended_vram_gb": 6,
        "description": "Small guidance-distilled model with a quality/speed balance.",
    },
    {
        "id": "hunyuan3d-2mini",
        "label": "Hunyuan3D 2 Mini",
        "engine": "v2",
        "repo": "tencent/Hunyuan3D-2mini",
        "subfolder": "hunyuan3d-dit-v2-mini",
        "parameters": "0.6B",
        "multiview": False,
        "turbo": False,
        "supports_text": True,
        "recommended_vram_gb": 6,
        "description": "Full-step compact model.",
    },
    {
        "id": "hunyuan3d-2-turbo",
        "label": "Hunyuan3D 2 Turbo",
        "engine": "v2",
        "repo": "tencent/Hunyuan3D-2",
        "subfolder": "hunyuan3d-dit-v2-0-turbo",
        "parameters": "1.1B",
        "multiview": False,
        "turbo": True,
        "supports_text": True,
        "recommended_vram_gb": 8,
        "description": "Fast full-size Hunyuan3D 2.0 geometry model.",
    },
    {
        "id": "hunyuan3d-2-fast",
        "label": "Hunyuan3D 2 Fast",
        "engine": "v2",
        "repo": "tencent/Hunyuan3D-2",
        "subfolder": "hunyuan3d-dit-v2-0-fast",
        "parameters": "1.1B",
        "multiview": False,
        "turbo": False,
        "supports_text": True,
        "recommended_vram_gb": 8,
        "description": "Guidance-distilled full-size model.",
    },
    {
        "id": "hunyuan3d-2",
        "label": "Hunyuan3D 2",
        "engine": "v2",
        "repo": "tencent/Hunyuan3D-2",
        "subfolder": "hunyuan3d-dit-v2-0",
        "parameters": "1.1B",
        "multiview": False,
        "turbo": False,
        "supports_text": True,
        "recommended_vram_gb": 8,
        "description": "Original full-step Hunyuan3D 2.0 model.",
    },
    {
        "id": "hunyuan3d-2mv-turbo",
        "label": "Hunyuan3D 2 Multi-view Turbo",
        "engine": "v2",
        "repo": "tencent/Hunyuan3D-2mv",
        "subfolder": "hunyuan3d-dit-v2-mv-turbo",
        "parameters": "1.1B",
        "multiview": True,
        "turbo": True,
        "supports_text": False,
        "recommended_vram_gb": 8,
        "description": "Fast front/left/right/back image-to-3D model.",
    },
    {
        "id": "hunyuan3d-2mv-fast",
        "label": "Hunyuan3D 2 Multi-view Fast",
        "engine": "v2",
        "repo": "tencent/Hunyuan3D-2mv",
        "subfolder": "hunyuan3d-dit-v2-mv-fast",
        "parameters": "1.1B",
        "multiview": True,
        "turbo": False,
        "supports_text": False,
        "recommended_vram_gb": 8,
        "description": "Guidance-distilled multi-view geometry model.",
    },
    {
        "id": "hunyuan3d-2mv",
        "label": "Hunyuan3D 2 Multi-view",
        "engine": "v2",
        "repo": "tencent/Hunyuan3D-2mv",
        "subfolder": "hunyuan3d-dit-v2-mv",
        "parameters": "1.1B",
        "multiview": True,
        "turbo": False,
        "supports_text": False,
        "recommended_vram_gb": 8,
        "description": "Highest-quality full-step multi-view model.",
    },
    {
        "id": "hunyuan3d-2.1",
        "label": "Hunyuan3D 2.1 + PBR",
        "engine": "v21",
        "repo": "tencent/Hunyuan3D-2.1",
        "subfolder": "hunyuan3d-dit-v2-1",
        "parameters": "3.3B",
        "multiview": False,
        "turbo": False,
        "supports_text": True,
        "recommended_vram_gb": 10,
        "description": "Highest-fidelity geometry with optional production-ready PBR materials.",
    },
]

MODEL_BY_ID = {model["id"]: model for model in MODELS}

PRESETS: dict[str, dict[str, Any]] = {
    "eco": {
        "label": "Low VRAM",
        "description": "Mini Turbo, CPU offload, 128 octree, no texture.",
        "model_id": "hunyuan3d-2mini-turbo",
        "num_inference_steps": 5,
        "guidance_scale": 5.0,
        "octree_resolution": 128,
        "num_chunks": 8000,
        "texture_mode": "none",
        "cpu_offload": True,
        "flashvdm": True,
    },
    "balanced": {
        "label": "Balanced",
        "description": "Full-size Turbo geometry with standard texture support.",
        "model_id": "hunyuan3d-2-turbo",
        "num_inference_steps": 5,
        "guidance_scale": 5.0,
        "octree_resolution": 256,
        "num_chunks": 12000,
        "texture_mode": "v2-turbo",
        "cpu_offload": True,
        "flashvdm": True,
    },
    "quality": {
        "label": "Quality / PBR",
        "description": "Hunyuan3D 2.1, 384 octree and PBR materials.",
        "model_id": "hunyuan3d-2.1",
        "num_inference_steps": 30,
        "guidance_scale": 5.0,
        "octree_resolution": 384,
        "num_chunks": 20000,
        "texture_mode": "pbr",
        "cpu_offload": True,
        "flashvdm": False,
    },
    "multiview": {
        "label": "Multi-view Fast",
        "description": "Multi-view Turbo using up to four reference views.",
        "model_id": "hunyuan3d-2mv-turbo",
        "num_inference_steps": 5,
        "guidance_scale": 5.0,
        "octree_resolution": 256,
        "num_chunks": 12000,
        "texture_mode": "v2-turbo",
        "cpu_offload": True,
        "flashvdm": True,
    },
}

_jobs: dict[str, dict[str, Any]] = {}
_processes: dict[str, subprocess.Popen] = {}
_lock = threading.RLock()
_generation_slot = threading.Semaphore(1)

_TERMINAL_STATES = {"completed", "failed", "cancelled"}
# Job-registry hygiene: keep a short history of finished jobs for status
# polling, but never let the in-memory dict grow with server uptime.
_MAX_FINISHED_JOBS = 20
_FINISHED_JOB_TTL_SECONDS = 3600
# Backpressure: the semaphore serializes GPU work, so anything beyond a
# handful of waiting jobs means something is stuck — reject early instead of
# accumulating blocked threads.
_MAX_ACTIVE_JOBS = 4
# Watchdog limits for a single worker: silence usually means a stalled
# download or a wedged CUDA context; the absolute cap covers slow-but-alive
# pathological runs.
_WORKER_INACTIVITY_LIMIT_SECONDS = 15 * 60
_WORKER_TIME_LIMIT_SECONDS = 2 * 3600


def _python_path() -> Path | None:
    candidates = [ENV_DIR / "python.exe", ENV_DIR / "bin" / "python"]
    return next((path for path in candidates if path.is_file()), None)


def installation_status() -> dict[str, Any]:
    python_path = _python_path()
    v2_source = VENDOR_DIR / "Hunyuan3D-2" / "hy3dgen"
    v21_source = VENDOR_DIR / "Hunyuan3D-2.1" / "hy3dshape"
    installed = bool(python_path and INSTALL_MARKER.is_file() and WORKER_PATH.is_file() and v2_source.is_dir() and v21_source.is_dir())
    return {
        "installed": installed,
        "python": str(python_path) if python_path else None,
        "v2_source": v2_source.is_dir(),
        "v21_source": v21_source.is_dir(),
        "isolated_runtime": True,
        "releases_vram_after_job": True,
        "install_hint": None if installed else "Run Maestro's standard Install or Update action.",
    }


def is_model_downloaded(model_id: str) -> bool:
    """Return whether the selected Hugging Face snapshot is cached locally."""
    model = MODEL_BY_ID.get(model_id)
    if model is None:
        return False
    repo_cache = HF_CACHE_DIR / "hub" / f"models--{model['repo'].replace('/', '--')}" / "snapshots"
    if not repo_cache.is_dir():
        return False
    snapshots = [path for path in repo_cache.iterdir() if path.is_dir()]
    if model["engine"] == "v21":
        return bool(snapshots)
    return any((snapshot / model["subfolder"]).is_dir() for snapshot in snapshots)


def models_sharing_repo(model_id: str) -> list[dict[str, Any]]:
    """Return sibling catalog entries whose weights live in the same HF repo.

    Several variants (mini/fast/turbo families) are subfolders of one shared
    Hugging Face repository, and the cache can only be removed per-repo:
    snapshot files are deduplicated blobs, so deleting a single subfolder
    could silently corrupt the remaining variants.
    """
    model = MODEL_BY_ID.get(model_id)
    if model is None:
        return []
    return [item for item in MODELS if item["repo"] == model["repo"] and item["id"] != model_id]


def delete_model_cache(model_id: str) -> list[str]:
    """Remove the upstream repository cache used by a Hunyuan3D variant.

    Deletion is repo-granular (see models_sharing_repo), so callers must
    surface the affected sibling variants to the user before invoking this.
    """
    model = MODEL_BY_ID.get(model_id)
    if model is None:
        raise ValueError(f"Unknown Hunyuan3D model: {model_id}")
    repo_cache = HF_CACHE_DIR / "hub" / f"models--{model['repo'].replace('/', '--')}"
    if not repo_cache.exists():
        return []
    shutil.rmtree(repo_cache)
    return [model["repo"]]


def capabilities() -> dict[str, Any]:
    with _lock:
        active = sum(1 for job in _jobs.values() if job["status"] in {"queued", "running"})
    return {
        "runtime": installation_status(),
        "models": MODELS,
        "presets": [{"id": key, **value} for key, value in PRESETS.items()],
        "texture_modes": [
            {"id": "none", "label": "Geometry only", "recommended_vram_gb": 6},
            {"id": "v2", "label": "Hunyuan3D Paint 2.0", "recommended_vram_gb": 16},
            {"id": "v2-turbo", "label": "Hunyuan3D Paint 2.0 Turbo", "recommended_vram_gb": 16},
            {"id": "pbr", "label": "Hunyuan3D Paint 2.1 PBR", "recommended_vram_gb": 21},
        ],
        "input_views": ["front", "left", "right", "back"],
        "output_formats": sorted(MODEL3D_EXTENSIONS),
        "active_jobs": active,
    }


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def _bounded_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def _prepare_request(body: dict[str, Any], image_paths: dict[str, str]) -> dict[str, Any]:
    preset_id = str(body.get("preset") or "balanced")
    preset = dict(PRESETS.get(preset_id, PRESETS["balanced"]))
    model_id = str(body.get("model_id") or preset["model_id"])
    if model_id not in MODEL_BY_ID:
        raise ValueError(f"Unknown Hunyuan3D model: {model_id}")
    model = dict(MODEL_BY_ID[model_id])

    prompt = str(body.get("prompt") or "").strip()
    clean_images = {key: value for key, value in image_paths.items() if key in {"front", "left", "right", "back"} and value}
    if model["multiview"]:
        if "front" not in clean_images:
            raise ValueError("Multi-view models require at least a front image")
    elif not clean_images and not prompt:
        raise ValueError("Provide an image or a text prompt")

    output_format = str(body.get("output_format") or "glb").lower().lstrip(".")
    if output_format not in MODEL3D_EXTENSIONS:
        raise ValueError(f"Unsupported 3D output format: {output_format}")

    texture_mode = str(body.get("texture_mode", preset["texture_mode"]))
    if texture_mode not in {"none", "v2", "v2-turbo", "pbr"}:
        raise ValueError(f"Unsupported texture mode: {texture_mode}")
    if texture_mode == "pbr" and model["engine"] != "v21":
        raise ValueError("PBR materials require the Hunyuan3D 2.1 model")
    if texture_mode == "pbr" and output_format != "glb":
        raise ValueError("PBR materials must be exported as GLB so all maps remain embedded")

    settings = {
        "prompt": prompt,
        "seed": _bounded_int(body.get("seed"), 1234, 0, 2**32 - 1),
        "num_inference_steps": _bounded_int(body.get("num_inference_steps", preset["num_inference_steps"]), preset["num_inference_steps"], 1, 100),
        "guidance_scale": _bounded_float(body.get("guidance_scale", preset["guidance_scale"]), preset["guidance_scale"], 0.0, 30.0),
        "octree_resolution": _bounded_int(body.get("octree_resolution", preset["octree_resolution"]), preset["octree_resolution"], 64, 512),
        "num_chunks": _bounded_int(body.get("num_chunks", preset["num_chunks"]), preset["num_chunks"], 1000, 500000),
        "texture_mode": texture_mode,
        "texture_resolution": _bounded_int(body.get("texture_resolution"), 512, 256, 1024),
        "remove_background": bool(body.get("remove_background", True)),
        "cpu_offload": bool(body.get("cpu_offload", preset["cpu_offload"])),
        "flashvdm": bool(body.get("flashvdm", preset["flashvdm"])),
        "compile": bool(body.get("compile", False)),
        "mc_algo": str(body.get("mc_algo") or "dmc"),
        "reduce_face": bool(body.get("reduce_face", False)),
        "target_face_num": _bounded_int(body.get("target_face_num"), 40000, 100, 1_000_000),
        "output_format": output_format,
    }
    if settings["mc_algo"] not in {"mc", "dmc"}:
        settings["mc_algo"] = "dmc"

    return {
        "preset": preset_id,
        "model": model,
        "images": clean_images,
        "settings": settings,
    }


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key not in {"request", "process"}}


def _prune_finished_jobs_locked() -> None:
    """Drop old terminal jobs; callers must hold _lock."""
    now = time.time()
    finished = sorted(
        (item for item in _jobs.items() if item[1]["status"] in _TERMINAL_STATES),
        key=lambda item: item[1].get("updated_at", 0.0),
        reverse=True,
    )
    for index, (job_id, job) in enumerate(finished):
        if index >= _MAX_FINISHED_JOBS or now - job.get("updated_at", 0.0) > _FINISHED_JOB_TTL_SECONDS:
            _jobs.pop(job_id, None)


def start_job(*, body: dict[str, Any], image_paths: dict[str, str], output_dir: str) -> dict[str, Any]:
    runtime = installation_status()
    if not runtime["installed"]:
        raise RuntimeError(runtime["install_hint"])

    with _lock:
        _prune_finished_jobs_locked()
        active = sum(1 for job in _jobs.values() if job["status"] in {"queued", "running"})
    if active >= _MAX_ACTIVE_JOBS:
        raise ValueError("Too many queued 3D jobs; wait for the current ones to finish or cancel them")

    request_data = _prepare_request(body, image_paths)
    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "phase": "queued",
        "message": "Queued Hunyuan3D generation",
        "error": None,
        "filename": None,
        "url": None,
        "model_id": request_data["model"]["id"],
        "created_at": time.time(),
        "updated_at": time.time(),
        "request": request_data,
    }
    with _lock:
        _jobs[job_id] = job
        initial_response = _public_job(dict(job))
    thread = threading.Thread(target=_run_job, args=(job_id, os.path.abspath(output_dir)), daemon=True)
    thread.start()
    return initial_response


def _update_job(job_id: str, **updates: Any) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = time.time()
        # The request payload (settings + image paths) is only needed while
        # the job runs; keeping it on finished jobs just bloats the registry.
        if job["status"] in _TERMINAL_STATES:
            job.pop("request", None)


def _run_job(job_id: str, output_dir: str) -> None:
    _generation_slot.acquire()
    try:
        with _lock:
            if _jobs.get(job_id, {}).get("status") == "cancelled":
                return
        _run_job_serialized(job_id, output_dir)
    finally:
        _generation_slot.release()


def _cleanup_partial_output(output_path: Path) -> None:
    """Remove a failed/cancelled job's half-written export and its preview."""
    for stale in (output_path, output_path.with_suffix(".preview.png")):
        try:
            stale.unlink(missing_ok=True)
        except OSError:
            pass


def _run_job_serialized(job_id: str, output_dir: str) -> None:
    python_path = _python_path()
    if not python_path:
        _update_job(job_id, status="failed", phase="failed", error="Hunyuan3D runtime is not installed")
        return

    with _lock:
        job = _jobs.get(job_id)
        request_data = job.get("request") if job else None
    if not request_data:
        # Cancelled (and possibly pruned) before the slot was acquired.
        return
    model_id = request_data["model"]["id"]
    output_format = request_data["settings"]["output_format"]
    safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_id)
    stamp = time.strftime("%Y-%m-%d-%Hh%Mm%Ss")
    filename = f"{stamp}_{safe_model}_{job_id[:8]}.{output_format}"
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    request_path = JOBS_DIR / f"{job_id}.json"
    request_path.write_text(json.dumps(request_data, indent=2), encoding="utf-8")
    pid_path = JOBS_DIR / f"{job_id}.pid"

    command = [str(python_path), str(WORKER_PATH), "--request", str(request_path), "--output", str(output_path)]
    env = os.environ.copy()
    # Built-in Hunyuan3D models are public. Do not inherit Pinokio's global HF
    # credentials: some Hub/proxy combinations omit X-Repo-Commit from
    # authenticated HEAD responses, which makes huggingface_hub incorrectly
    # report that model_index.json is missing.
    isolated_network_vars = (
        "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH",
        "HF_ENDPOINT", "HF_INFERENCE_ENDPOINT", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE",
    )
    for env_var in isolated_network_vars:
        env.pop(env_var, None)
    env.update({
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": str(HF_CACHE_DIR),
        "HUGGINGFACE_HUB_CACHE": str(HF_CACHE_DIR / "hub"),
        "HF_ENDPOINT": "https://huggingface.co",
        "HF_HUB_ETAG_TIMEOUT": "30",
        "HF_HUB_DOWNLOAD_TIMEOUT": "60",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    lines: list[str] = []
    try:
        _update_job(job_id, status="running", phase="starting", message="Starting isolated Hunyuan3D worker", progress=0.02)
        process = subprocess.Popen(
            command,
            cwd=str(SERVICE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with _lock:
            _processes[job_id] = process
        # Record the worker PID on disk so a hard-killed Maestro (SIGKILL,
        # OOM, reload) can reap the orphan on the next startup instead of
        # leaving it holding VRAM forever.
        try:
            pid_path.write_text(str(process.pid), encoding="utf-8")
        except OSError:
            pass

        activity = {"at": time.monotonic()}
        deadline = time.monotonic() + _WORKER_TIME_LIMIT_SECONDS
        timeout_reason: dict[str, str] = {}

        def _watchdog() -> None:
            while process.poll() is None:
                time.sleep(15)
                now = time.monotonic()
                if now - activity["at"] > _WORKER_INACTIVITY_LIMIT_SECONDS:
                    timeout_reason["error"] = (
                        f"3D worker produced no output for {_WORKER_INACTIVITY_LIMIT_SECONDS // 60} minutes"
                    )
                elif now > deadline:
                    timeout_reason["error"] = (
                        f"3D generation exceeded the {_WORKER_TIME_LIMIT_SECONDS // 3600}h limit"
                    )
                else:
                    continue
                process.kill()
                return

        threading.Thread(target=_watchdog, daemon=True).start()

        assert process.stdout is not None
        for raw_line in process.stdout:
            activity["at"] = time.monotonic()
            line = raw_line.rstrip()
            if not line:
                continue
            print(f"[Hunyuan3D] {line}")
            lines.append(line)
            lines = lines[-60:]
            if line.startswith("MAESTRO_EVENT "):
                try:
                    event = json.loads(line[len("MAESTRO_EVENT "):])
                    _update_job(
                        job_id,
                        phase=str(event.get("phase") or "running"),
                        message=str(event.get("message") or "Generating 3D asset"),
                        progress=max(0.0, min(0.99, float(event.get("progress", 0.0)))),
                    )
                except Exception:
                    pass

        exit_code = process.wait()
        with _lock:
            status = _jobs.get(job_id, {}).get("status")
        if status == "cancelled":
            _cleanup_partial_output(output_path)
            return
        if timeout_reason:
            raise RuntimeError(timeout_reason["error"])
        if exit_code != 0 or not output_path.is_file():
            detail = "\n".join(lines[-25:]) or f"Worker exited with code {exit_code}"
            raise RuntimeError(detail[-4000:])

        sidecar = output_path.with_suffix(".meta.json")
        sidecar.write_text(
            json.dumps(
                {
                    "generation_mode": "model3d",
                    "mode": "model3d",
                    "job_id": job_id,
                    "created_at": time.time(),
                    "params": {
                        **request_data["settings"],
                        "model_id": model_id,
                        "preset": request_data["preset"],
                        "images": request_data["images"],
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _update_job(
            job_id,
            status="completed",
            phase="completed",
            message="3D asset generated; worker exited and VRAM was released",
            progress=1.0,
            filename=filename,
            url=f"/api/v1/file/{filename}",
            size=output_path.stat().st_size,
        )
    except Exception as exc:
        with _lock:
            cancelled = _jobs.get(job_id, {}).get("status") == "cancelled"
        if not cancelled:
            _update_job(job_id, status="failed", phase="failed", message="Hunyuan3D generation failed", error=str(exc))
        _cleanup_partial_output(output_path)
    finally:
        with _lock:
            _processes.pop(job_id, None)
        for stale_path in (request_path, pid_path):
            try:
                stale_path.unlink(missing_ok=True)
            except Exception:
                pass


def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
        return _public_job(dict(job)) if job else None


def cancel_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
        process = _processes.get(job_id)
        if not job:
            return None
        if job["status"] in {"completed", "failed", "cancelled"}:
            return _public_job(dict(job))
        job.update({
            "status": "cancelled",
            "phase": "cancelled",
            "message": "3D generation cancelled",
            "updated_at": time.time(),
        })
        job.pop("request", None)
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    return get_job(job_id)


def cancel_all_jobs() -> int:
    with _lock:
        active_ids = [job_id for job_id, job in _jobs.items() if job["status"] in {"queued", "running"}]
    for job_id in active_ids:
        cancel_job(job_id)
    return len(active_ids)


def _is_hunyuan_worker(pid: int) -> bool:
    """Best-effort check that a PID belongs to one of our worker processes."""
    proc_cmdline = Path("/proc") / str(pid) / "cmdline"
    try:
        if proc_cmdline.is_file():
            cmdline = proc_cmdline.read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
            return WORKER_PATH.name in cmdline and "hunyuan3d" in cmdline
    except OSError:
        return False
    try:
        # Non-Linux platforms: only act when psutil can positively identify
        # the process; PIDs get recycled, so never kill blindly.
        import psutil

        cmdline = " ".join(psutil.Process(pid).cmdline())
        return WORKER_PATH.name in cmdline and "hunyuan3d" in cmdline
    except Exception:
        return False


def _reap_stale_jobs() -> None:
    """Clean up after a previous Maestro process that died mid-generation.

    Job state is in-memory and the atexit hook does not run on SIGKILL/OOM,
    so an interrupted run can leave an orphan worker holding VRAM plus stale
    request/pid files in JOBS_DIR. Everything found here predates this
    process and is stale by definition.
    """
    if not JOBS_DIR.is_dir():
        return
    for pid_path in JOBS_DIR.glob("*.pid"):
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = 0
        if pid > 0 and _is_hunyuan_worker(pid):
            print(f"[Hunyuan3D] Terminating orphaned worker from a previous run (pid {pid})")
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        try:
            pid_path.unlink(missing_ok=True)
        except OSError:
            pass
    for request_path in JOBS_DIR.glob("*.json"):
        try:
            request_path.unlink(missing_ok=True)
        except OSError:
            pass


atexit.register(cancel_all_jobs)

try:
    _reap_stale_jobs()
except Exception as exc:  # Never block Maestro startup on cleanup.
    print(f"[Hunyuan3D] Stale job cleanup skipped: {exc}")
