"""Local development adapter to the real queued-enhancement preparation path."""
from copy import deepcopy
import ipaddress
import os
from pathlib import Path
import platform
import time

from . import PROTOCOL_VERSION
from .evidence import digest, file_digest, source_fingerprint
from .trace import capture, expired, sanitize
from .experiments import EXPERIMENTS, experiment_context, validate_experiment

# The bench supports H3 writing, not preprocessing tools or generation side effects.
PARAM_KEYS = frozenset({
    "prompt", "model_type", "generation_mode", "resolution", "video_length", "seed",
    "num_inference_steps", "guidance_scale", "minimax_h3_references", "image_start", "image_end",
    "minimax_h3_reference_sequence", "minimax_h3_sequence_clip_frames",
    "minimax_h3_sequence_memory_override", "minimax_h3_sequence_continuity",
    "sliding_window_size", "sliding_window_overlap", "sliding_window_memory_override",
    "minimax_h3_multi_window",
})

# These are the built-in Reference entries available to the standalone suite
# validator.  The live endpoint additionally checks the resolved model
# definition, so aliases/custom definitions are classified by capability
# rather than by name or a broad substring match.
REFERENCE_MODEL_TYPES = frozenset({
    "minimax_h3_ref2va",
    "minimax_h3_ref2va_full",
    "minimax_h3_ref2va_fused_turbo",
})


def validate_workflow(value):
    if value not in ("studio", "director"):
        raise ValueError("Prompt bench workflow must be studio or director")
    return value


def _reference_model(model_type, model=None):
    if model is not None:
        return bool(
            model.get("omni_reference")
            or model.get("architecture") in {"minimax_h3_ref2va", "minimax_h3_ref2va_full"}
        )
    return model_type in REFERENCE_MODEL_TYPES


def local_request(request):
    try:
        return (ipaddress.ip_address(request.client.host).is_loopback
                and request.headers.get("x-maestro-prompt-bench") == "1")
    except (ValueError, AttributeError):
        return False


def validate_params(params, model=None):
    if not isinstance(params, dict) or set(params) - PARAM_KEYS:
        raise ValueError("Bench params contain unsupported fields; use the documented H3 fixture schema.")
    if not str(params.get("prompt") or "").strip():
        raise ValueError("A source prompt is required.")
    if len(params["prompt"]) > 100000:
        raise ValueError("Source prompt exceeds the pilot limit.")
    if params.get("generation_mode", "video") != "video":
        raise ValueError("The first bench supports H3 video prompting only.")
    frames = params.get("video_length", 0)
    if isinstance(frames, bool) or not isinstance(frames, int) or not 1 <= frames <= 2880:
        raise ValueError("Pilot video_length must be 1-2880 frames.")
    for path in [params.get("image_start"), params.get("image_end")]:
        if path and not Path(path).is_file():
            raise ValueError(f"Missing reference asset: {path}")
    refs = params.get("minimax_h3_references") or []
    if not isinstance(refs, list):
        raise ValueError("minimax_h3_references must be a list.")
    for ref in refs:
        if not isinstance(ref, dict) or not ref.get("path") or not Path(ref["path"]).is_file():
            raise ValueError("Every declared reference must have an existing file.")
    validated = deepcopy(params)
    if _reference_model(str(params.get("model_type") or ""), model):
        # Keep admission aligned with production Ref2VA validation.  In
        # particular, an audio reference does not satisfy the visual Reference
        # workflow and this must fail before the GPU slot or writer is touched.
        from models.minimax_h3.reference_manifest import validate_reference_manifest

        validated["minimax_h3_references"] = validate_reference_manifest(
            refs, require_files=True, require_visual=True,
        )
    return validated


def installed_writer(llm, model_id):
    entry = llm.MODEL_REGISTRY.get(model_id)
    if entry is None:
        raise ValueError("Choose an installed local writer from Maestro's registry.")
    stem = model_id.split("/")[-1].replace("-GGUF", "")
    directory = Path(llm.get_model_dir()) / (entry.get("cache_dir_override") or stem)
    names = [entry["gguf_file"]]
    if entry.get("mmproj_file"):
        # Match the production loader's preference for an existing projector
        # alias after an upstream rename; record the file it will actually use.
        projector = next((name for name in entry.get("mmproj_cache_aliases", [])
                          if (directory / name).is_file()
                          and (directory / name).stat().st_size > 0),
                         entry["mmproj_file"])
        names.append(projector)
    assets = []
    for name in names:
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Writer asset is not installed: {path}. The bench does not download models.")
        stat = path.stat()
        assets.append({"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {"model_id": model_id, "provider": "local", "assets": assets,
            "registry": deepcopy(entry), "identity_method": "registry + file size/mtime; weights not fully hashed"}


def runtime_identity(llm):
    """Inspect installed runtime files without running or updating the backend."""
    directory = Path(os.environ.get("MAESTRO_LLAMA_BIN") or getattr(llm, "DEFAULT_BIN_DIR", "."))
    names = ("llama-server.exe", "llama-server", "llama.dll", "libllama.so", "ggml-cuda.dll", "libggml-cuda.so")
    files = []
    for name in names:
        path = directory / name
        if path.is_file():
            files.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_digest(path)})
    receipt_reader = getattr(llm, "_read_llama_runtime_receipt", None)
    return {"python": platform.python_version(), "platform": platform.platform(), "files": files,
            "receipt": sanitize(receipt_reader(str(directory))) if receipt_reader else {},
            "scope": "server and core/CUDA libraries; external driver/runtime libraries not fully hashed"}


def register_routes(api, *, slot, settings, get_model, enhance, prepare, app_root):
    from fastapi import HTTPException, Request
    from services import llm_service as llm
    from services.studio_enhancement import enhancement_context, prepare_enhanced_job

    started_sources = source_fingerprint(app_root)
    started_digest = digest(started_sources)

    async def info(request: Request):
        if not local_request(request):
            raise HTTPException(403, "Prompt bench is available only to the local developer runner.")
        models = []
        for model_id in llm.MODEL_REGISTRY:
            try:
                models.append(installed_writer(llm, model_id))
            except (ValueError, KeyError):
                continue
        return {"protocol": PROTOCOL_VERSION, "source_digest": started_digest,
                "source_sha256": started_sources, "python": platform.python_version(),
                "writers": models, "runtime": runtime_identity(llm),
                "mode": "prepare_only", "generation_started": False, "experiments": EXPERIMENTS,
                "workflows": ["studio", "director"]}

    @slot
    async def run_owned(request: Request):
        body = await request.json()
        try:
            params = validate_params(body.get("params"))
            model = get_model(params.get("model_type")) or {}
            if not str(model.get("architecture") or "").startswith("minimax_h3") or model.get("minimax_h3_viggle"):
                raise ValueError("The bench supports H3 Frames/References generation models.")
            writer = installed_writer(llm, str(body.get("writer") or ""))
            timeout = float(body.get("timeout_seconds", 900))
            max_calls = int(body.get("max_calls", 32))
            experiment = validate_experiment(body.get("experiment", "baseline"))
            workflow = validate_workflow(body.get("workflow", "studio"))
            if workflow == "director" and experiment != "baseline":
                raise ValueError("Director experiments currently use the production baseline only")
            if not 30 <= timeout <= 1800 or not 1 <= max_calls <= 64:
                raise ValueError("Use a 30-1800 second timeout and 1-64 LLM calls per case.")
            if body.get("source_digest") != started_digest or source_fingerprint(app_root) != started_sources:
                raise ValueError("Bench source differs from the running server; restart while idle and capture a fresh baseline.")
        except (ValueError, TypeError) as error:
            raise HTTPException(400, str(error)) from error
        selected = {**settings(), "llm_provider": "local", "llm_model_id": writer["model_id"],
                    "llm_device": "cuda", "llm_remote_url": "", "enhance_llm_model_id": writer["model_id"],
                    "enhance_llm_device": "cuda", "nsfw_mode": False}
        started = time.monotonic()
        result = {"protocol": PROTOCOL_VERSION, "writer": writer, "source_digest": started_digest,
                  "request": params, "settings": sanitize(selected),
                  "generation_started": False, "mode": "prepare_only", "experiment": experiment, "workflow": workflow}
        with capture(timeout, max_calls) as trace:
            try:
                with enhancement_context(selected, expired), experiment_context(experiment):
                    if workflow == "director":
                        import asyncio
                        from .director import prepare_director
                        prepared = await asyncio.to_thread(prepare_director, params, model, selected)
                    else:
                        prepared = await prepare_enhanced_job(params, model, enhance, prepare)
                    result.update(status="complete", prepared=prepared)
            except Exception as error:
                result.update(status="failed", error=str(getattr(error, "detail", None) or error),
                              error_type=type(error).__name__)
            finally:
                result["writer_status"] = llm.get_status()
                process = getattr(llm, "_process", None)
                # Local llama-server arguments describe the GGUF, projection,
                # context and backend flags. No remote API credentials are used.
                result["runtime_command"] = getattr(process, "args", None)
                if llm.is_loaded():
                    try:
                        llm.unload_model()
                    except Exception as error:
                        result["cleanup_error"] = str(error)
                        result["status"] = "failed"
            result["calls"] = trace["calls"]
        result["seconds"] = round(time.monotonic() - started, 3)
        return result

    async def run(request: Request):
        # Reject before the ordinary slot wrapper can release a cached model.
        if not local_request(request):
            raise HTTPException(403, "Prompt bench is available only to the local developer runner.")
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object.")
            params = validate_params(body.get("params"))
            model = get_model(params.get("model_type")) or {}
            if not str(model.get("architecture") or "").startswith("minimax_h3") or model.get("minimax_h3_viggle"):
                raise ValueError("The bench supports H3 Frames/References generation models.")
            validate_params(params, model=model)
            validate_experiment(body.get("experiment", "baseline"))
            validate_workflow(body.get("workflow", "studio"))
        except (ValueError, TypeError) as error:
            raise HTTPException(400, str(error)) from error
        return await run_owned(request)

    api.add_api_route("/api/v1/dev/prompt-bench", info, methods=["GET"], include_in_schema=False)
    api.add_api_route("/api/v1/dev/prompt-bench", run, methods=["POST"], include_in_schema=False)
