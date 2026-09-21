"""Frozen, resumable enhancement followed by generation in one Studio job."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import re
import socket
import threading
from typing import Callable


_request_context: ContextVar[dict | None] = ContextVar("studio_enhancement", default=None)
_disk_lock = threading.RLock()
SETTING_KEYS = (
    "llm_provider", "llm_model_id", "llm_device", "llm_remote_url",
    "enhance_llm_model_id", "enhance_llm_device", "nsfw_mode",
)


def captured_settings(services: dict) -> dict:
    # Credentials are resolved at execution, never copied into jobs/sidecars.
    defaults = {"llm_provider": "local", "llm_remote_url": "", "llm_device": "cuda",
                "enhance_llm_model_id": "", "enhance_llm_device": "cuda", "nsfw_mode": False}
    result = {key: deepcopy(services.get(key, defaults.get(key))) for key in SETTING_KEYS}
    result["nsfw_mode"] = bool(result["nsfw_mode"])
    if result["enhance_llm_model_id"]:
        result.update(llm_model_id=result["enhance_llm_model_id"],
                      llm_device=result["enhance_llm_device"], llm_provider="local", llm_remote_url="")
    return result


def current_settings(services: dict) -> dict:
    context = _request_context.get()
    return {**services, **{key: value for key, value in context["settings"].items() if value is not None}} if context else services


def check_cancelled() -> None:
    context = _request_context.get()
    if context and context["cancelled"]():
        raise InterruptedError("Prompt enhancement cancelled.")


def is_cancellable() -> bool:
    return _request_context.get() is not None


def record_review_warning(message: str) -> None:
    context = _request_context.get()
    if context and message not in context["warnings"]:
        context["warnings"].append(message)


def enhancement_warnings() -> list[str]:
    context = _request_context.get()
    return list(context["warnings"]) if context else []


def cancellable_lines(response, **kwargs):
    """Keep cancellation responsive even before a server emits its first token.

    Windows can leave a socket read asleep after shutdown(), and close() waits
    on BufferedReader's lock. Isolate only network reads: the job still owns
    its GPU slot until its caller unloads the LLM during cancellation teardown.
    """
    if not is_cancellable():
        yield from response.iter_lines(**kwargs)
        return
    events = Queue(maxsize=32)
    stopped = threading.Event()

    def publish(kind, value=None):
        while not stopped.is_set():
            try:
                events.put((kind, value), timeout=0.1)
                return
            except Full:
                pass

    def read():
        try:
            for line in response.iter_lines(**kwargs):
                if stopped.is_set():
                    break
                publish('line', line)
        except Exception as error:
            publish('error', error)
        finally:
            response.close()
            publish('done')

    reader = threading.Thread(target=read, daemon=True, name='maestro_enhance_stream')
    response._maestro_stream_reader = reader
    reader.start()
    try:
        while True:
            check_cancelled()
            try:
                kind, value = events.get(timeout=0.1)
            except Empty:
                continue
            if kind == 'error':
                raise value
            if kind == 'done':
                break
            yield value
    finally:
        stopped.set()
        if reader.is_alive():
            raw = getattr(response, 'raw', None)
            sock = getattr(getattr(raw, '_connection', None), 'sock', None)
            if sock is None:
                buffered = getattr(getattr(raw, '_fp', None), 'fp', None)
                sock = getattr(getattr(buffered, 'raw', None), '_sock', None)
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        reader.join(timeout=0.1)


def close_response(response) -> None:
    """The network reader owns close while it may still hold the read lock."""
    reader = getattr(response, '_maestro_stream_reader', None)
    if reader is None or not reader.is_alive():
        response.close()


@contextmanager
def enhancement_context(settings: dict, cancelled: Callable[[], bool]):
    token = _request_context.set({"settings": settings, "cancelled": cancelled, "warnings": []})
    try:
        check_cancelled()
        yield
        check_cancelled()
    finally:
        _request_context.reset(token)


def new_enhancement(params: dict, services: dict) -> dict:
    original = deepcopy(params)
    for key in ("_enhance_on_generation", "_deferred_generation_prepare", "_deferred_prompt_enhance"):
        original.pop(key, None)
    return {
        "version": 1, "state": "pending", "original_prompt": str(params.get("prompt") or ""),
        "original_params": original, "settings": captured_settings(services),
        "enhanced_prompt": None, "warnings": [],
    }


def public_enhancement(record: dict | None, *, summary: bool = False) -> dict | None:
    if not record:
        return None
    keys = ["version", "state", "warnings", "error"]
    if not summary:
        keys.extend(["original_prompt", "enhanced_prompt"])
    return {key: deepcopy(record.get(key)) for key in keys}


def enhancement_request(params: dict, model: dict) -> tuple[dict, bool]:
    """Construct the same native writer inputs used by interactive Enhance."""
    h3 = str(model.get("architecture") or "").startswith("minimax_h3")
    omni = bool(model.get("omni_reference"))
    fps = float(model.get("fps") or (24 if h3 else 16))
    total = int(params.get("video_length") or model.get("frames_minimum") or 124)
    window = int((params.get("minimax_h3_sequence_clip_frames") if omni else 0) or 0)
    window = window or int(params.get("sliding_window_size") or model.get("frames_maximum") or total)
    sequence = h3 and total > window and bool(
        params.get("minimax_h3_reference_sequence") if omni else params.get("minimax_h3_multi_window")
    )
    sequence = sequence or bool(model.get("multi_window_sequence_controls") and
        params.get("ltx_multi_window") and total > window)
    images: list[str] = []
    context: list[str] = []
    if omni:
        from services.h3_sequence_planner import _reference_context
        from models.minimax_h3.ref2va import canonicalize_ref2va_reference_order
        references = params.get("minimax_h3_references") or []
        _, references, _ = canonicalize_ref2va_reference_order("", references)
        relationships, retention, _ = _reference_context(references)
        context.extend([relationships, retention])
        images.extend(ref["path"] for ref in references if ref.get("type") == "image" and ref.get("path"))
    else:
        for field, time in (("image_start", "0.00"), ("image_end", f"{total / fps:.2f}")):
            if params.get(field):
                images.append(params[field])
                context.append(f"At {time} seconds, <Picture {len(images)}> is an exact target frame.")
        positions = re.split(r"[\s,]+", str(params.get("frames_positions") or "").strip())
        keyframes = params.get("image_refs") or [] if "KFI" in str(params.get("video_prompt_type") or "") else []
        for path, position in zip(keyframes, positions):
            images.append(path)
            context.append(f"<Picture {len(images)}> is an exact injected frame at timeline position {position}.")
        if params.get("generation_mode") == "image" and not images:
            images.extend(params.get("image_refs") or [])
    payload = {
        "prompt": str(params.get("prompt") or ""),
        "mode": params.get("generation_mode") or "video", "model_type": params.get("model_type"),
        "planning_style": "adaptive" if h3 else "faithful",
        "duration_seconds": total / fps, "window_count": 1, "window_size_seconds": window / fps,
        "image_paths": images, "reference_context": "\n".join(context),
        "activated_loras": deepcopy(params.get("activated_loras") or []),
    }
    return payload, sequence


async def prepare_enhanced_job(params: dict, model: dict, enhance, prepare, *, previous_prepared: dict | None = None) -> dict:
    """Prepare on a copy; callers checkpoint the result only after full success."""
    body = deepcopy(params)
    if model.get("omni_reference"):
        from models.minimax_h3.ref2va import canonicalize_ref2va_reference_order
        body["prompt"], body["minimax_h3_references"], _ = canonicalize_ref2va_reference_order(
            str(body.get("prompt") or ""), body.get("minimax_h3_references") or [])
    for key in ("_enhance_on_generation", "_deferred_generation_prepare", "_deferred_prompt_enhance",
                "h3_window_plan", "h3_window_prompts", "h3_window_plan_signature", "_h3_window_plan_reviewed",
                "_h3_original_prompt", "_ltx_original_prompt"):
        body.pop(key, None)
    payload, sequence = enhancement_request(body, model)
    check_cancelled()
    result_warnings = []
    if sequence:
        body["minimax_h3_sequence_prompt_mode"] = "adaptive"
        body["minimax_h3_window_storyboard"] = True
        body["ltx_window_prompt_mode"] = "auto"
        from services.h3_plan_retry import retryable_windows
        previous_plan = (previous_prepared or {}).get("h3_window_plan")
        if retryable_windows(previous_plan):
            body["_h3_retry_plan"] = deepcopy(previous_plan)
    else:
        result = await enhance(payload)
        check_cancelled()
        result_warnings = result.get("warnings") or []
        text = str(result.get("enhanced") or "").strip()
        if not text:
            raise ValueError("The prompt enhancer returned an empty prompt.")
        body["prompt"] = text
        body["multi_prompts_gen_type"] = 2
        body["minimax_h3_window_storyboard"] = False
        body["minimax_h3_sequence_prompt_mode"] = "manual"
    prepared = await prepare(body, prepare_only=True)
    check_cancelled()
    if not isinstance(prepared.get("params"), dict):
        raise ValueError("Enhancement did not produce valid generation settings.")
    plan = prepared.get("h3_window_plan") or {}
    warnings = list(dict.fromkeys([*(plan.get("planning_warnings") or []), *result_warnings, *enhancement_warnings()]))
    # A fallback draft remains reviewable, but unattended generation must never
    # pass it off as a successful AI enhancement. Timing notes alone are fine.
    if "fallback" in str(plan.get("planned_by") or "") or warnings:
        prepared["enhancement_review_required"] = True
    prepared["enhancement_warnings"] = warnings
    return prepared


class StudioJobArchive:
    """Atomic checkpoints for opted-in Studio jobs, without browser state."""
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def _path(self, job_id: str) -> Path:
        if not job_id or not all(c.isalnum() or c == '-' for c in job_id):
            raise ValueError("Invalid Studio job ID")
        return self.directory / (job_id + ".json")

    def save(self, job: dict, *_args) -> None:
        if not job.get("enhancement"):
            return
        with _disk_lock:
            if job.get("dismissed"):
                self.remove(job["id"])
                return
            path = self._path(job["id"])
            self.directory.mkdir(parents=True, exist_ok=True)
            data = {key: deepcopy(value) for key, value in job.items() if not key.startswith('_')}
            temporary = path.with_suffix('.tmp')
            with temporary.open('w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)

    def recover(self) -> dict:
        result = {}
        for path in self.directory.glob('*.json'):
            try:
                job = json.loads(path.read_text(encoding='utf-8'))
                if self._path(job['id']) != path or not job.get('enhancement'):
                    continue
                if job['status'] in {'queued', 'running'}:
                    # Never rerun a possibly finished diffusion pass after an
                    # unclean shutdown. Run queue explicitly resumes it once.
                    job.update(status='held', phase='', message='Recovered after restart — ready to resume')
                    if job['enhancement']['state'] == 'enhancing':
                        job['enhancement']['state'] = 'pending'
                result[job['id']] = job
            except (ValueError, KeyError, TypeError, OSError):
                continue
        return result

    def remove(self, job_id: str) -> None:
        with _disk_lock:
            self._path(job_id).unlink(missing_ok=True)
