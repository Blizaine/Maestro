"""Additive Director controls for neighboring shot anchors and delivery."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool


def _timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return (
        f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},"
        f"{milliseconds:03d}"
    )


def lyrics_to_srt(
    lyrics: Any,
    *,
    timeline_start: float = 0.0,
    duration: float | None = None,
) -> str:
    """Create ordered SRT cues from timed lyric segments, clipped to the edit."""
    if not isinstance(lyrics, list):
        raise ValueError("This song has no timed lyrics to subtitle.")
    try:
        offset = float(timeline_start)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("The song subtitle timeline is invalid.") from exc
    if not math.isfinite(offset):
        raise ValueError("The song subtitle timeline is invalid.")
    try:
        max_duration = float(duration) if duration is not None else None
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("The song subtitle duration is invalid.") from exc
    if max_duration is not None and (
        not math.isfinite(max_duration) or max_duration <= 0
    ):
        raise ValueError("The song subtitle duration is invalid.")

    cues: list[tuple[float, float, str]] = []
    for entry in lyrics:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or entry.get("lyrics") or "").strip()
        if not text:
            continue
        try:
            start = float(entry.get("start")) - offset
            end = float(entry.get("end")) - offset
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        start = max(0.0, start)
        if max_duration is not None:
            end = min(max_duration, end)
        if end <= start:
            continue
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        cues.append((start, end, text))
    cues.sort(key=lambda item: (item[0], item[1]))
    if not cues:
        raise ValueError("This song has no usable timed lyrics to subtitle.")
    return "\n\n".join(
        f"{index}\n{_timestamp(start)} --> {_timestamp(end)}\n{text}"
        for index, (start, end, text) in enumerate(cues, start=1)
    ) + "\n"


def _song_timeline(state: dict) -> tuple[float, float | None]:
    clips = state.get("clips") or []
    planned = [
        clip.get("planned_clip") for clip in clips
        if isinstance(clip, dict) and isinstance(clip.get("planned_clip"), dict)
    ]
    starts = []
    ends = []
    for item in planned:
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start:
            starts.append(start)
            ends.append(end)
    if not starts or not ends:
        return 0.0, None
    first = min(starts)
    return first, max(ends) - first


def _director_state(out_dir: str, pipeline_id: str) -> tuple[dict, str]:
    from services import director_pipeline as pipeline

    state_path = pipeline._find_pipeline_file(out_dir, pipeline_id)
    if not state_path:
        raise ValueError(f"Pipeline {pipeline_id} not found")
    state = pipeline.load_pipeline_state(out_dir, pipeline_id)
    if not isinstance(state, dict):
        raise ValueError(f"Pipeline {pipeline_id} not found")
    return state, os.path.dirname(state_path)


def _set_song_subtitles(out_dir: str, pipeline_id: str, enabled: bool) -> dict:
    from services import director_pipeline as pipeline

    if not isinstance(enabled, bool):
        raise ValueError("Song lyric subtitles must be true or false.")
    if not pipeline._claim_pipeline_operation(pipeline_id):
        raise pipeline.PipelineBusyError(
            "Pipeline is still active; try again shortly."
        )
    try:
        result: dict[str, Any] = {}

        def apply(state: dict) -> None:
            if state.get("pipeline_type") != "music_video":
                if enabled:
                    raise ValueError(
                        "Song lyric subtitles are available for music videos."
                    )
            snapshot = state.get("_params_snapshot") or {}
            if enabled and not isinstance(snapshot.get("lyrics"), list):
                raise ValueError(
                    "This music video has no timed lyric analysis to subtitle."
                )
            state["song_subtitles"] = enabled
            result.update({
                "pipeline_id": pipeline_id,
                "song_subtitles": enabled,
            })

        with pipeline._pipeline_file_lock:
            updated = pipeline._update_saved_pipeline_locked(
                out_dir, pipeline_id, apply,
            )
        if updated is None:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        return result
    finally:
        pipeline._release_pipeline_operation(pipeline_id)


def _lyrics_subtitle_payload(state: dict) -> dict:
    if state.get("pipeline_type") != "music_video":
        raise ValueError("Song lyric subtitles are available for music videos.")
    if state.get("song_subtitles") is not True:
        raise ValueError("Turn on song lyric subtitles before exporting the track.")
    lyrics = (state.get("_params_snapshot") or {}).get("lyrics")
    timeline_start, duration = _song_timeline(state)
    content = lyrics_to_srt(
        lyrics,
        timeline_start=timeline_start,
        duration=duration,
    )
    return {
        "pipeline_id": state.get("pipeline_id"),
        "filename": f"director-{state.get('pipeline_id') or 'song'}-lyrics.srt",
        "content": content,
        "cue_count": content.count(" --> "),
    }


def _pipeline_features(state: dict) -> dict:
    from services import director_pipeline as pipeline

    model_def = {}
    getter = getattr(pipeline._wgp, "get_model_def", None)
    if callable(getter):
        try:
            model_def = getter(state.get("video_model") or "") or {}
        except Exception:
            model_def = {}
    from services.director_video_strategy import (
        BOUNDED_START_END,
        supports_director_seamless,
        video_strategy,
    )

    image_modes = str(model_def.get("image_prompt_types_allowed") or "")
    has_timed_lyrics = isinstance(
        (state.get("_params_snapshot") or {}).get("lyrics"), list,
    )
    return {
        "boundary_anchors": (
            video_strategy(model_def) == BOUNDED_START_END
            and "S" in image_modes
            and "E" in image_modes
        ),
        "motion_audio_continuation": supports_director_seamless(model_def),
        "song_lyric_subtitles": (
            state.get("pipeline_type") == "music_video" and has_timed_lyrics
        ),
        "song_subtitles": state.get("song_subtitles") is True,
        "joined_cut_upscale": False,
    }


def _enqueue_clip_upscale(
    out_dir: str,
    pipeline_id: str,
    clip_index: int,
    body: dict,
    enqueue_media_flow: Callable[[dict], Any] | None,
) -> dict:
    if not callable(enqueue_media_flow):
        raise RuntimeError("Director Delivery is not connected to Media Flow.")
    if not isinstance(body, dict):
        raise ValueError("Upscale settings must be a JSON object.")
    spatial = str(body.get("spatial_upsampling") or "").strip()
    if not spatial:
        raise ValueError("Choose an enabled finishing method to upscale this shot.")
    state, state_dir = _director_state(out_dir, pipeline_id)
    clips = state.get("clips") or []
    if clip_index < 0 or clip_index >= len(clips):
        raise ValueError(f"Clip index {clip_index} out of range")
    clip = clips[clip_index]
    if clip.get("video_stale"):
        raise ValueError("Regenerate this stale shot before upscaling it.")
    filename = clip.get("video_filename")
    if (
        not isinstance(filename, str)
        or not filename
        or os.path.basename(filename) != filename
    ):
        raise ValueError("This shot has no saved video take to upscale.")
    source_path = os.path.realpath(os.path.join(state_dir, filename))
    if os.path.dirname(source_path) != os.path.realpath(state_dir):
        raise ValueError("This shot is outside its Director project folder.")
    if not os.path.isfile(source_path) or os.path.getsize(source_path) <= 0:
        raise ValueError("The selected shot take is missing or empty.")
    options = {
        key: body[key]
        for key in ("dlss_intensity", "dlss_depth", "dlss_motion")
        if key in body
    }
    request = {
        "files": [{"path": source_path, "name": filename}],
        "spatial_upsampling": spatial,
        "temporal_upsampling": "",
        "workspace": state.get("workspace") or "default",
        **options,
    }
    return enqueue_media_flow(request)


def _current_joined_cuts(state: dict, state_dir: str) -> list[dict[str, str]]:
    """Return rejoined cuts whose source take manifest still matches the project."""
    pipeline_id = str(state.get("pipeline_id") or state.get("id") or "")
    clips = state.get("clips") or []
    if not pipeline_id or not isinstance(clips, list) or len(clips) < 2:
        return []

    expected_sources: list[str] = []
    canonical_dir = os.path.realpath(state_dir)
    for clip in clips:
        if not isinstance(clip, dict) or clip.get("video_stale"):
            return []
        filename = clip.get("video_filename")
        if (
            not isinstance(filename, str)
            or not filename
            or os.path.basename(filename) != filename
        ):
            return []
        source_path = os.path.realpath(os.path.join(canonical_dir, filename))
        if (
            os.path.dirname(source_path) != canonical_dir
            or not os.path.isfile(source_path)
            or os.path.getsize(source_path) <= 0
        ):
            return []
        expected_sources.append(filename)

    output_files = state.get("output_files") or []
    candidates = [
        filename for filename in output_files
        if isinstance(filename, str)
        and filename
        and os.path.basename(filename) == filename
        and "_multiclip" in os.path.splitext(filename)[0].lower()
    ]
    valid: list[dict[str, str]] = []
    seen: set[str] = set()
    for filename in reversed(candidates):
        if filename in seen:
            continue
        seen.add(filename)
        extension = os.path.splitext(filename)[1].lower()
        if extension not in {".mp4", ".mkv", ".webm", ".mov"}:
            continue
        cut_path = os.path.realpath(os.path.join(canonical_dir, filename))
        if (
            os.path.dirname(cut_path) != canonical_dir
            or not os.path.isfile(cut_path)
            or os.path.getsize(cut_path) <= 0
        ):
            continue
        meta_path = os.path.splitext(cut_path)[0] + ".meta.json"
        if os.path.dirname(os.path.realpath(meta_path)) != canonical_dir:
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, ValueError, TypeError):
            continue
        if (
            not isinstance(metadata, dict)
            or metadata.get("director_pipeline_id") != pipeline_id
            or metadata.get("director_artifact_kind") != "joined"
            or metadata.get("output_filename") != filename
        ):
            continue
        source_names = metadata.get("director_source_clips")
        if (
            not isinstance(source_names, list)
            or len(source_names) < len(expected_sources)
            or source_names[:len(expected_sources)] != expected_sources
        ):
            continue
        valid.append({"filename": filename})
    return valid


def _enqueue_joined_cut_upscale(
    out_dir: str,
    pipeline_id: str,
    body: dict,
    enqueue_media_flow: Callable[[dict], Any] | None,
) -> dict:
    if not callable(enqueue_media_flow):
        raise RuntimeError("Director Delivery is not connected to Media Flow.")
    if not isinstance(body, dict):
        raise ValueError("Upscale settings must be a JSON object.")
    spatial = str(body.get("spatial_upsampling") or "").strip()
    if not spatial:
        raise ValueError("Choose an enabled finishing method to upscale this cut.")
    state, state_dir = _director_state(out_dir, pipeline_id)
    filename = body.get("filename")
    if not isinstance(filename, str) or not filename:
        raise ValueError("Choose a current joined Director cut to upscale.")
    cut = next(
        (item for item in _current_joined_cuts(state, state_dir)
         if item["filename"] == filename),
        None,
    )
    if cut is None:
        raise ValueError(
            "This joined cut is missing, stale, or no longer matches the current shot takes. Re-join the current shots first."
        )
    source_path = os.path.realpath(os.path.join(state_dir, filename))
    options = {
        key: body[key]
        for key in ("dlss_intensity", "dlss_depth", "dlss_motion")
        if key in body
    }
    request = {
        "files": [{"path": source_path, "name": filename}],
        "spatial_upsampling": spatial,
        "temporal_upsampling": "",
        "workspace": state.get("workspace") or "default",
        **options,
    }
    return enqueue_media_flow(request)


def create_director_showrunner_router(
    *,
    authorize_pipeline: Callable[[str], Any],
    initialize_pipeline: Callable[[], Any],
    get_output_dir: Callable[[], str],
    enqueue_media_flow: Callable[[dict], Any] | None = None,
) -> APIRouter:
    """Build isolated Director routes, delegating upscale work to Media Flow."""
    router = APIRouter()

    def prepare(pid: str) -> str:
        authorize_pipeline(pid)
        initialize_pipeline()
        return get_output_dir()

    @router.get("/api/v1/director/pipelines/{pid}/showrunner")
    async def director_showrunner_info(pid: str):
        out_dir = prepare(pid)
        try:
            state, state_dir = await run_in_threadpool(_director_state, out_dir, pid)
            joined_cuts = await run_in_threadpool(
                _current_joined_cuts, state, state_dir,
            )
            features = _pipeline_features(state)
            features["joined_cut_upscale"] = bool(joined_cuts)
            return {
                "pipeline_id": pid,
                "features": features,
                "joined_cuts": joined_cuts,
            }
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.put("/api/v1/director/pipelines/{pid}/clips/{clip_index}/anchors")
    async def update_director_clip_anchors(
        pid: str, clip_index: int, request: Request,
    ):
        out_dir = prepare(pid)
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("Boundary anchor settings must be a JSON object.")
            from services import director_pipeline as pipeline

            return await run_in_threadpool(
                pipeline.update_clip_boundary_anchors,
                out_dir,
                pid,
                clip_index,
                start_from_previous=body.get("start_from_previous"),
                end_on_next=body.get("end_on_next"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            from services.director_pipeline import PipelineBusyError

            if isinstance(exc, PipelineBusyError):
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise

    @router.put("/api/v1/director/pipelines/{pid}/delivery")
    async def update_director_delivery(pid: str, request: Request):
        out_dir = prepare(pid)
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("Delivery settings must be a JSON object.")
            return await run_in_threadpool(
                _set_song_subtitles,
                out_dir,
                pid,
                body.get("song_subtitles"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            from services.director_pipeline import PipelineBusyError

            if isinstance(exc, PipelineBusyError):
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise

    @router.get("/api/v1/director/pipelines/{pid}/lyrics.srt")
    async def export_director_lyrics(pid: str):
        out_dir = prepare(pid)
        try:
            state, _ = await run_in_threadpool(_director_state, out_dir, pid)
            return _lyrics_subtitle_payload(state)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/v1/director/pipelines/{pid}/clips/{clip_index}/upscale")
    async def upscale_director_clip(
        pid: str, clip_index: int, request: Request,
    ):
        out_dir = prepare(pid)
        try:
            body = await request.json()
            return await run_in_threadpool(
                _enqueue_clip_upscale,
                out_dir,
                pid,
                clip_index,
                body,
                enqueue_media_flow,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.post("/api/v1/director/pipelines/{pid}/joined-cut/upscale")
    async def upscale_director_joined_cut(pid: str, request: Request):
        out_dir = prepare(pid)
        try:
            body = await request.json()
            return await run_in_threadpool(
                _enqueue_joined_cut_upscale,
                out_dir,
                pid,
                body,
                enqueue_media_flow,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return router
