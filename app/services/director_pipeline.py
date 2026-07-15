"""Server-side Director pipeline.

Orchestrates the full Director flow (LLM planning → image gen → video gen)
in a background thread so it can run without the browser being open.

Supports two planning backends:
  - Legacy: direct calls to llm_service (old monolithic approach)
  - New:    DirectorOrchestrator with layered architecture (planners → renderers → validators)

Controlled by feature flags in params or server config.
"""

import os
import re
import time
import json
import uuid
import threading
from typing import Optional

# These will be set by launch.py on startup
_jobs: dict = None          # reference to launch._jobs
_run_generation = None      # reference to launch._run_generation
_wgp = None                 # reference to wgp module
_gen_lock = None            # reference to launch._gen_lock
_active_gen_states = None   # reference to launch._active_gen_states (abort signaling)

_pipelines: dict = {}
_pipeline_lock = threading.Lock()


# ── Reference art-style lock ────────────────────────────────────────────
# Flux Klein only honors a reference's art style when the MEDIUM IS NAMED
# AT THE START of the prompt ("Maintain the same black and white hand
# drawn art style. ..."). A trailing referential anchor ("...preserve the
# art style of the reference image") demonstrably does NOT hold it — the
# output comes back photorealistic. So the pipeline asks the vision LLM
# once per run to NAME the reference's medium concretely, and the phrase
# is prepended to every image prompt deterministically at generation time
# (instead of trusting the 4B planner to follow a guide rule, which it
# provably doesn't do reliably).

_STYLE_DESCRIBE_PROMPT = (
    "Name the visual medium and art style of this image in one short phrase "
    "of 3 to 8 words. Examples: 'black and white hand-drawn pencil sketch', "
    "'watercolor illustration', 'flat-color anime', 'oil painting', "
    "'photorealistic photograph'. Reply with ONLY the phrase, nothing else."
)


def _normalize_style_phrase(raw: str) -> str:
    """Reduce the vision LLM's style answer to a clean, prefix-able phrase.

    Returns "" for photographic references (photorealism is the image
    model's default — a prefix would add nothing) and for answers that
    don't look like a short phrase (refusals, prose, thinking spill).
    """
    s = (raw or "").strip()
    if not s:
        return ""
    s = s.splitlines()[0].strip()
    s = s.strip('"').strip("'").lstrip("-*# ").rstrip(".").strip()
    if not s or len(s) > 80:
        return ""
    low = s.lower()
    if "photo" in low or "realistic" in low:
        return ""
    # Avoid "...style art style" when composing the prefix sentence.
    for suffix in (" art style", " style"):
        if low.endswith(suffix):
            s = s[: -len(suffix)].strip()
            break
    # Mid-sentence position: "Maintain the same simple black line..." —
    # the vision model tends to capitalize its answer.
    if s and s[0].isupper() and (len(s) < 2 or not s[1].isupper()):
        s = s[0].lower() + s[1:]
    return s


def _style_prefix_for(style: str) -> str:
    """The exact lead sentence validated to hold Klein to a medium."""
    style = (style or "").strip()
    return f"Maintain the same {style} art style. " if style else ""


# Motion-photography effects have no place in a START-FRAME prompt — the
# frame must be sharp for the video model to animate from. The music-video
# planner still writes them ("A strong motion blur effect is present on
# the background...") because its energy-focused rules leak into image
# prompts, and Klein complies with an image-wrecking smear. Deterministic
# strip, same philosophy as the style prefix: don't trust the 4B.
_MOTION_EFFECT_RE = re.compile(
    r"motion[- ]?blur|speed[- ]?lines|long[- ]?exposure|camera shake|blur effect",
    re.IGNORECASE,
)


def _strip_motion_effects(prompt: str) -> str:
    """Drop sentences/clauses that request motion-photography effects."""
    if not prompt or not _MOTION_EFFECT_RE.search(prompt):
        return prompt
    parts = re.split(r"(?<=[.;!?])\s+", prompt)
    kept = [s for s in parts if not _MOTION_EFFECT_RE.search(s)]
    cleaned = " ".join(kept).strip()
    return cleaned if cleaned else prompt

# ── Pipeline State Persistence ─────────────────────────────────────────────

PIPELINE_STATE_VERSION = 1
_PIPELINE_FILE_PREFIX = "_director_pipeline_"


def _save_pipeline_state(pid: str):
    """Serialize pipeline state to JSON on disk. Called at phase boundaries."""
    with _pipeline_lock:
        p = _pipelines.get(pid)
        if not p:
            return
        p = dict(p)  # shallow copy for safe access outside lock

    out_dir = p.get("out_dir") or (_wgp.save_path if _wgp else "outputs")
    params = p.get("params", {})

    # Build per-clip state
    clip_plans = p.get("clip_plans", [])
    clip_images = p.get("clip_images", [])
    pre_polish = p.get("_clip_plans_pre_polish", [])
    clip_timings = p.get("_clip_timings", {})

    # Per-clip video filenames. _clip_video_files was never populated by the
    # multi-clip runtime (the generation is one job whose outputs land in
    # output_files: one video per clip, in order, plus the *_multiclip.mp4
    # join), so derive positionally when the count matches. Seamless runs
    # produce a single combined file and never match.
    clip_videos = p.get("_clip_video_files") or []
    if not clip_videos:
        _outs = [f for f in (p.get("output_files") or []) if not f.endswith("_multiclip.mp4")]
        if len(_outs) == len(clip_plans):
            clip_videos = _outs

    clips = []
    for i, plan in enumerate(clip_plans):
        clip_state = {
            "index": i,
            "planned_clip": p.get("_planned_clips", [{}] * (i + 1))[i] if i < len(p.get("_planned_clips", [])) else None,
            "image_prompt": plan.get("image_prompt", ""),
            "video_prompt": plan.get("video_prompt", ""),
            "visual_changes": plan.get("visual_changes", []) or [],
            "image_source": plan.get("image_source", "original"),
            "keyframe_prompts": plan.get("keyframe_prompts", []) or [],
            "window_prompts": plan.get("window_prompts", []) or [],
            "window_count": plan.get("window_count", 1),
            "image_prompt_pre_polish": pre_polish[i].get("image_prompt", "") if i < len(pre_polish) else None,
            "video_prompt_pre_polish": pre_polish[i].get("video_prompt", "") if i < len(pre_polish) else None,
            # Per-window and per-keyframe pre-polish snapshots so the
            # Dashboard can show before/after diffs for windowed shots
            # (≥21s) and for keyframe prompts. Without these, windowed
            # shots showed no polish diff because video_prompt is
            # skipped by Pass 3 when window_prompts exist (its content
            # is unused at generation time anyway).
            "window_prompts_pre_polish": pre_polish[i].get("window_prompts", []) if i < len(pre_polish) else None,
            "keyframe_prompts_pre_polish": pre_polish[i].get("keyframe_prompts", []) if i < len(pre_polish) else None,
            "start_image_filename": clip_images[i] if i < len(clip_images) else None,
            "keyframe_filenames": (p.get("_clip_keyframes", []) or [])[i] if i < len(p.get("_clip_keyframes", [])) else [],
            "video_filename": clip_videos[i] if i < len(clip_videos) else None,
            "tag": (p.get("_clip_tags", []) or [])[i] if i < len(p.get("_clip_tags", [])) else None,
            "image_gen_time_sec": clip_timings.get(f"image_{i}"),
            "video_gen_time_sec": clip_timings.get(f"video_{i}"),
        }
        clips.append(clip_state)

    state = {
        "version": PIPELINE_STATE_VERSION,
        "pipeline_id": pid,
        "created_at": p.get("created_at"),
        "completed_at": p.get("_completed_at"),
        "status": p.get("status", "unknown"),
        "pipeline_type": params.get("pipeline_type", "music_video"),
        "scene_description": params.get("scene_description", ""),
        "reference_image_path": params.get("reference_image_path"),
        "character_ref_paths": params.get("character_ref_paths", []),
        "location_ref_paths": params.get("location_ref_paths", []),
        "auto_mode": params.get("auto_mode", True),
        "seamless": params.get("seamless", True),
        "image_model": params.get("image_model", ""),
        "video_model": params.get("video_model", ""),
        "image_loras": params.get("image_loras", {}),
        "video_loras": params.get("video_loras", {}),
        "image_params": params.get("image_params", {}),
        "video_params": params.get("video_params", {}),
        "llm_log": p.get("_llm_log"),
        "clips": clips,
        "output_files": p.get("output_files", []),
        "total_time_sec": (time.time() - p["created_at"]) if p.get("created_at") else None,
        # Full original request params, verbatim (it's the JSON dict the
        # endpoint received, so it's serializable). This is what makes a
        # crashed pipeline faithfully resumable — music-video mode in
        # particular depends on the analyzed audio track, character list, and
        # per-clip frame counts that the flattened per-clip state above does
        # not carry. resume_pipeline() rehydrates from here.
        "_params_snapshot": params,
    }

    try:
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, f"{_PIPELINE_FILE_PREFIX}{pid}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"[Pipeline] Failed to save state for {pid}: {e}")


def list_pipeline_states(out_dir: str) -> list[dict]:
    """Scan directory for saved pipeline state files. Returns summary list."""
    results = []
    if not os.path.isdir(out_dir):
        return results
    # Scan top-level and workspace subdirectories
    dirs_to_scan = [out_dir]
    for name in os.listdir(out_dir):
        sub = os.path.join(out_dir, name)
        if os.path.isdir(sub):
            dirs_to_scan.append(sub)

    for scan_dir in dirs_to_scan:
        for fname in os.listdir(scan_dir):
            if fname.startswith(_PIPELINE_FILE_PREFIX) and fname.endswith(".json"):
                try:
                    filepath = os.path.join(scan_dir, fname)
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # Detect stale "running" pipelines — if the JSON says running
                    # but there's no active in-memory pipeline, it crashed
                    status = data.get("status", "unknown")
                    pid = data.get("pipeline_id", "")
                    if status == "running" and pid not in _pipelines:
                        status = "crashed"
                        data["status"] = "crashed"
                        try:
                            with open(filepath, "w", encoding="utf-8") as fw:
                                json.dump(data, fw, indent=2, ensure_ascii=False, default=str)
                        except Exception:
                            pass
                    results.append({
                        "id": pid,
                        "status": status,
                        "pipeline_type": data.get("pipeline_type", ""),
                        "created_at": data.get("created_at"),
                        "clip_count": len(data.get("clips", [])),
                        "output_count": len(data.get("output_files", [])),
                        "scene_description": (data.get("scene_description", "") or "")[:100],
                        "workspace": os.path.basename(scan_dir) if scan_dir != out_dir else "default",
                        "_filepath": filepath,
                    })
                except Exception:
                    pass
    results.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
    return results


def _backfill_clip_video_filenames(state: dict, state_dir: str) -> dict:
    """Derive per-clip video filenames from output_files when absent.

    Multi-clip (non-seamless) runs produce one video per clip, in clip
    order, plus a trailing *_multiclip.mp4 join — but the runtime never
    recorded them per clip (_clip_video_files was a dead key), leaving
    every clip's video_filename null. That made the Dashboard count all
    clips as "missing" and broke Rejoin (needs >= 2 per-clip files).
    Fill only null entries (a rerun clip's filename must survive), only
    when the per-clip count matches exactly, and only for files that
    still exist next to the pipeline file. Seamless runs (one combined
    output) never match the count and are left untouched.
    """
    clips = state.get("clips") or []
    outputs = [f for f in (state.get("output_files") or []) if not f.endswith("_multiclip.mp4")]
    if not clips or len(outputs) != len(clips):
        return state
    for i, clip in enumerate(clips):
        if not clip.get("video_filename") and os.path.isfile(os.path.join(state_dir, outputs[i])):
            clip["video_filename"] = outputs[i]
    return state


def load_pipeline_state(out_dir: str, pid: str) -> Optional[dict]:
    """Load a saved pipeline state by ID. Searches out_dir and subdirectories."""
    target = f"{_PIPELINE_FILE_PREFIX}{pid}.json"
    # Search top-level
    filepath = os.path.join(out_dir, target)
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return _backfill_clip_video_filenames(json.load(f), out_dir)
    # Search subdirectories (workspaces)
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            sub = os.path.join(out_dir, name, target)
            if os.path.isfile(sub):
                with open(sub, "r", encoding="utf-8") as f:
                    return _backfill_clip_video_filenames(json.load(f), os.path.join(out_dir, name))
    return None


def update_clip_tag(out_dir: str, pid: str, clip_index: int, tag: Optional[str]) -> bool:
    """Update the tag on a specific clip in a saved pipeline state."""
    state = load_pipeline_state(out_dir, pid)
    if not state:
        return False
    clips = state.get("clips", [])
    if clip_index < 0 or clip_index >= len(clips):
        return False
    clips[clip_index]["tag"] = tag

    # Find and overwrite the file
    target = f"{_PIPELINE_FILE_PREFIX}{pid}.json"
    for search_dir in [out_dir] + [os.path.join(out_dir, d) for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]:
        filepath = os.path.join(search_dir, target)
        if os.path.isfile(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False, default=str)
            return True
    return False


def _find_pipeline_file(out_dir: str, pid: str) -> Optional[str]:
    """Find the JSON file path for a saved pipeline."""
    target = f"{_PIPELINE_FILE_PREFIX}{pid}.json"
    filepath = os.path.join(out_dir, target)
    if os.path.isfile(filepath):
        return filepath
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            sub = os.path.join(out_dir, name, target)
            if os.path.isfile(sub):
                return sub
    return None


def _update_saved_pipeline(out_dir: str, pid: str, updater) -> Optional[dict]:
    """Load a saved pipeline, apply an updater function, save back, and return the state."""
    filepath = _find_pipeline_file(out_dir, pid)
    if not filepath:
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        state = json.load(f)
    updater(state)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)
    return state


def rerun_clip_image(out_dir: str, pid: str, clip_index: int, prompt_override: str = None) -> dict:
    """Re-generate the start image for a single clip. Returns {job_id, filename} or raises."""
    state = load_pipeline_state(out_dir, pid)
    if not state:
        raise ValueError(f"Pipeline {pid} not found")
    clips = state.get("clips", [])
    if clip_index < 0 or clip_index >= len(clips):
        raise ValueError(f"Clip index {clip_index} out of range (0-{len(clips)-1})")

    clip = clips[clip_index]
    prompt = prompt_override or clip.get("image_prompt", "")
    if not prompt:
        raise ValueError("No image prompt for this clip")

    # Reference art-style lock: reruns re-apply the detected style prefix
    # (the pipeline prepends it at generation time, so the saved
    # image_prompt does not carry it). Motion-effect strip mirrors
    # _gen_image for the same reason.
    prompt = _strip_motion_effects(prompt)
    _style_prefix = _style_prefix_for((state.get("_params_snapshot") or {}).get("_reference_style") or "")
    if _style_prefix and not prompt.lower().startswith("maintain the same"):
        prompt = _style_prefix + prompt

    # Get image gen params from the saved pipeline state
    image_model = state.get("image_model") or "flux2_klein_9b"
    image_loras = state.get("image_loras") or {}
    image_params = state.get("image_params") or {}
    ref_path = state.get("reference_image_path") or ""

    # Build refs: main + character + location
    all_refs = []
    if ref_path and os.path.isfile(ref_path):
        all_refs.append(ref_path)
    for cp in (state.get("character_ref_paths") or []):
        if cp and os.path.isfile(cp):
            all_refs.append(cp)
    for lp in (state.get("location_ref_paths") or []):
        if lp and os.path.isfile(lp):
            all_refs.append(lp)

    # Determine the output directory from where the pipeline file lives
    pipeline_file = _find_pipeline_file(out_dir, pid)
    clip_out_dir = os.path.dirname(pipeline_file) if pipeline_file else out_dir

    gen_params = {
        "model_type": image_model,
        "prompt": prompt,
        "image_refs": all_refs if all_refs else [ref_path],
        "image_mode": 1,
        "image_prompt_type": "",
        "num_inference_steps": image_params.get("num_inference_steps", 8),
        "guidance_scale": image_params.get("guidance_scale", 1),
        "video_prompt_type": "KI",
        "resolution": image_params.get("resolution", "1280x720"),
        "seed": -1,
        "settings_version": 2.52,
        "generation_mode": "image",
        "repeat_generation": 1,
        "negative_prompt": "",
        "video_length": 1,
        "activated_loras": image_loras.get("activated_loras", []),
        "loras_multipliers": " ".join(
            m.split(";")[0] for m in (image_loras.get("loras_multipliers", "") or "").split(" ") if m
        ),
        "_director_pipeline_id": pid,
    }

    output_files = _submit_and_wait(gen_params, timeout_s=600, out_dir=clip_out_dir)
    new_filename = output_files[0] if output_files else ""

    if new_filename:
        # Update the saved pipeline state
        def _update(s):
            s["clips"][clip_index]["start_image_filename"] = new_filename
            if prompt_override:
                s["clips"][clip_index]["image_prompt"] = prompt_override
        _update_saved_pipeline(out_dir, pid, _update)

    return {"filename": new_filename, "clip_index": clip_index}


def _slice_audio_segment(src_path: str, start_sec: float, duration_sec: float, dst_path: str) -> None:
    """Cut [start, start+duration] out of the source audio with ffmpeg.

    Mirrors shared/utils/audio_video.py's plain-subprocess ffmpeg usage.
    Output is normalized wav so the generation's audio loader never has to
    care what container the song came in.
    """
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{max(0.0, float(start_sec)):.3f}",
        "-t", f"{max(0.1, float(duration_sec)):.3f}",
        "-i", src_path,
        "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
        dst_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def rerun_clip_video(out_dir: str, pid: str, clip_index: int, prompt_override: str = None) -> dict:
    """Re-generate the video for a single clip. Returns {job_id, filename} or raises."""
    state = load_pipeline_state(out_dir, pid)
    if not state:
        raise ValueError(f"Pipeline {pid} not found")
    clips = state.get("clips", [])
    if clip_index < 0 or clip_index >= len(clips):
        raise ValueError(f"Clip index {clip_index} out of range (0-{len(clips)-1})")

    clip = clips[clip_index]
    prompt = prompt_override or clip.get("video_prompt", "")
    if not prompt:
        raise ValueError("No video prompt for this clip")

    video_model = state.get("video_model") or "ltx2_22B_distilled_1_1"
    video_loras = state.get("video_loras") or {}
    video_params = state.get("video_params") or {}

    # Determine the output directory
    pipeline_file = _find_pipeline_file(out_dir, pid)
    clip_out_dir = os.path.dirname(pipeline_file) if pipeline_file else out_dir

    # Build start image path
    start_img = clip.get("start_image_filename")
    start_path = os.path.join(clip_out_dir, start_img) if start_img else ""
    has_start = start_path and os.path.isfile(start_path)

    # Use planned_clip for duration
    planned = clip.get("planned_clip") or {}
    duration_sec = planned.get("duration_sec", planned.get("end", 20) - planned.get("start", 0))
    if duration_sec <= 0:
        duration_sec = 20
    fps = 25  # LTX-2 default
    video_length = int(duration_sec * fps)

    gen_params = {
        "model_type": video_model,
        "prompt": prompt,
        "image_mode": 0,
        "image_prompt_type": "S" if has_start else "",
        "num_inference_steps": video_params.get("num_inference_steps", 8),
        "guidance_scale": video_params.get("guidance_scale", 1),
        "resolution": video_params.get("resolution", "1280x720"),
        "video_length": video_length,
        # One clip = ONE window — same convention as the original pipeline
        # (see the sliding_window_frames comment there): the window must be
        # STRICTLY greater than the clip's frame count after wgp's latent
        # quantization, or wgp splits the clip into multiple windows saved
        # as SEPARATE files and this rerun records only the first one (a
        # 13s clip came back as its first 5s, shifting every later clip in
        # the rejoined video and breaking lip sync). Without this key the
        # primary-settings default (129 frames) applied.
        "sliding_window_size": video_length + 8 + 1,
        "seed": -1,
        "settings_version": 2.52,
        "generation_mode": "video",
        "repeat_generation": 1,
        "negative_prompt": "",
        "activated_loras": video_loras.get("activated_loras", []),
        "loras_multipliers": " ".join(
            m.split(";")[0] for m in (video_loras.get("loras_multipliers", "") or "").split(" ") if m
        ),
        "_director_pipeline_id": pid,
    }
    if has_start:
        gen_params["image_start"] = start_path

    # Soundtrack conditioning. The original pipeline run passes the FULL
    # song as audio_guide (audio_prompt_type "A") and wgp slices it across
    # clips internally — a single-clip rerun gets none of that context, so
    # without this block the model invents its own audio and the
    # regenerated clip no longer matches the music video's soundtrack.
    # Slice the song to this clip's window and condition on it, mirroring
    # the segment the clip was originally generated against.
    snapshot = state.get("_params_snapshot") or {}
    pipeline_type = state.get("pipeline_type") or snapshot.get("pipeline_type") or "music_video"
    audio_path = snapshot.get("audio_path") or ""
    clip_start = planned.get("start")
    if pipeline_type != "short_film_story" and audio_path and os.path.isfile(audio_path) and clip_start is not None:
        slice_path = os.path.join(clip_out_dir, f"_rerun_audio_c{clip_index}.wav")
        try:
            _slice_audio_segment(audio_path, clip_start, duration_sec, slice_path)
            gen_params["audio_prompt_type"] = "A"
            gen_params["audio_guide"] = slice_path
            if snapshot.get("audio_scale") is not None:
                gen_params["audio_scale"] = snapshot["audio_scale"]
            print(f"[Pipeline {pid}] Clip {clip_index} rerun conditioned on song segment "
                  f"{float(clip_start):.1f}s-{float(clip_start) + float(duration_sec):.1f}s")
        except Exception as e:
            print(f"[Pipeline {pid}] Clip {clip_index} audio slice failed; "
                  f"regenerating without soundtrack conditioning: {e}")

    output_files = _submit_and_wait(gen_params, timeout_s=3600, out_dir=clip_out_dir)
    # Sliding-window generations save CUMULATIVE progress files (each save
    # is the video so far) — the LAST file is the complete clip. With the
    # single-window sizing above there is normally exactly one file, but
    # taking the last is correct in every case; taking the first recorded
    # a 5s preview of a 13s clip.
    new_filename = output_files[-1] if output_files else ""

    if new_filename:
        def _update(s):
            s["clips"][clip_index]["video_filename"] = new_filename
            if prompt_override:
                s["clips"][clip_index]["video_prompt"] = prompt_override
        _update_saved_pipeline(out_dir, pid, _update)

    return {"filename": new_filename, "clip_index": clip_index}


def rejoin_clips(out_dir: str, pid: str) -> dict:
    """Re-join all clips from a saved pipeline using current best versions. Returns {filename}."""
    state = load_pipeline_state(out_dir, pid)
    if not state:
        raise ValueError(f"Pipeline {pid} not found")

    pipeline_file = _find_pipeline_file(out_dir, pid)
    clip_out_dir = os.path.dirname(pipeline_file) if pipeline_file else out_dir

    clips = state.get("clips", [])
    video_files = []
    for clip in clips:
        vf = clip.get("video_filename")
        if vf:
            full_path = os.path.join(clip_out_dir, vf)
            if os.path.isfile(full_path):
                video_files.append(full_path)

    if len(video_files) < 2:
        raise ValueError(f"Need at least 2 video clips to rejoin, found {len(video_files)}")

    # Lay the pristine source song over the rejoined video, exactly like the
    # original pipeline's multiclip join does — per-clip embedded audio is a
    # windowed generation, the full track is the real soundtrack. Story-mode
    # pipelines (no song) concat with the clips' own audio.
    snapshot = state.get("_params_snapshot") or {}
    audio_path = snapshot.get("audio_path") or None
    if audio_path and not os.path.isfile(audio_path):
        audio_path = None

    import time as _time
    timestamp = _time.strftime("%Y-%m-%d-%Hh%Mm%Ss")
    output_name = f"{timestamp}_rejoin_multiclip.mp4"
    output_path = os.path.join(clip_out_dir, output_name)

    try:
        # concatenate_multi_clip_videos is the join the original pipeline
        # uses (ffmpeg concat FILTER, re-encodes to a uniform format). The
        # previously-called wgp.concatenate_videos never existed — this path
        # was unreachable until the video_filename backfill fix, so the
        # AttributeError only surfaced now.
        ok = _wgp.concatenate_multi_clip_videos(video_files, output_path, audio_path)
        if not ok or not os.path.isfile(output_path):
            raise RuntimeError("ffmpeg concatenation failed (see server log for the clip that broke it)")
        print(f"[Pipeline] Rejoined {len(video_files)} clips → {output_name}")

        # Update pipeline state
        def _update(s):
            if output_name not in s.get("output_files", []):
                s.setdefault("output_files", []).append(output_name)
        _update_saved_pipeline(out_dir, pid, _update)

        return {"filename": output_name}
    except Exception as e:
        raise RuntimeError(f"Rejoin failed: {e}")


def init(jobs_dict, run_gen_fn, wgp_module, gen_lock=None, active_gen_states=None):
    """Called by launch.py to wire up shared references."""
    global _jobs, _run_generation, _wgp, _gen_lock, _active_gen_states
    _jobs = jobs_dict
    _run_generation = run_gen_fn
    _wgp = wgp_module
    _gen_lock = gen_lock
    _active_gen_states = active_gen_states


def _submit_and_wait(params: dict, timeout_s: float = 600, workspace: str = None, out_dir: str = None) -> list[str]:
    """Submit a generation job and block until it completes.

    Returns list of output filenames. Raises on failure/timeout.
    """
    job_id = uuid.uuid4().hex[:8]
    job = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "step": 0,
        "total_steps": 0,
        "phase": "",
        "message": "Queued",
        "created_at": time.time(),
        "params": params,
        "output_files": [],
        "error": None,
        "workspace": workspace,
        "out_dir": out_dir,
    }
    _jobs[job_id] = job

    # Run generation in a separate thread (it acquires _gen_lock internally)
    # Non-daemon so the process stays alive if browser disconnects mid-generation
    thread = threading.Thread(target=_run_generation, args=(job_id,), daemon=False)
    thread.start()

    # Wait for completion, mirroring job progress to pipeline status
    deadline = time.time() + timeout_s
    _dir_pid = params.get("_director_pipeline_id")
    _abort_signalled = False
    while time.time() < deadline:
        j = _jobs.get(job_id)
        if not j:
            raise RuntimeError("Job disappeared")
        if j["status"] == "completed":
            return j.get("output_files", [])
        if j["status"] == "cancelled":
            # Keep whatever clips finished before the abort (multi-clip
            # jobs accrue output_files per clip) — callers tolerate a
            # partial or empty list and check the pipeline status.
            print(f"[Pipeline] Job {job_id} cancelled")
            return j.get("output_files", [])
        if j["status"] == "failed":
            err = j.get("error") or "Generation failed"
            print(f"[Pipeline] Job {job_id} failed: {err}")
            raise RuntimeError(err)
        # Backstop for stop_pipeline's abort: if the pipeline was cancelled
        # while this job runs (e.g. the job was submitted in the window
        # after the stop endpoint scanned _jobs), signal abort from here.
        if _dir_pid and not _abort_signalled:
            with _pipeline_lock:
                _cancelled = _pipelines.get(_dir_pid, {}).get("status") == "cancelled"
            if _cancelled:
                _abort_pipeline_jobs(_dir_pid)
                _abort_signalled = True
        # Mirror denoising step progress to pipeline status
        # Only update step/total_steps and message — preserve current/total for pipeline-level counts
        if _dir_pid and (j.get("step", 0) > 0 or j.get("total_steps", 0) > 0):
            with _pipeline_lock:
                p = _pipelines.get(_dir_pid)
                if p and "progress" in p:
                    p["progress"]["step"] = j.get("step", 0)
                    p["progress"]["total_steps"] = j.get("total_steps", 0)
                    p["progress"]["message"] = j.get("phase") or j.get("message") or "Generating..."
        time.sleep(1)

    raise RuntimeError("Generation timed out")


def _update_pipeline(pid: str, **kwargs):
    """Thread-safe update of pipeline state."""
    with _pipeline_lock:
        if pid in _pipelines:
            _pipelines[pid].update(kwargs)


def start_pipeline(params: dict) -> str:
    """Start a new director pipeline. Returns pipeline_id."""
    pid = uuid.uuid4().hex[:8]

    # Capture workspace at submission time — not at execution time
    workspace = params.pop("workspace", None)
    if workspace:
        # Resolve the output directory now, while we know the intended workspace
        from launch import _workspace_dir
        out_dir = _workspace_dir(workspace)
        print(f"[Pipeline] Workspace={workspace}, out_dir={out_dir}, wgp.save_path={_wgp.save_path}")
    else:
        out_dir = _wgp.save_path
        workspace = None
        print(f"[Pipeline] No workspace, using wgp.save_path={out_dir}")

    pipeline = {
        "id": pid,
        "status": "running",
        "phase": "planning",
        "auto_mode": params.get("auto_mode", True),
        "progress": {"current": 0, "total": 0, "message": "Starting...", "step": 0, "total_steps": 0},
        "clip_plans": [],
        "clip_images": [],         # filenames of generated start images
        "output_files": [],
        "error": None,
        "created_at": time.time(),
        "params": params,
        "pause_reason": None,
        "workspace": workspace,
        "out_dir": out_dir,
        # For LLM streaming: the frontend polls /api/v1/llm/stream-status
        "llm_streaming": False,
    }

    with _pipeline_lock:
        _pipelines[pid] = pipeline

    # Non-daemon so pipeline survives browser disconnect during overnight runs
    thread = threading.Thread(target=_run_pipeline, args=(pid,), daemon=False)
    thread.start()

    return pid


def get_pipeline(pid: str) -> Optional[dict]:
    with _pipeline_lock:
        p = _pipelines.get(pid)
        return dict(p) if p else None


def continue_pipeline(pid: str, updates: Optional[dict] = None):
    """Resume a paused pipeline, optionally with updated clip_plans."""
    with _pipeline_lock:
        p = _pipelines.get(pid)
        if not p or p["status"] != "paused":
            return False
        if updates:
            if "clip_plans" in updates:
                p["clip_plans"] = updates["clip_plans"]
        p["status"] = "running"
        p["pause_reason"] = None
    return True


def _find_pipeline_state_file(pid: str, out_dir: str) -> Optional[str]:
    """Locate a saved pipeline JSON by id under out_dir or a workspace subdir."""
    fname = f"{_PIPELINE_FILE_PREFIX}{pid}.json"
    candidates = [os.path.join(out_dir, fname)]
    try:
        for name in os.listdir(out_dir):
            sub = os.path.join(out_dir, name)
            if os.path.isdir(sub):
                candidates.append(os.path.join(sub, fname))
    except OSError:
        pass
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def resume_pipeline(pid: str, out_dir: str) -> tuple[bool, str]:
    """Rehydrate a crashed pipeline from disk and re-run it.

    Reuses the planning (and start images, when their files still exist)
    that completed before the crash; only the video phase re-runs. Returns
    (ok, message). Requires a state file that carries the full params
    snapshot (written since the resume feature shipped) — older crash files
    can't be resumed faithfully and report so.
    """
    with _pipeline_lock:
        existing = _pipelines.get(pid)
        if existing and existing.get("status") in ("running", "queued", "planning"):
            return False, "Pipeline is already running."

    state_path = _find_pipeline_state_file(pid, out_dir)
    if not state_path:
        return False, "No saved state found for this pipeline."
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, f"Could not read saved pipeline state: {e}"

    params = data.get("_params_snapshot")
    if not isinstance(params, dict):
        return False, (
            "This pipeline was created before resume support and can't be "
            "resumed — start a new generation."
        )

    # Rebuild the generation-driving structures from the saved per-clip state.
    saved_clips = data.get("clips", []) or []
    clip_plans = [{
        "image_prompt": c.get("image_prompt", ""),
        "video_prompt": c.get("video_prompt", ""),
        "visual_changes": c.get("visual_changes", []) or [],
        "image_source": c.get("image_source", "original"),
        "keyframe_prompts": c.get("keyframe_prompts", []) or [],
        "window_prompts": c.get("window_prompts", []) or [],
        "window_count": c.get("window_count", 1),
    } for c in saved_clips]
    planned_clips = [c.get("planned_clip") for c in saved_clips]
    clip_images = [c.get("start_image_filename") for c in saved_clips]
    clip_keyframes = [c.get("keyframe_filenames", []) or [] for c in saved_clips]

    workspace = data.get("workspace") if data.get("workspace") not in ("default", None) else None
    resume_out_dir = os.path.dirname(state_path)

    pipeline = {
        "id": pid,
        "status": "running",
        "phase": "resuming",
        "auto_mode": params.get("auto_mode", True),
        "progress": {"current": 0, "total": 0, "message": "Resuming…", "step": 0, "total_steps": 0},
        "clip_plans": clip_plans,
        "_planned_clips": planned_clips,
        "clip_images": clip_images,
        "_clip_keyframes": clip_keyframes,
        "output_files": data.get("output_files", []) or [],
        "_llm_log": data.get("llm_log"),
        "error": None,
        "created_at": data.get("created_at") or time.time(),
        "params": params,
        "pause_reason": None,
        "workspace": workspace,
        "out_dir": resume_out_dir,
        "llm_streaming": False,
    }
    with _pipeline_lock:
        _pipelines[pid] = pipeline

    thread = threading.Thread(target=_run_pipeline, args=(pid,), kwargs={"resume": True}, daemon=False)
    thread.start()
    return True, "resumed"


def _abort_pipeline_jobs(pid: str):
    """Signal wgp abort for this pipeline's queued/running generation jobs.

    Mirrors the Studio cancel endpoint (launch.cancel_job): flip the job's
    gen-state abort flag and the model's _interrupt so the denoise loop
    stops within a step. Without this, Stop only takes effect at the next
    phase/clip boundary — the in-flight clip runs to completion, 10+
    minutes of GPU work after the user pressed Stop on slower cards.
    """
    if not _jobs:
        return
    for job_id, job in list(_jobs.items()):
        params = job.get("params") or {}
        if params.get("_director_pipeline_id") != pid:
            continue
        status = job.get("status")
        if status == "queued":
            job["status"] = "cancelled"
            job["message"] = "Cancelled"
        elif status == "running":
            gen = (_active_gen_states or {}).get(job_id)
            if gen is not None:
                gen["abort"] = True
            model = getattr(_wgp, "wan_model", None) if _wgp is not None else None
            if model is not None and hasattr(model, "_interrupt"):
                model._interrupt = True
            print(f"[Pipeline {pid}] Abort signalled for in-flight job {job_id}")


def stop_pipeline(pid: str):
    with _pipeline_lock:
        p = _pipelines.get(pid)
        if p:
            p["status"] = "cancelled"
    _abort_pipeline_jobs(pid)


def _run_pipeline(pid: str, resume: bool = False):
    """Main pipeline thread — runs the full Director flow.

    When resume=True the pipeline was rehydrated from a crashed state
    (see resume_pipeline): planning + prompt-polish are skipped when the
    saved clip_plans are present, and start-image generation is skipped
    when the saved images still exist on disk. Only the (atomic) video
    generation phase re-runs — so a crash 2 hours into a run doesn't
    throw away the LLM planning that already succeeded.
    """
    try:
        p = _pipelines[pid]
        params = p["params"]
        pipeline_out_dir = p.get("out_dir") or _wgp.save_path
        pipeline_workspace = p.get("workspace")

        # Work already completed before a crash (empty on a fresh run).
        resume_plans = (p.get("clip_plans") or None) if resume else None
        resume_images = (p.get("clip_images") or None) if resume else None

        pipeline_type = params.get("pipeline_type", "music_video")  # music_video | short_film_audio | short_film_story
        auto_mode = params.get("auto_mode", True)

        # ── Disk preflight ─────────────────────────────────────────────
        # A Director run writes gigabytes (per-clip images + video + the
        # final concat). Fail fast with a clear message instead of dying
        # halfway through with a truncated "No space left on device" write.
        try:
            import shutil as _shutil
            free_gb = _shutil.disk_usage(pipeline_out_dir).free / (1024 ** 3)
            if free_gb < 3:
                raise RuntimeError(
                    f"Only {free_gb:.1f} GB free on the output drive — not "
                    f"enough for a Director run. Free up space and try again."
                )
        except RuntimeError:
            raise
        except Exception:
            pass  # disk_usage can fail on odd mounts; don't block on the check itself

        # ── Wait for GPU if jobs are running ────────────────────────────
        # LLM needs GPU (CUDA), so we must wait for generation queue to drain.
        # In auto mode this is expected (fire-and-forget). In non-auto mode
        # the user is waiting interactively, so we still wait but they can cancel.
        if not _wait_for_gpu(pid):
            return  # cancelled while waiting

        # ── Phase 1: LLM Planning ──────────────────────────────────────
        _update_pipeline(pid, phase="planning", llm_streaming=True,
                         progress={"current": 0, "total": 1, "message": "Planning with LLM...", "step": 0, "total_steps": 0})

        planning_start = time.time()
        if resume_plans:
            # Reuse the planning that already succeeded before the crash.
            clip_plans = resume_plans
            planned_clips = p.get("_planned_clips") or []
            print(f"[Pipeline {pid}] Resume: reusing {len(clip_plans)} planned clips — skipping LLM planning + polish")
        else:
            try:
                clip_plans, planned_clips = _run_planning(pid, params, pipeline_type)
            except Exception as plan_err:
                print(f"[Pipeline] Planning error: {plan_err}")
                import traceback
                traceback.print_exc()
                raise
        planning_time = time.time() - planning_start

        if not clip_plans:
            raise RuntimeError("Planning produced no clip plans")

        # Store planned clips for persistence
        _update_pipeline(pid, _planned_clips=planned_clips)

        # Capture LLM logs — collect all passes from the pipeline's accumulated log
        try:
            from services import llm_service
            # The pipeline accumulates logs via _append_llm_log during planning
            accumulated = _pipelines.get(pid, {}).get("_llm_passes", [])
            # Also capture the final state as a fallback
            if not accumulated:
                accumulated = [{
                    "pass": "planning",
                    "system_prompt": getattr(llm_service, '_last_system_prompt', '') or '',
                    "user_prompt": getattr(llm_service, '_last_user_prompt', '') or '',
                    "response_text": getattr(llm_service, '_stream_buffer', '') or '',
                    "thinking_text": getattr(llm_service, '_last_thinking_text', None),
                }]
            llm_log = {
                "provider": params.get("llm_provider", "local"),
                "model_id": params.get("llm_model_id", ""),
                "passes": accumulated,
                # Keep flat fields for backward compat — use last pass
                "system_prompt": accumulated[-1].get("system_prompt", "") if accumulated else "",
                "response_text": accumulated[-1].get("response_text", "") if accumulated else "",
                "thinking_text": accumulated[-1].get("thinking_text") if accumulated else None,
                "planning_time_sec": round(planning_time, 2),
            }
            # On resume, keep the rehydrated original log instead of clobbering
            # it with an empty re-capture (there was no fresh planning stream).
            if not resume_plans:
                _update_pipeline(pid, _llm_log=llm_log)
        except Exception:
            pass

        # ── Optional: Third-pass prompt polish ────────────────────────
        services = _wgp.server_config.get("services", {}) if _wgp else {}
        # Default "third_pass" — Pass 3 polish runs each generated prompt
        # through a model-specific dialect pass after planning, which
        # produces materially better output than relying on Pass 2 alone
        # with a single hardcoded dialect.
        polish_mode = services.get("director_prompt_polish", "third_pass")

        # Snapshot pre-polish prompts for comparison
        import copy
        _update_pipeline(pid, _clip_plans_pre_polish=copy.deepcopy(clip_plans))

        # On resume the saved clip_plans are ALREADY polished — re-polishing
        # would compound edits and drift the prompts, so skip the whole block.
        if resume_plans:
            pass
        elif polish_mode == "third_pass" and clip_plans:
            _update_pipeline(pid, phase="polishing_prompts", llm_streaming=False,
                             progress={"current": 0, "total": len(clip_plans), "message": "Polishing prompts (3rd pass)...", "step": 0, "total_steps": 0})
            try:
                from services.director.prompt_polish import polish_prompts_third_pass
                provider = services.get("llm_provider", "local")
                nsfw = services.get("nsfw_mode", False) and provider not in {"openai", "anthropic"}
                video_model = params.get("video_model", "")
                image_model = params.get("image_model", "")
                video_loras = (params.get("video_loras") or {}).get("activated_loras", [])
                image_loras = (params.get("image_loras") or {}).get("activated_loras", [])
                ref_paths = []
                rip = params.get("reference_image_path")
                if rip:
                    ref_paths.append(rip)
                for cp in (params.get("character_ref_paths") or []):
                    if cp:
                        ref_paths.append(cp)
                # Pass character profiles into polish so the LLM has a
                # definitive name → descriptor mapping. Without this, polish
                # silently substitutes generic "the woman" / "the man" for
                # any character name it encounters — catastrophic for
                # non-human characters (Lumi the unicorn became "the woman"
                # in test 03). characters comes from params.characters,
                # the same list passed to the planner.
                characters = params.get("characters", []) or []
                clip_plans = polish_prompts_third_pass(
                    clip_plans, video_model, image_model, nsfw,
                    video_loras=video_loras, image_loras=image_loras,
                    image_paths=ref_paths or None,
                    characters=characters,
                )
                _capture_llm_pass(pid, "third_pass_polish")
                print(f"[Pipeline] Third-pass polish completed for {len(clip_plans)} clips")
            except Exception as e:
                print(f"[Pipeline] Prompt polish failed (non-fatal): {e}")
        elif polish_mode in ("full_guide", "light_guide"):
            # For inject modes, polish happened inside the planner — note it in the log
            _update_pipeline(pid, _polish_mode_used=polish_mode)

        _update_pipeline(pid, clip_plans=clip_plans, llm_streaming=False)
        _save_pipeline_state(pid)  # Save after planning

        # Check cancellation
        if _pipelines[pid]["status"] == "cancelled":
            return

        # In non-auto mode, pause for user review after planning
        if not auto_mode:
            _update_pipeline(pid, status="paused", pause_reason="review_prompts",
                             progress={"current": 1, "total": 3, "message": "Review prompts", "step": 0, "total_steps": 0})
            _save_pipeline_state(pid)  # Save paused state so Dashboard shows it
            _wait_for_resume(pid)
            if _pipelines[pid]["status"] == "cancelled":
                return
            # Reload clip_plans in case user edited them
            clip_plans = _pipelines[pid]["clip_plans"]

        # ── Phase 2: Generate Start Images ──────────────────────────────
        # Always generate start images. When no reference image was provided,
        # _run_image_generation creates an establishing/anchor image first and
        # adopts it as the shared reference, so every clip shares a look —
        # instead of skipping image gen and going straight to text-to-video.
        _update_pipeline(pid, phase="generating_images",
                         progress={"current": 0, "total": len(clip_plans), "message": "Generating start images...", "step": 0, "total_steps": 0})

        # ── Detect the reference's art style while the LLM is still up ──
        # One vision call naming the medium concretely; the phrase gets
        # prepended to every image prompt in _run_image_generation (see
        # the module-level "Reference art-style lock" note). Skipped when
        # already detected (resume) or the reference is photographic.
        from services import llm_service
        _style_ref = params.get("reference_image_path") or ""
        if ("_reference_style" not in params and _style_ref and os.path.isfile(_style_ref)):
            _style_phrase = ""
            try:
                if llm_service.is_loaded() and getattr(llm_service, "_vision_available", False):
                    _style_raw = llm_service.generate(
                        _STYLE_DESCRIBE_PROMPT,
                        max_new_tokens=48,
                        temperature=0.1,
                        image_paths=[_style_ref],
                        enable_thinking=False,
                    )
                    _style_phrase = _normalize_style_phrase(_style_raw)
                    print(f"[Pipeline {pid}] Reference art style: {_style_phrase!r} (raw: {str(_style_raw)[:80]!r})")
            except Exception as e:
                print(f"[Pipeline {pid}] Style detection skipped (non-fatal): {e}")
            # Record even when empty ("" = photographic / undetected) so
            # resume doesn't re-run the detection.
            params["_reference_style"] = _style_phrase
            _update_pipeline(pid, _reference_style=_style_phrase)

        # Unload LLM to free VRAM
        try:
            if llm_service.is_loaded():
                llm_service.unload_model()
        except Exception as e:
            print(f"[Pipeline] LLM unload warning (non-fatal): {e}")

        # On resume, reuse the start images that already generated before the
        # crash — but only if every file still exists (a wiped/half-written
        # output dir falls back to regenerating them, which is safer than
        # feeding missing paths into video generation).
        _resume_imgs_ok = bool(resume_images) and all(
            f and os.path.isfile(os.path.join(pipeline_out_dir, f)) for f in resume_images
        )
        if _resume_imgs_ok:
            clip_images = resume_images
            clip_keyframes = p.get("_clip_keyframes") or [[] for _ in clip_images]
            print(f"[Pipeline {pid}] Resume: reusing {len(clip_images)} start images — skipping image generation")
        else:
            if resume_images:
                print(f"[Pipeline {pid}] Resume: saved start images missing on disk — regenerating")
            clip_images, clip_keyframes = _run_image_generation(pid, params, clip_plans, out_dir=pipeline_out_dir, workspace=pipeline_workspace)

        _update_pipeline(pid, clip_images=clip_images, _clip_keyframes=clip_keyframes)
        _save_pipeline_state(pid)  # Save after image generation

        if _pipelines[pid]["status"] == "cancelled":
            return

        # In non-auto mode, pause for image review
        if not auto_mode:
            _update_pipeline(pid, status="paused", pause_reason="review_images",
                             progress={"current": 2, "total": 3, "message": "Review images", "step": 0, "total_steps": 0})
            _wait_for_resume(pid)
            if _pipelines[pid]["status"] == "cancelled":
                return

        # ── Phase 3: Generate Video ─────────────────────────────────────
        _update_pipeline(pid, phase="generating_video",
                         progress={"current": 0, "total": 1, "message": "Generating video...", "step": 0, "total_steps": 0})

        output_files = _run_video_generation(pid, params, clip_plans, planned_clips, clip_images, clip_keyframes, out_dir=pipeline_out_dir, workspace=pipeline_workspace)

        # A Stop during the video phase lands here after the abort. Record
        # whatever clips finished (the Dashboard can rerun/rejoin them),
        # but don't overwrite the cancelled status with "completed".
        if _pipelines[pid]["status"] == "cancelled":
            print(f"[Pipeline {pid}] Cancelled during video generation — keeping {len(output_files or [])} finished clip(s)")
            _update_pipeline(pid, output_files=output_files or [])
            _save_pipeline_state(pid)
            return

        _update_pipeline(pid,
                         status="completed",
                         phase="completed",
                         output_files=output_files,
                         _completed_at=time.time(),
                         progress={"current": 3, "total": 3, "message": "Done!", "step": 0, "total_steps": 0})
        _save_pipeline_state(pid)  # Save on completion

    except Exception as e:
        import traceback
        # Special-case the safety scanner. Don't print a stack trace for
        # safety violations — they're a clean refusal, not a crash, and
        # the user-visible message is purpose-built. Other exceptions
        # keep the existing traceback dump for debugging.
        try:
            from services.director.safety_scan import SafetyViolationError
        except Exception:
            SafetyViolationError = None  # type: ignore
        if SafetyViolationError is not None and isinstance(e, SafetyViolationError):
            print(
                f"[Pipeline {pid}] Safety scan blocked generation. "
                f"source={e.source} matched={e.matched_terms}"
            )
            user_msg = (
                "Generation aborted: the input contained content involving "
                f"minors in a prohibited context (matched terms: "
                f"{', '.join(e.matched_terms)}). The system refuses to "
                f"generate this category of content. Please revise your "
                f"concept to use only adult characters (18+)."
            )
            _update_pipeline(
                pid, status="failed", error=user_msg,
                _completed_at=time.time(),
                progress={"current": 0, "total": 0,
                          "message": "Generation aborted (safety policy)",
                          "step": 0, "total_steps": 0},
            )
            _save_pipeline_state(pid)
            return
        traceback.print_exc()
        # Tag with OOM info if applicable so the UI can surface the
        # OOM recovery banner. detect_oom returns None for non-OOM
        # failures, in which case oom_info stays absent.
        _oom_info = None
        try:
            from services.oom_detect import detect_oom
            import wgp as _wgp_mod
            _coef = float(_wgp_mod.server_config.get("vram_safety_coefficient", 0.80))
            _oom_info = detect_oom(e, _coef)
        except Exception:
            pass  # Never fail a failure handler
        _update_pipeline(pid, status="failed", error=str(e),
                         oom_info=_oom_info,
                         _completed_at=time.time(),
                         progress={"current": 0, "total": 0, "message": f"Error: {e}", "step": 0, "total_steps": 0})
        _save_pipeline_state(pid)  # Save on failure too


def _wait_for_resume(pid: str, poll_interval: float = 1.0):
    """Block until pipeline is resumed, cancelled, or removed."""
    while True:
        with _pipeline_lock:
            p = _pipelines.get(pid)
            if not p:
                return
            if p["status"] != "paused":
                return
        time.sleep(poll_interval)


def _wait_for_gpu(pid: str, poll_interval: float = 2.0):
    """Block until no generation jobs are actively running on GPU.

    Checks both _gen_lock availability and active job statuses.
    Returns False if pipeline was cancelled while waiting.
    """
    _update_pipeline(pid, progress={
        "current": 0, "total": 1,
        "message": "Waiting for GPU (generation queue)...",
        "step": 0, "total_steps": 0,
    })

    while True:
        if _pipelines.get(pid, {}).get("status") == "cancelled":
            return False

        # Check if any jobs are currently running
        active_jobs = [j for j in _jobs.values()
                       if j.get("status") in ("queued", "running")]
        if not active_jobs:
            return True

        time.sleep(poll_interval)


# ── Planning Phase ──────────────────────────────────────────────────────

def _ensure_llm_loaded(params: dict):
    """Load/reload LLM if needed. Shared between legacy and new planning."""
    from services import llm_service

    services_cfg = _wgp.server_config.get("services", {}) if _wgp else {}
    desired_model = params.get("llm_model_id") or services_cfg.get("llm_model_id", "Abhiray/gemma-4-E4B-it-heretic-GGUF")
    desired_device = params.get("llm_device") or services_cfg.get("llm_device", "cpu")
    desired_provider = params.get("llm_provider") or services_cfg.get("llm_provider", "local")
    desired_remote_url = services_cfg.get("llm_remote_url", "")
    desired_api_key = ""
    if desired_provider == "openai":
        desired_api_key = services_cfg.get("openai_api_key", "")
    elif desired_provider == "anthropic":
        desired_api_key = services_cfg.get("anthropic_api_key", "")

    # Free GPU memory before running a local CUDA LLM. Director planning
    # fires right after image edits / audio analysis: memory profiles keep
    # the last generation model resident, and torch's caching allocator
    # holds whatever Whisper / the vocal separator reserved — none of it
    # available to the llama-server SUBPROCESS. The server then loads its
    # weights fine but aborts (CUDA OOM → connection reset by peer) when
    # the vision encode spikes during the first planning request; the
    # identical request verified fine on a free GPU. Guarded by _gen_lock
    # so an active generation is never released mid-run; wgp reloads the
    # gen model transparently on its next job (reload_needed).
    if desired_provider == "local" and desired_device == "cuda" and _wgp is not None:
        acquired = _gen_lock.acquire(blocking=False) if _gen_lock is not None else True
        if acquired:
            try:
                if getattr(_wgp, "wan_model", None) is not None:
                    print("[Pipeline] Releasing generation model VRAM before LLM planning")
                    _wgp.release_model()
                else:
                    import gc
                    import torch
                    if torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
            except Exception as e:
                print(f"[Pipeline] Pre-LLM VRAM release skipped: {e}")
            finally:
                if _gen_lock is not None:
                    _gen_lock.release()
        else:
            print("[Pipeline] Generation in progress — skipping pre-LLM VRAM release")

    if llm_service.is_loaded():
        status = llm_service.get_status()
        if status.get("model_id") != desired_model or status.get("provider") != desired_provider:
            llm_service.unload_model()
            llm_service.load_model(model_id=desired_model, device=desired_device, provider=desired_provider, remote_url=desired_remote_url, api_key=desired_api_key)
    else:
        llm_service.load_model(model_id=desired_model, device=desired_device, provider=desired_provider, remote_url=desired_remote_url, api_key=desired_api_key)


def _capture_llm_pass(pid: str, pass_name: str):
    """Capture the current LLM state as a pass and append to the pipeline's log.

    Captures both system_prompt AND user_prompt so the Director Dashboard
    can render the full LLM input. Previously the dashboard only stored
    system_prompt, which made it look like the user's story description
    was missing from Pass 1's input — but it was always being sent as
    a separate user message; the dashboard just wasn't capturing it.
    """
    try:
        from services import llm_service
        pass_entry = {
            "pass": pass_name,
            "system_prompt": getattr(llm_service, '_last_system_prompt', '') or '',
            "user_prompt": getattr(llm_service, '_last_user_prompt', '') or '',
            "response_text": getattr(llm_service, '_stream_buffer', '') or '',
            "thinking_text": getattr(llm_service, '_last_thinking_text', None),
        }
        with _pipeline_lock:
            p = _pipelines.get(pid)
            if p:
                passes = p.get("_llm_passes", [])
                passes.append(pass_entry)
                p["_llm_passes"] = passes
    except Exception:
        pass


def _run_planning(pid: str, params: dict, pipeline_type: str):
    """Run LLM planning and return (clip_plans, planned_clips).

    Uses the new DirectorOrchestrator when use_director_v2 flag is set,
    otherwise falls back to legacy llm_service calls.
    """
    _ensure_llm_loaded(params)

    # Default v2 — see launch.py services-config comment for rationale.
    # The params dict is built from servicesConfig in the frontend, so
    # this default only fires for direct API callers that didn't pass
    # the flag at all. Keeping it consistent with the services-config
    # default here so the legacy path isn't accidentally hit.
    use_v2 = params.get("use_director_v2", True)

    if use_v2:
        return _run_planning_v2(pid, params, pipeline_type)
    else:
        return _run_planning_legacy(pid, params, pipeline_type)


def _run_planning_v2(pid: str, params: dict, pipeline_type: str):
    """New architecture: DirectorOrchestrator with planners + renderers."""
    from services import llm_service
    from services.director.orchestrator import DirectorOrchestrator, DirectorFlags

    # Build feature flags from params
    flags_dict = params.get("director_flags", {})
    flags = DirectorFlags.from_dict(flags_dict) if flags_dict else DirectorFlags()

    # Wrap LLM functions to capture each pass for the dashboard log
    _pass_counter = [0]
    def _logged_generate(*args, **kwargs):
        result = llm_service.generate(*args, **kwargs)
        _pass_counter[0] += 1
        _capture_llm_pass(pid, f"generate_{_pass_counter[0]}")
        return result

    def _logged_streaming(*args, **kwargs):
        result = llm_service.generate_streaming(*args, **kwargs)
        _pass_counter[0] += 1
        _capture_llm_pass(pid, f"streaming_{_pass_counter[0]}")
        return result

    # Create orchestrator with logged LLM functions
    director = DirectorOrchestrator(
        llm_generate=_logged_generate,
        llm_generate_streaming=_logged_streaming,
        flags=flags,
    )

    # Map pipeline_type to skill_type
    skill_map = {
        "music_video": "music_video",
        "short_film_audio": "short_film",
        "short_film_story": "short_film",
        "podcast": "podcast",
        "viral_video": "viral_video",
    }
    skill_type = skill_map.get(pipeline_type, "music_video")

    # Build planner kwargs
    scene_description = params.get("scene_description", "")
    reference_image_path = params.get("reference_image_path")
    planned_clips = params.get("planned_clips", [])

    # Read NSFW from server config (persisted setting, not per-request)
    services_cfg = _wgp.server_config.get("services", {}) if _wgp else {}
    nsfw = services_cfg.get("nsfw_mode", False)
    # Multi-shot LoRA mode — passes through to Pass 2 so it can emit
    # storyboard-format video_prompts for medium-length shots. See
    # the toggle's comment in launch.py for behavior details.
    multishot_lora_mode = services_cfg.get("director_multishot_lora_mode", False)

    seamless = params.get("seamless", True)
    # Pass video_model and image_model to every planner so Pass 2 can
    # route its prompt guides correctly. Previously these only flowed
    # into polish_block construction (when polish_mode was on); now the
    # planner gets them unconditionally so it can pick the right
    # dialect-aware guide files (ltx2_shot_breakdown.md for LTX-2,
    # flux_image_edit_pass2.md for Flux.2 Klein, etc.).
    planner_kwargs = {
        "reference_image_path": reference_image_path,
        "speaker_mappings": params.get("speaker_mappings"),
        "characters": params.get("characters", []),
        "nsfw": nsfw,
        "seamless": seamless,
        "video_model": params.get("video_model", ""),
        "image_model": params.get("image_model", ""),
        "multishot_lora_mode": multishot_lora_mode,
    }

    if pipeline_type == "short_film_story":
        planner_kwargs.update({
            "story_description": scene_description,
            "target_duration": params.get("target_duration", 60),
            "target_scenes": params.get("target_scenes"),
            "narrative_mode": params.get("narrative_mode", False),
            "fps": params.get("fps", 16),
            "frames_steps": params.get("frames_steps", 8),
            "frames_minimum": params.get("frames_minimum", 41),
        })
    elif pipeline_type == "short_film_audio":
        planner_kwargs.update({
            "clips": planned_clips,
            "story_description": scene_description,
            "audio_path": params.get("audio_path"),
            "lyrics": params.get("lyrics"),
        })
    elif pipeline_type in ("podcast", "viral_video"):
        planner_kwargs.update({
            "clips": planned_clips if planned_clips else None,
            "transcript": params.get("lyrics"),
            "audio_path": params.get("audio_path"),
            "concept": scene_description,
            "visual_style": params.get("visual_style", ""),
            "target_duration": params.get("target_duration", 30),
            "platform": params.get("platform", "general"),
            "style": params.get("style", "cinematic"),
        })
    else:
        # Music video
        planner_kwargs.update({
            "clips": planned_clips,
            "scene_description": scene_description,
            "lyrics": params.get("lyrics"),
            "bpm": params.get("bpm", 120),
        })

    # Inject LoRA guides + model dialect guides into the planner only for
    # the full/light_guide inject modes (legacy paths). Default mode
    # "third_pass" deliberately skips this — model dialect is applied
    # per-prompt after planning by polish_prompts_third_pass(), which
    # avoids stacking conflicting dialect guidance into Pass 2's already
    # crowded system prompt.
    polish_mode = services_cfg.get("director_prompt_polish", "third_pass")
    if polish_mode in ("full_guide", "light_guide"):
        from services.director.prompt_polish import build_polish_block
        guide_mode = "full" if polish_mode == "full_guide" else "light"
        video_model = params.get("video_model", "")
        image_model = params.get("image_model", "")
        video_loras = (params.get("video_loras") or {}).get("activated_loras", [])
        image_loras = (params.get("image_loras") or {}).get("activated_loras", [])
        polish_block = build_polish_block(video_model, image_model, guide_mode,
                                          video_loras=video_loras, image_loras=image_loras)
        if polish_block:
            planner_kwargs["polish_block"] = polish_block
            print(f"[Pipeline {pid}] Injected {guide_mode} polish block ({len(polish_block)} chars)")

    # Also pass character/location ref labels and paths for image prompt rules
    planner_kwargs["character_ref_paths"] = params.get("character_ref_paths", [])
    planner_kwargs["character_ref_labels"] = params.get("character_ref_labels", [])
    planner_kwargs["location_ref_paths"] = params.get("location_ref_paths", [])
    planner_kwargs["location_ref_labels"] = params.get("location_ref_labels", [])

    # Plan
    print(f"[Pipeline {pid}] Planning with DirectorOrchestrator (skill={skill_type})...")
    plan = director.plan(skill_type, **planner_kwargs)

    # Store the production plan in pipeline state for later reference
    _update_pipeline(pid, production_plan=plan.to_dict())

    # Render prompts
    has_reference = bool(reference_image_path)
    rendered = director.render_plan(plan, prompt_type="both", has_reference=has_reference)
    clip_plans = director.plan_to_clip_plans(rendered)

    # Build planned_clips from shot data (for story mode which creates clips)
    if pipeline_type == "short_film_story":
        cumulative = 0.0
        # Get FPS from model definition for accurate frame count
        fps = params.get("fps", 16)
        try:
            vm = params.get("video_model", "")
            md = _wgp.get_model_def(vm) if vm else None
            if md and md.get("fps"):
                fps = md["fps"]
        except Exception:
            pass
        new_clips = []
        for shot in plan.shots:
            duration_frames = shot.metadata.get("duration_frames") if shot.metadata else int(shot.duration_sec * fps)
            new_clips.append({
                "start": cumulative,
                "end": cumulative + shot.duration_sec,
                "duration_sec": shot.duration_sec,
                "duration_frames": duration_frames,
                "label": shot.narrative_role or shot.scene_type or "scene",
                "beat_count": 0,
            })
            cumulative += shot.duration_sec
        planned_clips = new_clips

    # Normalize
    if clip_plans and isinstance(clip_plans[0], str):
        clip_plans = [{"video_prompt": p, "image_prompt": ""} for p in clip_plans]

    # Debug: log shot structure
    for idx, cp in enumerate(clip_plans):
        kf_count = len(cp.get("keyframe_prompts", []) or [])
        wc = cp.get("window_count", 1)
        pc = planned_clips[idx] if idx < len(planned_clips) else {}
        dur = pc.get("duration_sec", pc.get("duration_frames", "?"))
        print(f"[Pipeline] Shot {idx+1}: duration={dur}s, windows={wc}, keyframes={kf_count}, prompt_len={len(cp.get('video_prompt',''))}")

    return clip_plans, planned_clips


def _run_planning_legacy(pid: str, params: dict, pipeline_type: str):
    """Legacy planning: direct calls to llm_service functions."""
    from services import llm_service

    scene_description = params.get("scene_description", "")
    reference_image_path = params.get("reference_image_path")
    speaker_mappings = params.get("speaker_mappings", [])
    characters = params.get("characters", [])
    audio_path = params.get("audio_path")
    planned_clips = params.get("planned_clips", [])
    fps = params.get("fps", 16)
    frames_steps = params.get("frames_steps", 8)
    frames_minimum = params.get("frames_minimum", 41)

    if pipeline_type == "short_film_story":
        # Path C: Full story-based planning
        target_duration = params.get("target_duration", 60)
        narrative_mode = params.get("narrative_mode", False)

        result = llm_service.plan_short_film_from_story(
            story_description=scene_description,
            characters=characters,
            reference_image_path=reference_image_path,
            target_duration=target_duration,
            narrative_mode=narrative_mode,
            fps=fps,
            frames_steps=frames_steps,
            frames_minimum=frames_minimum,
        )
        planned_clips = result.get("clips", [])
        clip_plans = result.get("clip_plans", [])

    elif pipeline_type == "short_film_audio":
        # Path B: Short film with uploaded dialogue audio
        result = llm_service.plan_short_film_prompts(
            clips=planned_clips,
            scene_description=scene_description,
            lyrics=params.get("lyrics", ""),
            reference_image_path=reference_image_path,
            speaker_mappings=speaker_mappings,
            characters=characters,
            prompt_type="both",
        )
        clip_plans = result if isinstance(result, list) else result.get("clip_plans", [])

    else:
        # Music video flow
        result = llm_service.plan_clip_prompts_and_images(
            clips=planned_clips,
            scene_description=scene_description,
            lyrics=params.get("lyrics", ""),
            bpm=params.get("bpm"),
            reference_image_path=reference_image_path,
            speaker_mappings=speaker_mappings,
            prompt_type="both",
        )
        clip_plans = result if isinstance(result, list) else result.get("clip_plans", [])

    # Normalize clip_plans to list of dicts
    if clip_plans and isinstance(clip_plans[0], str):
        clip_plans = [{"video_prompt": p, "image_prompt": ""} for p in clip_plans]

    return clip_plans, planned_clips


# ── Image Generation Phase ──────────────────────────────────────────────

def _run_image_generation(pid: str, params: dict, clip_plans: list[dict], out_dir: str = None, workspace: str = None) -> tuple[list[str], list[list[str]]]:
    """Generate start images and keyframe images per clip.

    Returns:
        (clip_images, clip_keyframes) where:
        - clip_images[i] = start image filename for clip i
        - clip_keyframes[i] = list of keyframe image filenames for clip i (may be empty)
    """
    ref_image_path = params.get("reference_image_path")
    character_ref_paths = params.get("character_ref_paths", []) or []
    location_ref_paths = params.get("location_ref_paths", []) or []
    image_model = params.get("image_model", "flux2_klein_9b")
    image_params = params.get("image_params", {})
    image_loras = params.get("image_loras", {})

    # Diagnostic-only log: report what the frontend sent so a future
    # "I selected N LoRAs but only K were applied" report has data we
    # can correlate against the [LoRA] Loading line wgp prints.
    _activated_in = list(image_loras.get("activated_loras", []) or [])
    _mults_in = image_loras.get("loras_multipliers", "") or ""
    if _activated_in:
        print(
            f"[Pipeline {pid}] Image LoRAs received: {len(_activated_in)} | "
            f"model={image_model} | "
            f"names={[os.path.basename(n) for n in _activated_in]} | "
            f"multipliers={_mults_in!r}"
        )

    # ── Filter image LoRAs to those that exist in the image model's dir ──
    # The frontend's DirectorLoraSelector filters available LoRAs by
    # model directory, but `savedLoraPerMode.image` persists across
    # sessions and can hold stale activations from a previous model
    # selection (e.g. an LTX-2 LoRA name that's never been in the
    # flux2_klein_9b/ directory). Without this filter, wgp.validate_task
    # rejects the entire task with "The following Loras files are missing
    # or invalid: [...]" and image gen never starts.
    #
    # This is a file-EXISTENCE check only — no architecture detection,
    # no dim peeking. Just: is the .safetensors actually in the right
    # directory? If not, drop it with a clear warning so the user knows
    # to re-select their image LoRAs for the active model.
    try:
        if _activated_in:
            try:
                _lora_dir = _wgp.get_lora_dir(image_model)
            except Exception:
                _lora_dir = ""
            if _lora_dir and os.path.isdir(_lora_dir):
                _existing = {
                    f for f in os.listdir(_lora_dir)
                    if f.lower().endswith((".safetensors", ".sft"))
                }
                _mult_tokens = _mults_in.split()
                _kept: list[str] = []
                _kept_mults: list[str] = []
                _skipped: list[str] = []
                for _idx, _name in enumerate(_activated_in):
                    _basename = os.path.basename(_name)
                    if _basename in _existing:
                        _kept.append(_name)
                        if _idx < len(_mult_tokens):
                            _kept_mults.append(_mult_tokens[_idx])
                    else:
                        _skipped.append(_basename)
                if _skipped:
                    _warn = (
                        f"Skipped {len(_skipped)} image LoRA(s) not present in "
                        f"{os.path.basename(_lora_dir)}/: {_skipped}. These were "
                        f"likely activated when a different image model was selected, "
                        f"and the saved selection persisted across sessions. Re-select "
                        f"the LoRAs you want for {image_model} in the Director image "
                        f"LoRA panel to clear the stale entries."
                    )
                    print(f"[Pipeline {pid}] {_warn}")
                    _existing_warnings = _pipelines.get(pid, {}).get("lora_warnings", []) or []
                    _update_pipeline(pid, lora_warnings=[*_existing_warnings, _warn])
                _activated_in = _kept
                _mults_in = " ".join(_kept_mults)
                image_loras = {
                    "activated_loras": _activated_in,
                    "loras_multipliers": _mults_in,
                }
                print(
                    f"[Pipeline {pid}] Image LoRAs after existence filter: "
                    f"{len(_kept)} kept, {len(_skipped)} skipped"
                )
    except Exception as _e:
        print(f"[Pipeline {pid}] LoRA file-existence filter skipped: {_e}")

    resolution = image_params.get("resolution", "1280x720")
    steps = image_params.get("num_inference_steps", 8)
    guidance = image_params.get("guidance_scale", 1)
    spatial_upsampling = params.get("image_spatial_upsampling", "")
    film_grain_intensity = params.get("image_film_grain_intensity", 0)
    film_grain_saturation = params.get("image_film_grain_saturation", 0.5)

    # Build full refs list: main scene + character refs + location refs
    extra_refs = [p for p in (character_ref_paths + location_ref_paths) if p and os.path.isfile(p)]
    print(f"[Pipeline {pid}] Image refs: main={ref_image_path}, chars={len(character_ref_paths)}, locs={len(location_ref_paths)}, extra_valid={len(extra_refs)}")

    if not out_dir:
        out_dir = _wgp.save_path

    # Count total images to generate (start images + keyframes)
    total_images = len(clip_plans)
    for plan in clip_plans:
        kf = plan.get("keyframe_prompts", [])
        if kf:
            total_images += len(kf)

    clip_images: list[str] = []
    clip_keyframes: list[list[str]] = []
    image_count = 0

    # Reference art-style lock: the exact lead sentence validated to hold
    # Klein to a stylized medium. Applied to EVERY image prompt (start
    # images, keyframes, anchor) at generation time — after polish, and
    # regardless of whether the planner remembered to name the medium.
    _style_prefix = _style_prefix_for(params.get("_reference_style") or "")

    def _gen_image(prompt: str, source_ref: str, include_extra_refs: bool = True) -> str:
        """Generate a single image using source_ref + optional extra refs."""
        nonlocal image_count
        _pre_strip = prompt
        prompt = _strip_motion_effects(prompt or "")
        if prompt != _pre_strip:
            print(f"[Pipeline {pid}] Stripped motion-effect language from image prompt")
        if _style_prefix and not prompt.lower().startswith("maintain the same"):
            prompt = _style_prefix + prompt
        all_refs = [r for r in ([source_ref] + (extra_refs if include_extra_refs else [])) if r]
        print(f"[Pipeline {pid}] _gen_image: {len(all_refs)} refs: {[os.path.basename(r) for r in all_refs]}")
        gen_params: dict = {
            "model_type": image_model,
            "prompt": prompt,
            "image_refs": all_refs,
            "image_mode": 1,
            "image_prompt_type": "",
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            # 'I' carries an image reference; a ref-less anchor is plain T2I.
            "video_prompt_type": "KI" if all_refs else "",
            "resolution": resolution,
            "seed": -1,
            "settings_version": 2.52,
            "generation_mode": "image",
            "repeat_generation": 1,
            "negative_prompt": "",
            "video_length": 1,
            "activated_loras": image_loras.get("activated_loras", []),
            "loras_multipliers": image_loras.get("loras_multipliers", ""),
            "_director_pipeline_id": pid,
        }
        if spatial_upsampling:
            gen_params["spatial_upsampling"] = spatial_upsampling
        if film_grain_intensity > 0:
            gen_params["film_grain_intensity"] = film_grain_intensity
            gen_params["film_grain_saturation"] = film_grain_saturation

        output_files = _submit_and_wait(gen_params, timeout_s=600, workspace=workspace, out_dir=out_dir)
        image_count += 1
        return output_files[0] if output_files else ""

    # If no reference image was provided, generate a single establishing /
    # "anchor" image from the scene description and adopt it as the shared
    # reference, so every clip's start image keeps a consistent look instead of
    # each being generated independently with no visual through-line.
    if not (ref_image_path and os.path.isfile(ref_image_path)):
        scene_desc = (params.get("scene_description") or "").strip()
        anchor_prompt = scene_desc or (clip_plans[0].get("image_prompt", "") if clip_plans else "") or "cinematic establishing shot"
        total_images += 1
        _update_pipeline(pid, progress={
            "current": 0,
            "total": total_images,
            "message": "Generating establishing image",
            "step": 0, "total_steps": 0,
        })
        print(f"[Pipeline {pid}] No reference image — generating establishing/anchor image first.")
        anchor_file = _gen_image(anchor_prompt, "", include_extra_refs=False)
        anchor_path = os.path.join(out_dir, anchor_file) if anchor_file else ""
        if anchor_path and os.path.isfile(anchor_path):
            ref_image_path = anchor_path
            print(f"[Pipeline {pid}] Adopted establishing image as shared reference: {anchor_file}")

    for i, plan in enumerate(clip_plans):
        if _pipelines[pid]["status"] == "cancelled":
            return clip_images, clip_keyframes

        # ── Determine image source: original reference or previous scene's output ──
        image_source = plan.get("image_source", "original")
        source_ref = ref_image_path  # default: user's original reference

        if image_source == "previous" and i > 0 and clip_images[i - 1]:
            prev_img_path = os.path.join(out_dir, clip_images[i - 1])
            if os.path.isfile(prev_img_path):
                source_ref = prev_img_path
                print(f"[Pipeline {pid}] Shot {i+1}: using previous scene output as source ({clip_images[i-1]})")

        _update_pipeline(pid, progress={
            "current": image_count,
            "total": total_images,
            "message": f"Shot {i + 1}: generating start image ({image_source})",
            "step": 0, "total_steps": 0,
        })

        prompt = plan.get("image_prompt", "")
        ref_exists = os.path.isfile(source_ref) if source_ref else False
        print(f"[Pipeline {pid}] Shot {i+1} start image: source={image_source}, ref={source_ref} (exists={ref_exists}), prompt='{prompt[:60]}...'")

        img_t0 = time.time()
        try:
            if image_source == "previous" and source_ref != ref_image_path:
                # Dual reference: previous scene output as primary + original reference for character identity
                # _gen_image puts source_ref first, then extra_refs (which includes character/location refs).
                # We temporarily prepend the original ref to extra_refs so the model sees both.
                saved_extras = extra_refs[:]
                extra_refs.insert(0, ref_image_path)
                start_img = _gen_image(prompt, source_ref, include_extra_refs=True)
                extra_refs[:] = saved_extras  # restore
            else:
                start_img = _gen_image(prompt, ref_image_path)
            clip_images.append(start_img)
        except Exception as e:
            print(f"[Pipeline {pid}] Shot {i+1} start image failed: {e}")
            clip_images.append("")
        # Record per-clip image timing
        timings = _pipelines.get(pid, {}).get("_clip_timings", {})
        timings[f"image_{i}"] = round(time.time() - img_t0, 2)
        _update_pipeline(pid, _clip_timings=timings)

        # ── Generate keyframes (chained from previous output) ──
        keyframe_prompts = plan.get("keyframe_prompts", []) or []
        shot_keyframes: list[str] = []

        if keyframe_prompts and clip_images[-1]:
            # Chain: each keyframe edits from the previous image
            chain_ref = os.path.join(out_dir, clip_images[-1])  # start from the start image

            for ki, kf_prompt in enumerate(keyframe_prompts):
                if _pipelines[pid]["status"] == "cancelled":
                    break

                # Ensure kf_prompt is a string (LLM may return dicts or other types)
                if isinstance(kf_prompt, dict):
                    kf_prompt = kf_prompt.get("prompt", kf_prompt.get("image_prompt", str(kf_prompt)))
                elif not isinstance(kf_prompt, str):
                    kf_prompt = str(kf_prompt)
                if not kf_prompt or not kf_prompt.strip():
                    continue

                _update_pipeline(pid, progress={
                    "current": image_count,
                    "total": total_images,
                    "message": f"Shot {i + 1}: keyframe {ki + 1}/{len(keyframe_prompts)}",
                    "step": 0, "total_steps": 0,
                })

                print(f"[Pipeline {pid}] Shot {i+1} keyframe {ki+1}: chain_ref='{os.path.basename(chain_ref)}', prompt='{str(kf_prompt)[:60]}...'")

                try:
                    kf_img = _gen_image(kf_prompt, chain_ref)
                    shot_keyframes.append(kf_img)
                    # Chain: next keyframe edits from this one
                    if kf_img:
                        chain_ref = os.path.join(out_dir, kf_img)
                except Exception as e:
                    print(f"[Pipeline {pid}] Shot {i+1} keyframe {ki+1} failed: {e}")
                    shot_keyframes.append("")

        clip_keyframes.append(shot_keyframes)

    _update_pipeline(pid, progress={
        "current": total_images,
        "total": total_images,
        "message": "All images generated",
        "step": 0, "total_steps": 0,
    })

    return clip_images, clip_keyframes


# ── Video Generation Phase ──────────────────────────────────────────────

def _run_video_generation(pid: str, params: dict, clip_plans: list[dict],
                          planned_clips: list[dict], clip_images: list[str],
                          clip_keyframes: Optional[list[list[str]]] = None,
                          out_dir: str = None, workspace: str = None) -> list[str]:
    """Generate multi-clip video with optional keyframe injection. Returns list of output filenames."""
    video_model = params.get("video_model")
    if not video_model:
        # Fallback: use first available video model from server config
        available = _wgp.get_models_list() if _wgp else []
        video_models = [m for m in available if m.get("is_t2v") or m.get("is_i2v")]
        video_model = video_models[0]["model_type"] if video_models else "ltx2_22B_distilled"
        print(f"[Pipeline] No video_model in params, using fallback: {video_model}")
    video_params = params.get("video_params", {})
    video_loras = params.get("video_loras", {})
    # Mirror of the image-LoRA file-existence filter — see _run_image_generation
    # for the rationale. Filter video_loras to those actually present in
    # video_model's LoRA directory so a stale activation from a different
    # video model doesn't crash wgp validation upfront.
    try:
        _vid_activated = list(video_loras.get("activated_loras", []) or [])
        _vid_mults = video_loras.get("loras_multipliers", "") or ""
        if _vid_activated:
            print(
                f"[Pipeline {pid}] Video LoRAs received: {len(_vid_activated)} | "
                f"model={video_model} | "
                f"names={[os.path.basename(n) for n in _vid_activated]} | "
                f"multipliers={_vid_mults!r}"
            )
            try:
                _vid_lora_dir = _wgp.get_lora_dir(video_model)
            except Exception:
                _vid_lora_dir = ""
            if _vid_lora_dir and os.path.isdir(_vid_lora_dir):
                _vid_existing = {
                    f for f in os.listdir(_vid_lora_dir)
                    if f.lower().endswith((".safetensors", ".sft"))
                }
                _vid_mult_tokens = _vid_mults.split()
                _vid_kept: list[str] = []
                _vid_kept_mults: list[str] = []
                _vid_skipped: list[str] = []
                for _idx, _name in enumerate(_vid_activated):
                    _basename = os.path.basename(_name)
                    if _basename in _vid_existing:
                        _vid_kept.append(_name)
                        if _idx < len(_vid_mult_tokens):
                            _vid_kept_mults.append(_vid_mult_tokens[_idx])
                    else:
                        _vid_skipped.append(_basename)
                if _vid_skipped:
                    _warn = (
                        f"Skipped {len(_vid_skipped)} video LoRA(s) not present in "
                        f"{os.path.basename(_vid_lora_dir)}/: {_vid_skipped}. These "
                        f"were likely activated when a different video model was "
                        f"selected. Re-select your video LoRAs for {video_model}."
                    )
                    print(f"[Pipeline {pid}] {_warn}")
                    _exw = _pipelines.get(pid, {}).get("lora_warnings", []) or []
                    _update_pipeline(pid, lora_warnings=[*_exw, _warn])
                video_loras = {
                    "activated_loras": _vid_kept,
                    "loras_multipliers": " ".join(_vid_kept_mults),
                }
                print(
                    f"[Pipeline {pid}] Video LoRAs after existence filter: "
                    f"{len(_vid_kept)} kept, {len(_vid_skipped)} skipped"
                )
    except Exception as _e:
        print(f"[Pipeline {pid}] Video LoRA file-existence filter skipped: {_e}")

    audio_path = params.get("audio_path")
    seamless = params.get("seamless", True)
    pipeline_type = params.get("pipeline_type", "music_video")
    # Get FPS from model definition (reliable) — don't trust frontend default of 16
    fps = params.get("fps", 16)
    try:
        model_def = _wgp.get_model_def(video_model)
        if model_def and model_def.get("fps"):
            fps = model_def["fps"]
    except Exception:
        pass
    print(f"[Pipeline] Video gen: fps={fps}, video_model={video_model}")

    resolution = video_params.get("resolution", "1280x720")
    steps = video_params.get("num_inference_steps", 8)
    guidance = video_params.get("guidance_scale", 1)
    spatial_upsampling = params.get("video_spatial_upsampling", "")
    film_grain_intensity = params.get("video_film_grain_intensity", 0)
    film_grain_saturation = params.get("video_film_grain_saturation", 0.5)
    self_refiner = params.get("video_self_refiner", 0)

    if not out_dir:
        out_dir = _wgp.save_path

    # Quantize helper
    try:
        _min_f, _fs, _latent = _wgp.get_model_min_frames_and_step(video_model)
    except Exception:
        _min_f, _fs, _latent = 17, 8, 8

    def _quantize_frames(cf):
        return max((cf - 1) // _latent * _latent + 1, _min_f)

    def _quantize_frames_nearest(cf):
        # Round to the NEAREST valid (latent*n + 1) length instead of
        # flooring — used with a carry term for per-clip sequences where
        # floor-truncation would compound (see below).
        return max(round((cf - 1) / _latent) * _latent + 1, _min_f)

    # ── SEAMLESS MODE: one continuous rolling window generation ──────
    # Instead of separate per-clip jobs, build ONE generation that looks like
    # Studio mode: rolling windows with per-window prompts + keyframe injection.
    if seamless:
        window_prompts_all = []  # One prompt per rolling window
        keyframe_images = []     # All keyframe images in order
        keyframe_frame_positions = []  # Absolute frame numbers (1-indexed for wgp parser)

        # Track cumulative frame position as we go through scenes
        cumulative_frames = 0

        for i, plan in enumerate(clip_plans):
            pc = planned_clips[i] if i < len(planned_clips) else {}
            dur_sec = pc.get("duration_sec", pc.get("end", 0) - pc.get("start", 0))
            if dur_sec <= 0:
                dur_sec = 20
            scene_frames = round(dur_sec * fps)

            wp = plan.get("window_prompts") or []
            wp = [w.get("prompt", w.get("text", str(w))) if isinstance(w, dict) else str(w) for w in wp]
            if len(wp) > 1:
                for w_prompt in wp:
                    window_prompts_all.append(w_prompt)
            else:
                vp = plan.get("video_prompt", "")
                if vp:
                    window_prompts_all.append(vp)

            # Mid-scene keyframes from the LLM (injected at mid-point of this scene)
            if clip_keyframes and i < len(clip_keyframes):
                kf_list = clip_keyframes[i]
                if kf_list:
                    # Distribute mid-scene keyframes evenly across the scene
                    num_kf = len(kf_list)
                    for ki, kf_file in enumerate(kf_list):
                        if kf_file:
                            kf_path = os.path.join(out_dir, kf_file)
                            if os.path.isfile(kf_path):
                                # Position: evenly spaced within the scene
                                kf_pos = cumulative_frames + int(scene_frames * (ki + 1) / (num_kf + 1))
                                keyframe_images.append(kf_path)
                                keyframe_frame_positions.append(kf_pos + 1)  # 1-indexed for wgp parser

            # Scene boundary keyframe: inject next scene's start image at the end of this scene
            if i < len(clip_plans) - 1:
                next_img = clip_images[i + 1] if i + 1 < len(clip_images) else ""
                if next_img:
                    next_path = os.path.join(out_dir, next_img)
                    if os.path.isfile(next_path):
                        boundary_frame = cumulative_frames + scene_frames
                        keyframe_images.append(next_path)
                        keyframe_frame_positions.append(boundary_frame)  # 1-indexed (approx)

            cumulative_frames += scene_frames

        total_frames = _quantize_frames(cumulative_frames)
        sliding_window_frames = _quantize_frames(round(20 * fps))

        # First scene's start image
        first_start = ""
        if clip_images and clip_images[0]:
            first_path = os.path.join(out_dir, clip_images[0])
            if os.path.isfile(first_path):
                first_start = first_path

        prompt_text = "\n".join(window_prompts_all)

        print(f"[Pipeline {pid}] Seamless mode: {len(window_prompts_all)} windows, "
              f"{len(keyframe_images)} keyframes at frames {keyframe_frame_positions}, "
              f"{total_frames} total frames ({total_frames/fps:.1f}s)")

    # ── STANDARD MODE: separate per-clip generation ─────────────────
    else:
        prompts = []
        image_start_paths = []
        image_end_paths = []
        per_clip_frames = []
        has_sliding_window = False

        for i, plan in enumerate(clip_plans):
            wp = plan.get("window_prompts") or []
            wp = [w.get("prompt", w.get("text", str(w))) if isinstance(w, dict) else str(w) for w in wp]
            if len(wp) > 1:
                prompts.append("\n".join(wp))
            else:
                vp = plan.get("video_prompt", "")
                pc = planned_clips[i] if i < len(planned_clips) else {}
                dur = pc.get("duration_sec", pc.get("end", 0) - pc.get("start", 0))
                if dur > 32 and vp:
                    print(f"[Pipeline] WARNING: Clip {i+1} is {dur:.0f}s but has no window_prompts")
                prompts.append(vp)

            img_file = clip_images[i] if i < len(clip_images) else ""
            if img_file:
                img_path = os.path.join(out_dir, img_file)
                image_start_paths.append(img_path if os.path.isfile(img_path) else "")
            else:
                image_start_paths.append("")
            image_end_paths.append("")

            pc = planned_clips[i] if i < len(planned_clips) else {}
            window_prompts = plan.get("window_prompts", []) or []
            window_count = plan.get("window_count", 1) or 1
            if len(window_prompts) > 1 and window_count <= 1:
                window_count = len(window_prompts)
            has_keyframes = bool(plan.get("keyframe_prompts"))
            num_keyframes = len(plan.get("keyframe_prompts", []) or [])

            if window_count > 1 or has_keyframes:
                shot_duration = pc.get("duration_sec", pc.get("end", 0) - pc.get("start", 0))
                if shot_duration <= 0:
                    shot_duration = 20 * max(window_count, num_keyframes + 1)
                clip_frames = max(round(shot_duration * fps), round(5 * fps))
                per_clip_frames.append(clip_frames)
                has_sliding_window = True
            else:
                # SECONDS are the fps-agnostic ground truth. planned_clips
                # from plan_clip_structure carry start/end (+duration_frames)
                # but NO duration_sec — the old `get("duration_sec", 0)`
                # fell straight through to duration_frames, which the
                # frontend may have had computed at the WRONG model's fps
                # (modelOptions belongs to the Studio-selected model, e.g.
                # ACE-Step right after generating the track → fps 16). A
                # 26s clip became 26x16=416 frames, rendered at LTX-2's 25
                # fps = 16.6s — every music-video clip silently shortened
                # by 16/25.
                dur_sec = pc.get("duration_sec") or (pc.get("end", 0) - pc.get("start", 0))
                clip_frames = round(dur_sec * fps) if dur_sec > 0 else pc.get("duration_frames", round(20 * fps))
                if clip_frames > round(32 * fps):
                    has_sliding_window = True
                per_clip_frames.append(max(clip_frames, round(5 * fps)))

        # Quantize to the model's (latent*n + 1) frame lattice WITHOUT letting
        # the error compound. Floor-snapping each clip independently lost 0-7
        # frames per clip (an 8s clip = 200 frames @25fps floors to 193 —
        # exactly the "7 frames short" the user measured), while the song
        # plays on at true time — so cuts drifted off the planned musical
        # break points by seconds near the end of a song. Instead, round each
        # clip to the NEAREST valid length and carry the residual into the
        # next clip: every cumulative boundary stays within half a latent
        # step (±4 frames ≈ 0.16s) of the planned beat, forever.
        _carried: list[int] = []
        _carry = 0.0
        for _cf in per_clip_frames:
            _target = _cf + _carry
            _q = _quantize_frames_nearest(_target)
            _carry = _target - _q
            _carried.append(_q)
        per_clip_frames = _carried
        total_frames = sum(per_clip_frames)
        max_clip_frames = max(per_clip_frames) if per_clip_frames else round(5 * fps)
        # Single-window case: sliding_window_frames must be STRICTLY
        # greater than max_clip_frames after wgp's internal quantization
        # (line ~6725 of wgp.py), or wgp interprets `video_length >
        # sliding_window_size` and splits the clip into multiple
        # windows. Add `_latent + 1` frames of safety margin — one full
        # latent step plus one to guarantee strict-greater after the
        # `(x - 1) // latent * latent + 1` rounding. Multi-window
        # case (has_sliding_window=True) stays at 20s because the
        # whole point is to slide.
        #
        # Single-window clips are allowed up to 32s (was 22s): LTX-2.3
        # holds up well past its nominal ~20s window — user-validated at
        # 26s with the window sized to the clip — and one window beats
        # mid-clip window seams for music sync. plan_clip_structure caps
        # planned clips at MAX_CLIP_SECONDS=26 (the 75%-merge rule can
        # stretch a section to ~32s, hence the threshold).
        sliding_window_frames = (
            round(20 * fps) if has_sliding_window
            else max_clip_frames + _latent + 1
        )

        for ci, cf in enumerate(per_clip_frames):
            wp_count = len((clip_plans[ci].get("window_prompts") or []) if ci < len(clip_plans) else [])
            wc = clip_plans[ci].get("window_count", 1) if ci < len(clip_plans) else 1
            print(f"[Pipeline {pid}] Clip {ci+1}: {cf} frames ({cf/fps:.1f}s), windows={wc}, window_prompts={wp_count}")

    # Build audio params
    audio_params: dict = {}
    if pipeline_type == "short_film_story":
        audio_params["audio_prompt_type"] = ""
    elif audio_path:
        audio_params["audio_prompt_type"] = "A"
        audio_params["audio_guide"] = audio_path
        audio_scale = params.get("audio_scale")
        if audio_scale is not None:
            audio_params["audio_scale"] = audio_scale

    # ── Build gen_params based on mode ──────────────────────────────
    lora_params = {
        "activated_loras": video_loras.get("activated_loras", []),
        "loras_multipliers": " ".join(
            m.split(";")[0] for m in (video_loras.get("loras_multipliers", "") or "").split(" ") if m
        ),
    }

    if seamless:
        # Seamless: ONE generation job with rolling windows + keyframe injection
        gen_params: dict = {
            "model_type": video_model,
            "prompt": prompt_text,
            "image_mode": 0,
            "multi_prompts_gen_type": 0,  # Rolling window mode (one prompt per window)
            "image_prompt_type": "S" if first_start else "",
            "video_prompt_type": "",
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "resolution": resolution,
            "video_length": total_frames,
            "sliding_window_size": sliding_window_frames,
            "seed": -1,
            "settings_version": 2.52,
            "generation_mode": "video",
            "repeat_generation": 1,
            "negative_prompt": "",
            "self_refiner_setting": self_refiner,
            "_director_pipeline_id": pid,
            **lora_params,
            **audio_params,
        }
        if first_start:
            gen_params["image_start"] = first_start
        # Keyframe injection via image_refs + frames_positions (numeric absolute positions)
        if keyframe_images:
            gen_params["image_refs"] = keyframe_images
            gen_params["frames_positions"] = " ".join(str(p) for p in keyframe_frame_positions)
            existing_vpt = gen_params.get("video_prompt_type", "")
            if "KFI" not in existing_vpt:
                gen_params["video_prompt_type"] = existing_vpt + "KFI"
            print(f"[Pipeline {pid}] Seamless keyframes: {len(keyframe_images)} images at frames {keyframe_frame_positions}")

    else:
        # Standard: separate per-clip generation jobs
        CLIP_SEPARATOR = "\n---CLIP_BOUNDARY---\n"
        prompt_text = CLIP_SEPARATOR.join(prompts)

        has_any_start = any(p for p in image_start_paths)
        has_any_end = any(p for p in image_end_paths)
        if not has_any_start:
            image_start_paths = []
        if not has_any_end:
            image_end_paths = []

        ipt = "SE" if has_any_start and has_any_end else ("S" if has_any_start else "")

        gen_params: dict = {
            "model_type": video_model,
            "prompt": prompt_text,
            "image_mode": 0,
            "multi_prompts_gen_type": 3,  # Multi-clip mode
            "image_prompt_type": ipt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "resolution": resolution,
            "video_length": total_frames,
            "sliding_window_size": sliding_window_frames,
            "per_clip_frames": per_clip_frames,
            "seed": -1,
            "settings_version": 2.52,
            "generation_mode": "video",
            "repeat_generation": 1,
            "negative_prompt": "",
            "self_refiner_setting": self_refiner,
            "_director_pipeline_id": pid,
            **lora_params,
            **audio_params,
        }
        if has_any_start:
            gen_params["image_start"] = image_start_paths
        if has_any_end:
            gen_params["image_end"] = image_end_paths
        # Per-clip keyframe injection
        if clip_keyframes:
            per_clip_kf_paths: list[list[str]] = []
            for i, kf_list in enumerate(clip_keyframes):
                paths = []
                for kf_file in kf_list:
                    if kf_file:
                        kf_path = os.path.join(out_dir, kf_file)
                        if os.path.isfile(kf_path):
                            paths.append(kf_path)
                per_clip_kf_paths.append(paths)
            if any(paths for paths in per_clip_kf_paths):
                gen_params["per_clip_keyframes"] = per_clip_kf_paths
                print(f"[Pipeline {pid}] Keyframe injection: {[len(p) for p in per_clip_kf_paths]} keyframes per clip")

    # Common params
    voice_ref = params.get("voice_reference")
    if voice_ref:
        gen_params["voice_reference"] = voice_ref
        gen_params["identity_guidance_scale"] = params.get("identity_guidance_scale", 3.0)
        print(f"[Pipeline {pid}] Voice reference: {voice_ref}, identity_scale={gen_params['identity_guidance_scale']}")
    if spatial_upsampling:
        gen_params["spatial_upsampling"] = spatial_upsampling
    if film_grain_intensity > 0:
        gen_params["film_grain_intensity"] = film_grain_intensity
        gen_params["film_grain_saturation"] = film_grain_saturation

    # Track progress by monitoring the generation job
    output_files = _submit_and_wait(gen_params, timeout_s=7200, workspace=workspace, out_dir=out_dir)  # 2hr timeout for long videos
    return output_files
