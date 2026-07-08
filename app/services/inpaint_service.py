"""
Inpaint Service — Orchestrates LLM intent parsing + SAM segmentation + LTX retake.

The text-driven inpaint pipeline:
1. LLM parses user description → extract target object + modification prompt
2. SAM 3.1 microservice segments the target object and propagates mask across video
3. LTX MaskInjection regenerates the masked region with the modification prompt
"""

import json
import logging
import os
import time
from typing import Optional

import numpy as np
import requests

log = logging.getLogger("inpaint_service")

# SAM service URL — configurable via env or defaults to localhost:8321
SAM_SERVICE_URL = os.environ.get("SAM_SERVICE_URL", "http://127.0.0.1:8321")
SAM_PORT = int(os.environ.get("SAM_SERVICE_PORT", "8321"))
SAM_TIMEOUT = 300  # seconds — first use downloads ~5GB of checkpoints + loads model
_sam_process = None


def _find_sam_env():
    """Find the SAM conda env python path."""
    # Look relative to the app directory
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(app_dir, "services", "sam", "env", "python.exe"),  # conda on Windows
        os.path.join(app_dir, "services", "sam", "env", "bin", "python"),  # conda on Linux
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _find_sam_service_script():
    """Find the sam_service.py script."""
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(app_dir, "services", "sam", "sam_service.py")
    return path if os.path.isfile(path) else None


def ensure_sam_running() -> bool:
    """Start the SAM service if not already running. Returns True if available."""
    global _sam_process, SAM_SERVICE_URL

    # Check if already running
    status = check_sam_status()
    if status.get("status") in ("available", "ready"):
        return True

    # Try to start it
    python_path = _find_sam_env()
    script_path = _find_sam_service_script()
    if not python_path or not script_path:
        print("[SAM] Service not installed — cannot start on demand")
        return False

    import subprocess
    import socket

    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    SAM_SERVICE_URL = f"http://127.0.0.1:{port}"
    print(f"[SAM] Starting SAM service on port {port}...")

    # Don't capture stdout — let SAM's loading messages print to console
    _sam_process = subprocess.Popen(
        [python_path, script_path, "--port", str(port), "--idle-timeout", "120"],
    )

    # Wait for it to be ready — model loading can take 2-3 minutes on first use
    import time as _time
    max_wait = 300  # 5 minutes
    for i in range(max_wait * 2):  # check every 0.5s
        _time.sleep(0.5)
        if i % 20 == 0 and i > 0:
            print(f"[SAM] Waiting for SAM service to load... ({i // 2}s)")
        try:
            resp = requests.get(f"{SAM_SERVICE_URL}/health", timeout=2)
            if resp.ok:
                print(f"[SAM] Service ready: {SAM_SERVICE_URL}")
                return True
        except requests.ConnectionError:
            pass
        # Check if process died
        if _sam_process.poll() is not None:
            print(f"[SAM] Service process exited unexpectedly")
            return False

    print(f"[SAM] Service failed to start within {max_wait}s")
    return False


def unload_sam():
    """Tell the SAM service to unload model from VRAM (without shutting down)."""
    try:
        resp = requests.post(f"{SAM_SERVICE_URL}/unload", timeout=10)
        if resp.ok:
            print("[SAM] Model unloaded from VRAM")
        else:
            print(f"[SAM] Unload returned {resp.status_code}")
    except Exception as e:
        print(f"[SAM] Failed to unload: {e}")


def shutdown_sam():
    """Shut down the SAM service completely to free all VRAM including CUDA context."""
    global _sam_process
    try:
        requests.post(f"{SAM_SERVICE_URL}/shutdown", timeout=5)
    except Exception:
        pass
    if _sam_process is not None:
        try:
            _sam_process.terminate()
            _sam_process.wait(timeout=10)
        except Exception:
            pass
        _sam_process = None
    print("[SAM] Service shut down")


def check_sam_status() -> dict:
    """Check if the SAM segmentation service is running and available.

    Possible status values returned to the frontend:
      "ready" / "available" — service is running and reachable
      "installed"           — env is installed but service not yet running.
                              Will start on demand on first request.
      "not_installed"       — SAM env doesn't exist on disk. User needs to
                              run "Install Inpaint Support (SAM 3.1)" from
                              the Pinokio menu (sam_install.js). UI should
                              show a prominent install banner in this case.
      "unavailable"         — service is reachable but unhealthy, OR some
                              other failure. UI shows generic error.
    """
    try:
        resp = requests.get(f"{SAM_SERVICE_URL}/health", timeout=5)
        if resp.ok:
            return resp.json()
        return {"status": "unavailable", "model_loaded": False}
    except requests.ConnectionError:
        # Check if SAM env is installed (can be started on demand)
        if _find_sam_env() and _find_sam_service_script():
            return {"status": "installed", "model_loaded": False}
        # Distinct status so the UI can render the right "needs install"
        # affordance (point user at the Pinokio menu) rather than a
        # generic "service down" error which suggests something different.
        return {
            "status": "not_installed",
            "model_loaded": False,
            "error": "SAM service not installed. Open the Pinokio menu and click 'Install Inpaint Support (SAM 3.1)' to enable inpainting.",
        }
    except Exception as e:
        return {"status": "unavailable", "model_loaded": False, "error": str(e)}


def parse_inpaint_intent(description: str) -> dict:
    """
    Use LLM to parse a natural language edit description into:
    - target: the object/region to segment (for SAM)
    - prompt: the LTX generation prompt for the modified region
    - negative_prompt: what to avoid

    Falls back to heuristic parsing if LLM is not available.
    """
    # Try LLM parsing first
    try:
        from services.llm_service import is_loaded, generate

        if is_loaded():
            system_prompt = (
                "You are an AI video editor assistant. Given the user's edit instruction, extract:\n"
                "1. target: A short noun phrase describing the object or region to select/mask "
                "(for AI segmentation). Just the object, no verbs or actions. Examples: "
                "\"man's hands\", \"the sky\", \"her dress\", \"the car\".\n"
                "2. prompt: A full descriptive prompt for generating the modified content in "
                "the masked region. Include visual details.\n"
                "3. negative_prompt: Things to avoid in the regenerated region.\n\n"
                "Respond ONLY with valid JSON, no markdown:\n"
                "{\"target\": \"...\", \"prompt\": \"...\", \"negative_prompt\": \"...\"}"
            )

            result = generate(
                prompt=f'Edit instruction: "{description}"',
                system_prompt=system_prompt,
                max_new_tokens=256,
                temperature=0.3,
                enable_thinking=False,
            )
            # Parse JSON from response
            text = result.strip()
            # Handle potential markdown wrapping
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            # Handle potential leading text before JSON
            json_start = text.find("{")
            if json_start > 0:
                text = text[json_start:]
            parsed = json.loads(text)
            print(f"[Inpaint] LLM parsed intent: target='{parsed.get('target')}', prompt='{parsed.get('prompt')}'")
            return {
                "target": parsed.get("target", description),
                "prompt": parsed.get("prompt", description),
                "negative_prompt": parsed.get("negative_prompt", ""),
            }
    except ImportError:
        pass
    except Exception as e:
        print(f"[Inpaint] LLM intent parsing failed: {e}, falling back to heuristic")

    # Fallback: simple heuristic parsing
    return _heuristic_parse_intent(description)


def _heuristic_parse_intent(description: str) -> dict:
    """Simple rule-based intent parsing when LLM is not available."""
    desc_lower = description.lower().strip()

    # Common patterns: "make X do Y", "add X to Y", "change X to Y", "X is/are Y"
    target = description
    prompt = description

    # "make [target] [action]" pattern
    for prefix in ["make ", "have ", "let "]:
        if desc_lower.startswith(prefix):
            rest = description[len(prefix):]
            # Find verb boundary: common verbs that split target from action
            for verb in [" hold ", " wear ", " glow ", " burn ", " fly ", " turn ", " become ",
                         " shoot ", " emit ", " carry ", " catch ", " throw "]:
                idx = rest.lower().find(verb)
                if idx > 0:
                    target = rest[:idx].strip()
                    prompt = rest.strip()
                    return {"target": target, "prompt": prompt, "negative_prompt": ""}

    # "[target] is/are [state]" pattern
    for copula in [" is ", " are "]:
        idx = desc_lower.find(copula)
        if idx > 0:
            target = description[:idx].strip()
            prompt = description.strip()
            return {"target": target, "prompt": prompt, "negative_prompt": ""}

    # "add [thing] to [target]" pattern
    if desc_lower.startswith("add "):
        to_idx = desc_lower.find(" to ")
        if to_idx > 0:
            thing = description[4:to_idx].strip()
            target = description[to_idx + 4:].strip()
            prompt = f"{target} with {thing}"
            return {"target": target, "prompt": prompt, "negative_prompt": ""}

    # "change [target] to [new state]" / "replace [target] with [new]"
    for verb_pair in [("change ", " to "), ("replace ", " with "), ("turn ", " into ")]:
        if desc_lower.startswith(verb_pair[0]):
            split_idx = desc_lower.find(verb_pair[1])
            if split_idx > 0:
                target = description[len(verb_pair[0]):split_idx].strip()
                new_state = description[split_idx + len(verb_pair[1]):].strip()
                prompt = f"{target} transformed into {new_state}"
                return {"target": target, "prompt": prompt, "negative_prompt": ""}

    # Default: use full description for both
    return {"target": description, "prompt": description, "negative_prompt": ""}


def segment_video(
    video_path: str,
    text: str,
    frame_index: int = -1,
    start_time: float = 0,
    end_time: float = -1,
    fps: float = 25.0,
    mask_padding: int = 0,
) -> dict:
    """
    Call SAM service to segment an object in a video and get propagated masks.

    Returns dict with:
      - masks_path: path to .npy file containing [T, H, W] binary mask
      - mask_preview: base64 PNG of mask overlay on key frame
      - target: the text prompt used
      - frame_index: which frame was used for initial segmentation
    """
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time > 0 else -1

    # Auto-pick key frame: middle of the range
    if frame_index < 0:
        if end_frame > 0:
            frame_index = (start_frame + end_frame) // 2
        else:
            frame_index = start_frame

    payload = {
        "video_path": video_path,
        "text": text,
        "frame_index": frame_index,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "mask_padding": mask_padding,
    }

    try:
        resp = requests.post(
            f"{SAM_SERVICE_URL}/segment/video",
            json=payload,
            timeout=SAM_TIMEOUT,
        )
        if not resp.ok:
            try:
                error = resp.json().get("detail", "Segmentation failed")
            except Exception:
                error = resp.text[:200] or f"HTTP {resp.status_code}"
            raise RuntimeError(f"SAM service error: {error}")
        return resp.json()
    except requests.ConnectionError:
        raise RuntimeError("SAM segmentation service is not running. Start it first.")
    except requests.Timeout:
        raise RuntimeError("SAM segmentation timed out. Video may be too long.")


def segment_image_preview(
    video_path: str,
    text: str,
    frame_index: int = -1,
    start_time: float = 0,
    end_time: float = -1,
    fps: float = 25.0,
) -> dict:
    """
    Quick preview: extract a single frame, segment it, return mask overlay.
    Faster than full video segmentation — useful for preview before committing.
    """
    import cv2

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time > 0 else -1

    if frame_index < 0:
        if end_frame > 0:
            frame_index = (start_frame + end_frame) // 2
        else:
            frame_index = start_frame

    # Extract frame to temp image
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

    # Save temp frame
    temp_dir = os.path.join(os.path.dirname(video_path), ".temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"frame_{frame_index}_{int(time.time())}.png")
    cv2.imwrite(temp_path, frame)

    try:
        payload = {
            "image_path": temp_path,
            "text": text,
            "confidence": 0.3,
        }
        resp = requests.post(
            f"{SAM_SERVICE_URL}/segment/image",
            json=payload,
            timeout=SAM_TIMEOUT,
        )
        if not resp.ok:
            try:
                error = resp.json().get("detail", "Segmentation failed")
            except Exception:
                error = resp.text[:200] or f"HTTP {resp.status_code}"
            raise RuntimeError(f"SAM service error: {error}")
        result = resp.json()
        result["frame_index"] = frame_index
        result["fps"] = actual_fps
        return result
    except requests.ConnectionError:
        raise RuntimeError("SAM segmentation service is not running.")
    finally:
        # Cleanup temp frame
        try:
            os.remove(temp_path)
        except OSError:
            pass


def build_retake_params(
    video_path: str,
    masks_path: str,
    prompt: str,
    negative_prompt: str = "",
    model_type: str = "ltx2_22B_distilled_1_1",
    retake_strength: float = 0.85,
    seed: int = -1,
    guidance_scale: float = 1.0,
    num_inference_steps: int = 8,
    start_time: float = 0,
    end_time: float = -1,
    fps: float = 25.0,
    workspace: str = "default",
) -> dict:
    """
    Build the generation parameters for LTX retake with a spatial mask.

    The masks_path points to a .npy file containing [T, H, W] binary mask
    from SAM segmentation. This gets converted to the masking_source format
    expected by the LTX pipeline.
    """
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time > 0 else -1

    return {
        "video_path": video_path,
        "masks_path": masks_path,
        "start_time": start_time,
        "end_time": end_time,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model_type": model_type,
        "retake_strength": retake_strength,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "workspace": workspace,
        "generation_mode": "video",
        "inpaint": True,  # flag to indicate spatial mask inpaint
    }
