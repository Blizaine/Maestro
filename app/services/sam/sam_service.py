"""
SAM 3.1 Segmentation Microservice

Runs as a standalone FastAPI server in its own Python 3.12+ venv.
The main Maestro backend communicates with this service via HTTP.

Endpoints:
  POST /segment/image  — Segment objects in a single image by text prompt
  POST /segment/video  — Segment + propagate across video frames
  GET  /health         — Liveness check
  POST /shutdown       — Graceful shutdown (releases VRAM)

Usage:
  python sam_service.py --port 8321 --idle-timeout 300
"""

import argparse
import asyncio
import base64
import io
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="[SAM] %(message)s")
log = logging.getLogger("sam_service")

# -- Global state --
_predictor = None
_image_model = None
_processor = None
_last_used = 0.0
_idle_timeout = 300  # seconds
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model():
    """Load SAM 3.1 model on demand."""
    global _predictor, _image_model, _processor, _last_used
    if _predictor is not None:
        _last_used = time.time()
        return

    log.info("Loading SAM 3.1 model...")
    try:
        from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
        from sam3.model.sam3_image_processor import Sam3Processor

        # Download base checkpoint with fallback to ungated mirror
        base_ckpt = _download_checkpoint_with_fallback(
            official_repo="facebook/sam3", mirror_repo="jetjodh/sam3",
            filename="sam3.pt", label="SAM 3 base"
        )

        with torch.autocast(_device, dtype=torch.bfloat16):
            _image_model = build_sam3_image_model(checkpoint_path=base_ckpt, load_from_HF=False)
            _processor = Sam3Processor(_image_model, confidence_threshold=0.3)
            # Use SAM 3 base video predictor (not 3.1 multiplex)
            # SAM 3 base uses standard SDPA (no flash-attn-3), works on consumer GPUs
            _predictor = build_sam3_video_predictor(checkpoint_path=base_ckpt)

        _last_used = time.time()
        log.info("SAM 3.1 loaded successfully")
    except ImportError as e:
        log.error(f"SAM 3.1 not installed: {e}")
        log.error("Install with: pip install /path/to/sam3")
        raise
    except Exception as e:
        log.error(f"Failed to load SAM 3.1: {e}")
        raise


def _download_checkpoint_with_fallback(official_repo: str, mirror_repo: str, filename: str, label: str) -> str:
    """Download checkpoint from official HF repo, falling back to ungated mirror."""
    from huggingface_hub import hf_hub_download
    # Also download config.json alongside the checkpoint
    for repo in [official_repo, mirror_repo]:
        try:
            log.info(f"Downloading {label} from {repo}...")
            try:
                hf_hub_download(repo_id=repo, filename="config.json")
            except Exception:
                pass  # config.json is optional
            path = hf_hub_download(repo_id=repo, filename=filename)
            log.info(f"Downloaded {label}: {path}")
            return path
        except Exception as e:
            err_str = str(e).lower()
            if "gated" in err_str or "restricted" in err_str or "access to model" in err_str:
                log.warning(f"Gated access denied for {repo}, trying mirror...")
                continue
            elif repo == official_repo:
                log.warning(f"Failed to download from {repo}: {e}, trying mirror...")
                continue
            else:
                raise
    raise RuntimeError(f"Failed to download {label} from both {official_repo} and {mirror_repo}")


def _unload_model():
    """Unload model to free VRAM."""
    global _predictor, _image_model, _processor
    if _predictor is None and _image_model is None:
        return
    log.info("Unloading SAM 3.1 model...")
    _predictor = None
    _image_model = None
    _processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    log.info("SAM 3.1 unloaded, VRAM freed")


async def _idle_watchdog():
    """Background task to unload model after idle timeout."""
    while True:
        await asyncio.sleep(30)
        if _predictor is not None and (time.time() - _last_used) > _idle_timeout:
            log.info(f"Idle for {_idle_timeout}s, unloading model")
            _unload_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_idle_watchdog())
    yield
    task.cancel()
    _unload_model()


app = FastAPI(title="SAM 3.1 Segmentation Service", lifespan=lifespan)


# -- Request/Response models --

class ImageSegmentRequest(BaseModel):
    image_path: str
    text: str
    confidence: float = 0.3

class VideoSegmentRequest(BaseModel):
    video_path: str
    text: str
    frame_index: int = 0
    start_frame: int = 0
    end_frame: int = -1  # -1 = all frames
    mask_padding: int = 0

class SegmentResponse(BaseModel):
    mask_preview: str  # base64 PNG with mask overlay
    masks_path: str | None = None  # path to saved .npy mask array
    target: str
    frame_index: int
    num_objects: int
    scores: list[float]


# -- Endpoints --

@app.get("/health")
async def health():
    return {
        "status": "available",
        "model_loaded": _predictor is not None,
        "device": _device,
        "idle_timeout": _idle_timeout,
    }


@app.post("/segment/image", response_model=SegmentResponse)
async def segment_image(req: ImageSegmentRequest):
    """Segment objects in a single image by text prompt."""
    global _last_used
    _load_model()
    _last_used = time.time()

    if not os.path.isfile(req.image_path):
        raise HTTPException(400, f"Image not found: {req.image_path}")

    try:
        image = Image.open(req.image_path).convert("RGB")

        with torch.autocast(_device, dtype=torch.bfloat16):
            _processor.confidence_threshold = req.confidence
            state = _processor.set_image(image)
            state = _processor.set_text_prompt(state=state, prompt=req.text)

        masks = state["masks"]  # bool tensor [N, 1, H, W]
        scores = state["scores"].tolist()

        # Create preview: overlay mask on image
        preview = _create_mask_preview(image, masks)

        return SegmentResponse(
            mask_preview=preview,
            target=req.text,
            frame_index=0,
            num_objects=masks.shape[0],
            scores=scores,
        )
    except Exception as e:
        log.error(f"Segmentation failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/unload")
async def unload_model():
    """Unload SAM model from VRAM without shutting down the service."""
    _unload_model()
    return {"status": "unloaded"}


@app.post("/segment/video", response_model=SegmentResponse)
async def segment_video(req: VideoSegmentRequest):
    """Segment an object in a video using batched video tracking.

    Strategy: process the video in VRAM-safe batches (~200 frames each),
    using SAM's full video predictor for temporal consistency within each batch.
    The mask from the last frame of batch N seeds batch N+1 for continuity.

    This preserves SAM's temporal tracking (smooth masks, occlusion handling)
    while fitting in 24GB VRAM.
    """
    global _last_used
    _load_model()
    _last_used = time.time()

    if not os.path.isfile(req.video_path):
        raise HTTPException(400, f"Video not found: {req.video_path}")

    try:
        import cv2

        cap = cv2.VideoCapture(req.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        actual_start = max(0, req.start_frame)
        actual_end = min(req.end_frame, total_frames) if req.end_frame > 0 else total_frames
        frame_count = actual_end - actual_start

        if frame_count <= 0:
            raise HTTPException(400, "Invalid frame range")

        # Batch size: ~200 frames fits comfortably in 24GB with SAM loaded
        # SAM 3 base uses ~4GB VRAM. Frame memory scales ~50MB per frame.
        # 250 frames ≈ 16GB total, safe for 24GB GPUs.
        BATCH_SIZE = 250
        MIN_BATCH = 30  # Don't create tiny leftover batches
        batches = []
        for batch_start in range(actual_start, actual_end, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, actual_end)
            batches.append((batch_start, batch_end))
        # Merge tiny last batch into previous
        if len(batches) > 1 and (batches[-1][1] - batches[-1][0]) < MIN_BATCH:
            prev_start, _ = batches[-2]
            _, last_end = batches[-1]
            batches = batches[:-2] + [(prev_start, last_end)]

        log.info(f"Processing {frame_count} frames in {len(batches)} batch(es) "
                 f"(frames {actual_start}-{actual_end}, batch_size={BATCH_SIZE})")

        all_batch_masks = []  # list of [batch_frames, H, W] arrays
        scores = []
        preview_data = None  # (frame_rgb, masks_tensor) for preview
        carry_mask = None  # mask from last frame of previous batch for continuity

        for batch_idx, (b_start, b_end) in enumerate(batches):
            b_count = b_end - b_start
            log.info(f"Batch {batch_idx + 1}/{len(batches)}: frames {b_start}-{b_end} ({b_count} frames)")

            # Extract batch frames to temp JPEG folder (SAM loads from folder)
            batch_dir = os.path.join(os.path.dirname(req.video_path), ".sam_batch")
            if os.path.exists(batch_dir):
                import shutil
                shutil.rmtree(batch_dir)
            os.makedirs(batch_dir)

            cap = cv2.VideoCapture(req.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, b_start)
            for i in range(b_count):
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(os.path.join(batch_dir, f"{i:06d}.jpg"), frame)
            cap.release()

            with torch.autocast(_device, dtype=torch.bfloat16):
                resp = _predictor.handle_request(dict(
                    type="start_session",
                    resource_path=batch_dir,
                ))
                session_id = resp["session_id"]

                try:
                    if batch_idx == 0:
                        # First batch: use text prompt on middle frame
                        prompt_frame = min(b_count // 2, b_count - 1)
                        print(f"[SAM] Text prompt: '{req.text}' on frame {prompt_frame}")
                        prompt_resp = _predictor.handle_request(dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=prompt_frame,
                            text=req.text,
                        ))
                        # Check if the prompt detected anything
                        if isinstance(prompt_resp, dict) and "outputs" in prompt_resp:
                            out = prompt_resp["outputs"]
                            if isinstance(out, dict):
                                obj_ids = out.get("out_obj_ids", [])
                                probs = out.get("out_probs", [])
                                print(f"[SAM] add_prompt detected {len(obj_ids) if hasattr(obj_ids, '__len__') else '?'} objects, "
                                      f"probs: {probs if hasattr(probs, 'tolist') and len(probs) < 5 else '...'}")
                            else:
                                print(f"[SAM] add_prompt output type: {type(out)}")
                    else:
                        # Subsequent batches: re-prompt with text on frame 0
                        # Text prompts are more reliable than point prompts for initial detection
                        _predictor.handle_request(dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=0,
                            text=req.text,
                        ))

                    # Propagate and collect masks for this batch
                    batch_masks = {}
                    prop_count = 0
                    for out in _predictor.handle_stream_request(
                        dict(type="propagate_in_video", session_id=session_id)
                    ):
                        fidx = out["frame_index"]
                        outputs = out["outputs"]
                        prop_count += 1

                        # Log first output to debug format
                        if prop_count == 1:
                            log.info(f"First propagation output: type={type(outputs)}, "
                                     f"keys={list(outputs.keys()) if isinstance(outputs, dict) else 'N/A'}")

                        if outputs is None:
                            continue

                        # Extract binary masks from output dict
                        if isinstance(outputs, dict):
                            m = outputs.get("out_binary_masks")
                            s = outputs.get("out_probs", outputs.get("out_scores"))
                        elif isinstance(outputs, (tuple, list)) and len(outputs) >= 4:
                            video_res_masks = outputs[2]
                            m = (video_res_masks > 0.0) if torch.is_tensor(video_res_masks) else video_res_masks
                            s = outputs[3]
                        else:
                            continue

                        if m is None:
                            continue

                        # Convert to numpy, skip if no objects detected
                        if torch.is_tensor(m):
                            if m.shape[0] == 0:
                                continue
                            mask_np = m[0].cpu().numpy() if m.ndim > 2 else m.cpu().numpy()
                        elif isinstance(m, np.ndarray):
                            if m.shape[0] == 0:
                                continue
                            mask_np = m[0] if m.ndim > 2 else m
                        else:
                            continue

                        batch_masks[fidx] = mask_np
                        if not scores and s is not None:
                            if torch.is_tensor(s):
                                scores = s.flatten().tolist()
                            elif isinstance(s, np.ndarray):
                                scores = s.flatten().tolist()
                            elif hasattr(s, "tolist"):
                                scores = s.tolist()

                    log.info(f"Propagation yielded {prop_count} frames, {len(batch_masks)} had masks")

                    # Build ordered mask array for this batch
                    mask_h = mask_w = 0
                    if batch_masks:
                        sample = next(iter(batch_masks.values()))
                        mask_h, mask_w = sample.shape
                    elif carry_mask is not None:
                        # No masks in this batch — use carry mask dimensions
                        mask_h, mask_w = carry_mask.shape

                    if mask_h > 0:
                        batch_array = np.zeros((b_count, mask_h, mask_w), dtype=bool)
                        for i in range(b_count):
                            if i in batch_masks:
                                batch_array[i] = batch_masks[i]
                            elif batch_masks:
                                nearest = min(batch_masks.keys(), key=lambda k: abs(k - i))
                                batch_array[i] = batch_masks[nearest]
                            elif carry_mask is not None:
                                # No masks at all in this batch — extend last known mask
                                batch_array[i] = carry_mask

                        carry_mask = batch_array[-1].copy()
                        all_batch_masks.append(batch_array)

                    # Capture preview from first batch middle frame
                    if batch_idx == 0 and batch_masks:
                        mid = min(b_count // 2, b_count - 1)
                        if mid in batch_masks:
                            mid_abs = b_start + mid
                            cap2 = cv2.VideoCapture(req.video_path)
                            cap2.set(cv2.CAP_PROP_POS_FRAMES, mid_abs)
                            ret, frame = cap2.read()
                            cap2.release()
                            if ret:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                mask_t = torch.from_numpy(batch_masks[mid]).unsqueeze(0).unsqueeze(0)
                                preview_data = (frame_rgb, mask_t)
                finally:
                    _predictor.handle_request(dict(type="close_session", session_id=session_id))

            # Clean up batch frames
            try:
                import shutil
                shutil.rmtree(batch_dir)
            except OSError:
                pass

        if not all_batch_masks:
            raise HTTPException(400, f"SAM found no matches for '{req.text}'")

        print(f"[SAM] All batches complete: {len(all_batch_masks)} batch(es), "
              f"shapes: {[b.shape for b in all_batch_masks]}")

        # Concatenate all batches
        print(f"[SAM] Concatenating masks...")
        mask_array = np.concatenate(all_batch_masks, axis=0)  # [T, H, W]

        # Apply padding using efficient per-frame OpenCV dilation (much faster than scipy on large arrays)
        if req.mask_padding > 0:
            import cv2 as _cv2
            kernel_size = req.mask_padding * 2 + 1
            kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            print(f"[SAM] Applying mask padding ({req.mask_padding}px) to {mask_array.shape[0]} frames...")
            for i in range(mask_array.shape[0]):
                mask_array[i] = _cv2.dilate(mask_array[i].astype(np.uint8), kernel).astype(bool)

        # Save mask as compressed npz (much smaller than uncompressed npy for bool arrays)
        masks_dir = os.path.join(os.path.dirname(req.video_path), ".masks")
        os.makedirs(masks_dir, exist_ok=True)
        mask_filename = f"mask_{int(time.time())}_{os.path.basename(req.video_path)}.npy"
        masks_path = os.path.join(masks_dir, mask_filename)
        print(f"[SAM] Saving mask ({mask_array.shape}, {mask_array.nbytes / 1e6:.1f}MB)...")
        np.save(masks_path, mask_array)

        print(f"[SAM] Saved mask: {mask_array.shape} ({frame_count} frames, {len(batches)} batches)")

        # Create preview (resize to 720p max for faster base64 encoding)
        print(f"[SAM] Generating preview...")
        preview = ""
        if preview_data:
            frame_rgb, mask_t = preview_data
            pil_img = Image.fromarray(frame_rgb)
            # Downscale preview for large frames — resize both image and mask together
            if pil_img.width > 1280 or pil_img.height > 1280:
                max_dim = max(pil_img.width, pil_img.height)
                scale = 1280 / max_dim
                new_w = int(pil_img.width * scale)
                new_h = int(pil_img.height * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                # Resize mask to match: mask_t is [N, 1, H, W] or [N, H, W]
                mask_np = mask_t[0].cpu().numpy() if torch.is_tensor(mask_t) else mask_t
                if mask_np.ndim > 2:
                    mask_np = mask_np[0] if mask_np.ndim == 3 else mask_np
                mask_resized = np.array(Image.fromarray(mask_np.astype(np.uint8) * 255).resize((new_w, new_h), Image.NEAREST)) > 127
                mask_t = torch.from_numpy(mask_resized).unsqueeze(0).unsqueeze(0)
            preview = _create_mask_preview(pil_img, mask_t)
        print(f"[SAM] Done, returning response")

        mid_frame = actual_start + frame_count // 2
        return SegmentResponse(
            mask_preview=preview,
            masks_path=masks_path,
            target=req.text,
            frame_index=mid_frame,
            num_objects=1,
            scores=scores[:1] if scores else [1.0],
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Video segmentation failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/shutdown")
async def shutdown():
    """Graceful shutdown — unload model and exit."""
    _unload_model()
    log.info("Shutdown requested")
    # Schedule process exit
    asyncio.get_event_loop().call_later(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
    return {"status": "shutting_down"}


# -- Helpers --

def _extract_clip(video_path: str, start_frame: int, end_frame: int) -> str:
    """Extract a frame range from a video to a temp file for SAM processing.

    Returns the temp clip path, or the original path if the full video is needed.
    """
    import cv2
    if start_frame <= 0 and end_frame < 0:
        return video_path  # Use full video

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    actual_end = min(end_frame, total) if end_frame > 0 else total
    actual_start = max(0, start_frame)

    # Don't bother extracting if it's most of the video
    if (actual_end - actual_start) > total * 0.8:
        cap.release()
        return video_path

    clip_dir = os.path.join(os.path.dirname(video_path), ".clips")
    os.makedirs(clip_dir, exist_ok=True)
    clip_path = os.path.join(clip_dir, f"clip_{actual_start}_{actual_end}_{int(time.time())}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(clip_path, fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)
    for i in range(actual_start, actual_end):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

    frame_count = actual_end - actual_start
    log.info(f"Extracted clip: frames {actual_start}-{actual_end} ({frame_count} frames) from {total} total")
    return clip_path


def _create_mask_preview(image: Image.Image, masks: torch.Tensor) -> str:
    """Create a base64 PNG with mask overlay on the image."""
    img_array = np.array(image)
    overlay = img_array.copy()

    if masks.shape[0] > 0:
        # Combine all masks, use first one as primary
        combined = masks[0, 0].cpu().numpy()  # [H, W]
        # Green overlay on masked region
        overlay[combined] = (overlay[combined] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

    pil = Image.fromarray(overlay)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _create_video_mask_preview(video_path: str, frame_index: int, mask) -> str:
    """Extract a frame from video and overlay the mask."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return ""

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    overlay = frame_rgb.copy()

    if mask is not None:
        if isinstance(mask, np.ndarray):
            m = mask[0] if mask.ndim > 2 else mask
        elif torch.is_tensor(mask):
            m = mask[0].cpu().numpy() if mask.ndim > 2 else mask.cpu().numpy()
        else:
            m = None

        if m is not None:
            # Resize mask to frame size if needed
            if m.shape != frame_rgb.shape[:2]:
                m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(
                    (frame_rgb.shape[1], frame_rgb.shape[0]),
                    Image.NEAREST
                )) > 127
            overlay[m] = (overlay[m] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

    pil = Image.fromarray(overlay)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM 3.1 Segmentation Service")
    parser.add_argument("--port", type=int, default=8321)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--idle-timeout", type=int, default=300, help="Unload model after N seconds idle")
    args = parser.parse_args()

    _idle_timeout = args.idle_timeout

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
