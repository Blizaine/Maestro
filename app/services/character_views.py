"""Lossless character view caches, separate from H3 conditioning and previews."""
from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from pathlib import Path

VERSION = 1
MAX_VIEWS = 128
MAX_PIXELS = 32_000_000


def view_path(directory: Path, views: dict, view_id: str) -> Path:
    """Resolve only an indexed PNG inside this character's generated cache."""
    folder = str(views.get("directory", ""))
    if not re.fullmatch(r"views-[0-9a-f]{32}", folder):
        raise ValueError("Invalid character image cache.")
    item = next((item for item in views.get("items", []) if item.get("id") == view_id), None)
    if item is None or not re.fullmatch(r"view-[0-9]{4}", str(view_id)):
        raise ValueError("Character image not found.")
    base = directory.resolve()
    path = (base / folder / f"{view_id}.png").resolve()
    if path.parent.parent != base or not path.is_file():
        raise ValueError("Character image is missing. Recover its images again.")
    return path


def selection(views: dict, selected_ids, cover_id) -> dict:
    valid = {item["id"] for item in views["items"]}
    if (not isinstance(selected_ids, list) or not selected_ids
            or any(not isinstance(value, str) for value in selected_ids)
            or len(selected_ids) != len(set(selected_ids))
            or not set(selected_ids) <= valid or not isinstance(cover_id, str)
            or cover_id not in selected_ids):
        raise ValueError("Select at least one image and choose a cover from the selection.")
    return {**views, "selected_ids": selected_ids, "cover_id": cover_id}


def build_cache(directory: Path, producer) -> dict:
    """Publish only complete PNG sets; a failed recovery preserves the old set."""
    from PIL import Image

    base = directory.resolve()
    cache = base / f"views-{uuid.uuid4().hex}"
    cache.mkdir()
    try:
        result = producer(cache)
        items = result.get("items", [])
        if not 1 <= len(items) <= MAX_VIEWS:
            raise ValueError("A character needs between one and 128 recovered images.")
        ids = set()
        for item in items:
            view_id = item.get("id")
            if not isinstance(view_id, str) or not re.fullmatch(r"view-[0-9]{4}", view_id) or view_id in ids:
                raise ValueError("Invalid recovered character image ID.")
            ids.add(view_id)
            path = cache / f"{view_id}.png"
            with Image.open(path) as picture:
                if picture.format != "PNG" or picture.width * picture.height > MAX_PIXELS:
                    raise ValueError("Recovered character images must be PNGs up to 32 megapixels.")
                item.update(width=picture.width, height=picture.height)
                picture.verify()
        selected = result.get("selected_ids", [item["id"] for item in items[:min(4, len(items))]])
        cover = result.get("cover_id", selected[0] if selected else "")
        return selection({**result, "version": VERSION, "directory": cache.name}, selected, cover)
    except Exception:
        # This UUID folder was created here, resolved below the character root.
        if cache.resolve().parent == base:
            shutil.rmtree(cache)
        raise


def original_images(path: Path, kind: str, destination: Path, update) -> dict:
    """Prefer saved source pixels over reconstructing an encoded copy."""
    from PIL import Image, ImageOps

    if kind == "image":
        with Image.open(path) as picture:
            if picture.width * picture.height > MAX_PIXELS:
                raise ValueError("The character image exceeds 32 megapixels.")
            mode = "RGBA" if "A" in picture.getbands() or "transparency" in picture.info else "RGB"
            ImageOps.exif_transpose(picture).convert(mode).save(destination / "view-0001.png")
        return {"source": "original_image", "items": [{"id": "view-0001"}]}

    import cv2
    import numpy as np
    capture = cv2.VideoCapture(str(path))
    items = []
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if count < 1 or fps <= 0:
            raise ValueError("Could not read the character's saved video.")
        for frame_number in np.linspace(0, count - 1, min(32, count)).round().astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ok, pixels = capture.read()
            if not ok:
                continue
            if pixels.shape[0] * pixels.shape[1] > MAX_PIXELS:
                raise ValueError("The character video exceeds 32 megapixels per frame.")
            view_id = f"view-{len(items) + 1:04}"
            Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)).save(destination / f"{view_id}.png")
            items.append({"id": view_id, "source_frame": int(frame_number), "seconds": round(frame_number / fps, 3)})
            update(f"Saving source image {len(items)}…")
    finally:
        capture.release()
    return {"source": "original_video", "items": items}


def make_preview(directory: Path, views: dict, kind: str) -> Path:
    """Keep the legacy image/video reference channel while retaining PNG masters."""
    from PIL import Image

    if kind == "image":
        output = directory / "visual.png"
        shutil.copyfile(view_path(directory, views, views["items"][0]["id"]), output)
        return output
    output = directory / "visual.mp4"
    frames = []
    for item in views["items"][:8]:
        with Image.open(view_path(directory, views, item["id"])) as picture:
            # Only the small compatibility preview is resized/compressed.
            picture = picture.convert("RGB")
            picture.thumbnail((1024, 1024))
            if not frames:
                width, height = picture.size
            frames.append(picture.resize((width, height)).tobytes())
    repeats = max(1, (4 + len(frames) - 1) // len(frames))
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "2", "-i", "pipe:0", "-an",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output)], input=b"".join(frame * repeats for frame in frames),
        capture_output=True, check=True, timeout=120)
    return output


def validate_embedded_views(views: dict) -> None:
    if (not isinstance(views, dict) or views.get("version") != VERSION
            or views.get("source") not in {"refmod", "original_image", "original_video"}
            or not isinstance(views.get("items"), list) or not 1 <= len(views["items"]) <= MAX_VIEWS):
        raise ValueError("Invalid embedded character images.")
    ids = []
    for item in views["items"]:
        if (not isinstance(item, dict) or not re.fullmatch(r"view-[0-9]{4}", str(item.get("id", "")))
                or item.get("type") != "image" or item.get("extension") != ".png"):
            raise ValueError("Invalid embedded character image.")
        ids.append(item["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate character image IDs.")
    if not isinstance(views.get("notes", []), list) or any(not isinstance(note, str) for note in views.get("notes", [])):
        raise ValueError("Invalid character image notes.")
    selection(views, views.get("selected_ids"), views.get("cover_id"))


def embed_views(directory: Path, views: dict | None, tensors: dict, extension: dict) -> None:
    """Share selected PNG masters without changing any native RefMod fields."""
    from .refmod import embedded_media

    if not views:
        return
    for key in list(tensors):
        if key.startswith("maestro.image_views."):
            tensors.pop(key)
    items = []
    for view_id in views["selected_ids"]:
        path = view_path(directory, views, view_id)
        items.append({"id": view_id, **embedded_media(tensors, path,
            f"maestro.image_views.{view_id}.bytes", "image")})
    extension["image_views"] = {"version": VERSION, "source": views["source"], "items": items,
        "cover_id": views["cover_id"], "selected_ids": list(views["selected_ids"]), "notes": views.get("notes", [])}


def restore_views(directory: Path, views: dict, tensors: dict) -> dict:
    validate_embedded_views(views)
    def restore(cache):
        items = []
        for item in views["items"]:
            (cache / f"{item['id']}.png").write_bytes(tensors[item["tensor"]].numpy().tobytes())
            items.append({"id": item["id"]})
        return {**views, "items": items}
    return build_cache(directory, restore)
