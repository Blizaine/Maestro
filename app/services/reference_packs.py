"""Immutable, reviewed five-view character packs stored in Maestro's library.

No remote service or GPU is required. Import creates a draft revision; approving
it is a human review decision, not a claim of automated identity validation.
"""

from copy import deepcopy
import hashlib
import shutil
import time
import uuid
from pathlib import Path

from . import character_library as library

VIEWS = {
    "face_closeup": "Face close-up",
    "full_body_front": "Full body — front",
    "full_body_three_quarter": "Full body — three-quarter",
    "upper_body_three_quarter": "Upper body — three-quarter",
    "full_body_back": "Full body — back",
}


def _revision(record: dict, revision_id: str) -> dict:
    revision = next((r for r in record.get("reference_pack", {}).get("revisions", [])
                     if r["id"] == revision_id), None)
    if revision is None:
        raise ValueError("Reference Pack revision not found.")
    return revision


def image_path(character_id: str, revision_id: str, view: str) -> Path:
    record = library._get_character_record(character_id)
    revision = _revision(record, revision_id)
    if view not in VIEWS or view not in revision["images"]:
        raise ValueError("Reference Pack view not found.")
    root = library._character_directory(character_id)
    path = (root / revision["images"][view]["storage_name"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Reference Pack image is missing.")
    return path


def public_pack(character_id: str, pack: dict) -> dict:
    result = deepcopy(pack)
    root = library._character_directory(character_id)
    for revision in result["revisions"]:
        for view, image in revision["images"].items():
            image["path"] = str(root / image["storage_name"])
            image["url"] = f"/api/v1/characters/{character_id}/reference-packs/{revision['id']}/images/{view}"
            image["label"] = VIEWS[view]
            image.pop("storage_name", None)
    return result


def create_revision(*, name: str = "", images: dict, character_id: str | None = None,
                    base_revision_id: str | None = None, label: str = "") -> dict:
    """Copy all five images before publishing; revisions never share mutable files."""
    from PIL import Image, ImageOps

    if not isinstance(images, dict) or not images or set(images) - VIEWS.keys():
        raise ValueError("Supply images using the five Reference Pack view names.")
    new_character = character_id is None
    character_id = character_id or uuid.uuid4().hex[:16]
    directory = library._character_directory(character_id)
    revision_id = uuid.uuid4().hex[:16]
    lock = library._FILE_LOCKS[hash(character_id) % len(library._FILE_LOCKS)]
    with lock:
        if new_character:
            name = " ".join(str(name or "").split())[:120]
            if not name:
                raise ValueError("Give this character a name.")
            record = {"id": character_id, "name": name, "created_at": time.time(),
                      "voice": None, "reference_pack": {"version": 1, "revisions": []}}
        else:
            record = deepcopy(library._get_character_record(character_id))
            if not record.get("reference_pack"):
                raise ValueError("Create a Reference Pack character first.")
        base = _revision(record, base_revision_id) if base_revision_id else None
        if not base and set(images) != VIEWS.keys():
            raise ValueError("A new Reference Pack requires all five views.")
        staged = directory / "packs" / (".draft-" + revision_id)
        published = directory / "packs" / revision_id
        staged.mkdir(parents=True)
        try:
            revision = {"id": revision_id, "number": len(record["reference_pack"]["revisions"]) + 1,
                        "parent_revision_id": base_revision_id, "label": str(label or "").strip()[:200],
                        "created_at": time.time(), "approved": False, "images": {}}
            for view in VIEWS:
                if view in images:
                    source = library._source_path(images[view], library._IMAGE_EXTENSIONS)
                    with Image.open(source) as picture:
                        if picture.width * picture.height > 32_000_000 or getattr(picture, "n_frames", 1) != 1:
                            raise ValueError("Pack views must be single images up to 32 megapixels.")
                        ImageOps.exif_transpose(picture).convert("RGB").save(staged / f"{view}.png")
                else:
                    shutil.copy2(image_path(character_id, base_revision_id, view), staged / f"{view}.png")
                path = staged / f"{view}.png"
                with Image.open(path) as picture:
                    width, height = picture.size
                revision["images"][view] = {
                    "storage_name": f"packs/{revision_id}/{view}.png", "width": width, "height": height,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "inherited": view not in images,
                }
            if len({image["sha256"] for image in revision["images"].values()}) != 5:
                raise ValueError("Provide five distinct views; duplicate images were found.")
            staged.rename(published)
            record["reference_pack"]["revisions"].append(revision)
            record["updated_at"] = time.time()
            # The cover remains the original close-up for legacy library consumers.
            if new_character:
                record["visual"] = {"type": "image", "storage_name": revision["images"]["face_closeup"]["storage_name"],
                                    "filename": "face_closeup.png"}
            with library._LOCK:
                index = library._load_index()
                if new_character:
                    index["characters"].append(record)
                else:
                    # Preserve any independently attached voice or other library metadata.
                    current = next((r for r in index["characters"] if r["id"] == character_id), None)
                    if current is None:
                        raise ValueError("The character was deleted during import.")
                    current.update(reference_pack=record["reference_pack"], updated_at=record["updated_at"])
                    record = current
                library._save_index(index)
            return library._public_record(record)
        except Exception:
            shutil.rmtree(staged, ignore_errors=True)
            shutil.rmtree(published, ignore_errors=True)
            if new_character:
                shutil.rmtree(directory, ignore_errors=True)
            raise


@library._file_operation
def review_revision(character_id: str, revision_id: str, approved: bool) -> dict:
    if not isinstance(approved, bool):
        raise ValueError("Review must explicitly approve or unapprove the revision.")
    with library._LOCK:
        index = library._load_index()
        record = next((r for r in index["characters"] if r["id"] == character_id), None)
        if record is None:
            raise ValueError("Saved character not found.")
        revision = _revision(record, revision_id)
        if approved:
            for view, image in revision["images"].items():
                if hashlib.sha256(image_path(character_id, revision_id, view).read_bytes()).hexdigest() != image["sha256"]:
                    raise ValueError("Pack image changed on disk. Import a new revision.")
        revision.update(approved=approved, reviewed_at=time.time())
        record["updated_at"] = time.time()
        library._save_index(index)
        return library._public_record(record)


def validate_pack_references(references: list[dict], *, require_files: bool) -> None:
    """Reject partial/mixed revisions and stale approvals at the H3 boundary."""
    groups: dict[str, list[dict]] = {}
    for item in references:
        marked = item.get("reference_pack_revision_id") or item.get("reference_pack_view")
        if require_files and item.get("library_character_id") and item["type"] != "audio":
            try:
                marked = marked or library._get_character_record(item["library_character_id"]).get("reference_pack")
            except ValueError:
                pass
        if marked:
            for key in ("reference_pack_revision_id", "reference_pack_view"):
                if not isinstance(item.get(key), str) or not item[key]:
                    raise ValueError("Select a complete Reference Pack revision from the character library.")
            character_id = item.get("library_character_id")
            if not isinstance(character_id, str) or not character_id:
                raise ValueError("Reference Pack views need a saved character identity.")
            groups.setdefault(character_id, []).append(item)
    for character_id, items in groups.items():
        visual = [r for r in references if r.get("library_character_id") == character_id and r["type"] != "audio"]
        if len(items) != 5 or len(visual) != 5 or {r.get("reference_pack_view") for r in items} != VIEWS.keys():
            raise ValueError("Select all five views of one Reference Pack revision together.")
        revision_ids = {r.get("reference_pack_revision_id") for r in items}
        if len(revision_ids) != 1 or not next(iter(revision_ids)):
            raise ValueError("Do not mix Reference Pack revisions for the same character.")
        for item in items:
            if item["type"] != "image" or item.get("image_intent", "identity") != "identity" or item.get("refmod_path") or item.get("remove_background"):
                raise ValueError("Reference Pack views must retain their reviewed identity images.")
        if not require_files:
            continue
        record = library._get_character_record(character_id)
        revision_id = next(iter(revision_ids))
        revision = _revision(record, revision_id)
        if not revision["approved"]:
            raise ValueError("Review and approve this Reference Pack revision before using it.")
        for item in items:
            view = item["reference_pack_view"]
            path = image_path(character_id, revision_id, view)
            if Path(item["path"]).resolve() != path or hashlib.sha256(path.read_bytes()).hexdigest() != revision["images"][view]["sha256"]:
                raise ValueError("Reference Pack files no longer match the selected revision.")
            # The library, not arbitrary submitted names, owns pack identity labels.
            item["character_name"] = item["role"] = record["name"]


def create_router():
    from fastapi import APIRouter, HTTPException, Request
    from starlette.concurrency import run_in_threadpool
    from .win_safe_files import share_delete_file_response

    router = APIRouter(prefix="/api/v1/characters")

    async def save(request, character_id=None):
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "Expected a JSON object.")
        try:
            return await run_in_threadpool(create_revision, name=body.get("name", ""), images=body.get("images"),
                                          character_id=character_id, base_revision_id=body.get("base_revision_id"),
                                          label=body.get("label", ""))
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    @router.post("/reference-packs")
    async def create(request: Request):
        return await save(request)

    @router.post("/{character_id}/reference-packs")
    async def revise(character_id: str, request: Request):
        return await save(request, character_id)

    @router.put("/{character_id}/reference-packs/{revision_id}/review")
    async def review(character_id: str, revision_id: str, request: Request):
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "Expected a JSON object.")
        try:
            return await run_in_threadpool(review_revision, character_id, revision_id, body.get("approved"))
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @router.get("/{character_id}/reference-packs/{revision_id}/images/{view}")
    def media(character_id: str, revision_id: str, view: str):
        try:
            return share_delete_file_response(str(image_path(character_id, revision_id, view)))
        except ValueError as error:
            raise HTTPException(404, str(error)) from error

    return router


def pack_prompt_definitions(items: list[dict], subjects: dict[int, int]) -> dict[int, str]:
    """One definition per pack, using actual canonical Picture ordinals."""
    groups = {}
    picture = 0
    for index, item in enumerate(items):
        if item["type"] == "image":
            picture += 1
        if item.get("reference_pack_revision_id"):
            group = groups.setdefault(item["library_character_id"], [])
            group.append((index, picture, item))
    definitions = {}
    for entries in groups.values():
        first, _, item = entries[0]
        sources = "; ".join(f"<Picture {number}>: {VIEWS[row['reference_pack_view']]}" for _, number, row in entries)
        name = " ".join(str(item.get("character_name") or item.get("role") or "the character").split())
        definitions[first] = (
            f"<Subject {subjects[first]}> is {name}, ONE character jointly defined by five views: {sources}. "
            "The close-up defines facial identity; front and three-quarter views define body proportions, "
            "hair and clothing; the back view defines rear appearance. Preserve a single consistent identity "
            "and outfit across angles. These are complementary identity references, not five actors, "
            "a shot sequence, poses to reproduce, or target keyframes. Their backgrounds and framing do not define the scene."
        )
    return definitions
