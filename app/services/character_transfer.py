"""Background import/export routes for portable characters and H3 RefMods."""
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlparse

from fastapi import APIRouter, HTTPException, Request, UploadFile, File

from . import character_library as library
from .refmod import MAX_FILE_BYTES, inspect_refmod

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _headers() -> dict:
    headers = {"User-Agent": "Maestro-Character-Import"}
    try:
        from huggingface_hub import get_token
        token = get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    except ImportError:
        pass
    return headers


def remote_files(url: str) -> dict:
    import requests

    parsed = urlparse(str(url).strip())
    if parsed.scheme != "https" or parsed.hostname not in {"huggingface.co", "www.huggingface.co", "hf.co"} or parsed.username or parsed.password:
        raise ValueError("Paste an HTTPS Hugging Face repository or file URL. Files from other sites can be imported with Choose file.")
    parts = [unquote(part) for part in parsed.path.strip("/").split("/")]
    if len(parts) < 2 or any(part in {"", ".", ".."} or "\\" in part for part in parts):
        raise ValueError("Invalid Hugging Face URL.")
    repo = "/".join(parts[:2])
    revision = parts[3] if len(parts) >= 4 and parts[2] in {"blob", "resolve", "tree"} else "main"
    filename = "/".join(parts[4:]) if len(parts) >= 5 and parts[2] in {"blob", "resolve"} else None
    response = requests.get(f"https://huggingface.co/api/models/{quote(repo, safe='/')}/revision/{quote(revision, safe='')}",
                            headers=_headers(), timeout=20)
    response.raise_for_status()
    data = response.json()
    files = [item["rfilename"] for item in data.get("siblings", [])
             if isinstance(item, dict) and str(item.get("rfilename", "")).lower().endswith(".safetensors")]
    if filename:
        if filename not in files:
            raise ValueError("That file was not found in the Hugging Face repository.")
        files = [filename]
    if not files:
        raise ValueError("This repository has no safetensors character files.")
    return {"repo": repo, "revision": data.get("sha") or revision, "files": files[:5000]}


def download_refmod(url: str, filename: str, destination: Path, update) -> Path:
    import requests

    listing = remote_files(url)
    if not filename and len(listing["files"]) == 1:
        filename = listing["files"][0]
    if filename not in listing["files"] or ".." in PurePosixPath(filename).parts:
        raise ValueError("Choose the character file to import.")
    resolved = f"https://huggingface.co/{quote(listing['repo'], safe='/')}/resolve/{quote(listing['revision'], safe='')}/{quote(filename, safe='/')}"
    with requests.get(resolved, stream=True, headers=_headers(), timeout=(15, 60)) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length") or 0)
        if total > MAX_FILE_BYTES:
            raise ValueError("This file is larger than 256 MB. It may be a LoRA or checkpoint instead of a RefMod.")
        downloaded = 0
        with destination.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > MAX_FILE_BYTES:
                    raise ValueError("Character downloads must be smaller than 256 MB.")
                output.write(chunk)
                update(f"Downloading character… {downloaded / (1024*1024):.1f} MB")
        if total and downloaded != total:
            raise ValueError("The character download was incomplete. Try again.")
    inspect_refmod(destination)
    return destination


def _start(operation) -> dict:
    job_id = uuid.uuid4().hex
    with _lock:
        # Completed records are short lived; no media is deleted by this cleanup.
        for key, job in list(_jobs.items()):
            if job["status"] in {"completed", "failed"} and time.time() - job["updated_at"] > 3600:
                _jobs.pop(key, None)
        _jobs[job_id] = {"id": job_id, "status": "running", "message": "Preparing character…", "updated_at": time.time()}

    def update(message):
        with _lock:
            _jobs[job_id].update(message=message, updated_at=time.time())

    def run():
        try:
            result = operation(update)
            with _lock:
                _jobs[job_id].update(status="completed", result=result, message="Character ready", updated_at=time.time())
        except Exception as error:
            with _lock:
                _jobs[job_id].update(status="failed", error=str(error), message="Character transfer failed", updated_at=time.time())
    threading.Thread(target=run, daemon=True, name=f"character-{job_id[:8]}").start()
    return {"id": job_id, "status": "running"}


def build_router(run_codec) -> APIRouter:
    """The application supplies its shared generation lock/residency guard."""
    router = APIRouter()

    def import_file(path, name, update):
        from .character_codec import preview_refmod
        update("Reading character file…")
        metadata = inspect_refmod(path)["refmod"]
        return library.import_character(path, name=name, decode_preview=lambda *args:
            run_codec(lambda: preview_refmod(*args, metadata=metadata, update=update), update))

    @router.post("/api/v1/characters/{character_id}/images/recover")
    def recover_images(character_id: str, force: bool = False):
        try:
            library.get_character(character_id)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error
        def operation(update):
            from .character_codec import recover_refmod_images
            character = library.recover_character_images(character_id, force=force, update=update,
                decode_views=lambda *args: run_codec(lambda: recover_refmod_images(*args, update=update), update))
            return {"character": character}
        return _start(operation)

    @router.put("/api/v1/characters/{character_id}/images")
    async def select_images(character_id: str, request: Request):
        import asyncio
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "Expected a character image selection.")
        try:
            return await asyncio.to_thread(library.select_character_images, character_id,
                body.get("selected_ids"), body.get("cover_id"))
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @router.get("/api/v1/characters/{character_id}/images/{view_id}")
    def image_file(character_id: str, view_id: str, download: bool = False):
        from .win_safe_files import share_delete_file_response
        try:
            path = library.get_character_image(character_id, view_id)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error
        return share_delete_file_response(str(path), media_type="image/png",
            **({"filename": path.name} if download else {}))

    @router.get("/api/v1/characters/{character_id}/images.zip")
    def download_images(character_id: str):
        from .win_safe_files import share_delete_file_response
        try:
            path = library.export_character_images(character_id)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        return share_delete_file_response(str(path), filename=path.name, media_type="application/zip")

    @router.get("/api/v1/character-transfers/{job_id}")
    def status(job_id: str):
        with _lock:
            job = _jobs.get(job_id)
            if not job:
                raise HTTPException(404, "Character transfer not found.")
            return dict(job)

    @router.post("/api/v1/characters/import")
    async def upload(file: UploadFile = File(...), metadata_file: UploadFile | None = File(None)):
        if not str(file.filename or "").lower().endswith(".safetensors"):
            raise HTTPException(400, "Choose a .maestro.safetensors character or an H3 RefMod .safetensors file.")
        root = Path.cwd() / "uploads" / "character_imports"
        root.mkdir(parents=True, exist_ok=True)
        destination = root / f"{uuid.uuid4().hex}.safetensors"
        total = 0
        try:
            with destination.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_FILE_BYTES:
                        raise HTTPException(413, "Character files must be smaller than 256 MB.")
                    output.write(chunk)
            if metadata_file:
                data = await metadata_file.read(2 * 1024 * 1024 + 1)
                if len(data) > 2 * 1024 * 1024:
                    raise HTTPException(413, "RefMod metadata is too large.")
                destination.with_suffix(".json").write_bytes(data)
            inspect_refmod(destination)
        except Exception as error:
            destination.unlink(missing_ok=True)
            destination.with_suffix(".json").unlink(missing_ok=True)
            if isinstance(error, HTTPException):
                raise
            raise HTTPException(400, str(error)) from error
        finally:
            await file.close()
            if metadata_file:
                await metadata_file.close()

        def operation(update):
            try:
                return {"character": import_file(destination, "", update)}
            finally:
                destination.unlink(missing_ok=True)
                destination.with_suffix(".json").unlink(missing_ok=True)
        return _start(operation)

    @router.get("/api/v1/characters/remote-files")
    def find_remote_files(url: str):
        try:
            return remote_files(url)
        except Exception as error:
            raise HTTPException(400, str(error)) from error

    @router.post("/api/v1/characters/import-url")
    async def import_url(request: Request):
        body = await request.json()
        url, filename = str(body.get("url") or ""), str(body.get("filename") or "")
        def operation(update):
            root = Path.cwd() / "uploads" / "character_imports"
            root.mkdir(parents=True, exist_ok=True)
            destination = root / f"{uuid.uuid4().hex}.safetensors"
            try:
                download_refmod(url, filename, destination, update)
                return {"character": import_file(destination, str(body.get("name") or ""), update)}
            finally:
                destination.unlink(missing_ok=True)
        return _start(operation)

    @router.post("/api/v1/characters/{character_id}/export")
    def export(character_id: str):
        try:
            library.get_character(character_id)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error
        def operation(update):
            from .character_codec import encode_character_visual
            path = library.export_character(character_id, encode_visual=lambda *args:
                run_codec(lambda: encode_character_visual(*args), update))
            return {"character": library.get_character(character_id), "filename": path.name,
                    "url": f"/api/v1/characters/{character_id}/export-file"}
        return _start(operation)

    @router.get("/api/v1/characters/{character_id}/export-file")
    def download(character_id: str):
        from .refmod import character_filename
        from .win_safe_files import share_delete_file_response
        try:
            record = library.get_character(character_id)
            path = library._character_directory(character_id) / character_filename(record["name"])
        except ValueError as error:
            raise HTTPException(404, str(error)) from error
        if not path.is_file():
            raise HTTPException(404, "Export this character before downloading it.")
        return share_delete_file_response(str(path), filename=path.name, media_type="application/octet-stream")

    @router.put("/api/v1/characters/{character_id}/voice")
    async def update_voice(character_id: str, request: Request):
        import asyncio
        body = await request.json()
        try:
            return await asyncio.to_thread(library.attach_character_voice, character_id, str(body.get("voice_path") or ""))
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    return router
