"""HTTP boundary for optional music preparation, training and style imports."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
import threading
import zipfile

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from . import music_training as projects
from .music_styles import load_style, list_styles, save_style, style_directory

_export_lock = threading.Lock()


def create_router(submit, jobs):
    router = APIRouter()

    def current(project):
        job_id = project.get("job_id")
        if project.get("status") in {"queued", "preparing", "training", "auditioning"} and job_id not in jobs:
            project = projects.update_project(project["id"], status="interrupted",
                auditions=[{**row, 'status': 'cancelled', 'error': 'The app restarted during this audition; the checkpoint is saved. Retry this sample.'}
                           if row['status'] == 'running' else row for row in project.get('auditions', [])],
                message="The app restarted. Prepared data and saved checkpoints are available to resume.")
        if project.get("status") == "queued" and jobs.get(job_id, {}).get("status") == "cancelled":
            project = projects.update_project(project["id"], status="cancelled", message="Music job cancelled before starting")
        return {**project, 'reviewed_track_ids': [track['id'] for track in project['tracks']
            if project.get('reviews', {}).get(track['id']) == projects.review_fingerprint(track)]}

    @router.get("/api/v1/music-styles")
    def styles():
        return {"styles": list_styles()}

    @router.get("/api/v1/music-training/projects")
    def all_projects():
        return {"projects": [current(project) for project in projects.list_projects()]}

    @router.get("/api/v1/music-training/projects/{project_id}")
    def project(project_id: str):
        try:
            return current(projects.get_project(project_id))
        except ValueError as error:
            raise HTTPException(404, str(error)) from error

    @router.post("/api/v1/music-training/projects")
    async def create_project(request: Request):
        body = await request.json()
        try:
            if not isinstance(body, dict):
                raise ValueError("Expected a music project object")
            return current(await asyncio.to_thread(projects.create_project, body.get("name"), body.get("trigger"), body.get("tracks")))
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    @router.post("/api/v1/music-training/projects/{project_id}/{operation}")
    async def start(project_id: str, operation: str, request: Request):
        if operation not in {"prepare", "train", "audition-style", "reconstruct", "prepare-audio", "adapt-audio", "align-lyrics", "fork", "review-data", "review-track", "render-audition"}:
            raise HTTPException(404, "Unknown music training operation")
        body = await request.json()
        try:
            if not isinstance(body, dict):
                raise ValueError("Expected music job options")
            project = projects.get_project(project_id)
            if operation == "reconstruct":
                return submit(operation, project_id, body)
            project = current(project)
            if operation == 'review-track':
                return current(await asyncio.to_thread(projects.review_track, project_id, body.get('track_id'), body.get('reviewed') is True))
            if operation == 'render-audition':
                from .music_auditions import audition_options
                branch = body.get('branch')
                field = 'audio_checkpoints' if branch == 'audio' else 'checkpoints'
                if branch not in {'style', 'audio'} or body.get('checkpoint') not in {row['file'] for row in project.get(field, [])}:
                    raise ValueError('Choose a saved style or audio checkpoint')
                settings = audition_options(body.get('audition', project.get('audition_settings')))
                if not settings['enabled']:
                    raise ValueError('Set up a fixed audition request first')
                return submit(operation, project_id, {'branch': branch, 'checkpoint': body['checkpoint'], 'audition': settings})
            if operation == 'fork':
                return await asyncio.to_thread(projects.fork_project, project_id)
            if operation == "audition-style":
                from models.TTS.yue2.artist_adapter import read_upstream_adapter
                from models.TTS.yue2.music_assets import ensure_asset
                filename = str(body.get("checkpoint") or "")
                audio_filename = str(body.get('audio_checkpoint') or '')
                if audio_filename:
                    audio_row = next((row for row in project.get('audio_checkpoints', []) if row['file'] == audio_filename), None)
                    if not audio_row:
                        raise ValueError('Choose a saved audio checkpoint')
                    filename = audio_row['conditioning_checkpoint']
                if filename and filename not in {item["file"] for item in project.get("checkpoints", [])} or not filename and not audio_filename:
                    raise ValueError("Choose a checkpoint saved by this training project")
                def publish():
                    if filename:
                        if audio_filename:
                            from .music_styles import file_digest
                            expected = project.get('audio_training_options', {}).get('conditioning_sha256')
                            if file_digest(projects.project_directory(project_id) / 'checkpoints' / filename) != expected:
                                raise ValueError('The music checkpoint used for audio training has changed; cannot export a mismatched pair')
                        ar = read_upstream_adapter(projects.project_directory(project_id) / "checkpoints" / filename, "ar")
                    else:
                        import torch
                        from models.TTS.yue2.artist_adapter import target_shapes
                        ar = {name + suffix: torch.zeros(shape, dtype=torch.float32, device='cpu')
                              for name, inputs, outputs in target_shapes('ar')
                              for suffix, shape in (('.A', (1, inputs)), ('.B', (outputs, 1)))}
                    nar = read_upstream_adapter(projects.project_directory(project_id) / 'audio_checkpoints' / audio_filename
                                                if audio_filename else ensure_asset("nar"), "nar")
                    checkpoint_label = 'audio-' + audio_filename if audio_filename else filename
                    label = f"{project['name'][:65]} · {checkpoint_label.removesuffix('.safetensors')}"
                    return save_style(label, project["trigger"], ar, nar,
                                      training={"project_id": project_id, "checkpoint": filename or 'base-ar',
                                                "audio_checkpoint": audio_filename,
                                                "audio_contract": project.get('audio_training_options') if audio_filename else None,
                                                "dataset_digest": project["dataset_digest"]})
                return await asyncio.to_thread(publish)
            options = projects.training_options(body) if operation == "train" else projects.audio_training_options(body) if operation == 'adapt-audio' else {}
            return submit(operation, project_id, options)
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    @router.get('/api/v1/music-training/projects/{project_id}/recordings/{track_id}')
    def recording(project_id: str, track_id: str):
        try:
            project = projects.get_project(project_id)
            track = next((track for track in project['tracks'] if track['id'] == track_id), None)
            if track is None:
                raise ValueError('Recording not found')
            path = Path(track['audio_path'])
            if not path.is_file() or path.suffix.lower() not in projects.AUDIO_EXTENSIONS:
                raise ValueError('Recording not found')
            return FileResponse(path)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error

    @router.get('/api/v1/music-training/projects/{project_id}/auditions/{audition_id}')
    def audition_audio(project_id: str, audition_id: str):
        try:
            project = projects.get_project(project_id)
            row = next((row for row in project.get('auditions', []) if row['id'] == audition_id and row['status'] == 'completed'), None)
            if row is None:
                raise ValueError('Audition not found')
            directory = projects.project_directory(project_id) / 'auditions'
            path = (directory / row['file']).resolve()
            if path.parent != directory or path.suffix != '.flac' or not path.is_file():
                raise ValueError('Audition not found')
            return FileResponse(path, media_type='audio/flac')
        except ValueError as error:
            raise HTTPException(404, str(error)) from error

    @router.post("/api/v1/music-styles/import")
    async def import_style(name: str = Form(""), trigger: str = Form(""), ar: UploadFile = File(...),
                           nar: UploadFile | None = File(None)):
        def convert(directory):
            from models.TTS.yue2.artist_adapter import read_upstream_adapter
            from models.TTS.yue2.music_assets import ensure_asset
            if Path(ar.filename or "").suffix.lower() == ".zip":
                # Read only the three known entries. Do not extract archive paths.
                with zipfile.ZipFile(ar.file) as archive:
                    entries = archive.infolist()
                    if len(entries) != 3 or {item.filename for item in entries} != {"style.json", "ar.safetensors", "nar.safetensors"}:
                        raise ValueError("A music style bundle must contain its manifest and AR/NAR weights only")
                    for item in entries:
                        limit = 65536 if item.filename == "style.json" else 1024**3
                        if item.file_size > limit:
                            raise ValueError("The music style bundle exceeds the supported size")
                        with archive.open(item) as source, (Path(directory) / item.filename).open("wb") as destination:
                            import shutil
                            shutil.copyfileobj(source, destination, length=1024 * 1024)
                # Reuse exactly the same revision and checksum checks as generation.
                bundle = load_style(Path(directory).name, root=Path(directory).parent, verify=True)
                return save_style(name or bundle["name"], trigger or bundle["trigger"],
                                  read_upstream_adapter(Path(directory) / "ar.safetensors", "ar"),
                                  read_upstream_adapter(Path(directory) / "nar.safetensors", "nar"), training=bundle.get("training"))
            inputs = {}
            for branch, upload in (("ar", ar), ("nar", nar)):
                if upload is None:
                    inputs[branch] = ensure_asset("nar")
                    continue
                suffix = Path(upload.filename or "").suffix.lower()
                if suffix not in {".pt", ".safetensors"}:
                    raise ValueError("Choose a .pt or .safetensors YuE2 adapter")
                path = Path(directory) / (branch + suffix)
                total = 0
                with path.open("wb") as destination:
                    while chunk := upload.file.read(1024 * 1024):
                        total += len(chunk)
                        if total > 1024**3:
                            raise ValueError("Each music adapter must be smaller than 1 GB")
                        destination.write(chunk)
                inputs[branch] = path
            return save_style(name, trigger, read_upstream_adapter(inputs["ar"], "ar"),
                              read_upstream_adapter(inputs["nar"], "nar"))
        try:
            # TemporaryDirectory owns only its uniquely-created import directory.
            with tempfile.TemporaryDirectory(prefix="maestro-music-import-") as directory:
                return await asyncio.to_thread(convert, directory)
        except (ValueError, OSError, RuntimeError, zipfile.BadZipFile) as error:
            raise HTTPException(400, str(error)) from error

    @router.get("/api/v1/music-styles/{style_id}/export")
    def export_style(style_id: str):
        try:
            manifest = load_style(style_id, verify=True)
            directory = style_directory(style_id)
            output = directory / "music-style.zip"
            # Saved bundles are immutable. Publish the archive once, so a
            # second download never truncates a file already being streamed.
            with _export_lock:
                if not output.is_file():
                    with tempfile.NamedTemporaryFile(dir=directory, prefix=".export-", suffix=".zip", delete=False) as handle:
                        temporary = Path(handle.name)
                    try:
                        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
                            for filename in ("style.json", manifest["ar"]["file"], manifest["nar"]["file"]):
                                archive.write(directory / filename, filename)
                        os.replace(temporary, output)
                    finally:
                        temporary.unlink(missing_ok=True)
            return FileResponse(output, media_type="application/zip", filename=f"{style_id}-music-style.zip")
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    return router
