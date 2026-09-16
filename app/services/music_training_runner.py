"""Serialize optional music training with Maestro's existing GPU jobs."""
import threading
import time
import traceback
import uuid

from . import music_training as projects
from .job_lifecycle import (finish_job, generation_slot, is_cancel_requested, request_cancel,
                           register_abort_state, unregister_abort_state, try_start, update_job)


class MusicTrainingRunner:
    def __init__(self, jobs, generation_lock, active_states, release_models, workspace, output_directory=None):
        self.jobs, self.generation_lock, self.active_states = jobs, generation_lock, active_states
        self.release_models, self.workspace = release_models, workspace
        self.output_directory = output_directory
        self.submission_lock = threading.RLock()
        self.project_workers = set()

    def submit(self, operation, project_id, options):
        with self.submission_lock:
            if operation not in {"prepare", "train", "reconstruct", "prepare-audio", "adapt-audio", "align-lyrics", "review-data", "render-audition"}:
                raise ValueError("Unknown music operation")
            project = projects.get_project(project_id)
            active = self.jobs.get(project.get("job_id"), {})
            if project_id in self.project_workers or active.get("status") in {"held", "queued", "running"} or (
                active and project.get("status") in {"queued", "preparing", "training", "auditioning"}
            ):
                # Cancellation marks the queue row terminal immediately, while
                # its worker may still be saving the final checkpoint.
                raise ValueError("This music project already has a queued or running job")
            if operation in {"train", "adapt-audio", "prepare-audio", "align-lyrics"} and not project.get("prepared"):
                raise ValueError("Prepare this project's audio before starting training")
            if operation == 'adapt-audio' and not project.get('audio_prepared'):
                raise ValueError('Prepare source sound before audio adaptation')
            diagnostic = operation == "reconstruct"
            if operation in {'train', 'adapt-audio', 'render-audition'}:
                from .music_auditions import audition_options
                settings = audition_options(options.get('audition'))
                saved_settings = settings if settings['enabled'] else {**(project.get('audition_settings') or {}), 'enabled': False}
                projects.update_project(project_id, audition_settings=saved_settings)
            if diagnostic:
                from .music_reconstruction import reconstruction_options
                options = reconstruction_options(options, project)
                if self.output_directory is None:
                    raise ValueError("Music reconstruction output directory is not configured")
            job_id = uuid.uuid4().hex[:8]
            job = {"id": job_id, "kind": "music_reconstruction" if diagnostic else "music_training", "status": "queued", "progress": 0,
                   "step": 0, "total_steps": 0, "phase": "", "message": f"Music {operation} queued",
                   "created_at": time.time(), "output_files": [], "error": None,
                   "workspace": self.workspace(), "params": {"model_type": "yue2", "generation_mode": "audio",
                       "music_training_project": project_id, "operation": operation, "options": options}}
            self.jobs[job_id] = job
            if diagnostic:
                # The diagnostic reads checkpoints but never changes training's
                # status, progress, latest job, or resume state.
                job["reconstruction_output_dir"] = str(self.output_directory(job["workspace"]))
            else:
                projects.update_project(project_id, job_id=job_id, status="queued", progress=0,
                                        message=f"Waiting to {operation} music")
            self.project_workers.add(project_id)
            try:
                threading.Thread(target=self.run, args=(job_id,), daemon=False, name="Music training").start()
            except Exception as error:
                self.project_workers.discard(project_id)
                if not diagnostic:
                    projects.update_project(project_id, status="failed", message=str(error))
                finish_job(job, "failed", error=str(error), message=str(error))
                raise
            return {"job_id": job_id, "status": "queued", "project_id": project_id}

    def run(self, job_id):
        project_id = self.jobs[job_id]["params"]["music_training_project"]
        try:
            if self.jobs[job_id]["params"]["operation"] == "reconstruct":
                self._run_reconstruction(job_id)
            else:
                self._run(job_id)
        finally:
            with self.submission_lock:
                self.project_workers.discard(project_id)

    def _run_reconstruction(self, job_id):
        job = self.jobs[job_id]
        state = {"abort": False}
        cancelled = lambda: bool(state.get("abort")) or is_cancel_requested(job)
        with generation_slot(self.generation_lock, job) as acquired:
            if not acquired:
                return
            try:
                if not try_start(job, message="Reconstructing source music", phase="Music reconstruction"):
                    return
                if not register_abort_state(job, job_id, self.active_states, state):
                    return
                self.release_models()
                if cancelled():
                    raise InterruptedError("Music reconstruction cancelled")

                def report(message, percent=None):
                    changes = {"message": str(message)}
                    if percent is not None:
                        changes["progress"] = max(0, min(100, float(percent)))
                    update_job(job, **changes)
                    print(f"[Music reconstruct] {message}", flush=True)

                from models.TTS.yue2.reconstruction import reconstruct_project
                result = reconstruct_project(
                    projects.get_project(job["params"]["music_training_project"]),
                    job["params"]["options"], job["reconstruction_output_dir"], job_id,
                    report=report, cancelled=cancelled,
                    publish=lambda files: update_job(job, output_files=list(files)),
                )
                if cancelled():
                    raise InterruptedError("Music reconstruction cancelled")
                finish_job(job, "completed", progress=100, phase="", output_files=result["files"],
                           reconstruction_report=result["report"], message="Source reconstructions ready to compare")
            except InterruptedError:
                request_cancel(job, job_id=job_id, active_states=self.active_states)
            except Exception as error:
                traceback.print_exc()
                finish_job(job, "failed", error=str(error), message=str(error))
            finally:
                unregister_abort_state(job_id, self.active_states, state)

    def _run(self, job_id):
        job = self.jobs[job_id]
        project_id, operation = job["params"]["music_training_project"], job["params"]["operation"]
        preparation = operation in {'prepare', 'prepare-audio', 'align-lyrics', 'review-data'}
        abort_state = {"abort": False}
        cancelled = lambda: bool(abort_state.get("abort")) or is_cancel_requested(job)
        with generation_slot(self.generation_lock, job) as acquired:
            if not acquired:
                projects.update_project(project_id, status="cancelled", message="Music job cancelled before starting")
                return
            try:
                if not try_start(job, message=f"Starting music {operation}", phase="Music preparation" if preparation else "Music training"):
                    projects.update_project(project_id, status="cancelled", message="Music job cancelled before starting")
                    return
                if not register_abort_state(job, job_id, self.active_states, abort_state):
                    projects.update_project(project_id, status="cancelled", message="Music job cancelled before starting")
                    return
                self.release_models()
                if cancelled():
                    raise InterruptedError("Music job cancelled")
                projects.update_project(project_id, status="preparing" if preparation else "training")
                def report(message, percent=None):
                    if cancelled():
                        # Training finishes its current optimizer step and writes
                        # the checkpoint before observing cancellation itself.
                        return
                    changes = {"message": str(message)}
                    if percent is not None:
                        changes["progress"] = max(0, min(100, float(percent)))
                    update_job(job, **changes)
                    projects.update_project(project_id, **changes)
                    print(f"[Music {operation}] {message}", flush=True)
                project = projects.get_project(project_id)
                audition_failures = 0
                if operation == "prepare":
                    from models.TTS.yue2.music_tokenizer import prepare_project
                    prepare_project(project, report=report, cancelled=cancelled)
                    from .music_data_review import sequence_coverage
                    sequence_coverage(project, report=report, cancelled=cancelled)
                elif operation == 'prepare-audio':
                    from models.TTS.yue2.audio_training_data import prepare_audio
                    prepare_audio(project, report=report, cancelled=cancelled)
                elif operation == 'adapt-audio':
                    from models.TTS.yue2.acoustic_training import train_audio
                    from .music_auditions import train_with_auditions
                    audition_failures = train_with_auditions(project_id, job['params']['options'], train_audio, 'audio', report=report, cancelled=cancelled)
                elif operation == 'align-lyrics':
                    from models.TTS.yue2.lyric_alignment import align_project
                    align_project(project, report=report, cancelled=cancelled)
                elif operation == 'review-data':
                    from .music_data_review import draft_lyrics
                    draft_lyrics(project, report=report, cancelled=cancelled)
                elif operation == 'render-audition':
                    from .music_auditions import render_checkpoint
                    options = job['params']['options']
                    projects.update_project(project_id, status='auditioning')
                    render_checkpoint(project, options['branch'], options['checkpoint'], options['audition'], report=report, cancelled=cancelled)
                else:
                    from models.TTS.yue2.artist_training import train_project
                    from .music_auditions import train_with_auditions
                    audition_failures = train_with_auditions(project_id, job['params']['options'], train_project, 'style', report=report, cancelled=cancelled)
                if cancelled():
                    raise InterruptedError("Music job cancelled; saved work is available to resume")
                message = {'prepare': 'Audio tokens prepared', 'prepare-audio': 'Source sound prepared',
                           'align-lyrics': 'Lyric alignment prepared', 'review-data': 'Local lyric drafts ready; listen and review before applying',
                           'render-audition': 'Checkpoint audition ready to play'}.get(operation, 'Training complete; audition a saved checkpoint')
                if audition_failures:
                    message += f'; {audition_failures} audition(s) failed and can be retried'
                if operation == 'align-lyrics' and not projects.get_project(project_id).get('alignment', {}).get('ready'):
                    message = 'Lyric alignment needs review before timing-guided training'
                projects.update_project(project_id, status="prepared" if preparation else "completed", progress=100, message=message)
                finish_job(job, "completed", progress=100, phase="", message=message)
            except InterruptedError as error:
                projects.update_project(project_id, status="cancelled", message=str(error))
                request_cancel(job, job_id=job_id, active_states=self.active_states)
            except Exception as error:
                traceback.print_exc()
                projects.update_project(project_id, status="failed", message=str(error))
                finish_job(job, "failed", error=str(error), message=str(error))
            finally:
                unregister_abort_state(job_id, self.active_states, abort_state)
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
