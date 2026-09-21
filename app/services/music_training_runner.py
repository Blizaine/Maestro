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
        self.job_projects = {}

    def submit(self, operation, project_id, options):
        with self.submission_lock, projects._lock:
            if operation not in {"auto-train", "reconstruct-pair", "prepare-pair", "adapt-pair", "analyze-songs", "build-dataset", "prepare", "train", "train-joint", "reconstruct", "prepare-audio", "adapt-audio", "align-lyrics", "review-data", "render-audition"}:
                raise ValueError("Unknown music operation")
            project = projects.get_project(project_id)
            automatic = operation == 'auto-train'
            if not automatic and bool(project.get('preparation_draft')) != (operation in {'analyze-songs', 'build-dataset'}):
                raise ValueError('Create a reviewed training dataset from this draft first' if project.get('preparation_draft') else 'Choose a song preparation draft')
            if operation == 'build-dataset':
                from .music_dataset import dataset_selection
                dataset_selection(project)
            active = self.jobs.get(project.get("job_id"), {})
            owned = {project_id}
            if automatic:
                from .music_auto_training import auto_options, linked_projects
                options = auto_options(options, project)
                owned = linked_projects(project)
                for linked_id in owned:
                    linked = projects.get_project(linked_id)
                    if linked_id in self.project_workers or self.jobs.get(linked.get('job_id'), {}).get('status') in {'held', 'queued', 'running'}:
                        raise ValueError('A linked music project already has a queued or running job')
            if project_id in self.project_workers or active.get("status") in {"held", "queued", "running"} or (
                active and project.get("status") in {"queued", "preparing", "training", "auditioning"}
            ):
                # Cancellation marks the queue row terminal immediately, while
                # its worker may still be saving the final checkpoint.
                raise ValueError("This music project already has a queued or running job")
            if operation in {"train", "train-joint", "adapt-audio", "prepare-audio", "align-lyrics"} and not project.get("prepared"):
                raise ValueError("Prepare this project's audio before starting training")
            if operation in {'adapt-audio', 'train-joint'} and not project.get('audio_prepared'):
                raise ValueError('Prepare source sound before audio adaptation')
            if operation == 'adapt-pair' and not project.get('pair_prepared'):
                raise ValueError('Prepare sound-adaptation data first')
            diagnostic = operation in {"reconstruct", "reconstruct-pair"}
            if operation in {'train', 'train-joint', 'adapt-audio', 'render-audition'}:
                from .music_auditions import audition_options
                settings = audition_options(options.get('audition'))
                saved_settings = settings if settings['enabled'] else {**(project.get('audition_settings') or {}), 'enabled': False}
                projects.update_project(project_id, audition_settings=saved_settings)
            if diagnostic:
                from .music_reconstruction import reconstruction_options
                if operation == 'reconstruct-pair':
                    from .music_pair_adaptation import comparison_options
                    options = comparison_options(options, project)
                else:
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
            if automatic:
                from .music_auto_training import update_auto
                update_auto(project_id, version=1, **options, status='queued',
                            stage=(project.get('auto_training') or {}).get('stage', 'analyze-songs' if project.get('preparation_draft') else 'prepare-pair'))
                for linked_id in owned:
                    projects.update_project(linked_id, job_id=job_id, auto_training_root=project_id,
                                            status='queued', message='Auto training is waiting for the GPU')
            if diagnostic:
                # The diagnostic reads checkpoints but never changes training's
                # status, progress, latest job, or resume state.
                job["reconstruction_output_dir"] = str(self.output_directory(job["workspace"]))
            else:
                workflow = ('author' if operation in {'prepare-pair', 'adapt-pair'} else
                            'advanced' if operation in {'adapt-audio', 'train-joint'} else
                            'style' if operation == 'train' else project.get('training_workflow'))
                projects.update_project(project_id, job_id=job_id, status="queued", progress=0,
                                        message=f"Waiting to {operation} music", training_workflow=workflow)
            self.project_workers.update(owned)
            self.job_projects[job_id] = owned
            try:
                threading.Thread(target=self.run, args=(job_id,), daemon=False, name="Music training").start()
            except Exception as error:
                self.project_workers.difference_update(self.job_projects.pop(job_id, owned))
                if automatic:
                    update_auto(project_id, status='failed')
                if not diagnostic:
                    projects.update_project(project_id, status="failed", message=str(error))
                finish_job(job, "failed", error=str(error), message=str(error))
                raise
            return {"job_id": job_id, "status": "queued", "project_id": project_id}

    def run(self, job_id):
        project_id = self.jobs[job_id]["params"]["music_training_project"]
        try:
            if self.jobs[job_id]['params']['operation'] == 'auto-train':
                self._run_auto(job_id)
            elif self.jobs[job_id]["params"]["operation"] in {"reconstruct", "reconstruct-pair"}:
                self._run_reconstruction(job_id)
            else:
                self._run(job_id)
        finally:
            with self.submission_lock:
                self.project_workers.difference_update(self.job_projects.pop(job_id, {project_id}))

    def _claim_auto_project(self, job_id, project_id):
        with self.submission_lock, projects._lock:
            owned = self.job_projects[job_id]
            if project_id in self.project_workers and project_id not in owned:
                raise ValueError('A linked project is already in use. Resume Auto after that job finishes.')
            project = projects.get_project(project_id)
            if project.get('job_id') != job_id and self.jobs.get(project.get('job_id'), {}).get('status') in {'held', 'queued', 'running'}:
                raise ValueError('A linked project is already queued. Resume Auto after that job finishes.')
            owned.add(project_id)
            self.project_workers.add(project_id)
            projects.update_project(project_id, job_id=job_id,
                auto_training_root=self.jobs[job_id]['params']['music_training_project'])

    @staticmethod
    def _cleanup_music_memory():
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    def _run_auto(self, job_id):
        from .music_auto_training import run_auto, update_auto
        job = self.jobs[job_id]
        project_id = job['params']['music_training_project']
        state = {'abort': False}
        cancelled = lambda: bool(state.get('abort')) or is_cancel_requested(job)

        def finalize(status, message):
            update_auto(project_id, status=status)
            for linked_id in self.job_projects[job_id]:
                projects.update_project(linked_id, status=status, message=message,
                    **({'progress': 100} if status == 'completed' else {}))

        with generation_slot(self.generation_lock, job) as acquired:
            if not acquired:
                finalize('cancelled', 'Auto training cancelled before starting. Resume when ready.')
                return
            try:
                if not try_start(job, message='Starting Auto music training', phase='Auto music training') or not register_abort_state(job, job_id, self.active_states, state):
                    raise InterruptedError('Auto training cancelled before starting')
                self.release_models()
                if cancelled():
                    raise InterruptedError('Auto training stopped; saved work is available to resume')
                projects.update_project(project_id, status='training')
                update_auto(project_id, status='running')

                def report(message, percent=None):
                    changes = {'message': str(message)}
                    if percent is not None:
                        changes['progress'] = percent
                    if update_job(job, **changes):
                        projects.update_project(project_id, **changes)
                        print(f'[Music Auto] {message}', flush=True)

                result = run_auto(project_id, job['params']['options'], report=report, cancelled=cancelled,
                    claim=lambda target: self._claim_auto_project(job_id, target), cleanup=self._cleanup_music_memory)
                message = 'Auto training complete. Your LoRA is saved in My music.'
                if not finish_job(job, 'completed', progress=100, phase='', message=message, music_style_id=result['style_id']):
                    raise InterruptedError('Auto training stopped; saved work is available to resume')
                finalize('completed', message)
            except InterruptedError as error:
                request_cancel(job, job_id=job_id, active_states=self.active_states)
                finalize('cancelled', str(error))
            except Exception as error:
                traceback.print_exc()
                failed = finish_job(job, 'failed', error=str(error), message=str(error))
                finalize('failed' if failed else 'cancelled', str(error))
            finally:
                unregister_abort_state(job_id, self.active_states, state)
                self._cleanup_music_memory()

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
                if job['params']['operation'] == 'reconstruct-pair':
                    from models.TTS.yue2.pair_reconstruction import reconstruct_pair as reconstruct_project
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
        preparation = operation in {'prepare-pair', 'analyze-songs', 'build-dataset', 'prepare', 'prepare-audio', 'align-lyrics', 'review-data'}
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
                if operation == 'analyze-songs':
                    from .music_dataset_analysis import analyze_songs
                    analyze_songs(project, report=report, cancelled=cancelled,
                                  retry_voices=job['params']['options'].get('retry_voices') is True,
                                  track_id=job['params']['options'].get('track_id'),
                                  language=job['params']['options'].get('language'))
                elif operation == 'build-dataset':
                    from .music_dataset import build_dataset
                    build_dataset(project, report=report, cancelled=cancelled)
                elif operation == "prepare":
                    from models.TTS.yue2.music_tokenizer import prepare_project
                    prepare_project(project, report=report, cancelled=cancelled)
                    from .music_data_review import sequence_coverage
                    sequence_coverage(project, report=report, cancelled=cancelled)
                elif operation == 'prepare-audio':
                    from models.TTS.yue2.audio_training_data import prepare_audio
                    prepare_audio(project, report=report, cancelled=cancelled)
                elif operation == 'prepare-pair':
                    from models.TTS.yue2.pair_training_data import prepare_pair
                    prepare_pair(project, report=report, cancelled=cancelled)
                elif operation == 'adapt-pair':
                    from models.TTS.yue2.pair_training import train_pair
                    train_pair(project, job['params']['options'], report=report, cancelled=cancelled)
                elif operation == 'adapt-audio':
                    from models.TTS.yue2.acoustic_training import train_audio
                    from .music_auditions import train_with_auditions
                    audition_failures = train_with_auditions(project_id, job['params']['options'], train_audio, 'audio', report=report, cancelled=cancelled)
                elif operation == 'align-lyrics':
                    from models.TTS.yue2.lyric_alignment import align_project
                    align_project(project, report=report, cancelled=cancelled)
                elif operation == 'train-joint':
                    from models.TTS.yue2.joint_training import train_joint
                    from .music_auditions import train_with_auditions
                    audition_failures = train_with_auditions(project_id, job['params']['options'], train_joint, 'joint', report=report, cancelled=cancelled)
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
                message = {'prepare-pair': 'Sound-adaptation data ready. Start tokenizer + decoder adaptation.',
                           'adapt-pair': 'Sound adaptation complete. Choose a pair to train song style in a new project.',
                           'analyze-songs': 'Voice and clip drafts ready; listen and review before creating the dataset',
                           'build-dataset': 'Reviewed training dataset ready to open',
                           'prepare': 'Audio tokens prepared', 'prepare-audio': 'Source sound prepared',
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
