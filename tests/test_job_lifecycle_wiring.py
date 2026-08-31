"""Static wiring checks that avoid importing Maestro's heavyweight server."""
from __future__ import annotations

import ast
import os
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _parse(relative_path: str) -> ast.Module:
    with open(os.path.join(_ROOT, relative_path), "r", encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=relative_path)


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} not found")


def _called_names(node: ast.AST) -> set[str]:
    names = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.add(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.add(child.func.attr)
    return names


def _load_isolated_function(relative_path: str, name: str, namespace: dict):
    function = _function(_parse(relative_path), name)
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, relative_path, "exec"), namespace)
    return namespace[name]


class TestJobLifecycleWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launch = _parse("app/launch.py")

    def test_each_worker_uses_lifecycle_transitions(self):
        expected = {
            "_run_generation": {
                "try_start", "register_abort_state", "finish_job",
                "record_job_outputs",
            },
            "_run_recast": {"try_start", "register_abort_state", "try_requeue"},
            "_run_tool_upscale": {
                "try_start", "register_abort_state", "finish_job",
                "record_job_outputs",
            },
            "_run_tool_revoice": {"try_start", "register_abort_state", "finish_job"},
            "_run_blend_generation": {
                "register_abort_state", "finish_job", "record_job_outputs",
            },
            "_run_sfx_generation": {
                "finish_job", "record_job_outputs",
            },
        }
        for function_name, required in expected.items():
            with self.subTest(function=function_name):
                calls = _called_names(_function(self.launch, function_name))
                self.assertTrue(required <= calls, required - calls)

    def test_cancel_endpoint_routes_through_shared_helper(self):
        cancel = _function(self.launch, "cancel_job")
        self.assertIn("request_cancel", _called_names(cancel))
        self.assertFalse(any(
            isinstance(node, ast.Attribute) and node.attr == "_interrupt"
            for node in ast.walk(cancel)
        ))

    def test_studio_queue_uses_explicit_held_lifecycle(self):
        generate = _function(self.launch, "generate")
        generate_constants = {
            node.value
            for node in ast.walk(generate)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("_queue_mode", generate_constants)
        self.assertIn("held", generate_constants)
        self.assertTrue(any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isinstance(node.test.operand, ast.Name)
            and node.test.operand.id == "hold_for_queue"
            and any(
                isinstance(child, ast.Name)
                and child.id == "_run_generation"
                for child in ast.walk(node)
            )
            for node in ast.walk(generate)
        ))

        release_queue = _function(self.launch, "_start_held_studio_queue")
        self.assertIn("release_held", _called_names(release_queue))

        dispatcher = _function(self.launch, "_run_held_studio_jobs")
        self.assertIn("_run_generation", _called_names(dispatcher))

        endpoint = _function(self.launch, "start_studio_queue")
        self.assertIn("_start_held_studio_queue", _called_names(endpoint))

        list_jobs = _function(self.launch, "list_jobs")
        list_constants = {
            node.value
            for node in ast.walk(list_jobs)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("held", list_constants)

    def test_studio_queue_release_preserves_submission_order(self):
        started_threads = []

        class FakeThread:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                started_threads.append(self)

            def start(self):
                self.started = True

        jobs = {
            "later": {"status": "held", "created_at": 20},
            "active": {"status": "running", "created_at": 5},
            "earlier": {"status": "held", "created_at": 10},
        }

        def release(job, **updates):
            if job.get("status") != "held":
                return False
            job.update(updates)
            job["status"] = "queued"
            return True

        start_queue = _load_isolated_function(
            "app/launch.py",
            "_start_held_studio_queue",
            {
                "_jobs": jobs,
                "snapshot_job": lambda job: dict(job),
                "release_held": release,
                "threading": SimpleNamespace(Thread=FakeThread),
                "_run_held_studio_jobs": object(),
            },
        )

        self.assertEqual(start_queue(), ["earlier", "later"])
        self.assertEqual(jobs["earlier"]["status"], "queued")
        self.assertEqual(jobs["later"]["status"], "queued")
        self.assertEqual(jobs["active"]["status"], "running")
        self.assertEqual(len(started_threads), 1)
        self.assertEqual(
            started_threads[0].kwargs["args"],
            (["earlier", "later"],),
        )
        self.assertTrue(started_threads[0].started)

    def test_director_music_progress_validation_imports_regex_module(self):
        imported_modules = {
            alias.asname or alias.name.split(".", 1)[0]
            for node in self.launch.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        generate_music = _function(self.launch, "director_generate_music")
        uses_re_fullmatch = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fullmatch"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "re"
            for node in ast.walk(generate_music)
        )

        self.assertTrue(uses_re_fullmatch)
        self.assertIn("re", imported_modules)

    def test_director_dashboard_mutations_run_off_the_event_loop(self):
        expected = {
            "rerun_pipeline_clip_image": "rerun_clip_image",
            "rerun_pipeline_clip_video": "rerun_clip_video",
            "rejoin_pipeline_clips": "rejoin_clips",
        }
        for endpoint_name, worker_name in expected.items():
            with self.subTest(endpoint=endpoint_name):
                endpoint = _function(self.launch, endpoint_name)
                awaited_thread_targets = {
                    call.args[0].id
                    for node in ast.walk(endpoint)
                    if isinstance(node, ast.Await)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "to_thread"
                    for call in [node.value]
                    if call.args and isinstance(call.args[0], ast.Name)
                }
                self.assertIn(worker_name, awaited_thread_targets)

    def test_director_bulk_repair_routes_to_server_owned_worker(self):
        repair = _function(self.launch, "repair_saved_pipeline")
        cancel = _function(self.launch, "cancel_saved_pipeline_repair")
        self.assertIn("start_pipeline_repair", _called_names(repair))
        self.assertIn("cancel_pipeline_repair", _called_names(cancel))

    def test_blend_defers_generation_completion(self):
        blend = _function(self.launch, "_run_blend_generation")
        matching_calls = [
            node for node in ast.walk(blend)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_run_generation"
        ]
        self.assertEqual(len(matching_calls), 1)
        finalize = next(
            (kw.value for kw in matching_calls[0].keywords if kw.arg == "finalize"),
            None,
        )
        self.assertIsInstance(finalize, ast.Constant)
        self.assertIs(finalize.value, False)

    def test_wan_checks_abort_before_resetting_interrupt(self):
        wgp = _parse("app/wgp.py")
        generate = _function(wgp, "generate_video")
        self.assertIn(
            "_cleanup_generation_resources",
            {
                node.name for node in generate.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            },
        )
        with open(
            os.path.join(_ROOT, "app", "wgp.py"), "r", encoding="utf-8",
        ) as handle:
            source_lines = handle.read().splitlines()
        body = "\n".join(source_lines[generate.lineno - 1:generate.end_lineno])
        reset = body.index("wan_model._interrupt = False")
        before = body.rfind('if gen.get("abort", False):', 0, reset)
        after = body.find('if gen.get("abort", False):', reset)
        self.assertGreaterEqual(before, 0)
        self.assertGreater(after, reset)
        for check in (before, after):
            cleanup = body.find("_cleanup_generation_resources()", check)
            abort_return = body.find("return False", check)
            self.assertLess(check, cleanup)
            self.assertLess(cleanup, abort_return)

    def test_flashvsr_checks_cancel_before_replacing_source_video(self):
        upscale = _function(
            self.launch, "_apply_spatial_upsampling_to_file",
        )
        with open(
            os.path.join(_ROOT, "app", "launch.py"),
            "r", encoding="utf-8",
        ) as handle:
            launch_source = handle.read()
        source = ast.get_source_segment(launch_source, upscale)
        self.assertIsNotNone(source)
        for replacement in (
            "os.replace(tmp_muxed, video_path)",
            "os.replace(tmp_video, video_path)",
        ):
            replace_at = source.index(replacement)
            check_at = source.rfind("abort_check()", 0, replace_at)
            self.assertGreaterEqual(check_at, 0)

    def test_generation_stamps_partial_outputs_before_cancel_return(self):
        generation = _function(self.launch, "_run_generation")
        with open(
            os.path.join(_ROOT, "app", "launch.py"), "r", encoding="utf-8",
        ) as handle:
            launch_source = handle.read()
        source = ast.get_source_segment(launch_source, generation)
        self.assertIsNotNone(source)
        publish_at = source.index("record_job_outputs(\n                    job,")
        stamp_at = source.index("_write_output_sidecars(new_files)", publish_at)
        cancel_at = source.index("if cancelled or is_cancel_requested(job):", stamp_at)
        cancel_return = source.index("return False", cancel_at)
        self.assertLess(publish_at, stamp_at)
        self.assertLess(stamp_at, cancel_at)
        self.assertLess(cancel_at, cancel_return)

    def test_director_sidecars_cover_every_supported_media_extension(self):
        generation = _function(self.launch, "_run_generation")
        sidecar_writer = next(
            node for node in ast.walk(generation)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_write_output_sidecars"
        )
        referenced_names = {
            node.id for node in ast.walk(sidecar_writer)
            if isinstance(node, ast.Name)
        }
        self.assertIn("GENERATED_MEDIA_EXTENSIONS", referenced_names)
        self.assertIn("output_filename", {
            node.value for node in ast.walk(sidecar_writer)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        })

    def test_gallery_sidecars_use_active_generation_time(self):
        generation = _function(self.launch, "_run_generation")
        with open(
            os.path.join(_ROOT, "app", "launch.py"), "r", encoding="utf-8",
        ) as handle:
            launch_source = handle.read()
        source = ast.get_source_segment(launch_source, generation)
        self.assertIsNotNone(source)
        self.assertIn('cmd == "generation_time"', source)
        self.assertIn("active_generation_seconds_by_output", source)
        self.assertIn('"job_elapsed_time":', source)
        self.assertNotIn(
            '"generation_time": round(time.time() - start_time)',
            source,
        )

        generate_video = _function(_parse("app/wgp.py"), "generate_video")
        with open(
            os.path.join(_ROOT, "app", "wgp.py"), "r", encoding="utf-8",
        ) as handle:
            wgp_source = handle.read()
        wgp_body = ast.get_source_segment(wgp_source, generate_video)
        self.assertIsNotNone(wgp_body)
        self.assertIn('"generation_time",', wgp_body)
        self.assertIn('configs["generation_time_basis"] = "active"', wgp_body)

    def test_generation_duration_is_minutes_and_seconds(self):
        formatter = _load_isolated_function(
            "app/wgp.py",
            "format_generation_time",
            {},
        )
        self.assertEqual(formatter(128), "2m 8s")
        self.assertEqual(formatter(8), "0m 8s")
        self.assertEqual(formatter(3601), "60m 1s")

    def test_continuation_accepts_all_generated_video_containers(self):
        generation = _function(self.launch, "_run_generation")
        with open(
            os.path.join(_ROOT, "app", "launch.py"), "r", encoding="utf-8",
        ) as handle:
            launch_source = handle.read()
        source = ast.get_source_segment(launch_source, generation)
        continuation = source.split(
            "# Find the latest video explicitly registered by", 1,
        )[1].split("if latest_video:", 1)[0]
        for extension in (".mp4", ".webm", ".mkv", ".mov"):
            self.assertIn(extension, continuation)

    def test_failed_multiclip_concat_removes_partial_output(self):
        concatenate = _load_isolated_function(
            "app/wgp.py",
            "concatenate_multi_clip_videos",
            {"os": os},
        )
        with tempfile.TemporaryDirectory() as directory:
            clip = os.path.join(directory, "clip.mp4")
            output = os.path.join(directory, "joined.mp4")
            with open(clip, "wb") as handle:
                handle.write(b"clip")

            def fake_run(command, **kwargs):
                if "-filter_complex" in command:
                    with open(output, "wb") as handle:
                        handle.write(b"partial")
                    return SimpleNamespace(
                        returncode=1, stdout="", stderr="ffmpeg error",
                    )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch("subprocess.run", side_effect=fake_run):
                self.assertFalse(concatenate([clip], output))
            self.assertFalse(os.path.exists(output))

    def test_multiclip_external_audio_can_start_after_source_time_zero(self):
        concatenate = _load_isolated_function(
            "app/wgp.py",
            "concatenate_multi_clip_videos",
            {"os": os},
        )
        with tempfile.TemporaryDirectory() as directory:
            clip = os.path.join(directory, "clip.mp4")
            audio = os.path.join(directory, "song.wav")
            output = os.path.join(directory, "joined.mp4")
            for path in (clip, audio):
                with open(path, "wb") as handle:
                    handle.write(b"media")
            commands = []

            def fake_run(command, **kwargs):
                commands.append(command)
                if "-filter_complex" in command:
                    with open(output, "wb") as handle:
                        handle.write(b"joined")
                stdout = "25/1\n" if "stream=r_frame_rate" in command else ""
                return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

            with patch("subprocess.run", side_effect=fake_run):
                self.assertTrue(concatenate(
                    [clip], output, audio, audio_start_sec=2.0,
                ))

            command = next(c for c in commands if "-filter_complex" in c)
            filter_value = command[command.index("-filter_complex") + 1]
            self.assertIn(
                "[1:a]atrim=start=2.000000,asetpts=PTS-STARTPTS[outa]",
                filter_value,
            )
            self.assertIn("[outa]", command)

    def test_multiclip_concat_can_be_cancelled_during_ffmpeg(self):
        concatenate = _load_isolated_function(
            "app/wgp.py",
            "concatenate_multi_clip_videos",
            {"os": os},
        )
        with tempfile.TemporaryDirectory() as directory:
            clip = os.path.join(directory, "clip.mp4")
            output = os.path.join(directory, "joined.mp4")
            with open(clip, "wb") as handle:
                handle.write(b"clip")

            class FakeProcess:
                def __init__(self, *_args, **_kwargs):
                    self.returncode = None
                    self.finished = False
                    with open(output, "wb") as handle:
                        handle.write(b"partial")

                def communicate(self, timeout=None):
                    if not self.finished:
                        raise subprocess.TimeoutExpired("ffmpeg", timeout)
                    return "", ""

                def terminate(self):
                    self.finished = True
                    self.returncode = -15

                def kill(self):
                    self.finished = True
                    self.returncode = -9

                def poll(self):
                    return self.returncode

            probe = SimpleNamespace(returncode=0, stdout="", stderr="")
            with patch("subprocess.run", return_value=probe):
                with patch("subprocess.Popen", FakeProcess):
                    self.assertFalse(concatenate(
                        [clip],
                        output,
                        abort_callback=lambda: True,
                    ))
            self.assertFalse(os.path.exists(output))

    def test_multiclip_can_pad_short_audio_to_exact_video_timeline(self):
        concatenate = _load_isolated_function(
            "app/wgp.py",
            "concatenate_multi_clip_videos",
            {"os": os},
        )
        with tempfile.TemporaryDirectory() as directory:
            clip = os.path.join(directory, "clip.mp4")
            audio = os.path.join(directory, "source.mp4")
            output = os.path.join(directory, "joined.mp4")
            for path in (clip, audio):
                with open(path, "wb") as handle:
                    handle.write(b"media")
            commands = []

            def fake_run(command, **kwargs):
                commands.append(command)
                if "-filter_complex" in command:
                    with open(output, "wb") as handle:
                        handle.write(b"joined")
                stdout = "30/1\n" if "stream=r_frame_rate" in command else ""
                return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

            with patch("subprocess.run", side_effect=fake_run):
                self.assertTrue(concatenate(
                    [clip],
                    output,
                    audio,
                    pad_audio=True,
                    audio_duration_sec=1.25,
                ))

            command = next(c for c in commands if "-filter_complex" in c)
            filter_value = command[command.index("-filter_complex") + 1]
            self.assertIn(
                "[1:a]asetpts=PTS-STARTPTS,apad,"
                "atrim=duration=1.250000[outa]",
                filter_value,
            )
            self.assertIn("[outa]", command)
            self.assertIn("-shortest", command)

    def test_multiclip_dispatch_preserves_audio_origin(self):
        generation = _function(self.launch, "_run_generation")
        with open(
            os.path.join(_ROOT, "app", "launch.py"), "r", encoding="utf-8",
        ) as handle:
            source = ast.get_source_segment(handle.read(), generation)
        self.assertIn('raw_params.get("audio_frame_offset", 0)', source)
        self.assertGreaterEqual(source.count(
            '"audio_start_sec": multi_clip_audio_start_sec'
        ), 2)

    def test_director_multiclip_dispatch_uses_explicit_prompt_modes(self):
        generation = _function(self.launch, "_run_generation")
        with open(
            os.path.join(_ROOT, "app", "launch.py"), "r", encoding="utf-8",
        ) as handle:
            source = ast.get_source_segment(handle.read(), generation)
        self.assertIn('raw_params.pop(\n                    "per_clip_prompt_modes"', source)
        self.assertIn("explicit_prompt_mode", source)
        self.assertIn(
            'else (1 if "\\n" in clip_prompt else 0)',
            source,
        )

    def test_failed_audio_mux_removes_partial_output(self):
        combine = _load_isolated_function(
            "app/shared/utils/audio_video.py",
            "combine_and_concatenate_video_with_audio_tracks",
            {
                "os": os,
                "subprocess": subprocess,
                "get_mp4_audio_codec_settings": lambda _key: {
                    "codec": "aac", "bitrate": None,
                },
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "muxed.mp4")

            def fail_after_partial(command, **kwargs):
                with open(output, "wb") as handle:
                    handle.write(b"partial")
                raise subprocess.CalledProcessError(
                    1, command, stderr="mux failed",
                )

            with patch("subprocess.run", side_effect=fail_after_partial):
                with self.assertRaisesRegex(Exception, "FFmpeg error"):
                    combine(
                        output, "input.mp4", [], [], 0, 44100,
                    )
            self.assertFalse(os.path.exists(output))

    def test_wgp_audio_mux_always_cleans_raw_render_temp(self):
        generate = _function(_parse("app/wgp.py"), "generate_video")

        def calls_named(node, name):
            return any(
                isinstance(child, ast.Call)
                and (
                    isinstance(child.func, ast.Name)
                    and child.func.id == name
                    or isinstance(child.func, ast.Attribute)
                    and child.func.attr == name
                )
                for child in ast.walk(node)
            )

        cleanup_try = next(
            (
                node for node in ast.walk(generate)
                if isinstance(node, ast.Try)
                and calls_named(ast.Module(body=node.body, type_ignores=[]),
                                "combine_and_concatenate_video_with_audio_tracks")
                and calls_named(ast.Module(body=node.finalbody, type_ignores=[]),
                                "remove")
            ),
            None,
        )
        self.assertIsNotNone(cleanup_try)
        self.assertTrue(any(
            isinstance(child, ast.Name) and child.id == "save_path_tmp"
            for statement in cleanup_try.finalbody
            for child in ast.walk(statement)
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
