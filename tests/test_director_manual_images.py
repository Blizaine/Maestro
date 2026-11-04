"""Reviewed scene images must survive None image-model policy and reruns."""
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services import director_pipeline as pipeline, llm_service
import test_director_cancellation as cancellation_fixture
import test_director_model_compat as compat_fixture


class TestDirectorManualImages(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = self.temp.name

    def image(self, name):
        path = Path(self.directory, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"reviewed image")
        return str(path)

    def test_external_and_nested_images_become_valid_owned_saved_starts(self):
        outside = self.image("upload/room.png")
        nested = self.image("project/assets/character.png")
        directory = str(Path(self.directory, "project"))
        recorded = pipeline._prepare_reviewed_shot_images(
            [outside, nested, outside], 3, directory,
        )
        pipeline._require_video_start_images(recorded, 3, directory)
        self.assertEqual(recorded[0], recorded[2])
        for name in recorded:
            self.assertEqual(os.path.basename(name), name)
            self.assertEqual(Path(directory, name).read_bytes(), b"reviewed image")
        self.assertEqual(Path(outside).read_bytes(), b"reviewed image")

    def test_invalid_reviewed_image_is_not_silently_discarded(self):
        for paths in [["missing.png"], [self.image("not-an-image.txt")], []]:
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                pipeline._prepare_reviewed_shot_images(paths, 1, self.directory)

    def test_none_pipeline_skips_generator_but_preserves_reviewed_images(self):
        source = self.image("uploads/scene.png")
        output = str(Path(self.directory, "project"))
        Path(output).mkdir()
        params = {
            "pipeline_type": "short_film_story", "image_model": "__none__",
            "video_model": "test-video", "auto_mode": True, "seamless": False,
            "_director_shot_image_policy": "prompt_only",
            "prepared_clip_plans": [{"video_prompt": "A boat moves across a lake."}],
            "prepared_planned_clips": [{"start": 0, "end": 5, "duration_sec": 5}],
            "prepared_clip_image_paths": [source],
        }
        record = {"id": "manual", "status": "running", "params": params,
                  "out_dir": output, "clip_plans": [], "clip_images": [],
                  "output_files": [], "progress": {}}
        with ExitStack() as stack:
            stack.enter_context(patch.object(pipeline, "_wgp", SimpleNamespace(
                save_path=output, get_model_def=lambda _: {},
                server_config={"services": {"director_prompt_polish": "off"}},
            )))
            stack.enter_context(patch.dict(pipeline._pipelines, {"manual": record}))
            for name in ("_save_pipeline_state", "_validate_director_models", "_preflight_h3_director_prompts"):
                stack.enter_context(patch.object(pipeline, name))
            stack.enter_context(patch.object(pipeline, "_wait_for_gpu", return_value=True))
            stack.enter_context(patch.object(llm_service, "is_loaded", return_value=False))
            generator = stack.enter_context(patch.object(pipeline, "_run_image_generation"))
            render = stack.enter_context(patch.object(pipeline, "_run_video_generation", return_value=[]))
            pipeline._run_pipeline("manual")
        self.assertEqual(record["status"], "completed", record.get("error"))
        generator.assert_not_called()
        recorded = render.call_args.args[4]
        pipeline._require_video_start_images(recorded, 1, output)
        self.assertEqual(params["prepared_clip_image_paths"], [os.path.join(output, recorded[0])])

    def test_initial_frames_and_references_receive_each_reviewed_scene_image(self):
        paths = [self.image("one.png"), self.image("two.png")]
        for strategy in ("bounded_start_end", "omni_reference"):
            with self.subTest(strategy=strategy), ExitStack() as stack:
                model_type = "minimax_h3_ref2va" if strategy == "omni_reference" else "minimax_h3"
                model = {"architecture": model_type, "fps": 24, "frames_minimum": 124,
                         "frames_maximum": 345, "frames_steps": 17, "latent_size": 17,
                         "director_video_strategy": strategy, "returns_audio": True}
                stack.enter_context(patch.object(pipeline, "_wgp", SimpleNamespace(
                    save_path=self.directory, get_model_def=lambda _: model,
                    get_model_min_frames_and_step=lambda _: (124, 17, 17),
                    get_default_settings=lambda _: {},
                )))
                stack.enter_context(patch.object(pipeline, "_preflight_h3_director_prompts"))
                submit = stack.enter_context(patch.object(pipeline, "_submit_and_wait", return_value=["out.mp4"]))
                pipeline._run_video_generation("manual", {
                    "video_model": model_type, "pipeline_type": "short_film_story",
                    "seamless": False, "image_model": "__none__", "video_params": {},
                    "_director_shot_image_policy": "direct_references" if strategy == "omni_reference" else "prompt_only",
                    "prepared_clip_image_paths": paths,
                }, [{"video_prompt": "First scene."}, {"video_prompt": "Second scene."}],
                    [{"start": i * 124 / 24, "end": (i + 1) * 124 / 24,
                      "duration_frames": 124, "duration_sec": 124 / 24} for i in range(2)],
                    [os.path.basename(path) for path in paths], out_dir=self.directory)
                submitted = submit.call_args.args[0]
                if strategy == "omni_reference":
                    manifests = submitted["per_clip_minimax_h3_references"]
                    self.assertEqual([[r["path"] for r in manifest] for manifest in manifests], [[paths[0]], [paths[1]]])
                    self.assertTrue(all(manifest[0]["image_intent"] == "composition" for manifest in manifests))
                else:
                    self.assertEqual(submitted["image_start"], paths)
                    self.assertEqual(submitted["image_prompt_type"], "S")

    def test_none_rerun_keeps_reviewed_start_and_rejects_missing_file(self):
        fixture = cancellation_fixture.TestDirectorCancellation(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        source = fixture._write_media("manual.png", b"reviewed image")
        record = fixture._add_pipeline("manual-rerun", "completed")
        record["params"].update({
            "pipeline_type": "short_film_story", "image_model": "__none__",
            "_director_shot_image_policy": "prompt_only",
            "prepared_clip_image_paths": [source],
        })
        record["clip_plans"] = [{"video_prompt": "A boat crosses the lake."}]
        record["clip_images"] = ["manual.png"]
        record["_planned_clips"] = [{"start": 0, "end": 5, "duration_sec": 5}]
        self.assertTrue(pipeline._save_pipeline_state("manual-rerun"))
        with patch.object(pipeline, "_submit_and_wait", return_value=["new.mp4"]) as submit:
            pipeline.rerun_clip_video(fixture.temp_dir.name, "manual-rerun", 0)
            self.assertEqual(submit.call_args.args[0]["image_start"], source)
            self.assertEqual(submit.call_args.args[0]["image_prompt_type"], "S")
        os.remove(source)
        with patch.object(pipeline, "_submit_and_wait") as submit:
            with self.assertRaisesRegex(ValueError, "valid start image"):
                pipeline.rerun_clip_video(fixture.temp_dir.name, "manual-rerun", 0)
            submit.assert_not_called()

    def test_reference_model_accepts_only_reviewed_scene_images(self):
        source = self.image("manual.png")
        model_type = "minimax_h3_ref2va_pika_flex"
        model = compat_fixture._effective_h3_model(
            compat_fixture._pika_h3_default_model(model_type + ".json"),
        )
        with patch.object(pipeline, "_wgp", SimpleNamespace(
            get_model_def=lambda _: model,
            get_model_family=lambda *_args, **_kwargs: "minimax_h3",
            get_base_model_type=lambda _: model_type,
        )):
            pipeline._validate_director_models({
                "video_model": model_type, "image_model": "__none__",
                "pipeline_type": "short_film_story", "seamless": False,
                "_director_shot_image_policy": "direct_references",
                "prepared_clip_image_paths": [source],
            })

    def test_reference_rerun_keeps_reviewed_composition_with_none_image_model(self):
        fixture = cancellation_fixture.TestDirectorCancellation(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        source = fixture._write_media("manual.png", b"reviewed image")
        model_type = "minimax_h3_ref2va"
        model = {"architecture": model_type, "fps": 24, "frames_minimum": 124,
                 "frames_maximum": 345, "frames_steps": 17,
                 "director_video_strategy": "omni_reference", "returns_audio": True}
        pipeline._wgp.get_model_def = lambda _: model
        pipeline._wgp.get_model_min_frames_and_step = lambda _: (124, 17, 17)
        record = fixture._add_pipeline("manual-ref-rerun", "completed")
        record["params"].update({
            "pipeline_type": "short_film_story", "image_model": "__none__",
            "video_model": model_type,
            "_director_shot_image_policy": "direct_references",
            "prepared_clip_image_paths": [source],
        })
        record["clip_plans"] = [{"video_prompt": "A boat crosses the lake."}]
        record["clip_images"] = ["manual.png"]
        record["_planned_clips"] = [{"start": 0, "end": 124 / 24,
                                   "duration_sec": 124 / 24, "duration_frames": 124}]
        self.assertTrue(pipeline._save_pipeline_state("manual-ref-rerun"))
        with patch.object(pipeline, "_preflight_h3_director_prompts"), \
                patch.object(pipeline, "_validate_saved_profile_for_current_hardware"), \
                patch.object(pipeline, "_submit_and_wait", return_value=["new.mp4"]) as submit:
            pipeline.rerun_clip_video(fixture.temp_dir.name, "manual-ref-rerun", 0)
        submitted = submit.call_args.args[0]
        self.assertNotIn("image_start", submitted)
        self.assertEqual(submitted["minimax_h3_references"][0]["path"], source)
        self.assertEqual(submitted["minimax_h3_references"][0]["image_intent"], "composition")


if __name__ == "__main__":
    unittest.main()
