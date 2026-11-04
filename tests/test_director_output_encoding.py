"""Director rejoin forwards output settings without altering the saved clips."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services import director_pipeline as pipeline
import test_director_cancellation as fixture_module


class TestDirectorOutputEncoding(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.TestDirectorCancellation(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.directory = self.fixture.temp_dir.name
        record = self.fixture._add_pipeline("encoding", "completed")
        record["params"].update({"pipeline_type": "short_film_story",
                                  "_director_shot_image_policy": "prompt_only"})
        record["clip_plans"] = [{"video_prompt": "One."}, {"video_prompt": "Two."}]
        record["_clip_video_files"] = ["one.mp4", "two.mp4"]
        for name in record["_clip_video_files"]:
            self.fixture._write_media(name, b"original clip")
        self.assertTrue(pipeline._save_pipeline_state("encoding"))

    def test_rejoin_preserves_selected_codec_quality_container_and_metadata(self):
        for codec, container in [("libx265_8", "mp4"), ("libx264_lossless", "mkv")]:
            with self.subTest(codec=codec, container=container):
                pipeline._wgp.server_config = {"video_output_codec": codec, "video_container": container}
                def concatenate(_clips, output, *_args, **_kwargs):
                    Path(output).write_bytes(b"joined output")
                    return True
                pipeline._wgp.concatenate_multi_clip_videos = Mock(side_effect=concatenate)
                result = pipeline.rejoin_clips(self.directory, "encoding")
                self.assertTrue(result["filename"].endswith("." + container))
                args, kwargs = pipeline._wgp.concatenate_multi_clip_videos.call_args
                self.assertEqual(kwargs["codec_type"], codec)
                self.assertEqual(kwargs["container"], container)
                self.assertEqual([os.path.basename(p) for p in args[0]], ["one.mp4", "two.mp4"])
                metadata = Path(self.directory, os.path.splitext(result["filename"])[0] + ".meta.json")
                saved = json.loads(metadata.read_text(encoding="utf-8"))
                self.assertEqual(saved["params"]["video_quality"], codec)
                self.assertEqual(saved["params"]["video_container"], container)
                self.assertEqual(saved["director_pipeline_id"], "encoding")
                for source in args[0]:
                    self.assertEqual(Path(source).read_bytes(), b"original clip")

    def test_invalid_container_never_dispatches_an_encoder(self):
        pipeline._wgp.server_config = {"video_container": "not-a-container"}
        pipeline._wgp.concatenate_multi_clip_videos = Mock()
        with self.assertRaisesRegex(ValueError, "container"):
            pipeline.rejoin_clips(self.directory, "encoding")
        pipeline._wgp.concatenate_multi_clip_videos.assert_not_called()


if __name__ == "__main__":
    unittest.main()
