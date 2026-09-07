"""Face identity, timeline, source-protection and recipe regressions."""
import contextlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import numpy as np
import torch
from services import face_refiner as service
from services.media_flow import ProcessingCancelled
from postprocessing.h3_face_refiner import runtime, face


class FaceRefinerContracts(unittest.TestCase):
    def test_refinement_adapter_retains_huggingface_source_subfolder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def download(**kwargs):
                path = root / kwargs['targetFolderList'][0] / kwargs['sourceFolderList'][0] / kwargs['fileList'][0][0]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'adapter')
            def locate(relative, error_if_none=True):
                path = root / relative
                if path.is_file():
                    return str(path)
                if error_if_none:
                    raise FileNotFoundError(relative)
            with patch.dict(sys.modules, {'wgp': SimpleNamespace(process_files_def=download)}), \
                 patch('shared.utils.files_locator.locate_file', side_effect=locate):
                first = runtime.ensure_refinement_adapter()
                self.assertEqual(Path(first).read_bytes(), b'adapter')
                self.assertEqual(runtime.ensure_refinement_adapter(), first)

    def test_auto_and_limits(self):
        self.assertEqual(service.normalize_options()["face_count"], 0)
        for key, values in {"face_count": [-1, 6, True, 1.5], "steps": [3, 9],
                            "strength": [0, -1, 1.1, float("nan"), float("inf")],
                            "window_frames": [23, 362], "character_ids": [["../escape"]]}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    service.normalize_options({key: value})

    def test_schedule_starts_at_source_noise_and_keeps_all_steps(self):
        video, audio = runtime.refinement_sigmas(4, 0.45, 0.75)
        self.assertEqual(len(video), 5)
        self.assertAlmostEqual(video[0].item(), (12 * 0.5 / (1 + 11 * 0.5)) * 0.75)
        self.assertTrue(bool((video[:-1] > video[1:]).all()))
        self.assertEqual(video[-1], 0)
        self.assertEqual(audio[-1], 0)
        # Real scheduler must see exactly these effective (scaled) sigmas.
        from models.minimax_h3.scheduler import MiniMaxH3Scheduler
        scheduler = MiniMaxH3Scheduler()
        scheduler.set_timesteps(sigmas=video)
        self.assertEqual(len(scheduler.timesteps), 4)
        torch.testing.assert_close(scheduler.timesteps, 1 - video[:-1])

    def test_windows_cover_every_frame_without_gaps(self):
        for count in (1, 22, 124, 244, 500, 7200):
            starts = runtime.window_starts(count)
            self.assertEqual(starts[0], 0)
            covered = set()
            for start in starts:
                covered.update(range(start, min(start + 243, count)))
            self.assertEqual(len(covered), count)
        for window, overlap in ((362, 18), (23, 18), (124, 124), (22, -1)):
            with self.assertRaises(ValueError):
                runtime.window_starts(1000, window, overlap)

    def test_overlap_blending_ignores_model_default_device(self):
        previous = torch.full((3, 18, 8, 8), 20, dtype=torch.uint8, device='cpu')
        current = torch.full((3, 39, 8, 8), 220, dtype=torch.uint8, device='cpu')
        # MMGP changes the default device when loading H3. A meta default
        # reproduces accidental device inheritance without needing CUDA.
        with torch.device('meta'):
            result = service.blend_overlap(previous, current, 18)
        self.assertEqual(result.device.type, 'cpu')
        self.assertTrue(bool((result[:, 0] == 20).all()))
        self.assertTrue(bool((result[:, 17:] == 220).all()))
        self.assertGreater(int(result[0, 9, 0, 0]), 20)
        self.assertLess(int(result[0, 9, 0, 0]), 220)

    def test_mapping_identity_is_explicit_not_character_order(self):
        data = {"faces": [{"track_id": 1, "character_id": "a" * 16}, {"track_id": 2, "character_id": "b" * 16}]}
        mapped = service.resolve_mappings(data, [
            {"track_id": 2, "character_id": "a" * 16}, {"track_id": 1, "skip": True}])
        self.assertTrue(mapped[0]["skip"])
        self.assertEqual(mapped[1]["character_id"], "a" * 16)
        for assignments in ([{"track_id": 1}, {"track_id": 1}], [{"track_id": 3}], [{"track_id": True}]):
            with self.assertRaises(ValueError):
                service.resolve_mappings(data, assignments)

    def test_keep_original_overrides_auto_match(self):
        data = {"faces": [{"track_id": 1, "character_id": "a" * 16}]}
        self.assertIsNone(service.resolve_mappings(data, [{"track_id": 1, "character_id": None}])[0]["character_id"])

    def test_changed_source_or_count_rejects_old_mappings(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(service, "_root", return_value=Path(temp)):
            source = Path(temp) / "video.mp4"
            source.write_bytes(b"original")
            directory = service.analysis_directory("a" * 32)
            directory.mkdir()
            (directory / "analysis.json").write_text(json.dumps({"source": service.fingerprint(source), "face_count": 0}))
            service.load_analysis("a" * 32, source, 0)
            with self.assertRaisesRegex(ValueError, "count changed"):
                service.load_analysis("a" * 32, source, 2)
            source.write_bytes(b"different video")
            with self.assertRaisesRegex(ValueError, "source video changed"):
                service.load_analysis("a" * 32, source)
            with self.assertRaises(ValueError):
                service.analysis_directory("../escape")

    def test_character_ids_use_real_generation_manifest_key(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "video.mp4"
            source.with_suffix(".meta.json").write_text(json.dumps({"params": {"minimax_h3_references": [
                {"type": "video", "library_character_id": "a" * 16},
                {"type": "audio", "library_character_id": "a" * 16},
                {"type": "video", "library_character_id": "b" * 16}]}}))
            self.assertEqual(service.source_characters(source), ["a" * 16, "b" * 16])

    def test_track_survives_absence_and_cut_without_merging_people(self):
        left, right = [0, 0, 40, 40], [80, 0, 120, 40]
        a, b = np.array([1., 0.]), np.array([0., 1.])
        detections = [[left, right], [], [right, left]]
        tracks = face._associate_tracks(detections, [[a, b], [], [a, b]], 0.28, shot_boundaries=[2])
        self.assertEqual(len(tracks), 2)
        for track in tracks:
            if np.dot(track["anchor"], a) > 0.9:
                self.assertEqual(track["boxes"], [left, None, right])

    def test_composite_preserves_background_and_absent_frames(self):
        frames = torch.full((2, 128, 128, 3), 70, dtype=torch.uint8)
        refined = torch.full((3, 2, 64, 64), 200, dtype=torch.uint8)
        transform = {"boxes": [(40, 40, 48, 48)] * 2, "face_rect": [(16, 16, 32, 32)] * 2,
                     "canvas": (64, 64), "src_size": (128, 128), "weights": [1., 0.], "detected": [True, False]}
        with patch("torch.cuda.is_available", return_value=False):
            result = face.stitch(frames, refined, transform, feather=2, mask_dilation=0, colour_match=0)
        self.assertEqual(tuple(result.shape), (2, 128, 128, 3))
        self.assertTrue(bool((result[:, :20] == 70).all()))
        self.assertTrue(bool((result[1] == 70).all()))
        self.assertGreater(int(result[0, 64, 64, 0]), 70)

    def test_automatic_character_matching_rejects_weak_and_ambiguous_matches(self):
        frames = torch.zeros(1, 64, 64, 3, dtype=torch.uint8)
        box = [0, 0, 40, 40]
        references = [object(), object()]
        for similarities, expected in (([0.38, 0.2], False), ([0.52, 0.49], False), ([0.8, 0.3], True)):
            with self.subTest(similarities=similarities), \
                 patch.object(face, "_insightface_app"), \
                 patch.object(face, "_release_insightface", return_value=False), \
                 patch.object(face, "_aligned_embeddings", return_value=[np.array([1., 0.])]), \
                 patch.object(face, "_embeddings", side_effect=[[(box, np.array([s, (1 - s * s)**0.5]))] for s in similarities]):
                tracks = face._select_tracks(frames, [[box]], reference_images=references,
                    reference_threshold=0.5, reference_margin=0.08)
            self.assertEqual("reference_image" in tracks[0], expected)
            if expected:
                self.assertIs(tracks[0]["reference_image"], references[0])

    def test_no_faces_is_exact_copy_and_never_loads_h3(self):
        with tempfile.TemporaryDirectory() as temp:
            source, destination = Path(temp) / "source.mp4", Path(temp) / "copy.mp4"
            source.write_bytes(b"original video bytes")
            data = {"id": "a" * 32, "faces": [], "warnings": []}
            with patch.object(service, "analyze", return_value=data), patch.object(runtime, "refinement_model") as model:
                result = service.process_video(source, destination)
                self.assertTrue(result["unchanged"])
                self.assertEqual(source.read_bytes(), destination.read_bytes())
                model.assert_not_called()

    def test_cancel_does_not_publish_a_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            source, destination = Path(temp) / "source.mp4", Path(temp) / "copy.mp4"
            source.write_bytes(b"original")
            with patch.object(service, "analyze", return_value={"id": "a" * 32, "faces": [], "warnings": []}):
                with self.assertRaises(ProcessingCancelled):
                    service.process_video(source, destination, abort=lambda: True)
            self.assertFalse(destination.exists())
            self.assertEqual(source.read_bytes(), b"original")

    def test_cancellation_during_copy_removes_partial_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            source, destination = Path(temp) / "source.mp4", Path(temp) / "copy.mp4"
            source.write_bytes(b"a" * (3 * 1024 * 1024))
            with patch.object(service, "analyze", return_value={"id": "a" * 32, "faces": [], "warnings": []}):
                cancelled = iter([False, False, True])
                with self.assertRaises(ProcessingCancelled):
                    service.process_video(source, destination, abort=lambda: next(cancelled))
            self.assertFalse(destination.exists())

    def test_no_faces_transcodes_when_destination_container_differs(self):
        @contextlib.contextmanager
        def decoded(*args):
            yield np.zeros((2, 16, 16, 3), dtype=np.uint8), {"fps": 30}, None
        with tempfile.TemporaryDirectory() as temp:
            source, destination = Path(temp) / "source.webm", Path(temp) / "copy.mp4"
            source.write_bytes(b"webm bytes")
            with patch.object(service, "analyze", return_value={"id": "a" * 32, "faces": [], "warnings": []}), \
                 patch.object(service, "decoded_video", decoded), patch.object(service, "_encode") as encode:
                service.process_video(source, destination)
                encode.assert_called_once()

    def test_refinement_wrapper_pads_short_segment_and_discards_model_audio(self):
        class Model:
            _interrupt = False
            def generate(self, _prompt, **kwargs):
                self.kwargs = kwargs
                return {"x": kwargs["input_frames"], "audio": "must not escape"}
        model = Model()
        crop = torch.zeros(3, 9, 32, 32)
        output = runtime.refine_window(model, crop, "reference.png", fps=30,
            audio=np.zeros((9600, 2)), audio_rate=32000, options=service.normalize_options(), seed=17)
        self.assertEqual(model.kwargs["frame_num"], 22)
        self.assertEqual(model.kwargs["fps"], 30)
        self.assertEqual(tuple(output.shape), tuple(crop.shape))
        self.assertEqual(output.dtype, torch.uint8)


if __name__ == "__main__":
    unittest.main()
