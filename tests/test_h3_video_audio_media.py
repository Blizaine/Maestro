"""Exercise real CPU soundtrack extraction/cache without model inference."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.h3_reference_audio import canonicalize_video_audio_roles, measure_video_audio_metadata
from models.minimax_h3.reference_manifest import split_exact_drive_audio_reference


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe required")
class VideoAudioMediaTests(unittest.TestCase):
    def test_embedded_mkv_soundtrack_cache_concurrency_and_exact_drive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mkv"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=24:d=2",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000:duration=2",
                "-c:v", "mpeg4", "-c:a", "pcm_s16le", "-shortest", str(source),
            ], check=True, capture_output=True, timeout=30)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            original = [{"type": "video", "id": "clip", "path": str(source),
                         "has_audio": True, "video_intent": "motion", "video_audio_role": "drive"}]
            cache = root / "cache"
            def prepare():
                return canonicalize_video_audio_roles(original, cache_dir=cache)
            with ThreadPoolExecutor(max_workers=2) as pool:
                first, second = list(pool.map(lambda _: prepare(), range(2)))
            self.assertEqual(first, second)
            self.assertNotIn("audio_duration_seconds", original[0])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), source_hash)
            self.assertEqual(len(list(cache.glob("*.wav"))), 1)
            self.assertFalse(list(cache.glob("*.tmp")))
            measured = measure_video_audio_metadata(original, cache_dir=cache)
            self.assertEqual(len(measured), 1)
            self.assertAlmostEqual(measured[0]["audio_duration_seconds"], 2.0, delta=0.05)
            runtime, drive_path, ordinal = split_exact_drive_audio_reference(first)
            self.assertIsNone(ordinal)  # Exact video soundtrack has no Audio label to remove.
            self.assertEqual(len(runtime), 1)
            self.assertEqual(runtime[0]["type"], "video")
            self.assertEqual(Path(drive_path).read_bytes(), Path(first[1]["path"]).read_bytes())
            self.assertEqual(len(canonicalize_video_audio_roles(runtime, cache_dir=cache)), 1)


if __name__ == "__main__":
    unittest.main()
