import ast
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

from app.services.gallery_thumbnails import get_video_poster


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SIZES = (480, 960, 1920)


class _RouteError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _load_gallery_route_functions(root: Path):
    source = (ROOT / "app" / "launch.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        "_safe_join", "_get_active_workspace", "_workspace_dir",
        "_workspace_browse_dir", "_resolve_gallery_media_file",
        "serve_gallery_thumbnail",
    }
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    found = {node.name for node in functions}
    if found != names:
        raise AssertionError(f"Missing gallery route functions: {sorted(names - found)}")
    for function in functions:
        function.decorator_list = []
    namespace = {
        "os": os,
        "wgp": types.SimpleNamespace(server_config={
            "save_path": str(root / "outputs"),
            "services": {"active_workspace": "default"},
        }),
        "HTTPException": _RouteError,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "launch.py", "exec"), namespace)
    return namespace


class GalleryThumbnailTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"

    def _mock_ffmpeg(self, command, **_kwargs):
        Path(command[-1]).write_bytes(b"\xff\xd8\xffmock-jpeg")
        return types.SimpleNamespace(returncode=0)

    def test_real_ffmpeg_respects_requested_size_and_never_enlarges(self):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("ffmpeg is not installed")
        source = self.root / "wide.mp4"
        created = self._create_test_video(ffmpeg, source, "1280x720")
        if created.returncode != 0 or not source.is_file():
            self.skipTest("installed ffmpeg lacks the tiny-video test encoder")

        posters = {
            size: get_video_poster(
                str(source), size=size, cache_dir=str(self.cache), ffmpeg=ffmpeg
            )
            for size in SUPPORTED_SIZES
        }
        self.assertTrue(all(posters.values()))
        self.assertEqual(self._ffmpeg_image_dimensions(ffmpeg, posters[480]), (480, 270))
        self.assertEqual(self._ffmpeg_image_dimensions(ffmpeg, posters[960]), (960, 540))
        # A 1920 request must preserve a smaller 1280x720 source without enlarging it.
        self.assertEqual(self._ffmpeg_image_dimensions(ffmpeg, posters[1920]), (1280, 720))
        for poster in posters.values():
            self.assertTrue(Path(poster).read_bytes().startswith(b"\xff\xd8\xff"))

        small_source = self.root / "small.mp4"
        created = self._create_test_video(ffmpeg, small_source, "80x64")
        if created.returncode != 0 or not small_source.is_file():
            self.skipTest("installed ffmpeg could not create the small-video fixture")
        small_poster = get_video_poster(
            str(small_source), size=1920, cache_dir=str(self.cache), ffmpeg=ffmpeg
        )
        self.assertIsNotNone(small_poster)
        self.assertEqual(self._ffmpeg_image_dimensions(ffmpeg, small_poster), (80, 64))

    def _create_test_video(self, ffmpeg, source, dimensions):
        return subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-f", "lavfi", "-i", f"color=c=red:s={dimensions}:r=5:d=0.4",
             "-frames:v", "2", "-threads", "1", "-pix_fmt", "yuv420p",
             "-c:v", "mpeg4", str(source)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )

    def _ffmpeg_image_dimensions(self, ffmpeg, image_path):
        decoded = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", image_path,
             "-frames:v", "1", "-c:v", "ppm", "-f", "image2pipe", "-"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        self.assertEqual(decoded.returncode, 0)
        header = re.match(rb"P6\s+(\d+)\s+(\d+)\s+255\s", decoded.stdout)
        self.assertIsNotNone(header, "ffmpeg did not return a PPM frame")
        return int(header.group(1)), int(header.group(2))

    def test_cache_tracks_source_path_mtime_and_size(self):
        source = self.root / "clip.mp4"
        source.write_bytes(b"video-a")
        stamp = 1_700_000_000_000_000_000
        os.utime(source, ns=(stamp, stamp))
        with patch("app.services.gallery_thumbnails.subprocess.run", side_effect=self._mock_ffmpeg) as run:
            first = get_video_poster(str(source), cache_dir=str(self.cache), ffmpeg="ffmpeg")
            self.assertEqual(get_video_poster(str(source), cache_dir=str(self.cache), ffmpeg="ffmpeg"), first)
            self.assertEqual(run.call_count, 1)

            same_bytes_other_path = self.root / "same-content.mp4"
            same_bytes_other_path.write_bytes(source.read_bytes())
            os.utime(same_bytes_other_path, ns=(stamp, stamp))
            second = get_video_poster(str(same_bytes_other_path), cache_dir=str(self.cache), ffmpeg="ffmpeg")
            self.assertNotEqual(second, first)
            self.assertEqual(run.call_count, 2)

            source.write_bytes(b"video-a-now-larger")
            newer_stamp = stamp + 1_000_000
            os.utime(source, ns=(newer_stamp, newer_stamp))
            third = get_video_poster(str(source), cache_dir=str(self.cache), ffmpeg="ffmpeg")
            self.assertNotEqual(third, first)
            self.assertEqual(run.call_count, 3)

    def test_cache_includes_poster_format_version_and_requested_size(self):
        source = self.root / "clip.mp4"
        source.write_bytes(b"video")
        with patch("app.services.gallery_thumbnails.subprocess.run", side_effect=self._mock_ffmpeg) as run:
            small = get_video_poster(str(source), cache_dir=str(self.cache), ffmpeg="ffmpeg")
            large = get_video_poster(
                str(source), size=960, cache_dir=str(self.cache), ffmpeg="ffmpeg"
            )
            self.assertEqual(
                get_video_poster(str(source), size=960, cache_dir=str(self.cache), ffmpeg="ffmpeg"),
                large,
            )

        self.assertNotEqual(small, large)
        self.assertIn("jpg-v2-480-", Path(small).name)
        self.assertIn("jpg-v2-960-", Path(large).name)
        self.assertEqual(run.call_count, 2)

    def test_concurrent_sizes_do_not_share_inflight_render(self):
        source = self.root / "parallel.mp4"
        source.write_bytes(b"video")
        calls_lock = threading.Lock()
        calls = 0

        def slow_ffmpeg(command, **_kwargs):
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.15)
            Path(command[-1]).write_bytes(b"\xff\xd8\xffmock-jpeg")
            return types.SimpleNamespace(returncode=0)

        with patch("app.services.gallery_thumbnails.subprocess.run", side_effect=slow_ffmpeg):
            with ThreadPoolExecutor(max_workers=2) as pool:
                small = pool.submit(
                    get_video_poster, str(source), size=480,
                    cache_dir=str(self.cache), ffmpeg="ffmpeg",
                )
                large = pool.submit(
                    get_video_poster, str(source), size=960,
                    cache_dir=str(self.cache), ffmpeg="ffmpeg",
                )
                posters = (small.result(), large.result())

        self.assertEqual(calls, 2)
        self.assertNotEqual(posters[0], posters[1])

    def test_concurrent_cache_destinations_do_not_share_inflight_render(self):
        source = self.root / "parallel-destinations.mp4"
        source.write_bytes(b"video")
        calls_lock = threading.Lock()
        calls = 0

        def slow_ffmpeg(command, **_kwargs):
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.15)
            Path(command[-1]).write_bytes(b"\xff\xd8\xffmock-jpeg")
            return types.SimpleNamespace(returncode=0)

        cache_a = self.root / "cache-a"
        cache_b = self.root / "cache-b"
        with patch("app.services.gallery_thumbnails.subprocess.run", side_effect=slow_ffmpeg):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(
                    get_video_poster, str(source), size=480,
                    cache_dir=str(cache_a), ffmpeg="ffmpeg",
                )
                second = pool.submit(
                    get_video_poster, str(source), size=480,
                    cache_dir=str(cache_b), ffmpeg="ffmpeg",
                )
                posters = (first.result(), second.result())

        self.assertEqual(calls, 2)
        self.assertEqual(Path(posters[0]).parents[1], cache_a)
        self.assertEqual(Path(posters[1]).parents[1], cache_b)

    def test_unsupported_sizes_are_rejected_before_rendering(self):
        source = self.root / "unsupported.mp4"
        source.write_bytes(b"video")
        with patch("app.services.gallery_thumbnails.subprocess.run") as run:
            for size in (0, 720, 3840, True, "960"):
                self.assertIsNone(
                    get_video_poster(
                        str(source), size=size, cache_dir=str(self.cache), ffmpeg="ffmpeg"
                    )
                )
        run.assert_not_called()

    def test_identical_concurrent_requests_share_one_render(self):
        source = self.root / "parallel.webm"
        source.write_bytes(b"video")
        calls_lock = threading.Lock()
        calls = 0

        def slow_ffmpeg(command, **_kwargs):
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.15)
            Path(command[-1]).write_bytes(b"\xff\xd8\xffmock-jpeg")
            return types.SimpleNamespace(returncode=0)

        with patch("app.services.gallery_thumbnails.subprocess.run", side_effect=slow_ffmpeg):
            with ThreadPoolExecutor(max_workers=8) as pool:
                posters = list(pool.map(
                    lambda _: get_video_poster(str(source), cache_dir=str(self.cache), ffmpeg="ffmpeg"),
                    range(8),
                ))
        self.assertEqual(calls, 1)
        self.assertEqual(len(set(posters)), 1)
        self.assertTrue(Path(posters[0]).is_file())

    def test_distinct_concurrent_sources_are_limited_to_two_ffmpeg_jobs(self):
        sources = []
        for index in range(5):
            source = self.root / f"clip-{index}.mkv"
            source.write_bytes(b"video")
            sources.append(source)
        state_lock = threading.Lock()
        active = 0
        maximum = 0

        def slow_ffmpeg(command, **_kwargs):
            nonlocal active, maximum
            with state_lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.12)
            Path(command[-1]).write_bytes(b"\xff\xd8\xffmock-jpeg")
            with state_lock:
                active -= 1
            return types.SimpleNamespace(returncode=0)

        with patch("app.services.gallery_thumbnails.subprocess.run", side_effect=slow_ffmpeg):
            with ThreadPoolExecutor(max_workers=5) as pool:
                posters = list(pool.map(
                    lambda source: get_video_poster(str(source), cache_dir=str(self.cache), ffmpeg="ffmpeg"),
                    sources,
                ))
        self.assertEqual(len(posters), 5)
        self.assertEqual(maximum, 2)

    def test_missing_corrupt_and_nonvideo_sources_fail_safely(self):
        missing = self.root / "missing.mp4"
        self.assertIsNone(get_video_poster(str(missing), cache_dir=str(self.cache), ffmpeg="ffmpeg"))
        text_file = self.root / "readme.txt"
        text_file.write_text("not video", encoding="utf-8")
        self.assertIsNone(get_video_poster(str(text_file), cache_dir=str(self.cache), ffmpeg="ffmpeg"))

        corrupt = self.root / "corrupt.mkv"
        corrupt.write_bytes(b"not a video")
        with patch("app.services.gallery_thumbnails.subprocess.run",
                   return_value=types.SimpleNamespace(returncode=1)):
            self.assertIsNone(get_video_poster(str(corrupt), cache_dir=str(self.cache), ffmpeg="ffmpeg"))

    def test_route_resolver_keeps_workspace_upload_and_traversal_boundaries(self):
        app_root = self.root / "app"
        outputs = self.root / "outputs"
        folder_a = outputs / "Folder A"
        folder_b = outputs / "Folder B"
        uploads = app_root / "uploads"
        for directory in (outputs, folder_a, folder_b, uploads):
            directory.mkdir(parents=True, exist_ok=True)
        for directory, contents in (
            (outputs, b"default"),
            (folder_a, b"folder-a"),
            (folder_b, b"folder-b"),
            (uploads, b"upload"),
        ):
            (directory / "same.mp4").write_bytes(contents)
        (outputs / "only-default.mp4").write_bytes(b"default-only")
        outside = self.root / "outside.mp4"
        outside.write_bytes(b"outside")

        namespace = _load_gallery_route_functions(self.root)
        resolve = namespace["_resolve_gallery_media_file"]
        with patch("os.getcwd", return_value=str(app_root)):
            self.assertEqual(resolve("same.mp4", "Folder A"), str(folder_a / "same.mp4"))
            self.assertEqual(resolve("same.mp4", "Folder B"), str(folder_b / "same.mp4"))
            self.assertEqual(resolve("same.mp4", "__uploads__"), str(uploads / "same.mp4"))
            with self.assertRaises(_RouteError) as missing:
                resolve("only-default.mp4", "Folder A")
            self.assertEqual(missing.exception.status_code, 404)
            with self.assertRaises(_RouteError) as traversal:
                resolve(os.path.join("..", "..", "outside.mp4"), "Folder A")
            self.assertEqual(traversal.exception.status_code, 404)
            with self.assertRaises(_RouteError) as bad_workspace:
                resolve("same.mp4", os.path.join("..", "outside"))
            self.assertEqual(bad_workspace.exception.status_code, 400)

    def test_thumbnail_route_uses_shared_resolver_and_revalidation_headers(self):
        app_root = self.root / "app"
        folder = self.root / "outputs" / "Folder A"
        uploads = app_root / "uploads"
        folder.mkdir(parents=True)
        uploads.mkdir(parents=True)
        (folder / "clip.mp4").write_bytes(b"video")
        namespace = _load_gallery_route_functions(self.root)
        posters = []

        def response(path, *, media_type, headers):
            return {"path": path, "media_type": media_type, "headers": headers}

        namespace["FileResponse"] = response
        package = types.ModuleType("services")
        package.__path__ = []
        service = types.ModuleType("services.gallery_thumbnails")
        service.SUPPORTED_POSTER_SIZES = frozenset(SUPPORTED_SIZES)
        service.get_video_poster = lambda path, *, size=480: posters.append((path, size)) or str(self.cache / f"poster-{size}.jpg")
        with patch.dict(sys.modules, {"services": package, "services.gallery_thumbnails": service}):
            with patch("os.getcwd", return_value=str(app_root)):
                result = namespace["serve_gallery_thumbnail"]("clip.mp4", "Folder A")
                larger = namespace["serve_gallery_thumbnail"]("clip.mp4", "Folder A", 960)
                with self.assertRaises(_RouteError) as unsupported:
                    namespace["serve_gallery_thumbnail"]("clip.mp4", "Folder A", 720)

        self.assertEqual(posters, [(str(folder / "clip.mp4"), 480), (str(folder / "clip.mp4"), 960)])
        self.assertEqual(result["path"], str(self.cache / "poster-480.jpg"))
        self.assertEqual(result["media_type"], "image/jpeg")
        self.assertEqual(result["headers"]["Cache-Control"], "private, max-age=60, must-revalidate")
        self.assertEqual(larger["path"], str(self.cache / "poster-960.jpg"))
        self.assertEqual(unsupported.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
