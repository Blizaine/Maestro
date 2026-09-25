import ast
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
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

    def test_real_ffmpeg_video_generates_a_decodable_jpeg_when_available(self):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("ffmpeg is not installed")
        source = self.root / "tiny.mp4"
        created = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-f", "lavfi", "-i", "color=c=red:s=80x64:r=5:d=0.4",
             "-frames:v", "2", "-threads", "1", "-pix_fmt", "yuv420p",
             "-c:v", "mpeg4", str(source)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
        if created.returncode != 0 or not source.is_file():
            self.skipTest("installed ffmpeg lacks the tiny-video test encoder")

        poster = get_video_poster(str(source), cache_dir=str(self.cache), ffmpeg=ffmpeg)
        self.assertIsNotNone(poster)
        self.assertTrue(Path(poster).read_bytes().startswith(b"\xff\xd8\xff"))
        decoded = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", poster,
             "-frames:v", "1", "-f", "null", "-"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        self.assertEqual(decoded.returncode, 0)

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
        service.get_video_poster = lambda path: posters.append(path) or str(self.cache / "poster.jpg")
        with patch.dict(sys.modules, {"services": package, "services.gallery_thumbnails": service}):
            with patch("os.getcwd", return_value=str(app_root)):
                result = namespace["serve_gallery_thumbnail"]("clip.mp4", "Folder A")

        self.assertEqual(posters, [str(folder / "clip.mp4")])
        self.assertEqual(result["path"], str(self.cache / "poster.jpg"))
        self.assertEqual(result["media_type"], "image/jpeg")
        self.assertEqual(result["headers"]["Cache-Control"], "private, max-age=60, must-revalidate")


if __name__ == "__main__":
    unittest.main()
