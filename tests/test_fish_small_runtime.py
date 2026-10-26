from __future__ import annotations

import hashlib
import io
import os
import stat
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


_APP_DIR = Path(__file__).resolve().parents[1] / "app"
if os.fspath(_APP_DIR) not in sys.path:
    sys.path.insert(0, os.fspath(_APP_DIR))

from services import fish_small_runtime as runtime  # noqa: E402


def _zip_bytes(files: dict[str, bytes], *, symlink: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "../outside.txt")
    return buffer.getvalue()


def _tar_gz_bytes(files: dict[str, tuple[bytes, int]], *, link: bool = False) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, (contents, mode) in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            info.mode = mode
            archive.addfile(info, io.BytesIO(contents))
        if link:
            info = tarfile.TarInfo("linked")
            info.type = tarfile.SYMTYPE
            info.linkname = "../outside.txt"
            archive.addfile(info)
    return buffer.getvalue()


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _windows_fixture(*, missing: str | None = None) -> tuple[runtime._RuntimeSpec, tuple[bytes, bytes]]:
    bin_bytes = _zip_bytes(
        {
            "audiocpp_server.exe": b"server fixture",
            "ggml-cuda.dll": b"cuda backend",
            "LICENSE": b"upstream license",
        }
    )
    dependencies = {
        "cudart64_12.dll": b"cudart",
        "cublas64_12.dll": b"cublas",
        "cublasLt64_12.dll": b"cublasLt",
        "cufft64_11.dll": b"cufft",
    }
    if missing:
        dependencies.pop(missing)
    cudart_bytes = _zip_bytes(dependencies)
    assets = (
        runtime._Asset("bin.zip", hashlib.sha256(bin_bytes).hexdigest(), len(bin_bytes), "zip"),
        runtime._Asset("cudart.zip", hashlib.sha256(cudart_bytes).hexdigest(), len(cudart_bytes), "zip"),
    )
    spec = runtime._RuntimeSpec(
        variant="windows-x64-test",
        platform_name="windows",
        assets=assets,
        required_files=tuple(
            ["ggml-cuda.dll", "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cufft64_11.dll"]
        ),
        server_name="audiocpp_server.exe",
    )
    return spec, (bin_bytes, cudart_bytes)


def _linux_fixture() -> tuple[runtime._RuntimeSpec, tuple[bytes, bytes]]:
    bin_bytes = _tar_gz_bytes(
        {
            "./audiocpp_server": (b"linux server fixture", 0o755),
            "./LICENSE": (b"upstream license", 0o644),
        }
    )
    cudart_bytes = _tar_gz_bytes(
        {
            "./libcudart.so.12": (b"cudart", 0o644),
            "./libcublas.so.12": (b"cublas", 0o644),
            "./libcublasLt.so.12": (b"cublasLt", 0o644),
            "./libcufft.so.11": (b"cufft", 0o644),
        }
    )
    assets = (
        runtime._Asset("bin.tar.gz", hashlib.sha256(bin_bytes).hexdigest(), len(bin_bytes), "tar.gz"),
        runtime._Asset("cudart.tar.gz", hashlib.sha256(cudart_bytes).hexdigest(), len(cudart_bytes), "tar.gz"),
    )
    spec = runtime._RuntimeSpec(
        variant="linux-x64-test",
        platform_name="linux",
        assets=assets,
        required_files=("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12", "libcufft.so.11"),
        server_name="audiocpp_server",
    )
    return spec, (bin_bytes, cudart_bytes)


class TestFishSmallRuntime(unittest.TestCase):
    def _install_with_fixture(self, spec, archives, cache_root, *, progress=None, check_cancelled=None):
        responses = iter(archives)

        def open_url(_request, timeout):
            self.assertEqual(timeout, runtime._DOWNLOAD_TIMEOUT_SECONDS)
            return _Response(next(responses))

        with (
            patch.object(runtime, "_select_runtime_spec", return_value=spec),
            patch.object(runtime, "_runtime_base_dir", return_value=cache_root),
            patch.object(runtime.urllib.request, "urlopen", side_effect=open_url) as urlopen,
        ):
            result = runtime.ensure_runtime(
                check_cancelled=check_cancelled,
                progress=progress,
            )
        return result, urlopen

    def test_windows_uses_pinned_server_and_does_not_redownload_on_repeat(self):
        spec, archives = _windows_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            result, urlopen = self._install_with_fixture(spec, archives, Path(temporary))
            self.assertEqual(result.executable.name, "audiocpp_server.exe")
            self.assertEqual(result.root.name, spec.variant)
            self.assertEqual(result.backend, "cuda")
            self.assertEqual(result.release, runtime.RELEASE)
            self.assertIn("PATH", result.environment)
            self.assertTrue((result.root / "LICENSE").is_file())
            self.assertEqual(urlopen.call_count, 2)

            with (
                patch.object(runtime, "_select_runtime_spec", return_value=spec),
                patch.object(runtime, "_runtime_base_dir", return_value=Path(temporary)),
                patch.object(runtime.urllib.request, "urlopen", side_effect=AssertionError("redownloaded")),
            ):
                again = runtime.ensure_runtime()
            self.assertEqual(again.executable, result.executable)

    def test_linux_tar_runtime_sets_library_path_and_finds_server(self):
        spec, archives = _linux_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            result, _ = self._install_with_fixture(spec, archives, Path(temporary))
            self.assertEqual(result.executable.name, "audiocpp_server")
            self.assertTrue(result.executable.is_file())
            self.assertIn("LD_LIBRARY_PATH", result.environment)
            self.assertIn(os.fspath(result.root), result.environment["LD_LIBRARY_PATH"])

    def test_missing_cuda_dependency_fails_and_does_not_publish_runtime(self):
        spec, archives = _windows_fixture(missing="cufft64_11.dll")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(runtime.RuntimeDependencyError):
                self._install_with_fixture(spec, archives, Path(temporary))
            self.assertFalse((Path(temporary) / spec.variant).exists())
            self.assertFalse(any("staging" in child.name for child in Path(temporary).iterdir()))

    def test_bad_checksum_cleans_partial_install(self):
        spec, archives = _windows_fixture()
        bad_first = b"X" + archives[0][1:]
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(runtime, "_select_runtime_spec", return_value=spec),
                patch.object(runtime, "_runtime_base_dir", return_value=Path(temporary)),
                patch.object(runtime.urllib.request, "urlopen", return_value=_Response(bad_first)),
            ):
                with self.assertRaises(runtime.RuntimeIntegrityError):
                    runtime.ensure_runtime()
            self.assertFalse((Path(temporary) / spec.variant).exists())
            self.assertFalse(any("staging" in child.name for child in Path(temporary).iterdir()))

    def test_cancellation_during_download_removes_only_owned_partial(self):
        spec, archives = _windows_fixture()
        cancelled = False
        unrelated = None

        def check_cancelled():
            return cancelled

        def progress(stage, completed, _total):
            nonlocal cancelled
            if stage.startswith("download:") and completed > 0:
                cancelled = True

        with tempfile.TemporaryDirectory() as temporary:
            unrelated = Path(temporary) / "user-file.txt"
            unrelated.write_text("keep", encoding="utf-8")
            with self.assertRaises(runtime.RuntimeInstallCancelled):
                self._install_with_fixture(
                    spec,
                    archives,
                    Path(temporary),
                    progress=progress,
                    check_cancelled=check_cancelled,
                )
            self.assertTrue(unrelated.is_file())
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep")
            self.assertFalse((Path(temporary) / spec.variant).exists())
            self.assertFalse(any("staging" in child.name for child in Path(temporary).iterdir()))

    def test_owned_file_hash_mismatch_repairs_install(self):
        spec, archives = _windows_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            result, first_urlopen = self._install_with_fixture(spec, archives, Path(temporary))
            executable = result.root / "ggml-cuda.dll"
            executable.write_bytes(b"tampered1234")

            responses = iter(archives)

            def open_url(_request, timeout):
                self.assertEqual(timeout, runtime._DOWNLOAD_TIMEOUT_SECONDS)
                return _Response(next(responses))

            with (
                patch.object(runtime, "_select_runtime_spec", return_value=spec),
                patch.object(runtime, "_runtime_base_dir", return_value=Path(temporary)),
                patch.object(runtime.urllib.request, "urlopen", side_effect=open_url) as urlopen,
            ):
                repaired = runtime.ensure_runtime()
            self.assertEqual(repaired.executable, result.executable)
            self.assertEqual(urlopen.call_count, 2)
            self.assertEqual(first_urlopen.call_count, 2)
            self.assertEqual((repaired.root / "ggml-cuda.dll").read_bytes(), b"cuda backend")

    def test_zip_and_tar_reject_traversal_and_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "payload"
            root.mkdir()
            zip_path = Path(temporary) / "unsafe.zip"
            zip_path.write_bytes(_zip_bytes({"../outside.txt": b"escape"}))
            with self.assertRaises(runtime.UnsafeArchiveError):
                runtime._extract_archive(
                    zip_path,
                    root,
                    "zip",
                    seen={},
                    stage="extract",
                    check_cancelled=None,
                    progress=None,
                )
            self.assertFalse((Path(temporary) / "outside.txt").exists())

            symlink_zip = Path(temporary) / "symlink.zip"
            symlink_zip.write_bytes(_zip_bytes({"server.exe": b"x"}, symlink="link"))
            with self.assertRaises(runtime.UnsafeArchiveError):
                runtime._extract_archive(
                    symlink_zip,
                    root,
                    "zip",
                    seen={},
                    stage="extract",
                    check_cancelled=None,
                    progress=None,
                )

            traversal_tar = Path(temporary) / "unsafe.tar.gz"
            traversal_tar.write_bytes(_tar_gz_bytes({"../outside-tar.txt": (b"escape", 0o644)}))
            with self.assertRaises(runtime.UnsafeArchiveError):
                runtime._extract_archive(
                    traversal_tar,
                    root,
                    "tar.gz",
                    seen={},
                    stage="extract",
                    check_cancelled=None,
                    progress=None,
                )
            self.assertFalse((Path(temporary) / "outside-tar.txt").exists())

            link_tar = Path(temporary) / "link.tar.gz"
            link_tar.write_bytes(_tar_gz_bytes({}, link=True))
            with self.assertRaises(runtime.UnsafeArchiveError):
                runtime._extract_archive(
                    link_tar,
                    root,
                    "tar.gz",
                    seen={},
                    stage="extract",
                    check_cancelled=None,
                    progress=None,
                )

    def test_backend_and_platform_selection_are_explicit(self):
        with self.assertRaises(runtime.RuntimeInstallError):
            runtime._select_runtime_spec("cpu", None)
        with (
            patch.object(runtime.sys, "platform", "win32"),
            patch.object(runtime.platform, "machine", return_value="AMD64"),
        ):
            self.assertEqual(runtime._select_runtime_spec("cuda", (8, 6)).variant, "windows-x64-cuda12.4")
            self.assertEqual(runtime._select_runtime_spec("cuda", (8, 9)).variant, "windows-x64-cuda12.4")
            self.assertEqual(runtime._select_runtime_spec("cuda", (10, 0)).variant, "windows-x64-cuda13.3")
        with patch.object(runtime.sys, "platform", "darwin"):
            with self.assertRaises(runtime.RuntimeInstallError):
                runtime._select_runtime_spec("cuda", None)


if __name__ == "__main__":
    unittest.main()
