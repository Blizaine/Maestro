"""GPU runtime selection without loading a model or touching a real GPU."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services import llama_linux_runtime as runtime
from services import llm_service


class CudaProbeTests(unittest.TestCase):
    def test_only_successful_cuda_device_rows_count(self):
        cases = [
            (0, "Available devices:\n CUDA0: NVIDIA RTX 5090 (32768 MiB)\n CUDA1: NVIDIA RTX 4090 (24576 MiB)", ["CUDA0", "CUDA1"]),
            (0, "load_backend: loaded CUDA backend\nAvailable devices:\n (none)", []),
            (0, "Available devices:\n Vulkan0: NVIDIA RTX 5090", []),
            (1, "Available devices:\n CUDA0: NVIDIA RTX 5090", []),
        ]
        for code, output, expected in cases:
            with self.subTest(output=output), mock.patch.object(runtime.subprocess, "run", return_value=subprocess.CompletedProcess([], code, output, "")) as run:
                devices, _ = runtime.probe_cuda("llama-server", {"LD_LIBRARY_PATH": "/runtime"})
                self.assertEqual(devices, expected)
                self.assertEqual(run.call_args.args[0], ["llama-server", "--list-devices"])
                self.assertEqual(run.call_args.kwargs["env"], {"LD_LIBRARY_PATH": "/runtime"})

    def test_loader_error_or_timeout_never_claims_cuda(self):
        for error in (OSError("libcublas missing"), subprocess.TimeoutExpired("llama-server", 30)):
            with self.subTest(error=error), mock.patch.object(runtime.subprocess, "run", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, "cannot use a CUDA device"):
                    runtime.require_cuda("llama-server", None)


class LinuxRuntimeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.bin_dir = self.directory.name
        self.exe = Path(self.bin_dir, "llama-server")
        self.exe.write_bytes(b"cached CPU server")
        for patch in (
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.object(llm_service, "_llama_server_build", return_value=10964),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def test_cached_cpu_runtime_is_built_as_cuda_without_downloading_cpu_archive(self):
        release = {"tag_name": "b10964", "assets": [{"name": "llama-b10964-bin-ubuntu-x64.tar.gz", "browser_download_url": "https://example.test/cpu.tar.gz"}]}
        with mock.patch.object(runtime, "probe_cuda", return_value=([], "(none)")), mock.patch.object(runtime, "build_cuda_runtime") as build, mock.patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(release).encode())) as download:
            llm_service._ensure_llama_server(self.bin_dir, device="cuda")
        self.assertEqual(download.call_count, 1)  # metadata only
        build.assert_called_once_with(self.bin_dir, "b10964", llm_service._llama_server_env)
        self.assertEqual(llm_service._read_llama_runtime_receipt(self.bin_dir)["backend"], "cuda")

    def test_usable_cuda_build_is_reused_without_network(self):
        with mock.patch.object(runtime, "probe_cuda", return_value=(["CUDA0"], "")), mock.patch.object(runtime, "build_cuda_runtime") as build, mock.patch("urllib.request.urlopen") as download:
            llm_service._ensure_llama_server(self.bin_dir, device="cuda")
        build.assert_not_called()
        download.assert_not_called()

    def test_driver_problem_on_known_cuda_build_does_not_rebuild(self):
        llm_service._write_llama_runtime_receipt(self.bin_dir, tag="b10964", build=10964, backend="cuda")
        with mock.patch.object(runtime, "probe_cuda", return_value=([], "CUDA driver unavailable")), mock.patch.object(runtime, "build_cuda_runtime") as build, mock.patch("urllib.request.urlopen") as download:
            with self.assertRaisesRegex(RuntimeError, "CUDA driver unavailable"):
                llm_service._ensure_llama_server(self.bin_dir, device="cuda")
        build.assert_not_called()
        download.assert_not_called()

    def test_explicit_cpu_preserves_cached_runtime_without_cuda_probe(self):
        with mock.patch.object(runtime, "probe_cuda") as probe, mock.patch("urllib.request.urlopen") as download:
            llm_service._ensure_llama_server(self.bin_dir, device="cpu")
        probe.assert_not_called()
        download.assert_not_called()


class LinuxCudaBuildTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.bin_dir = self.directory.name
        self.exe = Path(self.bin_dir, "llama-server")
        self.exe.write_bytes(b"original")
        for patch in (
            mock.patch.object(runtime, "find_nvcc", return_value="/cuda/bin/nvcc"),
            mock.patch.object(runtime.shutil, "which", side_effect=lambda name: "/tools/" + name),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def test_build_targets_cuda_then_installs_server_and_soname_libraries(self):
        commands = []
        def fake_run(command, **kwargs):
            commands.append(command)
            if "--build" in command:
                output = Path(command[command.index("--build") + 1], "bin")
                output.mkdir(parents=True)
                (output / "llama-server").write_bytes(b"CUDA server")
                (output / "libggml-cuda.so").write_bytes(b"CUDA library")
                (output / "libllama.so.0").write_bytes(b"SONAME library")
            return subprocess.CompletedProcess(command, 0)
        with mock.patch.object(runtime.subprocess, "run", side_effect=fake_run), mock.patch.object(runtime, "require_cuda", return_value=["CUDA0"]) as verify:
            runtime.build_cuda_runtime(self.bin_dir, "b10964", lambda _: None)
        self.assertIn("b10964", commands[0])
        self.assertIn("-DGGML_CUDA=ON", commands[1])
        self.assertIn("-DLLAMA_BUILD_NUMBER=10964", commands[1])
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=native", commands[1])
        self.assertIn("-DCMAKE_CUDA_COMPILER=/cuda/bin/nvcc", commands[1])
        self.assertEqual(commands[2][commands[2].index("--target") + 1], "llama-server")
        self.assertEqual(self.exe.read_bytes(), b"CUDA server")
        self.assertEqual(Path(self.bin_dir, "libllama.so.0").read_bytes(), b"SONAME library")
        self.assertEqual(verify.call_count, 2)  # before and after relocation

    def test_build_failure_keeps_existing_runtime(self):
        with mock.patch.object(runtime.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "cmake")), mock.patch.object(runtime, "require_cuda") as verify:
            with self.assertRaisesRegex(RuntimeError, "existing runtime was kept"):
                runtime.build_cuda_runtime(self.bin_dir, "b10964", lambda _: None)
        self.assertEqual(self.exe.read_bytes(), b"original")
        verify.assert_not_called()

    def test_missing_toolkit_is_actionable_and_does_not_download(self):
        with mock.patch.object(runtime, "find_nvcc", return_value=None), mock.patch.object(runtime.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "CUDA toolkit.*nvcc"):
                runtime.build_cuda_runtime(self.bin_dir, "b10964", lambda _: None)
        run.assert_not_called()
        self.assertEqual(self.exe.read_bytes(), b"original")

    def test_failed_cuda_verification_keeps_existing_runtime(self):
        with mock.patch.object(runtime.subprocess, "run"), mock.patch.object(runtime, "require_cuda", side_effect=RuntimeError("No device")):
            with self.assertRaisesRegex(RuntimeError, "No device"):
                runtime.build_cuda_runtime(self.bin_dir, "b10964", lambda _: None)
        self.assertEqual(self.exe.read_bytes(), b"original")


class LlmDeviceLoadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = self.directory.name
        self.repo = "JonathanColetti/Qwen3.8-27B-Uncensored-GGUF"
        self.gguf = Path(self.cache, "weights.gguf")
        self.gguf.write_bytes(b"test weights")
        self.proc = mock.Mock()
        self.proc.poll.return_value = None
        self.events = []
        health = mock.Mock(status_code=200, text='{"status":"ok"}')
        health.json.return_value = {"status": "ok"}
        for patch in (
            mock.patch.object(llm_service, "is_loaded", return_value=False),
            mock.patch.object(llm_service, "_get_server_exe", return_value="llama-server"),
            mock.patch.object(llm_service, "get_model_dir", return_value=self.cache),
            mock.patch.object(llm_service.subprocess, "Popen", return_value=self.proc),
            mock.patch.object(llm_service, "_find_free_port", return_value=42123),
            mock.patch.object(llm_service, "_start_log_reader", side_effect=lambda _: self.events.append("reader")),
            mock.patch.object(llm_service.requests, "get", side_effect=lambda *a, **kw: self.events.append("health") or health),
            *(mock.patch.object(llm_service, name, value) for name, value in (
                ("_process", None), ("_model_id", ""), ("_device", ""),
                ("_provider", "local"), ("_remote_url", ""), ("_api_key", ""),
                ("_vision_available", False), ("_server_port", 0),
            )),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def test_unavailable_cuda_fails_before_downloading_weights(self):
        with mock.patch.object(runtime, "require_cuda", side_effect=RuntimeError("No CUDA")), mock.patch.object(llm_service, "_download_gguf") as download:
            with self.assertRaisesRegex(RuntimeError, "No CUDA"):
                llm_service.load_model(self.repo, device="cuda")
        download.assert_not_called()
        llm_service.subprocess.Popen.assert_not_called()

    def test_cpu_is_explicit_and_cuda_pins_verified_devices(self):
        for device in ("cpu", "cuda"):
            with self.subTest(device=device), mock.patch.object(runtime, "require_cuda", return_value=["CUDA1"]) as verify, mock.patch.object(llm_service, "_download_gguf", return_value=str(self.gguf)) as download:
                self.events.clear()
                llm_service.load_model(self.repo, device=device)
                command = llm_service.subprocess.Popen.call_args.args[0]
                self.assertEqual(command[command.index("--device") + 1], "CUDA1" if device == "cuda" else "none")
                self.assertEqual(verify.call_count, 1 if device == "cuda" else 0)
                self.assertEqual(self.events, ["reader", "health"])
                self.assertEqual(download.call_args_list[1].args[1], "mmproj-Qwen3.8-27B-Uncensored-F16.gguf")
                self.assertEqual(llm_service._device, device)

    def test_renamed_projector_reuses_cached_previous_file(self):
        cache_dir = Path(self.cache, "Qwen3.8-27B-Uncensored")
        cache_dir.mkdir()
        old = cache_dir / "Qwen3.8-27B-Uncensored-vision-f16.gguf"
        old.write_bytes(b"cached projector")
        with mock.patch.object(llm_service, "_download_gguf", return_value=str(self.gguf)) as download:
            llm_service.load_model(self.repo, device="cpu")
        self.assertEqual(download.call_count, 1)
        command = llm_service.subprocess.Popen.call_args.args[0]
        self.assertEqual(command[command.index("--mmproj") + 1], str(old))
        self.assertTrue(llm_service._vision_available)


if __name__ == "__main__":
    unittest.main()
