"""Bounded tests for opt-in native MTP loading in the local LLM service."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


_ROOT = Path(__file__).resolve().parents[1]
_APP = str(_ROOT / "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from services import llm_service


SALUKI_REPO = "Kujira/Underdog-Saluki-27B-1.0-MTP-Abliterated-GGUF"
QWEN38_REPO = "JonathanColetti/Qwen3.8-27B-Uncensored-GGUF"
GEMMA_REPO = "Abhiray/gemma-4-E4B-it-heretic-GGUF"
MTP_FLAGS = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]


class FakeProcess:
    def __init__(self, args):
        self.args = args
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


class TestLlmMtp(unittest.TestCase):
    def setUp(self):
        self.patchers = []
        for name, value in (
            ("_process", None),
            ("_model_id", ""),
            ("_device", ""),
            ("_server_port", 0),
            ("_vision_available", False),
            ("_mtp_acceleration", False),
            ("_provider", "local"),
            ("_remote_url", ""),
            ("_api_key", ""),
            ("_idle_timer", None),
            ("_idle_generation", 0),
            ("_active_uses", 0),
        ):
            patcher = mock.patch.object(llm_service, name, value)
            patcher.start()
            self.patchers.append(patcher)

        self.processes = []

        def start_process(args, **kwargs):
            process = FakeProcess(args)
            self.processes.append(process)
            return process

        self.popen_patcher = mock.patch.object(
            llm_service.subprocess, "Popen", side_effect=start_process
        )
        self.popen = self.popen_patcher.start()
        self.patchers.append(self.popen_patcher)

        self.get_patcher = mock.patch.object(
            llm_service.requests,
            "get",
            return_value=SimpleNamespace(
                status_code=200,
                text="{}",
                json=lambda: {"status": "ok"},
            ),
        )
        self.get_patcher.start()
        self.patchers.append(self.get_patcher)

        self.runtime_env_patcher = mock.patch.object(
            llm_service, "_llama_server_env", return_value={"PATH": "test"}
        )
        self.runtime_env_patcher.start()
        self.patchers.append(self.runtime_env_patcher)

        self.load_patches = [
            mock.patch.object(llm_service, "_get_server_exe", return_value="llama-server-test"),
            mock.patch.object(llm_service, "_download_gguf", side_effect=self.download),
            mock.patch.object(llm_service, "get_model_dir", return_value="cache"),
            mock.patch.object(llm_service.os, "makedirs"),
            mock.patch.object(llm_service.os.path, "getsize", return_value=1_000_000),
            mock.patch.object(llm_service, "_find_free_port", return_value=18123),
            mock.patch.object(llm_service, "_start_log_reader"),
            mock.patch.object(llm_service.time, "sleep"),
            mock.patch.object(llm_service, "_reset_idle_timer"),
            mock.patch.object(llm_service, "_check_mtp_runtime_capability"),
        ]
        self.load_mocks = [patcher.start() for patcher in self.load_patches]
        self.patchers.extend(self.load_patches)
        self.download_mock = self.load_mocks[1]
        self.capability_mock = self.load_mocks[-1]

        self.old_cache = dict(llm_service._MTP_RUNTIME_CAPABILITY_CACHE)
        llm_service._MTP_RUNTIME_CAPABILITY_CACHE.clear()

    def tearDown(self):
        try:
            llm_service._unload_inner()
            llm_service._MTP_RUNTIME_CAPABILITY_CACHE.clear()
            llm_service._MTP_RUNTIME_CAPABILITY_CACHE.update(self.old_cache)
        finally:
            for patcher in reversed(self.patchers):
                patcher.stop()

    @staticmethod
    def download(repo_id, filename, cache_dir):
        return os.path.join(cache_dir, filename)

    def load_local(self, repo_id=SALUKI_REPO, *, enabled=False):
        llm_service.load_model(
            model_id=repo_id,
            device="cpu",
            provider="local",
            mtp_acceleration=enabled,
        )

    def test_only_verified_saluki_weights_advertise_mtp(self):
        local_models = llm_service.get_available_models(provider="local")
        by_id = {model["id"]: model for model in local_models}
        self.assertIs(by_id[SALUKI_REPO]["supports_mtp"], True)
        self.assertIs(by_id[QWEN38_REPO]["supports_mtp"], False)
        self.assertIs(by_id[GEMMA_REPO]["supports_mtp"], False)
        self.assertIs(
            llm_service.MODEL_REGISTRY[SALUKI_REPO]["supports_mtp"], True
        )
        self.assertNotIn("supports_mtp", llm_service.MODEL_REGISTRY[QWEN38_REPO])

    def test_mtp_flags_are_opt_in_and_require_eligible_local_weights(self):
        self.load_local(SALUKI_REPO, enabled=False)
        self.assertFalse(_has_mtp_flags(self.processes[-1].args))

        self.load_local(QWEN38_REPO, enabled=True)
        self.assertFalse(_has_mtp_flags(self.processes[-1].args))
        self.capability_mock.assert_not_called()

        self.load_local(GEMMA_REPO, enabled=True)
        self.assertFalse(_has_mtp_flags(self.processes[-1].args))
        self.capability_mock.assert_not_called()

    def test_eligible_enabled_model_receives_exact_mtp_flags(self):
        self.load_local(SALUKI_REPO, enabled=True)
        command = self.processes[-1].args
        self.assertEqual(command[-4:], MTP_FLAGS)
        self.capability_mock.assert_called_once_with("llama-server-test")
        self.assertTrue(llm_service.get_status()["mtp_acceleration"])

    def test_remote_provider_bypasses_mtp_and_reports_it_disabled(self):
        llm_service.load_model(
            model_id="remote-model",
            provider="remote",
            remote_url="http://127.0.0.1:1234",
            mtp_acceleration=True,
        )
        self.assertEqual(self.processes, [])
        self.capability_mock.assert_not_called()
        self.assertFalse(llm_service.get_status()["mtp_acceleration"])

    def test_missing_runtime_flags_fail_before_weight_download(self):
        self.load_patches[-1].stop()
        self.runtime_help = mock.patch.object(
            llm_service.subprocess,
            "run",
            return_value=SimpleNamespace(
                stdout="--spec-type <type>",
                stderr="",
                returncode=0,
            ),
        )
        run = self.runtime_help.start()
        self.addCleanup(self.runtime_help.stop)

        with self.assertRaisesRegex(
            RuntimeError,
            "Update Maestro's llama.cpp runtime or disable MTP acceleration",
        ):
            self.load_local(SALUKI_REPO, enabled=True)

        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["llama-server-test", "--help"])
        self.assertEqual(run.call_args.kwargs["timeout"], 15)
        self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(run.call_args.kwargs["errors"], "replace")
        self.download_mock.assert_not_called()
        self.assertEqual(self.processes, [])
        self.assertFalse(llm_service._mtp_acceleration)

    def test_generic_spec_flags_without_draft_mtp_mode_are_rejected_and_cached(self):
        self.load_patches[-1].stop()
        with mock.patch.object(
            llm_service.subprocess,
            "run",
            return_value=SimpleNamespace(
                stdout="--spec-type none,draft-simple --spec-draft-n-max N",
                stderr="",
                returncode=0,
            ),
        ) as run:
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "missing draft-mtp"):
                    llm_service._check_mtp_runtime_capability("llama-server-test")

        run.assert_called_once()
        self.assertEqual(len(llm_service._MTP_RUNTIME_CAPABILITY_CACHE), 1)

    def test_runtime_capability_probe_is_bounded_and_cached(self):
        self.load_patches[-1].stop()
        with mock.patch.object(
            llm_service.subprocess,
            "run",
            return_value=SimpleNamespace(
                stdout="--spec-type none,draft-mtp --spec-draft-n-max <n>",
                stderr="",
                returncode=0,
            ),
        ) as run:
            llm_service._check_mtp_runtime_capability("llama-server-test")
            llm_service._check_mtp_runtime_capability("llama-server-test")

        run.assert_called_once()
        self.assertEqual(
            run.call_args.args[0], ["llama-server-test", "--help"]
        )
        self.assertEqual(run.call_args.kwargs["timeout"], 15)
        self.assertEqual(run.call_args.kwargs["env"], {"PATH": "test"})
        self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(run.call_args.kwargs["errors"], "replace")

    def test_transient_help_probe_failures_are_retried(self):
        self.load_patches[-1].stop()
        valid_help = SimpleNamespace(
            stdout="--spec-type none,draft-mtp --spec-draft-n-max <n>",
            stderr="",
            returncode=0,
        )
        with mock.patch.object(
            llm_service.subprocess,
            "run",
            side_effect=[
                llm_service.subprocess.TimeoutExpired(
                    ["llama-server-test", "--help"], timeout=15
                ),
                SimpleNamespace(stdout="", stderr="temporary failure", returncode=1),
                valid_help,
            ],
        ) as run:
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "Update Maestro's llama.cpp runtime"):
                    llm_service._check_mtp_runtime_capability("llama-server-test")
            llm_service._check_mtp_runtime_capability("llama-server-test")

        self.assertEqual(run.call_count, 3)
        self.assertEqual(len(llm_service._MTP_RUNTIME_CAPABILITY_CACHE), 1)

    def test_mode_change_reloads_once_and_same_mode_reuses_loaded_model(self):
        self.load_local(SALUKI_REPO, enabled=False)
        self.load_local(SALUKI_REPO, enabled=False)
        self.assertEqual(self.popen.call_count, 1)

        self.load_local(SALUKI_REPO, enabled=True)
        self.assertEqual(self.popen.call_count, 2)
        self.assertTrue(self.processes[0].terminated)
        self.assertEqual(self.processes[1].args[-4:], MTP_FLAGS)

        self.load_local(SALUKI_REPO, enabled=True)
        self.assertEqual(self.popen.call_count, 2)

    def test_unload_and_remote_switch_clear_actual_mtp_status(self):
        self.load_local(SALUKI_REPO, enabled=True)
        self.assertTrue(llm_service.get_status()["mtp_acceleration"])

        llm_service.unload_model()
        self.assertFalse(llm_service.get_status()["mtp_acceleration"])
        self.assertFalse(llm_service._mtp_acceleration)

        self.load_local(SALUKI_REPO, enabled=True)
        llm_service.load_model(
            model_id="remote-model",
            provider="remote",
            remote_url="http://127.0.0.1:1234",
            mtp_acceleration=True,
        )
        self.assertFalse(llm_service.get_status()["mtp_acceleration"])
        self.assertFalse(llm_service._mtp_acceleration)


def _has_mtp_flags(command):
    return any(command[index:index + 4] == MTP_FLAGS for index in range(len(command)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
