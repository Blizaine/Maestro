"""Controller tests: no real subprocesses, network calls or GPU work."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("test_fleet", ROOT / "app/scripts/test_fleet.py")
fleet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fleet)


class FleetTests(unittest.TestCase):
    def run_target(self, **kwargs):
        with tempfile.TemporaryDirectory() as temp:
            return fleet.run_target("pterm", {"ref": "pinokio://pc:42000/api/Maestro.git"},
                                    "smoke", Path(temp), 30, **kwargs)

    def test_offline_never_dispatches(self):
        with patch.object(fleet, "pterm_status", return_value={"ready": False}), patch.object(fleet.subprocess, "Popen") as launch:
            self.assertEqual(self.run_target()["status"], "deferred")
            launch.assert_not_called()

    def test_busy_never_dispatches(self):
        with patch.object(fleet, "pterm_status", return_value={"ready": True, "running": True}), patch.object(fleet, "endpoint", return_value="http://pc:1234"), patch.object(fleet, "latest", return_value=None), patch.object(fleet, "busy_jobs", return_value=["user-job"]), patch.object(fleet.subprocess, "Popen") as launch:
            self.assertIn("User jobs", self.run_target()["reason"])
            launch.assert_not_called()

    def test_uncertain_previous_receipt_is_not_retried(self):
        with patch.object(fleet, "pterm_status", return_value={"ready": True, "running": True}), patch.object(fleet, "endpoint", return_value="http://pc:1234"), patch.object(fleet, "latest", return_value={"status": "uncertain"}), patch.object(fleet.subprocess, "Popen") as launch:
            self.assertEqual(self.run_target()["status"], "deferred")
            launch.assert_not_called()

    def test_remote_loopback_is_not_used(self):
        status = {"source": {"local": False}, "ready_url": "http://127.0.0.1:1111", "external_ready_urls": [{"url": "http://pc:2222"}]}
        with patch.object(fleet, "api", return_value={}) as request:
            self.assertEqual(fleet.endpoint(status), "http://pc:2222")
            request.assert_called_once_with("http://pc:2222", "/api/v1/system-stats")

    def test_untrusted_artifacts_cannot_escape_host_or_workspace(self):
        for name in ("../../secret.json", "http://attacker/secret.zip", "nested/weights.zip", "weights.safetensors", "x.zip?token=abc"):
            with self.assertRaises(ValueError):
                fleet.file_url("http://pc:1234", name)

    def test_collect_only_does_not_dispatch_or_check_user_queue(self):
        with patch.object(fleet, "pterm_status", return_value={"ready": True, "running": True}), patch.object(fleet, "endpoint", return_value="http://pc:1234"), patch.object(fleet, "latest", return_value={"run_id": "saved", "status": "completed", "finished_at": "today"}), patch.object(fleet, "collect", return_value=[]), patch.object(fleet.subprocess, "Popen") as launch:
            self.assertEqual(self.run_target(collect_only=True)["status"], "collected")
            launch.assert_not_called()

    def test_collect_only_leaves_running_worker_unresolved(self):
        with patch.object(fleet, "pterm_status", return_value={"ready": True, "running": True}), patch.object(fleet, "endpoint", return_value="http://pc:1234"), patch.object(fleet, "latest", return_value={"run_id": "saved", "status": "running"}), patch.object(fleet, "collect") as collect, patch.object(fleet.subprocess, "Popen") as launch:
            self.assertEqual(self.run_target(collect_only=True)["status"], "deferred")
            collect.assert_not_called()
            launch.assert_not_called()

    def test_mismatched_fresh_receipt_is_not_claimed_as_our_result(self):
        status = {"ready": True, "running": True, "local_entries": [{"script": "start.js", "local": {"url": "http://127.0.0.1:4567"}}]}
        receipt = {"run_id": "other", "mode": "render", "status": "completed", "finished_at": "today"}
        with patch.object(fleet, "pterm_status", return_value=status), patch.object(fleet, "endpoint", return_value="http://pc:9999"), patch.object(fleet, "latest", side_effect=[None, receipt]), patch.object(fleet, "busy_jobs", return_value=[]), patch.object(fleet, "collect") as collect, patch.object(fleet.subprocess, "Popen"):
            self.assertEqual(self.run_target()["status"], "uncertain")
            collect.assert_not_called()

    def test_json_receipt_is_written_before_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            def start(*args, **kwargs):
                self.assertEqual(json.loads((output / "receipt.json").read_text())["status"], "dispatched")
                raise OSError("connection lost")
            with patch.object(fleet, "pterm_status", return_value={"ready": True, "running": True}), patch.object(fleet, "endpoint", return_value="http://pc:1234"), patch.object(fleet, "source_url", return_value="http://127.0.0.1:1234"), patch.object(fleet, "latest", return_value=None), patch.object(fleet, "busy_jobs", return_value=[]), patch.object(fleet.subprocess, "Popen", side_effect=start):
                result = fleet.run_target("pterm", {"ref": "pinokio://pc:42000/api/Maestro.git"}, "smoke", output, 30)
                self.assertEqual(result["status"], "uncertain")

    def test_direct_action_uses_source_loopback_and_collects_terminal_receipt(self):
        status = {"ready": True, "running": True, "local_entries": [{"script": "start.js", "local": {"url": "http://127.0.0.1:4567"}}]}
        receipt = {"run_id": "new", "mode": "smoke", "status": "completed", "finished_at": "today"}
        with patch.object(fleet, "pterm_status", return_value=status), patch.object(fleet, "endpoint", return_value="http://pc:9999"), patch.object(fleet, "latest", side_effect=[None, receipt]), patch.object(fleet, "busy_jobs", return_value=[]), patch.object(fleet, "collect", return_value=["new.zip"]), patch.object(fleet.subprocess, "Popen") as launch:
            record = self.run_target()
            command = launch.call_args.args[0]
            self.assertEqual(command[:3], ["pterm", "start", "test_bench.js"])
            self.assertIn("--base_url=http://127.0.0.1:4567", command)
            self.assertEqual(record["status"], "collected")
            self.assertEqual(record["files"], ["new.zip"])

    def test_worker_artifact_field_names_are_supported(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(fleet, "fetch", return_value=b"test"):
            output = Path(temporary)
            files = fleet.collect("http://pc:1234", {"artifact_filename": "run.zip", "report_path": "run.html"}, output)
            self.assertEqual(files, ["run.zip", "run.html"])
            self.assertEqual((output / "run.zip").read_bytes(), b"test")

    def test_source_local_url_is_never_guessed_from_external_port(self):
        with self.assertRaises(ValueError):
            fleet.source_url({"local_entries": [{"script": "start.js", "local": {"url": "http://pc:1234"}}]})

    def test_persistent_uncertain_dispatch_blocks_new_controller_invocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            state = output / "state.json"
            state.write_text('{"status":"uncertain"}', encoding="utf-8")
            with patch.object(fleet, "pterm_status", return_value={"ready": True, "running": True}), patch.object(fleet, "endpoint", return_value="http://pc:1234"), patch.object(fleet, "latest", return_value=None), patch.object(fleet.subprocess, "Popen") as launch:
                result = fleet.run_target("pterm", {"ref": "pinokio://pc:42000/api/Maestro.git"}, "smoke", output / "receipt", 30, state_path=state)
                self.assertEqual(result["status"], "deferred")
                self.assertEqual(json.loads(state.read_text())["status"], "uncertain")
                launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
