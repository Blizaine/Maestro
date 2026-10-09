import contextlib
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("test_ssh_client", ROOT / "app/scripts/test_ssh_client.py")
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class SshClientTests(unittest.TestCase):
    def with_record(self, record, callback):
        with tempfile.TemporaryDirectory() as temporary:
            app = Path(temporary)
            (app / "settings").mkdir()
            (app / "settings/test_ssh_server.json").write_text(json.dumps(record), encoding="utf-8")
            with patch.object(client, "APP", app), patch.object(client, "KEY_DIR", app / "settings/keys"), patch.object(client, "OUTPUT", app / "outputs"):
                callback(app)

    def test_host_shell_injection_rejected(self):
        def check(app):
            with patch.object(client, "execute") as execute:
                with self.assertRaises(ValueError):
                    client.diagnose()
                execute.assert_not_called()
        self.with_record({"host": "pc & echo injected", "account_name": "user"}, check)

    def test_root_traversal_rejected(self):
        def check(app):
            with patch.object(client, "execute") as execute:
                with self.assertRaises(ValueError):
                    client.diagnose()
                execute.assert_not_called()
        self.with_record({"host": "pc", "account_name": "user", "host_public_key": "ssh-ed25519 AAAA", "remote_root": "C:\\app\\..\\private"}, check)

    def test_host_key_change_rejected(self):
        def check(app):
            client.KEY_DIR.mkdir()
            (client.KEY_DIR / "known_hosts").write_text("pc ssh-ed25519 OLD\n", encoding="ascii")
            with patch.object(client, "execute") as execute:
                with self.assertRaisesRegex(ValueError, "host key changed"):
                    client.diagnose()
                execute.assert_not_called()
        self.with_record({"host": "pc", "account_name": "user", "host_public_key": "ssh-ed25519 NEW"}, check)

    def test_diagnostics_require_pinned_key_and_never_print_log_contents(self):
        def check(app):
            class Result:
                stdout = json.dumps({"user": "pc/user", "computer": "pc", "processes": [], "logs": {"llm": ["private prompt"]}})
            with patch.object(client, "execute", return_value=Result()) as execute, patch.object(client.shutil, "which", return_value="ssh"), patch("builtins.print") as output:
                client.diagnose()
                command = execute.call_args.args[0]
                self.assertIn("StrictHostKeyChecking=yes", command)
                self.assertIn("BatchMode=yes", command)
                self.assertIn("IdentitiesOnly=yes", command)
                self.assertNotIn("private prompt", output.call_args.args[0])
                self.assertIn("private prompt", (client.OUTPUT / "ssh-diagnostics.json").read_text())
        self.with_record({"host": "pc", "account_name": "user", "host_public_key": "ssh-ed25519 AAAA", "remote_root": "C:\\pinokio\\api\\Maestro.git"}, check)

    def test_subprocess_failure_reports_stderr_without_command_or_secret_arguments(self):
        private_values = ("ssh.exe", "encoded-script-secret", "private-key-bytes")
        failure = subprocess.CalledProcessError(
            255,
            [private_values[0], "-EncodedCommand", private_values[1], "-i", private_values[2]],
            stderr="Connection timed out",
        )
        output = io.StringIO()
        with patch.object(client, "diagnose", side_effect=failure), patch.object(
            sys, "argv", ["test_ssh_client.py", "--action=diagnose"]
        ), contextlib.redirect_stderr(output):
            self.assertEqual(client.main(), 2)

        diagnostic = output.getvalue()
        self.assertIn("status 255", diagnostic)
        self.assertIn("stderr: Connection timed out", diagnostic)
        for value in private_values:
            self.assertNotIn(value, diagnostic)

    def test_subprocess_failure_stderr_is_bounded(self):
        failure = subprocess.CalledProcessError(1, ["ssh"], stderr="x" * 5000)
        diagnostic = client.subprocess_failure_message(failure)
        self.assertLessEqual(len(diagnostic), 1600)
        self.assertIn("...", diagnostic)


if __name__ == "__main__":
    unittest.main()
