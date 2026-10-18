import contextlib
import io
import importlib.util
import json
import os
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
                self.assertNotIn("-tt", command)
                self.assertNotIn("private prompt", output.call_args.args[0])
                self.assertEqual((client.OUTPUT / "ssh-diagnostics.json").read_text(), Result.stdout)
        self.with_record({"host": "pc", "account_name": "user", "host_public_key": "ssh-ed25519 AAAA", "remote_root": "C:\\pinokio\\api\\Maestro.git"}, check)

    def test_pty_mode_uses_framed_chunked_utf8_json_and_sanitizes_terminal_sequences(self):
        def check(app):
            import base64
            payload = {"user": "pc/user", "computer": "pc", "processes": [],
                       "logs": {"sample": ["private diagnostic line " * 18]}}
            encoded = base64.b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii")
            chunks = [encoded[index:index + client.PTY_JSON_CHUNK_SIZE]
                      for index in range(0, len(encoded), client.PTY_JSON_CHUNK_SIZE)]
            self.assertGreater(len(chunks), 1)
            self.assertTrue(all(len(chunk) <= 64 for chunk in chunks))
            stdout = ("login banner\r\n\x1b]0;remote title\x07\x1b[32m" + client.PTY_JSON_BEGIN +
                      "\x1b[0m\r\n" + "\r\n".join(chunks) + "\r\n" + client.PTY_JSON_END +
                      "\r\n\x1b]8;;https://example.invalid\x1b\\prompt")
            result = subprocess.CompletedProcess([], 0, stdout=stdout)
            record_path = app / "settings/test_ssh_server.json"
            record = {"host": "pc", "account_name": "user", "host_public_key": "ssh-ed25519 AAAA",
                      "remote_root": "C:\\pinokio\\api\\Maestro.git", "request_pty": True}
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with patch.object(client, "execute", return_value=result) as execute, patch.object(
                client.shutil, "which", return_value="ssh"
            ), patch("builtins.print"):
                client.diagnose(record_path)
            command = execute.call_args.args[0]
            destination = "user@pc"
            self.assertLess(command.index("-tt"), command.index(destination))
            remote_script = base64.b64decode(command[command.index("-EncodedCommand") + 1]).decode("utf-16le")
            self.assertIn("[Text.Encoding]::UTF8.GetBytes($json)", remote_script)
            self.assertIn("WriteLine('MAESTRO_SSH_JSON_BEGIN')", remote_script)
            self.assertIn("WriteLine('MAESTRO_SSH_JSON_END')", remote_script)
            self.assertIn("$offset += 64", remote_script)
            artifact = (client.OUTPUT / "ssh-diagnostics-pc.json").read_text(encoding="utf-8")
            self.assertEqual(json.loads(artifact), payload)
            self.assertNotIn(client.PTY_JSON_BEGIN, artifact)
            self.assertNotIn(encoded, artifact)
        self.with_record({}, check)

    def test_pty_frame_rejects_missing_ambiguous_and_malformed_payloads(self):
        bad_outputs = (
            "remote banner without a frame",
            f"{client.PTY_JSON_BEGIN}\nnot!base64\n{client.PTY_JSON_END}",
            f"{client.PTY_JSON_BEGIN}\n{'A' * 65}\n{client.PTY_JSON_END}",
            f"{client.PTY_JSON_BEGIN}\n e30= \n{client.PTY_JSON_END}",
            f"{client.PTY_JSON_BEGIN}\ne30=\n{client.PTY_JSON_END}\n{client.PTY_JSON_END}",
        )
        for stdout in bad_outputs:
            with self.subTest(stdout=stdout[:48]), self.assertRaisesRegex(ValueError, "PTY diagnostics"):
                client.decode_pty_json(stdout)

    def test_request_pty_requires_a_json_boolean(self):
        def check(app):
            with patch.object(client, "execute") as execute:
                with self.assertRaisesRegex(ValueError, "request_pty must be a boolean"):
                    client.diagnose()
                execute.assert_not_called()
            self.assertFalse(client.KEY_DIR.exists())
        self.with_record({"host": "pc", "account_name": "user", "request_pty": "true"}, check)

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

    def test_domain_account_uses_separate_host_pin_for_another_machine(self):
        def check(app):
            client.KEY_DIR.mkdir()
            original = client.KEY_DIR / "known_hosts"
            original.write_text("first-pc ssh-ed25519 FIRST\n", encoding="ascii")
            record_path = app / "settings/test_ssh_laptop.json"
            record_path.write_text(json.dumps({
                "host": "192.168.68.98", "account_name": "cti-stl\\blaine.brown",
                "host_public_key": "ssh-ed25519 AAAA", "remote_root": "C:\\pinokio\\api\\Maestro.git-dev",
            }), encoding="utf-8")
            result = subprocess.CompletedProcess([], 0, stdout=json.dumps({"processes": []}))
            with patch.object(client, "execute", return_value=result) as execute, patch.object(client.shutil, "which", return_value="ssh"), patch("builtins.print"):
                client.diagnose(record_path)
            command = execute.call_args.args[0]
            self.assertIn("cti-stl\\blaine.brown@192.168.68.98", command)
            self.assertIn("UserKnownHostsFile=" + str(client.KEY_DIR / "known_hosts-192.168.68.98"), command)
            self.assertEqual(original.read_text(), "first-pc ssh-ed25519 FIRST\n")
            self.assertEqual((client.KEY_DIR / "known_hosts-192.168.68.98").read_text(), "192.168.68.98 ssh-ed25519 AAAA\n")
            self.assertTrue((client.OUTPUT / "ssh-diagnostics-192.168.68.98.json").exists())
            self.assertFalse((client.OUTPUT / "ssh-diagnostics.json").exists())
        self.with_record({}, check)

    def test_unsafe_or_malformed_domain_accounts_are_rejected(self):
        for account in ("domain\\user & echo injected", "domain\\user\\other", "\\user", "domain\\", "user@domain", "-l user"):
            with self.subTest(account=account):
                def check(app):
                    with patch.object(client, "execute") as execute:
                        with self.assertRaisesRegex(ValueError, "Invalid SSH host/account"):
                            client.diagnose()
                        execute.assert_not_called()
                self.with_record({"host": "pc", "account_name": account}, check)

    def test_selected_machine_host_key_change_is_rejected(self):
        def check(app):
            client.KEY_DIR.mkdir()
            known = client.KEY_DIR / "known_hosts-pc"
            known.write_text("pc ssh-ed25519 OLD\n", encoding="ascii")
            with patch.object(client, "execute") as execute:
                with self.assertRaisesRegex(ValueError, "host key changed"):
                    client.diagnose(app / "settings/test_ssh_server.json")
                execute.assert_not_called()
            self.assertEqual(known.read_text(), "pc ssh-ed25519 OLD\n")
        self.with_record({"host": "pc", "account_name": "domain\\user", "host_public_key": "ssh-ed25519 NEW"}, check)

    def test_cli_passes_selected_receipt_to_diagnostics(self):
        with patch.object(client, "diagnose") as diagnose, patch.object(sys, "argv", [
            "test_ssh_client.py", "--action=diagnose", "--server-config=other-server.json",
        ]):
            self.assertEqual(client.main(), 0)
        diagnose.assert_called_once_with(Path("other-server.json"))

    @unittest.skipUnless(sys.platform == "win32", "Windows PowerShell serialization regression")
    def test_windows_powershell_log_lines_stay_plain_and_bounded(self):
        powershell = Path(os.environ.get("WINDIR", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
        if not powershell.exists():
            self.skipTest("Windows PowerShell 5.1 is unavailable")

        def check(app):
            log = app / "logs/api/start.js/latest"
            log.parent.mkdir(parents=True)
            log.write_text("example log line\nsecond line\n", encoding="utf-8")
            record_path = app / "settings/test_ssh_server.json"
            record = json.loads(record_path.read_text())
            record["remote_root"] = str(app)
            record_path.write_text(json.dumps(record), encoding="utf-8")

            def run_script_locally(command, timeout=30):
                # Run the actual generated diagnostic body; no SSH connection
                # or model loading occurs in this platform regression check.
                encoded = command[command.index("-EncodedCommand") + 1]
                return subprocess.run([str(powershell), "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                                      capture_output=True, text=True, check=True, timeout=timeout)

            with patch.object(client, "execute", side_effect=run_script_locally), patch.object(client.shutil, "which", return_value="ssh"), patch("builtins.print"):
                client.diagnose()
            artifact = (client.OUTPUT / "ssh-diagnostics.json").read_text()
            self.assertLess(len(artifact), 20000)
            parsed = json.loads(artifact)
            self.assertEqual(parsed["logs"]["logs\\api\\start.js\\latest"], ["example log line", "second line"])
            self.assertNotIn("PSProvider", artifact)

        self.with_record({"host": "pc", "account_name": "user", "host_public_key": "ssh-ed25519 AAAA"}, check)

    def test_subprocess_failure_stderr_is_bounded(self):
        failure = subprocess.CalledProcessError(1, ["ssh"], stderr="x" * 5000)
        diagnostic = client.subprocess_failure_message(failure)
        self.assertLessEqual(len(diagnostic), 1600)
        self.assertIn("...", diagnostic)


if __name__ == "__main__":
    unittest.main()
