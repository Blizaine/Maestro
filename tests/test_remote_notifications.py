from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_APP_DIR = os.path.join(_ROOT, "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.remote_access import TailscaleManager  # noqa: E402
from services.web_push import WebPushService  # noqa: E402


def _subscription(endpoint: str = "https://push.example.test/device") -> dict:
    return {
        "endpoint": endpoint,
        "expirationTime": None,
        "keys": {"p256dh": "public-key", "auth": "auth-secret"},
    }


class TestWebPushService(unittest.TestCase):
    def test_vapid_key_is_machine_local_and_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            first = WebPushService(directory)
            second = WebPushService(directory)
            self.assertEqual(first.public_key, second.public_key)
            self.assertTrue(first.public_key.startswith("B"))
            self.assertGreater(len(first.public_key), 80)
            saved = json.loads(
                Path(directory, "web_push.json").read_text(encoding="utf-8")
            )
            self.assertIn("BEGIN PRIVATE KEY", saved["vapid_private_key"])

    def test_subscribe_updates_in_place_and_unsubscribe_removes(self):
        with tempfile.TemporaryDirectory() as directory:
            service = WebPushService(directory)
            first = service.subscribe(
                _subscription(),
                preferences={"notifyCompleted": False},
                origin="https://maestro.example.ts.net/path",
                label="Phone",
            )
            second = service.subscribe(
                _subscription(),
                preferences={"notifyCompleted": True},
                origin="https://maestro.example.ts.net",
                label="Same phone",
            )
            self.assertEqual(first["subscription_count"], 1)
            self.assertEqual(second["subscription_count"], 1)
            removed = service.unsubscribe(_subscription()["endpoint"])
            self.assertTrue(removed["unsubscribed"])
            self.assertEqual(removed["subscription_count"], 0)

    def test_delivery_honors_category_preferences(self):
        with tempfile.TemporaryDirectory() as directory:
            service = WebPushService(directory)
            service.subscribe(
                _subscription(),
                preferences={"notifyCompleted": False, "notifyFailed": True},
                origin="https://maestro.example.ts.net",
            )
            calls = []

            class FakeWebPushException(Exception):
                pass

            parsed_pem = []
            signer = object()

            class FakeVapid:
                @classmethod
                def from_pem(cls, private_key):
                    parsed_pem.append(private_key)
                    return signer

                @classmethod
                def from_string(cls, private_key):
                    raise AssertionError(
                        f"PEM key incorrectly parsed as encoded key: {private_key!r}"
                    )

            fake_module = types.SimpleNamespace(
                WebPushException=FakeWebPushException,
                webpush=lambda **kwargs: calls.append(kwargs),
            )
            fake_vapid = types.SimpleNamespace(Vapid=FakeVapid)
            with patch.dict(
                sys.modules,
                {"pywebpush": fake_module, "py_vapid": fake_vapid},
            ):
                completed = service._send(
                    category="completion",
                    title="Complete",
                    body="Done",
                    tag="complete-1",
                )
                failed = service._send(
                    category="failure",
                    title="Failed",
                    body="Nope",
                    tag="failed-1",
                )
            self.assertEqual(completed.attempted, 0)
            self.assertEqual(failed.delivered, 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(parsed_pem), 1)
            self.assertIn(b"BEGIN PRIVATE KEY", parsed_pem[0])
            self.assertIs(calls[0]["vapid_private_key"], signer)
            self.assertEqual(
                calls[0]["vapid_claims"]["sub"],
                "mailto:blizaine@users.noreply.github.com",
            )
            payload = json.loads(calls[0]["data"])
            self.assertEqual(payload["url"], "https://maestro.example.ts.net")
            self.assertTrue(payload["onlyWhenHidden"])

    def test_legacy_encoded_vapid_key_uses_string_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            service = WebPushService(directory)
            service.subscribe(
                _subscription(), origin="https://maestro.example.ts.net"
            )
            service._state["vapid_private_key"] = "legacy-encoded-key"
            parsed_strings = []
            signer = object()

            class FakeVapid:
                @classmethod
                def from_pem(cls, _private_key):
                    raise AssertionError("Legacy encoded key was treated as PEM")

                @classmethod
                def from_string(cls, private_key):
                    parsed_strings.append(private_key)
                    return signer

            calls = []
            fake_webpush = types.SimpleNamespace(
                WebPushException=RuntimeError,
                webpush=lambda **kwargs: calls.append(kwargs),
            )
            fake_vapid = types.SimpleNamespace(Vapid=FakeVapid)
            with patch.dict(
                sys.modules,
                {"pywebpush": fake_webpush, "py_vapid": fake_vapid},
            ):
                result = service.send_test()

            self.assertEqual(result.delivered, 1)
            self.assertEqual(parsed_strings, ["legacy-encoded-key"])
            self.assertIs(calls[0]["vapid_private_key"], signer)


class TestTailscaleManager(unittest.TestCase):
    def _completed(self, args, stdout="", stderr="", returncode=0):
        return subprocess.CompletedProcess(args, returncode, stdout, stderr)

    def test_status_reports_private_https_route_for_current_port(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = TailscaleManager(directory, 42014)

            def fake_run(_executable, arguments, **_kwargs):
                if arguments == ["status", "--json"]:
                    return self._completed(arguments, json.dumps({
                        "BackendState": "Running",
                        "Self": {
                            "Online": True,
                            "DNSName": "maestro.tailnet.ts.net.",
                        },
                    }))
                return self._completed(
                    arguments,
                    "https://maestro.tailnet.ts.net\n"
                    "|-- / proxy http://127.0.0.1:42014\n",
                )

            with patch(
                "services.remote_access.find_tailscale",
                return_value="tailscale",
            ), patch.object(manager, "_run", side_effect=fake_run):
                status = manager.status()
            self.assertTrue(status["connected"])
            self.assertTrue(status["configured"])
            self.assertEqual(
                status["https_url"], "https://maestro.tailnet.ts.net"
            )

    def test_enable_targets_loopback_and_persists_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = TailscaleManager(directory, 42123)
            calls = []

            def fake_run(_executable, arguments, **_kwargs):
                calls.append(arguments)
                if arguments == ["status", "--json"]:
                    return self._completed(arguments, json.dumps({
                        "BackendState": "Running",
                        "Self": {
                            "Online": True,
                            "DNSName": "maestro.tailnet.ts.net.",
                        },
                    }))
                if arguments[:2] == ["serve", "status"]:
                    return self._completed(
                        arguments,
                        "https://maestro.tailnet.ts.net\n"
                        "http://127.0.0.1:42123\n",
                    )
                return self._completed(arguments)

            with patch(
                "services.remote_access.find_tailscale",
                return_value="tailscale",
            ), patch.object(manager, "_run", side_effect=fake_run):
                result = manager.enable()

            self.assertTrue(result["enabled"])
            self.assertIn([
                "serve", "--bg", "--yes", "--https=443",
                "http://127.0.0.1:42123"
            ], calls)
            preference = json.loads(
                Path(directory, "remote_access.json").read_text(encoding="utf-8")
            )
            self.assertEqual(preference["target_port"], 42123)
            self.assertTrue(preference["pinokio_port_lock"])

    def test_refresh_skips_cli_when_saved_port_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            preference = {
                "version": 2,
                "enabled": True,
                "target_port": 42123,
                "pinokio_port_lock": True,
            }
            Path(directory, "remote_access.json").write_text(
                json.dumps(preference),
                encoding="utf-8",
            )
            manager = TailscaleManager(directory, 42123)

            with patch.object(manager, "enable") as enable:
                manager.refresh_if_enabled()

            enable.assert_not_called()

    def test_in_app_toggle_preserves_windows_restore_helper_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            preference = {
                "version": 3,
                "enabled": True,
                "target_port": 42123,
                "pinokio_port_lock": True,
                "windows_restore_task": True,
                "windows_restore_task_name": "Maestro Tailscale Serve",
            }
            path = Path(directory, "remote_access.json")
            path.write_text(json.dumps(preference), encoding="utf-8")
            manager = TailscaleManager(directory, 42123)

            manager._save_preference(False)
            saved = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(saved["version"], 3)
            self.assertFalse(saved["enabled"])
            self.assertTrue(saved["windows_restore_task"])
            self.assertEqual(
                saved["windows_restore_task_name"],
                "Maestro Tailscale Serve",
            )

    def test_enable_preserves_an_unrelated_existing_serve_route(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = TailscaleManager(directory, 42123)

            def fake_run(_executable, arguments, **_kwargs):
                if arguments == ["status", "--json"]:
                    return self._completed(arguments, json.dumps({
                        "BackendState": "Running",
                        "Self": {
                            "Online": True,
                            "DNSName": "maestro.tailnet.ts.net.",
                        },
                    }))
                return self._completed(
                    arguments,
                    "https://maestro.tailnet.ts.net\n"
                    "|-- / proxy http://127.0.0.1:9999\n",
                )

            with patch(
                "services.remote_access.find_tailscale",
                return_value="tailscale",
            ), patch.object(manager, "_run", side_effect=fake_run):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "already has a different Tailscale Serve route",
                ):
                    manager.enable()

            self.assertFalse(
                Path(directory, "remote_access.json").exists()
            )


class TestRemoteNotificationWiring(unittest.TestCase):
    def test_launcher_preserves_required_url_capture_contract(self):
        for filename in ("start.js", "start_sol.js"):
            source = Path(_ROOT, filename).read_text(encoding="utf-8")
            self.assertIn('"event": "/(http:\\/\\/[0-9.:]+)/"', source)
            self.assertIn('url: "{{input.event[1]}}"', source)
            self.assertIn("port", source)

    def test_launcher_never_uses_public_tailscale_funnel(self):
        source = Path(_ROOT, "tailscale_setup.js").read_text(encoding="utf-8")
        for token in (
            '"serve"',
            '"--bg"',
            '"--yes"',
            '"--https=443"',
            '"http://127.0.0.1:{{args.port}}"',
        ):
            self.assertIn(token, source)
        self.assertNotIn(" tailscale funnel ", source.lower())

    def test_tailscale_setup_uses_one_noninteractive_elevated_command(self):
        source = Path(_ROOT, "tailscale_setup.js").read_text(encoding="utf-8")
        self.assertEqual(source.count("sudo: true"), 1)
        self.assertNotIn("input: true", source)
        self.assertNotIn('}}\" up', source)
        self.assertNotIn("serve status", source)
        self.assertIn('"app/settings/remote_access.json"', source)
        self.assertIn("pinokio_port_lock: true", source)
        self.assertIn("scripts/tailscale_windows_setup.ps1", source)
        self.assertIn("windows_restore_task: isWindows", source)

    def test_windows_setup_installs_a_fixed_on_demand_restore_task(self):
        source = Path(
            _ROOT,
            "scripts",
            "tailscale_windows_setup.ps1",
        ).read_text(encoding="utf-8")
        self.assertIn('$taskName = "Maestro Tailscale Serve"', source)
        self.assertIn("New-ScheduledTaskAction", source)
        self.assertIn("-Execute $TailscalePath", source)
        self.assertIn("-RunLevel Highest", source)
        self.assertIn("Register-ScheduledTask", source)
        self.assertIn("Start-ScheduledTask", source)
        self.assertIn("-RestartCount 3", source)
        self.assertIn('http://127.0.0.1:$Port', source)
        self.assertNotIn("New-ScheduledTaskTrigger", source)
        self.assertNotIn("funnel", source.lower())

    def test_launchers_reuse_the_opted_in_tailscale_port(self):
        for filename in ("start.js", "start_sol.js"):
            source = Path(_ROOT, filename).read_text(encoding="utf-8")
            self.assertIn('method: "json.get"', source)
            self.assertIn("local.remote_access.pinokio_port_lock", source)
            self.assertIn("local.remote_access.target_port", source)

    def test_launchers_dispatch_windows_restore_without_elevation(self):
        for filename in ("start.js", "start_sol.js"):
            source = Path(_ROOT, filename).read_text(encoding="utf-8")
            self.assertIn("local.remote_access.windows_restore_task", source)
            self.assertIn('"schtasks.exe"', source)
            self.assertIn('"Maestro Tailscale Serve"', source)
            self.assertNotIn("sudo: true", source)

    def test_noop_update_still_repairs_web_push_runtime(self):
        source = Path(_ROOT, "update.js").read_text(encoding="utf-8")
        self.assertIn('id: "uptodate"', source)
        self.assertIn("uv pip install pywebpush==2.3.0", source)
        self.assertIn("Web Push runtime verified", source)

    def test_secure_access_action_is_discoverable_while_stopped(self):
        launcher = Path(_ROOT, "pinokio.js").read_text(encoding="utf-8")
        setup = Path(_ROOT, "tailscale_setup.js").read_text(encoding="utf-8")
        self.assertGreaterEqual(
            launcher.count("Secure Remote Access (Tailscale)"),
            3,
        )
        self.assertIn("which('tailscale') && !args.port", setup)
        self.assertIn("Start Maestro first", setup)

    def test_backend_routes_and_terminal_hooks_exist(self):
        source = Path(_APP_DIR, "launch.py").read_text(encoding="utf-8")
        self.assertIn('/api/v1/notifications/push/subscribe', source)
        self.assertIn('/api/v1/remote-access/tailscale/enable', source)
        self.assertIn('category="completion"', source)
        self.assertIn('category="failure"', source)
        self.assertIn('category="queue"', source)


if __name__ == "__main__":
    unittest.main()
