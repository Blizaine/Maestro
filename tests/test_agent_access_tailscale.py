"""Provisioning and validation coverage for the optional Tailscale TLS path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.agent_access.tailscale import (
    TailscaleCertificateError,
    TailscaleCertificateManager,
    normalize_tailscale_dns_name,
)

_HOSTNAME = "studio-node.example-tail.ts.net"


def _make_certificate_pair(
    hostname: str,
    *,
    days_valid: int = 90,
    self_signed: bool = False,
) -> tuple[bytes, bytes]:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    issuer_key = leaf_key if self_signed else ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)])
    issuer = subject if self_signed else x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Maestro test issuer")
    ])
    leaf = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(
            now - timedelta(days=2)
            if days_valid >= 0
            else now + timedelta(days=days_valid - 5)
        )
        .not_valid_after(now + timedelta(days=days_valid))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(hostname)]), critical=False)
        .sign(issuer_key, hashes.SHA256())
    )
    certificates = [leaf]
    if not self_signed:
        issuer_subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "Maestro test issuer")
        ])
        issuer_cert = (
            x509.CertificateBuilder()
            .subject_name(issuer_subject)
            .issuer_name(issuer_subject)
            .public_key(issuer_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=10))
            .not_valid_after(now + timedelta(days=1000))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(issuer_key, hashes.SHA256())
        )
        certificates.append(issuer_cert)
    chain = b"".join(
        certificate.public_bytes(serialization.Encoding.PEM)
        for certificate in certificates
    )
    key = leaf_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return chain, key


class FakeTailscale:
    def __init__(
        self,
        *,
        hostname: str = _HOSTNAME,
        connected: bool = True,
        certificate_hostname: str | None = None,
        days_valid: int = 90,
        self_signed: bool = False,
        mismatched_key: bool = False,
        fail_certificate: bool = False,
    ):
        self.hostname = hostname
        self.connected = connected
        self.certificate_hostname = certificate_hostname or hostname.rstrip(".")
        self.days_valid = days_valid
        self.self_signed = self_signed
        self.mismatched_key = mismatched_key
        self.fail_certificate = fail_certificate
        self.calls: list[list[str]] = []

    def run(self, command, **_kwargs):
        command = [str(value) for value in command]
        self.calls.append(command)
        if command[1:] == ["status", "--json"]:
            payload = {
                "BackendState": "Running" if self.connected else "Stopped",
                "Self": {"Online": self.connected, "DNSName": self.hostname},
                "Peers": {"should-not-be-read-or-returned": {"DNSName": "peer.ts.net"}},
            }
            return subprocess.CompletedProcess(
                command, 0, json.dumps(payload), ""
            )
        if command[0].lower().endswith("icacls.exe"):
            return subprocess.CompletedProcess(command, 0, "", "")
        if any(value.startswith("--cert-file=") for value in command):
            if self.fail_certificate:
                return subprocess.CompletedProcess(
                    command, 1, "raw peer.ts.net output", "PRIVATE KEY secret detail"
                )
            cert_path = Path(next(
                value.split("=", 1)[1]
                for value in command
                if value.startswith("--cert-file=")
            ))
            key_path = Path(next(
                value.split("=", 1)[1]
                for value in command
                if value.startswith("--key-file=")
            ))
            cert_bytes, key_bytes = _make_certificate_pair(
                self.certificate_hostname,
                days_valid=self.days_valid,
                self_signed=self.self_signed,
            )
            if self.mismatched_key:
                _, key_bytes = _make_certificate_pair(self.certificate_hostname)
            cert_path.write_bytes(cert_bytes)
            key_path.write_bytes(key_bytes)
            return subprocess.CompletedProcess(command, 0, "issued", "")
        raise AssertionError(f"Unexpected local command: {command[1:]}")


class TailscaleCertificateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Path(self.temp.name) / "settings"
        self.settings.mkdir()
        self.manager = TailscaleCertificateManager(self.settings)
        self.fake = FakeTailscale()
        self.patcher = patch(
            "services.agent_access.tailscale.find_tailscale",
            return_value="tailscale-test",
        )
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.run_patcher = patch(
            "services.agent_access.tailscale.subprocess.run",
            side_effect=self.fake.run,
        )
        self.run_patcher.start()
        self.addCleanup(self.run_patcher.stop)

    def test_dns_validation_allows_own_ts_net_name_and_rejects_urls_or_injection(self):
        self.assertEqual(
            normalize_tailscale_dns_name("Studio-Node.example-tail.ts.net."),
            _HOSTNAME,
        )
        for malformed in (
            "https://node.ts.net",
            "node.ts.net/path",
            "node.ts.net:443",
            "node.ts.net;--serve-demo",
            "node.other.net",
            ".ts.net",
            "node..ts.net",
            "node.ts.net..",
            "node ts.net",
            "équipe.ts.net",
        ):
            with self.subTest(malformed=malformed), self.assertRaises(
                TailscaleCertificateError
            ):
                normalize_tailscale_dns_name(malformed)

    def test_provisioning_uses_self_identity_and_publishes_complete_private_generation(self):
        material = self.manager.ensure_certificate()
        self.assertEqual(material.hostname, _HOSTNAME)
        self.assertTrue(material.cert_path.is_file())
        self.assertTrue(material.key_path.is_file())
        self.assertEqual(material.cert_path.parent.name, material.generation)
        pointer = json.loads((self.settings / "tailscale-certs" / "current.json").read_text())
        self.assertEqual(pointer, {"generation": material.generation, "hostname": _HOSTNAME})
        cert_command = next(call for call in self.fake.calls if "--cert-file=" in " ".join(call))
        self.assertEqual(cert_command[-1], _HOSTNAME)
        self.assertEqual(cert_command[:2], ["tailscale-test", "cert"])
        self.assertTrue(any(value.startswith("--min-validity=") for value in cert_command))
        detected = self.manager.detect()
        self.assertNotIn("peer.ts.net", repr(detected))
        self.assertNotIn("PRIVATE KEY", repr(material))
        if os.name != "nt":
            self.assertEqual(material.cert_path.parent.stat().st_mode & 0o777, 0o700)
            self.assertEqual(material.cert_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(material.key_path.stat().st_mode & 0o777, 0o600)

    def test_status_and_sni_load_are_cache_only(self):
        material = self.manager.ensure_certificate()
        before = len(self.fake.calls)
        with patch("services.agent_access.tailscale.subprocess.run", side_effect=AssertionError("unexpected subprocess")):
            status = self.manager.status()
            loaded = self.manager.load_certificate(_HOSTNAME)
        self.assertEqual(len(self.fake.calls), before)
        self.assertIsNone(status.connected)
        self.assertTrue(status.certificate_ready)
        self.assertEqual(loaded.generation, material.generation)
        self.assertNotIn("key_path", repr(status))

    def test_repeated_ensure_reuses_stable_certificate_without_another_issue(self):
        first = self.manager.ensure_certificate()
        second = self.manager.ensure_certificate()
        cert_calls = [
            call for call in self.fake.calls
            if any(value.startswith("--cert-file=") for value in call)
        ]
        self.assertEqual(first.generation, second.generation)
        self.assertEqual(len(cert_calls), 1)

    def test_malformed_identity_and_disconnected_status_never_issue_certificate(self):
        for fake in (
            FakeTailscale(hostname="https://attacker.ts.net"),
            FakeTailscale(connected=False),
        ):
            with self.subTest(hostname=fake.hostname, connected=fake.connected):
                self.fake = fake
                with patch(
                    "services.agent_access.tailscale.subprocess.run",
                    side_effect=fake.run,
                ), self.assertRaises(TailscaleCertificateError):
                    self.manager.ensure_certificate()
                self.assertFalse(any(
                    any(value.startswith("--cert-file=") for value in call)
                    for call in fake.calls
                ))

    def test_mismatched_key_wrong_san_and_self_signed_certificates_are_rejected(self):
        for fake, expected in (
            (FakeTailscale(mismatched_key=True), "do not match"),
            (FakeTailscale(certificate_hostname="other-node.example-tail.ts.net"), "does not cover"),
            (FakeTailscale(self_signed=True), "self-signed"),
        ):
            with self.subTest(expected=expected):
                self.manager._attempt_path.unlink(missing_ok=True)
                self.fake = fake
                with patch(
                    "services.agent_access.tailscale.subprocess.run",
                    side_effect=fake.run,
                ), self.assertRaisesRegex(TailscaleCertificateError, expected):
                    self.manager.ensure_certificate()
                self.assertFalse((self.settings / "tailscale-certs" / "current.json").exists())

    def test_expired_certificate_is_not_ready_and_explicit_ensure_reissues(self):
        self.fake = FakeTailscale(days_valid=-100)
        with patch(
            "services.agent_access.tailscale.subprocess.run",
            side_effect=self.fake.run,
        ), self.assertRaisesRegex(TailscaleCertificateError, "not currently valid"):
            self.manager.ensure_certificate()
        self.assertFalse((self.settings / "tailscale-certs" / "current.json").exists())

        self.fake = FakeTailscale(days_valid=90)
        self.manager._attempt_path.write_text(
            json.dumps({"attempted_at": 1}), encoding="utf-8"
        )
        with patch(
            "services.agent_access.tailscale.subprocess.run",
            side_effect=self.fake.run,
        ):
            material = self.manager.ensure_certificate()
        self.assertGreater(material.expires_at, datetime.now(timezone.utc))

    def test_issue_failure_is_redacted_and_keeps_previous_generation_active(self):
        first = self.manager.ensure_certificate()
        pointer_path = self.settings / "tailscale-certs" / "current.json"
        original_pointer = pointer_path.read_bytes()
        self.manager._attempt_path.write_text(
            json.dumps({"attempted_at": 1}), encoding="utf-8"
        )
        self.fake = FakeTailscale(fail_certificate=True)
        with patch(
            "services.agent_access.tailscale.subprocess.run",
            side_effect=self.fake.run,
        ), self.assertRaises(TailscaleCertificateError) as error:
            self.manager.renew()
        self.assertNotIn("PRIVATE KEY", str(error.exception))
        self.assertNotIn("peer.ts.net", str(error.exception))
        self.assertEqual(pointer_path.read_bytes(), original_pointer)
        self.assertEqual(self.manager.load_certificate(_HOSTNAME).generation, first.generation)

    def test_broad_local_permissions_fail_closed(self):
        material = self.manager.ensure_certificate()
        if os.name != "nt":
            material.key_path.chmod(0o644)
            with self.assertRaisesRegex(TailscaleCertificateError, "permissions are too broad"):
                self.manager.load_certificate(_HOSTNAME)
            self.assertFalse(self.manager.status().certificate_ready)

    def test_certificate_issue_cooldown_bounds_failed_retry_attempts(self):
        self.fake = FakeTailscale(fail_certificate=True)
        with patch(
            "services.agent_access.tailscale.subprocess.run",
            side_effect=self.fake.run,
        ):
            with self.assertRaises(TailscaleCertificateError):
                self.manager.ensure_certificate()
            cert_calls_after_first = sum(
                any(value.startswith("--cert-file=") for value in call)
                for call in self.fake.calls
            )
            with self.assertRaisesRegex(TailscaleCertificateError, "waiting before retry"):
                self.manager.ensure_certificate()
        self.assertEqual(
            sum(
                any(value.startswith("--cert-file=") for value in call)
                for call in self.fake.calls
            ),
            cert_calls_after_first,
        )

    def test_cross_process_contention_reuses_current_cert_and_preserves_other_stage(self):
        current = self.manager.ensure_certificate()
        other_stage = self.settings / "tailscale-certs" / "generations" / ".staging-other"
        other_stage.mkdir()
        sentinel = other_stage / "in-progress"
        sentinel.write_text("owned by another process", encoding="utf-8")

        class BusyLock:
            def __init__(self, _directory):
                pass

            def __enter__(self):
                raise RuntimeError("This installation's Agent Access controller is already running")

            def __exit__(self, *_args):
                return None

        with patch("services.agent_access.tailscale.ControllerLock", BusyLock):
            reused = self.manager.ensure_certificate()
        self.assertEqual(reused.generation, current.generation)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "owned by another process")


if __name__ == "__main__":
    unittest.main()
