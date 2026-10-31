"""TLS leaf provisioning, persistence, and explicit trust tests."""

from contextlib import redirect_stderr, redirect_stdout
import io
import ipaddress
from pathlib import Path
import socket
import ssl
import stat
import tempfile
import threading
import unittest

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

import sys

_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.agent_access.tls import (  # noqa: E402
    TLSAssetError,
    ensure_tls,
    hostname_for_installation_id,
    renew_tls,
)


_INSTALLATION_ID = "9be9a830-7c10-4f68-a50b-d16514f13730"


class TLSProvisioningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "tls"

    def test_hostname_is_stable_and_sans_cover_tls_and_current_lan_ips(self):
        material = ensure_tls(
            self.directory, _INSTALLATION_ID, ip_addresses=["10.20.30.40", "fd12::5"]
        )
        self.assertEqual(material.hostname, "maestro-9be9a8307c104f68.local")
        self.assertEqual(material.hostname, hostname_for_installation_id(_INSTALLATION_ID))
        certificate = x509.load_pem_x509_certificate(material.cert_path.read_bytes())
        san = certificate.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
        self.assertEqual(set(san.get_values_for_type(x509.DNSName)), {
            material.hostname, "localhost"
        })
        self.assertEqual(set(san.get_values_for_type(x509.IPAddress)), {
            ipaddress.ip_address("127.0.0.1"),
            ipaddress.ip_address("::1"),
            ipaddress.ip_address("10.20.30.40"),
            ipaddress.ip_address("fd12::5"),
        })
        constraints = certificate.extensions.get_extension_for_class(
            x509.BasicConstraints
        ).value
        usage = certificate.extensions.get_extension_for_class(x509.KeyUsage).value
        self.assertFalse(constraints.ca)
        self.assertFalse(usage.key_cert_sign)
        self.assertFalse(usage.crl_sign)
        self.assertTrue(usage.digital_signature)

    def test_key_certificate_and_fingerprint_are_reused_across_restart(self):
        first = ensure_tls(self.directory, _INSTALLATION_ID, ip_addresses=["10.0.0.7"])
        first_key = first.key_path.read_bytes()
        first_cert = first.cert_path.read_bytes()
        second = ensure_tls(self.directory, _INSTALLATION_ID, ip_addresses=["10.0.0.8"])
        self.assertEqual(second.key_path.read_bytes(), first_key)
        self.assertEqual(second.cert_path.read_bytes(), first_cert)
        self.assertEqual(second.fingerprint_sha256, first.fingerprint_sha256)
        self.assertEqual(second.ip_addresses, ("10.0.0.7", "127.0.0.1", "::1"))

    def test_dhcp_change_does_not_rotate_leaf_or_block_stable_dns(self):
        first = ensure_tls(self.directory, _INSTALLATION_ID, ip_addresses=["192.168.4.20"])
        second = ensure_tls(self.directory, _INSTALLATION_ID, ip_addresses=["192.168.4.21"])
        self.assertEqual(first.fingerprint_sha256, second.fingerprint_sha256)
        self.assertIn("192.168.4.20", second.ip_addresses)
        self.assertNotIn("192.168.4.21", second.ip_addresses)

    def test_corrupt_assets_fail_clearly_without_silent_replacement(self):
        material = ensure_tls(self.directory, _INSTALLATION_ID)
        original_key = material.key_path.read_bytes()
        material.cert_path.write_bytes(b"not a certificate")
        with self.assertRaisesRegex(TLSAssetError, "corrupt or unsupported"):
            ensure_tls(self.directory, _INSTALLATION_ID)
        self.assertEqual(material.key_path.read_bytes(), original_key)
        self.assertEqual(material.cert_path.read_bytes(), b"not a certificate")

    def test_mismatched_certificate_and_private_key_fail_clearly(self):
        material = ensure_tls(self.directory, _INSTALLATION_ID)
        unrelated = ec.generate_private_key(ec.SECP256R1())
        material.key_path.write_bytes(unrelated.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ))
        with self.assertRaisesRegex(TLSAssetError, "do not match"):
            ensure_tls(self.directory, _INSTALLATION_ID)

    def test_private_key_is_never_printed_and_permissions_are_restricted(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            material = ensure_tls(self.directory, _INSTALLATION_ID)
        combined_output = stdout.getvalue() + stderr.getvalue()
        self.assertNotIn("PRIVATE KEY", combined_output)
        self.assertNotIn("PRIVATE KEY", repr(material))
        if os_name_is_posix():
            self.assertEqual(stat.S_IMODE(self.directory.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(material.key_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(material.cert_path.stat().st_mode), 0o600)

    def test_self_signed_leaf_verifies_when_explicitly_trusted(self):
        material = ensure_tls(self.directory, _INSTALLATION_ID)
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.load_cert_chain(
            certfile=str(material.cert_path), keyfile=str(material.key_path)
        )
        client_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        client_context.check_hostname = True
        client_context.verify_mode = ssl.CERT_REQUIRED
        client_context.load_verify_locations(cafile=str(material.cert_path))
        if hasattr(ssl, "VERIFY_X509_PARTIAL_CHAIN"):
            client_context.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        server_error = []

        def serve_once():
            try:
                raw, _ = listener.accept()
                with raw, server_context.wrap_socket(raw, server_side=True) as secured:
                    secured.recv(1)
                    secured.sendall(b"ok")
            except Exception as exc:  # surfaced in the test thread below
                server_error.append(exc)
            finally:
                listener.close()

        worker = threading.Thread(target=serve_once, daemon=True)
        worker.start()
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
                with client_context.wrap_socket(raw, server_hostname=material.hostname) as secured:
                    secured.sendall(b"x")
                    self.assertEqual(secured.recv(2), b"ok")
        finally:
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(server_error, [])

    def test_explicit_renewal_changes_fingerprint_and_requires_retrust(self):
        first = ensure_tls(self.directory, _INSTALLATION_ID)
        renewed = renew_tls(self.directory, _INSTALLATION_ID, ip_addresses=["10.0.0.8"])
        self.assertTrue(renewed.requires_retrust)
        self.assertNotEqual(renewed.fingerprint_sha256, first.fingerprint_sha256)
        self.assertIn("10.0.0.8", renewed.ip_addresses)

    def test_partial_assets_and_invalid_address_fail_closed(self):
        self.directory.mkdir()
        (self.directory / "server-key.pem").write_text("partial", encoding="ascii")
        with self.assertRaisesRegex(TLSAssetError, "both be present"):
            ensure_tls(self.directory, _INSTALLATION_ID)
        with tempfile.TemporaryDirectory() as another:
            with self.assertRaisesRegex(ValueError, "Invalid IP address"):
                ensure_tls(Path(another), _INSTALLATION_ID, ip_addresses=["not-an-ip"])


def os_name_is_posix() -> bool:
    import os
    return os.name == "posix"


if __name__ == "__main__":
    unittest.main()
