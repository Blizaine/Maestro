"""CPU-only tests for stable ownership of the HTTP bind socket."""

from __future__ import annotations

import socket
import sys
from pathlib import Path
import unittest
from unittest import mock

_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.server_socket import reserve_server_socket  # noqa: E402


class _FakeSocket:
    def __init__(self, bind_error=None):
        self.bind_error = bind_error
        self.bound_address = None
        self.closed = False
        self.options = []

    def setsockopt(self, level, option, value):
        self.options.append((level, option, value))

    def bind(self, address):
        self.bound_address = address
        if self.bind_error is not None:
            raise self.bind_error

    def close(self):
        self.closed = True


class TestReserveServerSocket(unittest.TestCase):
    def test_port_zero_uses_os_assignment_and_retains_exclusive_bind(self):
        reserved = reserve_server_socket("127.0.0.1", 0)
        contender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            host, port = reserved.getsockname()[:2]
            self.assertEqual(host, "127.0.0.1")
            self.assertGreater(port, 0)
            with self.assertRaises(OSError):
                contender.bind((host, port))
        finally:
            contender.close()
            reserved.close()

    def test_occupied_preferred_port_falls_forward_and_keeps_fallback_bound(self):
        occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupied.bind(("127.0.0.1", 0))
        preferred = occupied.getsockname()[1]
        reserved = None
        try:
            reserved = reserve_server_socket("127.0.0.1", preferred, span=8)
            self.assertIsNotNone(reserved)
            self.assertNotEqual(reserved.getsockname()[1], preferred)
            self.assertEqual(reserved.getsockname()[0], "127.0.0.1")
        finally:
            if reserved is not None:
                reserved.close()
            occupied.close()

    def test_all_busy_candidates_return_none_and_close_every_failed_socket(self):
        created = []

        def make_socket(*_args):
            value = _FakeSocket(OSError("busy"))
            created.append(value)
            return value

        def resolve(host, port, type=0):
            self.assertEqual(host, "127.0.0.1")
            self.assertEqual(type, socket.SOCK_STREAM)
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (host, port))]

        with (
            mock.patch("services.server_socket.socket.socket", side_effect=make_socket),
            mock.patch("services.server_socket.socket.getaddrinfo", side_effect=resolve),
        ):
            result = reserve_server_socket("127.0.0.1", 41000, span=2)

        self.assertIsNone(result)
        self.assertEqual(len(created), 3)
        self.assertTrue(all(value.closed for value in created))
        self.assertTrue(all(not any(option[1] == socket.SO_REUSEADDR for option in value.options) for value in created))

    def test_actual_bind_host_is_used_in_resolution_and_bind(self):
        fake = _FakeSocket()
        address = ("0.0.0.0", 42000)
        with (
            mock.patch("services.server_socket.socket.getaddrinfo", return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", address),
            ]) as resolve,
            mock.patch("services.server_socket.socket.socket", return_value=fake),
        ):
            reserved = reserve_server_socket("0.0.0.0", 42000, span=0)

        self.assertIs(reserved, fake)
        self.assertEqual(resolve.call_args.args[:2], ("0.0.0.0", 42000))
        self.assertEqual(fake.bound_address, address)
        self.assertFalse(fake.closed)
        fake.close()

    def test_windows_requests_exclusive_address_use_without_reuse(self):
        fake = _FakeSocket()
        exclusive_option = getattr(socket, "SO_EXCLUSIVEADDRUSE", 0x100)
        address = ("127.0.0.1", 43000)
        with (
            mock.patch("services.server_socket.sys.platform", "win32"),
            mock.patch.object(socket, "SO_EXCLUSIVEADDRUSE", exclusive_option, create=True),
            mock.patch("services.server_socket.socket.getaddrinfo", return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", address),
            ]),
            mock.patch("services.server_socket.socket.socket", return_value=fake),
        ):
            reserved = reserve_server_socket("127.0.0.1", 43000, span=0)

        self.assertIs(reserved, fake)
        self.assertIn((socket.SOL_SOCKET, exclusive_option, 1), fake.options)
        self.assertFalse(any(option[1] == socket.SO_REUSEADDR for option in fake.options))
        fake.close()

    def test_port_and_span_ranges_are_validated(self):
        for preferred, span in ((-1, 0), (65536, 0), (41000, -1), (41000, 65536), (65535, 1)):
            with self.subTest(preferred=preferred, span=span):
                with self.assertRaises(ValueError):
                    reserve_server_socket("127.0.0.1", preferred, span)

    def test_invalid_bind_host_is_rejected(self):
        for host in ("", " 127.0.0.1", "127.0.0.1 ", None):
            with self.subTest(host=host):
                with self.assertRaises(ValueError):
                    reserve_server_socket(host, 41000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
