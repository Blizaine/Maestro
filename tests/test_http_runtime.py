"""Real Windows listener regressions for a failed AcceptEx completion."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services import http_runtime


def windows_error(code):
    error = OSError("simulated incoming connection error")
    error.winerror = code
    return error


class InstallPlatformTests(unittest.TestCase):
    def test_non_windows_does_not_touch_loop(self):
        loop = mock.Mock()
        with mock.patch.object(http_runtime.sys, "platform", "linux"):
            self.assertFalse(http_runtime.install_windows_accept_retry(loop))
        self.assertEqual(loop.mock_calls, [])


@unittest.skipUnless(sys.platform == "win32", "Windows IOCP integration")
class WindowsAcceptTests(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(self.loop)
        self.errors = []
        self.loop.set_exception_handler(lambda loop, context: self.errors.append(context))
        self.servers = []
        self.accepted_sockets = []
        proactor = self.loop._proactor
        self.original_register = proactor._register
        original_socket = proactor._get_accept_socket

        def record_socket(family):
            conn = original_socket(family)
            self.accepted_sockets.append(conn)
            return conn

        proactor._get_accept_socket = record_socket

    def tearDown(self):
        async def shutdown():
            for server in self.servers:
                server.close()
                await server.wait_closed()
            for conn in self.accepted_sockets:
                conn.close()
            pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await asyncio.sleep(0)

        self.loop.run_until_complete(shutdown())
        self.loop.close()
        asyncio.set_event_loop(None)

    def inject_accept_errors(self, codes):
        codes = list(codes)
        self.injected = []

        def register(ov, obj, callback):
            if callback.__name__ == "finish_accept" and codes:
                code = codes.pop(0)

                def failed_accept(transferred, key, completed):
                    # Finish real kernel I/O before injecting the same Windows
                    # error observed in production, at the actual IOCP boundary.
                    completed.getresult()
                    self.injected.append(code)
                    raise windows_error(code)

                callback = failed_accept
            return self.original_register(ov, obj, callback)

        self.loop._proactor._register = register

    async def server(self):
        async def handle(reader, writer):
            try:
                await reader.readuntil(b"\r\n\r\n")
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        self.servers.append(server)
        return server, server.sockets[0].getsockname()[1]

    async def request(self, port):
        reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 2)
        try:
            writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            return await asyncio.wait_for(reader.read(), 2)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    async def aborted_request(self, port):
        try:
            response = await self.request(port)
        except (OSError, asyncio.TimeoutError):
            return
        self.assertEqual(response, b"")

    def test_stock_proactor_reproduces_closed_listener(self):
        self.inject_accept_errors([64])

        async def exercise():
            server, port = await self.server()
            await self.aborted_request(port)
            await asyncio.sleep(0.02)
            self.assertEqual(server.sockets[0].fileno(), -1)
            # Python 3.10's asyncio.TimeoutError is separate from OSError;
            # a dead listener can refuse or time out depending on Windows.
            with self.assertRaises((OSError, asyncio.TimeoutError)):
                await self.request(port)
            self.assertTrue(any(error.get("message") == "Accept failed on a socket"
                                for error in self.errors))

        self.loop.run_until_complete(exercise())
        self.assertEqual(self.injected, [64])

    def test_reset_keeps_same_listener_and_serves_later_requests(self):
        http_runtime.install_windows_accept_retry(self.loop)
        self.inject_accept_errors([64, 64])

        async def exercise():
            server, port = await self.server()
            for _ in range(2):
                await self.aborted_request(port)
                await asyncio.sleep(0.07)
            for _ in range(3):
                self.assertIn(b"200 OK", await self.request(port))
            self.assertEqual(server.sockets[0].getsockname()[1], port)
            self.assertTrue(all(conn.fileno() == -1 for conn in self.accepted_sockets[:2]))
            self.assertEqual(self.errors, [])

        self.loop.run_until_complete(exercise())
        self.assertEqual(self.injected, [64, 64])

    def test_unrelated_os_error_is_not_hidden_or_retried(self):
        http_runtime.install_windows_accept_retry(self.loop)
        self.inject_accept_errors([5])

        async def exercise():
            server, port = await self.server()
            await self.aborted_request(port)
            await asyncio.sleep(0.02)
            self.assertEqual(server.sockets[0].fileno(), -1)
            with self.assertRaises((OSError, asyncio.TimeoutError)):
                await self.request(port)
            self.assertTrue(any(getattr(error.get("exception"), "winerror", None) == 5
                                for error in self.errors))
            self.assertEqual(self.accepted_sockets[0].fileno(), -1)

        self.loop.run_until_complete(exercise())
        self.assertEqual(self.injected, [5])

    def test_server_shutdown_cancels_pending_accept_and_closes_connection(self):
        http_runtime.install_windows_accept_retry(self.loop)

        async def exercise():
            server, _ = await self.server()
            await asyncio.sleep(0)
            server.close()
            await server.wait_closed()
            await asyncio.sleep(0.02)
            self.assertTrue(self.accepted_sockets)
            self.assertTrue(all(conn.fileno() == -1 for conn in self.accepted_sockets))
            self.assertEqual(self.errors, [])

        self.loop.run_until_complete(exercise())

    def test_cancellation_during_retry_does_not_attempt_again(self):
        listener = SimpleNamespace(fileno=lambda: 123)
        proactor = SimpleNamespace(_loop=self.loop)

        async def exercise():
            task = asyncio.create_task(http_runtime._accept_with_retry(proactor, listener))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        with mock.patch.object(http_runtime, "_accept_once", new_callable=mock.AsyncMock,
                               side_effect=windows_error(64)) as accept:
            self.loop.run_until_complete(exercise())
            self.assertEqual(accept.await_count, 1)

    def test_closed_listener_is_not_retried(self):
        listener = SimpleNamespace(fileno=lambda: -1)
        proactor = SimpleNamespace(_loop=self.loop)
        with mock.patch.object(http_runtime, "_accept_once", new_callable=mock.AsyncMock,
                               side_effect=windows_error(64)) as accept:
            with self.assertRaises(OSError):
                self.loop.run_until_complete(http_runtime._accept_with_retry(proactor, listener))
            self.assertEqual(accept.await_count, 1)

    def test_install_is_idempotent_and_scoped_to_one_proactor(self):
        other = asyncio.ProactorEventLoop()
        try:
            other_accept = other._proactor.accept
            self.assertTrue(http_runtime.install_windows_accept_retry(self.loop))
            installed = self.loop._proactor.accept
            self.assertTrue(http_runtime.install_windows_accept_retry(self.loop))
            self.assertIs(self.loop._proactor.accept, installed)
            self.assertEqual(other._proactor.accept, other_accept)
        finally:
            other.close()

    def test_selector_loop_is_unchanged(self):
        selector = asyncio.SelectorEventLoop()
        try:
            self.assertFalse(http_runtime.install_windows_accept_retry(selector))
        finally:
            selector.close()


if __name__ == "__main__":
    unittest.main()
