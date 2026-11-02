"""mDNS discovery validation and lifecycle tests with no real broadcasts."""

from pathlib import Path
import importlib.util
import ipaddress
import sys
import unittest
from unittest.mock import Mock, patch

if importlib.util.find_spec("zeroconf") is None:
    raise unittest.SkipTest("mDNS tests run in the separate Agent Access environment")

_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.agent_access.discovery import (  # noqa: E402
    SERVICE_TYPE,
    DiscoveryService,
)
from services.agent_access.tls import hostname_for_installation_id  # noqa: E402


_LOCAL_ID = "9be9a830-7c10-4f68-a50b-d16514f13730"
_PEER_ID = "6e8440b8-89cf-4412-859e-91a478e26958"
_LOCAL_WORKSTATION_ID = "80fd3c87-67de-55c8-bc21-df85a60009e1"
_PEER_WORKSTATION_ID = "23dd53fa-9d8c-5cd1-bef0-18a67bf0c356"
_FINGERPRINT = "a" * 64


def _service(**kwargs):
    defaults = {
        "installation_id": _LOCAL_ID,
        "hostname": hostname_for_installation_id(_LOCAL_ID),
        "port": 8765,
        "display_name": "Maestro Dev",
        "version": "2.4.0",
        "certificate_fingerprint": _FINGERPRINT,
        "addresses": ["10.1.2.3", "fd12::3"],
        "workstation_id": _LOCAL_WORKSTATION_ID,
        "enabled": True,
    }
    defaults.update(kwargs)
    return DiscoveryService(**defaults)


def _properties(
    installation_id=_PEER_ID,
    *,
    fingerprint=_FINGERPRINT,
    hostname=None,
    display_name="Maestro Main",
    version="2.4.0",
    pairing_required="true",
    workstation_id=None,
    extra=None,
):
    if hostname is None:
        hostname = hostname_for_installation_id(installation_id)
    values = {
        b"installation_id": installation_id.encode(),
        b"hostname": hostname.encode(),
        b"display_name": display_name.encode(),
        b"version": version.encode(),
        b"fingerprint_sha256": fingerprint.encode(),
        b"pairing_required": pairing_required.encode(),
    }
    if workstation_id is not None:
        values[b"workstation_id"] = workstation_id.encode()
    if extra:
        values.update(extra)
    return values


def _info(properties=None, *, port=9000, addresses=None):
    return type("FakeInfo", (), {
        "properties": _properties() if properties is None else properties,
        "port": port,
        "addresses": [
            ipaddress.ip_address("10.2.3.4").packed,
            ipaddress.ip_address("192.168.10.8").packed,
            ipaddress.ip_address("8.8.8.8").packed,
            ipaddress.ip_address("fd12::8").packed,
        ] if addresses is None else addresses,
    })()


class DiscoveryLifecycleTests(unittest.TestCase):
    @patch("services.agent_access.discovery.ServiceBrowser")
    @patch("services.agent_access.discovery.ServiceInfo")
    @patch("services.agent_access.discovery.Zeroconf")
    def test_constructor_and_disabled_start_create_no_sockets_or_browser(
        self, zeroconf_cls, service_info_cls, browser_cls
    ):
        disabled = _service(enabled=False)
        zeroconf_cls.assert_not_called()
        service_info_cls.assert_not_called()
        browser_cls.assert_not_called()
        disabled.start()
        disabled.stop()
        zeroconf_cls.assert_not_called()
        service_info_cls.assert_not_called()
        browser_cls.assert_not_called()

    @patch("services.agent_access.discovery.ServiceBrowser")
    @patch("services.agent_access.discovery.ServiceInfo")
    @patch("services.agent_access.discovery.Zeroconf")
    def test_enabled_start_advertises_only_small_pairing_metadata(
        self, zeroconf_cls, service_info_cls, browser_cls
    ):
        zeroconf = Mock()
        zeroconf_cls.return_value = zeroconf
        service_info_cls.return_value = Mock()
        browser = Mock()
        browser_cls.return_value = browser
        service = _service()
        self.assertEqual(zeroconf_cls.call_count, 0)

        service.start()

        zeroconf_cls.assert_called_once_with()
        service_info_cls.assert_called_once()
        args, kwargs = service_info_cls.call_args
        self.assertEqual(args[:2], (
            SERVICE_TYPE,
            "maestro-9be9a8307c104f68._maestro._tcp.local.",
        ))
        self.assertEqual(kwargs["port"], 8765)
        self.assertEqual(kwargs["server"], "maestro-9be9a8307c104f68.local.")
        properties = kwargs["properties"]
        self.assertEqual(set(properties), {
            b"installation_id", b"hostname", b"display_name", b"version",
            b"fingerprint_sha256", b"pairing_required", b"workstation_id",
        })
        self.assertEqual(properties[b"pairing_required"], b"true")
        self.assertEqual(properties[b"workstation_id"], _LOCAL_WORKSTATION_ID.encode())
        self.assertNotIn(b"url", properties)
        browser_cls.assert_called_once_with(zeroconf, SERVICE_TYPE, service._listener)

        service.stop()
        browser.cancel.assert_called_once_with()
        zeroconf.unregister_service.assert_called_once()
        zeroconf.close.assert_called_once_with()

    @patch("services.agent_access.discovery.ServiceBrowser")
    @patch("services.agent_access.discovery.ServiceInfo")
    @patch("services.agent_access.discovery.Zeroconf")
    def test_long_utf8_label_is_retained_and_advertisement_stays_within_64_bytes(
        self, zeroconf_cls, service_info_cls, browser_cls
    ):
        zeroconf = Mock()
        zeroconf_cls.return_value = zeroconf
        service_info_cls.return_value = Mock()
        full_label = "🎨" * 64
        service = _service(display_name=full_label)

        service.start()

        self.assertEqual(service.display_name, full_label)
        properties = service_info_cls.call_args.kwargs["properties"]
        advertised = properties[b"display_name"]
        self.assertEqual(len(advertised), 64)
        self.assertEqual(advertised.decode("utf-8"), "🎨" * 16)
        service.stop()


class DiscoveryPeerTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.service = _service(clock=lambda: self.now, peer_ttl=60, max_peers=8)
        self.zeroconf = Mock()
        self.zeroconf_cls_patcher = patch(
            "services.agent_access.discovery.Zeroconf", return_value=self.zeroconf
        )
        self.info_cls_patcher = patch("services.agent_access.discovery.ServiceInfo", return_value=Mock())
        self.browser_cls_patcher = patch("services.agent_access.discovery.ServiceBrowser")
        self.zeroconf_cls_patcher.start()
        self.info_cls_patcher.start()
        self.browser_mock = self.browser_cls_patcher.start()
        self.addCleanup(self.zeroconf_cls_patcher.stop)
        self.addCleanup(self.info_cls_patcher.stop)
        self.addCleanup(self.browser_cls_patcher.stop)
        self.service.start()

    def test_peer_urls_come_only_from_private_resolved_ipv4_and_srv_port(self):
        # The object passed to ServiceBrowser is the third positional argument.
        listener = self._captured_listener()
        properties = _properties(extra={
            b"url": b"https://attacker.example/submit",
            b"addresses": b"203.0.113.40,198.51.100.2",
        }, workstation_id=_PEER_WORKSTATION_ID)
        info = _info(properties)
        self.zeroconf.get_service_info.return_value = info
        service_name = "maestro-6e8440b889cf4412._maestro._tcp.local."
        listener.add_service(self.zeroconf, SERVICE_TYPE, service_name)

        peers = self.service.peers()
        self.assertEqual(len(peers), 1)
        peer = peers[0]
        self.assertEqual(peer["installation_id"], _PEER_ID)
        self.assertEqual(peer["workstation_id"], _PEER_WORKSTATION_ID)
        self.assertEqual(peer["port"], 9000)
        self.assertEqual(peer["mcp_url"],
                         "https://maestro-6e8440b889cf4412.local:9000/mcp")
        self.assertEqual(peer["ip_addresses"], ["10.2.3.4", "192.168.10.8"])
        self.assertEqual(peer["ip_urls"], [
            "https://10.2.3.4:9000/mcp", "https://192.168.10.8:9000/mcp"
        ])
        self.assertTrue(peer["pairing_required"])
        self.assertNotIn("url", peer)

        peer["ip_urls"].clear()
        self.assertEqual(len(self.service.peers()[0]["ip_urls"]), 2)

    def test_legacy_peer_without_workstation_id_is_accepted_as_unknown(self):
        listener = self._captured_listener()
        self.zeroconf.get_service_info.return_value = _info(_properties())
        listener.add_service(
            self.zeroconf, SERVICE_TYPE,
            "maestro-6e8440b889cf4412._maestro._tcp.local.",
        )
        self.assertEqual(self.service.peers()[0]["workstation_id"], None)

    def test_main_and_dev_installations_can_share_one_workstation_identity(self):
        listener = self._captured_listener()
        dev_id = "4e08c31c-e967-4a6e-b93b-12ed4b981f17"
        self.zeroconf.get_service_info.side_effect = [
            _info(_properties(workstation_id=_PEER_WORKSTATION_ID)),
            _info(_properties(
                installation_id=dev_id,
                workstation_id=_PEER_WORKSTATION_ID,
            )),
        ]
        listener.add_service(
            self.zeroconf, SERVICE_TYPE,
            "maestro-6e8440b889cf4412._maestro._tcp.local.",
        )
        listener.add_service(
            self.zeroconf, SERVICE_TYPE,
            "maestro-4e08c31ce9674a6e._maestro._tcp.local.",
        )
        peers = self.service.peers()
        self.assertEqual(len(peers), 2)
        self.assertEqual({peer["installation_id"] for peer in peers}, {_PEER_ID, dev_id})
        self.assertEqual({peer["workstation_id"] for peer in peers}, {_PEER_WORKSTATION_ID})

    def test_self_malformed_and_unpaired_advertisements_are_skipped(self):
        listener = self._captured_listener()
        service_name = "maestro-6e8440b889cf4412._maestro._tcp.local."
        malformed = [
            _info(_properties(fingerprint="not-a-fingerprint")),
            _info(_properties(hostname="other.local")),
            _info(_properties(pairing_required="false")),
            _info(_properties(workstation_id="not-a-uuid")),
            _info(_properties(installation_id=_LOCAL_ID)),
            _info(_properties(installation_id="not-a-uuid", hostname="invalid.local")),
        ]
        for info in malformed:
            self.zeroconf.get_service_info.return_value = info
            listener.add_service(self.zeroconf, SERVICE_TYPE, service_name)
        self.assertEqual(self.service.peers(), [])

    def test_peer_cache_expires_and_remove_event_clears_immediately(self):
        listener = self._captured_listener()
        service_name = "maestro-6e8440b889cf4412._maestro._tcp.local."
        self.zeroconf.get_service_info.return_value = _info()
        listener.add_service(self.zeroconf, SERVICE_TYPE, service_name)
        self.assertEqual(len(self.service.peers()), 1)
        listener.remove_service(self.zeroconf, SERVICE_TYPE, service_name)
        self.assertEqual(self.service.peers(), [])

        listener.add_service(self.zeroconf, SERVICE_TYPE, service_name)
        self.now += 61
        self.assertEqual(self.service.peers(), [])

    def test_peer_count_is_bounded_and_oldest_peer_is_evicted(self):
        small = _service(clock=lambda: self.now, max_peers=1)
        small.start()
        # Re-read browser call args after starting the second service.
        listener = self._captured_listener()
        first_id = _PEER_ID
        second_id = "4e08c31c-e967-4a6e-b93b-12ed4b981f17"
        self.zeroconf.get_service_info.side_effect = [_info(_properties(first_id)), _info(_properties(second_id))]
        listener.add_service(
            self.zeroconf, SERVICE_TYPE,
            "maestro-6e8440b889cf4412._maestro._tcp.local.",
        )
        self.now += 1
        listener.add_service(
            self.zeroconf, SERVICE_TYPE,
            "maestro-4e08c31ce9674a6e._maestro._tcp.local.",
        )
        self.assertEqual([peer["installation_id"] for peer in small.peers()], [second_id])
        small.stop()

    def _captured_listener(self):
        return self.browser_mock.call_args.args[2]


class DiscoveryValidationTests(unittest.TestCase):
    def test_identity_ports_fingerprint_and_text_are_validated(self):
        cases = [
            {"installation_id": "bad"},
            {"hostname": "other.local"},
            {"port": 65536},
            {"port": True},
            {"certificate_fingerprint": "abc"},
            {"display_name": "x" * 65},
            {"workstation_id": "not-a-uuid"},
            {"version": "v" * 33},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    _service(**overrides)
        self.assertEqual(_service(addresses=["8.8.8.8"]).addresses, ())


if __name__ == "__main__":
    unittest.main()
