"""Opt-in mDNS discovery for nearby Maestro installations.

Advertising and browsing are intentionally started only by ``start()``. A
discovered record is descriptive metadata only: the advertised fingerprint is
not treated as authentication, and no peer is contacted automatically.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ipaddress
from itertools import islice
import re
import threading
import time
from typing import Any, Callable
import uuid

from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf

from .tls import hostname_for_installation_id


SERVICE_TYPE = "_maestro._tcp.local."
_FINGERPRINT_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _valid_text(value: Any, name: str, max_bytes: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    value = value.strip()
    if not value or len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{name} is empty or too long")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} contains control characters")
    return value


def _validated_id(value: Any) -> tuple[str, str]:
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError("installation_id must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError("installation_id must be a UUID string") from exc
    return str(parsed), parsed.hex[:16]


def _validated_workstation_id(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError("workstation_id must be a UUID string")
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError) as exc:
        raise ValueError("workstation_id must be a UUID string") from exc


def _advertised_text(value: str, max_bytes: int) -> bytes:
    """Return a valid UTF-8 prefix that fits one bounded TXT field."""
    return value.encode("utf-8")[:max_bytes].decode("utf-8", "ignore").encode("utf-8")


def _fingerprint(value: Any) -> str:
    if not isinstance(value, str) or not _FINGERPRINT_RE.fullmatch(value):
        raise ValueError("certificate_fingerprint must be a 64-character SHA-256 hex digest")
    return value.lower()


def _port(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError("port must be an integer between 1 and 65535")
    return value


def _is_private_lan_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_unspecified
        and not address.is_reserved
    )


def _normalize_addresses(addresses: Sequence[str] | None) -> tuple[str, ...]:
    if addresses is None:
        return ()
    if isinstance(addresses, (str, bytes)):
        raise ValueError("addresses must be a list of IP address strings")
    if len(addresses) > 16:
        raise ValueError("At most 16 local discovery addresses may be advertised")
    result: set[str] = set()
    for value in addresses:
        if not isinstance(value, str):
            raise ValueError("Each address must be a string")
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise ValueError(f"Invalid discovery address: {value!r}") from exc
        if _is_private_lan_address(address):
            result.add(address.compressed)
    return tuple(sorted(result))


def _service_name(compact_id: str) -> str:
    return f"maestro-{compact_id}.{SERVICE_TYPE}"


def _txt_value(properties: Mapping[Any, Any], name: str, max_bytes: int) -> str | None:
    key_variants = (name.encode("ascii"), name)
    raw = None
    for key in key_variants:
        if key in properties:
            raw = properties[key]
            break
    if not isinstance(raw, (bytes, str)):
        return None
    try:
        value = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    except UnicodeDecodeError:
        return None
    if len(value.encode("utf-8")) > max_bytes:
        return None
    return value


def _private_ipv4(values: Any) -> list[str]:
    if values is None:
        return []
    result: set[str] = set()
    try:
        items = islice(iter(values), 32)
    except TypeError:
        return []
    for item in items:
        try:
            if isinstance(item, str):
                address = ipaddress.ip_address(item)
            elif isinstance(item, (bytes, bytearray, memoryview)):
                address = ipaddress.ip_address(bytes(item))
            else:
                continue
        except (TypeError, ValueError):
            continue
        if isinstance(address, ipaddress.IPv4Address) and _is_private_lan_address(address):
            result.add(address.compressed)
    return sorted(result)


class _DiscoveryListener:
    def __init__(self, owner: "DiscoveryService"):
        self.owner = owner

    def add_service(self, zeroconf: Any, service_type: str, name: str) -> None:
        self.owner._resolve_service(zeroconf, service_type, name)

    def update_service(self, zeroconf: Any, service_type: str, name: str) -> None:
        self.owner._resolve_service(zeroconf, service_type, name)

    def remove_service(self, zeroconf: Any, service_type: str, name: str) -> None:
        self.owner._remove_service(name)


class DiscoveryService:
    """Opt-in advertisement and bounded cache of nearby Maestro peers."""

    def __init__(
        self,
        installation_id: str,
        hostname: str,
        port: int,
        display_name: str,
        version: str,
        certificate_fingerprint: str,
        addresses: Sequence[str] | None = None,
        *,
        workstation_id: str | None = None,
        enabled: bool = False,
        clock: Callable[[], float] = time.monotonic,
        peer_ttl: float = 120.0,
        max_peers: int = 64,
        resolve_timeout_ms: int = 1200,
    ):
        normalized_id, compact_id = _validated_id(installation_id)
        expected_hostname = hostname_for_installation_id(normalized_id)
        normalized_hostname = _valid_text(hostname, "hostname", 253).rstrip(".").lower()
        if normalized_hostname != expected_hostname:
            raise ValueError("hostname must match installation_id")
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        if not isinstance(peer_ttl, (int, float)) or isinstance(peer_ttl, bool) or peer_ttl <= 0:
            raise ValueError("peer_ttl must be positive")
        if isinstance(max_peers, bool) or not isinstance(max_peers, int) or not 1 <= max_peers <= 256:
            raise ValueError("max_peers must be between 1 and 256")
        if (
            isinstance(resolve_timeout_ms, bool)
            or not isinstance(resolve_timeout_ms, int)
            or not 100 <= resolve_timeout_ms <= 5000
        ):
            raise ValueError("resolve_timeout_ms must be between 100 and 5000")

        self.installation_id = normalized_id
        self.hostname = expected_hostname
        self.port = _port(port)
        self.display_name = _valid_text(display_name, "display_name", 256)
        if len(self.display_name) > 64:
            raise ValueError("display_name must contain at most 64 characters")
        self.version = _valid_text(version, "version", 32)
        self.certificate_fingerprint = _fingerprint(certificate_fingerprint)
        self.workstation_id = (
            None if workstation_id is None else _validated_workstation_id(workstation_id)
        )
        self.addresses = _normalize_addresses(addresses)
        self.enabled = enabled
        self._clock = clock
        self._peer_ttl = float(peer_ttl)
        self._max_peers = max_peers
        self._resolve_timeout_ms = resolve_timeout_ms
        self._lock = threading.RLock()
        self._zeroconf: Zeroconf | None = None
        self._browser: ServiceBrowser | None = None
        self._service_info: ServiceInfo | None = None
        self._listener = _DiscoveryListener(self)
        self._peers: dict[str, tuple[dict[str, Any], float, str]] = {}

    def start(self) -> None:
        """Begin mDNS advertisement and browsing if enabled."""
        with self._lock:
            if not self.enabled or self._zeroconf is not None:
                return
            zc: Zeroconf | None = None
            try:
                zc = Zeroconf()
                addresses = [ipaddress.ip_address(value).packed for value in self.addresses]
                properties = {
                    b"installation_id": self.installation_id.encode("ascii"),
                    b"hostname": self.hostname.encode("ascii"),
                    b"display_name": _advertised_text(self.display_name, 64),
                    b"version": self.version.encode("utf-8"),
                    b"fingerprint_sha256": self.certificate_fingerprint.encode("ascii"),
                    b"pairing_required": b"true",
                }
                if self.workstation_id is not None:
                    properties[b"workstation_id"] = self.workstation_id.encode("ascii")
                info = ServiceInfo(
                    SERVICE_TYPE,
                    _service_name(self.installation_id.replace("-", "")[:16]),
                    addresses=addresses or None,
                    port=self.port,
                    properties=properties,
                    server=f"{self.hostname}.",
                )
                zc.register_service(info, allow_name_change=False)
                self._zeroconf = zc
                self._service_info = info
                self._browser = ServiceBrowser(zc, SERVICE_TYPE, self._listener)
            except Exception:
                self._zeroconf = None
                self._service_info = None
                self._browser = None
                if zc is not None:
                    try:
                        zc.close()
                    except Exception:
                        pass
                raise

    def stop(self) -> None:
        """Stop browsing and unregister the service. Safe to call repeatedly."""
        with self._lock:
            browser, zc, info = self._browser, self._zeroconf, self._service_info
            self._browser = None
            self._zeroconf = None
            self._service_info = None
            self._peers.clear()
        if browser is not None:
            try:
                browser.cancel()
            except Exception:
                pass
        if zc is not None:
            if info is not None:
                try:
                    zc.unregister_service(info)
                except Exception:
                    pass
            try:
                zc.close()
            except Exception:
                pass

    def peers(self) -> list[dict[str, Any]]:
        """Return fresh peer metadata without making any outbound requests."""
        now = self._clock()
        with self._lock:
            self._expire_peers(now)
            return [
                {
                    **record,
                    "ip_addresses": list(record["ip_addresses"]),
                    "ip_urls": list(record["ip_urls"]),
                }
                for record, _, _ in sorted(
                self._peers.values(), key=lambda item: (item[0]["display_name"].casefold(), item[0]["installation_id"])
                )
            ]

    def _expire_peers(self, now: float) -> None:
        stale = [
            installation_id
            for installation_id, (_, seen_at, _) in self._peers.items()
            if now - seen_at > self._peer_ttl
        ]
        for installation_id in stale:
            self._peers.pop(installation_id, None)

    def _remove_service(self, service_name: str) -> None:
        with self._lock:
            for installation_id, (_, _, cached_name) in list(self._peers.items()):
                if cached_name == service_name:
                    self._peers.pop(installation_id, None)

    def _resolve_service(self, zeroconf: Any, service_type: str, service_name: str) -> None:
        if service_type.rstrip(".").lower() != SERVICE_TYPE.rstrip(".").lower():
            return
        try:
            info = zeroconf.get_service_info(
                service_type, service_name, timeout=self._resolve_timeout_ms
            )
        except Exception:
            return
        if info is None:
            return
        try:
            peer = self._peer_record(service_name, info)
        except (ValueError, TypeError, UnicodeError, AttributeError):
            return
        if peer is None:
            return
        installation_id = peer["installation_id"]
        now = self._clock()
        with self._lock:
            if self._zeroconf is None:
                return
            self._expire_peers(now)
            if installation_id not in self._peers and len(self._peers) >= self._max_peers:
                oldest = min(self._peers, key=lambda key: self._peers[key][1])
                self._peers.pop(oldest, None)
            self._peers[installation_id] = (peer, now, service_name)

    def _peer_record(self, service_name: str, info: Any) -> dict[str, Any] | None:
        properties = info.properties
        if not isinstance(properties, Mapping):
            return None
        raw_id = _txt_value(properties, "installation_id", 64)
        raw_hostname = _txt_value(properties, "hostname", 253)
        raw_name = _txt_value(properties, "display_name", 64)
        raw_version = _txt_value(properties, "version", 32)
        raw_fingerprint = _txt_value(properties, "fingerprint_sha256", 64)
        workstation_key_present = b"workstation_id" in properties or "workstation_id" in properties
        raw_workstation_id = _txt_value(properties, "workstation_id", 64)
        pairing_required = _txt_value(properties, "pairing_required", 5)
        if None in (raw_id, raw_hostname, raw_name, raw_version, raw_fingerprint):
            return None
        if workstation_key_present and raw_workstation_id is None:
            return None
        if pairing_required != "true":
            return None
        try:
            installation_id, compact_id = _validated_id(raw_id)
            hostname = _valid_text(raw_hostname, "hostname", 253).rstrip(".").lower()
            display_name = _valid_text(raw_name, "display_name", 64)
            version = _valid_text(raw_version, "version", 32)
            fingerprint = _fingerprint(raw_fingerprint)
            workstation_id = (
                None if raw_workstation_id is None
                else _validated_workstation_id(raw_workstation_id)
            )
            port = _port(info.port)
        except ValueError:
            return None
        if installation_id == self.installation_id:
            return None
        if hostname != hostname_for_installation_id(installation_id):
            return None
        expected_name = _service_name(compact_id).lower()
        if not isinstance(service_name, str) or service_name.lower() != expected_name:
            return None

        # TXT data can contain arbitrary URLs or addresses. Ignore both and
        # derive endpoints only from validated resolved SRV addresses/port.
        ip_addresses = _private_ipv4(info.addresses)
        ip_urls = [f"https://{address}:{port}/mcp" for address in ip_addresses]
        return {
            "installation_id": installation_id,
            "workstation_id": workstation_id,
            "hostname": hostname,
            "display_name": display_name,
            "version": version,
            "fingerprint_sha256": fingerprint,
            "port": port,
            "mcp_url": f"https://{hostname}:{port}/mcp",
            "ip_addresses": ip_addresses,
            "ip_urls": ip_urls,
            "pairing_required": True,
        }


__all__ = ["DiscoveryService", "SERVICE_TYPE"]
