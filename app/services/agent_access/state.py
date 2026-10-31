"""Private control-plane identity, backend readiness and bounded audit records."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import platform
import socket
import sqlite3
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from .auth import AuthStore


def process_alive(pid: int) -> bool:
    """Read-only PID probe. Unknown/access-denied is treated as still alive."""
    if type(pid) is not int or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER: no PID
        try:
            exit_code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


class ControllerLock:
    """An OS-held lock; a crash releases it without deleting a stale PID file."""
    def __init__(self, directory):
        self.path = Path(directory) / "controller.lock"
        self.handle = None

    def __enter__(self):
        handle = self.path.open("a+b")
        try:
            if self.path.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise RuntimeError("This installation's Agent Access controller is already running") from error
        self.handle = handle
        return self

    def __exit__(self, *args):
        self.handle.close()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".agent-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Agent Access metadata is too large")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Agent Access metadata must be an object")
    return value


def machine_id() -> str:
    """Return a namespaced identifier, never the platform's raw machine ID."""
    identity = ""
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
                identity = str(winreg.QueryValueEx(key, "MachineGuid")[0])
        except OSError:
            pass
    else:
        for name in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                identity = Path(name).read_text(encoding="ascii").strip()
            except OSError:
                continue
            if identity:
                break
    identity = identity or f"{platform.node()}:{uuid.getnode()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "maestro-workstation:" + identity))


def private_lan_ipv4_addresses() -> tuple[str, ...]:
    """Resolve this computer's hostname locally and keep private IPv4 results.

    This does not probe the network. The result is used only for TLS SANs and
    mDNS endpoint records; callers still filter aliases through the active
    certificate before publishing direct-IP URLs.
    """
    try:
        records = socket.getaddrinfo(
            socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM,
        )
    except OSError:
        return ()
    addresses: set[str] = set()
    for record in records:
        try:
            address = ipaddress.ip_address(record[4][0])
        except (IndexError, TypeError, ValueError):
            continue
        if (
            isinstance(address, ipaddress.IPv4Address)
            and address.is_private
            and not address.is_loopback
            and not address.is_link_local
            and not address.is_multicast
            and not address.is_unspecified
            and not address.is_reserved
        ):
            addresses.add(address.compressed)
    return tuple(sorted(addresses))


class RuntimeState:
    """A stable endpoint configuration; app startup never starts network listeners."""

    def __init__(self, directory: Path, *, app_root: Path | None = None, port: int | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            self.directory.chmod(0o700)
        self.auth = AuthStore(self.directory)
        self.installation_id = str(self.auth.status()["installation_id"])
        self.hostname = "maestro-" + self.installation_id.replace("-", "")[:16] + ".local"
        self.app_root = Path(app_root).resolve() if app_root else None
        self._certificate_ip_addresses: tuple[str, ...] = ()
        self._lock = threading.RLock()
        self._config_path = self.directory / "runtime.json"
        self._initialize(port)
        self.auth.set_resource(self.mcp_url)
        self._security_required = bool(self.auth.status()["owner_configured"])
        if self._security_required:
            self.mark_security_configured()
        self._audit_db = self.directory / "audit.sqlite3"
        with sqlite3.connect(self._audit_db) as db:
            db.execute("CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, time REAL, actor TEXT, action TEXT, status TEXT)")
        os.chmod(self._audit_db, 0o600)

    def _initialize(self, requested_port: int | None) -> None:
        # The backend and controller can start together in separate processes.
        # SQLite releases this initialization lock even after an abrupt exit.
        with self._lock, sqlite3.connect(self.directory / "runtime-lock.sqlite3", timeout=15) as guard:
            guard.execute("BEGIN IMMEDIATE")
            binding = hashlib.sha256((machine_id() + "|" + os.path.normcase(str(self.app_root))).encode("utf-8")).hexdigest()
            if self._config_path.exists():
                value = read_json(self._config_path)
                if value.get("installation_id") != self.installation_id:
                    raise ValueError("Agent Access installation identity mismatch")
                if type(value.get("port")) is not int or not 1024 <= value["port"] <= 65535:
                    raise ValueError("Invalid Agent Access port")
                if value.get("installation_binding", binding) != binding:
                    raise ValueError("Agent Access settings belong to another installation. Do not copy paired credentials between Main, Dev or workstations.")
                if "installation_binding" not in value:
                    value["installation_binding"] = binding
                    atomic_json(self._config_path, value)
                return
            if requested_port is not None and (type(requested_port) is not int or not 1024 <= requested_port <= 65535):
                raise ValueError("Agent Access port must be between 1024 and 65535")
            initial = requested_port or (53000 + int(self.installation_id.replace("-", "")[:4], 16) % 5000)
            chosen = None
            for candidate in range(initial, min(initial + (1 if requested_port is not None else 64), 65536)):
                with socket.socket() as probe:
                    try:
                        probe.bind(("127.0.0.1", candidate))
                    except OSError:
                        continue
                    chosen = candidate
                    break
            if chosen is None:
                if requested_port is not None:
                    raise OSError("Could not reserve the requested Agent Access port")
                # Windows/Hyper-V can reserve whole high-port blocks. Let the
                # OS choose an available port once, then persist that choice.
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", 0))
                    chosen = probe.getsockname()[1]
            atomic_json(self._config_path, {
                "schema": 1, "installation_id": self.installation_id,
                "workstation_id": machine_id(), "port": chosen,
                "label": ((self.app_root.parent.name if self.app_root.name == "app" else self.app_root.name) if self.app_root else "Maestro")[:64],
                "discovery_enabled": True,
                "installation_binding": binding,
            })

    @property
    def config(self) -> dict:
        return read_json(self._config_path)

    @property
    def security_required(self) -> bool:
        # Default-off installations avoid a SQLite transaction on every image,
        # preview and polling request. Security never resets when MCP is off.
        if not self._security_required and (self.directory / "security.json").exists():
            self._security_required = True
        return self._security_required

    def mark_security_configured(self) -> None:
        atomic_json(self.directory / "security.json", {"owner_configured": True})
        self._security_required = True

    @property
    def port(self) -> int:
        return int(self.config["port"])

    @property
    def origin(self) -> str:
        return f"https://{self.hostname}:{self.port}"

    @property
    def mcp_url(self) -> str:
        return self.origin + "/mcp"

    @property
    def local_url(self) -> str:
        return f"https://127.0.0.1:{self.port}"

    @property
    def allowed_origins(self) -> tuple[str, ...]:
        hosts = [self.hostname, "127.0.0.1", "localhost", "::1"]
        hosts.extend(self._certificate_ip_addresses)
        return tuple(dict.fromkeys(f"https://{_format_host(host)}:{self.port}" for host in hosts))

    @property
    def allowed_authorities(self) -> tuple[str, ...]:
        hosts = [self.hostname, "127.0.0.1", "localhost", "::1"]
        hosts.extend(self._certificate_ip_addresses)
        return tuple(dict.fromkeys(f"{_format_host(host)}:{self.port}" for host in hosts))

    def configure_certificate_ips(self, ip_addresses) -> None:
        """Install only IPs present in the TLS certificate used by this server."""
        values = set()
        for item in ip_addresses or ():
            try:
                values.add(ipaddress.ip_address(str(item)).compressed)
            except ValueError:
                continue
        self._certificate_ip_addresses = tuple(sorted(values))

    def origin_for_scope(self, scope: dict, *, allow_insecure_remote: bool = False) -> str | None:
        """Return the exact approved endpoint origin represented by an ASGI scope."""
        scheme = str(scope.get("scheme", "")).lower()
        if scheme not in {"https", "http"}:
            return None
        header_values = [
            value for key, value in scope.get("headers", []) if key.lower() == b"host"
        ]
        if len(header_values) != 1:
            return None
        try:
            authority = header_values[0].decode("latin1")
            parsed = urlsplit(scheme + "://" + authority)
            port = parsed.port
        except (UnicodeError, ValueError):
            return None
        if (
            parsed.username is not None
            or parsed.password is not None
            or not parsed.hostname
            or parsed.path
            or parsed.query
            or parsed.fragment
            or port != self.port
        ):
            return None
        host = parsed.hostname.casefold().rstrip(".")
        loopback_hosts = {"127.0.0.1", "localhost", "::1"}
        allowed_hosts = {self.hostname.casefold(), *loopback_hosts, *self._certificate_ip_addresses}
        if host not in allowed_hosts:
            return None
        if scheme == "http":
            # Plain HTTP is accepted for direct loopback callers. An optional
            # read-only login landing page may also be served remotely; it
            # contains no credential form and links back to HTTPS.
            peer = str((scope.get("client") or ("", 0))[0])
            try:
                peer_loopback = ipaddress.ip_address(peer).is_loopback
                if not peer_loopback and not allow_insecure_remote:
                    return None
                if peer_loopback and host not in loopback_hosts and not allow_insecure_remote:
                    return None
            except ValueError:
                return None
            if any(key.lower() == b"forwarded" or key.lower().startswith(b"x-forwarded-") for key, _ in scope.get("headers", [])):
                return None
        return f"{scheme}://{_format_host(host)}:{self.port}"

    def canonical_resource(self, resource: str) -> str:
        """Map an exact approved controller/MCP alias to the canonical audience."""
        if not isinstance(resource, str) or len(resource) > 2048 or "?" in resource or "#" in resource:
            raise ValueError("resource must be an approved Maestro MCP URL")
        try:
            parsed = urlsplit(resource)
            port = parsed.port
        except ValueError as error:
            raise ValueError("resource must be an approved Maestro MCP URL") from error
        if (
            parsed.scheme.lower() != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path != "/mcp"
            or parsed.query
            or parsed.fragment
            or port != self.port
        ):
            raise ValueError("resource must be an approved Maestro MCP URL")
        host = parsed.hostname.casefold().rstrip(".")
        allowed_hosts = {
            self.hostname.casefold(), "127.0.0.1", "localhost", "::1",
            *self._certificate_ip_addresses,
        }
        if host not in allowed_hosts:
            raise ValueError("resource must be an approved Maestro MCP URL")
        return self.mcp_url

    def lan_endpoint_metadata(self, interface_addresses) -> dict:
        """Return public direct-IP URLs only where this cert has matching SANs."""
        current = set()
        for item in interface_addresses or ():
            try:
                address = ipaddress.ip_address(str(item))
            except ValueError:
                continue
            if (
                isinstance(address, ipaddress.IPv4Address)
                and address.is_private
                and not address.is_loopback
                and not address.is_link_local
                and not address.is_multicast
                and not address.is_unspecified
                and not address.is_reserved
            ):
                current.add(address.compressed)
        usable = sorted(current.intersection(self._certificate_ip_addresses))
        return {
            "certificate_ip_addresses": list(self._certificate_ip_addresses),
            "lan_mcp_urls": [f"https://{address}:{self.port}/mcp" for address in usable],
        }

    def update(self, *, label: str | None = None, discovery_enabled: bool | None = None) -> dict:
        with self._lock:
            value = self.config
            if label is not None:
                self.validate_settings(label=label)
                value["label"] = label.strip()
            if discovery_enabled is not None:
                if not isinstance(discovery_enabled, bool):
                    raise ValueError("discovery_enabled must be a boolean")
                value["discovery_enabled"] = discovery_enabled
            atomic_json(self._config_path, value)
            return value

    @staticmethod
    def validate_settings(*, label=None, discovery_enabled=None, enabled=None) -> None:
        if isinstance(label, str):
            try:
                label.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError("Use a valid Unicode display name") from error
        if label is not None and (not isinstance(label, str) or not label.strip() or len(label) > 64 or any(ord(c) < 32 for c in label)):
            raise ValueError("Use a display name between 1 and 64 characters")
        for name, value in (("discovery_enabled", discovery_enabled), ("enabled", enabled)):
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")

    def register_backend(self, url: str, *, ready: bool, session_id: str, launcher_script: str = "start.js") -> None:
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"} or not parsed.port or parsed.username or parsed.query or parsed.fragment:
            raise ValueError("The backend must be this installation's loopback HTTP listener")
        if launcher_script not in {"start.js", "start_sol.js", "start_classic.js"}:
            raise ValueError("Invalid Maestro launcher")
        atomic_json(self.directory / "backend.json", {
            "schema": 1, "installation_id": self.installation_id,
            "auth_boundary_version": 1,
            "url": url.rstrip("/"), "pid": os.getpid(), "ready": bool(ready),
            "session_id": session_id, "launcher_script": launcher_script, "updated_at": time.time(),
        })

    def backend(self) -> dict:
        value = read_json(self.directory / "backend.json")
        if not value:
            return {}
        parsed = urlsplit(str(value.get("url", "")))
        if value.get("installation_id") != self.installation_id or parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"} or not parsed.port:
            raise ValueError("Invalid backend registration")
        return value

    def audit(self, actor: str, action: str, status: str) -> None:
        # Never record request bodies, prompts, paths, tokens or raw exceptions.
        values = (time.time(), str(actor)[:80], str(action)[:160], str(status)[:80])
        with sqlite3.connect(self._audit_db) as db:
            db.execute("INSERT INTO events(time,actor,action,status) VALUES(?,?,?,?)", values)
            db.execute("DELETE FROM events WHERE id NOT IN (SELECT id FROM events ORDER BY id DESC LIMIT 2000)")

    def events(self, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self._audit_db) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (max(1, min(int(limit), 200)),))]

    def public_status(self) -> dict:
        auth = self.auth.status()
        config = self.config
        controller = read_json(self.directory / "controller.json")
        controller_status = controller.get("status", "not_started")
        if controller_status == "ready" and not process_alive(controller.get("pid")):
            controller_status = "stopped"
        backend = self.backend()
        lan_mcp_urls = controller.get("lan_mcp_urls", []) if auth.get("enabled") else []
        return {
            **auth, "available": True, "label": config["label"],
            "workstation_id": config["workstation_id"], "controller_url": self.origin,
            "local_setup_url": self.local_url, "mcp_url": self.mcp_url,
            "lan_mcp_urls": lan_mcp_urls,
            "controller_status": controller_status,
            "certificate_sha256": controller.get("certificate_sha256", ""),
            "discovery_status": controller.get("discovery_status", "disabled"),
            "backend_online": bool(backend.get("ready") and process_alive(backend.get("pid"))),
        }


def _format_host(host: str) -> str:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    return f"[{address.compressed}]" if isinstance(address, ipaddress.IPv6Address) else address.compressed
