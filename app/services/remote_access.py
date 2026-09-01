"""Optional Tailscale Serve integration for private HTTPS Maestro access."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any


_HTTPS_RE = re.compile(r"https://[^\s|]+", re.IGNORECASE)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            if os.path.exists(temporary):
                os.unlink(temporary)
        except OSError:
            pass


def _tailscale_candidates() -> list[str]:
    candidates: list[str] = []
    discovered = shutil.which("tailscale")
    if discovered:
        candidates.append(discovered)
    if os.name == "nt":
        for base_name in ("ProgramFiles", "ProgramFiles(x86)"):
            base = os.environ.get(base_name)
            if base:
                candidates.append(os.path.join(base, "Tailscale", "tailscale.exe"))
    elif platform.system() == "Darwin":
        candidates.extend([
            "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
            "/usr/local/bin/tailscale",
            "/opt/homebrew/bin/tailscale",
        ])
    else:
        candidates.extend(["/usr/bin/tailscale", "/usr/local/bin/tailscale"])
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = os.path.normcase(os.path.abspath(candidate))
        if normalized not in seen and os.path.isfile(candidate):
            seen.add(normalized)
            unique.append(candidate)
    return unique


def find_tailscale() -> str | None:
    candidates = _tailscale_candidates()
    return candidates[0] if candidates else None


def _install_url() -> str:
    system = platform.system()
    if system == "Windows":
        return "https://tailscale.com/download/windows"
    if system == "Darwin":
        return "https://tailscale.com/download/mac"
    return "https://tailscale.com/download/linux"


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0


def _has_serve_configuration(status_text: str) -> bool:
    """Return whether ``serve status`` describes an active route."""
    normalized = status_text.strip().lower()
    if not normalized or normalized in {"{}", "null", "[]"}:
        return False
    return not any(
        marker in normalized
        for marker in (
            "no serve config",
            "no serve configuration",
            "not currently serving",
        )
    )


class TailscaleManager:
    """Detect Tailscale, maintain a private HTTPS Serve route, and report it."""

    def __init__(
        self,
        settings_dir: str | os.PathLike[str],
        server_port: int,
    ):
        self._path = Path(settings_dir) / "remote_access.json"
        self._server_port = int(server_port)
        self._lock = threading.RLock()

    def _read_preference(self) -> dict[str, Any]:
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _save_preference(self, enabled: bool) -> None:
        current = self._read_preference()
        windows_restore_task = bool(current.get("windows_restore_task", False))
        _atomic_write(self._path, {
            "version": 3,
            "enabled": bool(enabled),
            "target_port": self._server_port,
            # Pinokio's launcher reads this opt-in flag and reuses the exact
            # backend port on later starts. PINOKIO_SHARE_LOCAL_PORT controls
            # a separate LAN proxy and cannot keep a Tailscale Serve target
            # stable by itself.
            "pinokio_port_lock": bool(enabled),
            # Preserve the one-time Windows helper marker when the in-app
            # toggle disables and later re-enables the route. The task itself
            # remains inert while `enabled` is false.
            "windows_restore_task": windows_restore_task,
            "windows_restore_task_name": (
                str(
                    current.get("windows_restore_task_name")
                    or "Maestro Tailscale Serve"
                )
                if windows_restore_task
                else None
            ),
        })

    def _run(
        self,
        executable: str,
        arguments: list[str],
        *,
        timeout: float = 12.0,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [executable, *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            creationflags=_creation_flags(),
        )

    def _status_payload(self, executable: str) -> tuple[dict[str, Any], str | None]:
        try:
            result = self._run(executable, ["status", "--json"])
        except (OSError, subprocess.SubprocessError) as exc:
            return {}, str(exc)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "Tailscale is not connected").strip()
            return {}, detail
        try:
            payload = json.loads(result.stdout)
            return (payload if isinstance(payload, dict) else {}), None
        except ValueError as exc:
            return {}, f"Could not read Tailscale status: {exc}"

    def _serve_status(self, executable: str) -> tuple[str, bool]:
        try:
            result = self._run(executable, ["serve", "status", "--json"])
            text = "\n".join(filter(None, [result.stdout, result.stderr])).strip()
            if result.returncode != 0 or not text:
                fallback = self._run(executable, ["serve", "status"])
                text = "\n".join(
                    filter(None, [fallback.stdout, fallback.stderr])
                ).strip()
        except (OSError, subprocess.SubprocessError):
            return "", False
        target_tokens = {
            f"127.0.0.1:{self._server_port}",
            f"localhost:{self._server_port}",
        }
        configured = any(token in text for token in target_tokens)
        return text, configured

    def status(self) -> dict[str, Any]:
        executable = find_tailscale()
        preference = self._read_preference()
        base = {
            "installed": bool(executable),
            "connected": False,
            "backend_state": "Missing" if not executable else "Unknown",
            "dns_name": None,
            "https_url": None,
            "configured": False,
            "enabled": bool(preference.get("enabled", False)),
            "target_port": self._server_port,
            "install_url": _install_url(),
            "platform": platform.system().lower(),
            "needs_login": False,
            "error": None,
        }
        if not executable:
            return base

        payload, error = self._status_payload(executable)
        backend_state = str(payload.get("BackendState") or "Unknown")
        self_info = payload.get("Self") if isinstance(payload.get("Self"), dict) else {}
        dns_name = str(self_info.get("DNSName") or "").strip().rstrip(".")
        connected = backend_state.lower() == "running" and bool(
            self_info.get("Online", True)
        )
        serve_text, configured = self._serve_status(executable) if connected else ("", False)
        matched_url = _HTTPS_RE.search(serve_text)
        https_url = matched_url.group(0).rstrip("/.,") if matched_url else None
        if not https_url and connected and dns_name:
            https_url = f"https://{dns_name}"
        base.update({
            "connected": connected,
            "backend_state": backend_state,
            "dns_name": dns_name or None,
            "https_url": https_url,
            "configured": configured,
            "needs_login": not connected,
            "error": error,
        })
        return base

    def enable(self) -> dict[str, Any]:
        with self._lock:
            executable = find_tailscale()
            if not executable:
                raise RuntimeError(
                    "Tailscale is not installed. Install it, sign in, then try again."
                )
            current = self.status()
            if not current["connected"]:
                raise RuntimeError(
                    "Tailscale is installed but not signed in. Open Tailscale, sign in, then try again."
                )
            serve_text, already_configured = self._serve_status(executable)
            if _has_serve_configuration(serve_text) and not already_configured:
                raise RuntimeError(
                    "This computer already has a different Tailscale Serve route. "
                    "Maestro left it unchanged. Remove or relocate that route before enabling Maestro private access."
                )
            target = f"http://127.0.0.1:{self._server_port}"
            try:
                result = self._run(
                    executable,
                    ["serve", "--bg", "--yes", "--https=443", target],
                    timeout=30.0,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "Tailscale Serve did not finish configuring. Use Maestro's Secure Remote Access action in Pinokio once."
                ) from exc
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "Unknown Tailscale error").strip()
                raise RuntimeError(
                    "Tailscale Serve could not be enabled. On Windows, use the Secure Remote Access action in Pinokio so it can request administrator permission. "
                    + detail
                )
            self._save_preference(True)
            refreshed = self.status()
            if not refreshed.get("https_url"):
                match = _HTTPS_RE.search("\n".join([result.stdout, result.stderr]))
                if match:
                    refreshed["https_url"] = match.group(0).rstrip("/.,")
            return refreshed

    def disable(self) -> dict[str, Any]:
        with self._lock:
            executable = find_tailscale()
            if executable:
                try:
                    result = self._run(
                        executable,
                        ["serve", "--https=443", "off"],
                        timeout=20.0,
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    raise RuntimeError(f"Could not disable Tailscale Serve: {exc}") from exc
                if result.returncode != 0:
                    detail = (result.stderr or result.stdout or "Unknown Tailscale error").strip()
                    raise RuntimeError(f"Could not disable Tailscale Serve: {detail}")
            self._save_preference(False)
            return self.status()

    def refresh_if_enabled(self) -> None:
        """Retarget Serve after Pinokio assigns a different startup port."""
        preference = self._read_preference()
        if not preference.get("enabled"):
            return

        # `tailscale serve --bg` persists across app and machine restarts. The
        # Windows launcher also dispatches the fixed, user-approved on-demand
        # restore task before starting this backend. Avoid a second CLI call
        # here because the backend intentionally runs without elevation.
        try:
            saved_port = int(preference.get("target_port"))
        except (TypeError, ValueError):
            saved_port = None
        if saved_port == self._server_port:
            if os.name == "nt" and preference.get("windows_restore_task"):
                print(
                    "[Remote Access] Reusing the saved Maestro port; "
                    "the Windows Tailscale restore helper was requested."
                )
            elif os.name == "nt":
                print(
                    "[Remote Access] Reusing the saved Maestro port. Run "
                    "Secure Remote Access once after updating to install the "
                    "restart-safe Windows helper."
                )
            else:
                print(
                    "[Remote Access] Reusing the saved Maestro port; "
                    "the persistent Tailscale route remains valid."
                )
            return

        def worker() -> None:
            try:
                refreshed = self.enable()
                if refreshed.get("https_url"):
                    print(
                        "[Remote Access] Private Tailscale URL: "
                        f"{refreshed['https_url']}"
                    )
            except Exception as exc:
                print(f"[Remote Access] Tailscale route refresh skipped: {exc}")

        threading.Thread(
            target=worker,
            daemon=True,
            name="maestro_tailscale_refresh",
        ).start()
