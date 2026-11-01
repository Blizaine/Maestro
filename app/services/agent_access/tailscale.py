"""Private Tailscale HTTPS certificate detection and provisioning.

This module owns only the optional public-CA certificate used for the exact
local machine's ``*.ts.net`` name. It does not configure Tailscale Serve,
Funnel, DNS, trust stores, or the Agent Access self-signed LAN certificate.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import (
    dsa,
    ec,
    ed448,
    ed25519,
    padding,
    rsa,
)
from cryptography.x509.oid import ExtendedKeyUsageOID

from ..remote_access import find_tailscale
from .state import ControllerLock

_DNS_LABEL_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_GENERATION_RE = re.compile(r"[0-9a-f]{32}\Z")
_MAX_STATUS_BYTES = 4 * 1024 * 1024
_MAX_CERT_BYTES = 256 * 1024
_MAX_KEY_BYTES = 32 * 1024
_RENEW_BEFORE = timedelta(days=30)
_ISSUE_COOLDOWN = timedelta(minutes=15)
_MAX_GENERATIONS = 3


class TailscaleCertificateError(RuntimeError):
    """Tailscale is unavailable or its certificate could not be trusted."""


@dataclass(frozen=True)
class TailscaleCertificateMaterial:
    """Validated certificate files ready for an SNI-specific TLS context."""

    hostname: str
    cert_path: Path
    key_path: Path = field(repr=False)
    expires_at: datetime
    generation: str


@dataclass(frozen=True)
class TailscaleStatus:
    """Sanitized status; no CLI output, peer data, file paths, or key material."""

    installed: bool | None
    connected: bool | None
    hostname: str | None
    certificate_ready: bool
    expires_at: str | None
    needs_renewal: bool
    error: str | None = None


@dataclass(frozen=True)
class _Identity:
    installed: bool
    connected: bool
    hostname: str | None
    error: str | None


def normalize_tailscale_dns_name(value: object) -> str:
    """Validate and normalize one exact machine DNS name under ``.ts.net``."""
    if not isinstance(value, str) or not value or len(value) > 254:
        raise TailscaleCertificateError("Tailscale returned an invalid DNS name")
    # Tailscale status commonly uses an absolute DNS name with one final dot.
    name = value.removesuffix(".")
    if not name or name.endswith(".") or name != name.strip():
        raise TailscaleCertificateError("Tailscale returned an invalid DNS name")
    try:
        name.encode("ascii")
    except UnicodeEncodeError as exc:
        raise TailscaleCertificateError(
            "Tailscale returned an invalid DNS name"
        ) from exc
    normalized = name.lower()
    labels = normalized.split(".")
    if (
        len(normalized) > 253
        or len(labels) < 3
        or labels[-2:] != ["ts", "net"]
        or any(not _DNS_LABEL_RE.fullmatch(label) for label in labels)
    ):
        raise TailscaleCertificateError("Tailscale returned an invalid DNS name")
    return normalized


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _certificate_time_bounds(certificate: x509.Certificate) -> tuple[datetime, datetime]:
    # Use the timezone-aware accessors where available (cryptography 42+).
    not_before = getattr(certificate, "not_valid_before_utc", None)
    not_after = getattr(certificate, "not_valid_after_utc", None)
    if not_before is None:
        not_before = _utc(certificate.not_valid_before)
    if not_after is None:
        not_after = _utc(certificate.not_valid_after)
    return _utc(not_before), _utc(not_after)


def _verify_certificate_signature(
    certificate: x509.Certificate, issuer_key: object
) -> None:
    signature = certificate.signature
    tbs = certificate.tbs_certificate_bytes
    algorithm = certificate.signature_hash_algorithm
    if isinstance(issuer_key, rsa.RSAPublicKey):
        parameters = getattr(certificate, "signature_algorithm_parameters", None)
        if not isinstance(parameters, (padding.PKCS1v15, padding.PSS)):
            parameters = padding.PKCS1v15()
        issuer_key.verify(signature, tbs, parameters, algorithm)
    elif isinstance(issuer_key, ec.EllipticCurvePublicKey):
        issuer_key.verify(signature, tbs, ec.ECDSA(algorithm))
    elif isinstance(issuer_key, dsa.DSAPublicKey):
        issuer_key.verify(signature, tbs, algorithm)
    elif isinstance(issuer_key, (ed25519.Ed25519PublicKey, ed448.Ed448PublicKey)):
        issuer_key.verify(signature, tbs)
    else:
        raise TypeError("unsupported certificate issuer key")


def _validate_certificate(
    cert_bytes: bytes,
    key_bytes: bytes,
    hostname: str,
    *,
    allow_expired: bool,
) -> datetime:
    if len(cert_bytes) > _MAX_CERT_BYTES or len(key_bytes) > _MAX_KEY_BYTES:
        raise TailscaleCertificateError("Stored Tailscale TLS assets are too large")
    try:
        chain = x509.load_pem_x509_certificates(cert_bytes)
        if not chain:
            raise ValueError("empty certificate chain")
        leaf = chain[0]
        private_key = serialization.load_pem_private_key(key_bytes, password=None)
        sans = leaf.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
        basic_constraints = leaf.extensions.get_extension_for_class(
            x509.BasicConstraints
        ).value
        extended_key_usage = leaf.extensions.get_extension_for_class(
            x509.ExtendedKeyUsage
        ).value
    except (ValueError, TypeError, x509.ExtensionNotFound) as exc:
        raise TailscaleCertificateError(
            "Stored Tailscale TLS certificate is invalid"
        ) from exc

    leaf_names = {
        name.lower()
        for name in sans.get_values_for_type(x509.DNSName)
        if isinstance(name, str)
    }
    if hostname not in leaf_names or basic_constraints.ca:
        raise TailscaleCertificateError(
            "Stored Tailscale TLS certificate does not cover this machine"
        )
    if ExtendedKeyUsageOID.SERVER_AUTH not in extended_key_usage:
        raise TailscaleCertificateError(
            "Stored Tailscale TLS certificate is not valid for HTTPS"
        )

    key_public = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    cert_public = leaf.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    if key_public != cert_public:
        raise TailscaleCertificateError(
            "Stored Tailscale TLS certificate and key do not match"
        )

    # A leaf that merely claims a different issuer is not sufficient. Verify
    # its signature with the matching issuer certificate delivered in the PEM
    # chain so a fabricated self-signed leaf cannot be accepted as trusted.
    if leaf.issuer == leaf.subject:
        raise TailscaleCertificateError(
            "Stored Tailscale TLS certificate is self-signed"
        )
    issuer = next((item for item in chain[1:] if item.subject == leaf.issuer), None)
    if issuer is None or issuer.subject == leaf.subject:
        raise TailscaleCertificateError(
            "Stored Tailscale TLS certificate issuer is missing"
        )
    try:
        _verify_certificate_signature(leaf, issuer.public_key())
        issuer_constraints = issuer.extensions.get_extension_for_class(
            x509.BasicConstraints
        ).value
        if not issuer_constraints.ca:
            raise ValueError("issuer is not a CA")
    except Exception as exc:
        # Cryptography's backend raises its own InvalidSignature type. Keep all
        # parser/verification details private and report only a stable message.
        raise TailscaleCertificateError(
            "Stored Tailscale TLS certificate signature is invalid"
        ) from exc

    not_before, not_after = _certificate_time_bounds(leaf)
    now = datetime.now(timezone.utc)
    if not_before > now or (not allow_expired and not_after <= now):
        raise TailscaleCertificateError(
            "Stored Tailscale TLS certificate is not currently valid"
        )
    return not_after


def _is_regular_file(path: Path) -> bool:
    return not path.is_symlink() and path.is_file()


class TailscaleCertificateManager:
    """Read Tailscale identity and maintain its isolated HTTPS certificate.

    ``status`` and ``load_certificate`` only inspect local files. ``detect``
    performs one bounded ``tailscale status --json`` read. Certificate
    issuance occurs only through ``ensure_certificate`` or ``renew``.
    """

    def __init__(self, settings_dir: str | os.PathLike[str]):
        self._settings_dir = Path(settings_dir)
        self._directory = self._settings_dir / "tailscale-certs"
        self._generations = self._directory / "generations"
        self._pointer = self._directory / "current.json"
        self._attempt_path = self._directory / "last-issue.json"
        self._lock = threading.RLock()

    def _run(
        self,
        executable: str,
        arguments: list[str],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [executable, *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
            creationflags=_creation_flags(),
        )

    def _detect_identity(self) -> _Identity:
        executable = find_tailscale()
        if not executable:
            return _Identity(False, False, None, "Tailscale is not installed")
        try:
            result = self._run(executable, ["status", "--json"], timeout=8.0)
        except (OSError, subprocess.SubprocessError):
            return _Identity(True, False, None, "Could not read Tailscale status")
        if result.returncode != 0 or not isinstance(result.stdout, str):
            return _Identity(True, False, None, "Could not read Tailscale status")
        if len(result.stdout.encode("utf-8", errors="replace")) > _MAX_STATUS_BYTES:
            return _Identity(True, False, None, "Tailscale status response is too large")
        try:
            payload = json.loads(result.stdout)
        except (ValueError, TypeError):
            return _Identity(True, False, None, "Could not read Tailscale status")
        if not isinstance(payload, dict):
            return _Identity(True, False, None, "Could not read Tailscale status")

        self_info = payload.get("Self")
        if not isinstance(self_info, dict):
            return _Identity(True, False, None, "Could not read Tailscale status")
        try:
            dns_name = normalize_tailscale_dns_name(self_info.get("DNSName"))
        except TailscaleCertificateError:
            return _Identity(True, False, None, "Tailscale returned an invalid DNS name")

        backend_state = payload.get("BackendState")
        connected = (
            isinstance(backend_state, str)
            and backend_state.casefold() == "running"
            and self_info.get("Online", True) is True
        )
        error = None if connected else "Tailscale is not connected"
        return _Identity(True, connected, dns_name, error)

    def _check_existing_directory(self, path: Path) -> None:
        if path.is_symlink() or not path.is_dir():
            raise TailscaleCertificateError(
                "Tailscale certificate storage is not a private directory"
            )
        if os.name != "nt":
            try:
                if path.stat().st_mode & 0o077:
                    raise TailscaleCertificateError(
                        "Tailscale certificate storage permissions are too broad"
                    )
            except OSError as exc:
                raise TailscaleCertificateError(
                    "Could not inspect Tailscale certificate storage"
                ) from exc

    def _ensure_private_directory(self, path: Path) -> None:
        try:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            if path.is_symlink() or not path.is_dir():
                raise TailscaleCertificateError(
                    "Tailscale certificate storage is not a private directory"
                )
            if os.name != "nt":
                os.chmod(path, 0o700)
                if path.stat().st_mode & 0o077:
                    raise TailscaleCertificateError(
                        "Could not secure Tailscale certificate storage"
                    )
            else:
                self._restrict_windows_acl(path, is_directory=True)
        except TailscaleCertificateError:
            raise
        except OSError as exc:
            raise TailscaleCertificateError(
                "Could not create private Tailscale certificate storage"
            ) from exc

    def _secure_file(self, path: Path) -> None:
        try:
            os.chmod(path, 0o600)
            if os.name != "nt" and path.stat().st_mode & 0o077:
                raise TailscaleCertificateError(
                    "Could not secure Tailscale certificate files"
                )
            if os.name == "nt":
                self._restrict_windows_acl(path, is_directory=False)
        except TailscaleCertificateError:
            raise
        except OSError as exc:
            raise TailscaleCertificateError(
                "Could not secure Tailscale certificate files"
            ) from exc

    @staticmethod
    def _restrict_windows_acl(path: Path, *, is_directory: bool) -> None:
        account = getpass.getuser()
        domain = os.environ.get("USERDOMAIN")
        if domain:
            account = f"{domain}\\{account}"
        access = "(OI)(CI)F" if is_directory else "F"
        try:
            result = subprocess.run(
                [
                    "icacls.exe",
                    str(path),
                    "/inheritance:r",
                    "/grant:r",
                    f"{account}:{access}",
                    f"SYSTEM:{access}",
                ],
                capture_output=True,
                timeout=5,
                check=False,
                creationflags=_creation_flags(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise TailscaleCertificateError(
                "Could not restrict Tailscale certificate permissions"
            ) from exc
        if result.returncode != 0:
            raise TailscaleCertificateError(
                "Could not restrict Tailscale certificate permissions"
            )

    def _prepare_store(self) -> None:
        base = self._settings_dir
        try:
            if base.is_symlink() or not base.is_dir():
                raise TailscaleCertificateError(
                    "Agent Access settings directory is not available"
                )
        except OSError as exc:
            raise TailscaleCertificateError(
                "Agent Access settings directory is not available"
            ) from exc
        self._ensure_private_directory(self._directory)
        self._ensure_private_directory(self._generations)

    def _check_store_for_read(self) -> bool:
        if not self._directory.exists():
            return False
        self._check_existing_directory(self._directory)
        if not self._generations.exists():
            return False
        self._check_existing_directory(self._generations)
        return True

    def _read_current_record(self) -> tuple[str, str] | None:
        if not self._check_store_for_read():
            return None
        if self._pointer.is_symlink():
            raise TailscaleCertificateError("Tailscale certificate metadata is invalid")
        if not self._pointer.exists():
            return None
        try:
            if not _is_regular_file(self._pointer) or self._pointer.stat().st_size > 4096:
                raise TailscaleCertificateError(
                    "Tailscale certificate metadata is invalid"
                )
            value = json.loads(self._pointer.read_text(encoding="utf-8"))
        except TailscaleCertificateError:
            raise
        except (OSError, ValueError, TypeError, UnicodeDecodeError) as exc:
            raise TailscaleCertificateError(
                "Tailscale certificate metadata is invalid"
            ) from exc
        generation = value.get("generation") if isinstance(value, dict) else None
        hostname = value.get("hostname") if isinstance(value, dict) else None
        if (
            not isinstance(generation, str)
            or not _GENERATION_RE.fullmatch(generation)
            or not isinstance(hostname, str)
            or normalize_tailscale_dns_name(hostname) != hostname
        ):
            raise TailscaleCertificateError("Tailscale certificate metadata is invalid")
        return generation, normalize_tailscale_dns_name(hostname)

    def _load_generation(
        self,
        generation: str,
        hostname: str,
        *,
        expected_hostname: str | None = None,
        allow_expired: bool = False,
    ) -> TailscaleCertificateMaterial:
        if not _GENERATION_RE.fullmatch(generation):
            raise TailscaleCertificateError("Tailscale certificate metadata is invalid")
        directory = self._generations / generation
        if directory.is_symlink() or not directory.is_dir():
            raise TailscaleCertificateError("Tailscale certificate generation is missing")
        self._check_existing_directory(directory)
        cert_path, key_path = directory / "cert.pem", directory / "key.pem"
        for path in (cert_path, key_path):
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise TailscaleCertificateError("Stored Tailscale TLS assets are invalid")
        if not cert_path.exists() or not key_path.exists():
            raise TailscaleCertificateError("Stored Tailscale TLS assets are incomplete")
        try:
            if cert_path.stat().st_size > _MAX_CERT_BYTES or key_path.stat().st_size > _MAX_KEY_BYTES:
                raise TailscaleCertificateError("Stored Tailscale TLS assets are too large")
        except OSError as exc:
            raise TailscaleCertificateError("Could not inspect stored Tailscale TLS assets") from exc
        if os.name != "nt":
            for path in (cert_path, key_path):
                if path.stat().st_mode & 0o077:
                    raise TailscaleCertificateError(
                        "Tailscale certificate file permissions are too broad"
                    )
        try:
            cert_bytes = cert_path.read_bytes()
            key_bytes = key_path.read_bytes()
        except OSError as exc:
            raise TailscaleCertificateError(
                "Could not read stored Tailscale TLS assets"
            ) from exc

        pointer_hostname = normalize_tailscale_dns_name(hostname)
        if expected_hostname is not None:
            expected = normalize_tailscale_dns_name(expected_hostname)
            if pointer_hostname != expected:
                raise TailscaleCertificateError(
                    "Stored Tailscale TLS certificate belongs to another machine"
                )
        expires_at = _validate_certificate(
            cert_bytes,
            key_bytes,
            pointer_hostname,
            allow_expired=allow_expired,
        )
        return TailscaleCertificateMaterial(
            hostname=pointer_hostname,
            cert_path=cert_path,
            key_path=key_path,
            expires_at=expires_at,
            generation=generation,
        )

    def load_certificate(
        self, expected_hostname: str | None = None
    ) -> TailscaleCertificateMaterial | None:
        """Return the validated active material without running Tailscale."""
        with self._lock:
            record = self._read_current_record()
            if record is None:
                return None
            generation, hostname = record
            material = self._load_generation(
                generation,
                hostname,
                expected_hostname=expected_hostname,
            )
            if material.expires_at <= datetime.now(timezone.utc):
                raise TailscaleCertificateError(
                    "Stored Tailscale TLS certificate is not currently valid"
                )
            return material

    def status(self) -> TailscaleStatus:
        """Read cached local certificate state only; never spawn a subprocess."""
        with self._lock:
            try:
                record = self._read_current_record()
                if record is None:
                    return TailscaleStatus(None, None, None, False, None, False)
                generation, hostname = record
                material = self._load_generation(
                    generation, hostname, allow_expired=True
                )
                now = datetime.now(timezone.utc)
                is_current = material.expires_at > now
                return TailscaleStatus(
                    None,
                    None,
                    material.hostname,
                    is_current,
                    material.expires_at.isoformat(),
                    material.expires_at <= now + _RENEW_BEFORE,
                    None if is_current else "Tailscale TLS certificate has expired",
                )
            except TailscaleCertificateError:
                return TailscaleStatus(
                    None,
                    None,
                    None,
                    False,
                    None,
                    False,
                    "Stored Tailscale TLS certificate is invalid",
                )

    def detect(self) -> TailscaleStatus:
        """Read only this machine's DNS identity from ``status --json``."""
        identity = self._detect_identity()
        material = None
        certificate_error = None
        if identity.hostname:
            try:
                with self._lock:
                    record = self._read_current_record()
                    if record is not None:
                        generation, hostname = record
                        material = self._load_generation(
                            generation,
                            hostname,
                            expected_hostname=identity.hostname,
                            allow_expired=True,
                        )
            except TailscaleCertificateError:
                material = None
                certificate_error = "Stored Tailscale TLS certificate is invalid"
        ready = bool(
            identity.connected
            and material is not None
            and material.hostname == identity.hostname
            and material.expires_at > datetime.now(timezone.utc)
        )
        expires_at = material.expires_at.isoformat() if material else None
        needs_renewal = bool(
            material
            and material.expires_at <= datetime.now(timezone.utc) + _RENEW_BEFORE
        )
        return TailscaleStatus(
            identity.installed,
            identity.connected,
            identity.hostname,
            ready,
            expires_at,
            needs_renewal,
            identity.error or certificate_error,
        )

    def probe(self) -> TailscaleStatus:
        """Compatibility alias for an explicit fresh read-only detection."""
        return self.detect()

    def _last_issue_time(self) -> datetime | None:
        if not self._attempt_path.exists() or self._attempt_path.is_symlink():
            return None
        try:
            if not _is_regular_file(self._attempt_path) or self._attempt_path.stat().st_size > 2048:
                return None
            value = json.loads(self._attempt_path.read_text(encoding="utf-8"))
            timestamp = value.get("attempted_at") if isinstance(value, dict) else None
            if not isinstance(timestamp, (int, float)):
                return None
            return datetime.fromtimestamp(float(timestamp), timezone.utc)
        except (OSError, ValueError, TypeError, OverflowError, UnicodeDecodeError):
            return None

    def _record_issue_time(self, at: datetime) -> None:
        payload = json.dumps({"attempted_at": at.timestamp()}, separators=(",", ":")).encode()
        _atomic_private_write(self._attempt_path, payload)

    def _issue_allowed(self, now: datetime) -> bool:
        last_attempt = self._last_issue_time()
        return last_attempt is None or now - last_attempt >= _ISSUE_COOLDOWN

    def _issue_certificate(self, executable: str, hostname: str) -> TailscaleCertificateMaterial:
        self._prepare_store()
        generation = uuid.uuid4().hex
        staging = self._generations / f".staging-{generation}"
        final = self._generations / generation
        published = False
        try:
            staging.mkdir(mode=0o700)
            if os.name == "nt":
                self._restrict_windows_acl(staging, is_directory=True)
            cert_path = staging / "cert.pem"
            key_path = staging / "key.pem"
            self._record_issue_time(datetime.now(timezone.utc))
            try:
                result = self._run(
                    executable,
                    [
                        "cert",
                        f"--cert-file={cert_path}",
                        f"--key-file={key_path}",
                        f"--min-validity={int(_RENEW_BEFORE.total_seconds())}s",
                        hostname,
                    ],
                    timeout=90.0,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise TailscaleCertificateError(
                    "Tailscale could not issue an HTTPS certificate"
                ) from exc
            if result.returncode != 0:
                raise TailscaleCertificateError(
                    "Tailscale could not issue an HTTPS certificate"
                )
            for path in (cert_path, key_path):
                if not _is_regular_file(path):
                    raise TailscaleCertificateError(
                        "Tailscale did not create certificate files"
                    )
                self._secure_file(path)
            cert_bytes = cert_path.read_bytes()
            key_bytes = key_path.read_bytes()
            expires_at = _validate_certificate(
                cert_bytes, key_bytes, hostname, allow_expired=False
            )
            confirmed_identity = self._detect_identity()
            if (
                not confirmed_identity.connected
                or confirmed_identity.hostname != hostname
            ):
                raise TailscaleCertificateError(
                    "Tailscale identity changed during certificate provisioning"
                )
            os.replace(staging, final)
            pointer = json.dumps(
                {"generation": generation, "hostname": hostname},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            _atomic_private_write(self._pointer, pointer)
            published = True
            self._prune_generations(keep={generation})
            return TailscaleCertificateMaterial(
                hostname=hostname,
                cert_path=final / "cert.pem",
                key_path=final / "key.pem",
                expires_at=expires_at,
                generation=generation,
            )
        except TailscaleCertificateError:
            self._remove_staging(staging)
            if not published:
                self._remove_generation(final)
            raise
        except (OSError, ValueError, TypeError) as exc:
            self._remove_staging(staging)
            if not published:
                self._remove_generation(final)
            raise TailscaleCertificateError(
                "Could not safely publish the Tailscale HTTPS certificate"
            ) from exc

    def _prune_generations(self, *, keep: set[str]) -> None:
        candidates: list[tuple[float, Path]] = []
        for path in self._generations.iterdir():
            if path.is_symlink() or not path.is_dir() or not _GENERATION_RE.fullmatch(path.name):
                continue
            if path.name in keep:
                continue
            try:
                candidates.append((path.stat().st_mtime, path))
            except OSError:
                continue
        # Keep two previous generations in case an already-open SNI context is
        # still using one while the atomic current pointer changes.
        for _modified, path in sorted(candidates, reverse=True)[_MAX_GENERATIONS - len(keep) :]:
            self._remove_generation(path)

    @staticmethod
    def _remove_generation(path: Path) -> None:
        if path.is_symlink() or not path.is_dir():
            return
        for name in ("cert.pem", "key.pem"):
            child = path / name
            if child.is_symlink() or (child.exists() and not child.is_file()):
                return
            try:
                child.unlink(missing_ok=True)
            except OSError:
                return
        try:
            path.rmdir()
        except OSError:
            return

    @classmethod
    def _remove_staging(cls, path: Path) -> None:
        if path.name.startswith(".staging-"):
            cls._remove_generation(path)

    def ensure_certificate(self) -> TailscaleCertificateMaterial:
        """Provision when missing/near expiry, otherwise reuse the stable cert."""
        return self._ensure_certificate(force=False)

    def provision(self) -> TailscaleCertificateMaterial:
        """Explicit provisioning alias for UI actions that enable this option."""
        return self.ensure_certificate()

    def renew(self) -> TailscaleCertificateMaterial:
        """Explicitly request a new certificate, subject to the issue cooldown."""
        return self._ensure_certificate(force=True)

    def _ensure_certificate(self, *, force: bool) -> TailscaleCertificateMaterial:
        with self._lock:
            identity = self._detect_identity()
            if not identity.installed:
                raise TailscaleCertificateError("Tailscale is not installed")
            if not identity.connected or not identity.hostname:
                raise TailscaleCertificateError(
                    identity.error or "Tailscale is not connected"
                )
            self._prepare_store()
            lock_directory = self._directory / "provision-lock"
            self._ensure_private_directory(lock_directory)
            guard = ControllerLock(lock_directory)
            try:
                guard.__enter__()
            except RuntimeError as exc:
                # Another process may have finished issuance while this
                # request waited. Only reuse its complete, currently valid
                # generation; never inspect or remove another process's stage.
                try:
                    current = self.load_certificate(identity.hostname)
                except TailscaleCertificateError:
                    current = None
                if current is not None:
                    return current
                raise TailscaleCertificateError(
                    "Tailscale certificate provisioning is already in progress"
                ) from exc
            try:
                return self._ensure_certificate_locked(force=force)
            finally:
                guard.__exit__(None, None, None)

    def _ensure_certificate_locked(
        self, *, force: bool
    ) -> TailscaleCertificateMaterial:
        # Refresh inside the cross-process lock so a hostname or connection
        # change while waiting cannot publish for a stale identity.
        identity = self._detect_identity()
        if not identity.installed:
            raise TailscaleCertificateError("Tailscale is not installed")
        if not identity.connected or not identity.hostname:
            raise TailscaleCertificateError(
                identity.error or "Tailscale is not connected"
            )
        executable = find_tailscale()
        if not executable:
            raise TailscaleCertificateError("Tailscale is not installed")

        current: TailscaleCertificateMaterial | None = None
        try:
            record = self._read_current_record()
            if record is not None:
                generation, hostname = record
                current = self._load_generation(
                    generation,
                    hostname,
                    expected_hostname=identity.hostname,
                    allow_expired=True,
                )
        except TailscaleCertificateError:
            current = None

        now = datetime.now(timezone.utc)
        if (
            not force
            and current is not None
            and current.expires_at > now + _RENEW_BEFORE
        ):
            return current
        if not self._issue_allowed(now):
            if current is not None and current.expires_at > now:
                # A recent attempt must not turn a still-valid certificate
                # into a renewal loop. The caller can report needs_renewal.
                return current
            raise TailscaleCertificateError(
                "Tailscale certificate request is waiting before retry"
            )
        return self._issue_certificate(executable, identity.hostname)


def _atomic_private_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        if os.name != "nt":
            os.chmod(path, 0o600)
            if path.stat().st_mode & 0o077:
                raise TailscaleCertificateError(
                    "Could not secure Tailscale certificate metadata"
                )
        else:
            TailscaleCertificateManager._restrict_windows_acl(path, is_directory=False)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


__all__ = [
    "TailscaleCertificateError",
    "TailscaleCertificateManager",
    "TailscaleCertificateMaterial",
    "TailscaleStatus",
    "normalize_tailscale_dns_name",
]
