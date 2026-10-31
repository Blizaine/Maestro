"""Private, self-signed TLS leaf provisioning for the agent-access listener."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import getpass
import hashlib
import ipaddress
import os
from pathlib import Path
import subprocess
from typing import Iterable
import uuid

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


class TLSAssetError(RuntimeError):
    """A stored TLS identity is incomplete, corrupt, or no longer valid."""


@dataclass(frozen=True)
class TLSMaterial:
    cert_path: Path
    key_path: Path
    fingerprint_sha256: str
    hostname: str
    ip_addresses: tuple[str, ...] = ()
    requires_retrust: bool = False


def hostname_for_installation_id(installation_id: str) -> str:
    """Return the stable local DNS name for a UUID-backed installation."""
    parsed = _installation_uuid(installation_id)
    return f"maestro-{parsed.hex[:16]}.local"


def _installation_uuid(installation_id: str) -> uuid.UUID:
    if not isinstance(installation_id, str) or not installation_id or len(installation_id) > 64:
        raise ValueError("installation_id must be a UUID string")
    try:
        return uuid.UUID(installation_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError("installation_id must be a UUID string") from exc


def _normalize_ips(ip_addresses: Iterable[str] | None) -> tuple[str, ...]:
    if ip_addresses is None:
        return ()
    if isinstance(ip_addresses, (str, bytes)):
        raise ValueError("ip_addresses must be a list of IP address strings")
    unique: dict[str, ipaddress.IPv4Address | ipaddress.IPv6Address] = {}
    for value in ip_addresses:
        if not isinstance(value, str):
            raise ValueError("Each IP address must be a string")
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise ValueError(f"Invalid IP address: {value!r}") from exc
        if isinstance(address, ipaddress.IPv6Address) and address.scope_id is not None:
            raise ValueError("Scoped IPv6 addresses cannot be stored in a TLS SAN")
        unique[address.compressed] = address
    for loopback in (ipaddress.ip_address("127.0.0.1"), ipaddress.ip_address("::1")):
        unique.pop(loopback.compressed, None)
    return tuple(sorted(unique))


def _credential_directory(directory: str | os.PathLike[str]) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise TLSAssetError("TLS credential directory must be a real directory")
    _secure_permissions(path, is_directory=True)
    return path


def _secure_permissions(path: Path, *, is_directory: bool) -> None:
    """Best-effort private directory/file permissions without global changes."""
    try:
        os.chmod(path, 0o700 if is_directory else 0o600)
    except OSError:
        pass

    if os.name != "nt":
        return
    # `chmod` on Windows controls only the read-only bit. Tighten the DACL on
    # this generated path when icacls is available; never touch parent/global
    # certificate stores or any unrelated directory.
    try:
        account = getpass.getuser()
        if os.environ.get("USERDOMAIN"):
            account = f"{os.environ['USERDOMAIN']}\\{account}"
        grant_suffix = "(OI)(CI)F" if is_directory else "F"
        subprocess.run(
            [
                "icacls.exe",
                str(path),
                "/inheritance:r",
                "/grant:r",
                f"{account}:{grant_suffix}",
                f"SYSTEM:{grant_suffix}",
            ],
            check=True,
            capture_output=True,
            timeout=5,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        # Windows ACL updates can be unavailable in restricted deployments;
        # the caller still supplied a private settings directory and the
        # generated files are created with owner-only POSIX mode hints.
        return


def _build_certificate(
    private_key: ec.EllipticCurvePrivateKey,
    hostname: str,
    ip_addresses: tuple[str, ...],
) -> x509.Certificate:
    now = datetime.now(timezone.utc)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)])
    names: list[x509.GeneralName] = [
        x509.DNSName(hostname),
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        x509.IPAddress(ipaddress.ip_address("::1")),
    ]
    names.extend(x509.IPAddress(ipaddress.ip_address(value)) for value in ip_addresses)
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=None,
                decipher_only=None,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .sign(private_key, hashes.SHA256())
    )


def _write_atomic(path: Path, data: bytes) -> None:
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _secure_permissions(temp_path, is_directory=False)
        os.replace(temp_path, path)
        _secure_permissions(path, is_directory=False)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _write_material(
    directory: Path,
    hostname: str,
    ip_addresses: tuple[str, ...],
    *,
    replace_existing: bool,
) -> TLSMaterial:
    cert_path = directory / "server-cert.pem"
    key_path = directory / "server-key.pem"
    for path in (cert_path, key_path):
        if path.is_symlink():
            raise TLSAssetError(f"TLS asset path must not be a symbolic link: {path.name}")
        if path.exists() and not path.is_file():
            raise TLSAssetError(f"TLS asset path is not a regular file: {path.name}")
    if not replace_existing and (cert_path.exists() or key_path.exists()):
        raise TLSAssetError("Refusing to overwrite existing TLS assets")

    private_key = ec.generate_private_key(ec.SECP256R1())
    certificate = _build_certificate(private_key, hostname, ip_addresses)
    key_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    cert_bytes = certificate.public_bytes(serialization.Encoding.PEM)
    # Each file replacement is atomic. A crash between replacements is
    # detected as a key/certificate mismatch next time; it never silently
    # changes an already trusted certificate during normal startup.
    _write_atomic(key_path, key_bytes)
    _write_atomic(cert_path, cert_bytes)
    return _material_from_certificate(cert_path, key_path, certificate, hostname)


def _load_existing(directory: Path, hostname: str) -> TLSMaterial:
    cert_path = directory / "server-cert.pem"
    key_path = directory / "server-key.pem"
    for path in (cert_path, key_path):
        if path.is_symlink():
            raise TLSAssetError(f"TLS asset path must not be a symbolic link: {path.name}")
        if path.exists() and not path.is_file():
            raise TLSAssetError(f"TLS asset path is not a regular file: {path.name}")
    if not cert_path.exists() and not key_path.exists():
        return _write_material(directory, hostname, (), replace_existing=False)
    if not cert_path.exists() or not key_path.exists():
        raise TLSAssetError("TLS certificate and private key must both be present")
    if cert_path.stat().st_size > 128_000 or key_path.stat().st_size > 16_000:
        raise TLSAssetError("TLS asset exceeds the configured size limit")

    _secure_permissions(directory, is_directory=True)
    _secure_permissions(key_path, is_directory=False)
    _secure_permissions(cert_path, is_directory=False)
    cert_bytes = cert_path.read_bytes()
    key_bytes = key_path.read_bytes()
    try:
        certificate = x509.load_pem_x509_certificate(cert_bytes)
        private_key = serialization.load_pem_private_key(key_bytes, password=None)
    except (ValueError, TypeError) as exc:
        raise TLSAssetError("Stored TLS assets are corrupt or unsupported") from exc
    if not isinstance(private_key, ec.EllipticCurvePrivateKey) or not isinstance(
        private_key.curve, ec.SECP256R1
    ):
        raise TLSAssetError("Stored TLS private key is not the expected P-256 key")

    public_from_key = private_key.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    public_from_cert = certificate.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    if public_from_key != public_from_cert:
        raise TLSAssetError("Stored TLS certificate and private key do not match")
    if not isinstance(certificate.public_key(), ec.EllipticCurvePublicKey) or not isinstance(
        certificate.public_key().curve, ec.SECP256R1
    ):
        raise TLSAssetError("Stored TLS certificate does not use the expected P-256 key")
    if certificate.issuer != certificate.subject:
        raise TLSAssetError("Stored TLS certificate is not self-signed")
    try:
        certificate.public_key().verify(
            certificate.signature,
            certificate.tbs_certificate_bytes,
            ec.ECDSA(certificate.signature_hash_algorithm),
        )
    except Exception as exc:
        raise TLSAssetError("Stored TLS certificate signature is invalid") from exc

    try:
        basic_constraints = certificate.extensions.get_extension_for_class(
            x509.BasicConstraints
        ).value
        key_usage = certificate.extensions.get_extension_for_class(x509.KeyUsage).value
        extended_key_usage = certificate.extensions.get_extension_for_class(
            x509.ExtendedKeyUsage
        ).value
        sans = certificate.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
    except x509.ExtensionNotFound as exc:
        raise TLSAssetError("Stored TLS certificate is missing required extensions") from exc
    dns_names = set(sans.get_values_for_type(x509.DNSName))
    ip_names = set(sans.get_values_for_type(x509.IPAddress))
    common_names = certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    if basic_constraints.ca or key_usage.key_cert_sign or key_usage.crl_sign:
        raise TLSAssetError("Stored TLS certificate must be a non-CA leaf")
    if not key_usage.digital_signature or ExtendedKeyUsageOID.SERVER_AUTH not in extended_key_usage:
        raise TLSAssetError("Stored TLS certificate is not valid for server authentication")
    if hostname not in dns_names or "localhost" not in dns_names:
        raise TLSAssetError("Stored TLS certificate does not cover the stable DNS names")
    required_loopbacks = {
        ipaddress.ip_address("127.0.0.1"), ipaddress.ip_address("::1")
    }
    if not required_loopbacks.issubset(ip_names):
        raise TLSAssetError("Stored TLS certificate does not cover loopback addresses")
    if not common_names or common_names[0].value != hostname:
        raise TLSAssetError("Stored TLS certificate has the wrong hostname")

    now = datetime.now(timezone.utc)
    if certificate.not_valid_before_utc > now or certificate.not_valid_after_utc <= now:
        raise TLSAssetError("Stored TLS certificate is not currently valid; explicit renewal is required")
    validity = certificate.not_valid_after_utc - certificate.not_valid_before_utc
    if validity > timedelta(days=367):
        raise TLSAssetError("Stored TLS certificate lifetime exceeds one year")

    return _material_from_certificate(cert_path, key_path, certificate, hostname)


def _material_from_certificate(
    cert_path: Path,
    key_path: Path,
    certificate: x509.Certificate,
    hostname: str,
    *,
    requires_retrust: bool = False,
) -> TLSMaterial:
    sans = certificate.extensions.get_extension_for_class(
        x509.SubjectAlternativeName
    ).value
    ip_addresses = tuple(sorted(
        address.compressed for address in sans.get_values_for_type(x509.IPAddress)
    ))
    fingerprint = hashlib.sha256(
        certificate.public_bytes(serialization.Encoding.DER)
    ).hexdigest()
    return TLSMaterial(
        cert_path=cert_path,
        key_path=key_path,
        fingerprint_sha256=fingerprint,
        hostname=hostname,
        ip_addresses=ip_addresses,
        requires_retrust=requires_retrust,
    )


def ensure_tls(
    directory: str | os.PathLike[str],
    installation_id: str,
    *,
    ip_addresses: list[str] | None = None,
) -> TLSMaterial:
    """Create or validate the stable self-signed TLS identity.

    Supplied IP SANs are included only on initial creation. Existing valid
    material is reused if DHCP later changes an address; callers can inspect
    ``TLSMaterial.ip_addresses`` before advertising an optional IP alias.
    There is no automatic trust rotation.
    """
    hostname = hostname_for_installation_id(installation_id)
    addresses = _normalize_ips(ip_addresses)
    path = _credential_directory(directory)
    cert_path, key_path = path / "server-cert.pem", path / "server-key.pem"
    if not cert_path.exists() and not key_path.exists():
        return _write_material(path, hostname, addresses, replace_existing=False)
    return _load_existing(path, hostname)


def renew_tls(
    directory: str | os.PathLike[str],
    installation_id: str,
    *,
    ip_addresses: list[str] | None = None,
) -> TLSMaterial:
    """Explicitly rotate the leaf certificate and private key.

    The returned material sets ``requires_retrust=True`` when an earlier
    identity existed, because the public certificate fingerprint changes.
    """
    hostname = hostname_for_installation_id(installation_id)
    addresses = _normalize_ips(ip_addresses)
    path = _credential_directory(directory)
    existed = (path / "server-cert.pem").exists() or (path / "server-key.pem").exists()
    material = _write_material(path, hostname, addresses, replace_existing=True)
    if not existed:
        return material
    return TLSMaterial(
        cert_path=material.cert_path,
        key_path=material.key_path,
        fingerprint_sha256=material.fingerprint_sha256,
        hostname=material.hostname,
        ip_addresses=material.ip_addresses,
        requires_retrust=True,
    )


__all__ = [
    "TLSAssetError",
    "TLSMaterial",
    "ensure_tls",
    "hostname_for_installation_id",
    "renew_tls",
]
