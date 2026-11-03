"""Local OAuth and owner-session primitives for Maestro Agent Access.

This module deliberately keeps bearer credentials out of SQLite. Only SHA-256
digests are stored for owner codes, authorization codes, access tokens, and
refresh tokens. HTTP handlers should pass the installation's trusted resource
identifier to :class:`AuthStore`; this module never derives it from a request.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.parse import urlsplit


OWNER_CODE_TTL = 10 * 60
OWNER_SESSION_TTL = 30 * 24 * 60 * 60
ACCESS_TOKEN_TTL = 15 * 60
REFRESH_TOKEN_TTL = 30 * 24 * 60 * 60
AUTHORIZATION_REQUEST_TTL = 10 * 60
AUTHORIZATION_CODE_TTL = 5 * 60

_SCOPE_ORDER = ("observe", "generate", "manage", "admin")
_ALL_SCOPES = frozenset(_SCOPE_ORDER)
_SCOPE_TOKEN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_PKCE_CHALLENGE = re.compile(r"^[A-Za-z0-9_-]{43}$")
_PKCE_VERIFIER = re.compile(r"^[A-Za-z0-9._~-]{43,128}$")

# These limits cap unauthenticated resource use and long lived state. Expired
# rows are removed opportunistically on each operation.
_MAX_CLIENTS = 256
_MAX_CLIENT_REGISTRATIONS_PER_HOUR = 32
_INACTIVE_CLIENT_TTL = 7 * 24 * 60 * 60
_MAX_PENDING = 128
_MAX_AUTHORIZATION_CODES = 256
_MAX_OWNER_CODES = 8
_MAX_GRANTS = 4096
_MAX_ACCESS_TOKENS = 8192
_MAX_REFRESH_TOKENS = 8192
_MAX_LOCAL_CLIENTS = 256


class AuthError(Exception):
    """A safe authentication error suitable for conversion to an HTTP error."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = int(status)
        self.message = str(message)


class _CommitAuthError(AuthError):
    """Internal AuthError whose security-state changes must be committed."""


@dataclass(frozen=True)
class Principal:
    """Authenticated identity passed between the auth core and route layer."""

    id: str
    client_id: str
    scopes: tuple[str, ...]
    owner: bool = False


class AuthStore:
    """Thread-safe SQLite store for local owner and OAuth credentials.

    ``resource`` is the canonical audience configured by the trusted launcher
    or route layer. OAuth resource indicators are compared to it exactly.
    ``clock`` is injectable and should return epoch seconds (``datetime`` is
    also accepted for convenient tests).
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        clock: Callable[[], float | datetime] = time.time,
        resource: str | None = None,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db_path = self.directory / "agent-access.sqlite3"
        self._clock = clock
        self._lock = threading.RLock()
        self._resource = self._validate_resource(resource) if resource is not None else None
        self._initialize()
        if self._resource is not None:
            self.set_resource(self._resource)
        else:
            self._resource = self._read_configured_resource()

    @property
    def enabled(self) -> bool:
        """Compatibility alias for the remote HTTP/OAuth permission."""
        with self._connection() as conn:
            return self._meta_bool(conn, "enabled", default=False)

    def pinokio_proxy_key(self) -> bytes:
        """Private installation key for the native-to-controller HTTPS hop.

        This authenticates an origin envelope, never a user or an MCP grant.
        It is deliberately absent from public status and client configuration.
        """
        with self._transaction() as conn:
            value = self._meta(conn, "pinokio_proxy_key")
            if not value:
                value = secrets.token_hex(32)
                conn.execute("INSERT INTO meta(key, value) VALUES('pinokio_proxy_key', ?)", (value,))
            return bytes.fromhex(value)

    @property
    def local_enabled(self) -> bool:
        with self._connection() as conn:
            return self._meta_bool(conn, "local_enabled", default=False)

    @property
    def owner_configured(self) -> bool:
        with self._connection() as conn:
            return self._meta_bool(conn, "owner_configured", default=False)

    def mark_owner_configured(self) -> None:
        """Persist explicit owner enrollment without creating a browser session."""
        with self._transaction() as conn:
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('owner_configured', '1')")

    def set_resource(self, resource: str) -> None:
        """Set the trusted canonical resource audience for this installation."""
        canonical = self._validate_resource(resource)
        with self._transaction() as conn:
            existing = self._meta(conn, "resource")
            if existing and existing != canonical:
                raise AuthError(409, "The Agent Access resource is already configured")
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('resource', ?)", (canonical,))
            self._resource = canonical

    def status(self) -> dict[str, Any]:
        """Return non-secret installation and grant status for the local UI."""
        now = self._now()
        with self._transaction() as conn:
            self._cleanup(conn, now)
            counts = conn.execute(
                "SELECT SUM(CASE WHEN revoked_at IS NULL THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN revoked_at IS NOT NULL THEN 1 ELSE 0 END) FROM grants"
            ).fetchone()
            return {
                "installation_id": self._meta(conn, "installation_id"),
                "owner_configured": self._meta_bool(conn, "owner_configured", default=False),
                "enabled": self._meta_bool(conn, "enabled", default=False),
                "remote_enabled": self._meta_bool(conn, "enabled", default=False),
                "local_enabled": self._meta_bool(conn, "local_enabled", default=False),
                "grants": {
                    "active": int(counts[0] or 0),
                    "revoked": int(counts[1] or 0),
                },
            }

    def set_enabled(self, enabled: bool, ownerPrincipal: Principal) -> None:
        self._require_owner(ownerPrincipal)
        if not isinstance(enabled, bool):
            raise AuthError(400, "enabled must be a boolean")
        with self._transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('enabled', ?)",
                ("1" if enabled else "0",),
            )

    def set_local_enabled(self, enabled: bool, ownerPrincipal: Principal) -> None:
        self._require_owner(ownerPrincipal)
        if not isinstance(enabled, bool):
            raise AuthError(400, "local_enabled must be a boolean")
        with self._transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('local_enabled', ?)",
                ("1" if enabled else "0",),
            )

    def create_local_client(
        self,
        name: str,
        scopes: str | Sequence[str],
        owner: Principal,
    ) -> tuple[dict[str, Any], str]:
        """Create a scoped stdio grant and return its opaque credential once.

        Callers must immediately persist the credential in the private
        installation directory. It is never stored in SQLite or exposed by a
        listing/status method.
        """
        self._require_owner(owner)
        if not isinstance(name, str) or not name.strip() or len(name) > 128 or any(ord(c) < 32 for c in name):
            raise AuthError(400, "Use a client name between 1 and 128 characters")
        try:
            name.encode("utf-8")
        except UnicodeEncodeError as error:
            raise AuthError(400, "Use a valid Unicode client name") from error
        selected = _parse_scopes(scopes)
        client_id = str(uuid.uuid4())
        grant_id = str(uuid.uuid4())
        credential = secrets.token_urlsafe(48)
        now = self._now()
        with self._transaction() as conn:
            self._cleanup(conn, now)
            active = int(conn.execute(
                "SELECT COUNT(*) FROM grants WHERE transport = 'local' AND revoked_at IS NULL"
            ).fetchone()[0])
            if active >= _MAX_LOCAL_CLIENTS:
                raise AuthError(429, "Too many active local clients")
            conn.execute(
                "INSERT INTO clients(client_id, client_name, redirect_uris, created_at, transport) "
                "VALUES(?, ?, '[]', ?, 'local')",
                (client_id, name.strip(), now),
            )
            conn.execute(
                "INSERT INTO grants(grant_id, client_id, scopes, resource, created_at, revoked_at, transport) "
                "VALUES(?, ?, ?, ?, ?, NULL, 'local')",
                (grant_id, client_id, json.dumps(selected), self._meta(conn, "resource") or "local-stdio", now),
            )
            conn.execute(
                "INSERT INTO local_credentials(client_id, token_hash, created_at) VALUES(?, ?, ?)",
                (client_id, _digest(credential), now),
            )
        return {
            "id": client_id,
            "name": name.strip(),
            "scopes": selected,
            "grant_id": grant_id,
            "revoked": False,
            "created_at": now,
        }, credential

    def list_local_clients(self, owner: Principal) -> list[dict[str, Any]]:
        self._require_owner(owner)
        with self._transaction() as conn:
            self._cleanup(conn, self._now())
            rows = conn.execute(
                "SELECT c.client_id, c.client_name, g.grant_id, g.scopes, g.created_at, g.revoked_at "
                "FROM clients c JOIN grants g ON g.client_id = c.client_id "
                "WHERE c.transport = 'local' AND g.transport = 'local' "
                "ORDER BY g.created_at DESC, c.client_id"
            ).fetchall()
            return [
                {
                    "id": str(row[0]),
                    "name": str(row[1]),
                    "grant_id": str(row[2]),
                    "scopes": _ordered_scopes(json.loads(row[3])),
                    "created_at": float(row[4]),
                    "revoked": row[5] is not None,
                }
                for row in rows
            ]

    def authenticate_local(self, token: str, client_id: str | None = None) -> Principal:
        """Authenticate a local stdio credential against current grant state."""
        if not isinstance(token, str) or not token or len(token) > 512:
            raise AuthError(401, "Invalid local client credential")
        token_hash = _digest(token)
        with self._connection() as conn:
            if not self._meta_bool(conn, "local_enabled", default=False):
                raise AuthError(401, "Local Agent Access is disabled")
            row = conn.execute(
                "SELECT c.client_id, g.grant_id, g.scopes, g.revoked_at, g.transport, lc.token_hash "
                "FROM local_credentials lc JOIN clients c ON c.client_id = lc.client_id "
                "JOIN grants g ON g.client_id = c.client_id "
                "WHERE lc.token_hash = ? AND c.transport = 'local' AND g.transport = 'local'",
                (token_hash,),
            ).fetchone()
            if (
                row is None
                or row[3] is not None
                or (client_id is not None and str(row[0]) != client_id)
                or not secrets.compare_digest(str(row[5]), token_hash)
            ):
                raise AuthError(401, "Invalid, revoked, or disabled local client")
            return Principal(str(row[1]), str(row[0]), tuple(_ordered_scopes(json.loads(row[2]))), False)

    def issue_owner_code(self) -> str:
        """Create a ten minute bootstrap/login code for private console output."""
        now = self._now()
        code = secrets.token_urlsafe(32)
        with self._transaction() as conn:
            self._cleanup(conn, now)
            count = conn.execute("SELECT COUNT(*) FROM owner_codes").fetchone()[0]
            if count >= _MAX_OWNER_CODES:
                conn.execute(
                    "DELETE FROM owner_codes WHERE code_hash IN "
                    "(SELECT code_hash FROM owner_codes ORDER BY created_at ASC LIMIT ?)",
                    (count - _MAX_OWNER_CODES + 1,),
                )
            conn.execute(
                "INSERT INTO owner_codes(code_hash, created_at, expires_at) VALUES(?, ?, ?)",
                (_digest(code), now, now + OWNER_CODE_TTL),
            )
        return code

    def redeem_owner_code(self, code: str) -> dict[str, Any]:
        if not isinstance(code, str) or not code or len(code) > 256:
            raise AuthError(401, "Invalid or expired owner code")
        now = self._now()
        access_token = secrets.token_urlsafe(32)
        principal = _owner_principal()
        with self._transaction() as conn:
            self._cleanup(conn, now)
            row = conn.execute(
                "SELECT expires_at FROM owner_codes WHERE code_hash = ?",
                (_digest(code),),
            ).fetchone()
            if row is None or float(row[0]) <= now:
                raise AuthError(401, "Invalid or expired owner code")
            conn.execute("DELETE FROM owner_codes WHERE code_hash = ?", (_digest(code),))
            return self._issue_owner_session(conn, now, access_token, principal)

    def issue_owner_session(self) -> dict[str, Any]:
        """Create an owner session for an already-validated server-side flow.

        Unlike :meth:`issue_owner_code`, this does not create a redeemable
        credential that can be copied or exchanged through another route.
        Callers must validate the authorization decision before using it.
        """
        now = self._now()
        access_token = secrets.token_urlsafe(32)
        principal = _owner_principal()
        with self._transaction() as conn:
            self._cleanup(conn, now)
            return self._issue_owner_session(conn, now, access_token, principal)

    def _issue_owner_session(self, conn, now: float, access_token: str, principal: Principal) -> dict[str, Any]:
        self._insert_access_token(
            conn,
            token=access_token,
            kind="owner",
            principal_id=principal.id,
            client_id=principal.client_id,
            scopes=principal.scopes,
            expires_at=now + OWNER_SESSION_TTL,
            family_id=None,
        )
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('owner_configured', '1')")
        return {
            "access_token": access_token,
            "expires_in": OWNER_SESSION_TTL,
            "principal": principal,
        }

    def authenticate(self, token: str) -> Principal:
        if not isinstance(token, str) or not token or len(token) > 512:
            raise AuthError(401, "Invalid or expired access token")
        now = self._now()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT kind, principal_id, client_id, scopes, expires_at, revoked_at "
                "FROM access_tokens WHERE token_hash = ?",
                (_digest(token),),
            ).fetchone()
            if row is None or row[5] is not None or float(row[4]) <= now:
                raise AuthError(401, "Invalid or expired access token")
            kind, principal_id, client_id, scope_json = row[:4]
            if kind == "owner":
                return _owner_principal()
            enabled = self._meta_bool(conn, "enabled", default=False)
            grant = conn.execute(
                "SELECT scopes, revoked_at, transport FROM grants WHERE grant_id = ?",
                (principal_id,),
            ).fetchone()
            if not enabled or grant is None or grant[1] is not None or grant[2] != "remote":
                raise AuthError(401, "Access grant is revoked or disabled")
            # Read current grant scopes to make any future grant-level reduction
            # take effect for already-issued access tokens.
            scopes = tuple(_ordered_scopes(json.loads(grant[0])))
            return Principal(str(principal_id), str(client_id), scopes, False)

    def list_grants(self, owner: Principal) -> list[dict[str, Any]]:
        self._require_owner(owner)
        now = self._now()
        with self._transaction() as conn:
            self._cleanup(conn, now)
            rows = conn.execute(
                "SELECT g.grant_id, g.client_id, c.client_name, g.scopes, g.resource, "
                "g.created_at, g.revoked_at, g.transport FROM grants g "
                "LEFT JOIN clients c ON c.client_id = g.client_id "
                "ORDER BY g.created_at DESC, g.grant_id"
            ).fetchall()
            return [
                {
                    "grant_id": row[0],
                    "client_id": row[1],
                    "client_name": row[2] or "",
                    "scopes": _ordered_scopes(json.loads(row[3])),
                    "resource": row[4],
                    "created_at": float(row[5]),
                    "revoked": row[6] is not None,
                    "revoked_at": float(row[6]) if row[6] is not None else None,
                    "transport": str(row[7]),
                }
                for row in rows
            ]

    def revoke_grant(self, grant_id: str, owner: Principal) -> bool:
        self._require_owner(owner)
        if not isinstance(grant_id, str) or not grant_id or len(grant_id) > 128:
            raise AuthError(400, "Invalid grant id")
        now = self._now()
        with self._transaction() as conn:
            row = conn.execute("SELECT grant_id FROM grants WHERE grant_id = ?", (grant_id,)).fetchone()
            if row is None:
                return False
            conn.execute("UPDATE grants SET revoked_at = COALESCE(revoked_at, ?) WHERE grant_id = ?", (now, grant_id))
            conn.execute("UPDATE refresh_tokens SET revoked_at = COALESCE(revoked_at, ?) WHERE grant_id = ?", (now, grant_id))
            conn.execute("UPDATE access_tokens SET revoked_at = COALESCE(revoked_at, ?) WHERE principal_id = ? AND kind = 'agent'", (now, grant_id))
            return True

    def logout(self, token: str) -> bool:
        if not isinstance(token, str) or not token or len(token) > 512:
            return False
        token_hash = _digest(token)
        now = self._now()
        with self._transaction() as conn:
            cur = conn.execute(
                "UPDATE access_tokens SET revoked_at = COALESCE(revoked_at, ?) WHERE token_hash = ?",
                (now, token_hash),
            )
            if cur.rowcount:
                return True
            row = conn.execute(
                "SELECT family_id FROM refresh_tokens WHERE token_hash = ?",
                (token_hash,),
            ).fetchone()
            if row is None:
                return False
            self._revoke_family(conn, str(row[0]), now)
            return True

    def register_client(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(metadata, Mapping):
            raise AuthError(400, "Client metadata must be an object")
        redirects = metadata.get("redirect_uris")
        if not isinstance(redirects, (list, tuple)) or not redirects or len(redirects) > 20:
            raise AuthError(400, "redirect_uris must contain between 1 and 20 URIs")
        safe_redirects: list[str] = []
        for uri in redirects:
            if not isinstance(uri, str) or len(uri) > 2048 or not _safe_redirect_uri(uri):
                raise AuthError(400, "Redirect URI must use HTTPS or loopback HTTP without userinfo, fragments, or wildcards")
            if uri not in safe_redirects:
                safe_redirects.append(uri)
        client_name = metadata.get("client_name", "")
        if client_name is None:
            client_name = ""
        if not isinstance(client_name, str) or len(client_name) > 128 or any(ord(char) < 32 for char in client_name):
            raise AuthError(400, "client_name must be a plain string of at most 128 characters")
        now = self._now()
        client_id = "ma_" + secrets.token_urlsafe(24)
        with self._transaction() as conn:
            self._cleanup(conn, now)
            self._require_enabled(conn)
            recent_registrations = int(conn.execute(
                "SELECT COUNT(*) FROM client_registration_events WHERE created_at > ?",
                (now - 60 * 60,),
            ).fetchone()[0])
            if recent_registrations >= _MAX_CLIENT_REGISTRATIONS_PER_HOUR:
                raise AuthError(429, "Client registration rate limit reached; try again later")
            count = int(conn.execute("SELECT COUNT(*) FROM clients").fetchone()[0])
            if count >= _MAX_CLIENTS:
                self._reclaim_inactive_clients(conn, now, limit=count - _MAX_CLIENTS + 1)
                count = int(conn.execute("SELECT COUNT(*) FROM clients").fetchone()[0])
            if count >= _MAX_CLIENTS:
                raise AuthError(429, "Client registration limit reached; remove an unused client or wait for inactive entries to expire")
            conn.execute(
                "INSERT INTO clients(client_id, client_name, redirect_uris, created_at) VALUES(?, ?, ?, ?)",
                (client_id, client_name, json.dumps(safe_redirects), now),
            )
            conn.execute("INSERT INTO client_registration_events(created_at) VALUES(?)", (now,))
            return {
                "client_id": client_id,
                "client_name": client_name,
                "redirect_uris": safe_redirects,
                "client_id_issued_at": int(now),
                "token_endpoint_auth_method": "none",
            }

    def get_client(self, client_id: str) -> dict[str, Any]:
        if not isinstance(client_id, str) or not client_id or len(client_id) > 128:
            raise AuthError(404, "Unknown client")
        with self._connection() as conn:
            row = conn.execute(
                "SELECT client_id, client_name, redirect_uris, created_at FROM clients WHERE client_id = ?",
                (client_id,),
            ).fetchone()
            if row is None:
                raise AuthError(404, "Unknown client")
            return {
                "client_id": row[0],
                "client_name": row[1],
                "redirect_uris": json.loads(row[2]),
                "client_id_issued_at": int(row[3]),
                "token_endpoint_auth_method": "none",
            }

    def create_authorization(
        self,
        client_id: str,
        redirect_uri: str,
        scope: str | Sequence[str],
        code_challenge: str,
        resource: str,
        state: str | None,
    ) -> dict[str, Any]:
        if not isinstance(code_challenge, str) or not _PKCE_CHALLENGE.fullmatch(code_challenge):
            raise AuthError(400, "A valid S256 PKCE code challenge is required")
        if state is not None and (not isinstance(state, str) or len(state) > 1024):
            raise AuthError(400, "state must be a string of at most 1024 characters")
        scopes = _parse_scopes(scope)
        now = self._now()
        request_id = secrets.token_urlsafe(24)
        with self._transaction() as conn:
            self._cleanup(conn, now)
            self._require_enabled(conn)
            self._require_resource(conn, resource)
            client = self._client_row(conn, client_id)
            if not any(_redirect_matches(registered, redirect_uri) for registered in json.loads(client[2])):
                raise AuthError(400, "redirect_uri is not registered for this client")
            count = int(conn.execute("SELECT COUNT(*) FROM pending_authorizations").fetchone()[0])
            if count >= _MAX_PENDING:
                raise AuthError(429, "Too many authorization requests are waiting for approval")
            conn.execute(
                "INSERT INTO pending_authorizations(request_id, client_id, redirect_uri, scopes, "
                "code_challenge, resource, state, created_at, expires_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    request_id,
                    client_id,
                    redirect_uri,
                    json.dumps(scopes),
                    code_challenge,
                    resource,
                    state,
                    now,
                    now + AUTHORIZATION_REQUEST_TTL,
                ),
            )
            return self._pending_public(conn, request_id)

    def get_authorization(self, request_id: str) -> dict[str, Any]:
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise AuthError(404, "Authorization request not found")
        now = self._now()
        with self._transaction() as conn:
            self._cleanup(conn, now)
            row = conn.execute("SELECT expires_at FROM pending_authorizations WHERE request_id = ?", (request_id,)).fetchone()
            if row is None:
                raise AuthError(404, "Authorization request not found")
            if float(row[0]) <= now:
                conn.execute("DELETE FROM pending_authorizations WHERE request_id = ?", (request_id,))
                raise AuthError(410, "Authorization request expired")
            return self._pending_public(conn, request_id)

    def approve_authorization(
        self,
        request_id: str,
        owner: Principal,
        approved_scopes: str | Sequence[str],
    ) -> dict[str, Any]:
        self._require_owner(owner)
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise AuthError(404, "Authorization request not found")
        selected = _parse_scopes(approved_scopes)
        now = self._now()
        code = secrets.token_urlsafe(32)
        with self._transaction() as conn:
            self._cleanup(conn, now)
            row = conn.execute(
                "SELECT client_id, redirect_uri, scopes, code_challenge, resource, state, expires_at "
                "FROM pending_authorizations WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is None:
                raise AuthError(404, "Authorization request not found")
            if float(row[6]) <= now:
                conn.execute("DELETE FROM pending_authorizations WHERE request_id = ?", (request_id,))
                raise AuthError(410, "Authorization request expired")
            requested = set(json.loads(row[2]))
            if not set(selected).issubset(requested):
                raise AuthError(403, "Approved scopes must be a subset of requested scopes")
            if not selected:
                raise AuthError(400, "At least one scope must be approved")
            if int(conn.execute("SELECT COUNT(*) FROM authorization_codes").fetchone()[0]) >= _MAX_AUTHORIZATION_CODES:
                raise AuthError(429, "Too many unredeemed authorization codes")
            conn.execute(
                "INSERT INTO authorization_codes(code_hash, client_id, redirect_uri, scopes, code_challenge, "
                "resource, state, created_at, expires_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _digest(code),
                    row[0],
                    row[1],
                    json.dumps(selected),
                    row[3],
                    row[4],
                    row[5],
                    now,
                    now + AUTHORIZATION_CODE_TTL,
                ),
            )
            conn.execute("DELETE FROM pending_authorizations WHERE request_id = ?", (request_id,))
            return {"redirect_uri": row[1], "code": code, "state": row[5]}

    def deny_authorization(self, request_id: str, owner: Principal) -> dict[str, Any]:
        self._require_owner(owner)
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise AuthError(404, "Authorization request not found")
        now = self._now()
        with self._transaction() as conn:
            self._cleanup(conn, now)
            row = conn.execute(
                "SELECT redirect_uri, state, expires_at FROM pending_authorizations WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is None:
                raise AuthError(404, "Authorization request not found")
            conn.execute("DELETE FROM pending_authorizations WHERE request_id = ?", (request_id,))
            return {"redirect_uri": row[0], "state": row[1], "error": "access_denied"}

    def exchange_code(
        self,
        client_id: str,
        code: str,
        redirect_uri: str,
        code_verifier: str,
        resource: str,
    ) -> dict[str, Any]:
        if not isinstance(code, str) or not code or len(code) > 512:
            raise AuthError(400, "Invalid authorization code")
        if not isinstance(code_verifier, str) or not _PKCE_VERIFIER.fullmatch(code_verifier):
            raise AuthError(400, "Invalid PKCE code verifier")
        if not _safe_redirect_uri(redirect_uri):
            raise AuthError(400, "Invalid redirect_uri")
        now = self._now()
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(48)
        family_id = secrets.token_urlsafe(24)
        with self._transaction() as conn:
            self._cleanup(conn, now)
            self._require_enabled(conn)
            self._require_resource(conn, resource)
            self._client_row(conn, client_id)
            auth = conn.execute(
                "SELECT client_id, redirect_uri, scopes, code_challenge, resource, expires_at "
                "FROM authorization_codes WHERE code_hash = ?",
                (_digest(code),),
            ).fetchone()
            if auth is None or float(auth[5]) <= now:
                raise AuthError(400, "Invalid or expired authorization code")
            expected = _b64url(hashlib.sha256(code_verifier.encode("ascii")).digest())
            if auth[0] != client_id or auth[1] != redirect_uri or auth[4] != resource:
                raise AuthError(400, "Authorization code binding does not match")
            if not secrets.compare_digest(str(auth[3]), expected):
                raise AuthError(400, "PKCE verification failed")
            if int(conn.execute("SELECT COUNT(*) FROM grants").fetchone()[0]) >= _MAX_GRANTS:
                raise AuthError(429, "Agent grant limit reached")
            if int(conn.execute("SELECT COUNT(*) FROM refresh_tokens").fetchone()[0]) >= _MAX_REFRESH_TOKENS:
                raise AuthError(429, "Refresh token limit reached")
            if int(conn.execute("SELECT COUNT(*) FROM access_tokens").fetchone()[0]) >= _MAX_ACCESS_TOKENS:
                raise AuthError(429, "Access token limit reached")
            grant_id = "gr_" + secrets.token_urlsafe(24)
            scopes = tuple(_ordered_scopes(json.loads(auth[2])))
            conn.execute(
                "INSERT INTO grants(grant_id, client_id, scopes, resource, created_at, revoked_at) VALUES(?, ?, ?, ?, ?, NULL)",
                (grant_id, client_id, json.dumps(scopes), resource, now),
            )
            self._insert_access_token(
                conn,
                token=access_token,
                kind="agent",
                principal_id=grant_id,
                client_id=client_id,
                scopes=scopes,
                expires_at=now + ACCESS_TOKEN_TTL,
                family_id=family_id,
            )
            conn.execute(
                "INSERT INTO refresh_tokens(token_hash, client_id, grant_id, resource, family_id, scopes, "
                "expires_at, used_at, revoked_at) VALUES(?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
                (
                    _digest(refresh_token),
                    client_id,
                    grant_id,
                    resource,
                    family_id,
                    json.dumps(scopes),
                    now + REFRESH_TOKEN_TTL,
                ),
            )
            conn.execute("DELETE FROM authorization_codes WHERE code_hash = ?", (_digest(code),))
            return _token_response(access_token, refresh_token, scopes)

    def refresh_token(self, client_id: str, refresh_token: str, resource: str) -> dict[str, Any]:
        if not isinstance(refresh_token, str) or not refresh_token or len(refresh_token) > 512:
            raise AuthError(401, "Invalid or expired refresh token")
        now = self._now()
        next_access = secrets.token_urlsafe(32)
        next_refresh = secrets.token_urlsafe(48)
        with self._transaction() as conn:
            self._cleanup(conn, now)
            self._require_enabled(conn)
            self._require_resource(conn, resource)
            self._client_row(conn, client_id)
            row = conn.execute(
                "SELECT client_id, grant_id, resource, family_id, scopes, expires_at, used_at, revoked_at "
                "FROM refresh_tokens WHERE token_hash = ?",
                (_digest(refresh_token),),
            ).fetchone()
            if row is None or float(row[5]) <= now:
                raise AuthError(401, "Invalid or expired refresh token")
            if row[6] is not None:
                self._revoke_family(conn, str(row[3]), now)
                raise _CommitAuthError(401, "Refresh token reuse detected; token family revoked")
            if row[7] is not None or row[0] != client_id or row[2] != resource:
                raise AuthError(401, "Invalid or revoked refresh token")
            grant = conn.execute("SELECT revoked_at FROM grants WHERE grant_id = ?", (row[1],)).fetchone()
            if grant is None or grant[0] is not None:
                self._revoke_family(conn, str(row[3]), now)
                raise _CommitAuthError(401, "Access grant is revoked")
            if int(conn.execute("SELECT COUNT(*) FROM access_tokens").fetchone()[0]) >= _MAX_ACCESS_TOKENS:
                raise AuthError(429, "Access token limit reached")
            if int(conn.execute("SELECT COUNT(*) FROM refresh_tokens").fetchone()[0]) >= _MAX_REFRESH_TOKENS:
                raise AuthError(429, "Refresh token limit reached")
            conn.execute("UPDATE refresh_tokens SET used_at = ? WHERE token_hash = ?", (now, _digest(refresh_token)))
            scopes = tuple(_ordered_scopes(json.loads(row[4])))
            next_refresh_expiry = now + REFRESH_TOKEN_TTL
            # Retain every consumed digest until the family expires so replay
            # remains detectable after a later rotation extends the session.
            conn.execute(
                "UPDATE refresh_tokens SET expires_at = MAX(expires_at, ?) WHERE family_id = ?",
                (next_refresh_expiry, row[3]),
            )
            self._insert_access_token(
                conn,
                token=next_access,
                kind="agent",
                principal_id=str(row[1]),
                client_id=client_id,
                scopes=scopes,
                expires_at=now + ACCESS_TOKEN_TTL,
                family_id=str(row[3]),
            )
            conn.execute(
                "INSERT INTO refresh_tokens(token_hash, client_id, grant_id, resource, family_id, scopes, "
                "expires_at, used_at, revoked_at) VALUES(?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
                (
                    _digest(next_refresh),
                    client_id,
                    row[1],
                    resource,
                    row[3],
                    json.dumps(scopes),
                    next_refresh_expiry,
                ),
            )
            return _token_response(next_access, next_refresh, scopes)

    def _initialize(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE IF NOT EXISTS meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS owner_codes (
                        code_hash TEXT PRIMARY KEY,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS clients (
                        client_id TEXT PRIMARY KEY,
                        client_name TEXT NOT NULL,
                        redirect_uris TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS client_registration_events (
                        event_id INTEGER PRIMARY KEY,
                        created_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS pending_authorizations (
                        request_id TEXT PRIMARY KEY,
                        client_id TEXT NOT NULL REFERENCES clients(client_id),
                        redirect_uri TEXT NOT NULL,
                        scopes TEXT NOT NULL,
                        code_challenge TEXT NOT NULL,
                        resource TEXT NOT NULL,
                        state TEXT,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS authorization_codes (
                        code_hash TEXT PRIMARY KEY,
                        client_id TEXT NOT NULL REFERENCES clients(client_id),
                        redirect_uri TEXT NOT NULL,
                        scopes TEXT NOT NULL,
                        code_challenge TEXT NOT NULL,
                        resource TEXT NOT NULL,
                        state TEXT,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS grants (
                        grant_id TEXT PRIMARY KEY,
                        client_id TEXT NOT NULL REFERENCES clients(client_id),
                        scopes TEXT NOT NULL,
                        resource TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        revoked_at REAL,
                        transport TEXT NOT NULL DEFAULT 'remote'
                    );
                    CREATE TABLE IF NOT EXISTS access_tokens (
                        token_hash TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        principal_id TEXT NOT NULL,
                        client_id TEXT NOT NULL,
                        scopes TEXT NOT NULL,
                        expires_at REAL NOT NULL,
                        revoked_at REAL,
                        family_id TEXT
                    );
                    CREATE TABLE IF NOT EXISTS refresh_tokens (
                        token_hash TEXT PRIMARY KEY,
                        client_id TEXT NOT NULL,
                        grant_id TEXT NOT NULL,
                        resource TEXT NOT NULL,
                        family_id TEXT NOT NULL,
                        scopes TEXT NOT NULL,
                        expires_at REAL NOT NULL,
                        used_at REAL,
                        revoked_at REAL
                    );
                    CREATE TABLE IF NOT EXISTS local_credentials (
                        client_id TEXT PRIMARY KEY REFERENCES clients(client_id) ON DELETE CASCADE,
                        token_hash TEXT NOT NULL UNIQUE,
                        created_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS pending_expiry ON pending_authorizations(expires_at);
                    CREATE INDEX IF NOT EXISTS auth_code_expiry ON authorization_codes(expires_at);
                    CREATE INDEX IF NOT EXISTS access_expiry ON access_tokens(expires_at);
                    CREATE INDEX IF NOT EXISTS refresh_expiry ON refresh_tokens(expires_at);
                    CREATE INDEX IF NOT EXISTS grants_created ON grants(created_at);
                    CREATE INDEX IF NOT EXISTS client_registrations_created
                        ON client_registration_events(created_at);
                    """
                )
                # Existing OAuth clients and grants are explicitly remote. The
                # additive columns preserve all current registrations/tokens.
                for table in ("clients", "grants"):
                    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                    if "transport" not in columns:
                        conn.execute(
                            f"ALTER TABLE {table} ADD COLUMN transport TEXT NOT NULL DEFAULT 'remote'"
                        )
                conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('installation_id', ?)", (str(uuid.uuid4()),))
                conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('enabled', '0')")
                conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('local_enabled', '0')")
                conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('owner_configured', '0')")
                conn.commit()
            finally:
                conn.close()
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            # Windows ACLs are inherited from the caller-supplied settings directory.
            pass

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 15000")
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._connect()
            try:
                yield conn
            finally:
                conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.commit()
            except _CommitAuthError:
                conn.commit()
                raise
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _now(self) -> float:
        value = self._clock()
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.timestamp()
        return float(value)

    def _cleanup(self, conn: sqlite3.Connection, now: float) -> None:
        conn.execute("DELETE FROM owner_codes WHERE expires_at <= ?", (now,))
        conn.execute("DELETE FROM pending_authorizations WHERE expires_at <= ?", (now,))
        conn.execute("DELETE FROM authorization_codes WHERE expires_at <= ?", (now,))
        conn.execute("DELETE FROM refresh_tokens WHERE expires_at <= ?", (now,))
        conn.execute("DELETE FROM access_tokens WHERE expires_at <= ? OR revoked_at IS NOT NULL", (now,))
        conn.execute("DELETE FROM client_registration_events WHERE created_at <= ?", (now - 60 * 60,))
        self._reclaim_inactive_clients(conn, now, older_than=now - _INACTIVE_CLIENT_TTL)

    @staticmethod
    def _reclaim_inactive_clients(
        conn: sqlite3.Connection,
        now: float,
        *,
        older_than: float | None = None,
        limit: int | None = None,
    ) -> int:
        """Remove stale DCR records that have no live approval or grant state.

        At the client cap, callers may evict the oldest unused registrations
        immediately. Routine cleanup only removes registrations older than the
        retention window. Active grants and in-flight OAuth requests are kept.
        """
        cutoff = "" if older_than is None else "AND c.created_at <= ?"
        parameters: list[Any] = [] if older_than is None else [older_than]
        query = (
            "SELECT c.client_id FROM clients c WHERE 1=1 " + cutoff + " "
            "AND NOT EXISTS (SELECT 1 FROM grants g WHERE g.client_id = c.client_id AND g.revoked_at IS NULL) "
            "AND NOT EXISTS (SELECT 1 FROM pending_authorizations p WHERE p.client_id = c.client_id) "
            "AND NOT EXISTS (SELECT 1 FROM authorization_codes a WHERE a.client_id = c.client_id) "
            "AND NOT EXISTS (SELECT 1 FROM access_tokens t WHERE t.client_id = c.client_id "
            "AND t.revoked_at IS NULL AND t.expires_at > ?) "
            "AND NOT EXISTS (SELECT 1 FROM refresh_tokens r WHERE r.client_id = c.client_id "
            "AND r.revoked_at IS NULL AND r.expires_at > ?) "
            "ORDER BY c.created_at ASC, c.client_id ASC"
        )
        parameters.extend((now, now))
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(max(0, int(limit)))
        client_ids = [row[0] for row in conn.execute(query, parameters).fetchall()]
        for client_id in client_ids:
            conn.execute("DELETE FROM refresh_tokens WHERE client_id = ?", (client_id,))
            conn.execute("DELETE FROM access_tokens WHERE client_id = ?", (client_id,))
            conn.execute("DELETE FROM authorization_codes WHERE client_id = ?", (client_id,))
            conn.execute("DELETE FROM pending_authorizations WHERE client_id = ?", (client_id,))
            conn.execute("DELETE FROM grants WHERE client_id = ?", (client_id,))
            conn.execute("DELETE FROM clients WHERE client_id = ?", (client_id,))
        return len(client_ids)

    def _insert_access_token(
        self,
        conn: sqlite3.Connection,
        *,
        token: str,
        kind: str,
        principal_id: str,
        client_id: str,
        scopes: Sequence[str],
        expires_at: float,
        family_id: str | None,
    ) -> None:
        conn.execute(
            "INSERT INTO access_tokens(token_hash, kind, principal_id, client_id, scopes, expires_at, revoked_at, family_id) "
            "VALUES(?, ?, ?, ?, ?, ?, NULL, ?)",
            (_digest(token), kind, principal_id, client_id, json.dumps(list(scopes)), expires_at, family_id),
        )

    def _pending_public(self, conn: sqlite3.Connection, request_id: str) -> dict[str, Any]:
        row = conn.execute(
            "SELECT p.request_id, p.client_id, c.client_name, p.redirect_uri, p.scopes, p.resource, p.state, "
            "p.created_at, p.expires_at FROM pending_authorizations p "
            "JOIN clients c ON c.client_id = p.client_id WHERE p.request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise AuthError(404, "Authorization request not found")
        return {
            "request_id": row[0],
            "client_id": row[1],
            "client_name": row[2],
            "redirect_uri": row[3],
            "scopes": _ordered_scopes(json.loads(row[4])),
            "resource": row[5],
            "state": row[6],
            "created_at": float(row[7]),
            "expires_at": float(row[8]),
        }

    def _client_row(self, conn: sqlite3.Connection, client_id: str) -> sqlite3.Row:
        if not isinstance(client_id, str) or not client_id or len(client_id) > 128:
            raise AuthError(401, "Unknown client")
        row = conn.execute(
            "SELECT client_id, client_name, redirect_uris, created_at FROM clients WHERE client_id = ?",
            (client_id,),
        ).fetchone()
        if row is None:
            raise AuthError(401, "Unknown client")
        return row

    def _require_enabled(self, conn: sqlite3.Connection) -> None:
        if not self._meta_bool(conn, "enabled", default=False):
            raise AuthError(403, "Agent Access is disabled")

    def _require_resource(self, conn: sqlite3.Connection, resource: str) -> None:
        configured = self._meta(conn, "resource")
        if not configured:
            raise AuthError(503, "Agent Access resource is not configured")
        if not isinstance(resource, str) or resource != configured:
            raise AuthError(400, "OAuth resource does not match this installation")

    def _read_configured_resource(self) -> str | None:
        with self._connection() as conn:
            return self._meta(conn, "resource")

    @staticmethod
    def _meta(conn: sqlite3.Connection, key: str) -> str | None:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row[0]) if row else None

    @classmethod
    def _meta_bool(cls, conn: sqlite3.Connection, key: str, *, default: bool) -> bool:
        value = cls._meta(conn, key)
        return default if value is None else value == "1"

    @staticmethod
    def _validate_resource(resource: str | None) -> str:
        if not isinstance(resource, str) or not resource or len(resource) > 2048 or any(ch.isspace() for ch in resource):
            raise AuthError(400, "resource must be a canonical absolute HTTP(S) URI")
        try:
            parsed = urlsplit(resource)
            _ = parsed.port
        except ValueError as exc:
            raise AuthError(400, "resource must be a canonical absolute HTTP(S) URI") from exc
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname or parsed.username is not None or parsed.password is not None or parsed.fragment or "#" in resource:
            raise AuthError(400, "resource must be a canonical absolute HTTP(S) URI")
        if "*" in resource:
            raise AuthError(400, "resource must be a canonical absolute HTTP(S) URI")
        return resource

    @staticmethod
    def _require_owner(owner: Principal) -> None:
        if not isinstance(owner, Principal) or not owner.owner:
            raise AuthError(403, "Owner session required")

    @staticmethod
    def _revoke_family(conn: sqlite3.Connection, family_id: str, now: float) -> None:
        conn.execute("UPDATE refresh_tokens SET revoked_at = COALESCE(revoked_at, ?) WHERE family_id = ?", (now, family_id))
        conn.execute("UPDATE access_tokens SET revoked_at = COALESCE(revoked_at, ?) WHERE family_id = ?", (now, family_id))


def _owner_principal() -> Principal:
    return Principal("owner", "owner-console", tuple(_ordered_scopes(_ALL_SCOPES)), True)


LOCAL_DEVICE_PRINCIPAL = Principal(
    "local-device", "local-device", tuple(_SCOPE_ORDER), True,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _ordered_scopes(scopes: Sequence[str] | set[str] | frozenset[str]) -> list[str]:
    selected = set(scopes)
    return [scope for scope in _SCOPE_ORDER if scope in selected]


def _parse_scopes(scope: str | Sequence[str]) -> list[str]:
    if isinstance(scope, str):
        values = scope.split()
    elif isinstance(scope, Sequence) and not isinstance(scope, (bytes, bytearray)):
        values = []
        for item in scope:
            if not isinstance(item, str):
                raise AuthError(400, "Scopes must be strings")
            values.extend(item.split())
    else:
        raise AuthError(400, "Scopes must be a space-delimited string or string list")
    if not values:
        raise AuthError(400, "At least one scope is required")
    selected: set[str] = set()
    for value in values:
        if not _SCOPE_TOKEN.fullmatch(value):
            raise AuthError(400, "Invalid scope")
        if value == "full":
            selected.update(_ALL_SCOPES)
        elif value in _ALL_SCOPES:
            selected.add(value)
        else:
            raise AuthError(400, f"Unknown scope: {value}")
    return _ordered_scopes(selected)


def _safe_redirect_uri(uri: str) -> bool:
    if not isinstance(uri, str) or not uri or len(uri) > 2048 or any(ord(ch) < 33 for ch in uri):
        return False
    if "*" in uri or "#" in uri:
        return False
    try:
        parsed = urlsplit(uri)
        port = parsed.port
    except ValueError:
        return False
    if port == 0:
        return False
    if parsed.username is not None or parsed.password is not None or not parsed.hostname or parsed.fragment:
        return False
    scheme = parsed.scheme.lower()
    if scheme == "https":
        return True
    if scheme != "http":
        return False
    host = parsed.hostname.lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _redirect_matches(registered: str, requested: str) -> bool:
    if registered == requested:
        return True
    if not _safe_redirect_uri(registered) or not _safe_redirect_uri(requested):
        return False
    try:
        registered_parts = urlsplit(registered)
        requested_parts = urlsplit(requested)
        registered_port = registered_parts.port
        requested_port = requested_parts.port
    except ValueError:
        return False
    host = (registered_parts.hostname or "").lower()
    if registered_parts.scheme.lower() != "http" or host not in {"localhost", "127.0.0.1", "::1"}:
        try:
            if not ipaddress.ip_address(host).is_loopback:
                return False
        except ValueError:
            return False
    # A registered port of 0 or no port marks a native loopback redirect as
    # variable-port. All other components remain an exact binding.
    if registered_port not in (None, 0) or requested_port in (None, 0):
        return False
    return (
        registered_parts.scheme.lower() == requested_parts.scheme.lower() == "http"
        and host == (requested_parts.hostname or "").lower()
        and registered_parts.path == requested_parts.path
        and registered_parts.query == requested_parts.query
    )


def _token_response(access_token: str, refresh_token: str, scopes: Sequence[str]) -> dict[str, Any]:
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in": ACCESS_TOKEN_TTL,
        "scope": " ".join(_ordered_scopes(scopes)),
    }
