"""Owner-scoped local MCP stdio credentials and transport."""
from __future__ import annotations

import os
import re
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path

from .auth import AuthError
from .context import credential_context, principal_context
from .private_permissions import protect_private_path


_CLIENT_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_CREDENTIAL = re.compile(r"^[A-Za-z0-9_-]{64}$")
_OBSERVE_TOOLS = frozenset({
    "get_status", "list_models", "get_model_options", "get_model_defaults",
    "help", "list_operations", "describe_operation", "get_job",
    "get_submission", "fetch_media", "list_workstations", "upload_asset_status",
    "get_creation_defaults", "get_creation_context", "get_authored_draft",
})
_GENERATE_TOOLS = frozenset({
    "prepare_generation", "prepare_authored_generation", "submit_authored_generation",
})
_MANAGE_TOOLS = frozenset({
    "upload_asset_begin", "upload_asset_chunk", "upload_asset_complete",
})


class LocalCredentialError(ValueError):
    """A local credential could not be stored or safely read."""

    status = 500


def validate_client_id(client_id: str) -> str:
    if not isinstance(client_id, str) or not _CLIENT_ID.fullmatch(client_id):
        raise LocalCredentialError("Invalid local client ID")
    try:
        if str(uuid.UUID(client_id)) != client_id:
            raise LocalCredentialError("Invalid local client ID")
    except (ValueError, AttributeError) as error:
        raise LocalCredentialError("Invalid local client ID") from error
    return client_id


def _is_link_or_reparse(path: Path) -> bool:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(details.st_mode):
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(details, "st_file_attributes", 0) & reparse)


def _protect_credential_path(path: Path, *, directory: bool) -> None:
    try:
        protect_private_path(path, directory=directory)
    except Exception as error:
        raise LocalCredentialError("Could not protect the local credential path") from error


def _credential_directory(settings_directory: str | os.PathLike[str], *, create: bool) -> Path:
    base = Path(settings_directory).absolute()
    if _is_link_or_reparse(base):
        raise LocalCredentialError("Agent Access settings directory must not be a link")
    if not base.exists() or not base.is_dir():
        raise LocalCredentialError("Agent Access settings directory is unavailable")
    real_base = base.resolve(strict=True)
    directory = base / "local-clients"
    if _is_link_or_reparse(directory):
        raise LocalCredentialError("Local credential directory must not be a link")
    created = False
    if not directory.exists():
        if not create:
            raise LocalCredentialError("Local credential directory is unavailable")
        directory.mkdir(mode=0o700)
        created = True
    if _is_link_or_reparse(directory) or not directory.is_dir():
        raise LocalCredentialError("Local credential directory is invalid")
    real_directory = directory.resolve(strict=True)
    if real_directory.parent != real_base:
        raise LocalCredentialError("Local credential directory escapes Agent Access settings")
    try:
        _protect_credential_path(directory, directory=True)
    except LocalCredentialError:
        if created:
            try:
                directory.rmdir()
            except OSError:
                pass
        raise
    if os.name != "nt":
        try:
            directory.chmod(0o700)
        except OSError as error:
            raise LocalCredentialError("Local credential directory permissions are unavailable") from error
    return directory


def _credential_path(settings_directory: str | os.PathLike[str], client_id: str, *, create_dir: bool) -> Path:
    identifier = validate_client_id(client_id)
    return _credential_directory(settings_directory, create=create_dir) / f"{identifier}.credential"


def write_local_credential(
    settings_directory: str | os.PathLike[str],
    client_id: str,
    credential: str,
) -> Path:
    """Create a new private token file; never overwrite a path or follow links."""
    if not isinstance(credential, str) or not _CREDENTIAL.fullmatch(credential):
        raise LocalCredentialError("Invalid local credential")
    path = _credential_path(settings_directory, client_id, create_dir=True)
    if _is_link_or_reparse(path):
        raise LocalCredentialError("Local credential path must not be a link")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise LocalCredentialError("Could not create a private local credential") from error
    try:
        _protect_credential_path(path, directory=False)
        with os.fdopen(descriptor, "w", encoding="ascii", newline="") as handle:
            handle.write(credential)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            path.chmod(0o600)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return path


def read_local_credential(settings_directory: str | os.PathLike[str], client_id: str) -> str:
    """Read one credential by UUID from the installation-owned private store."""
    path = _credential_path(settings_directory, client_id, create_dir=False)
    if _is_link_or_reparse(path):
        raise LocalCredentialError("Local credential path must not be a link")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise LocalCredentialError("Local client credential is unavailable") from error
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise LocalCredentialError("Local credential must be a regular file")
        try:
            current = path.stat(follow_symlinks=False)
        except OSError as error:
            raise LocalCredentialError("Local credential path changed during access") from error
        if (details.st_dev, details.st_ino) != (current.st_dev, current.st_ino):
            raise LocalCredentialError("Local credential path changed during access")
        with os.fdopen(descriptor, "r", encoding="ascii", newline="") as handle:
            value = handle.read(513)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    if not _CREDENTIAL.fullmatch(value):
        raise LocalCredentialError("Local credential file is invalid")
    return value


def delete_local_credential(settings_directory: str | os.PathLike[str], client_id: str) -> bool:
    """Remove an owned token file without following a substituted link."""
    path = _credential_path(settings_directory, client_id, create_dir=False)
    if _is_link_or_reparse(path):
        raise LocalCredentialError("Local credential path must not be a link")
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def local_launch_configuration(app_root: str | os.PathLike[str], client_id: str) -> dict[str, object]:
    """Build a non-secret stdio command that references only a client ID."""
    identifier = validate_client_id(client_id)
    root = Path(app_root).resolve()
    interpreter = (
        root / "env-agent" / "Scripts" / "python.exe"
        if os.name == "nt"
        else root / "env-agent" / "bin" / "python"
    )
    script = root / "maestro_mcp.py"
    return {"command": str(interpreter), "args": [str(script), "--client", identifier]}


def _tool_scope(name: str, arguments: dict) -> str | None:
    if name in _OBSERVE_TOOLS:
        return "observe"
    if name in _GENERATE_TOOLS:
        return "generate"
    if name in _MANAGE_TOOLS:
        # Upload status is the one read-only upload action.
        return "manage"
    if name == "workstation_lifecycle":
        action = arguments.get("action")
        return "observe" if isinstance(action, str) and action.strip().casefold() == "status" else "admin"
    # execute_operation and transfer reads resolve the required scope from the
    # live operation or owned artifact in BackendClient.
    return None


def local_server_class(auth, client_id: str, credential: str):
    """Return an MCPServer subclass that rechecks local policy per request."""
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    identifier = validate_client_id(client_id)

    class AuthorizedLocalMCPServer(MCPServer):
        def _principal(self, required_scope: str | None = None):
            try:
                principal = auth.authenticate_local(credential, identifier)
            except AuthError as error:
                raise ToolError(error.message) from None
            if required_scope and required_scope not in principal.scopes:
                raise ToolError(f"This operation requires the {required_scope} scope")
            return principal

        @contextmanager
        def _request_identity(self, principal):
            principal_token = principal_context.set(principal)
            credential_token = credential_context.set(credential)
            try:
                yield
            finally:
                credential_context.reset(credential_token)
                principal_context.reset(principal_token)

        async def call_tool(self, name, arguments, context=None):
            required = _tool_scope(name, arguments or {})
            principal = self._principal(required)
            with self._request_identity(principal):
                return await super().call_tool(name, arguments, context)

        async def read_resource(self, uri, context=None):
            principal = self._principal("observe")
            with self._request_identity(principal):
                return await super().read_resource(uri, context)

        async def get_prompt(self, name, arguments=None, context=None):
            principal = self._principal("observe")
            with self._request_identity(principal):
                return await super().get_prompt(name, arguments, context)

    return AuthorizedLocalMCPServer


def create_local_mcp_server(backend, app_root, version, auth, client_id, credential):
    """Create Maestro's shared toolset with a per-call local authorization hook."""
    from .mcp_server import create_mcp_server

    return create_mcp_server(
        backend,
        Path(app_root),
        version,
        server_class=local_server_class(auth, client_id, credential),
    )


def run_local_stdio(runtime, client_id: str, *, backend=None, version: str | None = None) -> None:
    """Run the bundled MCP SDK stdio transport for one saved local client."""
    credential = read_local_credential(runtime.directory, client_id)
    runtime.auth.authenticate_local(credential, client_id)
    if version is None:
        version_file = runtime.app_root.parent / "VERSION"
        version = version_file.read_text(encoding="utf-8").strip()
    if not version or len(version) > 32:
        raise ValueError("Invalid Maestro release version")
    created_backend = backend is None
    if created_backend:
        from .bridge import BackendClient
        backend = BackendClient(runtime)
    try:
        if created_backend:
            from .lifecycle import LifecycleManager
            backend.lifecycle_manager = LifecycleManager(runtime, idle_probe=backend.idle_probe)
        server = create_local_mcp_server(
            backend, runtime.app_root, version, runtime.auth, client_id, credential,
        )
        server.run("stdio")
    finally:
        import asyncio
        close = getattr(backend, "close", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                asyncio.run(result)


__all__ = [
    "LocalCredentialError", "create_local_mcp_server", "delete_local_credential",
    "local_launch_configuration", "local_server_class", "read_local_credential",
    "run_local_stdio", "validate_client_id", "write_local_credential",
]
