"""Safe, durable Pinokio lifecycle requests for the local Maestro install.

Lifecycle calls are delegated to Pinokio's own WebSocket API. The manager
never starts Python model code directly and never accepts a caller-provided
script path. A request receipt is committed before a Pinokio mutation is
dispatched so a lost acknowledgement cannot silently repeat the action.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
import inspect
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
from typing import Any, Protocol

from .state import process_alive


_ACTIONS = frozenset({"status", "start", "stop", "restart", "update"})
_ENGINE_SCRIPTS = ("start.js", "start_sol.js", "start_classic.js")
_STARTABLE_SCRIPTS = frozenset((*_ENGINE_SCRIPTS, "update.js", "agent_access.js"))
_MAINTENANCE_SCRIPTS = frozenset({
    "install.js", "update.js", "reset.js", "sol_install.js", "sam_install.js",
    "torch.js", "huggingface_login.js",
})
_KNOWN_AUXILIARY_SCRIPTS = _MAINTENANCE_SCRIPTS | {"agent_access.js"}
_ALLOWED_APP_NAMES = frozenset({"maestro", "maestro-dev", "maestro-main", "maestro.git", "maestro.git-dev", "maestro.git-main"})
_RECEIPT_LIMIT = 2048
_MAX_RECEIPT_BYTES = 32 * 1024


class LifecycleError(Exception):
    """Safe lifecycle error with an HTTP status for the bridge."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = int(status)
        self.status_code = self.status


class PinokioRPCError(Exception):
    """Pinokio helper failure annotated with whether a mutation may have sent."""

    def __init__(self, message: str, *, dispatched: bool):
        super().__init__(message)
        self.dispatched = bool(dispatched)


class PinokioRPC(Protocol):
    async def status(self, app_root: Path, pinokio_home: Path) -> Mapping[str, Any]: ...

    async def dispatch(
        self,
        operation: str,
        script_name: str,
        app_root: Path,
        pinokio_home: Path,
    ) -> Mapping[str, Any]: ...


class PinokioRPCClient:
    """Invoke the bundled Pinokio RPC helper without starting the pterm CLI.

    The pterm CLI installs process-exit handlers that stop its selected script.
    This client uses only the verified pterm `rpc.js` transport, so closing the
    short-lived helper socket never sends `kernel.api.stop` implicitly.
    """

    def __init__(self, *, timeout: float = 8.0):
        self.timeout = timeout

    async def status(self, app_root: Path, pinokio_home: Path) -> Mapping[str, Any]:
        result = await self._invoke({
            "action": "status",
            "app_root": str(app_root),
            "pinokio_home": str(pinokio_home),
        }, dispatched=False)
        if result.get("ok") is not True or not isinstance(result.get("status"), Mapping):
            raise LifecycleError(503, "Pinokio could not confirm this installation's running scripts")
        return result["status"]

    async def dispatch(
        self,
        operation: str,
        script_name: str,
        app_root: Path,
        pinokio_home: Path,
    ) -> Mapping[str, Any]:
        if operation not in {"start", "stop"} or script_name not in _STARTABLE_SCRIPTS:
            raise LifecycleError(400, "Unsupported Pinokio lifecycle target")
        result = await self._invoke({
            "action": "dispatch",
            "operation": operation,
            "script_name": script_name,
            "app_root": str(app_root),
            "pinokio_home": str(pinokio_home),
        }, dispatched=True)
        if result.get("ok") is not True:
            raise PinokioRPCError(
                "Pinokio did not confirm the lifecycle request",
                dispatched=result.get("dispatched") is True,
            )
        return result

    async def _invoke(self, payload: dict[str, Any], *, dispatched: bool) -> dict[str, Any]:
        home = Path(payload["pinokio_home"])
        node = shutil.which("node")
        if not node:
            candidates = (home / "bin" / "node.exe", home / "bin" / "node")
            node = next((str(path) for path in candidates if path.is_file()), None)
        if not node:
            raise PinokioRPCError("Pinokio's Node runtime is unavailable", dispatched=False)
        helper = Path(__file__).resolve().parents[3] / "scripts" / "pinokio" / "agent_access_rpc.js"
        if not helper.is_file():
            raise PinokioRPCError("Pinokio lifecycle helper is missing", dispatched=False)
        try:
            process = await asyncio.create_subprocess_exec(
                node,
                str(helper),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(helper.parents[2]),
            )
        except OSError as error:
            raise PinokioRPCError("Could not start the Pinokio lifecycle helper", dispatched=False) from error
        try:
            stdout, _stderr = await asyncio.wait_for(
                process.communicate(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError as error:
            if process.returncode is None:
                process.kill()  # Only the helper process; Pinokio receives no stop RPC.
            try:
                await process.communicate()
            except Exception:
                pass
            raise PinokioRPCError(
                "Pinokio acknowledgement timed out; inspect status before another request",
                dispatched=dispatched,
            ) from error
        if len(stdout) > _MAX_RECEIPT_BYTES:
            raise PinokioRPCError("Pinokio returned an oversized lifecycle response", dispatched=dispatched)
        try:
            result = json.loads(stdout.decode("utf-8"))
        except (UnicodeError, ValueError) as error:
            raise PinokioRPCError(
                "Pinokio returned an unreadable lifecycle acknowledgement",
                dispatched=dispatched,
            ) from error
        if not isinstance(result, dict):
            raise PinokioRPCError("Pinokio returned an unknown lifecycle acknowledgement", dispatched=dispatched)
        return result


class _ReceiptStore:
    """SQLite-backed idempotency receipts for lifecycle mutations."""

    def __init__(self, directory: Path):
        self.path = Path(directory) / "lifecycle-receipts.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS lifecycle_requests (
                    principal_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    state TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (principal_id, request_id)
                )"""
            )
        if os.name != "nt":
            os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    async def lookup(self, principal_id: str, request_id: str, action: str) -> dict[str, Any] | None:
        async with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT * FROM lifecycle_requests WHERE principal_id=? AND request_id=?",
                    (principal_id, request_id),
                ).fetchone()
                if row is None:
                    return None
                if row["action"] != action:
                    raise LifecycleError(409, "This request_id is already reserved for another lifecycle action")
                if row["state"] == "dispatching":
                    result = json.loads(row["result_json"])
                    result.update(
                        state="uncertain",
                        message="A prior request may have reached Pinokio. Inspect status; it will not be dispatched again.",
                    )
                    update = self._connect()
                    try:
                        update.execute(
                            "UPDATE lifecycle_requests SET state='uncertain', result_json=?, updated_at=? WHERE principal_id=? AND request_id=?",
                            (_json_encode(result), time.time(), principal_id, request_id),
                        )
                    finally:
                        update.close()
                    return result
                result = json.loads(row["result_json"])
                result["replayed"] = True
                return result
            finally:
                connection.close()

    async def reserve(
        self,
        principal_id: str,
        request_id: str,
        action: str,
        plan: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Commit the pending receipt; return an existing request if present."""
        async with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM lifecycle_requests WHERE principal_id=? AND request_id=?",
                    (principal_id, request_id),
                ).fetchone()
                if row is not None:
                    if row["action"] != action:
                        connection.rollback()
                        raise LifecycleError(409, "This request_id is already reserved for another lifecycle action")
                    if row["state"] == "dispatching":
                        result = json.loads(row["result_json"])
                        result.update(
                            state="uncertain",
                            message="A prior request may have reached Pinokio. Inspect status; it will not be dispatched again.",
                        )
                        encoded = _json_encode(result)
                        connection.execute(
                            "UPDATE lifecycle_requests SET state='uncertain', result_json=?, updated_at=? WHERE principal_id=? AND request_id=?",
                            (encoded, time.time(), principal_id, request_id),
                        )
                        connection.commit()
                        return result
                    result = json.loads(row["result_json"])
                    result["replayed"] = True
                    connection.commit()
                    return result
                count = connection.execute("SELECT COUNT(*) FROM lifecycle_requests").fetchone()[0]
                if count >= _RECEIPT_LIMIT:
                    connection.rollback()
                    raise LifecycleError(429, "The lifecycle receipt journal is full; contact the owner")
                now = time.time()
                result = {
                    "action": action,
                    "request_id": request_id,
                    "state": "dispatching",
                    "plan": dict(plan),
                    "message": "The Pinokio lifecycle request is being dispatched.",
                }
                encoded = _json_encode(result)
                connection.execute(
                    "INSERT INTO lifecycle_requests(principal_id,request_id,action,state,result_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                    (principal_id, request_id, action, "dispatching", encoded, now, now),
                )
                connection.commit()
                return None
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise
            finally:
                connection.close()

    async def finish(
        self,
        principal_id: str,
        request_id: str,
        state: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        encoded = _json_encode(dict(result))
        async with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "UPDATE lifecycle_requests SET state=?, result_json=?, updated_at=? WHERE principal_id=? AND request_id=?",
                    (state, encoded, time.time(), principal_id, request_id),
                )
                connection.commit()
            finally:
                connection.close()
        return dict(result)


def _json_encode(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > _MAX_RECEIPT_BYTES:
        raise LifecycleError(413, "Lifecycle receipt exceeds the configured size limit")
    return encoded


def _required_text(value: Any, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LifecycleError(400, f"{name} must be a non-empty string")
    value = value.strip()
    if len(value.encode("utf-8")) > maximum or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise LifecycleError(400, f"{name} is invalid")
    return value


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.realpath(left)) == os.path.normcase(os.path.realpath(right))


def _resolve_pinokio_home(override: Path | str | None = None) -> Path:
    if override is not None:
        candidate = Path(override).expanduser()
    else:
        config_path = Path.home() / ".pinokio" / "config.json"
        candidate = None
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError) as error:
                raise LifecycleError(503, "Pinokio configuration could not be read safely") from error
            home = config.get("home") if isinstance(config, dict) else None
            if isinstance(home, str) and home.strip():
                candidate = Path(home.strip()).expanduser()
        if candidate is None:
            env_home = os.environ.get("PINOKIO_HOME")
            if isinstance(env_home, str) and env_home.strip():
                candidate = Path(env_home.strip()).expanduser()
    if not candidate or not candidate.is_absolute():
        raise LifecycleError(503, "PINOKIO_HOME is unavailable; lifecycle control is disabled")
    resolved = candidate.resolve(strict=False)
    if not (resolved / "api").is_dir():
        raise LifecycleError(503, "The configured PINOKIO_HOME has no api directory")
    return resolved


def _installation_paths(app_root: Path | str, pinokio_home: Path) -> tuple[Path, Path]:
    supplied = Path(app_root).expanduser().resolve(strict=True)
    if supplied.name.casefold() == "app":
        project_root = supplied.parent
        app_directory = supplied
    elif (supplied / "app").is_dir():
        project_root = supplied
        app_directory = (supplied / "app").resolve(strict=True)
    else:
        raise LifecycleError(403, "Lifecycle control requires the Maestro app directory")
    has_maestro_source = (app_directory / "services" / "agent_access" / "__init__.py").is_file() and (app_directory / "launch.py").is_file() and (project_root / "VERSION").is_file()
    if project_root.name.casefold() not in _ALLOWED_APP_NAMES and not has_maestro_source:
        raise LifecycleError(403, "Lifecycle control is only enabled for the installed Maestro or Maestro-dev app")
    api_root = (pinokio_home / "api").resolve(strict=True)
    if not _same_path(project_root.parent, api_root):
        raise LifecycleError(403, "Lifecycle control is disabled outside the installed PINOKIO_HOME/api app root")
    if not _same_path(app_directory, project_root / "app"):
        raise LifecycleError(403, "The Maestro app directory is not a direct child of its launcher root")
    for name in ("pinokio.js", "start.js", "update.js"):
        if not (project_root / name).is_file():
            raise LifecycleError(503, "The installed Maestro Pinokio launcher is incomplete")
    return project_root.resolve(strict=True), app_directory


def _script_name(value: Any, project_root: Path) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().split("?", 1)[0]
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    candidate = candidate.resolve(strict=False)
    try:
        relative = candidate.relative_to(project_root.resolve(strict=True))
    except ValueError:
        return "__foreign__"
    return relative.as_posix()


def _running_scripts(status: Mapping[str, Any], project_root: Path) -> frozenset[str]:
    scripts = status.get("running_scripts")
    if not isinstance(scripts, list):
        raise LifecycleError(503, "Pinokio did not provide a trustworthy running-script list")
    result = set()
    for item in scripts:
        normalized = _script_name(item, project_root)
        if normalized is None:
            raise LifecycleError(409, "Pinokio returned an unknown running script; inspect the launcher before continuing")
        result.add(normalized)
    return frozenset(result)


def _acknowledgement(reply: Mapping[str, Any]) -> tuple[str, str]:
    """Classify one RPC response without treating it as completion."""
    if reply.get("already_running") is True:
        return "accepted", "Pinokio already reports this script as running. Use status to check readiness."
    if reply.get("already_stopped") is True:
        return "accepted", "Pinokio already reports this script as stopped."
    if reply.get("dispatched") is not True:
        return "rejected", "Pinokio did not receive the lifecycle request."
    packet = reply.get("packet")
    if not isinstance(packet, Mapping):
        return "uncertain", "Pinokio may have received the request, but no acknowledgement arrived. Inspect status; do not retry automatically."
    packet_type = packet.get("type")
    if packet_type in {"stream", "result"}:
        return "accepted", "Pinokio accepted the request; startup or update may still be in progress. Use status to reconcile."
    return "uncertain", "Pinokio returned an unknown acknowledgement. Inspect status; do not retry automatically."


class LifecycleManager:
    """Coordinate owner-authorized Maestro actions through Pinokio only."""

    def __init__(
        self,
        runtime,
        *,
        idle_probe: Callable[[], Awaitable[Mapping[str, Any]] | Mapping[str, Any]] | None = None,
        rpc: PinokioRPC | None = None,
        pinokio_home: Path | str | None = None,
        git_clean: Callable[[Path], bool] | None = None,
        reconcile_timeout: float = 4.0,
    ):
        self.runtime = runtime
        self.idle_probe = idle_probe
        self.rpc = rpc or PinokioRPCClient()
        self.pinokio_home = pinokio_home
        self.git_clean = git_clean or self._git_clean
        self.reconcile_timeout = max(0.1, float(reconcile_timeout))
        self.receipts = _ReceiptStore(Path(runtime.directory))

    async def execute(
        self,
        action: str,
        principal_id: str,
        request_id: str | None,
    ) -> dict[str, Any]:
        if not isinstance(action, str) or action.strip().casefold() not in _ACTIONS:
            raise LifecycleError(400, "Use one of the supported lifecycle actions: status, start, stop, restart, update")
        action = action.strip().casefold()
        home = _resolve_pinokio_home(self.pinokio_home)
        app_root = getattr(self.runtime, "app_root", None)
        if app_root is None:
            raise LifecycleError(503, "This controller has no verified Maestro app root")
        project_root, _app_directory = _installation_paths(app_root, home)

        if action == "status":
            status = await self._status(home, project_root)
            scripts = _running_scripts(status, project_root)
            if "__foreign__" in scripts:
                raise LifecycleError(409, "Pinokio reported a script outside the installed Maestro root")
            backend = self._backend_record()
            return {
                "action": "status",
                "state": "running" if any(script in scripts for script in _ENGINE_SCRIPTS) else "stopped",
                "launcher_scripts": sorted(script for script in scripts if script in _ENGINE_SCRIPTS),
                "agent_access_terminal": "agent_access.js" in scripts,
                "backend_online": bool(backend and backend.get("ready") and process_alive(backend.get("pid"))),
            }

        principal_id = _required_text(principal_id, "principal_id", 512)
        request_id = _required_text(request_id, "request_id", 256)
        existing = await self.receipts.lookup(principal_id, request_id, action)
        if existing is not None:
            return existing

        try:
            if action in {"stop", "restart", "update"}:
                await self._require_idle()
            status = await self._status(home, project_root)
            scripts = _running_scripts(status, project_root)
            backend = self._backend_record()
            engine_scripts = sorted(script for script in scripts if script in _ENGINE_SCRIPTS)
            self._verify_engine_ownership(engine_scripts, scripts, backend)
            maintenance = sorted(script for script in scripts if script in _MAINTENANCE_SCRIPTS)
            if maintenance:
                if action == "update" and maintenance == ["update.js"]:
                    if engine_scripts:
                        raise LifecycleError(409, "Pinokio reports an update and Maestro engine running together; inspect status")
                    result = await self.receipts.reserve(
                        principal_id, request_id, action, {"target": "update.js", "already_running": True}
                    )
                    if result is not None:
                        return result
                    result = self._accepted_result(action, request_id, "update.js", already_running=True)
                    return await self.receipts.finish(principal_id, request_id, "accepted", result)
                raise LifecycleError(409, "Another Maestro maintenance script is running; inspect Pinokio before continuing")
            unknown = sorted(script for script in scripts if script not in _ENGINE_SCRIPTS and script not in _KNOWN_AUXILIARY_SCRIPTS)
            if unknown:
                raise LifecycleError(409, "Another Maestro script is running; inspect Pinokio before continuing")

            if len(engine_scripts) > 1:
                raise LifecycleError(409, "More than one Maestro launcher script is running; inspect Pinokio before continuing")
            if action == "start" and engine_scripts:
                result = await self.receipts.reserve(
                    principal_id, request_id, action, {"target": engine_scripts[0], "already_running": True}
                )
                if result is not None:
                    return result
                result = self._accepted_result(action, request_id, engine_scripts[0], already_running=True)
                return await self.receipts.finish(principal_id, request_id, "accepted", result)
            if action == "restart" and len(engine_scripts) != 1:
                raise LifecycleError(409, "Maestro is not running in one verified Pinokio script; use Start or inspect status")
            if action == "update" and not self.git_clean(project_root):
                raise LifecycleError(409, "Update is blocked because the installed Maestro checkout has local changes")

            saved_profile = backend.get("launcher_script")
            if action == "start" and saved_profile not in _ENGINE_SCRIPTS:
                saved_profile = "start.js"
            target = engine_scripts[0] if engine_scripts else ("update.js" if action == "update" else saved_profile if action == "start" else "start.js")
            plan = {"target": target, "stop_before_update": bool(action == "update" and engine_scripts)}
            result = await self.receipts.reserve(principal_id, request_id, action, plan)
            if result is not None:
                return result
        except Exception as error:
            # Rejections before dispatch are safe to expose; the reservation
            # remains pending so the same request ID cannot be reinterpreted.
            rejected = {
                "action": action,
                "request_id": request_id,
                "state": "rejected",
                "message": str(error) if isinstance(error, LifecycleError) else "Lifecycle preflight failed; inspect status before making another request.",
            }
            await self.receipts.finish(principal_id, request_id, "rejected", rejected)
            raise

        dispatches: list[dict[str, Any]] = []
        try:
            if action in {"stop", "restart"} or (action == "update" and engine_scripts):
                if engine_scripts:
                    stop_result = await self._dispatch("stop", engine_scripts[0], home, project_root)
                    state, message = _acknowledgement(stop_result)
                    dispatches.append({"operation": "stop", "script": engine_scripts[0], "state": state})
                    if state != "accepted":
                        final = self._accepted_result(action, request_id, engine_scripts[0], state=state, message=message, dispatches=dispatches)
                        return await self.receipts.finish(principal_id, request_id, state, final)
                if action == "stop":
                    final = self._accepted_result(action, request_id, engine_scripts[0] if engine_scripts else None, already_stopped=not bool(engine_scripts), dispatches=dispatches)
                    return await self.receipts.finish(principal_id, request_id, "accepted", final)
                if action in {"restart", "update"} and engine_scripts:
                    stopped = await self._wait_until_stopped(home, project_root, engine_scripts[0])
                    if not stopped:
                        final = self._accepted_result(
                            action,
                            request_id,
                            engine_scripts[0],
                            state="uncertain",
                            message="Pinokio accepted stop, but status has not confirmed the backend stopped. No follow-up was dispatched.",
                            dispatches=dispatches,
                        )
                        return await self.receipts.finish(principal_id, request_id, "uncertain", final)

            start_target = target if action in {"start", "restart"} else "update.js"
            reply = await self._dispatch("start", start_target, home, project_root)
            state, message = _acknowledgement(reply)
            dispatches.append({"operation": "start", "script": start_target, "state": state})
            final = self._accepted_result(action, request_id, start_target, state=state, message=message, dispatches=dispatches)
            return await self.receipts.finish(principal_id, request_id, state, final)
        except PinokioRPCError as error:
            state = "uncertain" if error.dispatched else "rejected"
            final = self._accepted_result(action, request_id, target, state=state, message=str(error), dispatches=dispatches)
            return await self.receipts.finish(principal_id, request_id, state, final)
        except Exception:
            final = self._accepted_result(
                action,
                request_id,
                target,
                state="uncertain",
                message="The Pinokio request may have been sent. Inspect status; it will not be dispatched again.",
                dispatches=dispatches,
            )
            return await self.receipts.finish(principal_id, request_id, "uncertain", final)

    async def _status(self, home: Path, project_root: Path) -> Mapping[str, Any]:
        try:
            status = await self.rpc.status(project_root, home)
        except LifecycleError:
            raise
        except Exception as error:
            raise LifecycleError(503, "Pinokio status is unavailable; lifecycle actions are disabled") from error
        if not isinstance(status, Mapping):
            raise LifecycleError(503, "Pinokio returned an invalid status document")
        registered_path = status.get("path", status.get("app_path"))
        if not isinstance(registered_path, str) or not _same_path(Path(registered_path), project_root):
            raise LifecycleError(409, "Pinokio could not verify this exact installed Maestro app; lifecycle action refused")
        return status

    async def _dispatch(
        self,
        operation: str,
        script: str,
        home: Path,
        project_root: Path,
    ) -> Mapping[str, Any]:
        if operation == "start" and script not in _STARTABLE_SCRIPTS:
            raise LifecycleError(400, "Unsupported Pinokio start target")
        if operation == "stop" and script not in _ENGINE_SCRIPTS:
            raise LifecycleError(400, "Unsupported Pinokio stop target")
        try:
            reply = await self.rpc.dispatch(operation, script, project_root, home)
        except PinokioRPCError:
            raise
        except Exception as error:
            raise PinokioRPCError(
                "Pinokio lifecycle acknowledgement was lost; inspect status before another request",
                dispatched=True,
            ) from error
        if not isinstance(reply, Mapping):
            return {"dispatched": True, "packet": None}
        return reply

    async def _require_idle(self) -> Mapping[str, Any]:
        if self.idle_probe is None:
            raise LifecycleError(409, "The backend could not verify active work; lifecycle mutation is refused")
        try:
            result = self.idle_probe()
            if inspect.isawaitable(result):
                result = await result
        except Exception as error:
            raise LifecycleError(409, "The backend could not verify every queue and engine; lifecycle mutation is refused") from error
        if not isinstance(result, Mapping):
            raise LifecycleError(409, "The backend returned an unknown work status; lifecycle mutation is refused")
        if result.get("foreign") is True or result.get("certain") is not True or result.get("idle") is not True:
            reason = result.get("reason")
            if isinstance(reason, str) and reason.strip() and len(reason) <= 300:
                message = "Lifecycle mutation is refused: " + reason.strip()
            else:
                message = "Lifecycle mutation requires a verified idle backend and Director queue"
            raise LifecycleError(409, message)
        return result

    def _backend_record(self) -> dict[str, Any]:
        try:
            record = self.runtime.backend()
        except Exception as error:
            raise LifecycleError(409, "The backend registration is not trustworthy; inspect status before continuing") from error
        return record if isinstance(record, dict) else {}

    @staticmethod
    def _verify_engine_ownership(
        engine_scripts: list[str],
        running_scripts: frozenset[str],
        backend: Mapping[str, Any],
    ) -> None:
        alive = process_alive(backend.get("pid")) if backend.get("pid") is not None else False
        if len(engine_scripts) > 1:
            return  # The caller reports the more specific multiple-launcher error.
        if engine_scripts and not alive:
            raise LifecycleError(409, "A Maestro launcher is still active without a verified ready backend; inspect Pinokio first")
        if engine_scripts and backend.get("launcher_script") != engine_scripts[0]:
            raise LifecycleError(409, "The running backend does not match the active Pinokio launcher; lifecycle action refused")
        if not engine_scripts and alive:
            raise LifecycleError(409, "A Maestro backend process is not owned by a running Pinokio launcher; lifecycle action refused")
        if "__foreign__" in running_scripts:
            raise LifecycleError(409, "Pinokio reported a script outside the installed Maestro root")

    async def _wait_until_stopped(self, home: Path, project_root: Path, script: str) -> bool:
        deadline = asyncio.get_running_loop().time() + self.reconcile_timeout
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(min(0.2, max(0.01, deadline - asyncio.get_running_loop().time())))
            status = await self._status(home, project_root)
            scripts = _running_scripts(status, project_root)
            if script not in scripts:
                backend = self._backend_record()
                if backend.get("pid") is None or not process_alive(backend.get("pid")):
                    return True
        return False

    @staticmethod
    def _accepted_result(
        action: str,
        request_id: str,
        target: str | None,
        *,
        state: str = "accepted",
        message: str | None = None,
        already_running: bool = False,
        already_stopped: bool = False,
        dispatches: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        defaults = {
            "start": "Pinokio accepted start; the app may still be starting. Use status to reconcile.",
            "stop": "Pinokio accepted stop; shutdown may still be in progress. Use status to reconcile.",
            "restart": "Pinokio accepted restart; the app may still be starting. Use status to reconcile.",
            "update": "Pinokio accepted update; it may still be running. Use status to reconcile.",
        }
        result = {
            "action": action,
            "request_id": request_id,
            "state": state,
            "target": target,
            "message": message or defaults[action],
        }
        if already_running:
            result["already_running"] = True
            result["message"] = "The Pinokio script is already running; check status for readiness."
        if already_stopped:
            result["already_stopped"] = True
            result["message"] = "The Maestro backend is already stopped."
        if dispatches is not None:
            result["dispatches"] = dispatches
        return result

    @staticmethod
    def _git_clean(project_root: Path) -> bool:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and not result.stdout.strip()


__all__ = ["LifecycleError", "LifecycleManager", "PinokioRPC", "PinokioRPCClient", "PinokioRPCError"]
