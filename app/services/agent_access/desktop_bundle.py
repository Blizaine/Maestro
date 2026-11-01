"""Build a minimal, credential-free Claude Desktop MCPB for a local client."""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Mapping
from zipfile import ZIP_DEFLATED, ZipFile
from io import BytesIO


MAX_BUNDLE_BYTES = 64 * 1024
MAX_CONFIG_BYTES = 8 * 1024
MAX_ARGUMENTS = 64
MAX_VALUE_BYTES = 4096
MAX_CLIENT_NAME_BYTES = 256


class DesktopBundleError(ValueError):
    """Raised when a local-client launch config cannot be packaged safely."""


_RELAY_SOURCE = r'''"use strict";

const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const MAX_CONFIG_BYTES = 8192;
const MAX_ARGUMENTS = 64;
const MAX_VALUE_CHARS = 4096;
const CONFIG_PATH = path.join(__dirname, "launch-config.json");

function writeError(message) {
  try {
    process.stderr.write(`Maestro local MCP extension: ${message}\n`);
  } catch (_) {
    // The host may already have closed stderr.
  }
}

function hasControl(value) {
  return /[\u0000-\u001f\u007f]/.test(value);
}

function hasTraversal(value) {
  return value.split(/[\\/]+/).some((part) => part === "..");
}

function isAbsoluteExecutable(value) {
  return path.isAbsolute(value) || /^[A-Za-z]:[\\/]/.test(value) || /^\\\\/.test(value);
}

function validConfig(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const keys = Object.keys(value).sort();
  if (keys.length !== 2 || keys[0] !== "args" || keys[1] !== "command") return false;
  if (typeof value.command !== "string" || value.command.length === 0 || value.command.length > MAX_VALUE_CHARS) return false;
  if (!isAbsoluteExecutable(value.command) || hasControl(value.command) || hasTraversal(value.command)) return false;
  const executable = value.command.split(/[\\/]/).pop().toLowerCase();
  if (!/^(python(?:\d+(?:\.\d+)*)?|pypy(?:\d+(?:\.\d+)*)?)(?:\.exe)?$/.test(executable)) return false;
  if (!Array.isArray(value.args) || value.args.length !== 3 || value.args.length > MAX_ARGUMENTS) return false;
  const scriptPath = value.args[0];
  if (typeof scriptPath !== "string" || !isAbsoluteExecutable(scriptPath) || path.basename(scriptPath).toLowerCase() !== "maestro_mcp.py") return false;
  if (value.args[1] !== "--client" || typeof value.args[2] !== "string" || !/^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$/.test(value.args[2])) return false;
  let bytes = Buffer.byteLength(value.command, "utf8");
  for (const arg of value.args) {
    if (typeof arg !== "string" || arg.length > MAX_VALUE_CHARS || hasControl(arg) || hasTraversal(arg)) return false;
    const lower = arg.toLowerCase();
    if (/^--?(token|access[-_]token|refresh[-_]token|secret|credential)(?:=|$)/.test(lower)) return false;
    bytes += Buffer.byteLength(arg, "utf8");
    if (bytes > MAX_CONFIG_BYTES) return false;
  }
  return true;
}

function start() {
  let launch;
  try {
    const stat = fs.statSync(CONFIG_PATH);
    if (!stat.isFile() || stat.size > MAX_CONFIG_BYTES) throw new Error("size");
    launch = JSON.parse(fs.readFileSync(CONFIG_PATH, "utf8"));
  } catch (_) {
    writeError("Could not read the local launch configuration.");
    process.exitCode = 1;
    process.stdin.pause();
    return;
  }

  if (!validConfig(launch)) {
    writeError("The local launch configuration is invalid.");
    process.exitCode = 1;
    process.stdin.pause();
    return;
  }

  let child;
  try {
    child = spawn(launch.command, launch.args, {
      shell: false,
      windowsHide: true,
      stdio: ["pipe", "pipe", "pipe"],
    });
  } catch (error) {
    const code = error && typeof error.code === "string" ? ` (${error.code})` : "";
    writeError(`Could not start the local Maestro MCP service${code}.`);
    process.exitCode = 1;
    process.stdin.pause();
    return;
  }

  let finished = false;
  let stopping = false;
  let failed = false;
  let parentDisconnected = false;
  let parentSignal = null;
  let stopTimer = null;

  function finish(code) {
    if (finished) return;
    finished = true;
    if (stopTimer) clearTimeout(stopTimer);
    process.stdin.unpipe(child.stdin);
    process.stdin.pause();
    child.stdout.unpipe(process.stdout);
    child.stderr.unpipe(process.stderr);
    process.exitCode = parentSignal || parentDisconnected ? 0 : failed ? 1 : (Number.isInteger(code) ? code : 1);
  }

  function requestStop(signal = null, disconnected = false) {
    if (signal && !parentSignal) parentSignal = signal;
    if (disconnected) parentDisconnected = true;
    if (stopping || finished) return;
    stopping = true;
    try {
      child.stdin.end();
    } catch (_) {
      // Continue to the bounded child cleanup below.
    }
    stopTimer = setTimeout(() => {
      if (!finished && child.exitCode === null && !child.killed) child.kill();
    }, 1000);
    if (typeof stopTimer.unref === "function") stopTimer.unref();
  }

  process.stdin.pipe(child.stdin);
  child.stdout.pipe(process.stdout);
  child.stderr.pipe(process.stderr);
  child.stdin.on("error", (error) => {
    if (error && error.code !== "EPIPE") {
      failed = true;
      writeError("The local MCP service input stream failed.");
    }
  });
  process.stdin.on("error", (error) => {
    if (error && error.code !== "EPIPE") {
      failed = true;
      writeError("The extension input stream failed.");
    }
    requestStop();
  });
  process.stdout.on("error", (error) => {
    if (error && error.code === "EPIPE") requestStop(null, true);
  });
  process.stdin.once("end", () => requestStop(null, true));
  process.stdin.once("close", () => requestStop(null, true));
  process.once("SIGTERM", () => {
    requestStop("SIGTERM");
    try { child.kill(); } catch (_) { /* already stopped */ }
  });
  process.once("SIGINT", () => {
    requestStop("SIGINT");
    try { child.kill(); } catch (_) { /* already stopped */ }
  });
  child.once("error", (error) => {
    failed = true;
    const code = error && typeof error.code === "string" ? ` (${error.code})` : "";
    writeError(`Could not start the local Maestro MCP service${code}.`);
    finish(1);
  });
  child.once("close", (code) => finish(code));
}

start();
'''


def build_desktop_bundle(config: Mapping[str, Any], version: str = "4.0.0") -> bytes:
    """Return an in-memory `.mcpb` ZIP for one active local client.

    The config is intentionally restricted to ``id``, ``name``, ``command``,
    and ``args``. Extra data is rejected so credentials or unrelated settings
    cannot accidentally be copied into an extension package.
    """

    safe = _validate_config(config)
    safe_version = _validate_version(version)
    launch_config = {"command": safe["command"], "args": safe["args"]}
    launch_bytes = _json_bytes(launch_config)
    if len(launch_bytes) > MAX_CONFIG_BYTES:
        raise DesktopBundleError("The local launch configuration is too large to package.")

    client_slug = re.sub(r"[^a-z0-9_-]+", "-", safe["id"].lower()).strip("-_")
    manifest = {
        "manifest_version": "0.3",
        "name": f"maestro-local-{client_slug}",
        "display_name": f"Maestro Local — {safe['name']}",
        "version": safe_version,
        "description": "Connect Claude Desktop to this local Maestro installation on the same computer.",
        "author": {"name": "Maestro"},
        "server": {
            "type": "node",
            "entry_point": "server/index.js",
            "mcp_config": {
                "command": "node",
                "args": ["${__dirname}/server/index.js"],
            },
        },
        "compatibility": {
            "platforms": ["darwin", "win32", "linux"],
            "runtimes": {"node": ">=16.0.0"},
        },
    }
    manifest_bytes = _json_bytes(manifest)
    relay_bytes = _RELAY_SOURCE.encode("utf-8")

    archive = BytesIO()
    try:
        with ZipFile(archive, mode="w", compression=ZIP_DEFLATED, compresslevel=9) as output:
            output.writestr("manifest.json", manifest_bytes)
            output.writestr("server/index.js", relay_bytes)
            output.writestr("server/launch-config.json", launch_bytes)
        result = archive.getvalue()
    except (OSError, ValueError) as exc:
        raise DesktopBundleError("Could not assemble the local Claude Desktop extension.") from exc
    if len(result) > MAX_BUNDLE_BYTES:
        raise DesktopBundleError("The local Claude Desktop extension is too large to package.")
    return result


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise DesktopBundleError("A local client launch configuration is required.")
    if set(config) != {"id", "name", "command", "args"}:
        raise DesktopBundleError("The local client launch configuration contains unsupported fields.")

    client_id = _text(config["id"], "client ID", max_bytes=80)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", client_id):
        raise DesktopBundleError("The local client ID is invalid.")

    name = _text(config["name"], "client name", max_bytes=MAX_CLIENT_NAME_BYTES)
    if not name.strip():
        raise DesktopBundleError("The local client name cannot be empty.")

    command = _text(config["command"], "command", max_bytes=MAX_VALUE_BYTES)
    if not command.strip() or _has_path_traversal(command) or not _is_absolute_path(command):
        raise DesktopBundleError("The local client command must be an absolute executable path without traversal segments.")
    executable = PureWindowsPath(command).name if "\\" in command or re.match(r"^[A-Za-z]:", command) else PurePosixPath(command).name
    if not re.fullmatch(r"(?:python(?:\d+(?:\.\d+)*)?|pypy(?:\d+(?:\.\d+)*)?)(?:\.exe)?", executable, flags=re.IGNORECASE):
        raise DesktopBundleError("The local client command must be a Python executable path.")

    raw_args = config["args"]
    if not isinstance(raw_args, (list, tuple)) or len(raw_args) != 3 or len(raw_args) > MAX_ARGUMENTS:
        raise DesktopBundleError("The local client command arguments are invalid.")
    if raw_args[1] != "--client" or raw_args[2] != client_id:
        raise DesktopBundleError("The local client arguments must identify only this registered client.")
    script = _text(raw_args[0], "server script argument", max_bytes=MAX_VALUE_BYTES)
    script_name = PureWindowsPath(script).name if "\\" in script or re.match(r"^[A-Za-z]:", script) else PurePosixPath(script).name
    if not _is_absolute_path(script) or _has_path_traversal(script) or script_name.lower() != "maestro_mcp.py":
        raise DesktopBundleError("The local client must launch an absolute Python server script without traversal segments.")
    args: list[str] = []
    total_bytes = len(command.encode("utf-8"))
    for raw_arg in raw_args:
        arg = _text(raw_arg, "command argument", max_bytes=MAX_VALUE_BYTES)
        if _has_path_traversal(arg):
            raise DesktopBundleError("The local client command arguments cannot contain path traversal segments.")
        if re.match(r"^--?(?:token|access[-_]token|refresh[-_]token|secret|credential)(?:=|$)", arg, flags=re.IGNORECASE):
            raise DesktopBundleError("Credential arguments cannot be included in a local client extension.")
        total_bytes += len(arg.encode("utf-8"))
        if total_bytes > MAX_CONFIG_BYTES:
            raise DesktopBundleError("The local client launch configuration is too large to package.")
        args.append(arg)

    return {"id": client_id, "name": name, "command": command, "args": args}


def _text(value: Any, label: str, *, max_bytes: int) -> str:
    if not isinstance(value, str) or not value:
        raise DesktopBundleError(f"The local client {label} is invalid.")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DesktopBundleError(f"The local client {label} is invalid.") from exc
    if len(encoded) > max_bytes or any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise DesktopBundleError(f"The local client {label} is too long or contains control characters.")
    return value


def _is_absolute_path(value: str) -> bool:
    return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()


def _has_path_traversal(value: str) -> bool:
    return any(part == ".." for part in re.split(r"[\\/]+", value))


def _validate_version(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?",
        value,
    ):
        raise DesktopBundleError("The extension version must be a valid semantic version.")
    if len(value) > 64:
        raise DesktopBundleError("The extension version is too long.")
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, separators=(",", ": ")).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise DesktopBundleError("The local extension metadata is invalid.") from exc
