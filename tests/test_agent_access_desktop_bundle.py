"""Credential-free Claude Desktop MCPB packaging and STDIO relay tests."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.agent_access.desktop_bundle import (  # noqa: E402
    DesktopBundleError,
    MAX_VALUE_BYTES,
    build_desktop_bundle,
)


def _config(*, command: str | None = None, args: list[str] | None = None) -> dict[str, object]:
    client_id = "123e4567-e89b-12d3-a456-426614174000"
    return {
        "id": client_id,
        "name": "Claude Desktop · local client",
        "command": command or sys.executable,
        "args": args if args is not None else [str((_APP / "maestro_mcp.py").resolve()), "--client", client_id],
    }


def test_builds_minimal_mcpb_with_required_manifest_and_only_safe_config():
    bundle = build_desktop_bundle(_config(), version="4.0.0")
    assert isinstance(bundle, bytes)
    assert zipfile.is_zipfile(BytesIO(bundle))
    with zipfile.ZipFile(BytesIO(bundle)) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "server/index.js",
            "server/launch-config.json",
        }
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["manifest_version"] == "0.3"
        assert manifest["version"] == "4.0.0"
        assert manifest["name"] == "maestro-local-123e4567-e89b-12d3-a456-426614174000"
        assert manifest["author"]["name"] == "Maestro"
        assert manifest["server"] == {
            "type": "node",
            "entry_point": "server/index.js",
            "mcp_config": {"command": "node", "args": ["${__dirname}/server/index.js"]},
        }
        launch = json.loads(archive.read("server/launch-config.json"))
        assert launch == {"command": sys.executable, "args": _config()["args"]}
        assert set(launch) == {"command", "args"}
        assert all("credential" not in name.lower() for name in archive.namelist())
        assert b"access_token" not in bundle
        assert b"refresh_token" not in bundle
        assert b"credential_path" not in bundle
        assert b"/settings/agent-access" not in bundle


@pytest.mark.parametrize(
    "changes",
    [
        {"id": "../outside"},
        {"command": "relative/python"},
        {"args": ["C:\\app\\..\\secret.py"]},
        {"args": [str((_APP / "maestro_mcp.py").resolve()), "--access-token", "123e4567-e89b-12d3-a456-426614174000"]},
        {"args": ["x" * 17000]},
        {"args": ["script.py\n--client"]},
        {"token": "must-not-be-ignored"},
    ],
)
def test_rejects_invalid_or_sensitive_client_config(changes):
    config = _config()
    config.update(changes)
    with pytest.raises(DesktopBundleError):
        build_desktop_bundle(config)


def test_rejects_invalid_version_and_enforces_zip_size_cap(monkeypatch):
    with pytest.raises(DesktopBundleError, match="semantic version"):
        build_desktop_bundle(_config(), version="latest")

    from services.agent_access import desktop_bundle

    monkeypatch.setattr(desktop_bundle, "MAX_BUNDLE_BYTES", 1)
    with pytest.raises(DesktopBundleError, match="too large"):
        build_desktop_bundle(_config())


def test_enforces_launch_config_size_cap():
    client_id = "123e4567-e89b-12d3-a456-426614174000"
    command = "/" + ("x" * (MAX_VALUE_BYTES - 1 - len("/python"))) + "/python"
    script = "/" + ("y" * (MAX_VALUE_BYTES - 1 - len("/maestro_mcp.py"))) + "/maestro_mcp.py"
    with pytest.raises(DesktopBundleError, match="too large"):
        build_desktop_bundle(_config(command=command, args=[script, "--client", client_id]))


def _wait_for_pid_to_exit(pid: int, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return True
        time.sleep(0.05)
    return False


def test_node_relay_preserves_stdio_and_stops_only_its_local_child(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed to exercise the MCPB STDIO relay")

    fake_dir = tmp_path / "fake child 日本語 with spaces"
    fake_dir.mkdir()
    fake_script = fake_dir / "maestro_mcp.py"
    pid_file = fake_dir / "child pid.txt"
    fake_script.write_text(
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "Path(__file__).with_name('child pid.txt').write_text(str(os.getpid()), encoding='utf-8')\n"
        "sys.stderr.write('fake child ready ' + sys.argv[2] + '\\n')\n"
        "payload = sys.stdin.buffer.read()\n"
        "sys.stdout.buffer.write(payload)\n"
        "sys.stdout.buffer.flush()\n"
        "# Deliberately ignore stdin EOF; the extension relay must clean up this child.\n"
        "while True: time.sleep(1)\n",
        encoding="utf-8",
    )
    config = _config(args=[str(fake_script.resolve()), "--client", "123e4567-e89b-12d3-a456-426614174000"])
    bundle = build_desktop_bundle(config)
    extension_dir = tmp_path / "Claude Extension 日本語 with spaces"
    extension_dir.mkdir()
    with zipfile.ZipFile(BytesIO(bundle)) as archive:
        archive.extractall(extension_dir)

    payload = b'{"jsonrpc":"2.0","id":7,"method":"initialize"}\n'
    relay = subprocess.Popen(
        [node, str((extension_dir / "server" / "index.js").resolve())],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = relay.communicate(payload, timeout=6)
    except subprocess.TimeoutExpired:
        relay.kill()
        relay.communicate(timeout=2)
        pytest.fail("the local extension relay did not stop after its host disconnected")

    assert relay.returncode == 0
    assert stdout == payload, "STDIO JSON-RPC bytes must pass through unchanged"
    assert "fake child ready 123e4567-e89b-12d3-a456-426614174000" in stderr.decode("utf-8")
    assert pid_file.exists(), "the fake local MCP child received the Unicode/space-containing launch path"
    child_pid = int(pid_file.read_text(encoding="utf-8"))
    assert _wait_for_pid_to_exit(child_pid), "the relay should clean up only its spawned local client process"
