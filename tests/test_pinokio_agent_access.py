from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (_ROOT / path).read_text(encoding="utf-8")


def test_agent_access_terminal_uses_separate_cpu_environment_and_ready_url_capture():
    script = _read("agent_access.js")
    assert 'daemon: true' in script
    assert 'venv: "env-agent"' in script
    assert 'venv_python: "3.11"' in script
    assert "python -u -m services.agent_access.controller --app-root ." in script
    assert "{{args.setup ? '--setup' : ''}}" in script
    assert "MAESTRO_AGENT_ACCESS_READY" in script
    assert '\n      url: "{{input.event[1]}}"' not in script
    assert 'agent_access_url: "{{input.event[1]}}"' in script
    assert r"https:\/\/127\.0\.0\.1" in script
    assert "app/env-agent/pyvenv.cfg" in script
    assert "uv pip install -r requirements-agent-access.txt" in script


def test_every_model_launcher_autostarts_agent_sidecar_and_registers_exact_profile():
    for launcher in ("start.js", "start_sol.js", "start_classic.js"):
        script = _read(launcher)
        assert "node scripts/pinokio/agent_access_rpc.js autostart" in script
        assert f'MAESTRO_LAUNCHER_SCRIPT: "{launcher}"' in script
    # Existing Maestro readiness captures stay in place and keep their original URLs.
    assert r'"event": "/(http:\/\/[0-9.:]+)/"' in _read("start.js")
    assert r'"event": "/(http:\/\/[0-9.:]+)/"' in _read("start_sol.js")
    assert r'"event": "/(http:\/\/[0-9.:]+)/"' in _read("start_classic.js")


def test_pinokio_menu_exposes_agent_terminal_when_model_is_stopped():
    menu = _read("pinokio.js")
    assert 'info.running("agent_access.js")' in menu
    assert 'info.exists("app/env-agent/pyvenv.cfg")' in menu
    assert 'text: "Start Agent Access and show local setup code"' in menu
    assert 'text: "Agent Access Terminal"' in menu
    assert 'text: "Set up Agent Access"' in menu
    assert 'text: "Open Agent Access"' in menu
    # The installed-but-stopped branch is independent of any running model script.
    assert "...agentAccessMenu(info, agentAccessReady, running.agent_access)" in menu


def test_install_and_both_update_paths_install_pinned_cpu_requirements():
    install = _read("install.js")
    update = _read("update.js")
    assert 'venv: "env-agent"' in install
    assert "requirements-agent-access.txt" in install
    assert update.count("requirements-agent-access.txt") == 2
    assert update.count('venv: "env-agent"') == 2
    requirements = _read("app/requirements-agent-access.txt")
    assert "mcp==2.3.0" in requirements
    assert "torch" not in requirements.lower()


def test_helper_uses_fixed_local_pterm_rpc_without_cli_exit_hooks():
    helper = _read("scripts/pinokio/agent_access_rpc.js")
    lifecycle = _read("app/services/agent_access/lifecycle.py")
    assert 'require(path.join(pinokioPackage, "rpc.js"))' in helper
    assert 'source: "pterm"' in helper
    assert 'method: "kernel.api.stop"' in helper
    assert "rpc.stop(" not in helper
    assert 'require("pterm")' not in helper
    assert "kernel.api.stop" in lifecycle
    assert "asyncio.create_subprocess_exec" in lifecycle
    assert "shell=True" not in lifecycle
    assert 'new Set(["start.js", "start_sol.js", "start_classic.js"])' in helper
    assert '"update.js", "agent_access.js"' in helper
    assert "PINOKIO_HOME/api" in helper


def test_launcher_files_parse_with_node_when_available():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not on PATH in this test environment")
    for relative in (
        "agent_access.js",
        "pinokio.js",
        "start.js",
        "start_sol.js",
        "start_classic.js",
        "install.js",
        "update.js",
        "scripts/pinokio/agent_access_rpc.js",
    ):
        result = subprocess.run(
            [node, "--check", str(_ROOT / relative)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0, f"{relative}: {result.stderr}"


def test_agent_readiness_captures_url_inside_terminal_chunk():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not on PATH in this test environment")
    result = subprocess.run(
        [node, "-e", r'''
const assert = require("node:assert/strict");
(async () => {
  const script = await require(process.argv[1])();
  const shell = script.run.find(step => step.method === "shell.run" && step.params.on);
  const pattern = shell.params.on[0].event;
  const re = new RegExp(pattern.slice(1, pattern.lastIndexOf("/")));
  for (const output of [
    "MAESTRO_AGENT_ACCESS_READY https://127.0.0.1:55468\r\n",
    "Starting controller...\r\n\x1b[?25lMAESTRO_AGENT_ACCESS_READY https://127.0.0.1:55468\r\n\x1b[?25h",
  ]) {
    assert.equal(output.match(re)?.[1], "https://127.0.0.1:55468");
  }
  assert.equal(re.test("MAESTRO_AGENT_ACCESS_READY https://remote.invalid:55468"), false);
  assert.equal(re.test("Listening on https://127.0.0.1:55468"), false);
  assert.equal(shell.params.on[0].done, true);
})().catch(error => { console.error(error); process.exit(1); });
''', str(_ROOT / "agent_access.js")],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr
