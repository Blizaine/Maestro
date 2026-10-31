from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sqlite3
import sys

import pytest

_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.agent_access.lifecycle import LifecycleError, LifecycleManager


class FakeRuntime:
    def __init__(self, root: Path, *, backend=None):
        self.app_root = root / "app"
        self.directory = self.app_root / "settings" / "agent-access"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.backend_record = backend or {}

    def backend(self):
        return dict(self.backend_record)


class FakeRPC:
    _DEFAULT_PACKET = object()

    def __init__(self, root: Path, *, scripts=(), packet=_DEFAULT_PACKET, runtime=None):
        self.root = root
        self.scripts = set(scripts)
        self.packet = packet if packet is not self._DEFAULT_PACKET else {"type": "stream", "data": "accepted"}
        self.runtime = runtime
        self.calls = []
        self.path_override = None

    async def status(self, app_root: Path, pinokio_home: Path):
        return {
            "path": str(self.path_override or self.root),
            "running_scripts": sorted(self.scripts),
        }

    async def dispatch(self, operation: str, script_name: str, app_root: Path, pinokio_home: Path):
        self.calls.append((operation, script_name))
        if operation == "stop":
            self.scripts.discard(script_name)
            if self.runtime is not None:
                self.runtime.backend_record = {}
        else:
            self.scripts.add(script_name)
            if script_name in {"start.js", "start_sol.js", "start_classic.js"} and self.runtime is not None:
                self.runtime.backend_record = {
                    "ready": True,
                    "pid": os.getpid(),
                    "launcher_script": script_name,
                }
        if self.packet is None:
            return {"ok": True, "dispatched": True, "packet": None}
        return {"ok": True, "dispatched": True, "packet": self.packet}


@pytest.fixture
def installed(tmp_path):
    home = tmp_path / "pinokio"
    root = home / "api" / "Maestro-dev"
    app = root / "app"
    home.joinpath("api").mkdir(parents=True)
    app.mkdir(parents=True)
    for script in ("pinokio.js", "start.js", "update.js"):
        (root / script).write_text("// fixture\n", encoding="utf-8")
    runtime = FakeRuntime(root)
    return home, root, runtime


def manager(home, root, runtime, rpc, *, idle=None, git_clean=True, reconcile_timeout=0.15):
    return LifecycleManager(
        runtime,
        idle_probe=idle or (lambda: {"idle": True, "certain": True, "reason": "queues are idle"}),
        rpc=rpc,
        pinokio_home=home,
        git_clean=lambda _root: git_clean,
        reconcile_timeout=reconcile_timeout,
    )


def test_status_reports_only_verified_local_pinokio_state(installed):
    home, root, runtime = installed
    rpc = FakeRPC(root, scripts=("agent_access.js",))
    result = asyncio.run(manager(home, root, runtime, rpc).execute("status", "owner", None))
    assert result["state"] == "stopped"
    assert result["agent_access_terminal"] is True
    assert rpc.calls == []


def test_start_uses_fixed_script_and_duplicate_request_is_not_dispatched_twice(installed):
    home, root, runtime = installed
    rpc = FakeRPC(root)
    lifecycle = manager(home, root, runtime, rpc)
    first = asyncio.run(lifecycle.execute("start", "owner", "start-1"))
    replay = asyncio.run(lifecycle.execute("start", "owner", "start-1"))
    assert first["state"] == "accepted"
    assert "completed" not in first["state"]
    assert replay["replayed"] is True
    assert rpc.calls == [("start", "start.js")]


def test_start_preserves_the_last_registered_launcher_profile(installed):
    home, root, runtime = installed
    runtime.backend_record = {
        "ready": False,
        "pid": None,
        "launcher_script": "start_sol.js",
    }
    rpc = FakeRPC(root, runtime=runtime)
    result = asyncio.run(manager(home, root, runtime, rpc).execute("start", "owner", "start-sol-profile"))
    assert result["state"] == "accepted"
    assert rpc.calls == [("start", "start_sol.js")]


def test_start_receipt_is_committed_before_rpc_dispatch(installed):
    home, root, runtime = installed
    calls = []

    class ReceiptCheckingRPC(FakeRPC):
        async def dispatch(self, operation, script_name, app_root, pinokio_home):
            with sqlite3.connect(runtime.directory / "lifecycle-receipts.sqlite3") as database:
                state = database.execute(
                    "SELECT state FROM lifecycle_requests WHERE principal_id=? AND request_id=?",
                    ("owner", "start-before-rpc"),
                ).fetchone()[0]
            calls.append(state)
            return await super().dispatch(operation, script_name, app_root, pinokio_home)

    rpc = ReceiptCheckingRPC(root)
    result = asyncio.run(manager(home, root, runtime, rpc).execute("start", "owner", "start-before-rpc"))
    assert result["state"] == "accepted"
    assert calls == ["dispatching"]


def test_stop_requires_certain_idle_evidence(installed):
    home, root, runtime = installed
    rpc = FakeRPC(root, scripts=("start.js",), runtime=runtime)
    runtime.backend_record = {"ready": True, "pid": os.getpid(), "launcher_script": "start.js"}
    lifecycle = manager(home, root, runtime, rpc, idle=lambda: {"idle": False, "certain": False, "reason": "Director queue status is unknown"})
    with pytest.raises(LifecycleError, match="Director queue status is unknown"):
        asyncio.run(lifecycle.execute("stop", "owner", "stop-uncertain"))
    assert rpc.calls == []


def test_lifecycle_mutation_refuses_foreign_work_even_if_local_queue_looks_idle(installed):
    home, root, runtime = installed
    rpc = FakeRPC(root)
    lifecycle = manager(
        home,
        root,
        runtime,
        rpc,
        idle=lambda: {"idle": True, "certain": True, "foreign": True, "reason": "another engine owns the queue"},
    )
    with pytest.raises(LifecycleError, match="another engine owns the queue"):
        asyncio.run(lifecycle.execute("update", "owner", "foreign-work"))
    assert rpc.calls == []


def test_restart_waits_for_stop_reconciliation_then_starts_same_profile(installed):
    home, root, runtime = installed
    runtime.backend_record = {"ready": True, "pid": os.getpid(), "launcher_script": "start_sol.js"}
    rpc = FakeRPC(root, scripts=("start_sol.js",), runtime=runtime)
    result = asyncio.run(manager(home, root, runtime, rpc).execute("restart", "owner", "restart-1"))
    assert result["state"] == "accepted"
    assert rpc.calls == [("stop", "start_sol.js"), ("start", "start_sol.js")]


def test_lost_acknowledgement_is_uncertain_and_never_retried(installed):
    home, root, runtime = installed
    rpc = FakeRPC(root, packet=None)
    lifecycle = manager(home, root, runtime, rpc)
    first = asyncio.run(lifecycle.execute("start", "owner", "lost-ack"))
    replay = asyncio.run(lifecycle.execute("start", "owner", "lost-ack"))
    assert first["state"] == "uncertain"
    assert replay["state"] == "uncertain"
    assert "inspect status" in replay["message"].lower()
    assert rpc.calls == [("start", "start.js")]


def test_unknown_rpc_packet_is_uncertain(installed):
    home, root, runtime = installed
    rpc = FakeRPC(root, packet={"type": "telemetry"})
    result = asyncio.run(manager(home, root, runtime, rpc).execute("start", "owner", "unknown-packet"))
    assert result["state"] == "uncertain"
    assert rpc.calls == [("start", "start.js")]


def test_update_requires_clean_checkout_and_never_dispatches_when_dirty(installed):
    home, root, runtime = installed
    rpc = FakeRPC(root)
    lifecycle = manager(home, root, runtime, rpc, git_clean=False)
    with pytest.raises(LifecycleError, match="local changes"):
        asyncio.run(lifecycle.execute("update", "owner", "dirty-update"))
    assert rpc.calls == []


def test_update_uses_fixed_update_script_when_idle_and_stopped(installed):
    home, root, runtime = installed
    rpc = FakeRPC(root)
    result = asyncio.run(manager(home, root, runtime, rpc).execute("update", "owner", "update-1"))
    assert result["state"] == "accepted"
    assert rpc.calls == [("start", "update.js")]


def test_worktree_root_is_not_a_lifecycle_target(installed, tmp_path):
    home, root, _runtime = installed
    worktree = root / ".codex-tmp" / "worktree"
    (worktree / "app").mkdir(parents=True)
    runtime = FakeRuntime(worktree)
    rpc = FakeRPC(root)
    with pytest.raises(LifecycleError, match="installed Maestro"):
        asyncio.run(manager(home, root, runtime, rpc).execute("status", "owner", None))
    assert rpc.calls == []


def test_pinokio_must_report_the_exact_installed_root(installed):
    home, root, runtime = installed
    other = root.parent / "Maestro"
    other.mkdir()
    (other / "app").mkdir()
    rpc = FakeRPC(root)
    rpc.path_override = str(other)
    with pytest.raises(LifecycleError, match="exact installed Maestro app"):
        asyncio.run(manager(home, root, runtime, rpc).execute("status", "owner", None))


def test_foreign_or_unknown_running_script_blocks_mutation(installed):
    home, root, runtime = installed
    rpc = FakeRPC(root, scripts=("tools/custom.js",))
    with pytest.raises(LifecycleError, match="Another Maestro script is running"):
        asyncio.run(manager(home, root, runtime, rpc).execute("start", "owner", "foreign-script"))
    assert rpc.calls == []


def test_request_id_cannot_be_reused_for_a_different_action(installed):
    home, root, runtime = installed
    rpc = FakeRPC(root)
    lifecycle = manager(home, root, runtime, rpc)
    asyncio.run(lifecycle.execute("start", "owner", "same-id"))
    with pytest.raises(LifecycleError, match="another lifecycle action"):
        asyncio.run(lifecycle.execute("update", "owner", "same-id"))
    assert rpc.calls == [("start", "start.js")]
