"""Run bounded Maestro test actions on reachable Pinokio machines.

This controller never updates, starts or stops Maestro. Each remote launcher
runs its local loopback-only worker in that machine's managed environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.parse import quote, urlsplit
from urllib.request import urlopen

APP_ROOT = Path(__file__).resolve().parents[1]
MODES = ("smoke", "prompt", "render", "nightly", "compare")
TERMINAL = {"completed", "failed", "cancelled", "canceled", "error"}
PENDING = {"running", "uncertain", "dispatched", "restoration_pending", "interrupted"}
MAX_BYTES = 32 * 1024 * 1024
REF_PATTERN = re.compile(r"pinokio://[A-Za-z0-9.:-]+/api/[A-Za-z0-9_.-]+\Z")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def fetch(url):
    with urlopen(url, timeout=15) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("Response exceeds the 32 MiB artifact limit")
    return data


def api(base, path):
    return json.loads(fetch(base.rstrip("/") + path))


def file_url(base, filename):
    # A server receipt is data. It cannot select a different host or path.
    if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.(?:zip|html|json)", filename):
        raise ValueError("Invalid test artifact filename")
    return base.rstrip("/") + "/api/v1/file/" + quote(filename) + "?workspace=Test-Bench"


def pterm_status(executable, ref):
    result = subprocess.run([executable, "status", ref, "--probe"], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=30)
    if result.returncode:
        raise ValueError("Pinokio status failed: " + result.stderr[-800:])
    return json.loads(result.stdout)


def endpoint(status):
    local = bool((status.get("source") or {}).get("local"))
    candidates = [status.get("ready_url")] if local else []
    candidates += [entry.get("url") for entry in status.get("external_ready_urls", [])]
    for base in candidates:
        if not base:
            continue
        url = urlsplit(base)
        if url.scheme not in ("http", "https") or url.username or url.password or url.query or url.fragment:
            continue
        if not local and url.hostname in ("localhost", "127.0.0.1", "::1"):
            continue
        try:
            api(base, "/api/v1/system-stats")
            return base.rstrip("/")
        except (OSError, ValueError):
            continue
    raise ValueError("No caller-reachable Maestro URL; enable private Pinokio sharing")


def latest(base):
    try:
        return json.loads(fetch(file_url(base, "testbench-latest.json")))
    except (OSError, ValueError):
        return None


def source_url(status):
    for entry in status.get("local_entries", []):
        if entry.get("script") not in ("start.js", "start_sol.js"):
            continue
        value = (entry.get("local") or {}).get("url")
        if not isinstance(value, str):
            continue
        parsed = urlsplit(value)
        if parsed.scheme in ("http", "https") and parsed.hostname in ("127.0.0.1", "localhost", "::1") and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment and parsed.path in ("", "/"):
            return value.rstrip("/")
    raise ValueError("Pinokio did not report a captured loopback URL for the source machine")


def busy_jobs(base):
    result = api(base, "/api/v1/jobs")
    if not isinstance(result, dict) or not isinstance(result.get("jobs"), list):
        raise ValueError("Unknown jobs response; refusing to dispatch")
    return [job.get("job_id") for job in result["jobs"] if job.get("status") not in TERMINAL]


def collect(base, receipt, directory):
    collected = []
    for key in ("bundle", "artifact", "report", "artifact_filename", "report_path"):
        filename = receipt.get(key)
        if not filename or filename in collected:
            continue
        content = fetch(file_url(base, filename))
        (directory / filename).write_bytes(content)
        collected.append(filename)
    return collected


def run_target(executable, target, mode, directory, timeout, collect_only=False, state_path=None):
    ref = target["ref"]
    if not REF_PATTERN.fullmatch(ref):
        raise ValueError("Only canonical Pinokio app refs are accepted")
    directory.mkdir(parents=True, exist_ok=True)
    record = {"ref": ref, "label": target.get("label", ref), "mode": mode, "started_at": stamp(), "status": "checking"}
    receipt_path = directory / "receipt.json"
    saved = json.loads(state_path.read_text(encoding="utf-8")) if state_path and state_path.exists() else None

    def finish(status, reason=None):
        record.update(status=status, finished_at=stamp())
        if reason:
            record["reason"] = reason
        write_json(receipt_path, record)
        if state_path and status != "deferred":
            write_json(state_path, record)
        return record

    try:
        status = pterm_status(executable, ref)
        # Never wake, update or restart a machine during a recurring run.
        if not status.get("ready") or not status.get("running"):
            return finish("deferred", "Maestro is offline or still starting")
        base = endpoint(status)
        record["base_url"] = base
        before = latest(base)
        if collect_only:
            if not before:
                return finish("untested", "No saved test receipt")
            if before.get("status") in PENDING or not before.get("finished_at"):
                return finish("deferred", "The saved test has no terminal receipt; remote work was left intact")
            if saved and saved.get("status") in PENDING:
                if before.get("run_id") == saved.get("previous_run_id") or before.get("status") in PENDING or before.get("mode") != saved.get("mode"):
                    return finish("deferred", "Saved dispatch has no matching terminal receipt; inspect the remote logs")
            record["remote"] = before
            record["files"] = collect(base, before, directory)
            return finish("collected")
        if "test_bench.js" in status.get("running_scripts", []) or (before or {}).get("status") in PENDING or (saved or {}).get("status") in PENDING:
            return finish("deferred", "An existing test is running or requires reconciliation")
        active = busy_jobs(base)
        if active:
            return finish("deferred", "User jobs are pending: " + ", ".join(str(v) for v in active))
        worker_base = source_url(status)
        record.update(status="dispatched", previous_run_id=(before or {}).get("run_id"))
        # Persist BEFORE sending. A transport timeout must never cause a retry.
        write_json(receipt_path, record)
        if state_path:
            write_json(state_path, record)
        with (directory / "pinokio-action.log").open("w", encoding="utf-8") as output:
            process = subprocess.Popen([executable, "start", "test_bench.js", "--ref", ref,
                                        "--", "--mode=" + mode, "--base_url=" + worker_base],
                                       stdout=output, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + timeout
            exited_at = None
            while time.monotonic() < deadline:
                remote = latest(base)
                if remote and remote.get("run_id") != record["previous_run_id"]:
                    if remote.get("mode") != mode:
                        return finish("uncertain", "A different test mode published the new receipt; inspect remote logs before reconciliation")
                    record["remote"] = remote
                    if remote.get("status") not in PENDING and remote.get("finished_at"):
                        record["files"] = collect(base, remote, directory)
                        # The normal one-shot pterm process ends on its own.
                        return finish("collected")
                code = process.poll()
                if code is not None and code != 0:
                    return finish("uncertain", "Launcher returned an error; inspect logs and reconcile before retry")
                if code is not None:
                    exited_at = exited_at or time.monotonic()
                    if time.monotonic() - exited_at > 10:
                        return finish("uncertain", "Launcher ended without a new terminal receipt; update Dev or inspect logs")
                time.sleep(5)
        return finish("uncertain", "Controller deadline reached; remote work was left intact. Collect/reconcile before retry")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return finish("uncertain" if record["status"] == "dispatched" else "deferred", str(error))


def report(directory, records):
    write_json(directory / "fleet-results.json", {"created_at": stamp(), "targets": records})
    rows = []
    for record in records:
        remote = record.get("remote") or {}
        links = " ".join(f'<a href="{html.escape(str(index))}/{html.escape(name)}">{html.escape(name)}</a>'
                         for index in [records.index(record)] for name in record.get("files", []))
        cells = (record["label"], record["status"], remote.get("status", "untested"), record.get("reason", ""))
        rows.append("<tr>" + "".join("<td>" + html.escape(str(cell)) + "</td>" for cell in cells) + "<td>" + links + "</td></tr>")
    document = "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'><title>Maestro fleet tests</title><style>body{font:16px system-ui;background:#151515;color:#eee;padding:24px}td,th{padding:12px;text-align:left;border-bottom:1px solid #555}a{color:#ff993e}</style><h1>Maestro fleet tests</h1><p>Performance is compared within each machine. Cross-machine times are descriptive. Creative and video quality require review.</p><table><tr><th>Machine</th><th>Collection</th><th>Test outcome</th><th>Reason</th><th>Artifacts</th></tr>" + "".join(rows) + "</table>"
    (directory / "report.html").write_text(document, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(APP_ROOT / "settings" / "test_fleet.json"))
    parser.add_argument("--ref", action="append", help="Canonical Pinokio ref; overrides config")
    parser.add_argument("--mode", choices=MODES, default="smoke")
    parser.add_argument("--pterm", default=shutil.which("pterm"))
    parser.add_argument("--output")
    parser.add_argument("--timeout", type=int, default=2500, help="Per-machine controller ceiling; never kills remote jobs")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.pterm:
        parser.error("pterm was not found; run inside Pinokio or provide --pterm")
    if not 30 <= args.timeout <= 3600:
        parser.error("--timeout must be 30–3600 seconds")
    try:
        targets = [{"ref": ref} for ref in args.ref] if args.ref else json.loads(Path(args.config).read_text(encoding="utf-8"))["targets"]
        if not 1 <= len(targets) <= 4 or len({v["ref"] for v in targets}) != len(targets):
            raise ValueError("Configure 1–4 distinct targets")
        output = Path(args.output) if args.output else APP_ROOT.parent / ".codex-tmp" / "test-fleet" / datetime.now().strftime("%Y%m%d-%H%M%S")
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use a new output directory; previous receipts must not be overwritten")
        output.mkdir(parents=True, exist_ok=True)
        state_root = APP_ROOT / "settings" / "test-fleet-state"
        state_root.mkdir(parents=True, exist_ok=True)
        lock = state_root / "controller.lock"
        try:
            lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise ValueError("A fleet controller lock exists; inspect its process and receipts before removing it")
        with os.fdopen(lock_fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        records = []
        try:
            for index, target in enumerate(targets):
                if (output / "STOP").exists():
                    break
                print(f"Checking {target.get('label', target['ref'])}", flush=True)
                state_path = state_root / (hashlib.sha256(target['ref'].encode()).hexdigest() + ".json")
                records.append(run_target(args.pterm, target, args.mode, output / str(index), args.timeout, args.collect_only, state_path))
                report(output, records)
                print(json.dumps(records[-1], ensure_ascii=True), flush=True)
        finally:
            lock.unlink()
        print("Report: " + str(output / "report.html"), flush=True)
        return 2 if any(r["status"] == "uncertain" or (r.get("remote") or {}).get("status") in {"failed", "uncertain", "interrupted"} for r in records) else 0
    except (OSError, ValueError, KeyError) as error:
        print("test_fleet: " + str(error), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
