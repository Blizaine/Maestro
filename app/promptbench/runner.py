"""Bounded localhost runner. Uses only the Python standard library."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, build_opener, ProxyHandler

from . import PROTOCOL_VERSION
from .checks import evaluate
from .evidence import digest, file_digest, snapshot_source
from .server import validate_params, validate_workflow
from .experiments import EXPERIMENTS

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = Path(__file__).with_name("cases.json")
WRITERS = {"qwen": "JonathanColetti/Qwen3.8-27B-Uncensored-GGUF",
           "gemma": "Abhiray/gemma-4-E4B-it-heretic-GGUF"}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_suite(path):
    path = Path(path).resolve()
    suite = json.loads(path.read_text(encoding="utf-8-sig"))
    if suite.get("version") != 1 or not isinstance(suite.get("cases"), list) or not suite["cases"]:
        raise ValueError("Expected a version-1 suite with a nonempty cases list.")
    ids = set()
    for case in suite["cases"]:
        name = case.get("id", "")
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in name) or name in ids:
            raise ValueError("Case IDs must be unique lowercase letters, digits and hyphens.")
        ids.add(name)
        if case.get("split") not in {"development", "holdout"}:
            raise ValueError(f"{name}: declare development or holdout split.")
        params = case.get("params") or {}
        for field in ("image_start", "image_end"):
            if params.get(field):
                params[field] = str((path.parent / params[field]).resolve())
        for ref in params.get("minimax_h3_references") or []:
            if ref.get("path"):
                ref["path"] = str((path.parent / ref["path"]).resolve())
        validate_params(params)
        validate_workflow(case.get("workflow", "studio"))
        checks = case.get("checks", {})
        for line in checks.get("exact_dialogue", []):
            if not line.get("speaker") or not line.get("text") or line["text"] not in params["prompt"]:
                raise ValueError(f"{name}: exact dialogue must occur in the source prompt and name its speaker.")
    return suite


class Client:
    def __init__(self, base):
        parsed = urlparse(base)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("Use the running Maestro http://127.0.0.1:PORT URL. This bench sends no prompts to remote servers.")
        self.base = base.rstrip("/")
        self.opener = build_opener(ProxyHandler({}))

    def call(self, path, data=None, timeout=30):
        request = Request(self.base + "/api/v1/" + path,
                          data=json.dumps(data).encode() if data is not None else None,
                          headers={"Content-Type": "application/json", "X-Maestro-Prompt-Bench": "1"})
        with self.opener.open(request, timeout=timeout) as response:
            return json.load(response)

    def busy(self):
        jobs = self.call("jobs")
        stream = self.call("llm/stream-status")
        if not isinstance(jobs.get("jobs"), list) or "done" not in stream:
            raise ValueError("Cannot establish GPU admission state; no experiment was submitted.")
        return any(j.get("status") in {"queued", "running"} for j in jobs["jobs"]) or stream["done"] is not True


def asset_fingerprints(cases):
    paths = set()
    for case in cases:
        params = case["params"]
        paths.update(params[key] for key in ("image_start", "image_end") if params.get(key))
        paths.update(ref["path"] for ref in params.get("minimax_h3_references") or [])
    return [{"path": path, "sha256": file_digest(path), "bytes": Path(path).stat().st_size} for path in sorted(paths)]


def select_cases(suite, names, split):
    if names:
        unknown = set(names) - {case["id"] for case in suite["cases"]}
        if unknown:
            raise ValueError(f"Unknown cases: {sorted(unknown)}")
    cases = [case for case in suite["cases"] if (not names or case["id"] in names) and case["split"] == split]
    if not cases:
        raise ValueError("No cases selected for this split.")
    return cases


@contextmanager
def run_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".run.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise ValueError("Run directory is locked. Inspect its runner/process before removing a stale .run.lock.") from error
    try:
        with handle:
            json.dump({"pid": os.getpid(), "created_at": stamp()}, handle)
        yield
    finally:
        lock.unlink()


def run(args):
    with run_lock(Path(args.output).resolve()):
        return _run(args)


def _run(args):
    suite = load_suite(args.suite)
    cases = select_cases(suite, args.case, args.split)
    writers = list(dict.fromkeys(WRITERS.get(w, w) for w in args.writer))
    if not 1 <= args.repetitions <= 10 or not 1 <= args.limit <= 100 or not 30 <= args.max_seconds <= 43200:
        raise ValueError("Use 1-10 repetitions, a 1-100 attempt limit, and 30-43200 seconds.")
    if not 30 <= args.case_timeout <= 1800 or not 1 <= args.max_calls <= 64:
        raise ValueError("Use a 30-1800 second case timeout and 1-64 LLM calls per case.")
    output = Path(args.output).resolve()
    client = Client(args.base_url)
    info = client.call("dev/prompt-bench")
    if info.get("protocol") != PROTOCOL_VERSION:
        raise ValueError("Bench protocol mismatch; restart Maestro after enabling the bench.")
    installed = {w["model_id"] for w in info["writers"]}
    if set(writers) - installed:
        raise ValueError(f"Writers not installed: {sorted(set(writers) - installed)}")
    specification = {"cases": cases, "writers": writers, "repetitions": args.repetitions,
                     "candidate": args.candidate, "experiment": args.experiment, "case_timeout": args.case_timeout,
                     "max_calls": args.max_calls, "source_digest": info["source_digest"],
                     "writer_assets": [w for w in info["writers"] if w["model_id"] in writers],
                     "runtime": info.get("runtime"),
                     "assets": asset_fingerprints(cases)}
    fingerprint = digest(specification)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise ValueError("Output already has a run. Use --resume or a new directory.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["specification_digest"] != fingerprint:
            raise ValueError("Cannot resume: source, cases, assets, writers or settings changed.")
    else:
        output.mkdir(parents=True, exist_ok=True)
        baseline = snapshot_source(APP_ROOT, output)
        if baseline["source_digest"] != info["source_digest"]:
            raise ValueError("Running server does not match the source snapshot; restart while idle before testing.")
        manifest = {"created_at": stamp(), "specification_digest": fingerprint, "specification": specification,
                    "baseline": baseline, "server": info, "generation_started": False,
                    "cloud_judging": False, "status": "ready"}
        write_json(manifest_path, manifest)
        write_json(output / "suite.json", {"version": 1, "cases": cases})
    started = time.monotonic()
    submitted = 0
    stop_reason = "complete"
    tasks = [(writer, repetition, case) for writer in writers for repetition in range(1, args.repetitions + 1) for case in cases]
    try:
        for writer, repetition, case in tasks:
            key = f"{case['id']}--{digest(writer)[:10]}--{repetition}"
            path = output / "attempts" / f"{key}.json"
            if path.exists():
                prior = json.loads(path.read_text(encoding="utf-8"))
                if prior["status"] in {"running", "uncertain"}:
                    stop_reason = "uncertain prior attempt; inspect server and record its outcome before any retry"
                    break
                if prior["status"] != "busy":
                    continue
            remaining = args.max_seconds - (time.monotonic() - started)
            if submitted >= args.limit or remaining < args.case_timeout:
                stop_reason = "configured attempt/time limit reached"
                break
            if (output / "STOP").exists():
                stop_reason = "STOP file requested; no new case submitted"
                break
            if client.busy():
                stop_reason = "user work pending; resume after it finishes"
                break
            record = {"key": key, "case_id": case["id"], "writer": writer, "repetition": repetition,
                      "created_at": stamp(), "status": "running"}
            write_json(path, record)
            submitted += 1
            print(f"Starting {case['id']} / {writer} / repetition {repetition}", flush=True)
            try:
                result = client.call("dev/prompt-bench", {"writer": writer, "params": case["params"],
                    "source_digest": info["source_digest"], "timeout_seconds": args.case_timeout,
                    "max_calls": args.max_calls, "experiment": args.experiment,
                    "workflow": case.get("workflow", "studio")}, timeout=args.case_timeout + 120)
                record.update(status=result.get("status", "failed"), result=result, assessment=evaluate(case, result))
            except HTTPError as error:
                detail = error.read().decode("utf-8", "replace")
                record.update(status="busy" if error.code == 409 else "failed", error=detail, http_status=error.code)
            except (URLError, TimeoutError, OSError, ValueError) as error:
                # A transport failure does not prove the server stopped. Never
                # auto-repeat a POST that may still own the GPU.
                record.update(status="uncertain", error=str(error))
            record["finished_at"] = stamp()
            write_json(path, record)
            report(output)
            print(f"Finished {case['id']}: {record['status']}", flush=True)
            if record["status"] in {"busy", "uncertain"}:
                stop_reason = record["status"]
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted; an in-flight running record must be inspected before retry"
    finally:
        manifest.update(status=stop_reason, updated_at=stamp())
        write_json(manifest_path, manifest)
        report(output)
    print(f"Run stopped: {stop_reason}. Report: {output / 'report.html'}", flush=True)
    return 0


def report(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    cases = manifest["specification"]["cases"]
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((output / "attempts").glob("*.json"))]
    # A report can be re-derived without changing raw results or rerunning an
    # inference. Keep the analyzer used for that report alongside the baseline.
    analyzer_files = (Path(__file__), Path(__file__).with_name("checks.py"))
    analysis = {"generated_at": stamp(), "source_sha256": {p.name: file_digest(p) for p in analyzer_files},
                "baseline_source_digest": manifest.get("baseline", {}).get("source_digest"),
                "policy": "Only entries with an effective payload count as dispatched LLM requests; nested delegation wrappers are listed separately."}
    for path in analyzer_files:
        target = output / "report-source" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    write_json(output / "report-analysis.json", analysis)
    esc = lambda value: html.escape(str(value))
    body = ["<h1>Maestro prompt bench</h1>", "<p>Enhancement only. Mechanical checks do not establish creative or rendered quality.</p>",
            f"<p>Candidate: {esc(manifest['specification']['candidate'])} · {esc(manifest['status'])}</p>"]
    md = ["# Maestro prompt bench", "", "Enhancement only; no video generated. Creative and rendered quality require review.", "",
          "| Case | Writer | Repeat | Status | Seconds | Calls | Mechanical failures | Warnings |", "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for case in cases:
        body.append(f"<h2>{esc(case.get('title', case['id']))}</h2><details><summary>Original prompt and review criteria</summary><pre>{esc(case['params']['prompt'])}</pre><pre>{esc(json.dumps(case.get('review', []), indent=2))}</pre></details><div class='grid'>")
        for record in [r for r in records if r["case_id"] == case["id"]]:
            result = record.get("result") or {}
            assessment = evaluate(case, result) if result else record.get("assessment") or {}
            label = f"{record['writer']} · repeat {record['repetition']}"
            body.append(f"<article><h3>{esc(label)}</h3><p>{esc(record['status'])} · {esc(result.get('seconds', '?'))}s · {esc(assessment.get('calls', '?'))} calls</p>")
            if assessment.get("warnings") or (result.get("prepared") or {}).get("enhancement_review_required"):
                body.append("<p><strong>Draft requires review before generation.</strong></p>")
            if record.get("error") or result.get("error"):
                body.append(f"<pre>{esc(record.get('error') or result['error'])}</pre>")
            for index, prompt in enumerate(assessment.get("native_prompts", []), 1):
                body.append(f"<h4>Window {index}</h4><pre>{esc(prompt)}</pre>")
            body.append(f"<details><summary>Checks and telemetry</summary><pre>{esc(json.dumps({k: v for k, v in assessment.items() if k != 'native_prompts'}, indent=2, ensure_ascii=False))}</pre></details>")
            body.append(f"<p><a href='attempts/{esc(record['key'])}.json'>Complete request, stages and response</a></p></article>")
            md.append(f"| {case['id']} | {record['writer']} | {record['repetition']} | {record['status']} | {result.get('seconds', '')} | {assessment.get('calls', '')} | {assessment.get('failure_count', '')} | {len(assessment.get('warnings') or [])} |")
        body.append("</div>")
    total = len(cases) * len(manifest["specification"]["writers"]) * manifest["specification"]["repetitions"]
    body.insert(2, f"<p>{len(records)} recorded attempts / {total} scheduled. Unrun cases remain untested.</p>")
    document = "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Maestro prompt bench</title><style>body{font:16px system-ui;background:#151515;color:#ddd;max-width:1800px;margin:auto;padding:24px}h1,h2{color:#ff9138}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:16px}article{background:#222;padding:18px;border:1px solid #555;border-radius:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.55 system-ui}a{color:#8bc6ff}details{margin:14px 0}</style><body>" + "\n".join(body) + "</body></html>"
    (output / "report.html").write_text(document, encoding="utf-8")
    (output / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_json(output / "summary.json", {"scheduled": total, "recorded": len(records),
               "analysis": analysis,
               "statuses": {status: sum(r["status"] == status for r in records) for status in {r["status"] for r in records}},
               "generation_started": False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("enable", help="Opt in locally; takes effect at the next idle server restart.")
    check = commands.add_parser("validate", help="Validate suite and required assets without loading an LLM.")
    check.add_argument("--suite", default=str(DEFAULT_SUITE))
    show = commands.add_parser("report", help="Rebuild local review report without any model calls.")
    show.add_argument("output")
    execute = commands.add_parser("run", help="Run fresh enhancements with a bounded, resumable plan.")
    execute.add_argument("--base-url", required=True)
    execute.add_argument("--suite", default=str(DEFAULT_SUITE))
    execute.add_argument("--output", required=True)
    execute.add_argument("--writer", action="append", required=True, help="qwen, gemma or exact installed repository ID")
    execute.add_argument("--case", action="append")
    execute.add_argument("--split", choices=["development", "holdout"], default="development")
    execute.add_argument("--candidate", default="baseline")
    execute.add_argument("--experiment", choices=EXPERIMENTS, default="baseline", help="Explicit request-scoped behavior; --candidate is only a report label")
    execute.add_argument("--repetitions", type=int, default=1)
    execute.add_argument("--limit", type=int, default=4)
    execute.add_argument("--max-seconds", type=int, default=4200)
    execute.add_argument("--case-timeout", type=int, default=900)
    execute.add_argument("--max-calls", type=int, default=32)
    execute.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "enable":
            target = APP_ROOT / "settings" / "promptbench.enabled"
            target.parent.mkdir(exist_ok=True)
            target.write_text("Local development prompt bench enabled.\n", encoding="utf-8")
            print(f"Enabled: {target}. Restart Maestro while idle to register the endpoint.")
        elif args.command == "validate":
            suite = load_suite(args.suite)
            print(f"Validated {len(suite['cases'])} cases and all declared reference assets. No models loaded.")
        elif args.command == "report":
            report(args.output)
        else:
            return run(args)
        return 0
    except (ValueError, OSError, HTTPError, URLError) as error:
        print(f"Prompt bench: {error}", file=sys.stderr)
        return 2
