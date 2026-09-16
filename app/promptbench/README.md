# Maestro prompt enhancement bench

Development experiments are opt-in per request. `--experiment fight_choreography`
compares a more concrete action craft guide with the existing one, using the same
writer calls and timing code. It applies only where action craft is already selected.
`--experiment action_first` tests a
physical schedule before generated speech for multiwindow briefs without exact user
quotes. `--candidate NAME` remains a report label, not a behavior switch. Both the
experiment and exact source are saved in the run manifest; resume refuses changes.
Ordinary Studio/Director requests and the default `--experiment baseline` retain
the current production behavior. The candidate has not been approved by render tests.

Developer tooling for measuring H3 enhancement through Maestro's actual generation-preparation code. It captures a source snapshot, per-call effective payloads and drafts, final native prompts, warnings and telemetry. It prepares settings without starting video generation. GPT-5.6 Sol or another agent can operate it using [the operator skill](operator/SKILL.md).

## Setup

Use Maestro's existing Python environment. From `app`:

```sh
python -m promptbench enable
python -m promptbench validate --suite ../.codex-tmp/promptbench/my-suite.json
```

`enable` creates the ignored `app/settings/promptbench.enabled` marker. Restart Maestro through its normal launcher while it is idle. Ordinary installs do not expose the endpoint. Remove the marker and restart to disable it. No package installation, public release, global sampling change or API key is required.

## Pilot

Use the actual Maestro URL and an ignored output directory. From `app`:

```sh
python -m promptbench run --base-url http://127.0.0.1:PORT --suite ../.codex-tmp/promptbench/my-suite.json --writer qwen --writer gemma --case mountain-fight --case office-transition --limit 4 --max-seconds 4200 --output ../.codex-tmp/promptbench/pilot
```

Check `cases.json` for current IDs and copy the cases you need into an ignored private suite. The public Reference cases are text-only templates and deliberately do not contain a user's media path, so validating the unchanged public suite reports their missing media. `--suite PATH` supplies private cases/reference paths. Relative media paths resolve relative to the suite file. A case using an H3 **References** model must declare at least one existing image or video in `minimax_h3_references`; an audio reference alone is insufficient. Keep private media paths in an ignored private suite such as `.codex-tmp/promptbench/my-suite.json` rather than adding personal assets to the public fixtures. H3 **Frames** cases may remain text-only. The runner rejects missing or incompatible assets, unsupported generation controls, remote URLs, unknown writers, or writers whose model assets are not installed. It does not download weights. All experiments set mature mode off within their request only.

Defaults are one repetition, at most four attempts, 900 seconds and 32 LLM calls per case. The batch will not start another case unless its full case timeout fits inside the remaining batch time. Model startup/shutdown and cancellation cleanup can extend the wall-clock deadline; it is not a process-kill deadline. Model calls and context limits remain visible in the trace. A video seed in the fixture is not a guarantee that all planner calls use that seed; inspect each effective payload.

The first pilot is four complete enhancements. The subsequent baseline is eight cases × two writers × three repetitions (48 enhancements, potentially many more LLM calls). Use explicit larger `--limit`, `--repetitions` and time limits only for an authorized batch. Baseline calls are fresh: prepared-plan fields are rejected/cleared and backend prompt caching is disabled by the production path.

## Resume and review

For Director comparisons, set `"workflow": "director"` on a case (the default is
`"studio"`). This calls the real short-film planner, timeline preparation and H3
preflight without creating a saved project or starting a render. The result includes
the clip plans and exact timeline beside the final native prompts. Director chooses
shot boundaries, so use duration/continuity review rather than assuming Studio's
window count. Director currently supports the production baseline only. Its repair
diagnostics are in the captured writer calls, not Studio's warning ledger; the report
marks this coverage difference for review. Music-video planning has separate source
song contracts and is not exercised by this short-film workflow.

Re-run the same command with `--resume`. Completed and failed attempts stay recorded and are not repeated. Admission rejections (`busy`) can resume. An interrupted or transport-uncertain POST is never automatically repeated; inspect the server before resolving that record. Changed code, media, cases, writer assets or backend identity require a new run directory. A `.run.lock` prevents concurrent runners from using the same directory; remove a stale lock only after checking its PID and server state. Add a `STOP` file to the run directory to stop submission of new cases.

```sh
python -m promptbench report ../.codex-tmp/promptbench/pilot
```

- `report.html`: local responsive side-by-side prompts; model text is HTML-escaped and no network scripts are loaded.
- `report.md`, `summary.json`: run coverage and statuses, including failures and unfinished cases. Reports recompute mechanical checks from preserved raw results; `report-analysis.json` and `report-source/` identify the analyzer used. Nested delegation wrappers in early traces are listed separately from actual dispatched LLM requests.
- `manifest.json`, `suite.json`, `source/`, `source.patch`: baseline evidence including relevant uncommitted/untracked code. Large model files are identified by registry/path/size/mtime rather than copied or fully hashed; this limitation is explicit.
- `attempts/*.json`: original generation settings, prepared parameters, all captured calls and final checks. Each local writer call retains raw content and any returned reasoning before text cleanup, alongside the clean draft; interrupted streams are marked incomplete. Images in LLM payloads are represented by data-URL hashes instead of duplicate base64 blobs; reference files have independent SHA-256 hashes.

The bench pauses admission when user jobs are queued/running. The server owns the ordinary generation lock for every enhancement and its cleanup. Writer selection is scoped to that request and never written to preferences. Idle model residency can change, as in ordinary queued enhancement; the bench unloads its writer before releasing the slot. There are no separate uncoordinated GPU processes.

Mechanical checks identify format, missing exact speech, unexpected speech in silent scenes, window count, trace coverage and fallbacks. They do not certify speaker ownership, creative quality or rendered quality. Apply [the review rubric](rubric.md) separately. The report does not produce a misleading combined quality score.

## Developer API

The opt-in endpoint requires a loopback client and `X-Maestro-Prompt-Bench: 1`:

| Request | Purpose |
| --- | --- |
| `GET /api/v1/dev/prompt-bench` | Protocol, source fingerprint and installed local writer assets. |
| `POST /api/v1/dev/prompt-bench` | `{writer, params, source_digest, timeout_seconds, max_calls}`; returns prepared output and trace or a recorded writer failure. |

Generation fields are allowlisted in `server.py`. Cached plans, generation submission flags, preprocessors and remote writer credentials are not accepted. A 409 means no experiment started. A transport timeout does not mean the server stopped; the runner records an uncertain attempt and stops submitting.

For a matched Frames run, include the source job's `sliding_window_memory_override` along with its window size and overlap. Omitting a manual override lets the usual VRAM recommendation clamp the window size and can change the number of prompts. Keep the effective window geometry in the comparison; do not count a clamped probe as a settings-matched result.

## Validation

From the repository root:

```sh
app/env/Scripts/python.exe -m unittest discover -s tests -p test_promptbench.py
```

Use the corresponding `app/env/bin/python` on Linux. The tests exercise scoped/threaded trace capture, missing assets, HTML escaping, truthful grading, request isolation and non-duplicating resume without loading a model. Current scope is baseline acquisition and review; candidate author/adapter workflows, automatic cloud judging, render judging and overnight scheduling are later stages.
