---
name: maestro-promptbench
description: Run and review bounded Maestro prompt-enhancement experiments with installed local writers. Use for baseline comparisons, creative continuity evaluation, and prompt-workflow tuning.
---

# Maestro prompt bench

Use the repository's `app/promptbench` runner. Read its [usage guide](../README.md) and [rubric](../rubric.md) when operating it. When this skill is installed outside the repository, locate the user's Maestro checkout and read `app/promptbench/README.md` and `app/promptbench/rubric.md` there instead.

The user-facing enhancement code is the experiment subject. Keep the source prompt and declared reference roles separate from expected results and evaluation instructions. The local endpoint uses the same preparation path as queued Enhance, returns its final parameters, and never submits video generation. It selects one installed local writer within the ordinary GPU slot without saving preferences. It may release idle cached models and unloads its test writer before returning the slot.

## Run

1. Read the saved experiment plan and choose the requested cases, writer(s), repetitions, attempt limit and time limit. If none are specified for a pilot, use two cases, one attempt for each of Qwen and Gemma, four total attempts, and a 70-minute batch ceiling. This is a bounded enhancement pilot, not overnight scheduling or permission for cloud calls.
2. From `app`, run `python -m promptbench validate --suite PATH`. Required missing media must be corrected or marked untested; never replace it silently. Check the app's current jobs before an idle restart or test. Use its managed runtime and its actual local URL.
3. Run `python -m promptbench run --base-url URL --suite PATH --output PRIVATE_DIRECTORY --writer qwen --writer gemma --limit 4 --max-seconds 4200`. Select `--case` explicitly when using the larger suite. The runner freezes relevant source including uncommitted files, request inputs and asset hashes. Do not edit experiment sources during the run.
4. If user work is pending, leave the batch ready to resume; do not cancel their work. `--resume` reuses the same run specification. A `running` or `uncertain` attempt is not safe to retry automatically: inspect the server and its logs first. A `STOP` file in the output directory prevents new requests; an in-flight call retains the slot until it finishes or hits its server-side limit.

Run artifacts contain private prompts and media paths and belong under the repository's ignored `.codex-tmp/promptbench/` or other private storage. Do not copy credentials, model weights or user media into tracked files. No API key is needed for the local runner; this Codex task still uses its configured model and account limits.

## Review

Open `report.html` for side-by-side native prompts and inspect `attempts/*.json` for effective LLM payloads, intermediate outputs, call metrics and errors. A warning-free response is not creative acceptance. Use the rubric to review source intent, cause and effect, speaker/voice/face ownership, physical blocking, persistent scene state, pacing and an ending. Quote short evidence and separate observed text defects from hypotheses about H3 rendering.

Give each case criterion a 0-3 score only where observable, with critical errors listed separately. State when a dimension is untested or not applicable. Record every attempt, including failures; never report only the strongest draft. For candidate comparisons, keep the rubric fixed, use blinded packets when available, randomize order, and check agreement with a human-reviewed sample before scaling automated judging.

The initial task establishes a baseline. Do not alter production guides, sampling defaults or validators during that baseline. A later authorized experiment changes one hypothesis at a time in isolated candidate source and records all repairs and final compilation. Holdout results do not guide edits. Text grading cannot establish motion quality, identity retention or lip sync; render review is a separate authorized stage.

Deliver a concise report with artifact paths, exact writer identities, cases/repetitions, latency and token coverage, concrete successes and defects, incomplete work, and the next test justified by the evidence. Do not claim an overnight run was scheduled or a candidate was released unless that action was separately performed.
