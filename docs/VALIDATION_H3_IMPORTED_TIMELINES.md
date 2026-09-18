# H3 imported timelines — v2.2.3 fidelity and continuation fixes

Validated 17 September 2026 against v2.2.2; included in v2.2.3.

## Report and cause

A saved two-window enhancement rejected the second camera plan for dropping a
`before` relation. The source used the editorial comparison “~1s earlier than
before.” When source fragments were joined, the validator attached that word to
the next event as though it specified their order.

The same source used `0-3s: [TITLE]` headings. The line-timestamp parser required
a space before punctuation and did not recognize these headings. It produced
83 loose events instead of nine authored scenes. This also let setup headings,
prop descriptions and a truck-livery field enter the cast/dialogue maps.

## Changes

- Accept colon-adjacent line timestamps, optional bracketed titles, and common
  time-heading separators. Preserve exact source offsets and scene clocks.
- Keep vehicle/prop sections and trailing quality notes as production context.
  Honor an explicitly closed, declared cast without making camera/prop nouns actors.
- Distinguish clause-ending “than before” comparisons from real before/after
  dependencies. Genuine reversed action order still fails validation.
- Increment the planner version so fresh enhancement uses the corrected parse.

## Verification and limits

Replaying the exact saved source now yields nine timed scenes, one character,
zero spoken lines, correct source offsets and no false chronology dependency.
The original source and saved review were not edited.

The focused run passed 106 tests covering imported briefs, headings, cast,
dialogue and fidelity checks. The integration run passed 308 tests covering
story/window planning, Frames adaptation, camera coverage, reference speakers
and H3 runtime behavior. Eight new regressions are included in both runs.
`git diff --check` passed.

The initial parser-only checks did not run a live writer. The restarted user
retry correctly recognized all nine scenes but exposed a separate rejection:
a short nonverbal sound cue was required in visual action prose. The camera
writer used a synonymous sound in its effects field. That should not cause a
visual action repair or deterministic fallback.

## Follow-up: local audio and completed scene state

- Keep standalone source sound cues in their own event's sound-effects field.
  This is not permission to prove physical actions from SFX, another event, or
  global audio. Explicit before/after dependencies remain checked.
- For intermediate windows in a locked imported schedule, retain the camera
  writer's concrete completed state after validating the actual action cards.
  Previously the compiler discarded it and handed the next window a copy of
  the preceding event, inviting a second performance of the same action.
  The final source outcome and exact dialogue remain source-owned.
- Require explicit viewpoint ownership before adding foreground hands/props.
- When a paste explicitly separates a still-generation brief from a later video
  prompt, apply the still-only restrictions to that initial image; retain video
  restrictions and the complete source events. Ordinary first-frame instructions
  do not trigger this separation.

The expanded focused run passes 299 tests, including 18 new sound/handoff
regressions. Cases cover real omitted/reversed action, event-local audio,
still/video constraint scope, hand ownership, complete two-window planning,
completed-state handoff, and the final locked outcome.

Live writer and text-review results are recorded below. These are enhancement
tests only; no video was generated, and text checks cannot establish rendered
motion or visual quality.

## Live Qwen verification

Three bounded attempts used the same saved source and start image through the
app's prompt-bench preparation endpoint, with Qwen3.8 27B
(`JonathanColetti/Qwen3.8-27B-Uncensored-GGUF`, Q4_K_M, vision projector),
thinking off, temperature 0.7, top-p 0.8, top-k 20. Geometry remained 672 total
frames, 345-frame windows, 18-frame overlap, 1280x704, two native prompts.
The bench uses mature mode off and does not change user preferences.

| Attempt | Seconds | LLM calls | Observed result |
| --- | ---: | ---: | --- |
| Parser-only baseline | 106.813 | 4 | First Window 2 draft failed the sound check; focused repair succeeded. |
| Local sound fix | 92.109 | 3 | No warnings/repairs, but text review found a repeated door-opening action and bad handoff. |
| Sound + handoff corrections | 91.328 | 3 | No warnings/repairs; Window 2 starts with the eruption after the prior doors-open/candy-dumped state. |

Replaying the baseline's exact first three captured responses through the sound
fix also succeeds without requesting its fourth repair call. This controls for
writer variability when verifying the false rejection. The live timings are
single samples, not a general throughput estimate. All calls have complete
payload/token telemetry and completed without truncation. The last run used
16,417 prompt tokens and 2,976 answer tokens, with no reasoning tokens.

The final text retains the ordered arrival, filling, drinking, dumping, launch
and disappearance, keeps the camera inside the car, and preserves the cue in
Window 2's local audio. Its handoff no longer repeats the door opening or adds
the filmed subject's hands to the camera operator. The still-only prohibition
is absent from the video prompt. The source's copied vertical-format wording
still appears despite the selected landscape output geometry, and the shared
audio description includes later action sounds in the first window. Those are
separate remaining text-quality caveats, not the reported fidelity rejection.

Private source, image identity, all attempts, captured drafts, and rubric review
are retained under `.codex-tmp/promptbench/audio-fidelity-20260917-*`. No user
job/review was replaced. Maestro was restarted only while idle and now runs the
correction; the test writer was unloaded after completion. Nothing was published.

## Follow-up: ambiguous word-overlap rejections

Another ordinary user retry rejected Window 1's jug-filling action in both the
initial and repaired camera plans. The logged source fragment ended in a dangling
`Only`, split from `Only then`. The saved review contains the compiled fallback;
ordinary jobs did not capture the rejected writer responses, so those logs cannot
establish exactly which paraphrase or omission caused the failed overlap test.

The parser now keeps that transition together. More broadly, a low word-overlap
score is treated as a suspected omission, not proof that an action is absent.
Before the existing focused repair, one short non-thinking review checks only
flagged source requirements against the same event's action, framing and camera
fields. An acceptance must provide exact quoted evidence from those fields.
Other events, global notes and sound effects cannot stand in for visible action.

The review is capped at 12 flags per window. Plans without these lexical flags
make no extra LLM call. Missing/contradicted actions, invalid evidence, malformed
JSON or an unavailable reviewer continue through the existing repair/fallback.
Structural, dialogue, chronology and timing checks are not cleared by the review.
If repair changes an event's visual fields, that event's earlier approval expires.
Quoted evidence establishes provenance, not infallible semantic judgment; this
still needs live text review and does not guarantee rendered fidelity.

The focused suite passes 312 tests, including 11 new coverage-review tests and
two continuing-audio tests. The
existing genuine-omission test now exercises review followed by local repair and
still verifies that unrelated valid cards cannot be overwritten. Source, scenes,
dialogue, ownership, timing and continuation regressions remain covered.

A live follow-up exposed a sound-classification gap: `roar still going` was still
treated as visual action. Qwen requested a rewrite despite correct local audio;
Gemma accepted a visual description that only implied the sound. Continuing and
fading sound-only clauses now use the same deterministic audio retention as short
sound cues. Actor actions and before/after clauses remain excluded. Replaying
both exact captured three-call drafts now completes without review or repair,
retains the source cue in the final event's audio, and leaves visual action intact.

Four initial candidate enhancements (three Qwen, one Gemma) all completed without
fidelity warnings. This is operational validation, not full creative acceptance:
one Qwen repair added an early landing followed by further flight; another draft
described the camera outside the car in its handoff; Gemma inconsistently described
the jug as full after drinking. These text defects are recorded in the private
review and are not claimed fixed by accepting semantic paraphrases. The source's
vertical-format wording also continues to conflict with selected landscape output.

Final version 89 confirmations also completed without warnings or camera rewrites:

| Writer | Seconds | LLM calls | Coverage review |
| --- | ---: | ---: | --- |
| Qwen3.8 27B | 88.437 | 3 | Not needed |
| Gemma 4 E4B | 39.125 | 4 | 1.469 seconds; accepted `a quick, single rotation` as faithful to `spinning once` |

The final Gemma trace supplies a concrete positive paraphrase example, separate
from the earlier incorrectly classified audio cue. Its quoted action preserves
the requested single rotation. Both final native prompts retain continuing audio
locally without requiring it to be repeated in visible action. All seven live
attempts (one baseline, four first candidates, two final confirmations) and the
two deterministic replays are retained in private `coverage-fidelity-20260917-*`
artifacts. Six candidate enhancements completed warning-free, but this is not an
estimate of universal reliability or render quality. Gemma's final draft still
adds an orbit despite the in-car viewpoint and leaves a contradictory jug-state
handoff; the detailed private review records those remaining writing limitations.

The app was restarted through Pinokio while idle to load version 89. No saved user
job was replaced, preferences were not changed, and no release was published.
