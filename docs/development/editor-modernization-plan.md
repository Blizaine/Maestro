# Editor modernization and local auto-edit plan

Status: implementation started, 2026-10-28. Owner: Maestro development.

Build a readable, responsive timeline editor with non-destructive Maestro AI
operations. Complete and verify the manual editing foundation before enabling
automatic editing. Preserve existing projects, originals, Director provenance,
source audio choices, themes and mobile access. Public publication is a separate
release action; the current work targets local Dev review.

## Scope and sequence

| Phase | Work | Original recommendations | Completion evidence |
| --- | --- | --- | --- |
| 0. Project integrity | Explicit size/time validation, revisions/conflict protection, actionable save errors, recoverable project versions | 13 | CPU tests for valid legacy projects, oversized input, stale writes, atomic persistence and source preservation |
| 1. Workspace and selection | Resizable Media / viewer / inspector panes, full-width adjustable timeline, readable controls, narrow-screen drawers and mobile actions, visual takes and A/B review | 1, 3, 4, 8, 14 | Browser checks for resizing, persistence, keyboard/touch controls, selection, takes and desktop/mobile layouts |
| 2. Core editing and media | Adjustable tracks, visible trims/fades, gap operations, source In/Out preparation, media views/filters, timeline culling and bounded preview work/caches | 2, 5, 6 | Edit-operation tests, large-project browser fixture, proxy/cache checks, real CPU preview/export fixtures |
| 3. Finishing | Keyframed transforms/opacity/volume, basic color, correct transition semantics, audio envelopes/fades/meters, caption tracks and editable subtitle import/export | 10, 11, 12 | Shared evaluation/serialization tests, preview/export parity fixtures, audio/caption timing and undo tests |
| 4. AI within the edit | Contextual selected-clip/range AI workspace, explicit source/audio policy, durable operation-to-job identity, Director regeneration and take provenance, Media Flow returns | 7, 8 | Mocked queue/reload/cancel tests proving the exact result returns to its intended clip and original media survives |
| 5. Delivery and recovery | Export presets/readiness/stages, encoder diagnostics/cancellation, portable project packages, recovery and mobile completion checks | 13, 14 | Export and project-package round trips, cancellation/fallback tests, desktop/phone end-to-end checks |
| 6. Local auto-edit, last | Auto Assemble, Auto Polish, then optional missing-material proposals; natural-language revisions to an inspectable edit plan | 9 and auto-edit | Deterministic plan validation/application tests followed by bounded local-writer trials only when installed resources and idle state are verified |

Phases are dependency ordered. Independent UI and integrity work may proceed in
parallel. Keep a usable editor at each checkpoint; do not add decorative controls
without a working edit, preview, persistence and export path.

## Detailed acceptance criteria

1. **Workspace:** independently resize media and inspector widths and timeline
   height, preserve the viewer's usable size, save browser-local layout, provide
   reset/collapse controls and accessible separators. The timeline spans the
   available editor width. Intermediate viewports use drawers; phones use touch
   controls rather than squeezed desktop panes.
2. **Timeline:** readable filmstrips/waveforms/names, adjustable track heights,
   visible selected trim/fade handles, frame-aware snapping feedback, selection
   and context actions, gap removal and explicit source slip/rolling trims.
   Existing linked editing, markers, ripple behavior and undo remain intact.
3. **Visual hierarchy:** Graphite and other theme tokens remain authoritative;
   common controls use readable text and useful targets. Show state, shortcut and
   action labels consistently. Avoid reliance on color alone.
4. **Inspector:** selection-specific Basic, Audio and AI controls, with advanced
   sections; transforms and timing remain easy to reach. Preserve Director
   drafts, missing-media relink, track/project settings and current capabilities.
5. **Media preparation:** source preview with In/Out bounds, insert at playhead,
   grid/list and size controls, source metadata, project-use indicators and useful
   filtering. Previewing never adds media or changes the project by itself.
6. **Performance:** measure large-project interactions before claiming gains;
   cull off-screen timeline clips, deduplicate preview requests per source, bound
   concurrent preparation and cache growth, and avoid project-wide render/save
   work for playhead movement. Existing mobile proxy/deck behavior is preserved.
7. **In-editor AI:** keep the timeline accessible while configuring supported
   operations. Capture project/item/source revision, exact range, model settings,
   references and audio policy. A job receives an explicit operation ID; never
   infer ownership from the next unrelated queue item. Show progress/cancel and
   preserve drafts/results across reloads.
8. **Takes:** visual alternatives, synchronized comparison with a single audio
   master, source-specific trims, selection without destroying originals, and
   useful provenance. Director shot reruns retain their original context.
9. **Assistant:** reuse the final auto-edit operation model for commands such as
   shorten, reorder, insert a cutaway and adjust music. Apply accepted changes as
   a single undoable transaction while preserving unrelated manual edits.
10. **Audio:** clip/track gain, clear meters and fades, editable volume envelopes,
    normalization/ducking and range-based voice/music operations. Preserve source
    audio and distinguish it from generated replacements. Report analysis or
    processing progress and actual availability.
11. **Captions/text:** editable timed caption clips and subtitle import/export,
    consistent title styles and safe-area helpers. Transcription must use an
    installed supported service or clearly explain its unavailability; no model
    download or hidden generation on opening an editor panel.
12. **Finishing:** saved keyframes with linear/hold/ease interpolation; shared
    property evaluation for preview/export. Basic exposure/contrast/saturation
    and white balance. Transition overlaps have explicit timing and source-handle
    requirements and match preview/export behavior.
13. **Safety/delivery:** revision-aware saves, recovery copies, explicit validation
    errors instead of silent truncation, and portable media manifests. Export
    distinguishes project/output size and codec, confirms encoder readiness,
    exposes progress and cancellation, and classifies fallback failures.
14. **Mobile:** deliberate viewer/timeline/tool-sheet layout, responsive controls,
    larger trim targets, stable playback and recoverable projects. Validate both
    portrait and landscape without changing user preferences.

## Auto-edit architecture and release gate

Enable auto-edit only after phases 0–5 have working tests for the operations it
will call. A local writer proposes structured edit operations; the editor applies
them through the same deterministic command layer used by manual editing.

- Build an asset ledger from actual durations, source bounds, existing generation
  prompts, Director shot/timeline metadata, optional transcripts/music analysis
  and available vision analysis. Unknown content remains unknown.
- **Auto Assemble:** begin with Director shot packages, using known shot order,
  soundtrack timing and provenance. Support a target duration, aspect ratio,
  pacing brief and explicit media selection.
- **Auto Polish:** propose trims, pauses/gaps, captions, music levels and
  transitions on an existing timeline. Preserve locked tracks and user-pinned
  decisions.
- **Missing material:** propose additional shots/voice/music as separate queued
  generation operations with visible model/range/audio settings. Do not start
  generative work as a side effect of ordinary assembly.
- Store the brief, asset ledger version, plan, warnings, operation IDs and applied
  project revision. Show an understandable plan/diff and allow partial acceptance,
  rejection, revision and a single undo.
- Validate actual assets, finite times, source handles, durations and operation
  applicability. Repair recoverable structural issues within bounded attempts;
  retain useful drafts and explain unresolved issues. Do not reject creative
  choices through brittle prose rules.
- Revision conflicts must preserve the current timeline. Rebase or request review
  rather than silently overwriting intervening manual work.

## Verification and operating boundaries

Use synthetic projects and mocked browser HTTP first. Use temporary directories
for real CPU media fixtures. Run existing Editor import, Director draft,
persistence and export checks alongside new focused regressions. Exercise long
projects, invalid saves, missing media, queue interleaving and mobile layouts.

Do not benchmark performance by assertion alone: record project scale, source
format, cache state and hardware. Any live generation test must be bounded and
must not interrupt active or uncertain work. Do not download models, modify Main,
change OS/network settings, or push public branches as part of a local checkpoint.

## Progress

- Audit complete: current UI rendered with synthetic media; code-level integrity,
  queue-correlation and performance risks identified, not benchmarked incidents.
- Phases 0–5: implemented; final combined export/save gate in progress.
- Manual verification: 81 CPU tests passed, including real FFmpeg media fixtures.
  Browser checks cover layout, selection, source ranges, A/B takes, captions,
  animation, audio envelopes, recovery/packages, Director import/drafts and exact
  AI ownership, with all HTTP mocked. Portrait and touch landscape reviewed.
- Performance fixture: 2,400 clips across four tracks and 20 synthetic assets;
  60 initial / 52 far-scroll clip nodes, with distant selection verified. The
  measured 383 ms initial mount and 47 ms far-scroll render are synthetic browser
  observations, not playback or hardware benchmarks.
- Phase 6: waits for the final manual export gate; no auto-edit generation begun.

### Known verification limits

Actual GPU encoders, long 4K playback, installed speech recognition and creative
model quality still require user testing. Browser preview color filters and the
FFmpeg color pipeline are not certified pixel-identical. Preview peak metering is
pre-export; the final audio renderer applies a safety limiter, so deliberately
overloaded mixes can sound different. Reduce clipping before export. Dissolves
preserve authored transparency rather than forcing opaque sources. Current text
rotation is unsupported; legacy ignored rotation migrates visibly with a warning.

### Auto-edit first implementation contract

The first version has two paths. Director assembly uses known source/shot order
without a writer; creative assembly, polishing and revisions use an already loaded
local writer (or an explicitly configured private-network writer). Opening the
panel never loads a model, downloads weights or begins generation.

The writer receives a bounded, path-free ledger and returns whitelisted insert,
move, trim, remove, volume, gap and crossfade proposals plus separate missing-media
suggestions. It cannot run arbitrary tools or create media. Show action reasons,
timing changes, warnings and the limits of available content descriptions before
acceptance. Unknown video contents remain unknown. Locked tracks and pinned clips
are protected. Validate selected actions against the exact project snapshot; apply
them together through the manual command layer in a single undo transaction.

Keep failed writer drafts and reasons, permit one focused structural repair, and
let users revise the brief or accept only valid independent actions. Persist the
proposal and applied action IDs for later review. A changed project requires a new
review instead of silently applying an old plan. Missing-material suggestions do
not submit a render; ordinary Studio tools remain the explicit next step.
