# Maestro v2.2.0

Released 16 September 2026.

This release brings one adaptive Enhance workflow, enhancement inside the
generation queue, YuE2 music and experimental personal music styles, a searchable
gallery across folders, and stronger control over Director music videos.

## Unified Enhance and queued generation

- One adaptive writer develops short concepts and adapts detailed scripts to
  the selected model. Existing Faithful/Creative job settings remain compatible.
- **Enhance now** prepares a draft immediately. **Enhance on generation** lets
  you submit while the GPU is busy; the job enhances when its turn arrives and
  then generates. **Use by default** is optional, with a visible per-job override.
- Queued work retains its source prompt, references and generation settings.
  Prepared drafts and exact H3 window prompts can be reviewed, refreshed or
  reused. A completed draft is not automatically enhanced a second time.
- Enhancement errors and drafts requiring review stop for attention. Retry can
  resume the appropriate phase, and interrupted work is held after restarting.
- Completed jobs collapse into a counted history section with **View prompts**
  and **Clear completed**. Clearing history preserves generated media and projects;
  failed jobs remain visible for attention.

See [Studio controls](Studio-controls.md) and [enhancement API](Studio-enhancement-api.md).

## H3 prompt writing and continuity

Detailed production notes, role descriptions, visual instructions and silent
action stay separate from spoken dialogue. Exact supplied speech keeps its
speaker and timing; short creative ideas receive development without forcing
unnecessary dialogue into silent scenes.

Writing guidance now connects action preparation, contact, consequence and the
resulting scene state. It emphasizes actor/prop ownership, camera geography,
first-frame authority, explicit powers and limitations, and continuity between
windows. Timing and camera-plan repairs preserve usable writing where possible
rather than replacing a whole draft for a small structural mismatch. Larger
plans receive bounded writing budgets and duration-aware scheduling.

Director shares relevant adaptive guidance and preserves complete action,
camera, audio and ending descriptions through native H3 compilation. These
changes improve planning; they do not guarantee perfect choreography, actor
assignment or generated-video fidelity. Review remains available when needed.

The included [prompt bench](../app/promptbench/README.md) supports repeatable,
bounded local-writer experiments with saved intermediate/final prompts and
separate text and rendered-quality review.

## YuE2 3B music

- YuE2 is enabled and selected as the initial music model on install/update.
  Subsequent choices of another model are remembered.
- Generate **48 kHz stereo** songs from style descriptions and structured lyrics.
  **Direct generation** is the composition default. Melody/chord planning,
  melody-only planning, ABC scores and source-song covers remain optional.
- Song duration is a ceiling: generation may finish earlier at its musical
  ending. Models, source audio, scores and style settings are recorded in output
  metadata, with Editor and Director handoff support.
- Music controls and My music appear correctly after refresh. Missing/null
  composition settings now resolve to Direct generation instead of skipping the
  job; explicitly selected planning modes retain their values.

Optional score tools and model assets are acquired when their workflows need
them. YuE2 weights and real-audio tokenizer weights carry their own
**noncommercial terms**, separate from Maestro's code license.

## My music — Experimental

An optional YuE2 studio for learning personal music styles:

- Upload training and held-out recordings; play them, edit captions/lyrics and
  review local transcription drafts before preparation.
- Prepare real-audio training data with the pinned tokenizer, train style
  adapters, stop after a step, and resume from saved checkpoints.
- Optional aligned-lyric supervision and acoustic adaptation complement the
  style adapter. Training stays in the shared GPU workflow.
- Save checkpoints into the style library, create matched auditions with a
  fixed test song, and compare source reconstruction with adapter on/off.
- Import/export portable style bundles and control style strength during
  generation. Recording playback and audition/resume handling are fixed.

This feature is experimental. Style, rhythm, lyric alignment and vocal resemblance
vary by data and checkpoint. Our tests do not establish reliable cloning of a
specific singer on new songs. Reconstruction is a diagnostic, not proof of
new-song voice similarity. Local training was exercised on a 24 GB GPU.

See [YuE2 music and training](YuE2-music.md) for workflow, downloads and limits.

## Director music videos

- **Clip length** offers Auto or a custom model-aligned maximum. **Advanced →
  GPU clip limit**, beside steps, permits a remembered per-model override, such
  as 14.4s H3 Ref2VA on capable hardware. LoRAs/Advanced now sit directly below
  the model selector, before media inputs and analysis.
- **Cut Speed** works across −2…+2. Slower settings permit longer clips across
  musical sections. −2 uses the fewest clips that fit the selected maximum,
  placing nearby cuts on musical cues. In the tested two-minute song, a 14.4s
  cap gives nine clips of 12.3–14.2s, compared with thirteen at 0.
- Section changes, lyric phrases, performer changes and beat accents guide
  cuts. Internal section/percussion changes remain available to camera planning
  inside longer clips.
- Native generation padding is trimmed separately from visible clip timing,
  preserving full-song coverage, audio offsets, reruns and rejoining.
- Reference images reach the visual planner, improving identity guidance for
  lead singers and supporting musicians. Vocal-interval evidence discourages
  singing in instrumental passages. Instrument cutaways keep source vocals
  assigned to the singer off screen and direct non-singing musicians to keep
  their lips closed; explicit backing singers and wind instruments retain their
  intended performance.
- CPU audio analysis estimates sustained percussion and likely entrances.
  These are timing hints, not verified drum-stem separation or identification
  of every instrument.
- Per-shot **Save/Cancel** durably retains edits and invalidates stale prepared
  prompts. Focusing or typing in long text no longer jumps the sidebar away.

See [Director controls](Director-controls.md). Re-analyze/replan to apply new
analysis and prompt guidance to an older project; saved reviewed work is retained.

## Gallery, images and media

- **All folders** browses and searches the output library with bounded pages,
  origin-folder labels and folder-qualified actions. Browsing does not change
  the destination of newly generated media. Startup/refresh and stale-query
  handling keep the library from appearing empty after a refresh.
- Image generation adds **21:9**, repairs immediate enhancement, adapts detailed
  imported prompts to the selected image model, and exposes original/enhanced
  prompt details in the gallery.
- Multiline still-image prompts retain all lines; legacy image/prompt batching
  remains supported. Z-Image decoding matches the loaded VAE's precision.
- CivitAI imports rebuild changed architecture fields and reload affected warm
  pipelines when versions share a model ID.
- Repeated identical Editor uploads/relinks reuse content rather than creating
  another random-name copy. Existing duplicate files are not deleted, and the
  broader requested media-management controls remain separate work.

## TaoMate H3 three-step Frames

An optional experimental **TaoMate FL2VA — 3-step** adapter preset adds pinned
downloads, Euler/CFG settings and compatibility checks for H3 Frames. Existing
fused, PDD and Viggle paths retain their recipes. References, Viggle, VDN and
stacking another acceleration adapter are not offered with this preset.

Its internal token-refiner weights are included; this does not introduce a
separate second video-refinement pass or TaoLive's multi-GPU streaming runtime.
Local rendered checks cover pruned INT8 ConvRot, text/start-image input, 480p
and 124 frames on an RTX 4090. See [tested scope and provenance](TaoMate-H3.md).

## Runtime fixes

- Disable incompatible H3 PDD/audio-refinement combinations and clear stale
  refinement selections before generation.
- Avoid collapsing streaming-profile H3 residency when the estimated activation
  reserve cannot fit alongside the minimum weight slice. Manual preload remains
  authoritative; the reported RTX 5080 speed recovery still needs a hardware retest.
- Improve Windows HTTP connection recovery and keep local-writer loading off the
  web event loop while retaining ownership until loading actually finishes.
- Verify llama-server's actual CUDA devices on Linux. CPU-only cached runtimes
  upgrade through a compatible CUDA build when requested and prerequisites are
  present. Missing toolkit/compiler/driver libraries receive actionable errors.
- Correct the Qwen3.8 vision-projector filename while reusing an existing cache.
  Director can retry transcription on CPU/int8 for missing CUDA libraries,
  including errors raised during lazy segment iteration.

See [local LLM runtime](LLM-runtime.md). Native Linux CUDA build/generation still
requires reporter validation; mocked build tests are not a substitute for it.

## GitHub issues

Fixes and requested controls in this release address **#4, #121, #123, #124,
#125, #126, #127, #129 and #132**. Release comments describe the implemented
behavior and reproduction checks for each.

Related improvements also ship for **#95** (source/draft review), **#103**
(deduplicated uploads), **#115** (multi-window writing), **#131** (residency) and
**#135** (Linux LLM runtime). Remaining scope or reporter validation stays open.
The **#128/#130** stalls and **#134** visual-anchor request are not claimed fixed.

## Updating

Run **Update** in Pinokio, restart Maestro and refresh the browser. Models,
characters, generated media, workspaces, saved projects and personal training
data remain in place. Optional new models/tools download assets when used;
existing video/image workflows do not require those optional weights.

Run Enhance again from the original prompt, or create a new Director plan, to
apply the new writer guidance. See the [validation record](VALIDATION_V2.2.0.md)
for automated checks, previously exercised workflows and remaining limits.
