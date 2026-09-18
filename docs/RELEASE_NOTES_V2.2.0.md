# Maestro v2.2.4

Released 18 September 2026. Includes the v2.2.0 feature release from 16 September.

This release brings one adaptive Enhance workflow, enhancement inside the
generation queue, YuE2 music and experimental personal music styles, a searchable
gallery across folders, and stronger control over Director music videos.

The complete v2.2.0 feature release remains below, together with the v2.2.1–v2.2.3
fixes and these v2.2.4 prompt-review improvements.

## v2.2.4 targeted retries and clearer prompt review

- **Retry only flagged windows:** when a camera or dialogue check flags part of
  an H3 sequence, retry those windows while preserving the accepted prompts
  exactly. The shared story, dialogue schedule and continuity boundaries stay
  in place. Supports Frames and References, Enhance now and queued enhancement.
- **Keep repair progress:** saved drafts retain the information needed for
  targeted retries after restarting Maestro. A second unsuccessful attempt stays
  focused on the flagged windows. If the prompt, references or timing change,
  create a fresh draft so the repair uses the correct inputs.
- **Make the scope of each action clear:** **Generate all N windows with this
  draft** uses the full saved draft, including flagged windows, without running
  enhancement again. **Retry window N & generate** repairs flagged prompts and
  generates the full job only when checks pass. **Edit prompts in Studio** opens
  the draft without starting generation. **Other options** contains full rewrites
  and generation from the original source.
- **Enhance-now retries remain a review step:** the prompt field's retry action
  repairs flagged windows without submitting a generation job.
- **Silent product timelines remain visual:** fixes the exact product-brief
  parsing case from #115, where `Logotype timeline:` became a speaker and its
  visual directions consumed dialogue capacity. Silent-video instructions can
  include duration, aspect and product-style qualifiers; explicitly written
  dialogue still remains intact.

Use Pinokio's **Update**, restart Maestro and refresh the browser. Drafts created
before this update need a fresh enhancement to enable targeted repair. Manual
prompt edits invalidate the saved repair checkpoint; the edited prompts remain
available for generation. Shared story-level issues can still require a full
rewrite. No new dependencies or model downloads are required for this update.

See [Studio controls](Studio-controls.md), the [enhancement API](Studio-enhancement-api.md),
and the [validation record](VALIDATION_V2.2.0.md).

## v2.2.3 H3 enhancement and memory fixes

- **Fewer false camera-fidelity warnings:** imported `0-3s: [TITLE]` storyboards
  keep their authored scenes. Revision wording, production headings, written
  logos and sound cues are no longer mistaken for extra actors, speech or missing
  physical actions. A bounded meaning check can accept a faithful camera
  paraphrase using evidence from that event; actual omissions still need repair.
- **Complete conversations fit more reliably:** retain dialogue introduced by a
  named reaction, distribute whole turns across windows, and preserve every exact
  line in order. Scene-wide directions no longer consume their own action slots;
  brief facial reactions share spare time while physical actions retain theirs.
- **Correct reference and voice ownership:** queued enhancement recognizes the
  supplied names, upload ordering and reference roles. Character subjects remain
  separate from vocal-event IDs; invented voice-reference clauses are corrected
  using the actual uploads. Markdown wrappers are removed from native prompts.
- **Clear duration feedback:** if exact supplied dialogue cannot fit the chosen
  duration, explain the word/time conflict before loading the LLM. Keep the source
  unchanged and suggest more time or another window.
- **Less unnecessary rewriting:** stop repeated calls made solely to meet a
  preferred dialogue density. Copyedit overlong AI-written speech first and batch
  remaining shortening requests. Exact user lines, topic ownership and hard
  timing checks remain protected; this is not a promised fixed speedup.
- **Better continuation and prompt cleanup:** the next window inherits what the
  preceding scene accomplished. Still-image-only constraints stay scoped to the
  start image; prop-holding hands keep their owner. Spoken reports of arrivals
  do not become on-screen entrances, and nonverbal prop motion/sounds survive.
- **Lower H3 normalization memory peaks:** process large hidden sequences in
  bounded slices while retaining native RMSNorm precision and inference hooks.
  Addresses the allocation hotspot in [#139](https://github.com/Blizaine/Maestro/issues/139)
  without reducing reference detail or changing VRAM budgets. The isolated
  operation is validated; a full A100 reproduction remains unverified.

After updating and restarting Maestro, refresh the browser and enhance again
from the original source to use the corrected planner. Saved drafts remain
available; they are not silently rewritten. No additional dependencies or model
downloads are required for these fixes.

See the validation records for [reference dialogue](VALIDATION_H3_REFERENCE_DIALOGUE.md),
[imported timelines](VALIDATION_H3_IMPORTED_TIMELINES.md), and
[memory and retry behavior](VALIDATION_H3_MEMORY_AND_LATENCY.md). Warning-free
enhancement does not guarantee perfect generated staging or lip sync.

## v2.2.2 enhancement and Director fixes

- **Fewer false review failures:** distinguish writing instructions, conversation
  topics, character descriptions and background context from actions that must
  appear on camera. Straight, curly and mixed quotation marks retain complete
  user-written lines; descriptive speakers such as "the older man" keep ownership.
- **More practical dialogue timing:** keep useful shorter exchanges after bounded
  repair instead of rejecting them for missing a preferred minimum word count.
  Preserve conversation order, allocate speaking time and rewrite crowded AI-only
  lines when necessary. User-written quotes are never shortened automatically;
  genuinely overfull or incomplete drafts can still require review.
- **More reliable multi-window planning:** allow the required source events and
  supporting beats to fit the story schema, preserve ongoing conversation turns,
  and repair misplaced window labels when source events remain intact and ordered.
- **Working review continuation:** immediately show and poll accepted draft retries
  without waiting for queue-history refresh. Submission errors remain visible,
  including on mobile where controls stay open until a job is accepted.
- **Director vocal ownership:** retain analyzed vocal activity through planning,
  start images, ending poses and final H3 prompt compilation. During instrumental
  passages, the singer listens or moves with closed lips; instrumentalists do not
  acquire invented singing, bellows or vocal breaths. User-requested expressions
  and wind-instrument playing remain supported. Replan existing Director projects
  to apply the updated guidance and vocal evidence.
- **Developer tooling:** the prompt bench recognizes cached vision-projector
  aliases, including existing Qwen3.8 installations, just as normal loading does.

These changes improve reliability; LLM writing and generated staging still need
human judgment. The validation record separates warning-free enhancement from
creative quality and rendered results.

## v2.2.1 numbered-shot fix

**AI enhancement hotfix:** imported numbered shot headings such as
`SHOT 1 — 0:00–0:02` now retain their authored structure, including Markdown
variants and Windows line endings. Camera settings such as motion blur, light
rays and depth of field carry into their own shots without depending on the
writer to repeat each phrase. This fixes false story-schedule and camera-fidelity
warnings while preserving checks for missing actions, event order and exact
dialogue. Applies to Enhance now and Enhance on generation. Retry enhancement
from the original prompt to replace an older draft.

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

v2.2.4 fixes the remaining silent-product dialogue classification reproduced from
**#115** and adds targeted window retries for drafts needing local camera repair.
The reported prompt is covered by parser and three-window scheduler regressions.
The broader product-mode and entity-handling request in **#138** remains open.

v2.2.3 addresses the large-sequence RMSNorm allocation reported in **#139**.
The bounded operation and numerical equivalence are tested; complete generation
on the reporter's A100 remains for validation, so this is not a blanket OOM fix.

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
