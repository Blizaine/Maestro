# Maestro v2.0.0

Maestro 2.0 turns the app from a generation interface into a complete local
create-to-edit workflow. Director can develop the project, Studio can generate
or transform its individual assets, and the new Editor can assemble and finish
the result without leaving Maestro.

## The new Editor

Editor is now a full top-level Maestro mode alongside Director and Studio.

- Build edits on layered video, audio, and title tracks.
- Drag, trim, split, duplicate, copy/paste, move, and snap clips without
  changing the original media.
- Adjust speed, volume, opacity, fades, transitions, titles, fonts, and visual
  transforms from the clip inspector.
- Resize and position visual layers directly on the preview canvas with center
  and edge snap guides.
- Use markers, a draggable timeline playhead, keyboard shortcuts, undo/redo,
  track mute/lock, and project duplication.
- Browse media from the active workspace, every other workspace, uploads, or
  favorites.
- Import a complete Director production as separate shot clips. Music Videos
  also bring the full source song onto an audio track.
- Recognize Director productions by the real first frame of their first shot
  instead of a generic project icon.
- Choose **Edit clip with Maestro AI** to send a timeline clip through an
  appropriate Studio workflow and return the result as a new take in the edit.
- Export H.264, H.265, or AV1 with delivery resolution, frame-rate, quality,
  audio, and hardware-encoder controls. Exports can run immediately or wait in
  Maestro's universal queue.
- Edit standard, vertical, square, classic, and 21:9 ultrawide canvases.

The Editor is responsive: desktop keeps the complete multi-panel workspace,
while phone and tablet layouts turn the media browser, inspector, projects, and
timeline into focused panels suitable for quick edits and remote review. Its
iOS preview path uses a lightweight 30 fps proxy and starts playback directly
inside the initiating tap, avoiding Safari's delayed-playback throttling.

## A cleaner Studio

Studio workflows have been reorganized around what creators want to do:

- **Video:** create, continue with frames, extend, blend, retake, edit anything,
  outpaint, repaint, recast, upscale, and finish.
- **Image:** new image, edit, upscale, and outpaint.
- **Audio:** music, speech, sound effects, and revoice.

Video generation is now split cleanly between **Frames** and **References**.
Frames covers text generation, first/last and injected frames, control video,
and soundtrack-driven workflows using compatible H3 FL2VA and LTX models.
References contains H3 Omni and its character, image, video, and audio
conditioning tools. Image generation similarly uses Generate, Transform, and
Finish, with model choices narrowing automatically from the supplied media.

The old Studio Edit and Tools categories no longer compete with the new Editor
name. Reusable processing such as Film Grain now lives under Finish, where it
can be applied to any video—including a clip selected in Editor.

## Long-form creation without duration guesswork

- Choose one native window, a 30-second through 60-minute duration preset, an
  exact timecode, or a direct number of generation windows.
- Maestro translates friendly duration labels through the active model's real
  window, overlap, and discard geometry and always shows the resulting runtime
  and window count.
- **Auto** follows a soundtrack or other timed source exactly, honors explicit
  durations, and otherwise estimates a bounded runtime from visible story
  beats and exact dialogue.
- **AI — Faithful** preserves the user's ordered events and dialogue across the
  selected windows. **AI — Creative** can develop a concept with additional
  action and dialogue while keeping identity and continuity constraints.
- Compatible long-form audio workflows use the same duration language; native
  music-generator limits remain visible and enforced.
- Director stories longer than five minutes—including the 60-minute preset—are
  planned hierarchically as a complete-film story bible, one causal outline,
  chapters, and bounded screenplay sequences. The checkpointed bible locks the
  premise engine, tone, ending, canonical cast, location registry, persistent
  world rules, and forbidden story drift so each 90-second writing pass is
  making the same film rather than rediscovering it.
- Studio's hierarchical H3 story planner now carries the same story-bible,
  chapter-state, relevant-cast, and registered-location contracts into its
  bounded window batches, so hour-scale Studio sequences retain a global arc
  without asking one local-model response to author hundreds of prompts.
- Intentional recurring story machinery is now distinct from accidental
  repetition. A user-requested pattern such as visiting a new show, delivering
  a recurring line, and triggering a consequence can recur across its assigned
  chapters with a different setup, reaction, escalation, and handoff each
  time. One-time source events still occur exactly once in source order, and
  remain attached to their exact user-written dialogue.
- Chapter and sequence contracts now carry canonical location IDs, cast
  presence, inherited physical/story state, named character availability
  changes, and explicit recurring-motif IDs. A disappeared, dead, transformed,
  or injured character stays in that state until an explicit restoration event.
  Disappearances, injuries, knowledge, relationships, wardrobe damage, and
  prop ownership are handed forward instead of being reset at each local-model
  call. If a bounded H3 screenplay omits immutable dialogue after its focused
  retry, Maestro restores the exact words in canonical screenplay form.
- Expansion stories receive a focused pre-screenplay variety audit. If a local
  model still cycles through too few places after repair, Maestro schedules
  unused locations from the AI-authored story bible without inventing dialogue
  or plot. Per-sequence writers receive only their relevant cast and neighboring
  geography, while a final whole-film audit repairs phantom speakers, obvious
  name drift, overlong H3 prompts, and records non-fatal quality diagnostics in
  the resumable planning checkpoint.
- Every long-form sequence now carries its own explicit dialogue intent and
  timing target. Maestro rejects stale planning checkpoints, removes unowned
  event/dialogue references from neighboring chapter prose, and settles hard
  screenplay and spoken-word budgets before H3 dialogue becomes immutable or
  LTX image/video prompts are written. Long-form LTX sequences also receive a
  deterministic prompt-integrity pass: static image prompts, chronological
  video/window prompts, visible identities, opening causes, and outgoing
  handoffs are completed before a sequence is checkpointed. If a repaired H3
  visual plan supplies too few clips for the locked exchange, Maestro adds
  neutral same-scene coverage slots instead of rewriting or dropping dialogue.
- Director's hard minor-safety boundary now evaluates structured production
  plans one shot at a time and recognizes singular `minor` only when it refers
  to a person. Ordinary phrases such as "minor camera movement" can no longer
  combine with unrelated motion words elsewhere in a long plan and abort an
  otherwise valid generation.
- Long Director music-video, audio-film, and podcast timelines are planned in
  small batches. Planning progress is visible, completed batches are saved as
  they finish, and an interrupted run can resume at its first unfinished unit
  instead of discarding the complete outline and prior prompts. Resume now
  returns directly to the original Director chat, and a browser refresh
  reconnects its live gallery progress card to the server-side worker.
- Final long-form assembly now sanitizes empty dialogue rows returned by local
  structured-output models instead of crashing after every segment has
  finished. It also corrects unambiguous character-name typos, removes sound
  effects or camera directions miscast as speakers, rejects named phantom cast
  outside a complete story registry, and recompiles overlong H3 prompts from
  the authoritative shot fields while preserving multimodal references and
  exact dialogue. A durable quality report records duration, location and
  motif coverage, duplicate scene goals, prompt sizes, and applied repairs.
  Failed Director cards remain visible with the real error and saved counters,
  and can reopen or resume the original checkpoint directly.

Frequently reused Studio choices now survive browser refreshes and Maestro
restarts, including media mode, workflow, model, prompt-planning mode, H3
optimizations, and the open/closed Characters and H3 Optimizations panels.

## MiniMax H3 upgrades

- Added **768p**, the model's native trained resolution tier.
- Added **21:9** output canvases and the **H3 Regenerate 2K** workflow.
- Added separate Alibaba PAI eight-step acceleration presets for FL2VA and
  Ref2VA, including the official PDD parallel-head execution path.
- Improved Omni reference isolation so a character reference is not silently
  interpreted as the opening frame.
- Added a reusable named character library. Save an image and voice reference,
  or save a character video, then recall it in later Omni generations.
- Added optional per-character background removal to reduce color and scene
  leakage from identity photos while leaving the original reference available
  for cases where its lighting and context are useful.
- Locked every named character to one immutable Subject/Speaker ID and its own
  pictures, videos, and voice references. The enhancer rejects `<Subject N>`,
  unexpected Subject 3/4 entries, and cross-wired dialogue rather than passing
  an ambiguous two-character prompt into H3.
- Applied that same character binder at generation time, including when Prompt
  Enhance is off. Manual prompts can name the speaker naturally beside each
  quoted line or in the immediately preceding performance cue—even when the
  dialogue itself is already written as `<d>...</d>`. Maestro repairs a
  mismatched `(S#)` tag and rejects an ambiguous or nonexistent speaker instead
  of guessing by quote order.
- Automatically trims and budgets character video references to H3's supported
  per-reference and combined duration limits.
- Added stronger reference-role manifests, diagnostic logging, and tests for
  image, voice, video, full/pruned, and accelerated reference paths.

H3 prompt planning now follows the official Context-IR structure while keeping
Maestro's long-form strengths. Causal scene planning, character voice bibles,
dialogue table reads, locked spoken lines, conversation-aware shot packing, and
explicit opening/closing states improve the feeling that Director is making one
film rather than a collection of unrelated clips. Token fitting is based on the
actual tokenizer and prompt structure instead of a false universal 512-token
limit.

## Better local writing with Qwen3.8

Qwen3.8 27B Uncensored Q4_K_M is available as a local creative LLM, including
its vision projector. Maestro gives it bounded deep thinking for screenplay and
prompt-enhancement work, but disables thinking for grammar-constrained JSON and
deterministic repair passes. Quantized KV-cache planning targets useful context
on 24 GB systems, and generation telemetry separates reasoning tokens from the
final answer.

Gemma 4 remains the recommended fast default. The Director improvements are
shared across supported LLMs rather than being limited to Qwen.

## Completion alerts and private remote access

Maestro can now report completed, failed, and queued work through:

- in-app alerts;
- optional per-device chimes;
- a host-computer completion sound;
- browser system notifications; and
- encrypted Web Push for supported closed desktop browsers and an installed
  iPhone/iPad Maestro Home Screen app.

Optional Tailscale integration gives Maestro a private HTTPS address in the
user's own Tailscale network. It does not create a Maestro account, join a
Maestro-owned network, or expose Tailscale Funnel publicly. The one-time setup
remembers the selected backend port so the same saved address survives Maestro
restarts. Windows setup now installs a fixed on-demand restore helper, allowing
later Maestro starts to repair the private route without another UAC prompt.
See [Use Maestro Remotely with Tailscale](TAILSCALE_REMOTE_ACCESS.md).

## Interface and quality-of-life improvements

- New orange Maestro icon for the app, PWA, and shared header.
- Consistent responsive Director / Studio / Editor navigation, icon placement,
  and version display on desktop and mobile.
- Per-clip, multi-window, and complete Director ETA estimates, with observed
  First Block Cache acceleration reflected as generation progresses. A local
  timing history learns from comparable completed runs to improve later
  estimates without recording prompts or media paths.
- Expandable gallery details show generation model, resolution, LoRAs, H3
  optimization chips, effective prompts, scene/window timing, and per-window
  completion times. Gallery search can filter on those values, including terms
  such as Omni, Turbo, PDD, Sol, First Block, and LoRA names.
- Gallery activation follows the item being viewed or played; playback also
  activates and unmutes that media so Studio actions target the expected clip.
- Cleaner Pinokio Start and LoRA-folder menus without normal Classic UI links.
- Dynamic Pinokio ports remain the default unless a user explicitly enables
  persistent Tailscale access.
- The v1.9.1 llama.cpp nightly-download hotfix is retained, with a newer runtime
  floor needed by Qwen3.8.

## Updating

Use **Update** from Maestro's Pinokio page, then start Maestro normally. Existing
models, outputs, workspaces, Director projects, presets, and local settings are
preserved. A hard browser refresh may be needed if an old UI bundle remains
cached after the first v2.0 launch.

Tailscale is optional and is not installed or configured for users who do not
select **Secure Remote Access (Tailscale)**.

## Release validation

The v2.0 release candidate passes the production React build, the complete
frontend ESLint policy, launcher syntax checks, JSON grammar regression runner,
clean-repository boundary guard, first-party Python compile checks, and the
complete 1,207-test Python suite.
