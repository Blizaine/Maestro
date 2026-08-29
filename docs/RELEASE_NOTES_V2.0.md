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

The old Studio Edit and Tools categories no longer compete with the new Editor
name. Reusable processing such as Film Grain now lives under Finish, where it
can be applied to any video—including a clip selected in Editor.

## MiniMax H3 upgrades

- Added **768p**, the model's native trained resolution tier.
- Added **21:9** output canvases and the **H3 Regenerate 2K** workflow.
- Added separate Alibaba PAI eight-step acceleration presets for FL2VA and
  Ref2VA, including the official PDD parallel-head execution path.
- Improved Omni reference isolation so a character reference is not silently
  interpreted as the opening frame.
- Added a reusable named character library. Save an image and voice reference,
  or save a character video, then recall it in later Omni generations.
- Locked every named character to one immutable Subject/Speaker ID and its own
  pictures, videos, and voice references. The enhancer rejects `<Subject N>`,
  unexpected Subject 3/4 entries, and cross-wired dialogue rather than passing
  an ambiguous two-character prompt into H3.
- Applied that same character binder at generation time, including when Prompt
  Enhance is off. Manual prompts can name the speaker naturally beside each
  quoted line; Maestro repairs a mismatched `(S#)` tag and rejects an ambiguous
  or nonexistent speaker before loading H3 instead of guessing by quote order.
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
remembers the selected backend port so the same saved address normally survives
Maestro restarts. See [Use Maestro Remotely with Tailscale](TAILSCALE_REMOTE_ACCESS.md).

## Interface and quality-of-life improvements

- New orange Maestro icon for the app, PWA, and shared header.
- Consistent responsive Director / Studio / Editor navigation, icon placement,
  and version display on desktop and mobile.
- Per-clip, multi-window, and complete Director ETA estimates, with observed
  First Block Cache acceleration reflected as generation progresses.
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
complete 1,081-test Python suite.
