# Maestro Changelog

All notable changes to Maestro are documented here. The upstream WanGP
pipeline's own history lives in [app/docs/CHANGELOG.md](app/docs/CHANGELOG.md).

## [1.2.7] - 2026-07-16

Fix #17, the second domino behind #15 on Linked Model Folder installs:
the internal gemma folder v1.2.6 creates for the text-encoder weight
shadowed the linked install's complete folder for locate_folder, so
the tokenizer load crashed (sentencepiece 'not a string'). The
downloader now completes a partial target folder even when a linked
root holds the full set (self-healing, ~40MB once), and locate_folder
gained required_files so the gemma tokenizer lookups skip folders
without an actual tokenizer inside.

## [1.2.6] - 2026-07-16

Fix #15: on Linked Model Folder installs, text encoders (Gemma 13GB,
Qwen 8B) re-downloaded on EVERY generation and then crashed the load.
download_file moved the weight toward a folder that was never created
(the linked install had satisfied the tokenizer download read-only),
and shutil.move to a nonexistent directory renames the file to the
folder's own name - invisible to the locator forever after. The folder
is now created before the move, the misnamed leftover is cleaned up
automatically (existing victims self-heal), and a missing text encoder
raises a clear error instead of 'Loading Text Encoder None' plus a
TypeError two layers deeper.

## [1.2.5] - 2026-07-16

UI delivery hardening after a community black-screen report: MIME
types for the module bundle are forced server-side (Python reads them
from the Windows registry, which some machines have hijacked to
text/plain - browsers silently refuse module scripts served that way);
a boot watchdog replaces any silent load failure with a diagnostic
page after 10 seconds; and the /classic link works with or without
the trailing slash (the printed banner URL was a 404).

## [1.2.4] - 2026-07-15

Director art-style lock: a vision pass names the reference's medium
once per run and the validated lead sentence ("Maintain the same ...
art style.") is prepended to every image prompt deterministically at
generation time - trailing "preserve the art style" anchors provably
did nothing. Photographic references skip the prefix. Also: motion-
blur/speed-line language is stripped from start-frame prompts in code
(planner energy language leaked into stills), and the performer is
anchored to the reference image so the image model stops inventing a
new design for the star. See the
[README Updates section](README.md#updates).

## [1.2.3] - 2026-07-15

Community-driven round. Added: an Uploads view in the workspace
switcher (browse + reuse uploaded media), a manual model-unload button
in the System panel, and collapsible model families with whole-family
toggles (#14). Fixed: Director Stop aborts the in-flight clip instead
of letting it finish (#12); the Director composer auto-grows upward
(#11); stylized reference images keep their art style; instruction-
example content no longer bleeds into prompts (the dragon) and
user-specified locations are binding; speaker identification actually
runs now (checkpoints auto-download ungated) with music-tuned
clustering; the music Load Settings pencil restores caption, song
description, and the correct audio sub-tab. Changed: a page refresh
starts clean instead of restoring every edit (reverses v1.2.0
save-as-you-type restore; in-session mode-switch persistence stays).
See the [README Updates section](README.md#updates).

## [1.2.2] - 2026-07-14

Director "Analyzing" hang fix for smaller GPUs: the generation model's
VRAM is released before audio analysis loads the vocal separator and
Whisper (Windows' CUDA sysmem fallback made the overflow look like a
silent hang rather than an OOM). Also ships an int8 quanto variant of
the ACE-Step XL SFT transformer (5.5 GB vs 10 GB) so int8-quantized
installs download and load half the model.

## [1.2.1] - 2026-07-14

Fix for existing installs updating to v1.2.0: the enabled-models
whitelist stored in the browser never re-read the shipped defaults, so
the new ACE-Step XL SFT entries stayed hidden and the music default
stayed on Turbo. The curated defaults list is now versioned - new
entries merge into existing installs exactly once - and installs still
on the old music default follow it to XL SFT LM_4B with the model's
recommended settings applied.

## [1.2.0] - 2026-07-14

Two features: light themes (Ivory / Daylight / Pearl as daylight
variants of the three theme families) behind a Dark / Light / Auto
appearance mode that follows the OS, with a large legibility pass so
every status color works on paper; and ACE-Step v1.5 XL SFT, the
premium CFG music model, first shipped anywhere - consolidated weights
hosted at Blizaine/Maestro-Models, a new APG classifier-free guidance
sampling path, and set as the default music model.

Fixes: the vllm LM engine was silently disabled on Windows by a faulty
triton probe (song planning now dramatically faster); LM sampling
defaults now hydrate into the UI (temperature was stuck at 1.0);
Director planning crash on same-sized reference images + false OOM
popup; truncated song durations in the gallery (atomic audio writes);
edits persist as you type and the lyrics prompt survives refresh; new
ACE-Step models classify under Music. See the
[README Updates section](README.md#updates).

## [1.1.3] - 2026-07-12

Fixes: Director-mode start-image thumbnails no longer broken (uploads
endpoint falls back to output-workspace resolution, repairing existing
sidecars too); two-phase "a;b" LoRA multipliers accepted for
user-selected LoRAs on LTX-2 two-stage models (validation now uses the
model's phase capability instead of the request's guidance_phases);
Director LoRA selector uses theme-stable indicator colors so CivitAI
recommendations read green instead of amber on Golden Hour.

## [1.1.2] - 2026-07-12

Director dashboard repair arc: Re-join uses the real concat API with the
source song overlaid; clip reruns generate as a single window at full
planned length (a legacy 129-frame sliding-window default fragmented them
and kept only the first ~5s, breaking rejoin alignment and lip sync);
reruns record the final cumulative save; gallery refreshes after
dashboard actions. Verified end to end on a real 10-clip music video
(rejoined output sample-exact at 150.00s against the 150.00s song).

## [1.1.1] - 2026-07-12

Fixes: Director clip reruns keep the music video's soundtrack (sliced to
the clip's window); dashboard missing-count and Re-join repaired for
multi-clip runs (existing pipeline files backfilled on load); ACE-Step LM
runaway progress display corrected (generation was fine, the counter was
not); Auto-Tune now assigns audio its own memory profile so 12 GB+ cards
get the fast LM decoder instead of the legacy fallback. See the
[README Updates section](README.md#updates).

## [1.1.0] - 2026-07-10

See the [Updates section of the README](README.md#updates) for the
user-facing summary. Highlights: Linked Model Folders (reuse checkpoints
and LoRAs from other installs, read-only), Krea 2 models (Raw + Turbo),
10Eros v1.4 + Reference Pipeline toggle, the LTX-2 Dev quality fix
(leaked euler_ancestral sampler), working STG slider, Load Settings
pencil fix, theme contrast fix (#7), sticky NSFW toggles, and the UI
version badge backed by the repo-root VERSION file.

## [1.0.0] - 2026-07-08 - first public release

Initial public release of Maestro: a local AI video, image, and music studio
built on the [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) pipeline.

### Highlights

- **Studio mode** — manual generation across Video (Frames / Multi-Shot /
  Extend / Blend sub-modes, each with its own isolated working set), Image,
  and Audio. Unified media-driven Inputs panel: drop images/audio/video onto
  tiles and the pipeline (start/end frame, injected keyframes, soundtrack,
  control video, references) is selected automatically.
- **Director mode** — describe a music video or short film and a local LLM
  plans it end-to-end: writes the song (ACE-Step 1.5), analyzes the audio,
  plans per-clip prompts, and renders the full video. Multi-pass planning
  with JSON-grammar-constrained output for reliability on small local LLMs.
- **Music mode** — ACE-Step v1.5 XL music generation with an LLM song-writer
  (describe → Style + Lyrics, editable guide).
- **Edit modes** — Retake (regenerate a time region), Inpaint (SAM 3.1
  text-driven segmentation), Restyle, and Edit Anything (IC-LoRA).
- **Tools** — FlashVSR DiT video upscaling (2x/3x/4x, chunked for long
  videos) and SeedVC revoice with background preservation, usable on any
  gallery or uploaded clip.
- **Voice** — TTS voice cloning, per-speaker voice references, ID-LoRA voice
  identity preservation (experimental), cross-clip voice consistency.
- **Hardware auto-tune** — detects GPU/VRAM/RAM on first launch and picks a
  performance profile; OOM recovery banner with one-click fix.
- **LoRA management** — CivitAI browser with per-LoRA auto-generated prompt
  guides, weight recommendations, and per-checkpoint enhance guides.
- **100% local** — no telemetry, no accounts, no cloud dependency. Optional
  external LLM APIs are opt-in and off by default.

### Requirements

NVIDIA GPU (6GB+ VRAM; 24GB recommended for the full experience), Windows or
Linux, installed via [Pinokio](https://pinokio.computer). Models download on
first use per model (the default set is ~30GB; the full collection exceeds
300GB).
