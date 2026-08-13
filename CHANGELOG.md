# Maestro Changelog

All notable changes to Maestro are documented here. The upstream WanGP
pipeline's own history lives in [app/docs/CHANGELOG.md](app/docs/CHANGELOG.md).

## [Unreleased]

## [1.8.1] - 2026-08-13

MiniMax H3 model sharing: linked WanGP installations can now supply their
pruned FL2VA and Ref2VA rank-8 INT8 ConvRot transformers as load-compatible
alternatives to Maestro's scaled-FP8 exports. Maestro also resolves the shared
Qwen3-VL encoder variants from WanGP's folder layout, prefers an exact Maestro
asset when both exist, detects the checkpoint's QKV layout before loading, and
prints the source of every H3 component. Differently published VAE artifacts
remain separate rather than being treated as unsafe duplicates.

Account-free installation: Install no longer starts Pinokio's Hugging Face
device-login flow. Maestro's default managed models download anonymously from
public sources. A clearly labeled optional menu action remains available for
custom gated assets or higher download limits, and missing LTX-2.5 components
now report the relevant public-download checks instead of incorrectly telling
users that an account is required.

## [1.8.0] - 2026-08-13

LTX-2.5: added native local support for the official Distilled workflow with
synchronized audio and WanGP/MMGP memory management. The recommended model
runs the official eight-step base pass, learned latent upscaling, and
three-step full-resolution refinement, with persistent model reuse for faster
follow-up jobs. Optional Dev and NVFP4 variants are available through Settings
without crowding the default model list, and Advanced offers the fast video
decoder or optional NAD diffusion decoder.

Complete LTX workflows: LTX-2.5 supports text and image generation, first and
last frames, timed frame injection, soundtrack and control-video audio
conditioning, sliding-window continuation, synchronized native audio, and
compatible LTX-2/2.3 LoRAs from the existing shared LoRA library. The model is
available to compatible Music Video, Short Film, and seamless Director paths.
INT8 ConvRot LoRA application now preserves the native linear forwards instead
of producing noise, and loaded LTX-2.5 profiles remain resident between jobs.

LTX long-form generation: every LTX video family now shares an explicit
Multi-window Sequence workflow with single-pass, AI-planned, and exact manual
prompt modes. Maestro computes the real overlap/discard geometry, validates one
manual prompt per window before loading a model, expands an overall idea into
chronological window-local prompts, and displays the reviewed prompts in the
main editor. Full temporal prefix frames and matching audio history now carry
across LTX-2.5 windows without the distorted seam or motion stall seen in the
initial implementation.

Audio and saved-settings reliability: corrected generated-audio sample-rate
handling that could make H3 and LTX sound slowed down, repaired standalone
soundtrack routing for loaded sidecars, and made audio strength use the actual
backend parameter. Load Settings now restores LTX window mode and geometry,
LTX-2.5 decoder choice, H3 Turbo preset, Sol Engine, First Block Cache, text
encoder, and their associated tuning values instead of leaking the prior
clip's state.

Runtime and interface polish: compatible RTX 40- and 50-series systems now
receive the tested Python 3.11 / PyTorch 2.10 / CUDA 13 Sol-capable runtime
through the normal Install, Update, and Start actions. The Studio sidecar is
ordered as Duration, inputs or Omni references, H3 Optimizations, then
Multi-window Sequence. Install, update, reset, recovery, and startup checks
cover the new runtime and LTX-2.5 assets while retaining safe compatibility
fallbacks.

## [1.7.5] - 2026-08-11

MiniMax H3 performance controls: added an optional experimental Sol Engine
sparse-attention path for supported RTX 40- and 50-series GPUs. Studio now
groups Turbo, Sol Engine, and First Block Cache in one collapsible H3
Optimizations panel and allows the three accelerators to be used independently
or together. Unsupported Sol calls fail safely to the normal attention path,
the startup audit reports runtime readiness, and the first compilation is
cached for later generations.

Director optimization parity: H3 Director projects now expose the same Turbo
preset, Sol Engine, and First Block Cache controls as Studio. The selected
recipe and cache tuning are persisted with the project and applied consistently
to initial shots, Dashboard regeneration, repair, and resume. Projects saved by
older Maestro versions remain compatible and default the new options off.

Managed H3 Turbo presets: updated the default to the upstream v4-600 EMA LoRA
at six steps and strength 1.0, while retaining the earlier v1-500 preset as a
rollback choice. Managed assets are revision-pinned, size- and SHA-256-verified,
published atomically, and recorded with local receipts. A scheduled GitHub
workflow detects upstream repository changes and opens a review issue rather
than silently replacing a validated adapter.

Runtime and update reliability: the tested Python 3.11 / PyTorch 2.10 / CUDA
13 runtime is now the standard Install, Update, and Start path on compatible
RTX 40- and 50-series systems. Existing RTX 40 installs migrate automatically
to `env-sol` while retaining `env` as an emergency compatibility fallback; the
separate Sol Start and installer choices are no longer required. Interrupted
runtime and FlashAttention installs remain resumable, NVIDIA drivers older
than 580 are handled safely, and RTX 20/30 systems continue to use the existing
SageAttention path. Bundled Sol sources include pinned upstream provenance and
Apache-2.0 notices.

## [1.7.2] - 2026-08-11

H3 compatibility fixes: repaired legacy Director and uploaded-media frame
counts onto H3's native frame lattice without timeline drift. Fixed manual
First / Last multi-window prompts being repeated wholesale in every window,
and corrected mixed FP16/FP32 visual conditioning for H3 GGUF text encoders.

## [1.7.1] - 2026-08-10

MiniMax H3 memory stability: fixed a workload inversion where a full-duration
540p generation could retain too many transformer weights, use oversized
projection chunks, spill into Windows shared GPU memory, and eventually run
out of VRAM even though the corresponding 720p job completed normally. H3 now
blends its measured runtime-workspace requirement smoothly into the residency
budget as clip load approaches a full native window and uses allocation-safe
chunks for long sequences. The corrected policy was validated with Full and
Pruned checkpoints in First / Last and Omni workflows.

## [1.7.0] - 2026-08-10

MiniMax H3 native continuation: First / Last and Omni can now build long
multi-window sequences while carrying a configurable span of recent video
motion and matching stereo audio into the next window. Shared Studio controls
support total duration, one editable prompt per window, optional AI planning,
and independent hard-cut sequences. The sequence planner now uses a persistent
story ledger to assign actions, dialogue, camera coverage, sound, and concrete
handoffs once across the timeline instead of repeating or prematurely
completing them. Exact-duration assembly and saved runtime prompts keep the
result reproducible.

H3 source-media workflows: added multiple timed frame injection for FL2VA,
soundtrack- and control-audio-driven video, video-to-audio generation that
preserves the source pictures, and video-to-video editing over the whole frame,
inside a white mask, or outside a white mask with denoise and masking controls.
Ref2VA music and performance references now advance through the source timeline
across sequence clips instead of replaying the first segment, while reusable
voice references retain their intended behavior.

H3 memory and hardware support: window recommendations now account for the
checkpoint family, output resolution, and detected VRAM, expose the native
14.4-second ceiling, and allow durable per-combination overrides. Transformer
residency, activation workspace, streaming VAE decoding, system-RAM ceilings,
and pageable LoRA fallback were refined for Full and Pruned checkpoints. RTX 50
/ Blackwell systems now use a dedicated Python 3.11, PyTorch 2.10, CUDA 13
runtime with compatible acceleration kernels, automatic Update migration,
startup compatibility diagnostics, and a launcher repair action; earlier RTX
families retain the established runtime.

Interface and reliability: invalid H3 media, prompt counts, durations, and
sequence combinations fail before expensive model loading. Omni's mixed-media
picker no longer prevents valid audio selection on iOS, and file extensions are
used when mobile document providers report ambiguous MIME types. Output cards
now report active generation time in minutes and seconds without including
queue or model-loading time. Regression coverage was expanded across native
continuation, reference packing, audio offsets, frame injection, video editing,
memory policy, RTX 50 setup, and the shared multi-window interface.

## [1.6.5] - 2026-08-08

MiniMax H3 performance and memory: rebalanced transformer residency and
activation workspace, added bounded QKV/MLP processing, and made Studio and
Director choose native H3 window lengths from the selected resolution, model,
and detected GPU memory. The main resolution menu now uses an aligned
1280x704 consumer 720p tier, exposes 1080p as an experimental option with
shorter hardware-aware windows, and keeps the former 768p tier available for
saved settings and API compatibility without presenting it as the default.
Users can lock a manual window override, and an optional experimental First
Block Cache offers selectable speed/quality thresholds.

H3 Turbo and LoRAs: Turbo now runs on both Pruned 20B and Full 33B checkpoints
through automatic AdaLN adapter conversion. The managed preset uses six steps
and a default strength of 0.50 while remaining editable in Advanced and
Director settings. Full-model jobs that are unnecessarily expensive receive a
Pruned recommendation instead of a hard failure. The CivitAI browser now has
an H3 filter, and both CivitAI downloads and pasted Hugging Face H3 LoRA URLs
route to the shared MiniMax H3 LoRA folder. Required conversion support assets
are revision-pinned, verified, and atomically published before generation.

H3 long-form prompting: Studio can turn one long-video idea into a structured,
window-local storyboard. Each continuation receives its own complete
Context-IR prompt with stable subject and setting continuity but distinct
actions, dialogue, camera coverage, ambience, effects, and music. This keeps a
multi-window story from finishing and repeating in its first pass. Exact
generated prompts are saved with the job, remain editable before submission,
show the currently generating window, and expand to their full content without
nested scrollbars.

Director H3 execution: Director now uses the same model-specific resolution,
frame-grid, VRAM, and Turbo rules as Studio. It divides long scenes into valid
native shots before queueing, rejects unsafe runtime shrinkage, preserves one
native pass for Turbo shots, and supports per-LoRA strength controls. Prompt-
only independent shots receive self-contained world, cast, wardrobe, blocking,
dialogue, sound, and continuity anchors rather than rolling-window commands.

## [1.6.1] - 2026-08-06

MiniMax H3 Turbo: added the pinned Turbo adapter to the Full FL2VA and Ref2VA
LoRA catalogs as a managed first-use download. Full H3 models now expose an
experimental one-click Turbo mode that selects the adapter, sets six inference
steps, and starts at strength 0.70. The adapter remains visible in Advanced so
its strength can be tuned per generation, and the backend preserves that
user-selected value while preventing duplicate Turbo adapters. Turbo remains
hidden and rejected for incompatible Pruned H3 checkpoints.

## [1.6.0] - 2026-08-06

MiniMax H3 in Director: added model-aware bounded-shot workflows for both H3
families. FL2VA now powers story-generated short films with native 5-15 second
shot planning and start/end continuity when a scene is divided into multiple
parts. Ref2VA now supports music videos, uploaded-dialogue films, and
story-generated films using per-shot composition, character, location,
soundtrack, and voice-reference manifests. Dashboard repair and regeneration
rebuild the same inputs, while audio-driven projects condition each shot on its
exact source segment and retain one clean continuous soundtrack for the final
join.

MiniMax H3 Omni Reference: added the separate H3 Base Ref2VA checkpoint and an
ordered Studio reference workflow for images, videos, embedded video audio,
and standalone audio. References receive exact Picture/Video/Audio labels,
can be reordered by drag and drop, and retain optional role notes for Prompt
Enhance. The runtime follows the official reference packing, VAE conditioning,
shared audio/video timing, and target-only denoising path while sharing the
existing H3 conditioner and VAEs. Output-matched reference detail is the
consumer-GPU default, with the official maximum-detail preparation available.

H3 Omni prompting: added a dedicated six-section Prompt Enhance guide that
maps ordered reference labels to subjects, motion, voices, retained details,
dialogue, soundscape, and music without changing the working FL2VA Context-IR
workflow. Standalone audio can be explicitly used as a voice reference,
performance-driving/reused audio, or a sound and music style reference. Raw
prompts now receive automatic media relationships, voice references no longer
copy source speech by default, scene ambience and effects begin at the first
frame, and malformed local-LLM enhancements retry or fall back safely instead
of being truncated into an unusable prompt. Standard H3 and Omni Prompt
Enhance now validate that every user-written line survives verbatim inside an
H3 dialogue block and that vague discussion requests receive actual scripted
dialogue. Raw Omni prompts are compiled into full six-field Context-IR and
identity pictures are prevented from introducing their source background,
framing, pose, or an opening still. Both H3 enhancers now allocate short
dialogue inside duration-aware speech intervals, fill the opening and remainder
with active nonverbal action, and explicitly suppress voices, grunts, breathing,
and speech-like filler outside dialogue tags. Dialogue is no longer duplicated
as ordinary quoted text, and visual terms such as cinematic or epic no longer
cause an unrequested musical score.

MiniMax H3 model and memory options: the existing FL2VA and Ref2VA entries are
now clearly labeled as the recommended Pruned 20B variants, with optional Full
33B entries for both workflows. Advanced settings can select the recommended
NVFP4-AWQ Qwen3-VL encoder or lower-RAM GGUF Q2/Q4, Quanto INT8, and BF16
alternatives. H3 now probes full versus pruned checkpoints at load time,
restores ConvRot layouts where needed, splits fused Q/K/V projections for
streaming, and profiles the Qwen language and vision towers independently.

MiniMax H3 Turbo LoRA: added the optional LarryVRH low-step adapter for Full
33B FL2VA/Ref2VA models with true 4/6/8-evaluation sampling and independent
video/audio schedules. Fixed active LoRAs bypassing the Full model's ConvRot
activation math and corrected fused-QKV adapter splitting, which previously
produced colorful tiled noise even though the same Full model worked without
the adapter. Incompatible Pruned 20B selections are rejected before loading.

H3 Omni video-reference memory: fixed Match Output references being silently
expanded to a 768-pixel short edge even for 480p/544p output. Reference video
area is now bounded to the requested canvas, long packed projections are
chunked, and video-reference jobs reserve dedicated attention workspace and
reload an already-resident profile when it was loaded with too much transformer
weight on the GPU. This substantially reduces first-denoise VRAM peaks while
keeping Maximum Detail available as an explicit high-memory option.

H3 Studio timing and continuation: Ref2VA/Omni is now limited to its native
single-shot maximum of 345 frames (14.375 seconds at 24 FPS), with incompatible
sliding-window controls hidden and rejected by the backend. FL2VA/First & Last
uses the same 345-frame native window but can continue longer Studio timelines
by feeding each completed window's final frame into the next. One-frame overlap
is removed during assembly, the optional end image is reserved for the final
window, and joined video and audio are trimmed to the exact requested duration.
Portrait, landscape, square, and automatic aspect-ratio selections now remain
native throughout the H3 pipeline.

Director planning and dialogue reliability: model selection is now filtered by
the capabilities required by each Director workflow, preventing image-only,
control-only, fixed-length, or native-audio-output models from being routed into
incompatible jobs. H3 story planning can omit unnecessary image generation,
retains project/world, wardrobe, blocking, and location context in every
independent shot, and compiles locked screenplay dialogue into stable speaker
IDs and native H3 dialogue blocks. Duration-aware shot coalescing permits
multi-speaker exchanges and internal camera changes while preserving complete
lines, and deterministic repair paths recover incomplete local-LLM plans without
silently changing, moving, duplicating, or truncating scripted dialogue.

Interface and diagnostics: simplified H3 model names distinguish First & Last
from Omni while explaining recommended Pruned 20B versus optional Full 33B
weights. Native audio-output badges are no longer presented as audio-input
support, Turbo LoRA compatibility is identified before generation, and
successful high-frequency system-stat polling is filtered from the console
without hiding errors or meaningful API activity. Saved Director jobs whose
process disappeared are now reported as interrupted instead of missing.

## [1.5.5] - 2026-08-04

MiniMax H3: added native local H3 Base FL2VA generation for text, first-frame,
and first/last-frame video with synchronized 32 kHz stereo audio. The initial
integration supports approximately 5-15 second output at 24 FPS across native,
portrait, square, and lower-VRAM resolutions, with revision-pinned automatic
provisioning of the compact scaled-FP8 transformer, NVFP4 Qwen3-VL conditioner,
video/audio VAEs, tokenizer, and processor assets. Ref2VA reference-video/audio
conditioning and hosted 2K regeneration remain outside this initial release.

H3 prompting: added a model-specific local Context-IR Prompt Enhance workflow
with the required multimodal-description, soundscape, music, stable speaker-ID,
and dialogue-tag syntax. Vague discussion requests can be converted into short,
duration-aware scripts; supplied dialogue remains verbatim; and unused time is
assigned to silent visible action to reduce invented speech. H3 enhancement now
bypasses the generic cinematic enhancer and remains one native timeline rather
than receiving sliding-window paragraph instructions.

H3 runtime reliability: corrected compact Qwen3-VL prompt conditioning,
row-scaled INT8 embedding loading, NVFP4 scale application, and causal attention.
Fixed mixed-dtype MMGP profiling and start-frame CPU/CUDA mismatches. Added
bounded activation chunking, explicit transformer working-memory reservation,
and dtype locks so the large packed audio/video sequence can stream on consumer
GPUs without exhausting memory before denoising. Expanded model-free and runtime
regressions for prompt conditioning, quantization, keyframes, scheduling, native
audio, activation memory, and Context-IR formatting.

SCAIL-2 Recast: improved continuous multi-character shots by detecting cast
transitions and supplying late-arriving identities through hidden pre-roll
conditioning instead of publishing artificial visible cuts. Recast assembly
now verifies every generation segment and preserves the exact source timeline.

## [1.5.0] - 2026-08-02

SCAIL-2 editing: rebuilt Recast around native replacement conditioning with
automatic reference isolation, face-detail conditioning, optional official
relighting, bystander preservation, and VRAM-aware 480p/512p/704p profiles.
Added stable color-mapped replacement for up to five people and shot-aware
SAM3 tracking so identities are reacquired and correctly routed across camera
cuts, close-ups, wide shots, and group shots. Added Repaint as a first-class,
shot-aware Edit mode that preserves the source timeline and audio while
changing characters, objects, or scene styling.

LTX-2.3 editing: rebuilt Outpaint around the official In/Outpainting IC-LoRA
and mask-preserving source conditioning, including bounded seam blending,
marker-spill cleanup, accurate canvas geometry, and model-correct sampling.
Multi-scene sources are now split at camera cuts, processed independently,
and reassembled at the exact original length with source audio. Retake now
supports distilled and two-stage LTX-2.3 pipelines. This resolves #28 and #37.

Krea 2: added RAW and Turbo Identity Edit v1.2 models with Qwen3-VL vision
conditioning, instruction editing, inpainting/outpainting, background removal,
and multi-reference support. Added current Diffusers/Kohya LoRA and GGUF
compatibility, a dedicated CivitAI/My LoRAs Krea 2 filter, accurate companion-
weight readiness checks, and default visibility for all four Krea 2 models.
This resolves #35 and #43.

Studio and reliability: model visibility now persists server-side across
Pinokio ports and restarts; newly installed CivitAI checkpoints appear without
a restart; control-video motion is independent of generated, uploaded, or
source audio; Temporal Depth assets are provisioned and verified on demand;
and Voice Reference is enabled independently of experimental features.
Director no longer duplicates single-clip outputs, SCAIL-2 LoRA phases are
normalized correctly, and installed apps survive early GPU-detection failures.
This resolves #19, #36, and #40.

## [1.4.0] - 2026-07-20

Storage and library management: added the Storage Manager with usage
analytics, safe workspace/pipeline/LoRA deletion, duplicate detection and
reclamation across linked installs, and opt-in linked-copy removal through the
Windows Recycle Bin. LoRA views now show sizes, download/release dates, age
chips, and newest-first sorting; CivitAI browsing is cached to reduce repeated
requests and rate-limit pressure.

Director workflow: reference-free runs now create and persist a shared visual
anchor before generating shot images. The Dashboard gained a server-owned,
cancelable repair workflow that skips valid work, survives browser reloads,
resumes interrupted batches, and rejoins completed clips. Fixed missing
thumbnails and clip mappings, generated start images not reaching video jobs,
repairs stopping after one item, and unsafe rejoin of missing or stale media.

Music-video timing: Dashboard reruns now use the same model FPS, frame lattice,
carried frame schedule, and audio window as the original Director run. Rejoin
also preserves the planned source-audio origin while retaining one continuous
soundtrack, fixing shortened replacement clips, cumulative lip-sync drift, and
leading-silence offsets without reintroducing audible clip-boundary artifacts.

Reliability and safety: job cancellation is terminal and race-safe, pipeline
state writes and output ownership are deterministic, and failed media joins
clean up partial files. Model/LoRA downloads now validate complete payloads and
archives before atomic publication, prevent concurrent destination writes, and
offer clearer progress and retry states. Restored expanded Director minor-
content scanning, fixed conditional React hook crashes, tightened NVIDIA-only
launcher gating, and enabled Python regression tests on both public branches.

## [1.3.3] - 2026-07-17

Recast tracking resilience (cocktailpeanut's second report). When the
replace target left the scene mid-clip, or was absent from frame 0,
SAM3's propagation crashed the whole job with "No points are provided".
The mask driver now anchors on a frame where the keyword actually
detects, propagates both directions, and re-anchors past a mid-video
tracking collapse, keeping all masks produced so far; absent-target
frames get empty masks (original footage passes through). Also clamps
the batched grounding chunk window to the video length (latent upstream
IndexError exposed by re-anchored propagation). Both root bugs are
inherited from upstream WanGP's SAM3 tree.

## [1.3.2] - 2026-07-17

Community-report round. Fixed: the first Recast on a fresh install
crashed with "SAM3.1 checkpoint not found" (the masking pre-step runs
before the model download that carries the detector; it now fetches it
on first use); downloaded badges lied for weight-aliased models
(SCAIL-2 Fast, Z-Image ControlNets) because the checker iterated the
alias string character by character - resolution is now recursive and
also counts weight modules and bundled LoRAs; deleting a finetune
leaves shared base weights in place for its siblings; SCAIL-2's
image-reference mask falls back to broader keywords when the configured
phrase matches nothing. New: the download icon in Settings -> System ->
Enabled Models is a real button that pre-downloads everything a model
needs (GPU-free, progress in the banner) via the new
/api/v1/models/{type}/download endpoint.

## [1.3.1] - 2026-07-17

Fix #20: a stale local Hugging Face token made HF reject public files
with 401 ("OAuth token signature verification failed"), surfacing as
"Repository Not Found" for the SCAIL-2 checkpoint. All model-download
paths now retry anonymously when the token is rejected; valid tokens
are still tried first so gated repos keep working. Also hides Recast's
inert resolution/window controls (the endpoint pins SCAIL-2's native
operating point).

## [1.3.0] - 2026-07-17

SCAIL-2 character animation, ported from upstream WanGP v12.3 onto
Maestro's engine with the SAM3 "Magic Mask" stack. Added: SCAIL-2 14B
and SCAIL-2 14B Fast (bundled lightx2v distill, 6 steps, ~13x faster)
as default-enabled Video models; the Recast sub-mode in the Edit tab
(replace a person in a video with a reference character, automatic
keyword-driven masking, preview, scene and audio preserved); a Control
Video input tile for guide-driven models; and a "use current frame as
reference" button on gallery videos. Hardened through field testing:
model-default hydration plus server-side operating guards (sliding
windows, source-fps follow capped at 30, audio remux, true duration
math), a SCAIL-2-aware VRAM budget (in-context tokens), GPU-serialized
Recast detection, and mode-scoped model selection and validation. See
the [README Updates section](README.md#updates).

## [1.2.8] - 2026-07-16

Fix #16: the My LoRAs library view only walked Maestro's own loras
folder while the guide scan and Studio selectors already enumerated
Linked Model Folders. The installed-LoRAs endpoint now uses the scan's
enumeration (primary + linked roots, deduped, mirror-joined sidecars/
guides) and entries carry a Linked badge in the browser.

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
