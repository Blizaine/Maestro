# Maestro v4.0.0 — Dev release candidate

29–30 October 2026. The next major release is available on **public Dev** for
testing. Public Main remains v3.2.5 until its separate release review. This
candidate combines the changes below with the fixes already tested on Dev.

## Changes since v3.2.5 Main

- **A refreshed workspace:** Graphite Studio, an accent-matched M mark,
  resizable gallery thumbnails and a cleaner panel toggle. Studio keeps its
  generation controls above the resource monitor, with compact duration,
  enhancement and model controls. Startup and mobile recovery show loading
  states while controls and the workspace become ready. Near-widescreen videos
  use their native proportions to avoid narrow black gaps inside gallery cards.
- **A larger Editor upgrade:** a media bin, preview, Inspector and multi-track
  timeline; more direct trimming, linked edits, transitions, keyframes, color
  and audio controls; media previews/proxies, project recovery and export
  options. AI round trips retain source/take information. Local Auto-edit
  proposes a reviewable plan before changes are applied and supports undo;
  it does not silently replace an authored timeline.
- **Animated titles in Studio and Editor:** describe an animation in your own
  words, optionally choose a style, and use AI planning when wanted. Frames
  generation uses a matching canvas rather than unrelated character references.
  Choose chroma-key green/blue or an opaque black, white or custom background.
  The Editor can composite keyed overlays and export transparent ProRes 4444
  MOV files. H3 itself produces a background; transparency comes from keying.
- **More image creation and editing:** MOSS Edit SFT/HQ v5 with ordered
  references, HQ vision conditioning and optional caches; Fibo 2.0 Dev and
  Turbo editing; Qwen Image 2.1 INT8 reference caches in VRAM or RAM. Krea 2
  guides and updated FLUX/Qwen/image-writer guidance preserve the requested
  artistic medium. Fixes cover FLUX 3 model switching, MOSS mode switching and
  rectangular Fibo VAE decoding.
- **H3 workflows:** experimental Pika H3 Flex for Frames and References, now
  included in their default model lists. Its fixed six-evaluation recipe uses
  SLA on supported runtimes with a dense fallback. Experimental sliding
  windows carry recent motion frames, a boundary frame and stereo audio;
  References also retains the ordered source references. Additions include
  workflow-matched LightX2V accelerators, PTD RefAsymOPD presets and compatible
  two-pass latent refinement. Existing model-specific recipes remain separate.
- **Pika Flex controls:** Frames and References offer 480p, 540p, 720p, 768p
  and experimental 1080p. Reusing a native clip restores its single window;
  resolution and aspect changes retain explicitly chosen window lengths and
  rolling-window geometry. Unlocking the window still resumes Auto planning.
  Incompatible First Block and extra Turbo settings no longer carry into Flex;
  SOL and SLA attention remain available.
- **Finishing and experimental tools:** FlashVSR+ Turbo joins the existing
  upscaler, with fixes for its checkpoint layout and legacy v1.1 attention.
  Worlds provides queued camera moves and continuation when enabled in Settings;
  it is off by default. Profile 0/VMM is opt-in with compatibility gates.
  Optional kernel updates retain supported fallbacks. Successful HTTP access
  logs no longer bury generation progress; warnings and failures stay visible.
- **Runtime safeguards:** GGUF prompt embeddings decode only the requested
  token rows in bounded chunks. Legacy MMGP avoids allocating new pinned
  staging buffers when host RAM is low and reports allocation failures more
  clearly. Native H3 start-image bookkeeping keeps its audio and video on the
  same output timeline. These changes are being checked in the live matrix;
  they are not a universal low-memory or speed guarantee.
- **Director planning:** standalone style and continuity labels stay in the
  creative brief instead of becoming mandatory performed story actions.
  Actual character actions retain their ordered source obligations and physical
  state validation. This reduces unnecessary planning conflicts without
  accepting invented contact or missing physical state. Repair instructions
  distinguish proximity from actual touch and avoid placeholder support facts.
  Silent H3 stories with a single action now use the same constrained JSON
  planning path as longer action sequences, so physical typing is applied
  before a draft is returned.
- **PTD/PDD scheduling:** small schedule and adapter-plan tensors stay on CPU
  even when the renderer's default tensor device is CUDA. The schedule math
  and hop selection are unchanged; this avoids unintended GPU allocation
  while preparing relative heads.
- **H3 reference preprocessing:** image and video preprocessing stays on CPU
  even under a CUDA default tensor device. The vision encoder still receives
  the explicitly selected device and precision. This avoids unintended
  preprocessing allocations; it is not a general low-memory guarantee.
- **Reliable startup:** the API reserves its actual bind interface through
  server startup and announces the Maestro URL only after its listener is
  ready. Busy ports fall back without taking over a shared Pinokio service.

See the [October port scope](WanGP-October-2026-Ports.md),
[MOSS guide](MOSS-Image.md), [Fibo guide](Fibo-2.md),
[Pika Flex guide](guides/pika-h3-flex.md), and [Studio controls](Studio-controls.md)
for model-specific inputs and limitations.

## Qualification and limits

The release candidate is undergoing a bounded overnight check on the RTX 4090
(24GB), RTX 3080 Ti (12GB) and RTX 5070 Laptop (8GB), alongside CPU and isolated
UI regressions. Results must be reviewed before promoting Main; this document
does not claim that the overnight matrix has already passed.

Pika continuation has two decoded 480p functional checks on the 4090. Those
checks preserve the tested frame counts and actual motion/audio history, but
camera framing can drift and audio levels can change at joins. They are not a
seamless-quality or speed guarantee. Quantized caches may fall back or cost
transfer time. New image editors can still change composition or lettering.
Worlds is queued generation, not real-time navigation. Profile 0 has no
certified GPU OOM-recovery result and excludes Fish/CUDA graph capture.

Experimental features do not establish every GPU, model, long Director project,
Linux configuration or output-quality claim. Keep important projects and compare
actual outputs before adopting a new workflow.

## First opening v4

Maestro switches to **Graphite Studio** with **Dark** appearance once, including
for people who previously selected another theme, Light mode or Auto mode.
The change applies before the UI paints so the initial view matches the theme
shown in Settings.

Afterward, choose any theme or Dark, Light or Auto appearance in Settings.
That choice survives refreshes and later v4 patch updates. The introduction
is remembered per browser and site address; clearing site storage or opening
another address creates a new browser profile. Other settings stay intact.

## Testing Dev

Update the idle Dev installation through Pinokio, restart Maestro and refresh
the browser. The version next to the Maestro title should read **v4.0.0**.

For the earlier Dev port scope and validation, see the
[v3.3.0 porting notes](RELEASE_NOTES_V3.3.0.md). The
[Pika H3 Flex guide](guides/pika-h3-flex.md) describes its experimental Frames
and References recipes, attention choices and compatibility fallbacks.
The [complete changelog](../CHANGELOG.md) retains earlier version entries.
