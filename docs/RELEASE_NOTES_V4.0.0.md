# Maestro v4.0.0

Released 4 November 2026 on Main and Dev. These notes describe the changes
since v3.2.5, including the latest reliability and reference-enhancement fixes.

## Changes since v3.2.5 Main

- **A larger Editor upgrade:** a media bin, preview, Inspector and multi-track
  timeline; more direct trimming, linked edits, transitions, keyframes, color
  and audio controls; media previews/proxies, project recovery and export
  options. AI round trips retain source/take information. Local Auto-edit
  proposes a reviewable plan before changes are applied and supports undo;
  it does not silently replace an authored timeline. Tracks are shorter and
  the timeline height is adjustable. [Editor guide](Editor.md).
- **Pika H3 Flex:** experimental Frames and References models, included in
  their default enabled lists. Its fixed six-evaluation recipe uses SLA on
  supported runtimes with a dense fallback. Sliding windows carry recent
  motion frames, a boundary frame and stereo audio; References also retains
  ordered source inputs. Exact source-audio conditioning and per-shot
  soundtrack offsets support Director music videos. Experimental 1080p joins
  the 480p, 540p, 720p and 768p choices. [Pika guide](guides/pika-h3-flex.md).
- **Built-in Agent Access/MCP:** independent Local MCP over STDIO and Remote
  MCP over authenticated HTTPS, with optional Tailscale setup, scoped approval
  and revocable clients. Agents receive model capabilities, prompting guides,
  exact timelines and saved references. They can author Studio prompts and
  Director productions directly, including lyrics and music-video plans using
  native song/beat analysis. Existing Pinokio and LAN browser access remains
  available. [Setup and client requirements](Agent-Access.md).
- **Optional ComfyUI integration:** discover and reuse configured model
  folders without moving weights, import API workflows with typed inputs and
  component pickers, and validate them before generation. Supported Studio
  routes and Director paths can use explicitly selected ComfyUI mappings.
  Native remains the default. ComfyUI and its nodes must be installed and
  managed separately; importing an arbitrary graph does not make it compatible
  with every Maestro feature. [ComfyUI guide](ComfyUI.md).
- **Pinokio Remote integration:** detects this installation's published browser
  address, offers a QR code and optional MCP access through the same private
  HTTPS route, and avoids a second Tailscale setup. Existing local, LAN and
  standalone Tailscale connections are preserved.
- **Source-driven Pika music videos:** Pika References now accepts an exact
  soundtrack as clean target-audio conditioning during all six sampler steps.
  Director and MCP preserve each shot's source offset. Fixes also cover false
  continuation detection, duplicate character bindings from generic reference
  labels, and native Director callbacks importing a second server instance.
- **LoRA and failure review:** optional experimental style/content LoRAs on
  FastH3, a shared Turbo LoRA filter, model-aware H3 adapter checks and improved
  handling of BF16-rounded checkpoint tables. Failed Studio cards retain the
  actual settings and LoRAs for inspection after a failure or restart. Editor
  tracks are shorter and its timeline height is adjustable.
- **A refreshed workspace:** Graphite Studio, an accent-matched M mark,
  resizable gallery thumbnails and a cleaner panel toggle. Studio keeps its
  generation controls above the resource monitor, with compact duration,
  enhancement and model controls. Startup and mobile recovery show loading
  states while controls and the workspace become ready. Near-widescreen videos
  use their native proportions to avoid narrow black gaps inside gallery cards.
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
- **H3 accelerators and refinement:** workflow-matched LightX2V accelerators,
  PTD RefAsymOPD presets and compatible two-pass latent refinement. Existing
  model-specific recipes remain separate; two-pass is off by default and
  applies only to compatible standard H3 Euler jobs.
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
- **Reference enhancement and dialogue recovery:** video references expose
  Character / identity, Motion / camera and Scene roles independently from
  their audio role. Native Subject IDs and vocal IDs stay separate. Recover
  exact speaker ownership when a listener or named prop appears near a line,
  preserve authored speech rather than imposing a shorter writing target,
  and avoid duplicate native sound fields in fallback prompts. A narrowly
  identifiable final `<d>words<d>` typo is repaired without changing its words;
  ambiguous or nested tags return a clear error. Focused repairs receive the
  validation reasons and missing lines. Existing Motion or legacy soundtrack
  references are not silently converted to character or voice references.
- **Reliable startup:** the API reserves its actual bind interface through
  server startup and announces the Maestro URL only after its listener is
  ready. Busy ports fall back without taking over a shared Pinokio service.

See the [October port scope](WanGP-October-2026-Ports.md),
[MOSS guide](MOSS-Image.md), [Fibo guide](Fibo-2.md),
[Pika Flex guide](guides/pika-h3-flex.md), and [Studio controls](Studio-controls.md)
for model-specific inputs and limitations.

## Qualification and limits

Release audit repairs include wrapped H3 BF16 text/vision encoder namespaces,
explicitly reviewed Director scene images with **None — no generated images**,
and joined videos honoring the selected codec, quality and MP4/MKV container.
Joins still re-encode to a consistent output format; they do not promise
lossless stream-copy for mixed inputs. Live previews recover from browser tab
suspension while preserving a user pause. Director keeps standalone uploaded
character-selection instructions out of its performed-action ledger, and
whole-song outlines can allocate up to 48 source actions rather than sixteen.
These repairs have focused CPU/browser coverage; the reported external-writer
and affected-GPU cases still need their own end-to-end retests.

Final name-resolution checks also repaired older pose-validation and face-mask
errors, initialized audio overlap before multi-window conditioning, and reset
temporal-upscaling history at each new repeat. The dtype-policy cleanup removes
an unreachable line and does not change model precision.

The MCP protocol/security suite and focused UI/backend regressions pass locally.
Pika's exact-audio path completed two 124-frame reference shots and a 248-frame
join at 832×480 / 24fps / SLA on the RTX 4090. Full CPU decoding, frame counts,
source-audio correspondence and fixed target-audio runtime evidence passed.
This confirms functionality for the tested case, not perfect lip sync. These
results belong to the earlier v4 candidate and are not a blanket qualification
of every later fix or model setting.

The bounded three-workstation qualification completed on 3 November with the
RTX 4090 (24GB), RTX 3080 Ti (12GB) and RTX 5070 Laptop (8GB). Reviewed native
renders, ComfyUI outputs, Editor exports and CPU/browser checks were retained.
Real ComfyUI Wan generation was checked on all three machines; mapped Director
routes and reference-edit semantics have narrower CPU/mock or manual coverage.
Local MCP was checked on 01 and 02. It remained off and unowned on 03, so that
machine's owner-approval gate was respected rather than bypassed for a test.

The 4 November reference-enhancement fixes have CPU regression coverage and
fresh Gemma/Qwen prompt checks on 01. Four checks preserved the six supplied
spoken lines and their native speaker bindings without a final fallback.
These were enhancement checks, not new video renders. A writer can still
change action detail or return an inconclusive continuity review. The latest
revision has not been fully re-rendered across the fleet; earlier results do
not establish that every new fix has been rendered again on 12GB and 8GB hardware.

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

Model weights retain their own licenses and use restrictions. In particular,
review the [Fibo](Fibo-2.md) and [Worlds](Worlds-Hy-Nomas.md) guides before using
those optional models; Maestro's application license does not replace theirs.

## Upgrade and optional setup

Use Pinokio's Update action while the installation is idle, restart Maestro,
and refresh the browser. Existing projects and media remain available;
model-specific defaults and compatibility checks still apply when selecting a
workflow. The deliberate appearance change is described below.

ComfyUI, Local MCP and Remote MCP are optional. Local MCP needs no Tailscale;
Remote MCP can reuse an existing Pinokio Remote HTTPS route or the separate
Tailscale integration. Follow the client-specific Agent Access instructions.
Enabling an agent connection does not require changing ordinary LAN browser
access or automatically configuring a ComfyUI server.

When reusing older video references for a character, choose **Character /
identity** and, for a cloned speaking voice, **Voice reference** as the video's
audio role. A **Motion / camera** video or a legacy paired soundtrack keeps
that role until changed explicitly. A room or environment image belongs under
**Scene**; **Composition** controls framing and blocking rather than identity.

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
