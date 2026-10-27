# Maestro v3.3.0 — Dev testing

26–27 October 2026. Public Dev testing release of the October WanGP ports
and their integration fixes. Public Main remains v3.2.5. Existing sampling defaults and
saved v1.1 FlashVSR selections retain their behavior.

## What changed

- **Qwen Image 2.1:** Advanced adds INT8 reference K/V storage in VRAM or RAM,
  alongside the existing full-precision cache. Disabled remains the default.
  GPU/host-RAM checks may recompute the complete prefix instead; all references
  and requested output dimensions are retained. Guidance branches are separate,
  and step-varying LoRAs disable caching. See [Qwen Image 2.1](Qwen-Image-2.1.md).
- **H3 LightX2V:** add FL2VA four-step v1.3 / eight-step v1.1 and Ref2VA
  four-step v1.0 / eight-step v1.1 standard-LoRA recipes. Their sampling shifts
  reach the scheduler and reset between jobs. Fused, FastH3, VDN and PDD keep
  their dedicated recipes. See [H3 accelerators](H3-LightX2V-Accelerators.md).
- **FlashVSR+ Turbo:** add explicit Plus selections in generation Post
  Processing, Tools and Editor export. Its hybrid attention, streaming and
  continuation cache are separate from v1.1. Selected assets use pinned
  identities and integrity checks; weights download on first use. See
  [FlashVSR](FlashVSR.md).
- **Optional kernels:** the normal Pinokio Update path selects hash-pinned
  WGPKernels 1.4.2 and GGUF CUDA 1.0.29 on supported Windows/Linux runtime
  stacks. Stale builds update, newer compatible manual installations remain,
  and unsupported systems retain their existing fallback. See
  [kernel compatibility](optional-kernel-pins.md).

## Validation and limits

Focused CPU regressions, UI type-check/build, touched-UI lint, Python syntax
and undefined-name checks, documentation links and the clean-repository guard
passed. CPU coverage includes cache accuracy and memory accounting, scheduled
LoRA invalidation, H3 sigma schedules, versioned upscaler selections, tensor-only
checkpoint loading, synthetic asset checks, two-chunk continuation and cancellation.

Functional trials now cover the RTX 4090 (24GB), RTX 3080 Ti (12GB) and RTX
5070 Laptop (8GB). All four H3 LightX2V recipes completed on each machine;
the text-only clips looked 3D despite a flat-2D prompt, and some reference clips
added background detail. All 15 Qwen edits retained the test subject and the
requested background change. Several cache modes fell back on the smaller
GPUs; activation is unconfirmed where the logs lack a positive cache message.
These runs are not a controlled speedup or general quality benchmark.

The GPU tests exposed two FlashVSR integration problems, now fixed: v1.1's
automatic sparse-attention ratio could become zero, and the official Plus
transformer/projection checkpoints used names the loader did not normalize.
The final follow-up passed 27 CPU tests and all 12 latest image/video checks
across the three GPUs. It covers 2x upscaling from 256x256 to 512x512 and
33-frame, 24-fps video with the TCDecoder path. Outputs fully software-decode
and retain the test content; tiny text and checker patterns can be softened
or changed. Longer clips, larger canvases and the full-VAE path need their own
validation. Initial failed attempts were retained separately. First use still
hashes the selected large Plus files and adds disk-read startup time.

INT8 prefix storage introduces a small numerical change. RAM caching adds
transfers and can be slower; preflight memory checks do not reserve memory or
retry allocation failures. This portable path materializes one prefix layer
for the existing attention backend. It does not include native staged Sage
cache kernels or the separate new WGPKernels FP8/W8A16 dispatch.

## 27 October follow-up

- **MOSS reference editing:** separate SFT and HQ v5 editors, ordered image
  references, edit-specific prompt enhancement and request-local reference
  caching. HQ includes its required Qwen3.5 vision conditioning. Switching
  from other image models no longer leaks their reference selector or generic
  model-mode value into MOSS validation. References preserve their native
  proportions independently of the output canvas. See [MOSS Image](MOSS-Image.md).
- **Fibo 2.0:** Dev text-to-image and four-step reference/masked editing.
  The tiled VAE flush now handles rectangular canvases correctly. See
  [Fibo 2.0](Fibo-2.md).
- **H3:** dedicated Full/Pruned eight-hop PTD RefAsymOPD recipes and opt-in
  two-pass latent upscaling for compatible standard recipes. Fixes cover PTD
  output heads, wrapped/fused adapter layouts and second-pass admission. See
  [PTD](H3-PTD-RefAsymOPD.md) and [two-pass generation](H3-Two-Pass.md).
- **Worlds:** an experimental Hy-Nomas workspace with saved worlds, queued
  camera moves, continuation, capture and recording of saved clips. This is
  queued generation; it does not provide real-time navigation. PixelUMM's
  non-commercial license requires acknowledgement. See [Worlds](Worlds-Hy-Nomas.md).
- **Optional memory runtime:** experimental MMGP VMM/Profile 0 selection with
  restart and compatibility gates. Legacy remains the default; Fish and CUDA
  graph capture remain unavailable on the experimental runtime. See
  [Profile 0](Profile-0.md).

Bounded functional pilots exercised these ports on the 24GB, 12GB and 8GB
test systems, including verified native two-pass execution and decoded media.
They are not matched speed or general quality benchmarks. The later Fibo
rectangular-canvas correction and corrected MOSS HQ vision/mode-switch path
were additionally exercised on workstation 01. The corrected HQ path has not
been requalified on the smaller GPUs; the earlier latent-only HQ pilot is
documented separately in the MOSS guide. Precise compositing, hand replacement
and text preservation can still fail visually even when generation completes.
The experimental allocator's actual GPU OOM recovery is not certified.

See the [port scope](WanGP-October-2026-Ports.md) for remaining work.

## Update and test

The image-prompt follow-up adds [Krea 2 caption and edit guides](Krea-2-Prompting.md)
for Raw, Turbo and their Identity Edit variants. Studio's guided and dedicated
raw image writers now carry an explicit medium/style contract: anime,
illustration, painting, photography and 3D are followed when requested. Source
images used only for identity do not prescribe a target style. Director's image
guides and dialect rules retain opening-frame continuity without forcing a
photographic medium, static neutral poses or the source image's lighting.
The older Flux/Qwen/generic guides no longer impose those constraints.

FLUX 3 prompt enhancement now identifies the active variant: Klein is distilled,
while Base is undistilled and uses classifier-free guidance, consistent with
the [official Base model card](https://huggingface.co/black-forest-labs/flux-3-image-klein-base).
Both still return a caption rather than changing the selected sampler settings.
The runtime also stops treating a stale generic model selector as the image
workflow, fixing valid Image Generate requests rejected after model switching.

Routine successful HTTP access lines are hidden in the terminal. HTTP failures,
warnings, exceptions and startup messages remain visible. These backend changes
take effect after a normal restart. The style checks cover routing, prompt
assembly and mocked-writer contracts; they do not guarantee that every LLM draft
or rendered image will follow the requested style.

When idle, stop Maestro, select **Update** for the **Dev** installation in
Pinokio, restart it and refresh the browser. The UI should show **v3.3.0**.

- For Qwen reference editing, compare Disabled, Enabled, INT8 and INT8 RAM
  using the same inputs/settings. Record complete outputs, elapsed time and
  memory; a smaller persistent cache does not imply a faster whole job.
- For H3, choose a matching standard model/workflow and the named LightX2V
  recipe. These presets are not for the already fused/distilled variants.
- For upscaling, select an explicit **FlashVSR+ Turbo** method. Legacy
  FlashVSR options continue to use v1.1.
