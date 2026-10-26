# Maestro v3.3.0 — Dev testing

26 October 2026. Available on public Dev for testing the first October WanGP
porting batch. Public Main remains v3.2.5. Existing sampling defaults and
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

Actual new GPU renders, speed and peak memory have not been measured. The
12GB and 8GB workstations are intended testing targets, not verified speedup
claims. The real Plus checkpoints have not yet been downloaded/deserialized
in this implementation; first use hashes the selected large files and adds
disk-read startup time.

INT8 prefix storage introduces a small numerical change. RAM caching adds
transfers and can be slower; preflight memory checks do not reserve memory or
retry allocation failures. This portable path materializes one prefix layer
for the existing attention backend. It does not include native staged Sage
cache kernels or the separate new WGPKernels FP8/W8A16 dispatch.

MOSS editing, Fibo, PTD, flexible two-pass H3, Profile 0 and Worlds are later
batches. See the [port scope](WanGP-October-2026-Ports.md).

## Update and test

When idle, stop Maestro, select **Update** for the **Dev** installation in
Pinokio, restart it and refresh the browser. The UI should show **v3.3.0**.

- For Qwen reference editing, compare Disabled, Enabled, INT8 and INT8 RAM
  using the same inputs/settings. Record complete outputs, elapsed time and
  memory; a smaller persistent cache does not imply a faster whole job.
- For H3, choose a matching standard model/workflow and the named LightX2V
  recipe. These presets are not for the already fused/distilled variants.
- For upscaling, select an explicit **FlashVSR+ Turbo** method. Legacy
  FlashVSR options continue to use v1.1.
