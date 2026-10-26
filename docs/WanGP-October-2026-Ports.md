# WanGP October 2026 ports — first batch

This batch adapts selected features from [WanGP revision
9905f0d](https://github.com/deepbeepmeep/WanGP/tree/9905f0df941e5ec7b0c80e32555d3a61fe415726),
reviewed on 26 October 2026. It preserves existing defaults and saved workflows.
It has CPU regression coverage and a successful UI build. Bounded functional
GPU checks now cover 24GB, 12GB and 8GB NVIDIA workstations; they do not establish
a matched speedup or general quality claim. See the updated
[v3.3.0 validation notes](RELEASE_NOTES_V3.3.0.md).

## Available changes

- **Optional compute kernels:** the install/update helpers select WGPKernels
  1.4.2 and GGUF CUDA kernels 1.0.29 on their supported Windows/Linux stacks.
  The selected wheel URLs include SHA-256 digests. They update stale versions,
  retain a newer matching manual installation and keep existing fallbacks on
  unsupported systems. See [kernel pins and compatibility](optional-kernel-pins.md).
  The new WGPKernels FP8-weight API is a separate dispatch path and is not
  activated by this update.
- **Qwen Image 2.1 prefix caching:** Advanced offers the existing 16-bit cache,
  INT8 cache in VRAM, and INT8 cache in RAM. Disabled remains the default.
  Quantization uses per-token, per-head groups of 32 channels with FP32 scales;
  for 128-channel heads its persistent storage is 56.25% of a 16-bit cache.
  This ratio covers only K/V storage, not the model or whole-job peak memory.
  The portable attention path materializes one layer's prefix before attention;
  it does not include WanGP's native staged Sage cache kernels. RAM mode adds
  transfers and may be slower. Conservative GPU and host-RAM checks can refuse
  caching and recompute every reference without reducing output dimensions.
  These are preflight checks rather than memory reservations; an actual
  allocation failure can still fail the job instead of retrying it.
  CFG caches are independent, request-local, and disabled for scheduled LoRAs.
  INT8 introduces a small numerical change. The bounded robot-edit comparisons
  retained the requested content; cache fallback and activation limits are
  recorded in the release notes.
- **H3 LightX2V standard accelerators:** four explicit workflow-matched recipes
  add FL2VA 4-step v1.3 / 8-step v1.1 and Ref2VA 4-step v1.0 / 8-step v1.1.
  Pinned weights and hashes are in the manifest. Their video shifts reach the
  scheduler and reset between jobs. Fused, FastH3, VDN and PDD paths keep their
  dedicated recipes. See [H3 accelerators](H3-LightX2V-Accelerators.md).
- **FlashVSR+ Turbo:** a separate model choice in generation post-processing,
  Tools and Editor exports, with versioned method IDs. Existing FlashVSR IDs
  remain v1.1. Plus uses its own hybrid attention, streaming schedule and
  continuation cache; it does not substitute a new checkpoint into the old
  configuration. First use downloads its separate assets. See
  [FlashVSR workflows](FlashVSR.md).

Initial source implementation was followed by user-authorized GPU validation.
Saved preferences and unrelated local changes were preserved. The FlashVSR
follow-up fixes restore the legacy automatic attention ratio and normalize the
official Plus transformer and projection names; all 12 latest image/video
checks passed after 27 CPU regressions. The earlier failed attempts remain
part of the private evidence.

## MOSS editing follow-up

The next implementation adds separate MOSS Edit SFT and HQ armV5 presets,
ordered image references, the full VAE encoder, edit-specific enhancement and
request-local reference K/V caching. Base/SFT text-to-image and their layout
helper stay separate. The adapters and companion VAE have been checked against
the actual released file headers. See [MOSS editing and cache controls](MOSS-Image.md#reference-editing).

This follow-up has passed 248 targeted CPU regressions and the UI build, and has
not been published with the FlashVSR repairs. GPU editing quality, cache activation, memory use and speed on
the three workstation configurations remain to be measured. An available cache
option must not be reported as a demonstrated speedup.

## Next batches

The saved review still covers Fibo 2.0, H3 PTD,
two-pass refinement, Profile 0, Worlds and a media assistant. Those are separate
runtime/product integrations. MOSS caching depends on its editing integration;
the Qwen cache controls do not add references to text-only MOSS models.

Before enabling new defaults or claiming a speedup, use matched GPU tests with
the same inputs, settings, cache state and verified output. In particular,
RAM-backed caches do not establish an 8GB or 12GB speedup by themselves.
