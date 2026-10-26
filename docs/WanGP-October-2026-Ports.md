# WanGP October 2026 ports — first batch

This batch adapts selected features from [WanGP revision
9905f0d](https://github.com/deepbeepmeep/WanGP/tree/9905f0df941e5ec7b0c80e32555d3a61fe415726),
reviewed on 26 October 2026. It preserves existing defaults and saved workflows.
It has CPU regression coverage and a successful UI build; consumer-GPU speed,
peak memory and visual quality have not been benchmarked for these new paths.

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
  INT8 introduces a small numerical change; local visual comparison is pending.
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

No running Maestro instance or saved user settings were changed during this
implementation. No weights, optional wheels or generation jobs were started.

## Next batches

The saved review still covers MOSS reference editing, Fibo 2.0, H3 PTD,
two-pass refinement, Profile 0, Worlds and a media assistant. Those are separate
runtime/product integrations. MOSS caching depends on its editing integration;
the Qwen cache controls do not add references to text-only MOSS models.

Before enabling new defaults or claiming a speedup, use matched GPU tests with
the same inputs, settings, cache state and verified output. In particular,
RAM-backed caches do not establish an 8GB or 12GB speedup by themselves.
