# Maestro v2.4.1

September 25, 2026. Changes since v2.4.0.

## Qwen Image 2.1: editing, controls and acceleration

Qwen Image 2.1 now supports control-image transfer for pose, depth, edges,
shapes, grayscale and raw images, alongside its existing reference editing.
Inpaint accepts a source and mask, with Masked Denoising or LanPaint at
2/5/10/15 inner steps. Outpaint extends the canvas. Enhancement receives the
selected editing/control context and uses updated model-specific guides.

Native **2K** presets use approximately 4.2 megapixels: 2048x2048 square,
2752x1536 widescreen, 1536x2752 portrait, 2400x1792 / 1792x2400, and
3136x1344 ultrawide. Auto aspect fits the source within the same pixel budget.
These are native image-generation sizes; memory needs increase accordingly.

**Advanced → Image acceleration** offers managed Viggle Turbo v0.1 at four
steps, v0.2 at five and v0.2.1 at six. Maestro downloads the matching adapter
on first use and applies its guidance and sampling recipe. Switching back to
Base removes the managed adapter while preserving user style LoRAs.
Optional reference KV caching uses a memory check before retaining the cache.

New base defaults are **40 steps / CFG 4**; existing saved settings remain.
The recommended-settings button reapplies the recipe. An unset inpainting
mode no longer causes ordinary text-to-image generation to fail. VAE tiling
adapts to GPU capacity with wider overlap, addressing a plausible source of
lines/seams reported in **#153**. That report still needs visual confirmation;
the change is not a claim that every image artifact is resolved.

See the [Qwen Image 2.1 guide](Qwen-Image-2.1.md). Turbo quality and memory use
vary with the workflow, particularly masks, transparency, many references and
2K output. Existing Qwen model license terms still apply.

## H3 and LTX kernels

Automatic H3 VAE selection uses the **INT8 ConvRot video VAE** when the
transformer is INT8. Download and loader selection agree on the asset;
explicit VAE selections remain authoritative. Other automatic transformer
formats retain FP16, and the audio VAE is unchanged.

The shared requirements include pinned **Comfy Kitchen** support. H3 and
LTX2.x can use its compatible kernels on the supported RTX 50 architecture,
with established implementations retained for other GPUs and unavailable
acceleration. Kernel precision policy remains configurable. Upstream speed
figures are not Maestro benchmark guarantees.

## Experimental DLSS on Windows 10

An optional compatibility backend adds **1x neural enhancement** and
**1.5x, 1.724x, 2x and 3x DLSS upscaling plus neural enhancement** for images
and videos. It works through Tools → Upscale, Studio finishing and Media Flow.
This requires a separate, explicit installation and compatible NVIDIA hardware
and driver; normal Maestro updates do not enable it silently.

The backend uses separate native processes for Super Resolution and neural
enhancement. Processing avoids redundant frame copies and CPU scans, retains
the source frame rate, duration and soundtrack, and cleans up workers on
cancellation. Internal edge padding handles native row alignment and is
cropped before enhancement, retaining the requested public dimensions.

The existing Windows 11 integration stays available. **DLSS Frame Generation
still requires Windows 11** and supported hardware; RIFE remains an alternative
for interpolation. Offline finishing includes decoding, guide estimation and
encoding, so it is not equivalent to a game's real-time DLSS pipeline.

Read the [installation, compatibility and measured validation guide](DLSS5.md).
The Windows 10 backend is community compatibility work and remains experimental.

## Gallery and media management

- **Stable chronology:** results remain ordered during generation, refresh
  and pagination. Overlapping refreshes and selection changes retain the
  correct folder-qualified item. This addresses the gallery ordering report
  in **#155**.
- **Useful Media Info:** local timestamps and available measured dimensions,
  duration, frame rate/count and file size appear for generated and uploaded
  media, even without a generation prompt. Labels distinguish recorded
  generation/upload times from filesystem fallback dates.
- **Finishing history:** new processed media records the method, scale,
  before/after resolution and frame rate, source, elapsed time and relevant
  DLSS settings. Older results show only details that can be recovered from
  existing metadata and source files; missing history is not invented.
- **Delete uploads:** the gallery menu offers confirmation before deleting
  uploaded media. Inputs used by active jobs are protected. Failed deletion
  shows a useful error and restores playback; completed outputs are retained.

See [Studio controls](Studio-controls.md) for gallery behavior.

## Update

Use **Update** in Pinokio, restart Maestro, then refresh the browser. Existing
projects, settings, models, LoRAs and media stay in place. Optional model assets
download on first use. Install the experimental Windows 10 DLSS component
separately only if you want to use it.

[Validation and limitations](VALIDATION_V2.4.1.md) · [Changelog](../CHANGELOG.md)
