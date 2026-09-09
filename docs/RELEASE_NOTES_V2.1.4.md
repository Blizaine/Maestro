# Maestro v2.1.4

Released 9 September 2026.

This patch reduces avoidable system RAM use while loading H3/Viggle's video
decoder, improves decoding efficiency and restores the intended limits for
fractional memory profiles. The README now groups all v2.1.0–v2.1.4 highlights
into one overview; individual release notes and the changelog retain their history.

## Lower H3 and Viggle loading overhead

- Read the compact video VAE checkpoint in its native projection layout instead
  of splitting/reordering its attention weights and swapping feed-forward rows
  into additional CPU tensors.
- Preserve the original checkpoint storage and INT8 ConvRot row scales. Both
  the FP16 VAE used by Viggle and the INT8 VAE used by H3 Fused are supported.
- Reuse existing model files. No converted checkpoint, extra model download or
  dependency change is required.

A local component-loading comparison removed about **4.35 GiB of private
allocations**. This is a measured reduction in loading overhead, not a guarantee
of a particular total RAM requirement for every job.

## Faster video decoding

- Profile the video encoder and decoder separately through MMGP, following
  WanGP's H3 memory policy.
- On cards with at least **10 GiB VRAM**, keep decoder weights on the GPU across
  spatial tiles and temporal chunks, avoiding repeated transfers of the same
  weights. Smaller cards keep the selected profile's streaming behavior.
- Release decoder weights when another model stage starts. Encoding and
  decoding share the original modules rather than allocating another copy.
- Preserve Maestro's existing FP16 codec precision policy, including when the
  transformer uses BF16 or the decoder uses INT8 ConvRot weights.

On a local RTX 4090, decoding the same 51-frame synthetic clip with the FP16 VAE
took **8.891 seconds before versus 2.219 seconds after**, with identical output
for the residency-policy comparison. This uses more VRAM during decoding and
measures that stage only; it does not establish a fourfold speedup for a complete
Viggle generation. See the [validation record](VALIDATION_V2.1.4.md).

## Memory-profile corrections

- Profile **3.5** receives profile 3's 70% budget while retaining disabled pinning.
- Profile **4.5** receives profile 4's streaming limits while retaining disabled
  asynchronous transfers.
- Explicit preload settings and model-specific budgets remain respected.

Previously, the fractional number was converted to its base profile only after
the budget branches ran, causing both variants to miss their intended limits.
Whole-number profiles keep their existing policy. Automatic profile selection
and transformer workspace safeguards remain in place.

## Updating

Use **Update** in Pinokio, restart Maestro and refresh the browser. Existing
models, outputs, characters, workspaces and saved projects remain in place.
The application reads the new version and runtime code on its next restart.

The cumulative [README overview](../README.md#updates) includes the Studio
redesign, character sharing, Animate trim/frame selection, H3 tools, dialogue
planning and reliability fixes from this release series. Detailed history:
[v2.1.0](RELEASE_NOTES_V2.1.0.md), [v2.1.1](RELEASE_NOTES_V2.1.1.md),
[v2.1.2](RELEASE_NOTES_V2.1.2.md), [v2.1.3](RELEASE_NOTES_V2.1.3.md).
