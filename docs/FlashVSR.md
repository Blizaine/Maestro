# FlashVSR and FlashVSR+ Turbo

Maestro exposes the original FlashVSR v1.1 and the separate FlashVSR+ Turbo
checkpoint as distinct spatial upscaling methods. Existing `flashvsr...` and
`flashvsr2pass...` values continue to select v1.1. The Plus methods use
`flashvsrplus...` and `flashvsrplus2pass...` values, so a saved v1.1 method
does not switch checkpoint families.

Both methods use the FlashVSR extension mode, precision, persistence, and
attention-backend settings. The v1.1 method also uses the configurable top-k
setting; Plus Turbo uses its pinned hybrid-attention recipe with a fixed 2.5
top-k ratio. A missing or invalid `flashvsr_model` setting normalizes to
`v1_1`; explicit method IDs select their own model. The default spatial
upscaling method remains the existing v1.1 choice.

Plus Turbo uses four latent frames per stream chunk, hybrid spatial attention,
three cache windows for draft layers, one cache window for local layers, and a
19-frame continuation overlap. V1.1 retains its six-frame startup, two-frame
continuation schedule, and 11-frame overlap. Still images use the selected
model's own startup and warmed-frame handling.

## Pinned Plus Turbo assets

On first use, Turbo assets are downloaded into model-specific local filenames
under `FlashVSR/`; v1.1 files remain separate. The transformer, low-quality
projection, decoder, and optional Wan VAE are pinned to
`ruchangyao/FlashVSR_plus` revision
`46b79520946ac67dec94d1566490690953c7debc`. Their LFS SHA-256 values are
recorded in `FlashVSRBridge.PLUS_ASSET_METADATA` in
`app/postprocessing/flashvsr/wgp_bridge.py`.

The Plus positive prompt tensor comes from the official
`OpenImagingLab/FlashVSR_Plus` repository at commit
`b548a2cc2c929d3d4676f2973abe27876ec4dee7`. Maestro verifies its Git blob SHA-1
and byte size before atomically saving it. The prompt is a raw
`[1, 512, 4096]` 16-bit floating point tensor and loads with PyTorch's
`weights_only=True`. The `.ckpt` TCDecoder and optional `.pth` Wan VAE use the
same weights-only state-dictionary loader. The `full` decoder setting fetches
the pinned Wan VAE; `tiny` and `tiny-long` use the pinned TCDecoder.

No model weights are included in the source tree. FlashVSR+ source is licensed
under Apache-2.0; the WanGP-derived streaming and application portions remain
subject to the WanGP Community License 2.0. See [third-party notices](../THIRD_PARTY_NOTICES.md).
