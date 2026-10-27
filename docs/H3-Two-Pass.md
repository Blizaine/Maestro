# MiniMax H3 Two-Pass Latent Upscaling

H3 Two-Pass is an opt-in path for compatible standard H3 recipes. It generates
video at aligned half resolution, upsamples the normalized 24-channel H3
latent with the learned 3D upscaler, then runs a high-resolution Euler refine
pass. The second pass keeps the first pass's generated stereo audio fixed.

The control is Off by default. Old jobs, presets, and metadata without the
control stay on the ordinary one-pass path and do not request the upscaler
asset. The four custom-setting IDs are stable for metadata and preset
round-tripping:

| Setting | Default | Range | Meaning |
| --- | ---: | ---: | --- |
| `h3_two_pass_enabled` | `off` | `off` / `on` | Enables the optional asset and both passes. |
| `h3_phase2_steps` | 3 | 2–12 | Number of high-resolution Euler evaluations. |
| `h3_phase2_noise` | 0.9035 | 0.05–1.0 | Initial phase 2 noise; sigma then decreases linearly to zero. |
| `h3_phase2_accelerator_strength` | 1.0 | 0–2 | Phase 2 multiplier for a selected Turbo accelerator. |

Phase 2 settings are retained while Off and become active only when the enable
setting is On. The normal `guide_phases` and `switch_threshold` controls keep
their own meanings.

## Compatibility

Two-Pass requires the Euler sampler and a compatible standard H3 model. It is
not available with PTD/PDD heads, fused Turbo, FastH3, VDN, DaSiWa, baked
Turbo, Singularity, audio-only jobs, source-video continuation, Control Video,
edit masks/outpainting, partial denoising, Audio Refinement, or face
refinement. Unsupported combinations return a validation error instead of
silently changing the recipe.

The phase 1 adapter schedule uses the user's ordinary settings. During phase
2, ordinary adapters retain their configured phase 2 multiplier; selected
Turbo accelerators use `h3_phase2_accelerator_strength`. The original adapter
schedules are restored after phase 2, cancellation, or errors.

Phase 2 leaves phase 1 audio latents fixed. Maestro's H3 scheduler defines
`t = 1 - sigma` with clean audio at `t=1`; WanGP's phase 2 sends zero audio
sigma while refining video. The H3 transformer therefore receives audio time
`1` on every phase 2 step, while the audio scheduler update and audio rescaling
remain disabled.

## Resolution and memory

The low-resolution canvas rounds each target dimension to a 32-pixel-aligned
half-size grid. For example, a 512×864 target starts at 256×448, then phase 2
returns to 512×864. The target remains the output canvas, not a VRAM cap. The
high-resolution transformer pass still has target-size activation costs, so
Two-Pass can save time without guaranteeing lower peak VRAM. MMGP reserves
10 GiB of transformer working memory plus a 2 GiB budget for the upscaler.

The learned upscaler source is pinned to MIT commit
`98934c0b5f4a213323c4d1d0e47eb3d2f27cb743`; the BF16 weights are pinned to
`LBH-123-AI/Minimax_h3_latent_Upscaler` revision
`09592c6221ec95cc8e0fae67842e34926c4e668b`, SHA-256
`4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6`. The
code and model have separate licenses; see
[`H3-LATENT-UPSCALER-NOTICE.md`](../app/models/minimax_h3/H3-LATENT-UPSCALER-NOTICE.md)
and the adjacent MIT license file.

The source implementation uses 32-latent temporal chunks with a 5-latent
overlap for longer clips. GroupNorm spans time, so chunking follows the
upstream memory-bounded approximation rather than mathematically matching
full-context inference. The default 124-frame H3 window fits below that chunk
threshold and uses full-context upscaling.
