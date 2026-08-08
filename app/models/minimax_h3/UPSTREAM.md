# MiniMax H3 upstream components

The video VAE, audio VAE, scheduler, FL2VA packing, and Ref2VA reference
preparation/packing in this directory are derived from the Hugging Face
Diffusers MiniMax H3 implementation at commit
`abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`.

Those files retain their upstream Apache-2.0 copyright and license headers.
Maestro-specific model loading, packing, memory management, and Studio
integration are implemented separately in this directory.

The default runtime stack is pinned to:

- `MiniMaxAI/MiniMax-H3` commit `5d9b308a59ab12e67147f191e184baf704185bd1`
  for the official processor and text-encoder configuration.
- `Comfy-Org/MiniMax-H3` commit `0543966fbdce5ba05709a8f2031c94bdba629b4a`
  for the scaled-FP8 FL2VA and Ref2VA transformers, NVFP4-AWQ conditioner,
  and compact video/audio VAE checkpoints.
- `DeepBeepMeep/MiniMax-H3` commit
  `fec7846aef352e58a1cfb699455e3d104281e68b` for the full 33B FL2VA/Ref2VA
  checkpoints and the optional BF16, Quanto INT8, and GGUF Qwen3-VL
  conditioners.

The dual full/pruned checkpoint probe, full-checkpoint split Q/K/V loading,
fused pruned-checkpoint attention, ConvRot QKV restoration, independent Qwen
language/vision profiling, and selectable text-encoder catalog adapt the
MiniMax H3 implementation shipped in WanGP v12.41
(`4ed4c744a396e43294f851f35cab769e11a89f2d`). Maestro retains its existing
compact Comfy checkpoint loader and local Studio APIs around those
memory-management pieces.

The INT8 ConvRot consumer is adapted from WanGP's
`shared/qtypes/int8_convrot.py` at `b382d0940cdbab29cff5d33301b34b337ad5517e`
(handler revision `6b92c54f92bde24d6d309d6f61249353b0ec783d`). The ConvRot export stores
its fused QKV rows and row scales in logical grouped `[Q, K, V]` order, which
Maestro splits contiguously into independent streamable projections. Active
LoRAs retain ConvRot's native activation rotation for the quantized base branch
and apply their deltas from the original, unrotated module input.

The Ref2VA runtime preserves the official ordered reference presentation,
shared rotary clock, soundtrack-before-video row ordering, sampled/noised
visual VAE conditioning, clean audio conditioning, and target-only denoising.
Maestro defaults reference images to an output-matched, downscale-only detail
policy for consumer GPUs; the official 2048-pixel-short-edge preparation is
also available as the Maximum reference detail option. Reference videos now
follow the same output-matched pixel-area policy by default; this prevents a
480p/544p generation from silently encoding its reference at a 768-pixel short
edge and more than doubling the packed attention working set.

Optional low-step support for LarryVRH's MiniMax H3 Turbo LoRA follows the
adapter and custom-sampler contract published at
`larryvrh/MiniMax-H3-Turbo-Lora` and
`Larryvrh/ComfyUI-MiniMax-H3-Turbo` (inspected 2026-08-06). Maestro's native
runtime already advances video and audio on the required independent shift-12
and shift-3 schedules. The adapter's logical grouped fused-QKV updates are split
contiguously with the instantiated H3 module, and the requested 4/6/8 steps are
treated as actual model evaluations. Full/Pruned AdaLN LoRA conversion is
adapted from WanGP commit `1830091bf4b27df2f901920d55b1fb748f33e7eb`.
Its small FL2VA/Ref2VA rank-8 and rank-64 affine packages are downloaded from
that immutable revision, size/hash verified, and stored in the user's checkpoint
area on first H3 LoRA use. They are not redistributed with Maestro.

The H3 shared-attention path, early release of projection inputs, bounded
projection chunks, and optional First Block Cache behavior also follow the
memory-conscious H3 runtime in WanGP commit
`1830091bf4b27df2f901920d55b1fb748f33e7eb`. Maestro adapts those ideas to its
batched Diffusers-style row layout, uses the exact generated-row boundary for
FL2VA and Ref2VA cache residuals, and keeps First Block Cache experimental and
disabled by default.

The one-click experimental Turbo preset pins the Maestro-validated
`minimax_h3_turbo_4step_ckpt500.safetensors` file at repository revision
`7a44622816e16032cb0b6d044d8820da39a1dfdc` (SHA-256
`82d0acff583b04ad9a4238a7440b584b56094bfb7c4fdb2981f67c7a4784b62d`).
It uses six model evaluations and starts at adapter strength 0.50. The managed
adapter is also activated in Advanced so users can tune its strength for a
specific prompt while Turbo mode continues to own the six-step schedule. The
file is listed for Full and Pruned H3 checkpoints before installation and is
downloaded, verified, and atomically published on first use; it is not
distributed in the Maestro repo.

Those model weights are downloaded at runtime and are not distributed in the
Maestro repository. They remain governed by their respective model terms and
any authorization or waiver required for the user's location.
