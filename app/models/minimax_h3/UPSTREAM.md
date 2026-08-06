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
treated as actual model evaluations. Pruned-base time-embedding reinjection
remains unsupported, so Maestro rejects that combination before loading the
adapter.

Those model weights are downloaded at runtime and are not distributed in the
Maestro repository. They remain governed by their respective model terms and
any authorization or waiver required for the user's location.
