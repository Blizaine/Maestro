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

The optional H3 Sol Engine integration is adapted from WanGP commit
`7e45fe7e21105807b43f6285827d9ebb5fa72906`. The portable kernels under
`shared/sol_attn/triton_kernels` track NVIDIA Sana Sol-Attn commit
`46031940ba8af5d18054217e571149579424c0b1`; the optimized INT8-QK kernels
under `shared/sol_attn/saganaki` track ComfyUI-sol-attn v0.5.2 commit
`e2fc225`. Both sources are Apache-2.0 and retain their source headers and
bundled license. Maestro restricts this backend to compatible MiniMax H3
main-DiT attention, protects the complete packed conditioning prefix, and
falls back to the selected dense Sage/SDPA backend on unsupported calls or
kernel failure. RTX 40 users receive it through an isolated optional runtime;
the proven RTX 20/30/40 default environment is not upgraded in place.

The full FL2VA sliding-window continuation contract is adapted from WanGP
v12.44's H3 feature commit
`5c8b4ac3c5e15135b6510d9b6d4d57002e4bb5e4`, with follow-up fixes through
`639ee1351e5b57c5992903690199719607c3700e` (WanGP 12.441). A legal
`17*n+1` overlap is split into complete 17-frame visual-history chunks and one
regenerated boundary frame; the matching generated stereo-audio tail is VAE
encoded on the same rotary timeline. Maestro retains its exact-duration outer
assembler, prompt planner, and VRAM-aware pass limits around that native H3
conditioning. The same upstream follow-up also supplies the streaming video
VAE tile decoder used here, which decodes one tile at a time and preserves
already blended horizontal/vertical tails to reduce peak VRAM and avoid
multi-tile corner seams.

Phase 3 extends that v12.44 contract to Ref2VA. Each continuation pass keeps
the user's canonical ordered image/video/audio references, presents the prior
window boundary as an additional leading picture to Qwen, and packs clean
multi-frame video plus matching stereo-audio history ahead of the reference
rows. Picture labels in the generated Context-IR are shifted only for that
temporary boundary; canonical Video and Audio numbering remains stable.
Maestro's Multi-window sequence toggle uses this native path when continuity is
enabled and retains independent hard-cut clips when it is disabled. Window
prompts may come from the story-planning LLM or from an exact user-authored
one-non-empty-line-per-window mapping; the latter never loads the planner.

Phase 4 adapts WanGP v12.44's FL2VA multiple-frame injection contract. Extra
Studio Frame tiles are paired with exact target-frame anchors after carried
visual history is removed, while the H3 window planner gives those same images
window-local Picture numbers and timestamps. Start, final-end, continuation,
and injected pictures therefore share one deterministic presentation order
between Qwen conditioning, packed model conditions, saved plans, and resumed
jobs. Ref2VA deliberately does not advertise this FL2VA-only capability.

Phase 5 adapts WanGP v12.44's FL2VA media-source contract. An uploaded
soundtrack or the soundtrack extracted from a Control Video is VAE encoded
into clean, frozen target-audio rows so it can drive newly generated video.
The reciprocal video-to-audio mode VAE encodes the complete Control Video as
frozen target-video rows, skips visual denoising and decoding, and generates a
new synchronized soundtrack while returning the source pictures unchanged.
Maestro keeps its ordinary multi-window assembler around both paths,
de-duplicates the carried visual overlap from each combined guide window, and
uses H3's `17*n+5` frame offset when preparing Control Video windows.

Phase 6 adapts WanGP v12.44's FL2VA video-to-video editing contract. A Control
Video is VAE encoded as a clean reconstruction target and re-injected at the
matching scheduler noise level. Whole-frame mode uses denoising strength to
choose how late generation begins; masked modes use a white-edit/black-preserve
mask and masking strength to control how long protected pixels remain locked.
The same source can retain its soundtrack, generate a new synchronized stereo
track, or be driven by a separate soundtrack. Ref2VA remains reference-guided
and deliberately does not advertise this FL2VA-only editing pipeline.

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
