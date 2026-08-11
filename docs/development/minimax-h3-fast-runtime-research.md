# MiniMax H3 Fast Runtime Research and Integration Plan

Status: research complete; implementation deferred

Research date: 2026-08-07

Maestro checkpoint before this note: `6707da2` (`feat: add MiniMax H3 LoRA browser routing`)
Checkpoint verification: `58` MiniMax H3 regression tests passed

## Purpose

Preserve the research comparing Maestro's current MiniMax H3 Turbo implementation with three attention/cache-accelerated ComfyUI workflows, and record a safe implementation plan for later.

The workflows were supplied locally as:

- `video_minimax_h3_t2v_fast.json`
- `video_minimax_h3_i2v_fast.json`
- `video_minimax_h3_r2v1_fast.json`

The associated social post claims that Sol-Attn, SageAttention, and EasyCache reduce MiniMax H3 render time from about ten minutes to three minutes, or as much as 3.2x, without visual degradation.

## Executive conclusion

The workflows use a fundamentally different acceleration strategy from Maestro Turbo:

- Maestro Turbo uses the Full 33B H3 model, a low-step distillation LoRA at `0.70`, and six model evaluations.
- The ComfyUI workflows use the Pruned 20B H3 models at 20 steps, then accelerate attention and reuse intermediate model output.

The approaches may be complementary, but they should remain separately named and controlled. Add a future **H3 Fast Runtime (Experimental)** option rather than redefining Maestro's existing **Turbo LoRA** mode.

The claimed 3.2x speedup is plausible for a particular GPU, resolution, duration, and warmed kernel cache. It is not a universal expectation, and "zero degradation" is not technically defensible because both sparse attention and model-output caching are approximations.

## Supplied workflow inventory

All three workflows contain 24 nodes and share the same acceleration chain:

```text
UNETLoader
  -> PathchSageAttentionKJ
  -> SolAttnPatch
  -> EasyCache
  -> BasicScheduler / BasicGuider
  -> SamplerCustomAdvanced
```

### Common generation settings

- Sampler: `res_multistep`
- Scheduler: `simple`
- Steps: `20`
- Denoise: `1.0`
- Output frame rate: `24 fps`
- Video CRF: `19`
- No MiniMax H3 Turbo/distillation LoRA is present.

### Common SageAttention settings

- Backend: `sageattn_qk_int8_pv_fp8_cuda++`
- Compilation toggle: off
- This is SageAttention2++ behavior using INT8 QK, FP8 PV, and mixed FP32/FP16 accumulation.

### Common Sol-Attn settings

- Tau: `1.2`
- Active range: `0.20` to `0.90`
- Minimum tokens: `4096`
- INT8 QK: off
- Conditioning sink: `exact_kv`
- Morton reordering: on
- Morton curve: `3d`
- Verbose logging: off

### Common EasyCache settings

- Reuse threshold: `0.30`
- Active range: `0.20` to `0.90`
- Verbose logging: off

### Text-to-video workflow

- Model: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- Conditioning: text only through FL2VA
- Nominal canvas: `1344x768`

### Image-to-video workflow

- Model: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- Conditioning: one start image through FL2VA
- Input image is scaled to one megapixel using nearest-exact sizing.

### Reference-to-video workflow

- Model: `minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- Conditioning: ordered Ref2VA visual references
- Reference detail: maximum
- Example length: `124` frames

## How the three accelerators actually compose

The nodes do compose, but SageAttention and Sol-Attn do not both process the same eligible attention call sequentially.

1. Sol-Attn is installed after SageAttention and captures Sage as the prior attention override.
2. Sol-Attn takes eligible calls first.
3. Calls that are masked, out of range, too short, explicitly dense, unsupported, or fail the sparse kernel are delegated to SageAttention.
4. EasyCache operates at a higher level. When its change estimate remains below the threshold, it skips the full transformer call and reconstructs output from cached deltas.

Therefore:

- Sol-Attn is the sparse first-choice attention kernel.
- SageAttention is the dense/fallback kernel.
- EasyCache can bypass both by avoiding an entire model evaluation.
- Their individual speedup factors cannot be multiplied together because their savings overlap.

## Comparison with Maestro's current H3 runtime

| Area | Maestro Turbo | Supplied ComfyUI fast workflows |
|---|---|---|
| Transformer | Full 33B | Pruned 20B |
| Sampling | 6 model evaluations | 20 model evaluations |
| Main acceleration | Distillation LoRA | Attention kernels, sparse attention, and output reuse |
| Turbo LoRA | Yes, weight `0.70` | None |
| SageAttention | Installed globally, but H3 bypasses it | Used as dense/fallback attention |
| Sparse attention | None | Sol-Attn |
| Model-call cache | None for H3 | EasyCache |
| Approximation risk | Distillation | Quantized attention, sparse attention, and cached output |

Relevant Maestro code:

- `app/models/minimax_h3/turbo.py`: Turbo LoRA selection, six-step preset, and `0.70` default weight.
- `app/models/minimax_h3/transformer.py`: H3 attention currently calls `torch.nn.functional.scaled_dot_product_attention` directly.
- `app/shared/attention.py`: Maestro's existing SageAttention backends, currently bypassed by H3.
- `app/models/minimax_h3/minimax_h3_main.py`: each denoising evaluation invokes the transformer once and jointly predicts video and audio velocities.
- `app/models/minimax_h3/minimax_h3_handler.py`: FL2VA/Ref2VA model selection, Full/Pruned behavior, and sliding-window limits.

T2V, I2V, and Ref2VA share the same H3 transformer core in Maestro. A correctly designed acceleration layer can therefore serve all three modes, with stricter conditioning rules for Ref2VA.

## Current local compatibility snapshot

### August 11, 2026 upstream status update

- Maestro does **not** currently include or expose Sol-Attn. Its selectable
  attention backends remain SDPA, FlashAttention, and SageAttention; First
  Block Cache is a separate optimization.
- Current WanGP `main` now vendors an Apache-2.0 Sol-Attn backend under
  `shared/sol_attn` and wires it into MiniMax H3. WanGP's optimized INT8-QK
  path requires compute capability SM89 or newer, BF16 head-dimension-128
  attention, PyTorch 2.10+, and Triton 3.6+.
- The current Pinokio WanGP launcher migrated RTX 20/30/40/50 systems to
  Python 3.11, PyTorch 2.10, and CUDA 13. Maestro still deliberately keeps
  RTX 20/30/40 on its tested Python 3.10 / PyTorch 2.7.1 / CUDA 12.8 runtime.
- Therefore Sol support should remain a separate, checkpointed runtime and
  model-integration project. It is unrelated to a broken optional
  FlashAttention DLL on an RTX 3090, and WanGP's present optimized Sol path
  would not enable on that SM86 GPU.

Test machine:

- GPU: NVIDIA RTX 4090, compute capability 8.9, 24 GB VRAM
- PyTorch: `2.7.1+cu128`
- CUDA runtime: `12.8`
- Triton Windows: `3.3.1.post19`
- SageAttention: `2.2.0+cu128torch2.7.1`

Implications:

- SageAttention can be integrated into H3 without introducing a new package on this machine.
- The community ComfyUI Sol-Attn extension reports successful H3 testing on RTX 4090 and RTX 5090, but describes itself as experimental and compiles Triton kernels on first use.
- NVIDIA's current official Sol-Engine sparse backend documents newer PyTorch and Triton requirements than Maestro currently ships. Dropping it directly into Maestro would require a potentially disruptive runtime upgrade.
- The community Sol-Attn repository's license must be confirmed before copying implementation code. Prefer an independently written integration against appropriately licensed primary code.

## Primary-source findings

### SageAttention

SageAttention2/2++ replaces dense attention with quantized kernels. The official project supports Ampere, Ada, and Hopper GPUs and specifically reports RTX 4090 support. Its minimum documented environment is compatible with Maestro's installed PyTorch, Triton, and CUDA versions.

Source: <https://github.com/thu-ml/SageAttention>

### Sol-Attn

Sol-Attn uses block-sparse attention with on-the-fly routing and correction. Its quality/speed tradeoff is controlled by tau, schedule range, exact conditioning sinks, and dense fallback policy.

Sources:

- <https://nvlabs.github.io/Sana/Sol-Attn/>
- <https://github.com/NVlabs/Sana/tree/sol-engine>
- <https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3>
- <https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends>
- <https://github.com/kijai/ComfyUI-SolAttn_triton>

NVIDIA's H3-specific Sol-Engine results are useful for understanding where speed comes from, but not for predicting Maestro performance: their published H3 benchmark used eight GB200 GPUs, a 50-step workload, and different runtime optimizations.

Within that benchmark, caching contributed the largest incremental gain. Sol-Attn provided a smaller additional gain after the dense kernel line was already optimized, demonstrating that accelerator benefits overlap.

### EasyCache

EasyCache estimates changes between adjacent denoising steps. If accumulated change stays under a threshold, it skips the full model call and applies a cached output delta. ComfyUI's H3-aware implementation caches paired video and audio outputs together.

Sources:

- <https://docs.comfy.org/built-in-nodes/EasyCache>
- <https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_easycache.py>
- <https://arxiv.org/abs/2507.02860>

EasyCache is not mathematically lossless. Threshold `0.30` is relatively aggressive and requires H3-specific audio, dialogue, reference, and motion testing.

## Risks in copying the supplied settings directly

### Audio and reference conditioning

The workflows use Sol-Attn's `exact_kv` sink. This makes conditioning keys and values exact, but does not make all conditioning query rows dense. The community node offers `exact_kv_and_rows`, which costs more but is intended to keep the generated audio stream exact.

For Maestro, audio is a first-class H3 output. A visual-only comparison is insufficient. Dialogue intelligibility, voice identity, sound effects, music, stereo coherence, and synchronization must all be tested.

### Morton ordering

The workflows enable 3D Morton ordering. NVIDIA's native H3 Sol-Engine notes that H3's grid is already contiguous and does not require Morton reordering. The community node defaults to per-frame 2D Morton for H3 because H3's temporal spacing is nonuniform.

Do not copy the supplied `morton=true, curve=3d` choice as Maestro's initial default. Begin with Morton disabled and enable it only if controlled measurements show a benefit without temporal or audio degradation.

### Aggressive caching

The supplied `0.30` reuse threshold may maximize benchmark speed at the expense of subtle motion, identity, dialogue, or sound. Maestro should begin around `0.15-0.20`, validate output, and expose stronger settings only as experimental presets.

### Six-step Turbo interaction

At six evaluations:

- SageAttention can still accelerate every dense attention call.
- EasyCache has little safe redundancy to exploit and could remove too much of the already short trajectory.
- Sol-Attn's 20-90% schedule covers only a few evaluations, so compilation/routing overhead may outweigh the saved compute.

Do not enable caching by default with Turbo LoRA. Treat any Turbo-plus-Sol combination as a separate experiment.

## Recommended product design

Keep these controls conceptually separate:

### Turbo LoRA

- Existing Full-model distillation path
- Six steps
- Default LoRA weight `0.70`
- Current behavior remains unchanged

### H3 Fast Runtime (Experimental)

Potential presets after validation:

**Off**

- Existing PyTorch SDPA path

**Balanced**

- SageAttention2++
- Conservative H3-aware cache
- Sol-Attn off initially
- Intended default fast-runtime preset if quality gates pass

**Maximum**

- SageAttention2++ dense fallback
- Conservative-to-moderate H3-aware cache
- Sol-Attn with exact conditioning protection
- Explicit experimental warning

Possible expert controls can remain hidden behind Advanced settings:

- Cache threshold and active range
- Sol tau and active range
- Conditioning sink mode
- Dense block overrides
- Morton mode
- Diagnostic logging

## Staged implementation plan

### Phase 0: Establish a reproducible benchmark harness

Create fixed-seed fixtures for:

- FL2VA text-to-video
- FL2VA image-to-video
- Ref2VA image reference
- Ref2VA image plus voice/audio reference
- Five-second and approximately 14-second outputs
- 16:9 and 9:16 outputs
- Pruned and Full models where supported
- Turbo LoRA on Full models

Record separately:

- Cold-start time
- Warm-generation time
- Text encoder time
- Transformer denoising time
- VAE/audio decode time
- Peak VRAM
- Peak system RAM
- Number of full transformer evaluations
- Number of cached evaluations
- Number of sparse versus dense attention calls

Preserve output pairs for blind visual/audio review.

### Phase 1: Wire SageAttention into H3

1. Add an H3 attention-backend abstraction around `MiniMaxH3Attention`.
2. Retain PyTorch SDPA as the guaranteed fallback.
3. Route compatible BF16, head-dimension-128 packed attention to SageAttention2++.
4. Preserve H3 padding masks and packed text/video/audio layout exactly.
5. Fall back automatically for unsupported devices, dtypes, shapes, or kernel errors.
6. Benchmark T2V, I2V, and Ref2VA before adding any cache or sparse attention.

This is the safest first implementation because SageAttention is already installed and H3 currently bypasses it.

### Phase 2: Add H3-aware caching

1. Implement caching at the H3 model-evaluation level.
2. Cache and reuse video and audio velocities as an inseparable pair.
3. Start with a conservative threshold around `0.15` and a limited middle schedule.
4. Never cache the first or final evaluations.
5. Reset all cache state between jobs, windows, reference manifests, dimensions, and model/LoRA changes.
6. Disable by default for six-step Turbo runs.
7. Add counters and concise diagnostics so speedups can be verified rather than assumed.

Evaluate both EasyCache-style output-delta reuse and NVIDIA's H3-oriented FirstBlockCache design before choosing the final algorithm.

### Phase 3: Add optional Sol-Attn

1. Resolve implementation licensing and dependency strategy first.
2. Avoid a mandatory PyTorch/Triton upgrade for all users if possible.
3. Require BF16, head dimension 128, supported NVIDIA architecture, and compatible sequence layout.
4. Preserve all text, visual-reference, and audio-conditioning rows exactly.
5. Begin with Morton disabled.
6. Keep early steps and sensitive transformer blocks dense.
7. Use automatic dense fallback for masks, short sequences, unsupported shapes, or kernel errors.
8. Verify both video and audio quality before exposing it broadly.

### Phase 4: Combination tuning

Benchmark these independently before combining them:

1. Baseline SDPA
2. Sage only
3. Cache only
4. Sol only
5. Sage plus cache
6. Sage plus Sol
7. Sage plus Sol plus cache
8. Full-model Turbo plus Sage
9. Full-model Turbo plus Sage and Sol, experimental only

Do not infer combined performance by multiplying standalone speedup numbers.

### Phase 5: UI, rollout, and safe fallback

1. Add one simple **Fast Runtime (Experimental)** control to MiniMax H3 models.
2. Keep detailed controls in Advanced settings.
3. Detect support before showing or enabling a preset.
4. Display a one-time first-run compilation notice for Triton kernels.
5. Fall back to SDPA automatically and visibly if an accelerator fails.
6. Store accelerator settings in generation metadata and Load Settings.
7. Add regression tests for T2V, I2V, Ref2VA, aspect ratio, audio conditioning, Turbo compatibility, cancellation, and model switching.

## Acceptance criteria

A runtime preset should not ship as the default unless it meets all of the following:

- No crashes or corrupted output across supported H3 models and modes.
- No significant increase in OOM frequency.
- Deterministic fallback when a kernel is unsupported.
- Meaningful warm-run end-to-end speedup, not merely a faster attention microbenchmark.
- No obvious loss of identity, motion, prompt adherence, image conditioning, or reference mapping.
- No added gibberish, missing dialogue, voice drift, sound dropouts, stereo artifacts, or audio/video desynchronization.
- Settings persist and reload correctly.
- Turbo LoRA behavior remains unchanged when Fast Runtime is off.

## Likely outcome

The most promising order is:

1. SageAttention for the lowest-risk immediate gain.
2. Conservative H3-aware caching for the largest likely additional gain on normal 20-step runs.
3. Sol-Attn for optional maximum acceleration after audio/reference-safe tuning.

A 3x warm-run improvement may be achievable for some 20-step Pruned-model workloads on a 4090, but it should be treated as a benchmark target rather than a product promise.

## Resume checklist

When this work resumes:

1. Re-check upstream SageAttention, Sol-Attn, EasyCache, ComfyUI H3, and Wan2GP changes.
2. Re-check Maestro's installed PyTorch, CUDA, Triton, and SageAttention versions.
3. Re-confirm the licensing of any community implementation considered for reuse.
4. Build the benchmark harness before changing H3 attention.
5. Implement and measure SageAttention alone first.
6. Preserve baseline output files and timing data for every later phase.
