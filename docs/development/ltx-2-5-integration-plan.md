# LTX-2.5 Integration Plan

Status: Native persistent WanGP/MMGP integration implemented locally. The
two-stage T2V/I2V/audio workflow, decoder selection, shared LTX-2/2.3 LoRAs,
managed upscaling, and warm model reuse are wired as of 2026-08-13; hands-on
output and timing parity remain the release gate.

Research baseline: 2026-08-11

WanGP v12.50 decoder audit: 2026-08-12

Pinned references:

- Lightricks `LTX-2` commit `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`
- ComfyUI-LTXVideo commit `ac4d99839020b983e956a8ab67ec38aec1b6e65a`
- LTX-2.5 model revision `28dac7acdc1f78a70e98687db261a949754f8941`
- Official Python packages `ltx-core==1.2.0` and `ltx-pipelines==1.2.0`
- WanGP native LTX-2.5 baseline `37f9111` (v12.50-era implementation)

## Architecture decision

LTX-2.5 is a parallel Maestro family (`ltx2_25`), not an in-place upgrade of
LTX-2.3. Existing 2.3 models, saved jobs, Retake, Outpaint, Director, and custom
MMGP behavior remain unchanged.

The first validation bridge used the official Transformers 5.x runtime in an
isolated per-generation process. It proved the workflow but reloaded the full
BF16 stack for every job, bypassed MMGP's normal dynamic LoRA path, and made
warm follow-up generations impossible.

That bridge is now retired. Maestro ports WanGP's native LTX-2.5 additions into
its existing LTX2 engine, loads the INT8 ConvRot checkpoint by default, profiles
all components through MMGP, applies LoRAs dynamically, and retains the loaded
model between compatible jobs. LTX-2.5 remains a separate Maestro family, so
the native port does not replace LTX-2.3 or its saved model definitions.

## Official workflow audit

Lightricks currently publishes nine 2.5 example workflows:

1. T2V/I2V single-stage distilled
2. T2V/I2V two-stage distilled
3. text-to-audio single-stage distilled
4. Ingredients IC-LoRA
5. inpainting IC-LoRA
6. outpainting IC-LoRA
7. motion-track IC-LoRA
8. union-control IC-LoRA
9. general V2V IC-LoRA

The official single-stage distilled workflow uses:

- the split 2.5 Distilled transformer;
- the LTX-specific Gemma 4 12B encoder with bundled projection;
- the fast convolutional video VAE by default, the optional NAD diffusion
  video VAE, and the audio VAE;
- 24 FPS;
- dimensions divisible by 32;
- an `8n+1` frame lattice;
- nine fixed sigma points/eight denoising evaluations: `1.0, 0.99375,
  0.9875, 0.98125, 0.975,
  0.909375, 0.725, 0.421875, 0.0`;
- Euler ancestral sampling with CFG 1; and
- image conditioning strength 0.7 in the published I2V example.

LTX-2.5's distilled release uses the ancestral stage-one sampler. Reusing the
deterministic 2.3 sampler is not equivalent.

The official Comfy workflows explicitly reuse a tested subset of 2.3 IC-LoRAs
with the 2.5 distilled transformer: Ingredients, In/Outpainting, Motion Track,
Union Control, and Instant Shave/V2V. Maestro will whitelist these by workflow;
it will not infer compatibility for every 2.3 LoRA from its filename.

The Python and Comfy examples expose both direct and multiscale variants. The
production Maestro path now follows the official Python `DistilledPipeline`:
eight ancestral evaluations at half resolution, learned latent x2 spatial
upscaling, then three deterministic evaluations at target resolution. This is
also the successful high-resolution structure used by current WanGP. Saved
single-stage request payloads are migrated to this supported path.

## Components and access

The native runtime downloads:

- `ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors`
- `gemma4-12b-ltx-v1` (INT8 encoder plus tokenizer files)
- split video/audio embedding connectors matching the transformer quantization
- `ltx-2.5-22b_video_vae_bf16.safetensors` (Fast VAE)
- `ltx-2.5-22b_diffusion_video_vae_bf16.safetensors` (optional NAD decoder)
- the native audio VAE, vocoder, text projection, and spatial upscaler

The Hugging Face repository is gated. Users must accept its terms and sign in
with a read token. Maestro's existing Hugging Face downloader supplies the
saved token and its LTX-2.5 error path explains 401/403 access failures.

Later phases add the temporal upscaler, Dev transformer, distilled LoRA, DFR
detailing LoRA, and optional duration head only when their workflow needs them.

## Phase 1 — implemented locally

- New `LTX-2.5 — Distilled` Studio model.
- Initial official direct distilled T2V baseline (superseded by Phase 2 for
  production generation).
- Start-image and end-image conditioning.
- Synchronized native audio generation.
- Official 8-step ancestral sampler and fixed CFG 1.
- Automatic `8n+1` repair for saved/raw frame counts.
- 64-pixel two-stage resolution alignment and 24 FPS.
- Native INT8 ConvRot transformer plus MMGP profiling/offload.
- Fast convolutional VAE is the default decoder, matching WanGP v12.50 and the
  official memory-efficient ConvVAE path.
- Optional experimental NAD Diffusion VAE remains selectable in Advanced;
  only NAD receives the diffusion-specific tiling, allocator, NATTEN, and
  Triton compatibility path.
- Persistent in-process model reuse across compatible follow-up jobs.
- Shared LTX-2/2.3 LoRA discovery and WanGP-compatible dynamic key mapping.
- Cancellation uses Maestro's normal sticky interrupt lifecycle.
- Compatible Director workflows are exposed through the existing model audit.

Phase 1 release gates:

1. Install the gated assets successfully with a valid HF read token.
2. Generate 576p T2V at 121 frames.
3. Repeat with a start image and confirm 0.7 conditioning behavior.
4. Confirm native audio exists and remains synchronized.
5. Test portrait output.
6. Cancel during model loading and during denoising.
7. Switch back to LTX-2.3 and MiniMax H3 without restarting.
8. Record peak VRAM, RAM, load time, and denoise time on the RTX 4090.

Local engineering validation completed before the first interactive Studio
test:

- official T2V worker: 256x256, 9 frames, 24 FPS, all 8 evaluations;
- official start-image I2V and combined first/last-frame workers at
  conditioning strength 0.7;
- H.264 MP4 with exactly 9 frames plus 48 kHz stereo AAC audio; and
- complete Maestro subprocess bridge returning a
  `(1, 9, 256, 256, 3)` frame tensor and `(16384, 2)` native-audio array.

The 576p/121-frame quality, memory, cancellation, portrait, model-switching,
and timing checks above remain the first hands-on release gates.

## Phase 1.1 — persistence and UX

1. **Completed:** replace the per-generation worker with the native persistent
   MMGP model lifecycle.
2. Add a gated-repository preflight before the large downloads begin.
3. Add disk-space estimates and component-by-component Storage Manager labels.
4. Benchmark the implemented Fast VAE / NAD Diffusion VAE selector across 16,
   24, and 32 GB cards. Fast is the default; NAD is explicitly experimental.

## Phase 2 — two-stage implemented; native multi-shot next

Completed locally:

- Official 8-step half-resolution base pass.
- Official LTX-2.5 learned latent x2 spatial upscaler as a managed gated asset.
- Official 3-step deterministic full-resolution refinement pass.
- Pass-aware 11-step progress and explicit upscaler/decode status.
- Existing aggressive transformer-to-VAE cleanup retained after refinement.
- LTX-2.5-specific 64-pixel-aligned 480p, 540p, 720p, portrait, square, and
  experimental 1080p canvases.
- Standard, Sol, and RTX 50 launches share the normal Maestro runtime path;
  there is no LTX-2.5-specific sidecar lifecycle.

Next:

1. Add native multi-shot prompting as one model call, preserving identity,
   environment, lighting, voice, and audio across internal cuts.
2. Build a 2.5 prompt enhancer for chronological prose, explicit cut timing,
   recurring subject identifiers, and continuous sound design.
3. Show the enhanced prompt before generation.
4. Add the optional duration head only after manual durations are stable.

## Phase 3 — Dev and DFR

1. Add the Dev transformer and official guidance schedules.
2. Add the 2.5 distilled LoRA where the upstream pipeline requires it.
3. Add DFR spatial refinement and its official pixel-detailing IC-LoRA.
4. Add optional temporal x2/x4 refinement.
5. Benchmark BF16+FP8 cast, full BF16, and any official prequantized path that
   becomes available on 16, 24, and 32 GB GPUs.

## Phase 4 — official editing/control workflows

Add and validate independently:

1. Ingredients/reference composition;
2. inpainting;
3. outpainting;
4. motion tracking;
5. union pose/depth/edge control;
6. general V2V/Instant Shave;
7. audio-to-video and text-to-audio;
8. Retake and DubIt;
9. keyframe interpolation; and
10. HDR/EXR.

Only the exact IC-LoRAs demonstrated by Lightricks are eligible for the initial
2.5 whitelist. Each Maestro workflow still needs mask polarity, conditioning
strength, colour, timing, and output regression tests.

## Phase 5 — Director

First add Distilled as a normal per-shot renderer. Then add native multi-shot
blocks:

1. Group adjacent shots sharing cast, location, time, and visual world.
2. Render each group in one native 2.5 call with internal cuts and continuous
   audio.
3. Save the native block, its source shot IDs, and expected cut boundaries.
4. Make Dashboard repair group-aware; initially regenerate the whole block.
5. Enable Music Video audio-driven use only after the 2.5 A2V timing path
   passes rejoin regression tests.

## Full validation matrix

- Windows RTX 4090 24 GB and RTX 5090 32 GB first; 16 GB after memory tuning.
- Standard CUDA 12.8 runtime and CUDA 13/Sol host runtimes.
- T2V, I2V, first/last frames, both aspect ratios, lattice boundaries, and
  supported durations.
- Model switching between LTX-2.3, LTX-2.5, and MiniMax H3.
- Cancellation, restart, interrupted downloads, missing gated credentials,
  update, repair, reset, and storage cleanup.
- Fixed prompt/seed/resolution comparisons against the pinned official Comfy
  workflow.

## Primary references

- Workflows: <https://github.com/Lightricks/ComfyUI-LTXVideo/tree/master/example_workflows/2.5>
- Model card and gated weights: <https://huggingface.co/Lightricks/LTX-2.5>
- Official source: <https://github.com/Lightricks/LTX-2>
- Prompting guide: <https://docs.ltx.io/open-source-model/usage-guides/prompting-guide>
- Community license: <https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md>
