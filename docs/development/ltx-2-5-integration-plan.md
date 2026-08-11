# LTX-2.5 Research and Maestro Integration Plan

Status: research complete; implementation deferred until the MiniMax H3 Sol Engine work is tested and checkpointed

Research date: 2026-08-11

## Recommendation

Add LTX-2.5 as a parallel Maestro model family instead of replacing LTX-2.3.
Ship a focused Studio implementation of the distilled model first, follow it
with native multi-shot generation, and preserve the existing LTX-2.3 Retake,
Outpaint, control, and Director workflows until each one is explicitly
validated against 2.5.

The native multi-shot pipeline is the feature with the highest Maestro value.
Unlike separately generated clips that Maestro later joins, LTX-2.5 can place
multiple camera shots and cuts in one model generation while maintaining
identity, environment, lighting, style, voice, and audio continuity.

## Official upstream findings

The initial release includes:

- native multi-shot video and synchronized audio generation;
- a custom Gemma 4 12B text encoder and bundled projection;
- a lighter convolutional video decoder and a higher-quality diffusion video
  decoder;
- an improved distilled transformer with its prescribed low-step schedule;
- a Dev transformer and optional distilled LoRA for higher-quality pipelines;
- DFR (Diffusion Fidelity Rendering), including spatial upscaling and optional
  temporal refinement;
- an optional duration predictor; and
- official T2V, I2V, audio-driven, Retake, interpolation, IC-LoRA, DubIt, and
  HDR/EXR pipeline references.

LTX-2.5 is not a checkpoint-only update. Its transformer, custom text encoder,
video decoder, audio components, and upscalers are separate components. The
current LTX-2.3 Gemma 3 encoder and decoder path cannot be substituted.

The official Python implementation supports BF16 checkpoints with runtime FP8
casting and CPU offload. The supplied INT8 ConvRot artifacts are currently
documented for ComfyUI and should not be exposed through Maestro until a
verified Python/MMGP loader exists.

The approximate shared quality component pack is 66 GiB. Installing both Dev
and Distilled transformers can push the total beyond 110 GiB, so model download
UX and Storage Manager accounting are release requirements rather than later
polish.

Official generation constraints include dimensions divisible by 32 and a
frame lattice of `8n + 1`. The initial official examples use 24 FPS. Maestro
must represent these as a separate model family's constraints so saved LTX-2.3
runs and timing are not changed.

## Access and license requirements

The Hugging Face repository is gated. Before downloading, Maestro should:

1. Explain that the user must accept the model terms on Hugging Face.
2. Verify that a read token can access the repository.
3. Convert HTTP 401/403 responses into a useful authentication message.
4. Estimate required disk space before starting.
5. Download atomically so interrupted files never appear installed.

The current LTX-2.x Community License permits commercial use without a paid
license below its stated annual-revenue threshold and requires a paid license
for larger entities. Maestro should present and link the upstream terms rather
than interpreting them for users.

## Maestro compatibility audit

Maestro's existing LTX implementation already has useful reusable boundaries:

- component-based checkpoint paths;
- separate transformer, text encoder, VAE, vocoder, connector, and upscaler
  loading;
- MMGP profiling and CPU/GPU offload;
- model-aware Studio and Director options;
- native audio/video output handling; and
- download, cancellation, progress, and storage infrastructure.

However, `app/models/ltx2` also contains extensive LTX-2.3-specific behavior,
including Retake, Outpaint blending, union-control IC-LoRAs, ID-LoRA,
transition handling, and decoder assumptions. Updating that namespace in
place would create a high regression risk.

Recommended internal structure:

- a new architecture key such as `ltx2_25`;
- a parallel implementation namespace such as `app/models/ltx25`;
- an exact pinned Lightricks source revision recorded in `UPSTREAM.md`;
- a versioned, hashed component manifest under an LTX-2.5 checkpoint folder;
- reuse of Maestro's MMGP adapter and job lifecycle instead of relying on an
  unreleased Diffusers build; and
- shared 2.5 components reused between Distilled and Dev without mixing them
  with 2.3 files.

Suggested public names:

- `LTX 2.5 — Distilled`
- `LTX 2.5 — Dev`
- Advanced decoder choice: `Fast (Conv)` or `Quality (Diffusion)`

Keep LTX-2.3 installed, selectable, and the default until 2.5 clears the full
regression matrix.

## Phased implementation

### Phase 0 — Upstream and download foundation

1. Recheck Lightricks, WanGP, ComfyUI, and Diffusers immediately before coding;
   release-day manifests and documentation may change quickly.
2. Pin exact `ltx-core` and `ltx-pipelines` revisions and resolve any discrepancy
   about whether 2.3 or 2.5 upscalers are expected.
3. Add the model-family identifier and component metadata without exposing a
   generation option yet.
4. Implement gated Hugging Face authentication, disk estimates, hashes,
   resumable downloads, and atomic publication.
5. Integrate the files with Storage Manager and duplicate/shared-file
   accounting.

### Phase 1 — Studio distilled MVP

Support:

- text-to-video;
- start-image-to-video;
- native synchronized audio;
- the official distilled sampling schedule and CFG behavior;
- 24 FPS, `8n + 1` frames, and dimensions divisible by 32;
- Maestro cancellation, progress, download, and output metadata; and
- calibrated MMGP offload profiles for 16, 24, and 32 GB GPUs.

Use the ConvVAE as the default stability/performance decoder. Hide LTX-2.3-only
controls such as existing Outpaint, Retake, union control, ID-LoRA, and
transition LoRA until separately validated. Use BF16 weights with FP8 casting
and offload instead of the ComfyUI-only INT8 checkpoints.

### Phase 2 — Native multi-shot and quality decoder

1. Add the diffusion decoder as an optional Quality mode.
2. Add a native multi-shot generation mode that remains one model call.
3. Build an LTX-2.5 prompt enhancer that writes chronological natural prose,
   explicit transitions, recurring subject identifiers, re-established
   composition after cuts, and continuous or intentionally changed audio.
4. Target two to four shots per generation by default.
5. Show the enhanced prompt before generation.
6. Add duration prediction as an optional Auto Duration control while keeping
   manual duration available for reproducibility.

### Phase 3 — Dev model and DFR

1. Add the Dev transformer and its official guided/two-stage schedules.
2. Add the optional distilled LoRA where the official pipeline calls for it.
3. Add DFR spatial refinement.
4. Add optional temporal x2/x4 refinement only after spatial DFR is stable.
5. Benchmark Dev versus Distilled and ConvVAE versus DiffVAE across each
   supported runtime.
6. Consider NVFP4 only after an official Python path or a verified MMGP loader
   is available.

### Phase 4 — Editing and control validation

Validate and expose features individually:

1. audio-driven video;
2. Retake and DubIt;
3. keyframe interpolation;
4. inpainting and outpainting;
5. control-video and other IC-LoRAs;
6. HDR/EXR output; and
7. LTX-2.3 LoRA compatibility.

Do not list every 2.3 adapter for 2.5 based only on its filename. Record target
architecture metadata and enable only tested combinations. Current upstream
compatibility wording is not consistent enough to treat all older adapters as
safe automatically.

### Phase 5 — Director integration

First add Distilled as a normal per-shot renderer. Then add native multi-shot
groups:

1. Group two to four adjacent shots that share cast, location, time, and visual
   world.
2. Render the group in a single LTX-2.5 call with explicit cuts and audio
   continuity.
3. Save the native multi-shot result as a first-class Director block.
4. Store expected internal cut boundaries and source shot IDs.
5. Make Dashboard repair group-aware. Until internal-shot replacement is proven,
   repairing one embedded shot should regenerate its entire native block.
6. Enable Music Video audio-driven use only after the 2.5 A2V path passes timing
   and rejoin regression tests.

### Phase 6 — Validation and release gates

Test at minimum:

- 16 GB, RTX 4090 24 GB, and RTX 5090 32 GB systems;
- Maestro's standard runtime and the Sol/CUDA 13 runtime;
- T2V, I2V, audio, both aspect ratios, frame-lattice boundaries, and supported
  durations;
- model switching between 2.3 and 2.5 in the same process;
- cancellation, restart, interrupted downloads, storage cleanup, and missing
  gated credentials;
- native multi-shot identity, dialogue, audio, location, and lighting
  continuity; and
- Director save, resume, repair, regeneration, and final joining.

Compare a fixed prompt/seed/resolution matrix against the official LTX Python
or ComfyUI workflow wherever exact parity is possible. Release only when 2.3
regressions remain at zero and incomplete 2.5 component sets cannot be selected.

## Recommended release order

1. Distilled Studio T2V/I2V/audio.
2. Native multi-shot and the dedicated prompt enhancer.
3. Quality diffusion decoder.
4. Dev and DFR.
5. Editing/control features.
6. Director native multi-shot groups.

This order puts LTX-2.5's genuinely new cinematic capability into Maestro
early without coupling the first release to every advanced 2.3 workflow.

## Primary references

- Model card and gated weights: <https://huggingface.co/Lightricks/LTX-2.5>
- Official source: <https://github.com/Lightricks/LTX-2>
- Prompting guide: <https://docs.ltx.io/open-source-model/usage-guides/prompting-guide>
- Community license: <https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md>
- DFR reference pipeline: <https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/dfr_pipeline.py>
- WanGP status to recheck before implementation: <https://github.com/deepbeepmeep/Wan2GP>
