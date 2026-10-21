# Maestro v3.1.0

19 October 2026. Available on public **Main and Dev**, following Dev testing.

## MOSS Image 5B

- Add MOSS Image 5B Base and SFT to Studio → Image → Generate. SFT appears once
  in the enabled model list; Base is available through Manage models. Existing
  model selections and visibility preferences are retained.
- Generate from captions, tags or structured layouts. The visual Prompt Helper
  supports object, character and text boxes, canvas-aware coordinates and exact
  lettering, with single-image, multi-panel and collage arrangements.
- Align enhancement with the official MOSS guidance: preserve minimalist intent,
  counts, style and exact visible words; compile new top-left-pixel layout drafts
  into centered model coordinates in code. Validate the selected canvas and boxes
  without silently clipping. Keep existing compiled layouts, extra fields and
  comments intact. Classic enhancement also preserves literal artist tags and
  email addresses rather than interpreting `@` as an instruction separator.
- Public, pinned BF16 and INT8 ConvRot downloads use Maestro's normal model
  management and offloading. No Hugging Face login is required.
- These checkpoints use 50 steps by default and support text/layout generation;
  they do not accept reference images or masks. The 5B name describes the image
  transformer; the separate text encoder also uses memory and disk space.
- Enable compatible image-transformer LoRAs for Base and SFT, including
  per-step multiplier schedules. The MOSS Image browser filter includes both
  Civitai base labels; supported imports share the `moss_image` LoRA folder.

See the [MOSS usage and validation guide](MOSS-Image.md).

## H3 community checkpoint imports

- Accept retained FP32 input/output and compressed AdaLN projection weights in
  compatible BF16 and INT8 ConvRot exports, with FP32 preservation during model
  loading. Existing shape, finite-curve, quantization and checksum checks remain.
- Ask for the creator's QKV row order when a SafeTensors header omits it,
  matching the existing explicit selection workflow for GGUF imports.
- Keep optional Turbo LoRAs separate from baked checkpoint acceleration.
  Read the base model's “steps without the LoRA” count and avoid interpreting
  “not res_multistep” as sampler advice.

See the [H3 import guide](H3-checkpoint-import.md). Structural import verification
and tiny CPU loader tests do not establish full checkpoint generation quality.

## H3 References speaker compatibility

- Restore support for unambiguous older `(S1) says ...` prompts when an explicit
  Subject/Speaker declaration or an earlier attributed line establishes the
  character within the same clip.
- Share the ownership resolver across generation, prompt enhancement and
  per-window voice selection. Retain independent Subject, Speaker and Audio
  numbering, including a second referenced character speaking first.
- Keep named guests separate from saved characters and preserve exact scripted
  dialogue, including French. Shot, scene, timestamp and production labels do
  not become speaking characters.
- Reject contradictory or unproven ownership when multiple referenced
  characters remain possible. The fix does not guess a character from the
  number in `(S1)`.

This addresses [issue #185](https://github.com/Blizaine/Maestro/issues/185).

## FLUX 3 Klein LoRAs

- Restore LoRAs & presets for both Distilled and Base.
- Add a Flux 3 Klein Civitai browser filter covering Distilled and Base tags,
  with automatic imports into their shared `flux3_klein` LoRA folder. Recognize the
  official Hugging Face base-model metadata and Klein URL hints as well.
- Load compatible native and supported Diffusers adapter targets, including
  fused QKV layouts, and apply multiplier schedules at each denoising step.
- Keep incompatible targets subject to validation. FLUX 2 adapters are not
  interchangeable with FLUX 3 adapters.

## Krea 2 Turbo imports

- Recognize Civitai's explicit Krea 2 Turbo base label and register compatible
  checkpoints using Turbo defaults. Older generic Krea 2 uploads retain the
  RAW/Turbo recipe choice.
- Include both labels in the Krea 2 browser filter, add a Turbo-only filter,
  and route compatible LoRAs to the shared `krea2` folder. Tensor-layout and
  supported-format checks still apply before checkpoint registration.
- Recognize logical tensor dimensions for the supported packed NVFP4
  SafeTensors layout, requiring its matching scale descriptors. Packed bytes
  are no longer mistaken for a different architecture. This does not enable
  unrelated or unverified quantization formats.

## H3 VAE x2, Viggle Animate and Refine faces

- Replace removed Hugging Face revision pins with a verified asset revision.
  Existing downloaded files are reused.
- Repair the H3 VAE x2 selection download as well. The checkpoint, license and
  notice now use the verified replacement revision; the checkpoint's original
  byte size and SHA-256 checks remain unchanged.
- Preserve H3's decoded 8-bit pixels during face refinement. The previous
  double conversion saturated those pixels and washed out facial features.
  Signed floating-point output remains supported with one conversion.
- Retain source dimensions, frame count/rate and copied soundtrack audio in
  the validated refinement workflow.

## Recast memory and diagnostics

- Release full-resolution tracking arrays and their local views after saving
  reference, mask and optional shot files, before video generation.
- Log a SCAIL-2 workload/placement summary and actionable host-RAM diagnostics.
  Preserve saved precision, steps, profiles and memory settings.
- Fast Recast retains its 480p/eight-step/CFG 1 starting recipe. The reported
  20-hour RTX 4080 case had no logs or saved settings available, so its exact
  cause and a speedup on that configuration are not established.

See the [workflow troubleshooting notes](V3-Workflow-Fixes.md).

## Updating and testing

1. Use **Update** on your Main or Dev installation in Pinokio while Maestro is idle.
2. Restart Maestro, refresh the browser and check that the UI shows **v3.1.0**.
3. Try your existing reference-dialogue prompts, adapters, H3 VAE x2, Viggle and
   Refine faces workflows. MOSS downloads its required components when selected.
4. Include saved settings, GPU/RAM details and the console error or workload
   summary with a bug report.

No model-cache deletion or manual prompt migration is required for an
unambiguous supported legacy binding.

## Validation scope

Release checks passed: 4,376 Python unit tests (12 skipped), 138 standalone
preview/media regressions, JSON grammar regressions, Python syntax and undefined
name checks, frontend lint and production build. Desktop/mobile browser checks
covered the MOSS layout helper and FLUX 3 LoRA/reference controls. The public
repository boundary guard passed with the new files staged.

The MOSS LoRA and Civitai follow-up passed 83 targeted Python regressions,
33 URL-hint and 10 import-request checks, frontend lint and production build.
The numerical MOSS check uses a tiny transformer with real MMGP adapter loading
on CPU, verifying zero/nonzero scheduled weights across CFG branches and batches.
An authenticated, bounded header read confirmed the Krea NVFP4 packing and scale
descriptors used by the import check; full generation with that community
checkpoint has not been tested.

The H3 mixed-precision follow-up passed 292 focused CPU regressions, the import
UI/API-client checks, frontend lint and production build. Tiny real SafeTensor
loads through MMGP preserve FP32 AdaLN weights and the stock FP16 path. Both
reported community file headers and bounded quantization markers passed
verification; the BF16 export requires explicit QKV layout selection. Full
generation with these newly admitted files and arbitrary Turbo adapters remains
untested.

The MOSS official-guide follow-up passed 42 CPU regressions, desktop/mobile
layout-browser checks, frontend lint and the production build. Live LLM quality
and new image quality were not measured by those CPU checks. The H3 VAE x2
download follow-up verified all three replacement URLs and passed 34 related
CPU regressions; the checkpoint size and SHA-256 remain unchanged.

The final runtime commit passed the complete public CI suite before release.
Publication changes the release documentation only; the validated runtime is
unchanged.

Workstation 01 (RTX 4090 / 24 GB VRAM / 128 GB RAM) completed normal-API MOSS
Base/SFT caption, layout and batch smoke tests, a Viggle video, compatible
FLUX 3 test-adapter generation, and face refinement with CPU media/audio
verification. H3 compatibility has CPU coverage through both generation
compilation and the real enhancement validation/repair flow with a mocked
writer. It does not depend on a GPU render to reproduce the parser error.

The FLUX 3 GPU adapter fixture has zero weights: it verifies loading and
scheduling, not a trained adapter's style fidelity. MOSS BF16 generation,
8/12 GB MOSS GPU runs, the reporter's Recast setup and arbitrary third-party
adapters still need wider testing. These checks do not establish output-quality
parity with every upstream pipeline or a new performance benchmark.
