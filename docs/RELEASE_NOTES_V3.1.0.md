# Maestro v3.1.0 — Dev testing

19 October 2026. This update is for the public **Dev** branch. Main remains on
v3.0.0 while the new features and fixes receive wider testing.

## MOSS Image 5B

- Add MOSS Image 5B Base and SFT to Studio → Image → Generate. SFT appears once
  in the enabled model list; Base is available through Manage models. Existing
  model selections and visibility preferences are retained.
- Generate from captions, tags or structured layouts. The visual Prompt Helper
  supports object and text boxes, canvas-aware coordinates and exact lettering.
  Dedicated prompt enhancement preserves the layout and selected canvas.
- Public, pinned BF16 and INT8 ConvRot downloads use Maestro's normal model
  management and offloading. No Hugging Face login is required.
- These checkpoints use 50 steps by default and support text/layout generation;
  they do not accept reference images or masks. The 5B name describes the image
  transformer; the separate text encoder also uses memory and disk space.

See the [MOSS usage and validation guide](MOSS-Image.md).

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

## Viggle Animate and Refine faces

- Replace removed Hugging Face revision pins with a verified asset revision.
  Existing downloaded files are reused.
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

1. Use **Update** on your Dev installation in Pinokio while Maestro is idle.
2. Restart Maestro, refresh the browser and check that the UI shows **v3.1.0**.
3. Try your existing reference-dialogue prompts, FLUX 3 adapters, Viggle and
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
