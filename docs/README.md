# Maestro guides

Start with the [README](../README.md) for installation and the feature overview. These guides cover the controls and model-specific workflows in more detail.

## Create and direct

- [Studio controls](Studio-controls.md) — video, image and audio workflows, queues, gallery search and saved settings.
- [Director controls](Director-controls.md) — music videos, short films, soundtrack analysis and reviewable production stages.
- [Prompt enhancement API](Studio-enhancement-api.md) — queued enhancement, saved drafts and programmatic control.
- [Characters and RefMods](Maestro-Characters.md) — reusable appearance references, portable characters and voice sharing.
- [Speech characters](TTS-Characters.md) — saved voices across speech workflows.

## Video and reference workflows

- [FastH3](FastH3.md) — V2, OmniRef, supported formats and experimental limits.
- [H3 LightX2V accelerators](H3-LightX2V-Accelerators.md) — workflow-matched four/eight-step recipes.
- [H3 Fused LoRAs](H3-Fused-LoRAs.md) — adapter compatibility and sampling guidance.
- [H3 community checkpoints](H3-checkpoint-import.md) — supported imports, verification and workflow pairing.
- [DaSiWa models](DaSiWa-models.md) — checkpoint variants and native formats.
- [H3 reference audio and saved drafts](H3-Reference-Audio-and-Drafts.md) — source soundtracks, generated audio and enhancement recovery.
- [Viggle Animate](Viggle-Animate.md) — character replacement and control-video setup.
- [H3 Face Refiner](H3-Face-Refiner.md) — tracked faces and saved character references.
- [H3 VAE x2](H3-VAE-x2.md) — decoding and upscaling options.
- [TaoMate H3](TaoMate-H3.md) — the optional three-step Frames recipe.
- [H3 Singularity](H3-Singularity.md) — experimental reference-generation workflow.
- [Generation previews](Generation-preview.md) — live video and frame preview choices.
- [FlashVSR](FlashVSR.md) — v1.1 and Plus Turbo upscaling.
- [DLSS finishing](DLSS5.md) — optional neural rendering and frame generation on supported systems.

## Image models

- [MOSS Image](MOSS-Image.md) — Base/SFT generation, layouts and prompt enhancement.
- [Qwen Image 2.1](Qwen-Image-2.1.md) — generation, reference editing, Turbo, transparency and VAE memory.
- [Flux 3 Klein](Flux-3-Klein.md) — model variants, references and LoRAs.

## Speech and music

- [Fish Audio Drama 3](Fish-Drama-3.md) — speech, voice cloning, performance cues and acceleration.
- [Long-form speech writing](Speech-writing.md) — duration, writing style and draft recovery.
- [H3 Voice Audio](H3-Voice-Audio.md) — voice references, speech and general audio.
- [YuE2 music](YuE2-music.md) — songs, instrumental generation, covers and experimental personal styles.

## Performance and access

- [Optional CUDA kernels](optional-kernel-pins.md) — supported stacks, pinned downloads and fallbacks.
- [Performance Auto-Tune](Performance-auto-tune.md) — hardware recommendations and manual controls.
- [Duration recommendations](Duration-recommendations.md) — tested and estimated starting points by hardware and workload.
- [H3 on 16GB GPUs](H3-16GB-placement.md) — memory placement and the reported regression.
- [Model downloads](Download-controls.md) — progress, cancellation and retries.
- [Local language models](LLM-runtime.md) — built-in writers, vision and runtime setup.
- [LM Studio memory](LM-Studio-memory.md) — releasing external writer memory before generation.
- [Private remote access and notifications](TAILSCALE_REMOTE_ACCESS.md) — Tailscale HTTPS and device notification setup.

## Updates and development

- [Complete update history](UPDATE_HISTORY.md) — every dated update announcement, including those summarized in the README.
- [WanGP October ports](WanGP-October-2026-Ports.md) — first-batch scope and validation limits.
- [Detailed changelog](../CHANGELOG.md) — development changes and release entries.
- [Dev v3.3.0 release notes](RELEASE_NOTES_V3.3.0.md) — new ports, validation and testing instructions.
- [Main v3.2.5 release notes](RELEASE_NOTES_V3.2.5.md) — stable release behavior and known limits.
- [API examples](../README.md#notification-and-remote-access-api) and [developer prompt bench](../README.md#developer-prompt-bench).
