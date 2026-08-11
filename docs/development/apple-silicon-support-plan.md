# Apple Silicon Support Research and Implementation Plan

Status: research complete; implementation deferred

Research date: 2026-08-11

Audited Maestro baseline: `5b0df69` (`v1.7.1`)

## Purpose

Preserve the research and implementation strategy for running a useful subset
of Maestro natively on Apple Silicon. The goal is not to force the current
CUDA/WanGP stack through PyTorch MPS. The safer design is one Maestro product
with a shared UI, Director pipeline, projects, Dashboard, and storage layer,
backed by platform-specific generation engines.

The initial Mac release should support the workflows that already have credible
Metal or MLX implementations:

- local LLM prompting and Director automation;
- selected LTX video workflows;
- MiniMax H3 First / Last and Omni-reference workflows;
- selected image generation and editing models;
- synchronized video and audio where the selected native engine supports it.

CUDA-only features such as SCAIL-2, Wan pipelines, SAM3, and some audio models
can remain unavailable in the first Apple Silicon alpha. The UI must explain
capability differences rather than exposing controls that cannot run.

## Executive decision

Create a backend capability layer and keep the existing Windows/NVIDIA runtime
intact. Add isolated Mac workers for LTX, H3, image generation, and the local
LLM. Each worker should run in its own process and dependency environment so
MLX version requirements and model memory can be managed independently.

```text
Maestro UI / API / Director / Dashboard / Storage
                        |
              Capability + backend registry
                        |
        +---------------+---------------+---------------+
        |               |               |               |
  CUDA WanGP/MMGP   Mac LTX worker   Mac H3 worker   Mac image/LLM workers
   (unchanged)       (MLX engine)      (h3.c)        (mflux + llama.cpp)
```

This avoids a high-risk rewrite and lets each platform advance independently.
It also provides a clean place to report which modes, references, LoRAs,
resolutions, and duration features are available for the active backend.

## Available development hardware

The current local hardware is enough to build a meaningful alpha:

| Machine | Primary role | Expected coverage |
|---|---|---|
| 24 GB M4 MacBook Pro | Compact-memory reference | Installer/startup, LLM, Director, image generation, low-memory LTX, cancellation, memory-pressure behavior, and very short H3 compatibility probes |
| 64 GB M1 Studio Ultra | H3/LTX correctness reference | H3 T2V, I2V, First / Last, Ref2VA image/video/audio references, full LTX workflows, engine switching, and longer Director runs |

The M1 Ultra is a useful correctness and memory-capacity machine, but should not
be treated as a performance benchmark for newer M-series systems. The 24 GB M4
is the compact-memory baseline; it should not be used to imply that practical
H3 generation is supported at 24 GB until testing proves it.

Initial public positioning should remain conservative:

> Apple Silicon alpha: 24 GB or more for Maestro, selected LTX and image
> workflows, and Director automation. MiniMax H3 requires 64 GB or more.
> Additional configurations are experimental.

Sixteen-gigabyte LTX support and H3 on 48 GB systems should remain experimental
until validated on hardware outside the two-machine development set.

## Candidate native engines

### Local LLM: llama.cpp Metal

Use `llama-server` with its native Metal backend. Maestro already speaks to a
llama.cpp-compatible local server, so this should require platform-aware binary
installation and GPU-layer defaults rather than a new application protocol.

### LTX: ltx-2-mlx

`dgrauet/ltx-2-mlx` is a native MLX implementation with text-to-video,
image-to-video, audio-to-video, retake, extend, keyframes, IC-LoRA, HDR, and
other advanced paths. It supports quantized weights and block streaming for
lower-memory systems. Its feature surface is the strongest candidate for the
first Mac video backend.

Integration should use a long-lived subprocess with a small JSON or local HTTP
protocol. Maestro should own job state, cancellation, output metadata, and
project organization; the worker should own model loading and inference.

### MiniMax H3: h3.c

`antirez/h3.c` is a native C/Objective-C implementation using Metal and
MPSGraph. It exposes a CLI and a static C API and currently supports text and
audio generation, First / Last conditioning, and ordered Ref2VA image, video,
and audio references with synchronized H.264/AAC output.

The project is still marked `0.1.0-dev` and is moving quickly. The public API
reviewed for this plan does not yet expose Maestro's full LoRA or sliding-window
control surface. Treat h3.c as a replaceable worker contract, pin a tested
commit for each Maestro release, and initially expose only capabilities proven
against that commit.

Phosphene currently integrates H3 through its own MLX engine rather than h3.c.
Its useful lesson is architectural: it isolates H3 and LTX in separate workers
and environments, unloads one engine before starting the other, and keeps the
application UI independent of the model runtime.

### Image generation: mflux

`mflux-community/mflux` provides native Apple Silicon paths for a selected set
of image models and editors, including members of the FLUX, Krea, Qwen Image,
Z-Image, and related families. Start with a tested subset rather than mirroring
the entire Windows model catalog.

## Current Maestro blockers

The current application assumes NVIDIA/CUDA before a user selects a model:

1. `install.js` rejects Windows systems without NVIDIA and is not structured
   around an Apple runtime profile.
2. `torch.js` installs Windows/Linux CUDA-specific packages and wheels.
3. `app/launch.py` imports the WanGP runtime at startup.
4. `app/wgp.py` performs CUDA capability queries while the module is imported.
5. The main requirements set includes MMGP and CUDA-oriented packages that a
   Mac-only process should never need to import.
6. Model definitions currently describe generation features, but not a strict
   backend capability contract suitable for hiding unsupported UI controls.

The first architectural task is therefore boot isolation, not model inference.
Maestro must be able to launch, show its UI, run the local LLM, and inspect Mac
hardware without importing CUDA-only modules.

## Proposed backend contract

Each backend should report capabilities rather than relying on model-name
special cases. A capability record should include at least:

- operating systems and accelerator families;
- T2V, I2V, start/end frame, and reference media support;
- accepted image, video, audio, and voice-reference roles;
- native audio generation and audio-driven-video support;
- LoRA, sliding-window, keyframe, mask, and edit support;
- legal resolutions, frame lattice, duration limits, and FPS;
- quantization formats and estimated unified-memory requirements;
- progress, cancellation, unload, and health-check operations.

The Director model catalog and Studio controls should consume the same record.
This prevents Director from selecting a theoretically known model whose active
platform backend cannot perform the requested workflow.

## Implementation phases

### Phase 0: Feasibility harnesses

Before changing Maestro startup, validate the native engines directly on both
Macs:

1. Run a short quantized LTX T2V and I2V generation with `ltx-2-mlx`.
2. Run short H3 T2V, start-image, and ordered-reference tests with h3.c on the
   64 GB M1 Ultra.
3. Probe the smallest practical H3 configuration on the 24 GB M4 without
   claiming support if swap or memory pressure is excessive.
4. Record model download size, peak resident memory, swap, load time, denoise
   time, decode time, output dimensions/FPS, audio synchronization, and unload
   behavior.
5. Confirm each engine can be interrupted without leaving large allocations or
   a child process behind.

Exit criterion: at least one repeatable LTX workflow on each Mac and a
repeatable H3 reference workflow on the 64 GB Mac.

### Phase 1: Capability registry and engine interface

1. Define a backend protocol for model discovery, validation, generation,
   progress, cancellation, result metadata, and unload.
2. Wrap the existing WanGP/MMGP path behind the protocol without changing its
   Windows behavior.
3. Replace Director and Studio model-name checks with capability queries where
   practical.
4. Add regression tests proving the Windows catalog and execution parameters
   remain unchanged.

### Phase 2: Mac boot and runtime profile

1. Add an Apple Silicon launcher/runtime profile and platform-aware dependency
   installation.
2. Split common server requirements from CUDA, MLX, and native-worker
   requirements.
3. Lazy-load the CUDA backend only when selected and available.
4. Add Metal/unified-memory system statistics and user-facing backend health.
5. Install and configure llama.cpp Metal for prompt enhancement and Director.
6. Verify install, start, update, reset, and hard-refresh behavior through
   Pinokio on both Macs.

### Phase 3: LTX MLX worker

1. Create an isolated `ltx-2-mlx` worker and pinned environment.
2. Implement T2V and I2V first, then add audio, retake, extend, keyframes, and
   other proven features incrementally.
3. Translate Maestro job parameters into the worker's native request schema.
4. Return ordinary Maestro outputs and metadata so Dashboard, favorites,
   workspaces, and saved settings work without a separate Mac UI.
5. Add model download/storage routing and unified-memory safety presets.

### Phase 4: H3 native worker

1. Pin and build a tested h3.c revision as an isolated worker.
2. Add FL2VA T2V, first image, last image, and first/last image support.
3. Add Ref2VA ordered image, video, audio, and voice-reference mapping.
4. Preserve synchronized stereo audio and validate 16:9 and 9:16 output.
5. Expose only the frame lengths, resolutions, LoRAs, and continuation features
   supported by the pinned engine version.
6. Keep the protocol replaceable so a stronger MLX H3 backend can be evaluated
   later without rewriting Studio or Director.

### Phase 5: Native image worker

1. Select a small, high-value mflux model set for generation and editing.
2. Map Maestro's image controls to per-model capabilities.
3. Share downloaded models safely where formats are actually compatible; do
   not assume CUDA safetensors can be reused by every MLX engine.
4. Add background removal or other utilities only after their Mac dependency
   path is validated.

### Phase 6: Director integration

1. Let Director choose only models supported by the active Mac backends.
2. Reuse Maestro's existing LLM screenplay, shot planning, continuity,
   cancellation, Dashboard repair, and join logic.
3. Add model-aware prompt renderers for the native LTX and H3 workers where
   their prompting contracts differ from the CUDA implementation.
4. Complete one short-film project end to end on each Mac, including saved
   state and Dashboard repair.

### Phase 7: Public alpha and hardening

1. Add structured diagnostics export and opt-in tester reporting.
2. Validate additional M-series generations and memory sizes.
3. Publish a precise compatibility table rather than a blanket "Mac support"
   claim.
4. Document separate weight formats, first-run downloads, expected generation
   times, and unsupported features.
5. Promote individual engines out of experimental status only after repeated
   clean-install, cancellation, and memory-pressure testing.

## External validation matrix

The two local Macs do not cover the installed Apple Silicon base. Recruit
testers for at least:

- 16 GB M1/M2/M3 for launch and constrained LTX behavior;
- 32-48 GB Pro/Max systems for practical LTX and experimental H3;
- 64 GB newer Max systems for H3 performance comparison with the M1 Ultra;
- 96-128 GB systems for larger/full checkpoints and longer reference jobs;
- M5 hardware when available to validate newer Metal behavior.

Every report should include:

- chip, GPU-core count, unified memory, macOS version, and Metal family;
- Maestro commit/version and exact engine/model revision;
- quantization, resolution, frames, steps, references, and seed;
- model load, conditioning, denoise, decode, and total elapsed time;
- peak resident memory, swap, and memory-pressure state;
- cancellation/unload result and any retained child process;
- an anonymized log bundle with personal prompts and media paths redacted.

## Acceptance criteria for the first alpha

- Fresh installation and application startup complete without importing CUDA.
- The UI reports Apple Silicon hardware and only exposes supported models and
  controls.
- The local LLM and Director planning work through Metal.
- A selected LTX model completes T2V and I2V in both 16:9 and 9:16.
- H3 completes T2V/I2V and an ordered-reference generation on the 64 GB Mac,
  with synchronized audio when requested.
- Stop/cancel terminates the native worker operation and releases memory.
- Switching LTX to H3 and back does not require restarting Maestro.
- Outputs appear normally in workspaces, favorites, Dashboard, and saved
  settings.
- A short Director project completes and can resume or repair after restart.
- Existing NVIDIA test suites and launcher behavior remain unchanged.

## Principal risks

- Native engines may require separate model formats and duplicate large model
  downloads.
- H3.c is pre-release and its CLI/API may change between commits.
- Feature parity for LoRAs, sliding windows, masks, and advanced conditioning
  will not exist on day one.
- MLX packages may require incompatible versions, making worker isolation
  mandatory.
- Unified memory can fail through swap or memory pressure even when a model
  technically loads; RAM capacity alone is not a sufficient support claim.
- Model licenses and redistribution terms must be reviewed independently from
  each engine's source-code license.

## Primary sources

- Phosphene: <https://github.com/mrbizarro/Phosphene>
- Phosphene H3 engine notes: <https://github.com/mrbizarro/Phosphene/blob/main/docs/H3_ENGINE.md>
- h3.c: <https://github.com/antirez/h3.c>
- h3.c public API: <https://github.com/antirez/h3.c/blob/main/h3.h>
- LTX-2 MLX: <https://github.com/dgrauet/ltx-2-mlx>
- mflux: <https://github.com/mflux-community/mflux>
- llama.cpp Metal build documentation: <https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#metal-build>

These sources were reviewed on 2026-08-11. The h3.c review included commit
`8974cc0`. Pin exact revisions and re-audit capabilities and licenses when
implementation begins; do not build a release against a moving default branch.
