# Maestro v2.7.0

October 8, 2026. Prepared release candidate; not published. Changes since the
public v2.6.0 Main release.

## Memory management and performance

Maestro now includes an adaptation of WanGP 17 and MMGP 4 for its existing
generation pipelines. It adds optional VRAM and RAM allocator choices, smarter
memory pinning, Windows checkpoint read-ahead, per-output dynamic or manual
weight preloading, compatible attention options and automatic INT8 kernel
selection. The base allocator default remains PyTorch; explicitly applying
Auto-tune recommends MMGP optimized where supported. Unsupported devices
and model layouts retain their compatible path, and saved allocator changes
that require a restart are reported in Settings.

**Settings → Performance → Auto** now considers the detected GPU, VRAM and
system RAM when it recommends memory settings. It can learn from completed,
comparable local generations and choose temporary H3 placements when repeated
results and the model's memory guard support a choice. Learning records stay on
the machine and do not contain prompts or local file paths. Auto does not start
benchmark generations in the background. The profile shown in Settings is a
starting recommendation; a particular H3 job can need a different guarded
placement based on its duration, resolution, references and LoRAs.

H3 also reduces repeated plain-text conditioning work with a small cache that
stores embeddings on the CPU and returns cloned tensors on the active
generation device. Multimodal and keyframe conditioning bypasses that cache.
Spatial VAE decoding can process two tiles together when the CUDA device and
available memory meet its conservative limits; other cases continue one tile
at a time. These paths retain fallback behavior when the hardware, checkpoint
layout or runtime does not support an optimization.

Several auxiliary assets are now downloaded only when a selected model or
preprocessor needs them. This avoids fetching every optional pose, depth,
scribble, flow and audio-conditioning asset during unrelated setup. The first
workflow that needs an asset still downloads it automatically.

See [Performance Auto-Tune](Performance-auto-tune.md), the [WanGP 17 memory
port](development/wan2gp-17-port.md), and the [memory test bench](development/memory-benchmark.md)
for available controls, measurement guidance and limitations.

Bundled MMGP licensing, source revision and allocator binary hashes are recorded in
[Third-party notices](../THIRD_PARTY_NOTICES.md) and the [MMGP provenance
record](../app/mmgp/PROVENANCE.json).

### What the available measurements show

The documented screening used one RTX 4090 with 24 GiB VRAM and 128 GB system
RAM. In matched warm Fused H3 runs, Auto INT8 measured 24.8% lower total time
and 32.1% lower denoising time than Triton. A separate longer reload comparison
measured 27.2% lower total time and 29.5% lower denoising time. In that longer
comparison, Balanced with an 8192 MiB manual preload used 3.72 GiB less observed
peak VRAM than Full with Auto INT8, with essentially the same denoising time.
These are comparisons between settings in the candidate build; they do not
measure a speedup over v2.6.0, establish uncertainty bounds, or predict other
workloads. Peak-memory polling can miss short spikes, and the manual preload
amount is not a cap on total generation VRAM.

The documented measurements do not establish safe limits or performance for
6, 8 or 10 GB GPUs paired with 16 GB system RAM. Start with the machine's Auto
recommendation and increase duration, resolution, references and LoRAs in
separate steps while checking the saved effective settings and output.

### Lower-memory Windows measurements

On the 12-GB RTX 3080 Ti / 32-GB RAM test machine, a matched H3 Pruned Profile 5
repeat took 433.969 seconds on public Main and 207.594 seconds on an earlier
optimized Dev snapshot: 2.09x throughput for that recorded workload. Final
Auto-tune 480p runs took 210.390–211.641 seconds with tighter host-memory limits.
The cross-app repeats placed final manual Dev about 10% behind Wan2GP at 480p,
and Auto-tune Dev 13.6% behind at 720p. Preview settings and supported runtimes
differed, so these are installed-application comparisons, not isolated source
speedups. Changing the optional RAM allocator alone improved measured repeat
time by less than 1%; it remains optional.

A full 23-clip Director project using Fused References, six steps, 720p and a
10.1-second maximum clip length also completed on that machine. Its 3m48.5s
combined video and all source clips decoded cleanly. The recorded video job
took about 2h52m. Media validation establishes completed outputs and timing;
visual quality, singing and lip sync still need human review. See the
[validation record](VALIDATION_V2.7.0.md) for exact settings and limits.

## Shared runtime compatibility

The memory port includes bounded attention staging/head splitting for supported
Sage2 and Sol routes, shorter intermediate-buffer lifetimes, automatic INT8
backend dispatch and VAE convolution/activation changes. Sol remains approximate
and can change generated results. Quantized FP8, NVFP4 and GGUF paths retain their
format-specific kernels and compatible fallbacks.

Explicit-device factories and model placement updates cover eligible Wan, H3,
Flux, Qwen Image, Hunyuan, HiDream, Krea2, Z-Image and audio pipelines. LTX,
LongCat and other routes that still need implicit factories retain their device
wrapper. Kandinsky no longer replaces the selected allocator at import time.
These are updates to existing workflows; new model families and upstream
upsamplers are not included in this release. Optional attention imports and CPU
CI fixtures were updated to verify the retained paths.

## Director workflow

The live generation previews introduced in v2.6.0 now appear in Director's
active pipeline tile as clips render. The tile updates as new previews arrive
without requiring a browser refresh, and Director's render jobs no longer add
duplicate preview tiles to the gallery. Click or tap the preview to show or
hide its information overlay. The same overlay toggle is available in Studio,
so mobile users can see the motion without the progress text covering it. Pause
and resume remain separate controls. Director retains whole-project Stop and ETA;
preview preferences remain consistent as it advances between clips.

Director H3 jobs now use the hardware-aware text-encoder recommendation when
the user has not selected an encoder, and retain that choice in the saved
execution profile for initial generation, regeneration, repair and resume.
An explicit valid encoder remains in effect. The recommendation uses the
detected machine and available H3 variants; it does not guarantee that every
checkpoint or workload fits in memory.

Single-shot Director music videos preserve multiline Context-IR prompts as
one complete shot. Explicit clip boundaries still split actual multi-clip
plans, and legacy Studio prompts retain their paragraph-based behavior.

## Prompt writing and local LLM startup

CUDA device discovery for the local llama-server writer now allows up to 120
seconds for startup. If the probe times out, Maestro reports that CUDA support
could not be verified, includes captured runtime output and suggests commands
for checking the local driver and server. It retains the selected device
instead of silently switching to CPU. The [LLM runtime guide](LLM-runtime.md)
covers retry steps and distinguishes a writer startup problem from an H3
generation failure.

Dialogue enhancement now explains why a draft was rejected and includes the
selected clip duration in a final error. When AI-written speech exceeds the
clip's word budget, Maestro can copyedit the generated lines while preserving
speaker ownership and turn order, then make up to two focused repairs to the
overlong lines. User-supplied lines remain unchanged. If a valid exchange still
cannot be completed, the error identifies the last validation reason and, for
an over-budget draft, suggests increasing the duration or shortening the
dialogue.

## Developer diagnostics and benchmarks

The manual **Developer tests** bench combines diagnostics, optional prompt
regressions, a fixed H3 cold/warm baseline and a bounded memory-profile
comparison. A manual full-suite action is available; installing the harness
does not schedule it. Reports and bounded log bundles are kept in the local
Test-Bench workspace. The [test-bench guide](development/test-bench.md)
documents the worker, supported modes and artifact API.

An optional fleet controller runs tests serially on ready, idle Maestro
installations over the existing private Pinokio app connection. It skips busy
or offline machines, does not start, stop, update or restart them, and leaves
ambiguous remote work running until its receipt can be reconciled. Cross-machine
timings are descriptive; comparisons should be made within the same machine,
runtime and workload.

Optional Windows SSH diagnostics support bounded process and log inspection
after a user sets up a dedicated key and verifies the server host key. Normal
test collection does not require SSH. The prepared candidate also reports an SSH
subprocess exit status or timeout and includes at most 1,500 characters of stderr
without echoing command arguments. Diagnostics collect selected process
metadata and bounded recent log lines; inspect reports for personal prompts or
paths before sharing them.

## Update and limitations

When v2.7.0 is released, use **Update** in Pinokio while idle, restart Maestro
to load the new backend, and refresh the browser. On an existing installation,
use **Settings → Performance → Re-detect** to explicitly apply the expanded
Auto-tune recommendations. Migration preserves existing manual choices. Check
the active allocator and restart if another allocator change is pending. Some
optional assets download on first use.
The new memory paths have fallbacks, but their behavior and performance depend on
GPU architecture, driver, PyTorch/CUDA runtime, checkpoint format, attention,
precision, resolution, duration, available RAM and other running programs.
Compare complete generated media as well as runtime and memory measurements.

The upstream WanGP release's speed claims are not Maestro measurements. The
current evidence covers selected candidate-build comparisons on the documented
machines; it is not a universal speed, OOM or output-quality guarantee.

[Validation and limitations](VALIDATION_V2.7.0.md) · [Changelog](../CHANGELOG.md)