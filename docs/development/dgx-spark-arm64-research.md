# NVIDIA DGX Spark / ARM64 Support Research and Plan

Status: research complete; implementation deferred

Research date: 2026-08-11

Audited Maestro baseline: `5b0df69` (`v1.7.1`)

Tracking issue: <https://github.com/Blizaine/Maestro/issues/53>

Community installer and field notes: <https://github.com/gizmax/maestro-dgx-spark>

## Purpose

Preserve the investigation of GitHub issue #53, which documents a working
manual Maestro installation on NVIDIA DGX Spark hardware, and record a safe
path toward supported Linux ARM64 operation.

The reported test system is:

- NVIDIA DGX Spark / GB10 Grace-Blackwell
- Ubuntu 24.04 on `aarch64`
- Compute capability `sm_121`
- CUDA 13
- 128 GB unified CPU/GPU memory
- Python 3.12 with an official ARM64 CUDA 13 PyTorch build

The contributor successfully launched Maestro and completed MiniMax H3
generations. The remaining work is primarily packaging, platform detection,
telemetry, optional dependency handling, and hardware validation rather than a
fundamental incompatibility in Maestro's generation runtime.

## Executive conclusion

Issue #53 remains valid but contains a mixture of current gaps and findings
that have already been superseded by later Maestro releases.

Current conclusions:

- Maestro's normal installer is not ARM64-safe.
- Pinokio currently ships x86_64-only Linux builds, so a supported DGX Spark
  path must remain a manual installer until that external limitation changes.
- GPU telemetry still incorrectly reports the GPU as unavailable when NVML
  supports utilization but not unified-memory information.
- MiniMax H3 attention acceleration is already resolved in Maestro v1.6.5 and
  later: compatible CUDA attention now routes through Maestro's shared
  Sage/SDPA selector.
- Current H3 start, end, injected, and Omni reference image paths normalize
  images to aligned canvases. The older image-grid failure is likely covered,
  but it still needs an explicit arbitrary-source-size regression test.
- Official support should not begin with a broad TorchCodec migration. Maestro
  already has an FFmpeg decoder that can provide a smaller, safer Decord
  fallback.

Do not close issue #53 until a current dev build has been validated on actual
GB10 hardware.

## Current gaps

### 1. Launcher and runtime detection

`launcher_profile.js` recognizes RTX 50 systems through `sm_120` or an RTX 50
model-name match. DGX Spark reports `sm_121`, so it falls into Maestro's legacy
CUDA 12.8 / Python 3.10 profile.

`torch.js` then installs Linux x86_64-only SageAttention, FlashAttention,
LightX2V, and Nunchaku wheels. Those packages cannot be installed on ARM64.

DGX Spark must have its own explicit runtime profile. It should not be treated
as an RTX 50 desktop card merely because both are Blackwell-family devices.

### 2. Requirements that do not resolve on ARM64

The current `app/requirements.txt` includes several assumptions that fail on
the reported system:

| Dependency | ARM64 issue | Working approach reported in #53 |
|---|---|---|
| `onnxruntime-gpu` | No compatible ARM64 wheel from Maestro's configured index | Use CPU `onnxruntime` |
| `rembg[gpu]` | Pulls the unavailable GPU ONNX runtime | Use ordinary `rembg` |
| `torchcodec==0.10.0` | No compatible wheel and tied to an older Torch ABI | Use a Torch-compatible current ARM64 release |
| `decord==0.6.0` | No ARM64 wheel; upstream source needs modern FFmpeg fixes | Build from source or use Maestro's FFmpeg fallback |
| `taichi==1.7.4` | No ARM64 wheel | Make optional; disable or fall back for affected SCAIL pose rendering |
| x86_64 acceleration wheels | Wrong architecture | Skip or compile supported kernels from source |

Requirement markers alone are insufficient for Decord because
`app/shared/utils/utils.py` imports it eagerly. Some model-specific modules
also import it directly. Platform gating must be paired with optional imports
and a clear runtime fallback.

### 3. Unified-memory GPU telemetry

On GB10, `nvmlDeviceGetUtilizationRates()` works while
`nvmlDeviceGetMemoryInfo()` raises `NVMLError_NotSupported`.

Both current telemetry implementations query utilization and memory inside one
exception boundary:

- `app/services/live_stats.py`
- `app/shared/utils/stats.py`

The memory exception therefore discards a valid utilization result and reports
the entire GPU as unavailable.

The safe behavior is:

1. Resolve the NVML device handle.
2. Query utilization independently.
3. Query NVML memory independently.
4. If only the memory query is unsupported, use
   `torch.cuda.mem_get_info()` for free and total CUDA-visible memory.
5. Keep the GPU available when either valid utilization or CUDA memory data is
   present.
6. Preserve the existing zero/unavailable behavior when CUDA itself is not
   usable.

This fallback needs isolated tests so a future NVML refactor does not regress
unified-memory systems.

### 4. Optional media and SCAIL functionality

Maestro already contains useful building blocks:

- `app/shared/utils/video_decode.py` provides FFmpeg-based decoding.
- `app/models/wan/scail/cylinder_renderer.py` contains a CPU cylinder-renderer
  fallback.
- Taichi imports are generally lazy, so the main application does not need to
  load Taichi at startup.

However, the main SCAIL NLF path imports the Taichi-only `render_whole`
implementation, bypassing the separate CPU renderer. ARM64 support therefore
needs either a compatible adapter or an explicit, user-facing feature
availability message.

### 5. Single CUDA context behavior

The contributor reports that GB10 currently permits only one usable CUDA
context at a time in their environment. A second GPU process may receive
`CUDA_ERROR_NO_DEVICE`, and context teardown can take tens of seconds.

Official documentation should instruct users to stop Ollama, vLLM, ComfyUI,
and other CUDA services before launching Maestro, then wait for the prior
context to finish tearing down. This is a platform constraint rather than a
Maestro queueing bug.

## Findings already resolved in current Maestro

### MiniMax H3 attention backend

The v1.5.5 report correctly observed that H3 called PyTorch SDPA directly.
Since v1.6.5, `app/models/minimax_h3/transformer.py` routes compatible CUDA
FP16/BF16 attention through `shared.attention.pay_attention`, retaining SDPA
for CPU, FP32, masked, or unsupported cases.

No additional H3 SageAttention routing change is required for issue #53.
Actual GB10 kernel availability still depends on whether SageAttention can be
built successfully for `sm_121`.

### H3 input-image geometry

Current H3 paths normalize visual inputs before conditioning:

- First/Last keyframes use `prepare_keyframe_image` on the selected output
  canvas.
- Timed injected frames use the same canvas preparation.
- Omni image and video references use resolutions aligned to H3's canvas
  multiple.

This likely prevents the reported Qwen visual-grid split mismatch. Add tests
covering odd-sized landscape and portrait source images before declaring the
old failure resolved.

## Recommended implementation phases

### Phase 1: Low-risk telemetry and regression coverage

1. Add a shared NVML/CUDA memory-query helper.
2. Update both live and legacy stats implementations to use it.
3. Add tests for:
   - ordinary NVML utilization and memory;
   - valid utilization with unsupported NVML memory;
   - CUDA memory fallback failure;
   - a genuinely unavailable GPU.
4. Add arbitrary-dimension H3 start/reference-image processor tests.

This phase is useful beyond DGX Spark and can ship independently.

### Phase 2: ARM64-safe dependency graph

1. Add architecture markers for ONNX Runtime, rembg, TorchCodec, Decord, and
   Taichi.
2. Remove the unused eager Decord import from shared utilities.
3. Route general video decoding to Maestro's existing FFmpeg path when Decord
   is unavailable.
4. Guard model-specific Decord use with a clear capability error or compatible
   fallback.
5. Route SCAIL pose rendering to a CPU fallback where practical; otherwise
   disable only that preprocessing option with a clear explanation.
6. Verify that the application imports and launches in a clean ARM64 virtual
   environment without optional acceleration packages.

### Phase 3: Dedicated DGX Spark runtime and installer

1. Add explicit detection for Linux `arm64` + NVIDIA `sm_121` / GB10.
2. Use a separate Python 3.12 / CUDA 13 runtime profile.
3. Install official ARM64 PyTorch packages without x86_64-only wheels.
4. Make SageAttention source compilation optional and non-fatal.
5. Adapt the community installer for current Maestro rather than copying its
   v1.5.5 package list unchanged.
6. Document launch, update, reset, model storage, and remote browser access.

Until Pinokio provides a Linux ARM64 build, expose this as an official manual
installation path rather than claiming normal one-click Pinokio support.

### Phase 4: Unified-memory budgeting audit

1. Record peak shared-memory use for H3 Pruned and Full, First/Last and Omni.
2. Confirm MMGP's reserved-RAM ceiling does not double-count the same unified
   memory pool as both system RAM and VRAM.
3. Validate H3 weight residency and activation reserves at 480p, 720p, and
   1080p.
4. Confirm that model unloading returns enough unified memory before another
   application starts.
5. Add a conservative DGX-specific profile only if measurements show the
   normal high-memory profile is unsafe.

### Phase 5: Hardware validation and rollout

Ask the issue reporter or another GB10 owner to validate:

- clean installation from a current dev commit;
- startup and GPU telemetry;
- T2V and I2V H3 First/Last;
- Omni image, video, audio, and voice references;
- uploaded video decoding;
- background removal through CPU ONNX Runtime;
- SCAIL behavior with Taichi absent;
- cancellation, model switching, and shutdown/context teardown;
- update and reinstall behavior.

Only after that validation should Maestro advertise DGX Spark support or close
issue #53.

## Out of scope for the first implementation

- Replacing all Decord use with TorchCodec.
- Promising SageAttention availability on every ARM64 Blackwell device.
- Treating unified memory as equivalent to discrete VRAM without measurement.
- Bundling or redistributing third-party source builds without reviewing their
  licenses and update strategy.
- Claiming Pinokio one-click support while Pinokio itself remains x86_64-only.

## Acceptance criteria

- A clean ARM64 requirements installation completes without manual editing.
- Maestro launches with optional Decord, Taichi, GPU ONNX Runtime, and
  x86_64-only acceleration packages absent.
- The UI correctly identifies the GB10 and displays usable memory telemetry.
- Current x86_64 Windows and Linux installs remain unchanged.
- H3 generation succeeds with text and arbitrary-sized start/reference images.
- Unsupported optional features fail locally with an actionable explanation,
  not at application startup.
- The manual installer is idempotent and has a tested update path.
- A real DGX Spark owner validates the release candidate.

## Resume checklist

When this work resumes:

1. Re-read issue #53 and check for new reporter comments or patches.
2. Re-check the community installer against Maestro's current requirements and
   PyTorch version.
3. Confirm current Pinokio Linux ARM64 availability.
4. Confirm the current official CUDA 13 ARM64 PyTorch and TorchCodec versions.
5. Implement and test the telemetry fallback first.
6. Make dependencies optional without weakening x86_64 pins.
7. Add the dedicated runtime/installer only after the import-only ARM64 path is
   clean.
8. Request hardware validation before closing the issue.
