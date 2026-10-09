# WanGP 17 memory and speed port

Maestro's local port uses WanGP 17.0 plus its initial 17.01 fixes, pinned to
[`0e58385fbde7ff102d276e4a9e490845de76b4ea`](https://github.com/deepbeepmeep/Wan2GP/tree/0e58385fbde7ff102d276e4a9e490845de76b4ea)
on 2026-10-05. This is an adaptation to Maestro's existing pipelines, imported
checkpoint layouts, multi-window generation, previews and editor workflows.
It is not a replacement of Maestro with upstream's UI or configuration.

## What the upstream update changes

The main change is MMGP 4, now shipped as source inside WanGP. It combines
block prefetching, reusable VRAM slots, parallel RAM pinning and a small pinned
staging ring. Its optional CUDA allocator reuses freed VRAM more efficiently;
dynamic preload measures activation pressure and retains extra weights only
when there is room. These improvements primarily affect streamed models and
long or high-resolution generations, rather than reducing sampling steps.

The attention work releases K/V and intermediate buffers sooner, adds staged
Sage2 quantization, and splits supported projections by attention heads. Other
changes shorten activation lifetimes, band large VAE convolutions and make
tensor devices explicit to avoid PyTorch's per-call default-device overhead.

The upstream release advertises substantial H3 VRAM reductions and up to 25%
faster Profile 4 / 50% faster Profile 5 generation. Those are **upstream
measurements, not verified Maestro results**. Quantization, attention, LoRAs,
resolution, duration, RAM and GPU architecture all affect the result. See the
[upstream release notes](https://github.com/deepbeepmeep/Wan2GP#readme).

## Included locally

| Area | Integration |
| --- | --- |
| MMGP 4 | Bundled `app/mmgp`; deterministic local imports, pinned dependencies, source revision and allocator binary SHA-256 hashes. |
| VRAM allocation | PyTorch default, MMGP optimized, and MMGP optimized with RAM spilling. Installed before CUDA allocations, with visible fallback and restart status. |
| RAM transfers | Smart pinning, Windows read ahead, and a configurable reserved-RAM ceiling. |
| Preload | Separate Video / Image / Audio choices: profile default, dynamic, or manual MB. Legacy preload values and CLI overrides remain authoritative. |
| Attention | Staged Sage2 and guarded head splitting for compatible Wan and H3 routes; custom checkpoint/LoRA paths retain ordinary forwards when grouping is unsupported. |
| Checkpoint kernels | INT8 row projection provider, optional Kitchen backend, FP8 channel scales and activation quantization, and NVFP4 scale preservation through MMGP routing. |
| Preview decoding | Bounded temporal decoder pieces, preserved recurrent frame state, selected-frame filtering and cancellation checks. |
| Tensor placement | Paired model factory fixes and a guarded generation context that removes the default-device wrapper only from audited model routes. |
| Startup | Optional Windows process power-throttling opt-out, reduced CUDA stack reserve, lazy GGUF CUDA probing and optional allocator diagnostics. |

The lazy GGUF probe also caches failure diagnostics as text rather than an
exception object. This prevents a failed optional CUDA probe from retaining
loader frames and mapped checkpoint storage after model unload; a Windows
fixture-cleanup test and caller-lifetime regression cover that path.

### Model-family audit

The MMGP engine applies to existing managed pipelines. Removing PyTorch's
default-device wrapper is a separate, guarded optimization: a route must
declare `device_explicit` after its tensor factories and callers are paired.

| Family | Model-side integration / placement policy |
| --- | --- |
| H3 | Maestro's custom transformer, VDN and Sol activation lifetimes and attention routes were adapted. Its default-device wrapper remains enabled. Fused/interleaved checkpoint layouts and unsupported LoRA projections keep their compatible path. |
| Wan | Attention, chunked FP32 RoPE, explicit rotary grids, WanMove trajectories, OVI and SCAIL2 factory changes. OVI is audited; vanilla Wan retains the wrapper because common inputs and some SCAIL / diffusion-forcing preprocessing still use implicit devices. |
| Flux / Flux2 | VAE activation handoffs, bounded normalization/upsampling/convolutions and paired position/timestep factories. Placement opt-in covers Flux, Schnell, Klein 4B/9B, Flux2 Dev, Chroma Radiance, Dev USO and Pi Flux2. |
| Qwen Image / Qwen21 | VAE convolution bands and activation/cache lifetime changes preserve Maestro's tiling and streaming. Qwen Image 20B T2I, edit, edit-plus, edit-plus2 and layered routes, plus Qwen21, opt in. |
| Hunyuan / HiDream / Krea2 | Paired device factories. Hunyuan 1.5 T2V/I2V opt in; other Hunyuan variants retain their wrapper. HiDream and Krea2 opt in. |
| Z-Image | VAE and pipeline memory/factory fixes; control retains its wrapper, other routes opt in. |
| LongCat | Factory and mask placement fixes; its existing avatar/reference routes retain their wrapper. |
| LTX2 / LTX audio | Schedules create tensors on the requested device; mel transforms and channel padding stay with their waveform. The wrapper remains enabled because transformer RoPE factories need a separate complete audit. Existing keyframe/retake callers were paired without adding new workflows. |
| Audio / music | HeartMula, Qwen3 TTS, Yue2, ACE-Step/1.5 and base IndexTTS2 opt in after paired factories. Other IndexTTS2 variants retain their wrapper. |
| Kandinsky5 | Removed its import-time allocator and cuDNN-policy overrides so startup settings remain authoritative. |

Maestro's per-job H3 transformer residency controls, preview cotenancy,
checkpoint verification, native quantized forwards, existing INT8 fallback
repairs and generation cancellation remain in place. The profile defaults now
retain MMGP's transformer allowances when the model provides none: 3000 MB
for Profile 2, 1200 MB for Profile 4 and 400 MB for Profile 5. Model-supplied
budgets, manual preload and explicit job allowances take precedence.

## Settings and a useful first test

Restart Maestro once to load the new backend. Open **Settings → System →
RAM / VRAM Management**. Saved allocator preferences are preserved; the new
allocator is opt-in. A recommended first comparison is:

1. Keep the same checkpoint, prompt, seed, resolution, steps, window count,
   preview mode and memory profile for both runs.
2. Select **MMGP optimized**, keep **Smart Memory Pinning** enabled, and
   restart Maestro. Check the setting's **Active** line and terminal startup
   message; unsupported configurations show their fallback reason.
3. With Profile 2, 4 or 5, choose **Dynamic** Video preload. Profile 3.5 is
   also supported, as is Profile 3, because Maestro gives them a budgeted,
   asynchronous path.
4. For long H3/Wan sequences, try **Medium** Attention Head Split and Sage2.
   Try **Auto** INT8 backend for compatible INT8 checkpoints. Compare image
   quality as well as time and peak memory. Unsupported paths fall back.
5. Record checkpoint loading/pinning time separately from denoising and
   decoding. Compare a warmed second run as well as the first run: dynamic
   preload learns the current model's memory use.

Test one setting at a time when investigating a regression. Maestro's existing
duration warnings are estimates; this port does not establish new universal
duration or packed-row limits for every GPU and checkpoint.

## Defaults, timing and limitations

- **Allocator:** remains PyTorch by default. MMGP's native allocator needs a
  supported NVIDIA/CUDA driver, PyTorch and Windows/Linux x64 runtime. CPU,
  HIP and unsupported native-library configurations retain PyTorch. The RAM
  spilling option trades speed for the ability to handle modest VRAM overflow.
- **Smart pinning:** enabled by default. A reserved-RAM ceiling of zero uses
  MMGP's automatic/environment policy plus Maestro's existing H3 Full sizing.
  It is a ceiling, not an eager reservation of that fraction of RAM.
- **Read ahead:** off by default. It is intended for Windows checkpoint
  loading and can increase the OS file-cache footprint.
- **Preload:** defaults to the existing profile policy; saved positive legacy
  preload MB migrates to Manual for each output. `--preload` still overrides
  per-output preferences. Dynamic needs an active MMGP allocator and async
  transfers. Profile 1 retains its normal policy; 4.5 disables async
  transfers and cannot use dynamic preload. A manual value does not change
  Profile 3/3.5's 70% budget.
- **Head split:** off by default, applies to the next generation. Short
  sequences, unsupported weight layouts or adapter types retain their normal
  path. Staged Sage2 and Sol paths have their own hardware/runtime guards.
  Sage2 staging uses supported SM89/SM120 kernels. Sol's staged route needs
  BF16, 128-wide heads and a sufficiently long sequence (8192+ H3 tokens).
  Sol now follows upstream's inline-Q dispatch on eligible long sequences;
  it remains approximate attention, so compare rendered quality with your
  usual settings as well as memory and speed.
- **INT8 backend:** a saved enabled legacy toggle stays Triton; disabled stays
  PyTorch. Auto tries Kitchen, Triton, then PyTorch. The terminal reports the
  resolved backend. FP8/NVFP4/GGUF formats retain their respective kernels.
- **Applying changes:** allocator changes need a process restart. Pinning,
  read ahead, preload and INT8 changes reload models on the next generation.
  A launch argument `--vram-allocator` overrides the saved allocator; remove
  it when restarting if the UI choice should control startup.

Optional diagnostics: launch with `--vram-debug 64` while using an MMGP
allocator to save reports for allocations of at least 64 MB under
`outputs/vram_debug`. `WANGP_AUDIT_DEFAULT_DEVICE=<file>` records implicit
tensor factory sites instead of removing the default-device wrapper.

## Scope reviewed separately

The H3 learned VAE x2 decoder was integrated separately after the memory
port; see [its guide](../H3-VAE-x2.md) and the model provenance in
`app/models/minimax_h3/UPSTREAM.md`. The LTX detail refiner, LTX tiled fusion
and HDR/keyframe features, SeedVR2 changes, Deepy UI changes and new model
packages remain separate workflows or absent local modules. They are not
advertised as enabled by this memory port and need their own validation. Maestro's llama-server writer is a separate
engine; MMGP/attention gains do not imply the same speedup for that writer.

Qwen3.5 GDN recurrent-state replay was reviewed separately. Maestro's shared
nanoVLLM engine has no MTP/speculative verification-and-commit lifecycle, which
the upstream replay kernel requires. Adding only those kernel hooks would be
unreachable or unsafe; that optimization belongs with a separate speculative
decoding engine integration.

## Verification

Validation includes CPU numerical parity and fallback tests, allocator startup
ordering, exception-safe device restoration, settings migration/API validation,
profile budgets, preview temporal continuity, checkpoint adapters, local package
selection and allocator binary provenance. The React interface is type-checked,
linted and built.

On the local RTX 4090 with PyTorch 2.10.0 / CUDA 13, isolated bounded probes
verified both native allocator options and release of their 320 MB allocation.
These did not force an actual system-RAM spill. Staged Sage2, head-group Sage2,
and staged Sol matched their ordinary attention outputs exactly for the tensors
tested. Sol's test included an unaligned 37-token conditioning prefix and
in-place query output. A tiny MMGP Profile 5 streamed model, including dynamic
preload measurements and cleanup, matched its CPU reference within 1.7e-8.
Auto selected Comfy Kitchen CUDA for an 8192-row BF16 INT8 projection; four
128-row output groups matched the layer's ordinary output exactly, and MMGP
consumed the handed-off input as intended.

Small Flux/Flux2 VAE encode/decode comparisons matched the pre-port source
exactly. Qwen Image's single- and multi-frame decoding differed by less than
8e-7; Qwen21 decoding matched exactly. Banded convolution tests also compare
input/weight/bias gradients, not just shapes.

Final integration checks: **3321 unit tests, 3310 passed / 11 skipped**, plus
**138 standalone pytest checks** and **5 JSON-grammar checks**, all passed.
The skipped unit tests require separate CUDA/model setups; they are not counted
as passes. UI lint and type-check/build passed. Python 3.10 syntax checks covered
all changed/new Python sources; undefined-name checks, whitespace validation
and the clean-repository guard (including new files) passed.

A full-generation follow-up on **2026-10-07** completed 21 H3 generations on
the RTX 4090 with MMGP Optimized active. It compared INT8 kernels, streaming
profiles, fixed/manual/dynamic preload, pinning, read ahead, Sage2 head splitting
and Sol, including a longer reference-and-LoRA confirmation. The reusable
[memory test bench and measured results](memory-benchmark.md) document the
controls, timings, peak VRAM and limits. These compare options in the ported
build; they do not establish a speedup over an earlier release or compatibility
and performance on every other GPU/runtime.

## Attribution and license

The port and Maestro adaptations were made on **2026-10-05** from the pinned
WanGP revision above. MMGP's package metadata/license pointers were adapted for
bundled imports. Startup, settings, attention and model changes were integrated
with Maestro's existing implementations rather than overwriting those pipelines.

WanGP/MMGP contributions retain **WanGP Community License 2.0**, included in
[`app/mmgp/LICENSE.txt`](../../app/mmgp/LICENSE.txt). Original third-party
copyright and licenses (including TAEHV, model components and Sol/Saganaki)
remain in their source directories. This component notice does not replace
Maestro's existing licensing. No affiliation or endorsement is implied.

See [`app/mmgp/PROVENANCE.json`](../../app/mmgp/PROVENANCE.json) for the source
revision, modification record and allocator binary hashes.
