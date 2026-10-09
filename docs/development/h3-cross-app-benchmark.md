# Cross-app H3 Pruned comparison

Use an idle machine and run one application at a time. The reference machine is AI-WRKST-02: RTX 3080 Ti 12GB, 32GB RAM. Pin each application commit and record Python, Torch, driver and allocator versions; the installed apps have different Torch versions, so this measures their supported installations, not just isolated source changes.

## Shared workload

Use `MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors`, the Q2_K H3 text/vision encoder and INT8 ConvRot video VAE. Share the original files rather than quantizing or downloading a different variant. Match seed 424242, 20 steps, CFG 1, flow shift 7, Sage2 attention, no LoRAs, no enhancement, and preview off where supported. The installed WanGP 17.17 benchmark used legacy RGB previews while Maestro previews were off; record this difference rather than treating preview settings as identical. The prompt is a red kite flying above a grassy field with wind and distant birds, without speech or music.

Start at 864x480, 124 frames at 24 fps (5.167 seconds), one window. Run Profile 4 and Profile 5 with one first-load run and one repeat. A repeat means no explicit model-release request; each runtime can still reload internally. Download and initial installation time are separate from render time. Retain failures, and exclude incomplete, mismatched or unverified outputs from speed comparisons.

Record API wall time, denoising phase, peak physical GPU use, system commit, available RAM, paging activity and actual output metadata. Decode each completed output to verify dimensions and frame count. For WanGP, use its documented asynchronous MCP v2 API, discover model defaults/capabilities, and inspect saved media settings. Its model-specific encoder/VAE selectors and model names differ from Maestro's.

## Follow-up cases

1. Compare the best successful profiles between Main, Dev and WanGP.
2. Restart Dev with RAM allocator `default` and `mmgp` for a controlled opt-in comparison. Do not switch allocators inside a live process. Record active/fallback state, not only the saved preference.
3. Run a 1280x704, 124-frame case with the winning low-memory placement, then a longer window or a two-window job if memory margins permit. Lock the requested window size and validate the final decoded output, because automatic window splitting changes the workload.
4. Keep live preview and enhancement timing as separate end-to-end cases. Their extra work must not be presented as transformer performance.

Restore changed settings, stop only benchmark-owned jobs, and leave the machine idle after testing. Keep full configuration backups private. `benchmark_memory.py` refuses busy Maestro queues and restores the selected settings; `benchmark_wangp.py` submits and cancels only its owned jobs. WanGP process-level settings are configured while stopped and restored by the test operator.

For long WanGP runs, the benchmark CLI also writes `<output>.progress.json` (or `--progress-file <path>`) atomically after submission and polling. It retains the owned job ID, case, elapsed time, cancellation state and small structured progress fields, excluding prompts, images and stream logs. If this optional sidecar cannot be written, monitoring continues and the CLI reports one warning. Reconnect monitoring to the saved job ID after a client failure; do not submit a duplicate render.

## Optional RAM allocator

Settings > System > RAM / VRAM Management now includes RAM Allocator. MMGP optimized reuses freed CPU tensor blocks, with a bounded cache that is returned under RAM pressure and when a Studio job or model release finishes. It does not reduce the size of live model weights. The default remains PyTorch until hardware-specific evidence justifies a change. Changing it requires a restart; active status and a fallback reason are visible in the UI/API. `--ram-allocator` overrides the saved choice.

The allocator source and Windows/Linux binaries come from the pinned WanGP 17.17 commit recorded in `app/mmgp/PROVENANCE.json`. The main MMGP engine remains at its separately recorded version. Native support is checked before installation; unsupported systems retain the PyTorch allocator. Linux GPU performance requires its own measurements.

## Confirming Auto-tune

The controlled Maestro matrix deliberately disables Auto-tune while comparing explicit settings. Confirm automatic placement separately: apply Auto-tune while idle, keep it enabled, omit `override_profile` and manual window-memory overrides, and inspect the completed job's `performance_plan`. Record the final transformer budget, reserved RAM fraction, profile and read-ahead settings rather than inferring placement from an earlier VRAM-guard candidate log.

A matching final placement can reuse a resident model. A real change in available RAM, activation allowance, profile or explicit preload may still require a reload. Model release and allocator changes also invalidate a warm comparison. Retain both first and repeat timings and explain any internal reload.
## Native H3 attention component check

`tests/test_int8_backend_rows.py` contains an opt-in CUDA integration check.
Set `MAESTRO_TEST_INT8_ROWS_NATIVE_SLA=1` in an idle, isolated process, then run
`python -m unittest discover -s tests -p test_int8_backend_rows.py` with the
installed Maestro GPU runtime. Default discovery skips this CUDA case.

The check uses native-size H3 geometry (219,886 packed rows), real ConvRot
INT8 projections, RMS/RoPE and sparse attention. It retains a block residual,
assembles the full attention output, checks all eight head groups and protects
1,606 media-prefix rows. It verifies that the original normalized input is
released before the first sparse call and caps the process allocator at
12 GiB. This component check on a larger GPU does not establish a complete
render on a physical 12-GB GPU or account for all model weights and VAE work.

The Triton row provider requires inference-only compatible CUDA FP16/BF16
ConvRot INT8 weights, group size 256, compatible locked activation-quantization
K tiles and no active LoRA. It retains ordinary native dispatch and unsupported
paths. Preparation leaves the shared source untouched for mixed formats;
rotation uses bounded tiles. GPU recorders must replace functions directly:
`MagicMock` call history can retain full-size tensor arguments and invalidate
memory measurements.

Grouped H3 Q/K normalization caps each input chunk at 64 MiB rather than the
unsplit path's 256 MiB default. PyTorch 2.7 can materialize a FP32 normalization
temporary twice the BF16 input size; the larger chunk exhausted a physical
12-GB card during the first native-size attention block. The smaller chunk
preserves per-head RMSNorm and RoPE math. A complete saved-video run on the
physical machine remains the acceptance check; CPU parity and component
memory checks alone do not establish end-to-end compatibility.

For long compatible grouped H3 routes, the output projection processes token
chunks and compacts them into a prefix of the owned contiguous attention
result. When output width does not exceed input width, each completed chunk
ends before the next unread input row. This avoids a second full-sequence
output allocation; the returned view is contiguous and retains the original
storage until its residual consumer releases it. Autograd, autocast, active
LoRAs and incompatible storage/projection layouts retain the ordinary path.

## Verified physical native follow-up

On October 9, WanGP 17.17 H3 Pruned at 20 steps completed native 1920x1088,
362 frames at 24 fps on the 12-GB / 32-GB machine in 12,893.633 seconds. Maestro
Dev `11ce7e1` separately completed Fused References at six steps in 1,924.016
seconds, with one image, an uploaded soundtrack and live Tiny VAE preview. Both
saved outputs passed complete audio/video decoding. These are capability checks
with different models, attention and conditioning, not matched speed results.

To reproduce the tested Maestro placement, use Settings > System > RAM / VRAM
Management: MMGP optimized VRAM allocator (restart if needed), PyTorch default
RAM allocator, Smart Memory Pinning on, Read Ahead off, and Attention Head Split
Medium. Set Video Profile 4 in the advanced Performance controls and select the
Q2_K H3 Text Encoder in generation settings. Use SLA, six steps, no active LoRAs
and no upsampling. Lock a single approximately 15.08-second window with Allow
30s clips enabled. This manual override switches away from ordinary Auto limits.

Peak API-sampled VRAM was 11.75 GiB. RAM was nearly exhausted and paging occurred;
free host memory and the Windows pagefile mattered. The ordinary 14.4-second
recommendation has not been widened. See the [validation record](../VALIDATION_V2.7.0.md)
for exact geometry, timing boundaries, runtime, independent host metrics and
failed controls. The earlier 12-GiB-capped component probe remains separate from
this physical-machine saved-video proof.
