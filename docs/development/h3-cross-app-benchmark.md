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

## Optional RAM allocator

Settings > System > RAM / VRAM Management now includes RAM Allocator. MMGP optimized reuses freed CPU tensor blocks, with a bounded cache that is returned under RAM pressure and when a Studio job or model release finishes. It does not reduce the size of live model weights. The default remains PyTorch until hardware-specific evidence justifies a change. Changing it requires a restart; active status and a fallback reason are visible in the UI/API. `--ram-allocator` overrides the saved choice.

The allocator source and Windows/Linux binaries come from the pinned WanGP 17.17 commit recorded in `app/mmgp/PROVENANCE.json`. The main MMGP engine remains at its separately recorded version. Native support is checked before installation; unsupported systems retain the PyTorch allocator. Linux GPU performance requires its own measurements.

## Confirming Auto-tune

The controlled Maestro matrix deliberately disables Auto-tune while comparing explicit settings. Confirm automatic placement separately: apply Auto-tune while idle, keep it enabled, omit `override_profile` and manual window-memory overrides, and inspect the completed job's `performance_plan`. Record the final transformer budget, reserved RAM fraction, profile and read-ahead settings rather than inferring placement from an earlier VRAM-guard candidate log.

A matching final placement can reuse a resident model. A real change in available RAM, activation allowance, profile or explicit preload may still require a reload. Model release and allocator changes also invalidate a warm comparison. Retain both first and repeat timings and explain any internal reload.