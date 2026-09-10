# Maestro v2.1.5 release validation

Validated on 10 September 2026. See the [release notes](RELEASE_NOTES_V2.1.5.md).

## Release boundary

Public `dev` and `main` both pointed to
`cd48033f3723c2d57c5ca81b0964bc8e502257f5` before this release. That commit had
already published the bounded Viggle control-frame conversion after v2.1.4.
This release adds H3 prompt adaptation, LLM lifecycle, INT8 dispatch, shared
audio assembly, Director duration and automatic-performance revision fixes.
The application version comes from the root `VERSION` file.

No launcher, dependency manifest, model weight, private character, local
configuration or generated content-guide changes are included. Local logs,
benchmark inputs and generated media remain outside the published source.

## Combined automated checks

The release checks used Windows, Python 3.11 and the installed PyTorch 2.10 /
CUDA 13.0 environment. CUDA was hidden for the combined suite so it could run
without disturbing an active generation.

| Check | Result |
| --- | --- |
| Full Python discovery | 1,567 tests in 89.814 seconds; passed, with two CUDA-only skips |
| Standalone JSON grammar checks | All five passed |
| Python compilation | Services, launch, H3, shared audio/INT8 modules, generation entry point and scripts passed |
| Undefined-name checks | F821/F823 passed for services, launch, H3 and changed shared modules |
| UI type check and production build | `tsc -b && vite build` passed; 1,858 modules transformed |
| Public-repository boundary guard | Passed across 2,213 staged/tracked files |
| Release version and documentation | App version reader returned `2.1.5`; all 55 checked local links resolved |
| Staged whitespace check | Passed |

The skipped tests are the CUDA temporal-convolution comparison and an explicitly
opted-in INT8 dispatch/retry check. The latter was exercised separately while
the GPU was idle: all 16 dispatch tests passed. A separate 119-test H3 regression
run also passed before the combined release run.

New or expanded coverage includes:

- Timed and flattened authored briefs, two-character profiles, shared production
  notes, silence versus actual speech, dialogue ownership and exact-line checks.
- Source-event/camera assignment, phase order and timing, physical ending and
  window handoff, and preservation of complete action/audio fields.
- Concurrent LLM use and stale idle-unload timers.
- Director multi-window frame validation while retaining independent-shot limits.
- Stereo/mono prefixes, silent gaps, joins and metadata using real FFmpeg output,
  plus CPU audio preparation under a different default tensor device.
- Automatic-setting revisions, manual overrides, failed detection and memory
  failure classification.
- INT8 dtype preservation, bounded kernel selection, cache behavior, memory
  checks and fallback/error handling.

The UI build retains its existing nonfatal mixed static/dynamic import and
large-bundle warnings. No UI dependency or chunking change is part of this patch.

## Real AI Faithful evaluations

Two installed local models were evaluated through Maestro's production planning
and LLM service using the same long, silent martial-arts brief: Gemma 4 E4B
Q4_K_M and Qwen3.8 27B Q4_K_M. The supplied 30-second script was adapted to two
windows totaling 28 seconds. These were text-only checks: no reference images
were attached and no video generation was submitted by the evaluation.

| Result | Gemma | Qwen |
| --- | ---: | ---: |
| Authored action phases retained | 8 | 8 |
| Principal characters | 2 | 2 |
| Shots per window | 4 / 4 | 4 / 4 |
| LLM writing calls | 3 | 3 |
| Repairs, deterministic fallbacks or review warnings | 0 | 0 |
| Final window prompt tokens | 1,462 / 1,711 | 1,471 / 1,646 |

Reading the resulting action and camera text confirmed the eight main fight
phases, their order, the handoff and final collision. This does not establish
detail-perfect fidelity or rendered-video quality. Qwen inferred hairstyles
that were not specified and omitted a secondary cloud effect; Gemma condensed
one cliff-impact transition. Users should still review appearance, secondary
effects, spatial relationships and timing in the exact H3 prompts.

## Real Viggle checkpoint layer measurements

An isolated RTX 4090 test used Maestro's MMGP loader and the installed
`Viggle-Animate-pruned_rank8_int8_convrot.safetensors` checkpoint. Inputs were
BF16 and stored scales FP32. The test applied ConvRot activation rotation and
timed three warmed repetitions with CUDA events, taking their median.

Here M is the activation row count, K the input width and N the output width.
The Q projection uses the first 7,168 rows of the loaded QKV weight.

| Layer | M / K / N | Old FP32 fallback | Corrected BF16 fallback | Triton | Automatic |
| --- | --- | ---: | ---: | ---: | ---: |
| Q projection | 23,296 / 5,376 / 7,168 | 39.25 ms | 12.84 ms | 10.00 ms | 9.93 ms |
| MLP first projection | 8,192 / 5,376 / 28,672 | 53.61 ms | 17.03 ms | 13.48 ms | 13.60 ms |
| MLP second projection | 8,192 / 14,336 / 5,376 | 28.26 ms | 9.37 ms | 7.02 ms | 6.99 ms |

The corrected fallback was about 3.0–3.1 times faster than the previous fallback.
Triton still won on this GPU and automatic selection retained it in all three
cases. Small timing differences between forced and automatic Triton are normal
measurement variation.

Relative RMS error against the old FP32 reference on sampled output rows was
0.0031–0.0036 for corrected BF16 and approximately 0.0062 for Triton. Lower
precision is not bit-identical to FP32; this test verifies the intended dtype
and numerical behavior, not video-quality equivalence.

These layer measurements exclude LoRA work, attention, model-weight streaming,
window preparation and decoding. They do not establish a full-generation
speedup or predict performance on an RTX A4500.

## Remaining limits

The reported RTX A4500 / 28 GB RAM environment was unavailable locally. A
matched full-window test is still needed to establish its remaining speed and
memory behavior. Automatic profiles below 32 GB RAM remain conservative; this
release does not promote those machines to Profile 4.

The shared audio tests establish channel preservation and CPU allocation
behavior, not that every source of an out-of-memory failure has been removed.
Existing exported mono media cannot recover a discarded stereo channel.

This patch does not include a fresh Pinokio installation test or a complete
end-to-end generation benchmark. GitHub CI runs the clean-repository guard,
Python checks and UI build on both publication branches.
