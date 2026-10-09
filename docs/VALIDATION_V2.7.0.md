# Maestro v2.7.0 validation

Release candidate prepared October 8, 2026, with an H3 VAE 2× follow-up on
October 9. Public v2.6.0 Main baseline:
`31d7396eb3b5586c1828fc69a6af5eb2a9e83bff`. The initial reviewed memory and
Director implementation is `8831f34809f003f446167c54e47cdfb0507fc756`.
The H3 decoder follow-up is `76dcc51aea5fd5983674bb60f608353001c8fc2f`.
Dev includes bounded SSH diagnostics and release metadata/documentation.
Main has not been updated.

## Automated checks

[Dev CI run 37861066113](https://github.com/Blizaine/Maestro/actions/runs/37861066113)
passed for `8831f34`: clean-repo guard, Python syntax and undefined-name checks,
full backend unittest discovery (3,490 tests, 55 skipped), JSON grammar and
standalone preview/source-camera/media checks (137 passed, one skipped), plus
frontend lint, TypeScript and production build. That result
belongs to the committed implementation snapshot; it does not cover the later
local SSH diagnostics or release documents.

The Director preview follow-up passed focused ownership/lifecycle/preview
regressions and an isolated browser integration suite at desktop and mobile
widths. Coverage includes discovery without refresh, clip transitions, ownership
filtering, startup races, detached repairs, accepted retries, whole-project Stop,
exact prompt disclosure, image previews and finishing-phase tile deduplication.
Browser fixtures do not establish physical iPhone playback compatibility.

Local follow-up checks passed: six SSH diagnostic tests, Python syntax and
undefined-name checks for both SSH files, and the backend's real VERSION reader
returned 2.7.0 without importing the generation runtime. All 133 local file
references across the four release Markdown documents resolve. The private
Director receipt is Git-ignored. Staged diff whitespace passed, and the source
boundary guard passed on all 2,839 tracked/staged files, including the new release
documents. No model weights, generated media, credentials or private reports are
included in the snapshot.

### H3 VAE 2× follow-up

[Dev CI run 37880485581](https://github.com/Blizaine/Maestro/actions/runs/37880485581)
passed for `76dcc51`: clean-repo guard, Python syntax and undefined names,
3,505 backend tests (55 skipped), standalone regressions (137 passed, one
skipped), JSON grammar checks, frontend lint, TypeScript and production build.
Focused local x2 tests also verify decoder-only weight replacement, native
ConvRot metadata, absence of meta tensors after loading, exact 345-frame
streamed decoding parity, preserved input buffers, cancellation, output
geometry and model reloads when switching the decoder. The isolated browser
suite checks Studio submission, active Director model capabilities and clearing
unsupported image/audio/non-H3 selections.

The optional checkpoint revision, file size and SHA-256 are pinned. Its downloaded
2,834,844,787-byte INT8 ConvRot asset was verified on the test machine before
loading. Both normal and x2 paths blend in the native decoder grid and convert
finalized temporal chunks to CPU bytes. No model weights are committed.

## Measured generation evidence

### 24 GB VRAM / 128 GB RAM workstation

The October 7 memory bench completed 21 H3 generations, including a longer
reference-and-LoRA confirmation. It compared INT8 backends, streaming profiles,
preload modes, pinning, read ahead, head splitting and Sol. Component CUDA
probes and small Flux/Qwen VAE parity checks are documented in the
[port review](development/wan2gp-17-port.md) and
[memory benchmark](development/memory-benchmark.md). These are not full
end-to-end verification of every affected model family.

### 12 GB RTX 3080 Ti / 32 GB RAM Windows machine

Existing private reports in the Test-Bench workspace record these results:

- Matched H3 Pruned Profile 5 repeat: 433.969 seconds on public Main versus
  207.594 seconds on an earlier optimized Dev snapshot (2.09x throughput for
  that workload). Final manual Profile 4 repeat was 184.437 seconds; final
  Auto-tune 480p runs were 210.390–211.641 seconds with tighter RAM budgets.
- Auto-tune reduced sampled peak system commit from 36.26 GiB in the final
  manual repeat to 26.16 GiB by automatic repeat 3. These are system-wide
  sampled values, not a cap on process memory or a 16-GB compatibility claim.
- Warm Wan2GP 480p repeat was 167.484 seconds; final manual Dev took about
  10% longer. At 720p, Auto-tune Dev was 552.922 seconds versus Wan2GP's
  486.656 seconds (13.6% more elapsed time). Applications used different
  supported Torch/Python runtimes and preview settings, so this is not an
  isolated source-only comparison.
- Changing only the optional RAM allocator improved repeat elapsed time by
  less than 1% in these measured cases. The RAM allocator remains PyTorch
  by default; do not advertise it as a universal speed improvement.
- All 24 successful matched clips in the H3 comparison passed full decode.
  Reference-sequence tests independently validated joined outputs and confirmed
  restoration of the prior settings and Auto flag.
- Omni Pruned Director tests rendered only a 10.125-second song excerpt.
  Measured wall times were 14m03s at 540p/Q2, 24m16s at 720p/Q2 and 19m02s
  at 720p/cached NVFP4. These differ in shot layout and setup; they do not
  establish a full 4m15s render time or isolate encoder-only speed gains.

The matched methodology and limits are in the
[cross-app benchmark guide](development/h3-cross-app-benchmark.md).
Private reports, prompts, music, media, hardware identifiers and SSH material
remain excluded from source control.

### H3 decoder follow-up on the 12-GB machine

Six single-window GPU runs completed: a matched Fused Frames 540p / 124-frame
six-step normal/x2 comparison with one cold and one warm run per decoder, plus
Fused References six-step / 345-frame tests with an image and exact music.
The four short tests used `68a88de`; the two long tests used `76dcc51`.
All six saved videos passed complete FFmpeg audio/video decoding and actual
frame-count/dimension checks.

The matched warm normal run took 65.453 seconds and x2 took 67.891 seconds
(3.7% more elapsed time in this one repeat). Outputs were respectively 960×544
and 1920×1088, with 124 frames at 24 fps. API phase timing includes final file
writing; this is not an isolated CUDA decoder measurement.

The 14.375-second normal 720p reference test completed in 507.828 seconds,
with 11.88 GiB sampled peak device memory. The 540p x2 reference test produced
1920×1088 in 260.735 seconds with 8.94 GiB sampled peak device memory. Both
contain 345 frames and retain the uploaded stereo soundtrack; aligned decoded
PCM correlation with the source excerpt was 0.9991. These different base
resolutions are not a matched quality or source-only speed comparison.

Before updating, the user's older-Dev 720p job completed denoising but failed
at whole-video float normalization while allocating another 3.47 GiB. The new
bounded output path removes that allocation pattern; the follow-up 720p test
used a different prompt with the same duration, resolution, model/steps and
image/music reference workload. The user's queued native 1080p job separately
failed during transformer packed-embedding assembly (2.09 GiB allocation).
Native 1080p denoising at that duration remains unverified; x2 decoding from
540p must not be presented as equivalent evidence.

These fixed-control benchmarks temporarily disabled Auto and restored its
original value and system settings afterward. System-RAM pressure remains:
three-second host samples reached 39.53 GiB whole-system commit in the long
720p case and 36.57 GiB in the long x2 case, with paging during checkpoint
loading. Peaks can be missed between samples. These runs do not establish
16-GB RAM compatibility or stability with other reference/LoRA combinations.
Private evidence and sampled frames remain in the Test-Bench workspace and
`.codex-tmp/h3-x2-20261008/`, excluded from source control.

### Full Director project

The user's 23-clip H3 Fused References project on the 12-GB machine completed
on remote Dev `6d7c5b9` and saved all clips plus a combined video. Settings were
720p, six steps, Q2_K and approximately 10.1 seconds maximum per generated clip.
All 24 files passed complete FFmpeg audio/video decoding with no reported error.
The combined video contains 5,484 H.264 frames at 1280x704 / 24 fps, lasts
228.500 seconds and has 48-kHz stereo AAC lasting 228.501 seconds. The source
song lasts 228.501333 seconds; final video timing is within one frame. Summed
per-clip planning targets include one additional frame, which the final
source-audio-duration trim removes.

Recorded video-job elapsed time was 10,317 seconds (2h51m57s); summed per-window
generation time was 10,258 seconds, with individual windows between 415 and
456 seconds. This is a measured 3m48.5s project, not a 4m15s estimate, and the
receipt does not establish continuous peak-memory stability or creative quality.
Technical evidence is private at `.codex-tmp/v270/director-full-validation.json`.
This run predates the final Director preview follow-up; it cannot validate that
UI fix in the running remote application. Human review of visual continuity,
lip sync, sound and editing remains separate from metadata and decode checks.

## Remaining validation limits

- 6/8/10 GB VRAM and 16 GB RAM recommendations are conservative extrapolations,
  not measured generation compatibility. The pure recommendation function
  selects Profile 4.5 for 32-GB RAM / 6–10-GB VRAM and Profile 5 for 16-GB RAM.
  Low-memory H3 uses Q2_K, smaller residency and workload guards; these cannot
  guarantee that all checkpoints, reference sets or window lengths fit.
- Linux CPU CI does not verify Linux CUDA model startup, allocator operation
  or end-to-end generation. A Linux CUDA smoke remains recommended.
- Allocator startup/release probes did not force an actual system-RAM spill;
  RAM-spill performance and recovery under genuine overflow remain unverified.
- Representative non-H3 end-to-end video/image/audio generations, a fresh
  installation and a normal public-v2.6.0-to-candidate Update/restart check
  remain recommended before promotion to Main.
- Physical iPhone testing of Director preview appearance without refreshing
  remains pending after updating/restarting the idle test machine.

## Packaging and update behavior

The product version is stored in the repository-root VERSION file. Pinokio's
script schema version and the frontend package's private tooling version are
independent and remain unchanged. The backend reads VERSION at startup.

The application loads reviewed MMGP 4 from `app/mmgp`; requirements explicitly
pin its runtime dependencies instead of fetching MMGP 3.8.2 again. Existing
saved manual controls retain ownership. Re-detect explicitly applies Auto-tune
recommendations; memory allocator changes require a process restart. Performance
learning and test reports remain local, bounded and optional.

Attribution, upstream license copies and native allocator provenance are in
[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md),
[MMGP provenance](../app/mmgp/PROVENANCE.json) and the
[port review](development/wan2gp-17-port.md). The optional H3 VAE 2× integration
is documented separately in the [decoder guide](H3-VAE-x2.md). The LTX 2.5 detail refiner and other unlisted
WanGP 17 workflows or model packages are not included.
