# Maestro v2.7.0 validation

Release candidate prepared October 8, 2026, with H3 follow-ups on October 9
and final local preparation on October 10. Public v2.6.0 Main baseline:
`31d7396eb3b5586c1828fc69a6af5eb2a9e83bff`. The initial reviewed memory and
Director implementation is `8831f34809f003f446167c54e47cdfb0507fc756`.
The H3 decoder follow-up is `76dcc51aea5fd5983674bb60f608353001c8fc2f`.
The October 10 candidate also includes the accumulated local Auto, Director,
model-picker, H3 decoder resolver and Qwen Turbo changes. Main has not been
updated; no push, tag or release publication is part of this preparation.

The October 12 follow-up is prepared for public Dev testing so existing Dev
installations can receive the local Director and dialogue changes through the
normal Pinokio Update action. This promotion targets Dev only; Main and release
tags remain unchanged. The planning and media limitations below still apply.

## Automated checks

The October 13 gallery restore follow-up passed mocked browser checks for
individual Director clips (including index zero), joined-project routing,
archived media URLs, exact prompt/seed/attention restoration, enhancement
ownership and errors that leave the active settings unchanged. Existing Editor
draft-persistence checks, frontend lint, TypeScript and the production build
also passed. Isolated checks against four saved outputs covered an original
first clip, an original seventh clip, its later rerun and the joined video.
Individual recipes retained their own prompts, seeds, six-step Veda selection,
345 native frames and 14.375-second soundtrack slices; the joined recipe retained
the full run. Saved pipeline and sidecar hashes were unchanged. These checks
did not submit a new generation or establish an attention-quality comparison.
The final combined Director/H3/gallery/settings/lifecycle suite passed 845
tests. A concurrent soundtrack-cache check passed 80 repeated runs after
serializing cache publication for Windows file-sharing behavior.
After an idle restart, the live metadata API returned the expected scoped
recipes for those four outputs, and their archived reference/soundtrack URLs
were reachable. The server also served the current frontend bundle.

Expanded H3 identity regressions passed 102 focused tests. An isolated final
compilation of all 16 saved music-video plans produced two uniquely mapped
subjects per clip, with no duplicate third subject. The historical prompts
remain available unchanged for matched attention tests. This verifies the
compiler repair, not a retroactive change to existing rendered video.

The October 13 Editor rerun follow-up passed 783 focused Director/H3 tests and
20 additional Editor project tests. An isolated replay of a saved shot and two
edited-prompt cases passed both production preflights with unique image-subject
bindings; submission was intercepted before GPU enqueue and existing media
was retained. A mocked browser integration test covered draft persistence,
failed reruns, blank-input rejection, duplicate-submit prevention and edits
made to multiple shots while a rerun/save was pending. Frontend lint,
TypeScript and the production build passed. These checks establish prompt
compilation and Editor behavior, not the visual quality of a new render.

The October 12 Dev preflight identified an avoidable recurrence-parser hot path
in the one-hour Studio planning fixture. Checking recurrence eligibility before
parsing connective prose reduced that fixture from a timed-out run to 2.935
seconds on the local machine. All 37 long-form planning tests and 19 recurrence
tests passed after the change. These are CPU planning fixtures, not video-render
timings.

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
Native 1080p denoising was unverified in that earlier test; x2 decoding from
540p must not be presented as equivalent evidence. See the separate native
follow-up below.

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

## Native H3 follow-up — October 9

A separate physical-machine check confirmed upstream WanGP 17.17 can save a
native 1920×1088, 362-frame / 24-fps video on the RTX 3080 Ti 12GB / 32GB RAM
machine. The pinned upstream revision is
`6479db36bdc2619a904a852bba9c2d78e1a83f82`. Its H3 Pruned run used 20 steps,
Q2_K, Sage2, Medium head splitting, Profile 4, MMGP VRAM spill/RAM allocators,
smart pinning and read ahead. There were no LoRAs or upsampling passes.

Measured wall time was 12,893.633 seconds (3h34m54s), with 12,507.038 seconds
in denoising. Three-second host samples reached 9.943 GiB whole-device VRAM
and 36.547 GiB system commit; available system RAM fell to 0.219 GiB and paging
occurred. Full FFmpeg audio/video decoding passed, and six evenly spaced
frames showed a coherent bird and landscape. Required saved model, geometry,
seed, steps, attention, profile and upsampling fields matched; upstream did
not report the empty enhancement selector. This verifies one Windows recipe,
not fast generation or universal compatibility on 12GB cards or 16GB RAM.

Maestro Dev `11ce7e1` then completed its separate Fused References six-step
case with an image reference, exact soundtrack and live Tiny VAE video preview.
The saved video contains 362 native 1920×1088 frames at 24 fps (15.083333 seconds),
with no spatial or temporal upsampling. Full FFmpeg audio/video decoding passed;
six evenly spaced frames show a coherent subject and scene without noise
collapse. Aligned source/output audio correlation was 0.99909 after AAC encoding.
This sampled visual check does not establish every frame's creative quality.

Total API wall time was 1,924.016 seconds (32m04s). Poll-attributed denoising was
1,706.968 seconds and final decoding/save was 191.828 seconds; the generation
console's denoising timer reported 28m39s. These timers use different phase
boundaries. Two-second API samples peaked at 11.75 GiB whole-device VRAM and
31.68 GiB RAM used. Independent three-second host samples peaked at 11.547 GiB
VRAM and 38.980 GiB system commit against a 39.832-GiB commit limit; available
RAM fell to 0.091 GiB, with paging observed. Different sampling cadences explain
the observed VRAM peaks and can miss short spikes.

The tested manual recipe used H3 Fused 4-Step — References (Experimental),
custom six steps, SLA at 90% requested sparsity, Medium head splitting (eight
groups), Profile 4, the Q2_K H3 encoder, active MMGP optimized VRAM allocation,
PyTorch default RAM allocation, smart pinning on and read ahead off. The runtime
was PyTorch 2.7.1+cu128, using ComfyKitchen CUDA INT8 ConvRot dispatch. Its
221,791 packed rows included a 3,511-row protected media prefix. All 2,400 sparse
calls completed, with 88.4% effective block sparsity and no dense fall-throughs.

Earlier physical attempts failed at a large Q/K RMSNorm temporary and then a
second full attention-output projection buffer. Grouped normalization now uses
64-MiB input chunks, and compatible grouped output projections compact token
chunks into their owned attention storage. The failed controls remain failures;
the final saved-video run provides the end-to-end evidence for these fixes.
[Dev CI for the implementation](https://github.com/Blizaine/Maestro/actions/runs/37941355624)
passed 3,531 backend tests (56 skipped), 137 standalone regressions (one skipped),
source checks and the frontend checks/build. A small real CUDA Triton projection
check on the 24-GB workstation was bitwise equal with and without storage reuse;
it does not establish Kitchen kernel parity or identical full generated videos.

These recipes differ in model, step count, attention, conditioning and allocator
settings. Their times must not be treated as a matched cross-app speed test.
Native 1080p tests use a manually locked 15.083-second window above the ordinary
14.4-second recommendation. This does not make native 15-second 1080p an Auto
default. The later October 10 duration policy starts native 1080p shorter and
promotes only the separately measured hardware/resolution tiers. No active
LoRAs were tested in these native follow-ups.

## October 10 local candidate validation

The final local preparation starts from public Dev
`d870fe6a4f5af3d50ddb53be5cf008016396d30b`, which is 27 commits ahead of the
v2.6.0 Main baseline, and includes the intended local source and regressions.
The exact final commit and source ZIP checksum are recorded in the local release
manifest. This is local verification; it is not a new GitHub CI result.

The backend rerun explicitly sets `CUDA_VISIBLE_DEVICES=-1` on Windows and uses
isolated pytest tooling. An initial run without the test dependency and with an
empty CUDA setting was invalid for the intended CPU gate. Two stale fixtures
were also corrected: the quote-admission engine stub now exposes atomic recipe
preparation, and the H3 defaults assertion reflects the curated References
selection while still checking existing-install migrations. CI now uses the
explicit disabled-device value and includes Qwen 2.1 in syntax/undefined-name
checks. Product behavior was not relaxed to satisfy those fixtures.

The final Windows CPU backend discovery gate passes **3,630 tests, with 13
skipped**, using Python 3.10.20 and PyTorch 2.7.1+cu128 with CUDA explicitly
disabled. The final local frontend lint, TypeScript build and production Vite build pass.
The full responsive sidebar browser suite and eleven focused UI runners pass,
covering Auto advice, clip defaults, model visibility, Director setup, Qwen image
workflows and H3 x2. The standalone preview/source-camera/media gate passes all
138 tests with CUDA explicitly disabled. JSON grammar, Python compilation and
undefined-name checks pass. The final source boundary guard, metadata/link check
and archive verification are recorded in the local release manifest. All 146
local links across nine release/feature documents resolve, all five changed JSON
assets parse, and the source boundary guard passes on 2,873 tracked files. The
14 bundled MMGP allocator/provenance files match their recorded SHA-256 values.

The resource policy is shared as versioned JSON between the backend and React
UI. Tests cover RAM/VRAM/resolution bounds, excluded checkpoints, reference
margins, late hardware detection, manual ownership, effective head splitting
and workload fingerprints. Director and Studio use the same measured tiers;
manual clip durations and cut speed remain in control. Long native-1080p Pruned
runs that take hours are treated as impractical evidence, not successful speed
benchmarks. LoRAs and additional reference rows retain uncertainty warnings.

Browser regressions cover contextual model names, nested workflow groups,
independent visibility, persistent manual hiding, reset without deleting models,
new Director defaults, scroll behavior after audio analysis, duration defaults
and the Qwen Turbo recipe. They use isolated fixtures rather than submitting
real jobs to the running application. The full sidebar suite also checks that a
reviewed H3 plan uses first-window and later-window spans consistent with the
backend overlap/discard contract before submission.

Qwen Turbo used real INT8 text-to-image and single-reference-edit runs on the
24-GB workstation at 1024×1024. The downloaded checkpoint's SHA-256 matched
`2f70ddf24247932852b8477744fce33af505a4aa462acb77b5dfe903c6f22069`.
Both saved outputs use eight steps, CFG 1 and the Turbo solver. The edit changed
the requested object's color while retaining the scene and sign in the sampled
review. The first run included its checkpoint download; elapsed times are not
advertised as a matched benchmark. BF16, 2K, mask/outpaint and stress tests with
many references or lower VRAM remain unverified for this Turbo integration.

The H3 x2 resolver regressions cover flattened and nested Hugging Face layouts,
shared roots, missing/corrupt files, and verified alternate copies. Pinned size
and SHA checks remain required before loading. Fused label cleanup does not
change the bundled checkpoint files or their recipes.

## Director prompt-contract follow-up — October 10

The final focused gates pass 292 Director tests and 80 H3 dialogue tests.
New cases cover anonymous visual cast/reference matching, ambiguous identities,
name boundaries, reviewed cast recovery, duplicate picture ownership, undefined
subjects, clip-local shot numbering and timestamps, exact performance-direction/world-anchor
recompilation, and preserving rendered-prompt provenance when editing a saved
clip. H3 planning's JSON schema now requires identity keys; its semantic check
requires a meaningful name or stable semantic ID. Only affected planning rows
enter the existing bounded repair path. The planner checkpoint revision was
advanced so an older cached plan cannot bypass these checks.

The final full backend gate passes **3,660 tests, with 13 skipped**, on the
Windows CPU test runtime. Python AST/control-character and undefined-name checks pass for
all changed Python files; JSON-grammar regressions pass. No frontend code or
launcher scripts changed in this follow-up; the preceding frontend gates apply
to their unchanged source snapshot.

All 23 reviewed prompts in the saved Director project pass the final compiler
contract with their actual clip durations. Corrections preserve the existing
media filenames, source-song timing and waveform, and backup prompts. The
remote update verified that all 30 existing media files retained their sizes
and modification times, started zero renders, and marked the 23 changed clips
as requiring regeneration. Private backups, per-shot prompts and the review
report remain excluded from Git.

These checks validate prompt construction and saved-project integrity. They
are model-free: no new writer call or video generation was run for this
follow-up, and the existing videos do not demonstrate the corrected prompts.
Human review after regeneration remains necessary for anatomy, scene
continuity, point of view and identity retention.

## Director transition-numbering regression — October 10

A new Gemma E4B music-video plan failed H3 preflight on clip 3. Its first six
planner rows ended in empty `<scenetrans>` shells tagged with global clip
numbers. Replaying the saved 23-clip plan reproduced numbering errors in clips
3–6; clip 2 also contained an unintended empty second cut. The compiler now
removes empty shells and localizes supported timed visual cut labels while
preserving exact dialogue and cut action/timestamps. The planner guide states
that numbering restarts inside every clip. Direct malformed payloads still
undergo the substantive timing and shot-order checks.

All 23 saved clips pass strict compilation, including reference images and
driving-audio mappings. Source prompts and visual shot labels remain stable
through recompilation. The 292 Director tests, 147 H3 dialogue tests and 7 new
numbering regressions pass. A pre-existing second-pass cast-alias normalization
in one saved clip was reproduced on the preceding build and recorded separately.
No new LLM calls or video renders were run; these checks do not establish
rendered motion quality.

## Overnight workstation review — October 10–11

This manual batch used the RTX 4090 workstation with 24 GB VRAM and 128 GB RAM.
The active Director run on the 12-GB workstation was left untouched. No public
branch, tag or release was written. Raw writer attempts, failed controls,
rendered diagnostics and local reports remain excluded from Git.

The baseline source was frozen at `6de3eff` until its GPU tests ended. Reviewed
speaker, source-language, opening-state and native-timing fixes were integrated
at `0687a1e`; a further bounded speaker-parser fix was integrated at `609e1ad`.
The final full backend CPU gate at **`609e1ad` passed 3,813 tests, with 12
skipped**, across 312 test files. It used the managed Python 3.11.13 / PyTorch
2.10 CUDA-13.0 runtime with `CUDA_VISIBLE_DEVICES=-1`; CUDA/training-specific
skips are not generation-validation evidence. Source hashes and a clean
checkout were checked before and after the run. Frontend and launcher source
were unchanged by these overnight fixes; their earlier checks retain their
original source attribution. Subsequent documentation-only commits do not
represent another backend run.

The fixes preserve authored Director duration before native frame fitting,
retime local visual cues, and retain strict manual-edit validation through
save/resume/regeneration. Speaker regressions include real failed outputs,
possessive listener names, response cues, parenthesized voice IDs, ambiguous
speakers and wrong-owner negatives. Other checks distinguish an open-palm
adjective from an opening action and keep continuation instructions out of
cast names. Explicit spatial setup now takes precedence over identity/pose
descriptions for short-film opening states. These are bounded corrections,
not a complete semantic story validator.

### Local writer results

Five prepare-only cases covered exact dialogue, creative dialogue, two-window
prop/knowledge continuity, rider/reference continuity and Director planning,
using installed Qwen3.8 27B and Gemma E4B writers. Every captured response and
final native prompt was reviewed. The original ten attempts had nine completed
and one failed; eight completed attempts used fallback. The matched ten-case
retest at `0687a1e` completed all ten, with seven using fallback in at least one
window. Completion did not remove causal-order, silence or prop-state defects.

Four fresh follow-up attempts at `609e1ad` produced:

| Case | Writer | Result | Calls | Fallback | Elapsed |
| --- | --- | --- | ---: | --- | ---: |
| Exact supplied dialogue | Qwen3.8 27B | Completed, correct owners/words | 1 | No | 16.2 s |
| Exact supplied dialogue | Gemma E4B | Completed, correct owners/words | 1 | No | 8.7 s |
| Creative group exchange | Qwen3.8 27B | Failed; five-word Ron-only draft | 2 | No final prompt | 11.7 s |
| Creative group exchange | Gemma E4B | Completed through source fallback | 7 | Yes | 23.2 s |

The exact-dialogue improvements agree with replay of previously rejected raw
outputs. These are single attempts across different source snapshots, not a
statistical latency or general writer-quality claim. Gemma's creative dialogue
JSON had correct owners; its frame drafts first omitted dialogue tags, then
disagreed with the deterministic owner resolver. Qwen's focused repair addressed
word count without restoring the missing speakers. An additional clean-reference-
caption diagnostic still used fallback and invented music. None of these
prepare-only checks establishes voice, lip-sync or generated-motion quality.

### Speed and installed-model checks

The controlled baseline used `6de3eff`, seed 424242, explicit settings with Auto
temporarily disabled, and sequential cold/warm pairs. Cold means model-unloaded,
with filesystem and compiled-kernel caches retained. The isolated INT8 comparison
kept Profile 1, default preload and head split Off; Auto resolved to Comfy Kitchen
CUDA. Warm results for the same 1280×704, 243-frame workloads were:

| Model/workload | INT8 policy | Wall | Denoising | Peak VRAM |
| --- | --- | ---: | ---: | ---: |
| Pruned, 20 steps | Triton | 648.9 s | 599.1 s | 21.77 GiB |
| Pruned, 20 steps | Auto | 483.3 s | 442.5 s | 21.43 GiB |
| Fused, 6 steps | Triton | 185.6 s | 135.9 s | 19.12 GiB |
| Fused, 6 steps | Auto | 131.2 s | 92.8 s | 19.11 GiB |

Measured denoising was approximately 26% shorter for Pruned and 32% shorter for
Fused. One pair per setting, cache/order/restart effects and variable cold loading
limit inference; Fused's Auto cold wall time was slightly longer. All compared
files passed full decoding and sampled-frame review, which does not establish
pixel-identical output or gains on other hardware.

A separate Profile-2/dynamic-preload/INT8-Auto Pruned diagnostic stopped advancing
after five denoising steps and was cancelled after about ten minutes. Those three
changes were bundled, so its cause remains unisolated. It is a failed performance
test, not an OOM or a successful benchmark. API cancellation preceded actual
engine teardown; a main-workstation-only restart restored an idle runtime.

A corrected native-1080p Fused test explicitly locked **345 frames / 14.375 s at
1920×1088**, six steps, Profile 2, fixed 8192-MB preload, Low head split and INT8
Auto. It passed full video/audio decoding in **609.7 s**, with **15.28 GiB** sampled
peak VRAM. This is an advanced 24-GB capacity test above Auto's recommendation,
not a 12-GB result or a validated-safe default.

Installed compatibility smokes also passed full decoding: LTX-2.5 Distilled video
with audio, Flux JPEG generation, Qwen Image 2.1 Turbo PNG generation, ACE-Step
12-second stereo audio, and learned H3 x2 decoding from 960×544 to 1920×1088.
The x2 decoder's pinned size and SHA checks passed. The first Flux request omitted
image mode and produced a single-frame MP4; the corrected image request is
separate. LTX returned the requested 49-frame output but normalized its internal
window to 58, so it is compatibility evidence rather than an exact-workload timing
pass. An initial ACE cold-load polling timeout was retained before its successful
retry. Flux fetched a small preview decoder; this batch did not download another
large compatibility checkpoint.

### Director media and unresolved quality findings

A fresh baseline short film (four clips, 40.5 s) and a 30.17-second diagnostic
segment of a fresh 23-clip full-song plan were rendered. All clips and joins passed
full video/audio decoding. All 23 song prompts were reviewed; unique wording still
allowed repeated frontal-flight motifs. Sampled music-segment frames showed the
Kitten in front of the Dragon without the requested mounted support. Source-song
alignment had zero best offset and 0.995 PCM correlation in the checked segment;
this does not establish lip-sync or perceptual audio quality. The LLM→video→LLM
handoff completed without a device failure, while its writer still used fallback.

A fresh five-clip diagnostic at `609e1ad` rendered in 834.2 s. Its native schedule
rounded the requested 972 frames / 40.5 s to **977 frames / 40.708 s**. The strict
exact-request check rejected that difference. A separately acknowledged diagnostic
check verified all five clips and the join against the actual native schedule;
it did not relabel the result as an exact-duration pass. All 12 reviewed source
hashes and the clean checkout remained frozen through rendering and verification.

Review of every native prompt, one-second samples and all exact first/last frames
found material defects despite successful generation:

- Compression invented `KITTEN (Implied)` in an action-only screenplay. Planning
  propagated it into empty-text dialogue beats and literal speech tags; compilation
  preserved `<d>[English] Implied</d>` despite the empty dialogue manifest. Music
  instructions also survived the explicit no-music brief, sometimes inside visual
  prose while the official music field said N/A. These are prompt findings; no
  perceptual audio review was performed.
- Canonical opening states coexisted with contradictory model prose. The continuity
  helper overwrote a flight clip's ending with the next grounded opening using a
  generic movement instruction, without establishing a landing action. Exact output
  boundaries showed a hanging-to-seated boarding jump and a grounded-to-airborne
  reset around a repeated landing. Dismount samples preceded folded wings.
- The authored sheathed-sword/left-hip state disappeared during compression and
  remained absent from all five final prompts. Strap and snout contact were present,
  and supported riding was visible in this sample; that does not resolve the
  separate music-video relation-state failure or guarantee all motion/identity.

These failures remain open. The overnight assessment recommends holding Main
promotion for authoritative silence/music handling, source-event/prop retention,
and causal clip-boundary validation. A passing CPU suite and decodable media do
not establish that those story contracts are satisfied. Saved performance settings
were restored, Auto remained enabled, resident models were released, and the main
application was left running and idle.

## October 11 remediation follow-up (local, not published)

The candidate now carries original speech/music permissions and source events
outside screenplay compression, preserves explicit prop details and authored
endings, and validates source allocation and declared physical continuity before
queuing video. Save/resume and clip regeneration preserve those contracts.
Source-owned action prose cannot append a model-invented transition action.
The silent H3 repair now requests one initial fact set and ordered state changes;
Maestro derives carried openings, transition from-values and closings in code.
Shot count, timing, camera choices and other authored fields stay fixed during
that repair. Source and physical validators still run afterward.

The v12 candidate recorded **3,978 passing CPU tests and 12 skips**, with unchanged
source throughout the 495.39-second run. Its source digest is
`6f92449ac574f4718172cbf424ce2636fbf3b05005d70f09695292ce342d8a02`.
Earlier failing runs remain retained. The two v11 speaker fixtures exposed a
real canonical-marker insertion defect; v12 corrects that defect while retaining
the ownership checks. Local name and pronoun cues must prove the source speaker
before Maestro inserts an adjacent canonical ID. Unknown names, a preceding name
used only as an object, competing owners, and changed dialogue still fail.
Captured-output replays establish parser behavior, not semantic acceptance.

The subsequent v13 source-owner guard passed **157 focused regressions** across
physical/source contracts, compact repairs, native serialization, music state
proposals and long-form planning. Explicit named support/contact relations bind
to their source actor and target; reversed support and unrelated actors do not
satisfy them. Single-event boarding requires its matching transition. Multiple
events require the actor/target in that actor's declared trajectory without
guessing which event owns a clip endpoint. Ambiguous actors and pronoun targets
remain outside this lexical check; it is not a general semantic proof. The full
CPU count above belongs to v12; v13 received these focused checks and lint, with
no new live writer or Director attempt. The restarted local runtime exactly
matches frozen v13 digest
`866786bf506758856a2c244fde53d867cf0f25079ebc4ba45b45ae437664d95c`.

Automatic duration-derived dialogue minima are writing targets for otherwise
complete exchanges; explicit spoken-word minima, speaker/topic/quote checks and
hard duration maxima remain enforced. An incomplete one-speaker exchange now
reports its missing speakers before the automatic word-count target.

All ten v11 writer attempts were reviewed, including their 44 effective calls,
five source fallbacks and one failed anonymous exchange. Missing spoken tags,
wrong speaker order and weak endings remain genuine quality findings. Two fresh
v12 creative-group attempts completed without fallback: Qwen3.8 27B used two
calls in 24.203 seconds; Gemma E4B used three in 11.515 seconds. Both final native
prompts preserve all three generated locked lines in order with correct adjacent
owners. These are single stochastic attempts, not a matched performance study;
the terse creative writing and speech timing still need rendered review. No
motion, voice, identity or lip-sync acceptance is claimed.

The 40.5-second v8 silent Director plan passed mechanical validation, but primary
text review rejected it before rendering: the typed states contained reciprocal
support, a misplaced lowering action and stale seated/airborne facts after
landing and dismounting. Earlier failed plans and this rejected plan remain
retained. Mechanical state consistency alone does not establish source fidelity.
The v8 full-song music plan failed before rendering because two independently
generated continuity markers disagreed. New compact H3 proposals now derive
physical continuity from their single cut strategy; legacy ledgers still undergo
strict validation. The v10 Director repair exposed a separate representation
gap: a sparse opening could not introduce a source-supported support/contact
relation at a later event. Its patch was rejected atomically, so the final stale
rows' missing-event error is not evidence that the repair omitted that event.
The next representation supports source-proven establishment of absent
support/contact/grounded facts at their exact event. It cannot overwrite a known
fact or guess an unknown starting value. Native serialization and JSON
round-trip checks retain that event provenance and do not print null states.
The music next-batch context now supplies only authoritative closing facts,
rather than replaying the preceding clip's old opening/actions; continuous clips
inherit those carried facts in code.

Fresh planning still failed before GPU submission. Gemma's v11 short-film
repair exceeded the eight-transition limit in its final clip. Qwen's v12 repair
included all source events but attempted a posture change without a known
starting state, so its patch was rejected atomically. The v11 full-song music
plan checkpointed two of 23 rows, then failed because a prop transition had no
declared current fact. All completed LLM passes and errors were retained. No
fresh Director or music plan was approved for rendering; **Main promotion remains
held for planning reliability and subsequent semantic/media review**. Passing
CPU checks and short enhancement attempts do not clear that gate.

Three matched six-step H3 Pruned diagnostics on the 24-GB workstation completed
full decoding. Profile 1/default, Profile 2/default and Profile 2/dynamic preload
took respectively 267.328, 205.125 and 261.687 seconds overall, with denoising
times of 139.265, 139.281 and 191.656 seconds. Dynamic preload was about 38%
slower during denoising and reached 23.50 GiB sampled device memory. There was
one cold reload per setting; cache/loading effects confound wall-time comparisons.
The earlier 20-step stall was not reproduced by these six-step diagnostics.
These observations do not justify new defaults or broad speed claims.

Private source snapshots, complete writer calls, failed Director passes and receipts are under
`.codex-tmp/director-remediation-20261011/` and the local Test-Bench report.
No changes have been committed or pushed by this follow-up. AI-WRKST-02 was
checked read-only: its clean `dev` branch already tracks `origin/dev`; normal
Pinokio Update will retrieve these fixes after they are published to Dev. All
39 original system-setting keys and 31 service-setting keys matched the pre-test
snapshot at the final comparison; the test writer was unloaded and the local
application was confirmed idle.

## October 11 native H3 music-clip planning follow-up

A fresh run on the 12-GB test machine failed before GPU submission because
the music planner supplied rolling-window prompts inside already segmented
native H3 clips. Joining those windows replaced each authored clip prompt with
several timelines that restarted shot numbering and local clocks. Captured
outputs reproduced the exact consecutive-shot-number validation failure.

H3 planning now constrains `window_prompts` to an empty array and rejects
nonempty arrays during bounded row repair. All local cuts belong in the single
`video_prompt`. The strict final timing/numbering checks remain unchanged, and
the planner revision invalidates obsolete planning checkpoints. Non-H3 window
support is retained.

The focused music planning, native timing, long-form recovery and shot-number
suite passed **91 tests**. Offline replay of the captured first clip verified
bounded repair with a mocked provider response and strict compilation, retaining
the authored cuts and action. All 23 original standalone clip prompts also
compiled in the diagnostic replay. These are CPU/replay checks, not a fresh
writer attempt or a rendered music-video acceptance test.

A narrow backport was installed on the idle 12-GB test machine and Maestro
was restarted through Pinokio. The installed file hash and healthy, idle API
were verified; original outputs and the failed pipeline were retained. This is
one private source-file modification against the older public Dev checkout;
public Dev and Main were not pushed. Reconcile that backport when installing
the eventual published update. The broader release validation holds above
remain in effect.

### Ordered music-video story actions — October 11

Inspection of the next kitten/dragon run found no climb in any of its 23 actual
clip prompts. The overall brief contained the climb, while clip 2 opened with
the kitten already perched on the dragon. Independent cuts bypassed the local
state handoff, and music planning did not allocate the requested source actions.

Music planning now extracts a conservative, finite set of concrete action
clauses from the original brief and allocates them to ordered global clips in
one bounded, checkpointed writer pass. Rows must preserve their assigned IDs
and concrete action clauses; a pending climb cannot be skipped by an independent
cut or baked into an identity definition. Final H3 compilation checks only the
actual detailed visual action for coverage. Scene summaries and global context
cannot satisfy it. The metadata persists across save, resume and regeneration.
Performance-only briefs do not require the extra allocation call.

The latest focused results cover **138 distinct passing tests**, including
music performance, timeline/timing, bounded planning, checkpoint recovery and
native H3 numbering. Nine integration/failure-recovery checks also passed against the exact
three-file private backport for the test machine's older Dev checkout. A CPU
replay rejected the captured mounted opening; four corrected milestone rows
survived actual H3 compilation. These are prompt checks, not proof that a video
model renders the requested motion. The separate one-hour Studio stress test
was stopped at an existing CPU hotspot in `h3_recurrence.py`; it is excluded
from this focused result.

Preliminary live writer attempts were rejected before media generation.
They exposed over-literal wording checks, including an abstract “teams up”
milestone and a physically described but paraphrased climb. Abstract social
verbs are no longer milestones. Bounded climb/flight/swoop/fire inflections now
accept the captured climb wording while checking the actor, boarding target,
sentence-local negation and action order. Wrong-actor and wrong-target tests
remain rejected. Failed planning calls are retained for diagnosis and their
streaming flag is cleared.

A further captured draft exposed partial native formatting: a freeform visual
prompt with only soundscape/music sections at the end was treated as a full
Context-IR prompt without a detailed description. The validator now retains
that visual prefix. Four real writer rows passed before and after H3
compilation, including a climb expressed through direct target contact and
upward body movement. Mounted-only and floating-only prompts still fail.

The old run was cancelled with its nine completed clips retained. The idle
test machine received a hash-verified, backed-up private update and restarted
through Pinokio. A fresh plan uses the same song, kitten reference, 23-clip
timeline, 720p resolution and eight-step Fused References settings. Public Dev
and Main remain unchanged. Finite verb extraction and actor/pronoun aliases remain
heuristic. Fresh run `ff9971bc` planned all 23 clips in eight batches without
repairs and passed native H3 preflight. Review confirmed the climb in clip 1,
flight in clip 2 and a consistent two-subject mapping in the reviewed draft.
The full eight-step render completed on AI-WRKST-02. The October 12 audit
below supersedes that draft-only identity finding and the pending join status.

### Music-video creative regression and final-input audit — October 12

Completed run `ff9971bc` produced
`2026-10-12-03h10m00s_seed281641005_multiclip.mp4`: 23 clips, 228.5 seconds,
1280x704, eight steps. All final clip prompts and sampled frames across this
render, the October 8 original and its October 10 repaired join were reviewed.
This was not a continuous playback or a controlled model-quality comparison:
the original used six steps and a slightly different brief.

The action allocator assigned all four source milestones to clips 1-4:
boarding, flight, swoop and fire attack. The allocator asks for enabling actions
early but has no whole-song pacing or creative-arc acceptance check. This
exhausted the major requested progression in approximately 40 seconds, while
the original's first five clips developed solo performance and movement before
mounting in clip 6. Later new prompts mostly vary the same mounted state,
battlefield and attack imagery. Ordered action coverage therefore passed while
the user's creative acceptance failed. This run does not establish that the
new planner improves general music-video quality.

The final generation metadata also defines the kitten as both Subject 1 and
Subject 3 in all 23 clips, although the reviewed draft had two subjects. The
final prompt compilation/reference mapping path still needs reconciliation
with the local identity fixes and validation at the actual render boundary.
Do not report this duplicate-identity issue as resolved on the test machine.

The original generation metadata retains its original prompts, whereas its
saved pipeline now contains later repairs. Original clips 11-12 and 21-23 put
the global concept into detailed visual instructions instead of a distinct
local action, reintroducing the climb/flight/fire sequence. That explains a
separate mechanism for the original's repeated scenes; the new run's loss of
variety is a planning regression rather than evidence that this original
mechanism was the only cause of repetition.

General remediation should plan section-aware progression across the complete
song before detailed batches, distinguish persistent physical facts from
creative staging, review semantic action/camera/environment diversity, and
validate the final H3 inputs. Preserve good existing shots during targeted
repairs. Evaluate narrative, performance-only and abstract concepts without
hardcoding this cast or requiring every music video to follow one story arc.
No further generation or public branch update was started for this audit.

### Whole-song direction and lyric control — October 12

The local candidate now saves an opt-in **Use lyrics to guide visuals** choice,
off for new music-video projects. Both planning API paths and saved project
restoration carry it. With it off, transcript words are excluded from creative
writer requests while vocal activity, performer mapping, musical timing and
the supplied soundtrack remain available. Changing it invalidates local visual
plans without discarding audio analysis or the timeline. A browser regression
exercises the real UI/store/request/restore paths with all requests intercepted;
it passed, as did the production frontend build.

Four-or-more-clip music plans now receive a bounded whole-song outline before
detailed writing. Narrative actions are distributed through that outline;
performance and abstract concepts can develop without an imposed story.
Physical continuity permits cutaways without replaying the transition that
established a state. Creative review inspects actual visual bodies and records
subjective or ungrounded judgments as diagnostics. Proven repetitions receive
focused repairs; unchanged neighbors are retained and their boundaries are
revalidated. Checkpoints preserve accepted repairs and revisioned review state.
Final H3 compilation checks picture ownership, reconciles article variants,
and filters shared placeholder names while preserving stable subject IDs.
An explicit user prompt edit retains its override of the literal-brief guard.

The final integrated CPU regression passed **230 tests**, excluding the previously
identified one-hour Studio CPU hotspot. This includes native-timing and
unattached-alignment checks. The captured 23-clip kitten/dragon final-input
replay retained two subjects and compiled idempotently after alias repair.
This is local compiler evidence; the remote machine's completed video is not
retroactively changed.

Bounded live tests used the installed local
`Abhiray/gemma-4-E4B-it-heretic-GGUF` and
`JonathanColetti/Qwen3.8-27B-Uncensored-GGUF`, without video renders, cloud calls,
saved-project writes or preference changes. Every attempt, request, response,
source hash and failure is retained privately under
`.codex-tmp/music-direction-20261012`. Sources stayed frozen during each run,
and the harness checked for user jobs before each call and unloaded its writer.

The initial pilot completed 23-clip narrative and eight-clip abstract outlines;
a fixed-stage outline exposed camera vocabulary being misread as a physical
action. The first full-plan run completed the fixed-stage case, but rejected
abstract and journey plans. These failures exposed missing validation feedback,
an action clause swallowing later actions, and invented event-ID variants.
The repair prompts now receive the actual error, source clauses are separated,
and the outline grammar limits event IDs to the source contract. Placeholder
performer labels found during final compilation were also corrected.

A second full-plan run completed four-clip fixed-stage and abstract plans in
84.8 and 62.0 seconds. Final native prompts kept three distinct band members,
a locked wide camera, and changing performance energy. The abstract plan added
no people and developed ink threads into clouds and still translucent layers.
An unrelated synthetic lyric line was absent from all writer requests with the
control off. The journey case failed because a valid shoreline walk contained
intervening descriptive words, exposing an overly literal action matcher. A
third attempt passed the walk but rejected a stop performed by the stable cast
ID `lead_performer` instead of the source label "singer." The validator now
accepts bounded same-verb inflections and adverbs with actor, object/place,
spatial-relation and negation checks. A singer-role bridge requires exactly one
explicitly declared vocalist ID; ambiguous casts remain strict. Both captured
stop drafts pass replay. These failures are retained rather than omitted from
the results.

A fresh journey retest then completed all four clips and final native
compilation in **77.4 seconds / nine writer calls**. The accepted plan moves
from a shoreline walk through the lighthouse stop to the sun disappearing,
retaining one singer. Disabled lyric text stayed out of the writer requests.

The captured fixed-stage opening also exposed a conflicting whole-cast
closed-mouth adjunct alongside the assigned vocalist's lip-sync direction.
Performance normalization now removes that bounded collective closure when
vocals are active and a vocalist is explicit, while retaining individual
instrumentalist directions, silent gaps and explicitly requested silence.
The captured opening passes final native validation and idempotent compilation
after the fix; the original defect remains recorded in the private evidence.

The full-length, 23-clip candidate attempts remain recorded,
including failures. They exposed duplicate source-event assignments, an overly
literal outline/action matcher, and detailed repairs losing accepted actions
while fixing physical state. Source events now use a keyed assignment map;
each required event has exactly one clip index. Malformed maps are repaired
without crashing. Repairs receive the rejected draft in place, and stale
action errors are cleared once fixed. A captured battlefield-running draft
passes replay with its terrain established in the visual body; wrong actors,
terrain, negations and directed transitions remain checked. Outline shorthand
such as "takes flight" produces diagnostic notes when literal matching is
inconclusive. Detailed source-action validation remains authoritative at the
generation handoff.

Long-plan failures also exposed an overly broad contact check: camera holds
and a performer holding a pose were interpreted as physical contact because
the global concept mentioned riding later. The check now distinguishes those
phrases from actual prop holding or support/contact actions. Complete shots
with compact-state materialization errors can receive metadata-only repairs;
incomplete shot rows still use full-row repair. The metadata path retains
visible prose and action beats and uses the same downstream validators and
attempt limit. Both captured Qwen state repairs passed in one call each
(5.25 and 5.64 seconds), with visible fields unchanged. Gemma's corresponding
bounded state-repair attempts remained unsuccessful; those failures are kept.

A subsequent fresh Qwen 23-clip run caught a repeated outline beat, accepted
five detailed clips, then stopped at clip six after two repairs. Its final
boarding draft used a pronoun in a compound action, which the actor check
rejected. The run took 293.8 seconds including model setup, with six writer
calls. It is recorded as failed, rather than counting the accepted outline or
prefix as a completed long plan. A narrow actor-binding fix now accepts that
unchanged boarding draft while retaining ordinary pronoun behavior and checks
against wrong or ambiguous actors and negated actions. Its six-clip prefix
passes final native compilation and subject-binding checks. Both recorded
metadata responses also pass the final repair eligibility guard and full-row
validators without further writer calls or visible text changes. These captured
replays do not establish a successful fresh full-length plan. All 17 live
attempts and 90 writer calls, including failures, remain in the private evidence.

Gemma 4B's subjective critic was frequently inconclusive about exact evidence;
those outputs are diagnostics, not proof of creative acceptance. Deterministic
literal-repeat/full-brief checks and contract validation remain active. Some
writer prose is still verbose or awkward. Structural completion is not full
creative acceptance. Text checks cannot establish rendered motion, identity,
lip-sync, or whether a full-length video is interesting; a matched media review
remains necessary before making those claims.

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
- Selected installed LTX, Flux, Qwen Turbo and ACE-Step smokes extend non-H3
  video/image/audio coverage, but do not verify every shared pipeline. A fresh
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
saved manual controls retain ownership. With Auto enabled, revised owned
recommendations apply once at startup; Re-detect can explicitly reapply them.
Memory allocator changes require a process restart. Performance
learning and test reports remain local, bounded and optional.

Attribution, upstream license copies and native allocator provenance are in
[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md),
[MMGP provenance](../app/mmgp/PROVENANCE.json) and the
[port review](development/wan2gp-17-port.md). The optional H3 VAE 2× integration
is documented separately in the [decoder guide](H3-VAE-x2.md). The LTX 2.5 detail refiner and other unlisted
WanGP 17 workflows or model packages are not included.
