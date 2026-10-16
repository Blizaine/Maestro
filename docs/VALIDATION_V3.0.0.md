# Maestro v3.0.0 validation

The unreleased v2.7.0 candidate was renamed v3.0.0 on October 15, 2026.
Publication targets the public Dev branch only; Main remains public v2.6.0
at `31d7396eb3b5586c1828fc69a6af5eb2a9e83bff`. Earlier dated entries below retain
their original candidate versions and describe the evidence available then.
The latest candidate includes the Flux 3 integration and prompting guides.
RTX 3080 Ti Flux 3 generation remains a user test, not an established result.

## WanGP v18.10 follow-up — October 15

The source port is pinned to WanGP `de302eec64b692dc83949d687cadf7d60bd74091`.
Flux 3 uses the public `DeepBeepMeep/Flux3` snapshot
`1e7c96984937fef03326e3b297fb44a28c30736d`; anonymous downloads verified the
Distilled INT8, Base INT8 and Qwen INT8 files against their published size and
SHA-256. Gate acceptance or a Hugging Face token was not needed. Model licenses
and source notices are retained.

Initial INT8 renders loaded and decoded but failed visual prompt adherence.
The audit found that the new encoder requires quantization-map prefix remapping
and inverse rotation of gathered ConvRot word embeddings. Both were corrected;
a real tiny MMGP embedding regression checks the latter without expanding the
vocabulary table. Fresh public INT8 Qwen features have cosine 0.999967 against
the original BF16 encoder for the tested prompt, with identical token IDs.

The following isolated production-adapter checks ran on RTX 4090, Torch
2.10/CUDA 13, vendored MMGP 4.0, Profile 4 and Auto resolving to Kitchen. Each
checked API/Director idleness, finite RGB pixels, geometry and its output image.
Times are rendering only; allocated memory excludes other device use.

| Flux 3 check | Render time | Peak allocated | Result |
| --- | ---: | ---: | --- |
| Distilled INT8, 512 square, 16 steps | 8.49 s | 1,746 MiB | Requested red paper airplane |
| Distilled INT8, 1024 square, 16 steps | 8.72 s | 4,442 MiB | Requested red paper airplane |
| Distilled INT8, two explicit references, 512 square | 8.56 s | 1,750 MiB | Supplied red/blue shapes in requested left/right positions |
| Base INT8, 512 square, 50 steps, CFG 3.5 | 45.48 s | 1,734 MiB | Requested teapot and two lemons |

The earlier vague two-reference still-life request changed the supplied objects;
it is recorded as execution-only evidence, not reference-fidelity acceptance.
The original BF16/BFL parity audit below remains scoped to its original workload.
New NVFP4 variants, physical RTX 3080 Ti Flux generation and larger reference
workloads still require their own tests.

WGPKernels 1.1.0 installed through the pinned optional helper on both local
CUDA 13/Torch 2.10 and CUDA 12.8/Torch 2.7.1 runtimes. Real MMGP ConvRot layers
passed normal forward and output-row projection against dequantized FP32 math
(relative L2 0.001673). A separate CUDA 12.8 check with M=1, K=6144 and N=2048
passed both paths with relative L2 0.001636 and 0.003253. This is kernel
compatibility evidence, not a generation speed comparison. The ACE-Step 1.5 LM
also receives upstream's activation-outlier precision marker. No package was
installed on workstation 02.

The pinned H3 LightVAE INT8 checkpoint loaded through the production loader on
Torch 2.7.1/CUDA 12.8. It retained the original encoder and all 12 decoder layers.
A real CUDA decode converted a `(1,24,7,1,1)` latent to finite RGB
`(1,3,22,16,16)` in 4.02 seconds, peaking at 1,205 MiB allocated. Auto resolved
to Triton. This small decoder check does not establish full-video image quality
or a long-resolution memory ceiling. CPU decoding hit a Quanto scale-dtype
limitation; no production workaround was added for that unsupported H3 path.
The selected decoder is preserved through Director child jobs and restoration.
The dated TaoMate recipe passed eligibility, configuration and pinned-download
checks; a full generation with that new three-step recipe remains untested.

Fish Drama 3's pinned INT4 assets passed a real CPU pipeline load on Python
3.10/Torch 2.7.1. Native FP8/INT4 expert math and rotated portable fallback
parity passed. All three bundled CPU source/library hashes were verified.
Fresh INT4 generations also passed on the RTX 4090 with Torch 2.7.1/CUDA 12.8:

| Fish engine | Generation time | Audio duration | Peak allocated / reserved |
| --- | ---: | ---: | ---: |
| Legacy, native CPU experts | 63.11 s | 4.46 s | 3.74 / 4.06 GiB |
| vLLM, complete GPU expert bank | 51.29 s | 4.97 s | 19.43 / 19.77 GiB |

Both produced finite mono audio. CPU Whisper-small independently transcribed
both as the requested “Good morning, this is a short fish audio drama 3 smoke
test.” Timing includes first-use staging and kernel work; differing output
lengths also prevent treating these two runs as a speed comparison. During
these runs, an unsupported Triton INT8 launch safely fell back to Quanto.
The later WGPKernels CUDA 12.8 shape check passed, but the speech runs were not
repeated with that backend. MMGP-managed embeddings, native KV-head caching,
two/three-dimensional attention shapes and explicit CUDA-device handling were
corrected during the real engine checks and have focused regressions.

The accelerated engine requires Triton, FlashAttention 2 and both FLA kernels;
unsupported runtimes expose Legacy only. Low-memory profiles keep experts on
the CPU. CUDA Graphs and partial expert GPU residency are outside this port's
scope. Physical 12-GB Fish generation and full three-voice or long-script media
fidelity remain untested.

Desktop/mobile mounted Prompt Helper tests passed drawing, source-region edits,
removal, save/reopen and Y/X coordinate serialization. Frontend lint, TypeScript,
production build, the existing Flux browser regression, 138 standalone media
checks and the JSON grammar runner passed. Full backend discovery passed
4,232 tests with 12 skips before the final Fish runtime repairs; all 26 focused
Fish tests passed afterward. The published Dev commit's CI verifies the final
source. Release source excludes all weights,
private reports and generated media. Main, workstation 02 and saved user
settings are unchanged by this follow-up.

## Dev preparation — October 15

The v3.0.0 Dev candidate includes all changes since public v2.6.0 plus the
native Flux 3 adapter, its generation/edit enhancement guides and browser
submission fixes. The application version reader returns `3.0.0`. Release
notes and validation links now point to the v3.0.0 documents; the old v2.7.0
pages retain forwarding links and the dated evidence is preserved here.

Local preflight passed Python syntax and undefined-name checks, frontend lint,
TypeScript and the production build, the mounted Flux 3 browser regression,
138 standalone preview/media tests and the JSON-grammar regression runner.
The staged-source guard passed with the new files included. Packaging checked
166 relative document links, parsed the new model definition, retained launcher
scripts unchanged and excluded weights, generated outputs and private test
artifacts. CI now includes the Flux 3 source in syntax and undefined-name checks.

The first full local backend discovery ran 4,171 tests with 12 skips and found
three subtest errors plus one failure. These were isolated test fixtures:
alignment cleanup imported WGP's CLI during guide lookup, and the TTS identity
test extracted unrelated Director restore code without its outer bindings.
The corrected alignment test passed all three cases; all eight TTS character
tests passed. No production generation change was needed for these failures.
A full rerun and the Dev commit's CI follow those repairs; their final status
is available in the [Dev CI runs](https://github.com/Blizaine/Maestro/actions?query=branch%3Adev).

Target-stack review found no new dependency requirement or unsupported API for
Python 3.10 / Torch 2.7.1 / Transformers 4.57.1. The local older-runtime model
probe was blocked before inference by a NumPy DLL access-denied error, so it
does not count as a target-runtime model test. Hosted CPU CI uses Python 3.10
and Torch 2.7.1. Physical RTX 3080 Ti Flux 3 generation, memory fit and output
quality remain unverified until the user's test. Start with a simple single
image, sixteen steps and Auto-tune before adding references.

Only Dev is being published. No Main promotion, release tag or GitHub release
is part of this preparation, and neither workstation needs to be stopped by
the publisher. Main's remaining Director media, silent short-film and public
update-path acceptance gates remain in effect.

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

## Release readiness — October 13

The release candidate is prepared locally as v2.7.0. The tested public Dev code
snapshot is `ad3293339c8ecad68c4a7800c7bf7f41a048b3fb`; public Main remains the
v2.6.0 baseline `31d7396eb3b5586c1828fc69a6af5eb2a9e83bff`. Release preparation
does not publish Main, a tag or a GitHub release, and does not restart either
workstation or interrupt user generations in that October 13 preparation. Later entries supersede earlier
failures only for the scope actually retested; historical evidence is retained.

| Gate | Current evidence and status |
| --- | --- |
| Automated source/build checks | Passed on the Dev code snapshot: [CI run 38480363663](https://github.com/Blizaine/Maestro/actions/runs/38480363663), both jobs successful. Backend discovery ran 4,042 tests with 57 skipped; standalone regressions passed 137 with one skipped. Source-boundary, syntax, undefined-name, JSON-grammar, frontend lint, TypeScript and production build checks passed. |
| Director music planning regressions | Later fresh 16-clip narrative, performance and boarding plans reached native preflight; bounded recovery was disclosed. The specific future-event/closing-beat failure was replayed successfully after its fix, and the original project passed all 16 final native prompts and entered generation. These are planning and handoff checks. |
| Editor and per-clip Studio restore | Focused CPU and real mounted-browser checks passed, including retained drafts, unique reference ownership, exact recipe restoration, one-window geometry and switching attention without resizing the pass. |
| Final Director media acceptance | Pending the user's latest render review. Check sustained motion, flashes, singing/lip-sync, physical transitions and whole-video scene progression. Earlier successful decoding does not clear semantic or visual findings. |
| Silent/action-only short films | Pending a fresh end-to-end check of explicit no-speech/no-music requests, prop retention and causal transitions after the latest fixes. Earlier failures were followed by code and contract tests, but no fresh approved short-film render clears this gate. |
| Public-version update path | Pending an idle v2.6.0-to-candidate Update/restart smoke, including preserved workspaces, manual preferences and model links. Existing installed Dev updates are not a substitute for this baseline check. |

No failing CI/build gate is known on this snapshot. Main promotion remains held
for the three pending acceptance/update gates above. Experimental Veda is opt-in;
the reported one-frame flash/motion change requires a matched comparison before
making quality claims. Linux CUDA and 6/8/10-GB VRAM or 16-GB RAM validation remain
separate coverage limits, rather than tested compatibility claims. The remaining
validation limits near the end of this document still apply.

Release notes and the Auto-tune guide now match the shared resource policy:
the tested 12-GB / at-least-30-GiB-usable-RAM tier recommends 345-frame, approximately
14.4-second 720p Pruned and Fused passes. The Pruned tier includes repeated user
success reports and removes its older reference margin for this tier only.
Native 1080p remains conservative. No generation settings are changed by this
release-document preparation.

Local packaging preflight checked 152 relative document links, parsed all 13
changed application JSON files and verified the 14 bundled MMGP allocator files
against their recorded SHA-256 values. TypeScript's commented configuration
is covered by the successful frontend compiler check. The source-boundary guard
passed on all 2,938 tracked files. Only release documentation changed after the
tested Dev code snapshot during that preparation; no new model weights, private
reports or generated media were included. The October 14 gallery follow-up below
is a subsequent local code change and is not covered by that October 13 CI run.

## Gallery duration follow-up — October 14

Gallery video cards and side thumbnails now display measured duration before
playback, retaining image posters and `preload="none"`. A workspace-qualified
header-facts endpoint reuses the bounded, file-stat-invalidated media cache.
Visible UI consumers share requests, with bounded queue/concurrency/cache and
an abort timeout. Unknown, invalid and boolean durations do not become labels;
a matching measured metadata seed can survive a failed lookup.

All 29 focused thumbnail, media-facts and gallery-library backend tests passed,
including no-frame-decode, workspace/upload boundaries, actual versus requested
duration, unreadable files and replacement invalidation. The isolated desktop
and phone browser checks passed with no video-byte requests before playback.
They cover visible-only loading, shared lookups, same-name workspace separation,
fallbacks, stat-only and URL invalidation, formatting and control clearance.
Existing desktop/StrictMode/mobile gallery navigation and clip-action menus at
three viewport sizes passed. Frontend lint, TypeScript and production build,
Python syntax and undefined-name checks passed.

After jobs, the Director queue and saved pipelines were confirmed idle, only
the local development app was restarted through Pinokio. The new endpoint read
two existing 345-frame / 24-fps videos as 14.375 seconds, with file sizes matching
the listing, and the server served the rebuilt frontend. No render, preference,
saved prompt or media file was modified; the second workstation was untouched.
The release notes include this follow-up. Main, public Dev and release tags
were not pushed by this change.

## Director clip visibility follow-up — October 15

Saving exact per-output Director settings exposed an older gallery rule that
collapsed every `multi_clip_info` group. Standard independent Director shots
were treated as superseded cumulative segments and disappeared from listings
when the final shot completed. The files and saved pipeline were retained.
The gallery now preserves independent Director clips, including older sidecars;
cumulative sequence and Studio filtering retain their previous behavior.

All 41 gallery CPU regressions and the Editor draft round-trip test passed,
covering completion, pre-join visibility, late sidecar publication, pagination,
favorites/search and cumulative filtering. A read-only check of the actual saved
nine-shot production returned all nine shot files and its joined output using
the corrected library. All nine files also passed the running Editor media-probe
endpoint with valid dimensions, audio and measured durations. The existing
Editor project was not modified by these checks.

Frontend lint, TypeScript and the production build passed. The Editor's saved
run import is independent of gallery visibility. Its partial-run fallback now
distinguishes an actual joined or seamless master from the last individual clip,
so a cancelled standard run imports only its surviving completed prefix.
The isolated real-store browser regression passed for a final-only media library
importing nine distinct clips, their prompts and master audio; seamless master
source ranges; and an interrupted run with one missing completed file. All
requests were mocked, with no live project writes or generation requests.
No video generation, app restart, second-workstation update or Main publication
was performed for this follow-up.

### Collapsible Director groups

The gallery now opts into grouping independent Director shot files beneath the
latest joined output, collapsed by default. Groups are formed before pagination
and their members load separately in clip order with workspace-qualified URLs.
Search and Favorites retain direct matching shots; unjoined or interrupted runs
retain their finished files. The default ungrouped output API and Editor import
remain compatible. Older clip and rejoin ownership is recovered read-only from
stat-cached saved plan records, without rewriting source sidecars.

Rejoin uses the Director project's recorded current takes, writes ownership
metadata for the new master, and avoids replacing a previous master when names
collide. The gallery, Director cancellation/rejoin and API lifecycle suites
passed 49, 70 and 28 CPU tests respectively. The actual nine-shot production
resolved to one parent group and nine correctly ordered members. These checks
used existing file facts and mocked concatenation; no user video was rejoined.
The Editor also prefers the latest joined master when it must reconstruct
editable ranges from a combined file. Four isolated real-store browser import
cases passed, including retained older masters and a missing individual source.
The mounted desktop/mobile group regression passed lazy loading, sorted takes,
collapse selection, favorite/load-settings/delete/move actions, safe Director
rejoin, parent deletion recovery and stale-scope retry. Stable bounds and pointer
hit tests confirm the controls are visible with a normal gap before the next
card. Existing desktop, StrictMode and mobile gallery navigation passed, as did
clip-action menus at three viewport sizes. Frontend lint, TypeScript, production
build and test-script syntax checks passed. All browser traffic was mocked;
no live generation, rejoin, saved project edit or app restart was performed.

## Duration policy follow-up — October 14

The shared matrix is now versioned `2026-10-14`. Studio and Director use the
same RAM, VRAM, canvas and checkpoint tiers. The 12-GB / 32-GB 720p Pruned and
Fused default remains 345 frames; the 24-GB / 128-GB Fused Frames native-1080p
default now uses 345 frames, anchored to the completed six-step SLA control.
Intermediate resource classes and 16-GB-host-RAM systems have explicit
estimated starting points. The [matrix and evidence](Duration-recommendations.md)
separate measured, reported and inferred capability. Longer native-1080p
12-GB results remain conditional, rather than a broadly verified default.

The UI labels the basis of each recommendation. Exceeding a starting point
does not become an automatic OOM prediction. Native versus chunked RMSNorm
guidance no longer claims a calibrated whole-render speed tier. The older
116-minute Pruned dense/Sage2 observation is not applied to SLA or Veda as a
sparse-attention timing warning. Manual durations and the separate experimental
30-second opt-in retain ownership.

All 278 focused H3 definition, Fused, performance-policy, duration, geometry,
extended-window, reference-window and Director timing/model tests passed. The
duration-policy tests also check the Director clip-limit response's provenance.
Both focused UI unit scripts and both mounted-browser duration suites passed,
including desktop/mobile layout, late hardware detection, manual choices and
Studio/Director agreement. Scoped frontend lint, TypeScript and production
build passed. These are policy and UI checks; no new GPU generation was run
for this refresh, and no physical 8-GB / 16-GB result is claimed.

After confirming jobs, the Director queue, saved pipelines and the writer were
idle, only the local development app was restarted through Pinokio. Its live
model policy reports version `2026-10-14` and all 101 resource rows. Read-only
Director clip-limit requests returned 345 frames with tested provenance for
Fused Frames at 1920×1088 and Fused References at 1280×704, and the rebuilt UI
asset was served. The second workstation was untouched. Main, public Dev and
release tags were not pushed.

## Local writer catalog follow-up — October 14

The local picker adds Kujira's Underdog Saluki 27B IQ2-Mix Uncensored, using
the verified GGUF filename and the original project's Q8 vision projector.
The profile has a 32K context and Q8 K/V cache; Gemma remains the default.
Bounded reads of the actual GGUF metadata confirmed the Qwen hybrid architecture,
its optional NextN head and the embedded thinking/reasoning controls. The
existing compatible llama.cpp loader handles this layout with drafting off;
native MTP speculative decoding is not enabled by default for this profile.

All 53 local-LLM registry, payload, lifecycle, responsiveness and UTF-8 tests
passed, including five new Saluki contracts. Five standalone JSON-grammar
checks, Python syntax and undefined-name checks passed. These are registry,
metadata and mocked request checks. The full model and projector were not
downloaded, and no Saluki inference or physical 12-GB memory test was run.
The 12-GB picker hint is an estimate, not a measured compatibility claim.

The idle local development app was restarted through Pinokio. Its live local
catalog exposes the Saluki entry and estimated 12-GB hint. The selected writer,
provider, device and enhancement-writer settings matched the pre-restart
snapshot; no model was selected or loaded by the check. The second workstation
and public branches were untouched by this follow-up.

## Experimental MTP settings follow-up — October 14

Integrations now exposes an off-by-default MTP acceleration toggle for local
writers with verified draft weights, including a separate local enhancement
writer. Model catalog capability metadata keeps no-MTP Qwen and Gemma weights
out of this route. The saved preference is a validated boolean and is frozen
with queued enhancement settings; legacy snapshots retain ordinary decoding.
Saving it does not load or unload a writer. The next eligible load compares
the requested mode with the actual resident mode and reloads only when needed.
Settings reports the actual loaded writer's MTP state.

All 221 focused LLM, runtime, services-settings, enhancement and Director
compatibility tests passed. The mounted desktop/mobile browser suite passed
preference persistence, supported/unsupported weights, separate enhancement
writers, actual-mode display, remote exclusion and no load/unload requests on
save. Scoped frontend lint, TypeScript and production build, plus Python
undefined-name checks passed. A CPU-only help probe of the installed runtime
confirmed both flags and the native `draft-mtp` mode. Capability checks also
cover missing modes, bounded timeouts, transient retries and cached results.
The five standalone JSON-grammar checks also passed.

The local development app was restarted only after the generation queues,
Director pipelines and writer were idle. Live API checks confirmed the Saluki
capability, default-off preference, actual unloaded/off status and rebuilt UI
asset. The selected writer, provider, device and enhancement-writer preferences
matched the pre-restart snapshot. Machine 02 and public branches were untouched.

These checks do not establish native-MTP generation speed, output quality or
long-run stability. No MTP inference or model download was started for this
follow-up. The experimental label and default-off behavior retain that limit.

## Automated checks

### Music-video repetition follow-up — October 14

Two consecutive Saluki music-video planning attempts failed in the whole-song
outline check. Three saved 29-row drafts reproduce the rejection, including
both near-matching flight beats and a literal duplicate outline beat. The old
comparison removed treatment details and normalization rejected the draft
before retaining it for an in-place repair. The repair could rewrite the whole
arc and fail on a different pair.

Outline repetition is now separate from structural validation. Similarity
requires a close beat and treatment; justified reprises are allowed. A bounded
repair edits only flagged later rows, preserving valid neighbors and immutable
source-action assignments. A malformed or still-repetitive optional repair
retains the valid outline and saves advisory notes for detailed writing.
Subjective detailed-review findings also retain validated originals after the
repair budget. Literal complete-native-prompt recycling, whole-project resets,
source-action obligations and timeline coverage keep their strict checks.

The earlier Dev CI failure was seven stale exact-call assertions in the
enhancement-provider routing tests after the MTP loader argument was added.
Their expectations now explicitly require MTP to remain off for those cases.

Validation: the 845-test broader compatibility suite was green with three
environment skips, and the final 89-test arc/review/routing/long-form integration
suite passed. The latter includes the 42 focused arc/review regressions and six
optional-review regressions. All three captured rejected outlines now pass six
CPU replay cases (unchanged versus malformed repair), preserving their original
beats and all four source-event assignments. Duplicate required source events
still fail validation. Python compilation, undefined-name and whitespace checks
passed. No new writer inference or video generation was submitted for this fix;
these checks establish planning recovery, not rendered variety or lip sync.

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

The Director clip window follow-up exercises the actual mounted Studio duration
controls during model and resolution hydration. Frames and References recipes
retain a single 345-frame window even when the hardware Auto profile recommends
243 frames, including two mounted panels and a Veda-to-SLA change. The intercepted
References submission retains the exact prompt, seed and single-window geometry;
ordinary new Auto generations still follow the hardware profile. Replaying the
old unconditional selection reset fails this browser regression. Existing
Studio duration, measured-default, reference-duration and Editor draft browser
checks passed, along with 151 focused H3/settings/duration Python tests, frontend
lint, TypeScript and the production build. The prior CI source assertion was
updated to accept Director's exact-prompt restore while retaining ordinary Studio
writer-mode restoration. The running server serves the rebuilt UI without a
restart. These checks used intercepted submissions and did not interrupt or
enqueue a live generation.

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

## October 13 Director planning follow-ups

Private frozen receipts at `f8fd02e` record four fresh 16-clip plans passing
final H3 native preflight: two narrative variants, a fixed-stage performance
and a separate boarding concept. One narrative case used source-compiled
recovery for one clip; the others did not. The corresponding focused gate
passed 709 tests with one skip. No video was submitted by these planning trials,
and their receipt explicitly does not verify rendered quality.

Later trials at `7150bb3` retained their failed controls and exposed mixed
writer numbering, a visual action misclassified into audio, and a closing beat
that performed a future action. The latter is now included in raw validation
and focused repair before native compilation. The exact rejected closing draft
passes recovery and final native validation with earlier accepted rows unchanged.
A separate fresh 16-clip boarding plan passed native preflight in 164.208 seconds.
The user's original 16-clip project also passed final preflight and entered
generation after recovery. These results supersede the specific captured
future-event failure, without establishing that every stochastic plan succeeds.

Text review still recorded an ambiguous mounted-versus-side-by-side description
in one closing beat. Subsequent fixes preserve visible physical direction,
performer/vocalist roles, assigned source actions and unique picture ownership;
tests and captured replays verify those compiler contracts. The current user
render reached generation with 18 planned clips and no pipeline error when
checked read-only during release preparation. Completion and human acceptance
of that render remain pending, as do matched attention-quality checks. No
generation was cancelled, restarted or enqueued for this preparation.

## Remaining validation limits

### October 14 YuE2 download repair

AI-WRKST-02's failed Director song request completed lyric writing, then received
HTTP 404 for `yue2/vae_config.json` at the retired DeepBeepMeep/TTS snapshot.
The replacement pin `6e87e439357e25ebb563628a2ea6e93a46a8c4b1` matches all 23
checked-in asset entries' sizes and recorded LFS hashes. The small configuration
and tokenizer were fetched and their Git blob hashes checked locally; the test
machine also fetched and parsed the replacement configuration in memory with
HTTP 200. No full weights or new songs were generated for these checks.

Offline regressions cover actual inference/default/training descriptors and
loading an existing artist-style bundle with the original compatibility identity.
The YuE2 and music suites passed 131 tests (seven hardware-dependent skips);
changed modules also passed syntax and undefined-name checks.
The base weights and style contract are unchanged. End-to-end song generation
after updating the test machine remains a separate validation step.

### Other validation limits

- A captured H3 music-video request mixing singing with reciprocal combat
  produced nine solo-performance clips in its text plan. The original action
  parser extracted no source events from its broad fight/battle wording; the
  reference images still reached the renderer. Regression coverage now checks
  collective participant binding, source-grounded outline recovery, and later
  combat continuation without requiring combat or both characters in every clip.
  The broader Director CPU suite passed 597 tests; focused combat, source-action,
  recovery and repetition checks also passed. Six captured-source replays cover
  named and unnamed reference hints and three possible fight assignments, keeping
  the other eight outline beats intact. No fresh writer or video call was made
  while the user's render was active.
  These are text and planner checks. A fresh rendered comparison remains needed
  to establish battle execution, effects, motion and lip-sync quality.
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
  installation remains recommended. The public-v2.6.0-to-candidate Update/restart
  smoke is a pending Main gate in the [readiness record](#release-readiness--october-13).
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

## Flux 3 Klein local integration — October 15

The local candidate adds the official distilled FLUX 3 Image Klein architecture,
native flow-map sampling, Qwen3-VL-4B hidden features and FP32 VAE. The meta-built
DiT has 5,548,681,984 parameters, matching the published checkpoint metadata. A
reduced-width, identically weighted comparison with the official reference code
passed for independent packs with and without reference images; maximum absolute
error was 1.79e-7. This is numerical adapter evidence, not generated-image quality
evidence or a speed benchmark.

58 CPU checks passed across the new runtime and surrounding download/visibility
contracts. They cover dispatch, checksum-verified downloads, complete component
readiness, shared model paths, cancellation, inactive controls, sampler state,
reference selection, output geometry and serial batch progress. Mounted desktop
and mobile browser checks cover model selection, the 16-step default, ten-image
reference controls, workflow filtering and persisted visibility. Scoped lint,
TypeScript and the production UI build passed.

The primary environment (PyTorch 2.10.0, Transformers 4.57.1) also loaded a tiny
two-shard fixture through the actual installed MMGP loader, including the Qwen
prefix/head mapping. The following real-checkpoint runs separately exercise the
full Qwen model, native DiT/VAE and CUDA MMGP hooks.

After the user received Hugging Face access, the production model-download
endpoint downloaded the pinned official checkpoint and required components.
The main checkpoint passed its declared size and SHA-256 checks. Three serial
API jobs completed in the isolated `Flux3-Test-20261015` workspace on the
RTX 4090 / 24-GB VRAM / 128-GB RAM workstation, all at 1024 × 1024 and 16 steps:

| Job | Test | API completion time | Denoising | Sampled peak VRAM |
| --- | --- | ---: | ---: | ---: |
| `ad17c96c` | Text-to-image, including initial loading | 44.48 s | 8.172 s | 13.24 GB |
| `8a47de64` | Single-reference color edit, model loaded | 25.58 s | 16.125 s | 14.14 GB |
| `4caafaa5` | Two-reference composition, model loaded | 35.89 s | 28.266 s | 15.08 GB |

Peak VRAM is the generated sidecar's device-wide NVML sampling, including other
GPU use; brief peaks may be missed. An independent nvidia-smi sampler peaked at
13,131, 14,059 and 15,017 MiB respectively. API completion includes queue handoff
and output saving, so it is not interchangeable with the denoising measurement.
MMGP logs confirm separate hooks for the real transformer, Qwen3VLModel and
native AutoEncoder, with profile 1 and Windows read-ahead enabled.

All saved files decoded as finite, nonblank 1024 × 1024 RGB images. Visual review
confirmed a red mug with the requested MAESTRO label, its cobalt-blue edit with
the composition retained, and a two-object result with the red mug on the left
and blue mug on the right. Private job results, samples, output metadata and
the local HTML report retain the underlying evidence. These checks clear the
initial real-checkpoint generation/edit gate for this tested workload; they do
not verify lower-memory systems, maximum reference counts, large resolutions,
LoRAs or unsupported checkpoint formats.

The Flux 3 changes remain local and unpromoted. No saved performance settings,
active workspace selection, user project, public branch or machine-02 install
was changed by these tests. The agent did not restart the application. See the
[Flux 3 guide](Flux-3-Klein.md).

### Ten-reference submission follow-up — October 15

User job `35748f1c` loaded ten reference paths, then failed model validation in
0.14 seconds with no denoising and a 1.94-GB device-wide sampled peak. The logged
reason was an unsupported reference selector, not CUDA OOM or the reference
count. Image submission could merge a previous model's selector flags with
Flux 3's `I` flag. Flux 3 submission now uses its current valid native choice
and clears legacy flags, including when references are removed. Other models'
reference and masked-edit modes retain their existing contracts.

Real API job `e3b2109b` replayed the same ten uploaded files with selector `I`
at 1280 × 720, 16 steps and seed 424245 in `Flux3-Ten-Refs-20261015`. It completed
in 75.19 seconds API time, with 67.531 seconds denoising and 17.44 GB peak
device-wide NVML sampling; an independent nvidia-smi sampler peaked at
17,433 MiB. The saved file decoded as nonblank 1280 × 720 RGB. The native
ten-reference conditioning path completed without dropping inputs. Visual
review counted nine people in the requested ten-person group: runtime capacity
passed, but complete subject retention did not. The test did not reuse the
unavailable full original prompt; its recorded group-photo prompt is retained
with the private request and output metadata.

Two additional CPU cases verify all ten references retain their order and
distinct negative native conditioning IDs, and that an eleventh is rejected
instead of silently discarded. The full focused runtime suite passes 15 tests.
The mounted UI regression exercises actual generation submission with ten
ordered uploads and stale flags, plus empty-reference cleanup and the existing
Flux 2 reference mode and Qwen masked editing. All 22 focused integration/runtime
CPU tests pass, along with the mounted browser runner, scoped ESLint, TypeScript,
production Vite build and tracked-source guard. Existing saved settings,
workspaces and public branches remain unchanged; the generated test image is
in its own workspace. The local UI build is served immediately; refreshing the
page applies the submission fix without restarting the model server.

### Flux 3 enhancement guides — October 15

The native Flux 3 architecture now resolves separate text-to-image and reference
editing guides instead of the generic Flux 2 guide. The image-enhancement
payload carries an exact upload-order map from UI Image 1 to `<ref_image_0>`
through UI Image 10 to `<ref_image_9>`. The output contract permits these tokens
and preserves complete binding phrases through cleanup. A synthetic ten-binding
caption reproduced truncation by the generic substring-loop heuristic; the
Flux 3 path now retains the full caption and its exact quoted lettering. Other
models retain their current cleanup and guide selection.

The guides distinguish bounded changes, new compositions and donor roles;
preserve explicit counts, nonhuman subjects, ownership, style and text; and do
not generate unsupported mask/layout commands or model settings. Flux 3 uses
the selected model-specific LLM path when the legacy cinematic enhancer is
enabled. No BFL prompt-upsampler weights or new dependencies were installed.

All 46 focused CPU tests passed across guide selection/overrides, native
reference context and cleanup, provider routing, active LoRA guidance and H3
performance-audio behavior. Python syntax, local documentation links and diff
whitespace checks passed. The earlier generated-image smokes predate these
guides: no fresh local-writer or matched render comparison establishes improved
subject retention yet. The current prompt-bench endpoint is H3-video-only and
was not used to imply image-workflow coverage. A normal Maestro restart loads
the new Python routing before user testing; no application restart, user-job
interruption or public push was performed for this guide update.

### Flux 3 numerical integration audit — October 15

The user's unenhanced beach pair used the same prompt, seed `366749024`, and
1088 × 1920 output geometry, with references and LoRAs disabled. Flux 2 Klein
9B used four steps; distilled Flux 3 Klein used its intended sixteen steps.
The user's visual preference for Flux 2 is not itself an inference-parity test.

Additional real-checkpoint CPU diagnostics compared the local adapter with BFL
reference revision `45e57c9fb06f9751cdced29a65c5d2171003be05`:

- The actual `model_factory` / MMGP loader retained all 1,323 component tensors
  bit-for-bit: 359 DiT, 713 bare Qwen encoder, and 251 VAE tensors. DiT/Qwen
  floating weights were BF16; the VAE remained FP32.
- With the real 4B Qwen weights and a short beach-photo prompt, chat templates,
  token IDs, masks, four-row position IDs, and the selected 80 × 20,480 hidden
  features matched exactly. Both default SDPA and Maestro's custom SDPA/GQA
  handler matched the upstream-style causal-LM wrapper on CPU.
- The full 5.55B-parameter DiT matched exactly at three tested timesteps with
  small token packs, including a reference-conditioned case, using common
  CPU math attention. This exercises the full model, not full image geometry.
- Real-weight VAE encoding/decoding matched exactly at 32 × 32 and 64 × 64,
  including decoding synthetic latents. Batch-norm statistics and epsilon
  matched the reference.

Private diagnostic scripts and JSON receipts live under
`.codex-tmp/flux3-quality-20261015/`. All four diagnostics explicitly disabled
CUDA and ran in the existing Torch 2.10 / Transformers 4.57 runtime. These
checks do not establish CUDA kernel or full-resolution pipeline parity, nor
parity with BFL's separately pinned dependency environment. The subsequent GPU
replay below compares the same prompt, seed, dimensions and sixteen-step
schedule, capturing text features, per-step latents/velocities and FP32 VAE
pixels. The reference replay stages components between CPU/GPU to fit the
24-GB card; the native replay uses Maestro's actual MMGP runtime.

BFL's demo enables a separate prompt upsampler by default; Maestro's raw-prompt
test does not use it. This is a comparison variable, not a demonstrated cause
of the user's realism concern. No production inference code, saved settings,
running job, public branch or machine-02 install changed during this audit.

### Flux 3 full-resolution GPU reference parity — October 15

With the user's explicit authorization for exclusive workstation-01 use,
Maestro was stopped and restarted through Pinokio. No generation was active
when it was stopped. Three serial GPU runs used the user's exact beach prompt,
seed `366749024`, 1088 × 1920 output and sixteen steps, without prompt
enhancement, reference images or LoRAs:

1. Pinned BFL `Pipeline.generate`, using its BF16 DiT and FP32 VAE. Complete
   components were staged between CPU/GPU to fit the 24-GB card.
2. Maestro's actual `model_factory.generate`, MMGP Profile 1, saved optimized
   allocator, normal empty-LoRA lifecycle, and WGP's explicit-device context.
3. The same native path with only DiT attention set to BFL's cuDNN function;
   Qwen encoding, memory management, weights and sampling remained native.

All nine used reference-source files matched immutable GitHub commit URLs
byte-for-byte. Both branches used the existing Torch 2.10.0+cu130 /
Transformers 4.57.1 environment; BFL's separately pinned environment was not
installed. The raw decoder tensor was captured before clamping on both paths.

GPU text features, initial noise, image/text/decode IDs and all sampling times
were bit-exact. Native default SDPA selected PyTorch efficient attention on
this runtime, whereas BFL forces cuDNN. Default attention produced numerical
divergence starting at the first DiT step; the final PNG's mean absolute
channel difference was 1.676 out of 255 (RMS 6.546, with sparse larger local
differences). The rendering retained the same subject, composition and light.

With only the native DiT attention backend matched to BFL, every latent,
velocity, time and target tensor across all sixteen steps matched bit-for-bit.
The final latent, raw FP32 decoder output and every final PNG pixel also
matched exactly. Native VAE decoder weights and running mean/variance remained
FP32 during GPU execution. This establishes numerical pipeline parity for
this full-resolution text-to-image workload in the tested runtime. It does
not establish a realism advantage over Flux 2 or a quality gain from cuDNN.

Elapsed diagnostic times were 36.0 s for the staged reference, 31.4 s for
native default and 29.9 s for native with cuDNN. These include different
loading/staging and trace overhead and are not clean speed benchmarks.
Native allocator peaks were about 11.84 GiB allocated and 12.22/12.31 GiB
reserved; these are process allocator statistics, not device-wide NVML peaks.

The local `Flux3-Quality-20261015` report includes the original Flux 2/3 images,
all three replay outputs, source hashes and numerical receipts. Private
scripts and per-step tensors remain under the numerical audit directory.
No production inference code, saved performance settings, public branches,
machine-02 installation or active workspace selection changed. The audit
does not add reference-image parity coverage or lower-memory hardware coverage.
