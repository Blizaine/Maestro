# Maestro v3.0.0

Prepared October 15, 2026. Supersedes the unreleased v2.7.0 candidate.
Public Dev testing candidate; not promoted to Main.
Changes since the public v2.6.0 Main release.

This update brings the adapted WanGP 17 / MMGP 4 memory engine, workload-aware
Auto-tune, longer recommended 720p H3 clips on the tested 12-GB tier, experimental
SLA/Veda choices, H3 VAE 2× decoding, Director planning and editing fixes, a simpler
model picker, Flux 3 Klein, Qwen Image 2.1 Turbo, and experimental MTP for
compatible local writers. Earlier Dev baselines passed CI; the later gallery
and Flux 3 follow-ups passed focused backend and browser checks. Main
publication awaits the remaining review and update checks in the
[readiness record](VALIDATION_V3.0.0.md#release-readiness--october-13).

Director now keeps explicit speech/music permissions and source actions outside the shortened
screenplay, preserves authored endings, and checks declared support/contact and
physical transitions before generation. Save, resume and clip regeneration keep
these contracts. Creative dialogue repair checks participant coverage and
duration together; a proven existing line can receive its native speaker marker
locally. A numeric duration label is no longer interpreted as another speaker.
These checks validate prompt contracts; rendered motion and audio require
separate review. Later 16-clip music plans passed native preflight after bounded
repairs. Earlier failures remain in the validation history; they do not establish
the current candidate's rendered quality or clear the separate short-film review.

An additional music-video planning check allocates concrete requested story
actions to ordered clips before the writer drafts them. For example, a requested
climb must appear in an actual clip before the rider is shown mounted, even
across independent cuts. Final H3 prompt compilation rechecks the action;
global project descriptions do not count as a performed action. Ordinary
performance-only briefs do not need this extra allocation pass.

Music videos with four or more clips now start with a compact plan for the
whole song before detailed shot writing. It gives later batches the full
progression, supports intentional refrains, and keeps continuing physical
facts separate from creative staging. Reviews inspect visible action rather
than repeated identity/soundscape fields. Focused repairs preserve accepted
neighboring shots and recheck their physical boundaries. Subjective reviewer
suggestions are retained as diagnostics rather than rewriting accepted shots.

Whole-song beat similarity no longer aborts a structurally valid music-video
plan. The check considers visible treatment, attempts a bounded repair of the
flagged beats, and preserves accepted neighbors and source-action assignments.
Unresolved similarity is saved as an advisory and passed to detailed writing;
it is not relabeled as an intentional reprise. Recurring performance and
continuing scene state are allowed. Timeline coverage, required story actions,
literal recycling of final visual prompts and whole-project resets retain
their strict checks.

**Use lyrics to guide visuals** is optional and off for new projects. Enable
it to use the lyrics as thematic inspiration alongside the visual concept.
Transcription, section/beat analysis, the supplied soundtrack and assigned
lip-sync continue in either setting. Projects save the choice; changing it
clears their cached visual plans while retaining audio analysis and timing.
Final H3 compilation also checks picture ownership, handles article
variants in reference names without adding a duplicate subject, and keeps
placeholder names such as “Unnamed” from merging distinct performers.

Editor Director prompt edits now save as per-shot drafts, separately from the
prompt used for the last rendered take. Deselection and failed reruns retain
the draft; errors remain beside the rerun button and identify the actual
Director clip. H3 reference matching also recognizes semantic cast IDs such as
`dragon_creature`. On rerun it repairs unambiguous legacy duplicates created by
the compiler while preserving authored action text and distinct characters.
Appearance wording such as “scary-looking” is retained during identity cleanup.
Qualified character names and semantic IDs now match their unique reference
roles too, without merging distinct picture owners or multiple characters.

The gallery's **Load Settings** pencil opens an individual Director clip in
Studio with its own rendered prompt, seed, duration and media inputs. A supplied
soundtrack is restored at that clip's interval, and Frames clips retain their
start/end images. The joined video's pencil reopens the complete Director
project. Supported older sidecars are reconstructed without rewriting their
saved prompts or media; unavailable or ambiguous recipes show an error.
Prompt enhancement is initially off for a restored clip so attention comparisons
can use the saved prompt unchanged. Its original duration and window geometry
survive model/resolution hydration and attention changes, instead of Auto
silently splitting a single native clip into multiple windows.

Completed Director productions show the latest joined video with **Show clips**,
collapsed by default. Expand it to browse the individual shot files beneath the
final, retaining their normal media actions. **Edit Director shots** opens the
saved production for prompt edits and replacement reruns; **Rejoin** assembles
its current takes into a new final and keeps the previous video. Search and
Favorites still find individual shots, and runs without a final show their
finished clips directly. Cumulative intermediates retain their existing filtering.
To edit the full production as separate shots, open **Editor → Media → Director**
and choose **Add run**. This imports the saved shot files, their Director prompts
and the original soundtrack; adding the joined video as ordinary media imports
one file. Interrupted standard runs import their surviving completed shots.

Video cards and the side thumbnail strip also show the actual clip duration
before playback. Labels use measured file-header facts, rather than the requested
generation duration, while retaining image posters and lazy video loading on
desktop and mobile. Visible cards share bounded lookups, file replacements
invalidate the cached duration, and unreadable files omit the label. A lightweight
`GET /api/v1/media-info/{filename:path}?workspace=...` endpoint exposes the same
cached measured facts without loading generation metadata or decoding frames.

## Memory management and performance

The October 12 Dev follow-up adds staged SageAttention for SM86 GPUs such as
the RTX 3080 Ti, releasing temporary Q/K/V allocations earlier while retaining
the ordinary dispatcher when optional kernels are unavailable. Existing
SM89/SM120 staging and manual attention choices are preserved.

Full and Pruned H3 in Frames and References now offer **Veda Sparse Attention
— Experimental** in Studio and Director. It is off by default and mutually
exclusive with Sol and SLA. Its learned predictor is downloaded on first use,
verified against pinned size and SHA-256 values, and managed with transformer
weights by MMGP. Standard ConvRot models use an approximately 281-MB predictor;
other eligible weights use the approximately 551-MB BF16 predictor. Fused Frames
and References also permit an explicit experimental Veda selection, retaining
their default SLA recipe. Specialty checkpoints are excluded. CPU regressions cover dispatch, reference
geometry, selection and asset handling. Selected user hardware tests have run,
but no controlled release-wide speed or output-quality claim is made.
[Integration and comparison guide](development/h3-veda-port.md).

Maestro now includes an adaptation of WanGP 17 and MMGP 4 for its existing
generation pipelines. It adds optional VRAM and RAM allocator choices, smarter
memory pinning, Windows checkpoint read-ahead, per-output dynamic or manual
weight preloading, compatible attention options and automatic INT8 kernel
selection. The base allocator default remains PyTorch; Auto-tune recommends MMGP optimized where supported. Unsupported devices
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
[validation record](VALIDATION_V3.0.0.md) for exact settings and limits.

### Native 1080p on the 12-GB test machine

A manual H3 Fused References run now completed 362 native 1920×1088 frames at
24 fps (15.08 seconds), using six steps, SLA, Medium head splitting, Profile 4
and the Q2_K encoder. It included an image reference, uploaded soundtrack and
live preview, with upsampling off. Total generation took 32m04s; sampled peak
VRAM was 11.75 GiB. Full media decoding and soundtrack checks passed, and sampled
frames showed a coherent scene. System RAM was nearly full and paging occurred.

The native test used a locked window with **Allow 30s clips · Experimental**
enabled. This does not promote 15-second native 1080p to an Auto duration default.
Auto starts shorter at this resolution, as described below. This verifies one
12-GB/32-GB Windows recipe, not all checkpoints,
LoRAs or lower-memory machines. The separate WanGP Pruned 20-step confirmation
used different settings and is not a matched speed comparison. See the
[validation record](VALIDATION_V3.0.0.md#native-h3-follow-up--october-9).

## Automatic H3 windows and head splitting

Studio and Director now share versioned duration recommendations based on usable
VRAM, system RAM, resolution and checkpoint. On the tested nominal 12-GB GPU /
32-GB RAM tier, Auto recommends approximately **14.4 seconds for 720p Fused
and Pruned** and **5.2 seconds for native 1080p**. The Pruned increase includes
repeated successful user tests, rather than a controlled timing study. Reference
and rolling-window policies can subtract a safety margin or include history in
the actual pass. On the tested 24-GB / high-RAM tier, Fused References also uses
the longer 720p recommendation. The October 14 refresh adds the tested
24-GB / 128-GB Fused Frames native-1080p 14.4-second route and explicit estimated
starting points for intermediate GPUs and lower host RAM. Studio and Director
show tested, reported, estimated or conservative provenance. Above a starting
recommendation means additional unverified load, rather than an automatic
likely-OOM warning. Manual durations remain available, including the separately
enabled experimental range. See the [resource matrix and evidence](Duration-recommendations.md).

For eligible H3 video jobs on the 12-GB / at-least-30-GiB-usable-RAM tier, Auto
chooses Off, Low or Medium head splitting from the actual generation workload.
Short 1080p and ordinary 720p jobs can retain the faster unsplit route; longer
1080p jobs use more splitting. This applies only while Auto owns that control
and does not rewrite the saved setting. Manual values remain in effect. The
performance card and duration guidance explain the recommendation and flag
unmeasured references/LoRAs and costly long Pruned jobs. One 10.1-second native
1080p Pruned dense-attention test took about 116 minutes; this older warning
is not treated as a sparse-attention ETA for SLA or Veda. Fitting in VRAM does not ensure an
acceptable render time. These observations do not establish comparable limits
on 6/8/10-GB cards or 16-GB RAM systems.

## Optional H3 VAE 2× video decoding

Studio and Director offer a learned H3 decoder that doubles both output
dimensions while keeping denoising at the selected resolution. For example,
960×544 generation produces 1920×1088 output. Its pinned INT8 ConvRot asset
(about 2.8 GB) downloads only when selected. The existing H3 encoder and
reference preparation are retained. Standard and x2 H3 decoding now convert
finalized temporal chunks directly to CPU uint8 frames, reducing the peak
GPU and host memory needed for final video normalization.

This is a separate learned-decoder workflow, not native 1080p denoising at
540p memory cost. It adds decoding work and does not establish upstream
15-second native 1080p claims on every machine. Current Maestro support is
for standard H3 video generation; audio-only, Viggle and frozen-source audio
workflows are excluded. The resolver also handles Hugging Face’s nested first-
download layout and shared model roots, verifying size and SHA-256 before loading
the decoder. See the [guide and model terms](H3-VAE-x2.md).

## Shared runtime compatibility

The memory port includes bounded attention staging/head splitting for supported
Sage2, compatible H3 SLA and Sol routes, shorter intermediate-buffer lifetimes,
automatic INT8 backend dispatch and VAE convolution/activation changes. Sol remains approximate
and can change generated results. Quantized FP8, NVFP4 and GGUF paths retain their
format-specific kernels and compatible fallbacks.

Full and Pruned H3 now expose **SLA Sparse Attention — Experimental** in Studio
and Director for Frames and References. It defaults off and preserves the
selected checkpoint, sampler and step count. Fused retains its default SLA
recipe. Director saves SLA separately from Sol with its project. Sparse
attention can change motion, reference fidelity and audio; compare warmed runs
before adopting it. See the [SLA comparison guide](Performance-auto-tune.md#experimental-sla-for-full-and-pruned-h3).

Explicit-device factories and model placement updates cover eligible Wan, H3,
Flux, Qwen Image, Hunyuan, HiDream, Krea2, Z-Image and audio pipelines. LTX,
LongCat and other routes that still need implicit factories retain their device
wrapper. Kandinsky no longer replaces the selected allocator at import time.
These are updates to existing workflows. The optional H3 learned x2 decoder
is included separately, along with the official Turbo checkpoint for the existing
Qwen Image 2.1 pipeline. The LTX detail refiner is not included in this release. Optional attention imports and CPU
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

New Director projects prefer **H3 Fused 4-Step References** when enabled and
compatible. Existing drafts retain their model. Auto cut speed favors fewer,
longer H3 clips, and Auto maximum clip length follows the same resource policy
as Studio. Explicit cut speed and maximum-duration choices remain in control.
After uploaded audio analysis, Director brings character/reference setup back
into view so it is easier to complete before planning.

Long music-video planning now validates each batch, preserves valid checkpoints
and repairs the invalid tail in smaller batches. Continuity and action-diversity
checks reduce repeated scene descriptions. Native H3 reference workflows keep
reference images as identity/design guidance rather than silently copying them
into opening frames. These checks reduce the observed repeated-action and
reference-still problems; generated stories still benefit from human review.

Ordinary pose changes and camera descriptions no longer create auxiliary
physical-state obligations that stop an otherwise usable music-video plan.
Requested actions, their actors and timing remain checked. Focused repairs retain
authored shots; when bounded writer repair still cannot preserve an assigned
action, Maestro can compile the affected clip from those source actions and
validate it again. Saved metadata and the console identify this recovery.
Final compilation also preserves explicit vocalist direction and avoids
duplicating an authored action in both the shot body and its closing beat.

H3 music planning now supplies each clip's local duration and requests stable
cast identities. Malformed or out-of-range visual timecodes enter bounded
planning repair, with a final check before generation. Reference pictures bind
to the existing visible subject when its identity is unambiguous; ambiguous
matches require a distinct cast/reference name. Reviewed numbered cast
clauses survive recompilation, including characters without their own picture.
Audio fields exclude visual shot markers, and compiler-owned performance
instructions and world anchors are rebuilt once instead of accumulating
through preflight.

Director compilation also removes empty scene-transition shells and repairs
global numbering on timed visual cuts before H3 preflight. Actual scene action,
cut timestamps and exact dialogue remain intact; malformed substantive prompts
still fail validation.

H3 music-video planning now requires one complete prompt per native clip.
Unwanted rolling-window prompts enter focused planning repair before rendering,
preventing separate zero-based timelines from being joined into one clip.

When native frame fitting changes a planned clip's length, Director retains the
authored duration and adjusts local visual cues to the fitted clip. The timing
provenance survives save, resume and regeneration. Manually edited prompts keep
strict timing validation instead of receiving an inferred duration exemption.
Short-film opening states prefer explicit spatial setup over pose-bearing
identity descriptions.

Planner guidance also covers rider scale, anatomy, viewpoint and physical
transitions. Saving a changed prompt preserves its rendered-prompt provenance
and marks an existing clip as needing regeneration.

## Simpler model selection

**Manage models** remains available from every model picker, including References.
Settings groups models under **Video**, **Image** and **Audio**, with expandable
workflow groups such as Frames, References, Avatar, Extend and Blend. Enabling a
model in one workflow does not force it into every other workflow; downloaded
weights remain shared. Small download icons keep rows the same height, and
contextual model names omit redundant mode labels while retaining variants,
steps, model size and quantization. H3 Fused labels no longer say Experimental.

Fresh-install video defaults are:

- **Frames:** H3 Pruned, H3 Fused 4-Step and LTX-2.5 Distilled.
- **References:** H3 Pruned and H3 Fused 4-Step.
- **Avatar:** both LongCat Avatar models.
- **Extend and Blend:** Frames and References variants of H3 Pruned and H3 Fused
  4-Step, plus LTX-2.5 Distilled.

**Reset**, beside **Enable more models**, restores the current workflow's default
visibility after explaining that no model files are deleted. Hidden models can
be enabled again. Existing saved choices are preserved on update.

## WanGP v18.10 additions

**Flux 3 Klein** offers Distilled (16 steps, no CFG) and Base (50 steps, CFG 3.5)
with BF16 and INT8 ConvRot checkpoint choices. Separate NVFP4 entries require
supported RTX 50 hardware and kernels. All components now download on demand
from pinned public WanGP sources, so a Hugging Face login or prior gate approval
is not required. Model license notices remain included. Existing weights are
retained, and ten ordered references remain supported.

**Prompt Helper** adds a visual Flux 3 layout editor for output boxes, reference
source regions and removals. Saved prompts retain the caption, element IDs,
reference bindings and JSON together. Boxes guide composition rather than
acting as exact masks. Reference IDs map Image 1 to `<ref_image_0>` through
Image 10 to `<ref_image_9>`. Enhance preserves supplied layout data and uses
model-specific generation/edit guidance. See the [Flux 3 guide](Flux-3-Klein.md).

**Fish Audio Drama 3** adds expressive multi-speaker speech, up to three cloned
voice references, automatic transcription for missing reference transcripts,
and line-based long-script chunks with recent speech as context. FP8 and INT4
expert variants are optional model choices. Legacy uses the native CPU expert
kernel on supported systems, with a portable PyTorch fallback. The graph-free
vLLM path is offered only when Triton, FlashAttention 2 and the required FLA
kernels are usable; expert GPU staging also checks profile and free VRAM.
CUDA Graphs and partial expert GPU residency are not enabled in this port.
The INT4 expert file is about 16.6 GB; FP8 is about 32 GB before the ordinary
layers and codec. See [Fish Drama 3](Fish-Drama-3.md) for resource requirements.

**WGPKernels** is an optional INT8 backend in RAM / VRAM Management. Normal
Install/Update attempts to install the pinned wheel matching CUDA 12.8 or 13
on supported Windows/Linux x64 RTX 30-series or newer systems. Auto verifies
available kernels and tries WGPKernels, Kitchen, Triton, then PyTorch. Explicit
selection reports an unavailable backend rather than pretending to use it.

**H3 LightVAE** adds BF16 and INT8 ConvRot 12-layer decoder choices under
Advanced → H3 Video Decoder. Start/end frames and references retain the original
H3 encoder. Auto retains the established full decoder behavior; LightVAE can
change decoded detail. H3 VAE x2 selects the dedicated x2 decoder.

**TaoMate FL2VA 3-step, 2026-10-15** is a separate dated accelerator recipe for
Frames models. Its LoRA downloads only when selected. Existing recipes remain
available; the new FL2VA adapter is excluded from References models.

The earlier official BF16 Flux 3 adapter passed real 4090 text-to-image,
single/two-reference and ten-reference capacity tests, plus a full-resolution
numerical comparison against BFL. Those results apply to their recorded
workloads; new variant and lower-memory evidence is tracked in the
[validation record](VALIDATION_V3.0.0.md).

## Qwen Image 2.1 Turbo

Official **Qwen Image 2.1 Turbo 7B** is a separate image generation/editing option,
adapted from [PR #175](https://github.com/Blizaine/Maestro/pull/175). It uses the
published eight-step schedule with CFG 1, shares the existing 2.1 text encoder
and VAE, and offers BF16 or INT8 ConvRot checkpoints through normal on-demand
downloads. The setting and saved output metadata enforce the Turbo recipe,
including when older queued parameters requested another step count. Ordinary
compatible style LoRAs remain available; incompatible managed acceleration
recipes are kept separate.

The option is enabled once on update without selecting it or downloading its
weights. A user's later decision to hide it persists. Real INT8 text-to-image
and reference-edit checks passed on the 24-GB workstation. This does not verify
every image size, BF16, mask workflow or lower-memory machine. Qwen's research
and evaluation model license applies; commercial use requires a separate
license. See the [Qwen guide and terms](Qwen-Image-2.1.md).

## Prompt writing and local LLM startup

The local writer picker adds **Underdog Saluki 27B IQ2-Mix Uncensored**,
with the upstream Q8 vision projector, a 32K context and Qwen thinking controls.
Its weights are approximately 8.35 GB, plus 0.63 GB for vision. Downloads occur
on first use; Gemma remains the default. **MTP acceleration — Experimental**
is an optional, default-off local writer setting. It applies only to verified
MTP-capable weights and takes effect on the next load, with actual mode shown
in Settings. Current work continues when the preference is saved. MTP may
improve speed but uses additional memory and needs quality/stability testing.
The 12-GB picker estimate is a starting hardware estimate, not a measured fit
or quality result.

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

Speaker attribution also handles possessive listener references, bounded
response cues and parenthesized voice IDs more carefully. Source fallbacks
avoid duplicate voice markers. Continuity instructions do not become cast
names, and an adjective such as an open palm is distinguished from an opening
action. Fresh exact-dialogue checks with both local writers completed without
fallback; creative dialogue still has recorded failures and repair overhead.

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

Director music videos now retain explicit fights and battles alongside vocal
performance. Both opponents remain part of the required visual action; continuing
combat can develop after its first assigned beat. A song merely mentioning an
antagonist does not require that character onscreen. Missing combat anchors can
be restored from the source concept while keeping the other planned shots.

YuE2 song generation now downloads from a replacement pinned snapshot after the
upstream TTS repository consolidated its history. The required files and recorded
weight hashes match the previous manifest; existing weights and saved artist
styles remain compatible. Missing companion assets still download on first use.

To test v3.0.0 on Dev, use **Update** in Pinokio while idle, restart Maestro
to load the new backend, and refresh the browser. When Performance Auto is
enabled, revised recommendations apply once at startup while preserving manual
customizations. **Settings → Performance → Re-detect** remains available to
explicitly reapply Auto recommendations. Check
the active allocator and restart if another allocator change is pending. Some
optional assets download on first use.
The new memory paths have fallbacks, but their behavior and performance depend on
GPU architecture, driver, PyTorch/CUDA runtime, checkpoint format, attention,
precision, resolution, duration, available RAM and other running programs.
Compare complete generated media as well as runtime and memory measurements.

The upstream WanGP release's speed claims are not Maestro measurements. The
current evidence covers selected candidate-build comparisons on the documented
machines; it is not a universal speed, OOM or output-quality guarantee.

The [October 13 Dev CI](https://github.com/Blizaine/Maestro/actions/runs/38480363663)
ran 4,042 backend tests successfully (57 skipped) and passed 137 standalone
tests (one skipped), source-boundary, syntax, undefined-name and JSON-grammar checks, plus frontend
lint, TypeScript and the production build. Local browser regressions cover the
latest Editor and clip-restore behavior. Selected earlier installed video,
image, audio and x2-decoder smokes also passed.

The remaining Main gates are human review of the latest Director motion,
lip-sync and scene progression; a fresh silent/action-only short-film check
against the earlier speech/music, prop and transition findings; and an idle
v2.6.0-to-candidate update/restart smoke. Veda remains experimental, and the
reported flash/motion change still needs a matched attention comparison.
Creative enhancement can still need source fallback or fail validation on a
particular draft. Full media decoding and passing text tests do not establish
story fidelity. See the [readiness record](VALIDATION_V3.0.0.md#release-readiness--october-13)
for the evidence and remaining installation/device limits.

[Validation and limitations](VALIDATION_V3.0.0.md) · [Changelog](../CHANGELOG.md)
