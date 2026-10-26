# Maestro v3.2.0

22 October 2026. Public Main and Dev release, including the 20–22 October Dev
preview fixes. FastH3 models retain their experimental status.

## Dev follow-up fixes — 23–24 October

- **Fish smaller-memory acceleration:** reusable single-frame expert scratch,
  packed GPU expert caching and a partial audio-tower graph. Honor declared
  vLLM offload support, preserve CPU/eager fallbacks and report the actual
  backend. Kernel/cache checks passed on 24GB, 12GB and 8GB GPUs; complete
  speech on the smaller cards remains pending setup. See [Fish details](Fish-Drama-3.md).
- **Fish whole-frame CUDA Graphs:** eligible vLLM profiles replay the codebook
  tower, sampling and next backbone step together. Capture runs once per speech
  part, with the prompt caches and random state restored after warmup. All
  decode weights and expert banks must fit on one GPU; offloaded profiles and
  unavailable capture retain eager decoding. Fix a traversal recursion caused by
  MMGP's parent references, including with no LoRAs selected. One matched RTX 4090
  short take decoded in 1.66s versus 26.76s with identical audio samples; total
  job time was 21s versus 45s. This is a single-take result, not a general speed claim.
- **Fish decoding overhead:** remove redundant expert-ID bound checks from
  the internal GPU router path, avoiding eighty CUDA scalar reads per decoded
  frame. Direct expert calls retain validation and CPU math is unchanged.
  A GPU speedup from the bound-check change alone is not yet measured.
- **Long-form speech:** saved writing instructions control the speech writer's
  personality and style. Long story/dialogue drafts are assembled from bounded
  sections; requested runtime is approximate. Fish automatically chunks long
  scripts and assembles one file, retaining incomplete-part warnings and a
  whole-output limit up to one hour. See [speech writing](Speech-writing.md).
- **Fish speech length:** Auto lets the model finish naturally under a generous
  script/context safety budget and a one-hour total bound. Maximum length
  remains optional, and short manual limits now reach generation unchanged.
  Saved positive limits are preserved. Auto does not force pacing or remove
  model-generated pauses; these changes have CPU and mocked browser coverage.
- **Fish speech sampling:** Top-k 0 correctly removes the sampling cap rather
  than becoming greedy. New Fish Drama 3 defaults use no cap, matching the
  mitigation in Fish's demo for speech that falls silent after a short opening.
  Advanced Generation exposes Top-k and Top-p, preserving saved positive values.
  Completion and voice fidelity still need confirmation with an actual render.
- **Eros Max beta6:** validate its mixed dense/INT8 layout and quantized
  condition, output and refiner projections. The known file's published
  identity/checksum selects both Frames and References, retaining the FL2VA
  basis and creator's baked Turbo recipe: Euler, 8–12 steps, shifts 10/3.
  Quantized output heads preserve their FP32 compute lock; BF16 CPU forwards
  retain the original FP32 scale storage. These paths have numeric CPU coverage.
- **One H3 import, both modes:** clarify the checkpoint's native-basis selector
  and enable newly imported Frames/References companions even when old scoped
  visibility defaults contain their IDs. Both entries share one weight file;
  existing visibility choices for known models remain unchanged.
- **16GB H3 placement:** for Auto-owned native H3 video on Windows 16GB
  Blackwell / nominal 64GB RAM hosts, use Profile 4 and retain a feasible
  guard-approved residency allowance instead of the extra 3 GiB ceiling.
  Manual placement remains authoritative; clamped jobs retain streaming
  defaults and other hardware tiers keep their existing policy. CPU regressions
  cover the guard-to-loader handoff, cached-model reload/reuse and restoration.
  The reported RTX 5080 slowdown has not been reproduced or benchmarked on that
  machine. See [the diagnosis and reporter test](H3-16GB-placement.md).
- **Eros Max import:** support the beta5 file's mixed W4A8/INT8 ConvRot layers
  and per-layer weight groups of 4, 8 and 16, while retaining strict verification.
  Let both quantization handlers load their own layers in the same model.
  Accept finite BF16/FP16 AdaLN tables for the native FP32 interpolation path.
  Bounded descriptor reads retry a transient failure once and preserve specific
  transport/access errors in the browser. The real file's header and markers
  were inspected; a complete weight download and GPU render were not performed.
  See [supported formats and verification limits](H3-checkpoint-import.md).
- **Concurrent soundtrack preparation:** on Windows, reuse an already published
  nonempty WAV when another window's reader prevents replacing it. Preserve
  atomic publication and temporary-file cleanup; permission errors still fail
  when no usable cache exists.

## Dev follow-up fixes — 22 October

- **UI startup:** prioritize service preferences, model controls and saved
  choices before gallery scans and remote writer discovery. Load Settings,
  browsing, dashboard and editor panels on first use. The initial JavaScript
  bundle is about 24% smaller; this is a payload measurement, not a measured
  reduction in cold-start seconds. Compress large catalogue/options JSON and
  hashed JavaScript/CSS without changing generated media or event streams.
- **Startup recovery:** share concurrent pending reads, bound request waits and
  retry temporary failures. Show **Loading model controls** and a visible Retry
  action instead of leaving controls empty after a restart. Recover on browser
  return/reconnection while preserving prompts, edited settings and saved
  preferences. A failed preferences read cannot overwrite the server record.
- **Mobile sidecar:** distinguish an actual keyboard from browser toolbar and
  restored-tab viewport changes. Keep modes, workflows, references and hardware
  visible when no text entry is active; restore them on blur and page return.
  Preserve keyboard compaction while moving between text fields.
- **Gallery startup:** display **Loading workspace** through saved-folder
  restoration and the first media listing, closing the blank interval after
  model controls load. Preserve folder selection and reject stale listings
  during a folder switch. This changes loading feedback, not scan throughput.
- **H3 reference validation:** stop treating motion-video retention entries as
  additional characters, avoiding false rejection of valid source-video prompts.
- **Rejected writer drafts:** save drafts and specific rejection reasons locally
  for Studio and Director, including failed or interrupted enhancement requests.
  Review them through Studio prompt history or the Director Dashboard.
- **Multi-window recovery:** preserve accepted windows during focused repairs,
  retain usable camera cards if a repair request fails, and keep incomplete
  schedules reviewable. Do not replace multiple windows with a repeated copy of
  the original prompt. See [H3 reference audio and drafts](H3-Reference-Audio-and-Drafts.md).
- **Video soundtrack roles:** give source videos the audio-role choices available
  to audio references in Studio and Director: preserve soundtrack, voice,
  music/sound style, sound effect, ignore, or the previous generative reference
  behavior. Preserve the meaning of older saved checkbox settings.
- **Director style and pacing:** carry the shot's visual style through the full
  H3 frame and preserve it across saved projects, resume, native timeline splits
  and clip reruns, with edited prompts remaining authoritative. Improve
  instructions for visible performance, timed stillness and matching loop
  composition/camera position. These improve prompt handoff; they do not guarantee
  that every rendered frame follows the requested style, motion or loop.
- **Director source actions:** stop turning choreography fragments such as
  "dancing with spins" or the requester in "I want" into mandatory events.
  Preserve genuine first-person actions and the original creative brief.
- **Additional download repairs:** restore fifteen MiniMax-Music3 optimized
  downloads, three local-writer vision projectors, the hidden Z-Image eight-step
  ControlNet BF16 export and the YuE2 model-weight license link. Reuse compatible
  cached files and retain pinned-source verification.

## Dev follow-up fixes — 21 October

- **H3 VDN downloads:** repair Full/Pruned VDN modules, default LoRAs and the
  bundled eight-step Turbo preset after the old Hugging Face revision stopped
  serving them. Redirect four known obsolete URLs retained in saved presets
  and reuse existing local files.
- **Director source actions and outline contracts:** keep timeline-formatting
  instructions, schema examples and native identity/retention/soundscape metadata
  out of the mandatory story-action ledger. Preserve actual narrative actions,
  including JSON `visual_beat` text, and keep native identity definitions available
  for participant resolution. The sixteen-action planning bound remains; genuine
  actions are not silently truncated. Make the compact-outline stage explicit to
  local and external writers, and keep optional repair examples consistent with
  the requested event-assignment schema. New planning invalidates older cached
  results. Users can supply a normal creative brief without maintaining an internal
  JSON template. Clarify that the displayed minimum clip count applies to the
  selected track and maximum clip length, rather than a twelve-clip requirement.
- **Explicit runtime upgrades:** opening an installed launcher with missing
  performance-runtime markers no longer automatically starts Update and its
  `git pull`. Keep the existing compatibility-start option and a running app's
  Web UI accessible. Updating remains an explicit user action. Director and
  launcher regressions passed 292 CPU checks; this does not certify a live run
  with the reporter's Gemma/LM Studio configuration.
- **Viggle Animate trim:** use the video's measured picture duration for the
  timeline and submitted range. A soundtrack that ends later no longer makes
  an untouched full-video selection fail as out of range. Check restored sources,
  invalidate old bounds when the source changes in any workflow, and verify
  ranges before character preparation. Existing full-file selections can be
  reconciled against measured audio tails; genuinely invalid ranges still fail.
- **Face Refiner (#188 / PR #189):** clone H3 inference-mode uint8 windows
  before in-place overlap blending. This retains the earlier color-range fix
  while allowing clips longer than one refinement window to finish. The fix
  follows [glenndav's PR #189](https://github.com/Blizaine/Maestro/pull/189);
  a CPU two-window regression checks blending, source preservation and export.
- **Optional finishing (#190):** a failed Studio DLSS/RIFE finishing stage keeps
  the generated original and reports a non-fatal warning in completion notices,
  completed queue entries and the output processing metadata. Warnings survive
  browser reconnection. Cancellation remains cancellation; standalone Tools and
  Media Flow processing failures remain failed jobs.
- **DLSS availability (#190):** installed files alone no longer establish
  Windows 11 Neural Rendering availability. A bounded, cached synthetic-frame
  evaluation now exercises an unaligned 1.5x request through the same padding
  and crop path as processing. Failed probes disable the option with a reason;
  refresh reruns the check while idle. A small passing probe does not certify
  every scale or clip. The native binary pins are unchanged. [DLSS details](DLSS5.md).
- **DLSS output alignment (#190):** apply internal right/bottom edge padding to
  the standard Windows 11 backend for the reported 64-pixel output constraint.
  Pad the image, motion and depth together, then crop back to the requested
  even-rounded dimensions. Both requested and padded sizes must fit the native
  limit. The separate Windows 10 experimental path keeps its existing policy.
  CPU regressions cover the reported sizes, guides, cropping and failure cleanup;
  the workaround still needs native confirmation on the reporter's RTX 5090.
- **WGPKernels and Auto-tune (#191):** split activation preparation into bounded
  calls below the native 65,535-row limit, preserving row order and projection
  reuse. Reapplying Auto-tune keeps a valid selected manual INT8 backend and
  reports the actual applied settings. CPU regressions cover boundary-sized
  inputs and longer sequences; these are not full RTX 5090 render benchmarks.
- **Qwen processor downloads (#192):** the repair already included in Dev uses
  a replacement immutable revision and canonical local filenames. All ten
  replacement processor files returned public HTTP 200 during review. Existing
  fresh-install and cached-download regressions pass; no model weights were
  downloaded for this verification.
- **External LLM Director planning (#193):** accept unambiguous wrapped music-arc
  responses instead of mistaking a JSON object container for one timeline row.
  Make the fixed timeline and output object explicit, and use bounded corrective
  retries for malformed responses. This music-arc call opts into the
  OpenAI-compatible strict JSON Schema format, with one unconstrained fallback
  when the endpoint explicitly rejects that format. Other remote calls keep
  their existing behavior. Duplicate/missing groups and contradictory
  source-action assignments remain errors; the fix does not invent missing beats
  or turn off continuity checks.
- **Outline compatibility:** retain the complete required response format in
  writer prompts when an endpoint cannot enforce structured output. Accept
  unambiguous row-local `beat` and `prompt` aliases for `visual_beat`; conflicting
  aliases, incomplete rows and invalid timeline/action assignments still fail.

## FastH3 V2 and OmniRef

- Enable **FastH3 V2 8-Step — Text to Video (Experimental)** in the Frames model picker for
  text-to-video with generated audio. It uses its own fixed eight-step DMD
  schedule and trained VSA-H3 attention.
- Enable **FastH3 OmniRef 8-Step (Experimental)** in References for ordered image,
  video and audio conditioning. At least one image or video reference is required.
  Its eight evaluations use 32 PDD output heads and a separate sparse-attention
  pattern. This is a different model and recipe from V2 and H3 Fused 4-Step.
- The normal OmniRef model offers official BF16, INT8 ConvRot and FP8 checkpoints
  through the Transformer Quantization setting. Existing verified INT8 downloads
  are reused because the official file is byte-identical to the earlier upload.
- A separate **FastH3 OmniRef QAD NVFP4 8-Step (Experimental)** option uses the
  QAD-trained Comfy export. It requires supported NVIDIA Blackwell hardware and
  a working native NVFP4 kernel; unsupported systems receive an actionable error.
- Checkpoints use immutable source revisions, exact size/SHA-256 verification,
  format-specific geometry checks and the matching grouped-QKV layout. Retained
  FP32 output heads and AdaLN projections keep their source precision.
- Start at **480p / 5.2 seconds**. OmniRef does not support Director, continuation,
  start/end frames, masks, additional LoRAs or ordinary H3 audio refinement.
  FastH3 uses its own attention instead of SLA or Veda. Existing H3 hardware
  duration tiers are not treated as verified FastH3 limits.

The prior INT8 OmniRef 4090 image-reference smoke test completed an audio/video
render in 86 seconds of active generation. The new format regressions passed,
and real FP8/NVFP4 compression gates executed on an RTX 5070 Laptop GPU, with
NVFP4 using its native kernel. Those layer checks do not verify a complete
BF16/FP8/NVFP4 render, output fidelity or consumer-GPU throughput.

FastVideo's published near-real-time OmniRef benchmarks used **eight GB200 GPUs**.
Its 5090 and 4090 launch recipes do not establish comparable speeds in Maestro.
Maestro uses BF16/FP16 VSA and does not yet include FastVideo's full FP4-attention
and NVFP4-decoder acceleration stack. See the [FastH3 guide](FastH3.md) for sources,
formats, hardware requirements and validation scope.

## Qwen Image 2.1 downloads and VAE decoding

- Repair processor assets after upstream repository moves. Generation and prompt
  enhancement use a verified replacement revision and canonical filenames;
  existing compatible local assets are retained.
- Omit unused temporal feature caches during single-image decoding to reduce
  final VAE memory use. Low memory use during numbered transformer steps does
  not guarantee that final decoding fits the same budget.
- Address thin tile-boundary lines using full-sized tiles at image edges,
  at least 512px decoder context in Medium/Aggressive modes, overlap and FP32
  CPU blending of reliable tile interiors before one final pixel conversion.
  Increased overlap can add decode work; it does not repeat denoising.
- In a decoder-only comparison at 1088×1920, Full peak allocated VRAM fell from
  6.85GiB to 3.26GiB on the 4090. An isolated 5070 Laptop decoder check also
  completed at 3.26GiB allocated and 4.22GiB reserved. These are decoder checks,
  not proof that every resolution, reference count or complete render fits 8GB.

See the [Qwen 2.1 guide](Qwen-Image-2.1.md) for workflow and measurement details.

## Updating to v3.2.0

1. Let current work finish, then stop Maestro and use **Update** for your
   installation in Pinokio.
2. Restart Maestro, refresh the browser and confirm **v3.2.0** beside the title.
3. Enable the desired experimental FastH3 model through **Manage models**.
   Weights download when requested; saved model visibility and user preferences
   are retained.

This release is available on public Main and Dev. Existing user duration
recommendations are unchanged; experimental model limits remain as described above.
