# Maestro v3.2.0

20 October 2026. Public **Dev** preview for testing. Main remains v3.1.0.

## Dev follow-up fixes — 21 October

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

## Updating the Dev build

1. Let current work finish, then stop Maestro and use **Update** for your Dev
   installation in Pinokio.
2. Restart Maestro, refresh the browser and confirm **v3.2.0** beside the title.
3. Enable the desired experimental FastH3 model through **Manage models**.
   Weights download when requested; saved model visibility and user preferences
   are retained.

This push updates public Dev only. It does not publish a stable GitHub release,
move Main or change existing user duration recommendations.
