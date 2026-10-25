# Maestro v3.2.5

24 October 2026. Public Main and Dev release of the updates tested on Dev since
v3.2.0. Existing models, outputs, projects and saved manual preferences remain
available.

## Fish speech generation

- **Whole-frame CUDA Graph decoding** accelerates full Fish when its dense
  decode weights and complete expert banks fit on one GPU. Capture restores
  prompt caches and random state after warmup. Offloaded or ineligible profiles
  retain eager decoding.
- **Smaller-memory acceleration** adds reusable expert buffers, packed GPU
  expert caching and a partial audio-tower graph. Memory guards retain GPU and
  system-RAM headroom; low available RAM can keep experts on CPU. Supported
  RTX 30-series runtimes now use Fish-specific attention and kernel checks.
- **Fish Drama 3 Small** adds the community English/Japanese derivative through
  a pinned, verified audio.cpp runtime. It supports voice references, expert
  offload and bounded streaming synthesis assembled into one completed gallery
  WAV. This is internal streaming, rather than live browser audio playback.
  Its runtime and model download only when requested; full Fish remains available.
- **Auto length** allows Fish to finish from its script under bounded safety
  limits. An optional manual maximum, including short limits, is preserved.
  Duration is a cap rather than a pacing control. Long scripts are chunked and
  joined, with incomplete-part reports and a one-hour whole-output bound.
- **Sampling and conditioning** correct Top-k 0 to mean no cap, expose Top-k and
  Top-p, preserve native paired tags and record effective per-part conditioning.
  Recent complete-turn history can help consistency, but does not guarantee an
  identical voice register or delivery between chunks.

See [Fish Drama 3](Fish-Drama-3.md) for setup, controls and runtime details.

## Speech writing and prompt enhancement

Saved Speech writer instructions let you request a personality or writing style.
The approximate draft target follows the selected duration unless the brief or
a positive word-count override supplies a target. Long stories and dialogues
use bounded planning, section drafting, continuation and focused expansion.

Fish guidance now includes inline emotion, delivery and word-level cues, with
syntax-aware cuts and conservative performance edits. Repeated storytelling
phrases no longer cause the generic repetition cleaner to discard a complete
story's ending. Incomplete drafts, rejected edits and available partial repairs
remain reviewable with specific reasons. Requested duration remains approximate;
local writers can still produce short, incomplete or poorly tagged drafts.

See [Speech writing](Speech-writing.md).

## H3 checkpoints and placement

- **Eros Max beta5:** accept the verified mixed W4A8/INT8 ConvRot layout and
  supported per-layer groups while retaining descriptor validation. Bounded
  descriptor reads retry transient failures once and retain useful access,
  network and range errors.
- **Eros Max beta6:** validate retained BF16 main layers plus quantized auxiliary
  projections. The recognized file identity/checksum selects its verified
  hybrid basis and creator's 8–12-step Euler recipe. Quantized output heads
  retain FP32 compute locks and scale storage.
- **One compatible import, both modes:** Ref2VA and verified hybrid imports can
  enable Frames and References using the same downloaded file. Existing opt-outs
  are preserved. This does not make every Frames-only checkpoint compatible
  with References.
- **Windows 16GB RTX 50-series / 64GB RAM:** Auto-owned native H3 jobs use Profile
  4 and retain the workspace guard's feasible residency allowance instead of
  an additional 3 GiB ceiling. Manual placement, clamped-workload protection and
  other hardware tiers retain their policies.

The reported RTX 5080 slowdown has not been reproduced on matched hardware.
The placement correction addresses identified policy differences; it is not a
measured claim that every affected render recovers its old speed.
See [16GB H3 placement](H3-16GB-placement.md) and
[checkpoint imports](H3-checkpoint-import.md).

## Director reliability

H3 story planning can recover repeated source-action allocations and missing
physical-continuity metadata through at most two compact repair requests after
the normal whole-plan repair. Accepted story content, timing, dialogue and
camera instructions are preserved. Strict validation still stops plans that
cannot be repaired; the screenshot case has CPU regression coverage, rather
than a reproduction with the reporter's exact writer and inputs.

Concurrent soundtrack preparation on Windows now reuses an already published,
usable cache WAV when another reader prevents replacement. Missing or unusable
cache entries still raise an error; temporary files remain request-specific.

## Validation and limits

- Full Fish and Small short speech tests completed on the RTX 3080 Ti 12GB and
  RTX 5070 Laptop 8GB, with CPU decoding and transcript checks. Small's final
  public-Dev tests produced 12.63 seconds of audio in 39 seconds of active
  generation on the 12GB machine, and 10.54 seconds in 32 seconds on the 8GB
  machine. These are individual runs, not matched speedup ratios.
- One matched full-Fish RTX 4090 take decoded the same 131 frames in 1.66 seconds
  with the whole-frame graph versus 26.76 seconds eager; decoded samples matched.
  Total job time was 21 versus 45 seconds. The decode-stage ratio is not a
  whole-application speed claim.
- Cold setup, model loading, available host RAM, expert placement and script
  delivery materially affect results. The short speech tests do not establish
  long-story completeness, emotion quality or cloned-voice fidelity. Linux
  Small generation and matched RTX 5080 placement remain unverified.
- CPU regressions cover the native result-report path for both full Fish and
  Small, Director metadata recovery, speech drafting, quantized checkpoint
  handling and memory-policy handoffs. UI lint, type checks and production
  build remain part of release CI.

## Update

When the installation is idle, stop Maestro in Pinokio, select **Update** on
Main or Dev, restart Maestro and refresh the browser. The UI should show
**v3.2.5**. New optional Small assets download when selected; installed full
Fish models and saved settings remain available. No reset is required.
