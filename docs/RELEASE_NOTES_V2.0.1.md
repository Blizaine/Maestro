# Maestro v2.0.1

Maestro 2.0.1 is a focused stability and workflow-fidelity update. It fixes a
critical interface startup regression, makes Video Extend's duration promise
match the work actually queued, and turns gallery **Load Settings** into a real
round trip for the full Maestro 2 workflow surface.

## Critical interface fix

- Fixed the black or blank interface reported in GitHub issue #97.
- The cause was a circular LTX state update: Auto duration derived a timeline
  from the current native window while the native window simultaneously tried
  to follow that generated duration. Some restored/default LTX-2.5 states could
  make enough synchronous updates to trip React's maximum-update-depth guard.
- A one-window Auto plan now stays on its stable model default. When the story,
  soundtrack, or requested runtime genuinely needs multiple windows, Maestro
  jumps directly to the safe long-window ceiling instead of walking there in a
  render loop.
- Explicit **Time** and **Windows** planning retain their existing behavior,
  including user-saved window overrides.

## Video Extend fixes

- One selected continuation window now queues exactly one generation pass.
- The first pass subtracts only the source-tail frames used as continuation
  context; those frames are no longer counted as newly generated duration.
- Duration presets, direct window counts, prompt boxes, queue validation, and
  backend routing share the same continuation-aware geometry.
- The duration summary explains how much fresh footage the first pass adds and
  how far each later sliding-window pass advances.
- AI Faithful no longer reports an invented large spoken-word count when the
  user's extension contains only a short line of dialogue.

## H3 prompt integrity

- Dialogue parsing no longer lets a bare instructional opening `<d>` token
  swallow all later scene prose before a real tagged line.
- Malformed or nested dialogue markers are bounded at the next opening marker,
  allowing the real line to be recovered without treating camera, timing, and
  soundscape instructions as speech.
- `integrated_multimodal_description` and the other Context-IR field labels are
  explicitly excluded from screenplay-speaker inference.
- If a user enhances one H3 First/Last prompt and Auto later expands it to
  multiple windows, the planner uses the preserved original story prompt. It
  no longer treats the enhanced Context-IR document as a new source screenplay.

## Complete Load Settings round trips

The gallery pencil now restores the workflow that actually created an output,
not merely its prompt and model. Restored data includes the relevant sources,
references, masks, timing, LoRAs, optimizations, and specialized controls for:

- Video Frames and References;
- Extend and Blend;
- Retake, Prompt Edit, Inpaint, Outpaint, Repaint, and Recast;
- Image Generate, Edit, Inpaint/Outpaint, and finishing workflows;
- Music, Speech, Sound Effects, Revoice, and Audio Mixer;
- Upscale and Film Grain;
- Director revisions; and
- Editor project exports.

Large source videos are restored as lightweight named browser objects and
continue streaming from Maestro instead of being copied into browser memory.
Older sidecars fall back through their legacy fields. Loading a new item also
clears incompatible multi-window, reference, TTS, and edit state left by the
previous item.

## Better output records and gallery details

- Generation sidecars now retain scalar and multi-file source names for image,
  video, voice, mask, anchor, blend, and edit inputs.
- Blend, Retake, Inpaint, Outpaint, Recast, Repaint, Revoice, and Mixer retain
  the settings necessary to reopen their UI faithfully.
- Mixer outputs now write metadata and refresh into the gallery immediately.
- Expanded details show generation time for a normal single-window clip. When
  active timing is available, it excludes queue wait and model loading.
- **Original Prompt** now has its own Copy button, independent of the effective
  or AI-generated prompt.

## Background notification delivery

- Fixed completed and failed jobs not reaching enrolled iPhone, iPad, or
  desktop Web Push subscriptions even though foreground tests worked.
- Maestro now parses its persisted VAPID PEM key explicitly before signing a
  push and uses a public-domain contact claim accepted by Apple's push service.
  Existing device subscriptions, permissions, and Tailscale Home Screen apps
  remain valid; users do not need to enroll again.
- Notification settings now distinguish the foreground-only test from the
  closed-app test, so a local browser alert cannot be mistaken for proof that
  background delivery is working.

## Release validation

- Added regression coverage for startup duration stability, continuation-window
  math, H3 dialogue parsing, source-prompt preservation, sidecar contracts, and
  workflow restoration.
- Expanded GitHub Actions dependencies so the complete release suite can run
  portably with CPU PyTorch.
- Attention capability discovery now uses a safe CPU-only sentinel when CUDA is
  unavailable, without changing the GPU runtime path.

## Updating

Use **Update** from Maestro's Pinokio page, then start Maestro normally. Models,
outputs, workspaces, presets, Director projects, and Editor projects are
preserved. If a browser still has the v2.0.0 bundle open, perform one hard
refresh after Maestro restarts.
