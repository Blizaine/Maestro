# Maestro v2.1.2

Released 9 September 2026.

This patch fixes H3 Omni enhancement failures, incorrect speech detection and
reference-audio roles, along with speaker assignment and two video-tool errors.

## H3 Omni prompt enhancement

- Fixed `prompt enhancement failed: name 're' is not defined` (issue #102).
  A missing import could fail the request after the LLM successfully responded,
  particularly when the original request contained no dialogue. The regression
  checks cover full, pruned and fused Omni models in Faithful and Creative modes.
- Malformed AI-written speaker assignments now reach the validation and repair
  retry instead of aborting enhancement immediately.
- Explicit requests for music only or no dialogue reject unwanted AI-written
  speech through the existing retry and fallback path.

## Descriptions, dialogue and audio references

- Quoted descriptions such as "older woman", "English speaker", character
  names, visual styles and reference metadata are no longer automatically
  converted into spoken lines. Speech cues and explicit `<d>` dialogue remain
  supported throughout enhancement, reference preparation and story planning.
- Quotations inside existing `<d>` blocks retain their wording without creating
  nested speech tags.
- Explicit audio roles take priority over descriptive text. A music/style
  reference containing the word "voice" no longer becomes a character voice.
  Silent windows retain music references without selecting voice references
  because of quoted visual labels.

## Speaker assignment

- Added support for attribution after a line, such as `"Go," Alex says`, while
  keeping a following character's next turn separate.
- Named guests no longer inherit a preceding saved character's identity and
  voice through backward context matching.
- Speaker IDs are validated separately from saved reference IDs, allowing
  scenes with more speaking characters than saved character references.
- Replaced the hard-coded Yoda example in the ambiguous-speaker error with a
  generic example. That diagnostic never read the user's character library.
- Ambiguous user-authored dialogue still requires a named speaker or an
  explicit `<Subject N>` beside the line when multiple references are present.

## Other stability fixes

- Blend now calculates video strength before using it in its generation setup.
- Video inpainting reads source dimensions before optional mask downscaling.
- Corrected a missing Director annotation import and added CI checks for
  undefined names and local variables used before assignment.

## Updating and validation

Use **Update** in Pinokio, restart Maestro and refresh the browser. Enhance
affected prompts again before generating; updating does not rewrite prompts
already saved in projects or queued jobs.

See the [validation record](VALIDATION_V2.1.2.md). Automated tests exercise the
real prompt-processing modules with synthetic inputs and mocked LLM responses;
they do not establish generated audio/video quality or reproduce every reported
prompt exactly.
