# Maestro v2.4.2

September 27, 2026. Changes since v2.4.1.

## Gallery audio and clip trimming

The gallery **More** menu now sends audio to compatible inputs in the active
sidebar. This includes Director music/dialogue and voice references, Studio
soundtracks and audio references, individual Speech voices, Revoice/SeedVC,
mixer tracks and YuE2 source songs when those inputs are available.

Choosing an audio or video input opens a trim preview:

- Use the audio waveform or video filmstrip to drag the start and end handles.
- Enter exact times or choose a **3-, 5- or 10-second** excerpt.
- **Preview selection** plays the chosen range; **Zoom to selection** makes
  short excerpts from long sources easier to adjust.
- **Use selection** creates a separate excerpt. **Use full clip** skips trimming.

Original media stays unchanged. The source folder remains attached when browsing
All folders, and Maestro checks that the chosen destination is still available
before sending. Each input keeps its normal format and size limits. Gallery
image and current-video-frame actions continue to work directly.

## Krea 2 Identity Edit

Both RAW and Turbo Identity Edit expose controls under **Advanced → Generation**:

- **Subject likeness:** 0–10, default 1, with a **Strong likeness · 4** shortcut.
- **Scene likeness:** separate control for two-reference edits. Image 1 is the
  scene and Image 2 the subject; with one image, Subject likeness controls it.
- **Grounding resolution:** 384–1536 pixels, default 768, controlling the longest
  reference edge sent to the vision encoder. This is separate from output size.

RAW and Turbo remember their values independently across model switches and
restarts. Queued jobs, saved presets, Media Info and **Load settings** retain
the recipe so comparisons can use the same prompt, references and seed.

Reference preparation now preserves the uploaded image's aspect ratio before
grounding. A portrait reference is no longer padded into a landscape canvas
before the vision encoder sees it. Reference latents fit the output grid without
stretching. Other models and explicit outpaint canvas handling retain their
existing behavior.

Turbo's guidance has always been disabled in the pipeline; Maestro now records
zero consistently and hides the ineffective control. RAW guidance remains
adjustable.

Higher likeness can resist edits or copy too much of the reference. Grounding
above the trained 384–768 range is experimental and uses more memory; smaller
source images are not enlarged. These controls do not guarantee a particular
degree of likeness. See [Studio controls](Studio-controls.md#krea-2-identity-edit).

## Maintenance and update

This release includes the already-public Linux DLSS protocol-test portability
hotfix that followed v2.4.1. It does not introduce another DLSS runtime change.

Use **Update** in Pinokio, restart Maestro and refresh the
browser. Existing models, LoRAs, projects and media stay in place.

[Validation and limitations](VALIDATION_V2.4.2.md) · [Changelog](../CHANGELOG.md)
