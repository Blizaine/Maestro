# Maestro v2.5.0

September 28, 2026. Changes since v2.4.2.

## A clearer LoRA library

LoRAs now use creator titles where available. Use the pencil beside a LoRA to
choose your own display name, or reset it to the automatic name. Studio,
Director and My LoRAs share these labels. The underlying filename, activation,
weights and update identity remain unchanged.

Separate version and variant labels help distinguish releases with the same
title. Age labels and New sorting use release dates when available, then
download/file dates. Editing names or generating guides no longer makes an old
adapter appear newly downloaded. Existing metadata supplies these improvements
without requiring a library-wide guide regeneration.

## LoRA-aware prompt enhancement

Active adapters' usage guides and creator-declared trigger words are supplied
to the prompt writer, including H3 Frames and References and Director prompts.
Only selected adapters contribute context. External notes are bounded and kept
separate from the user's scene and its continuity requirements.

CivitAI and Hugging Face downloads automatically save available creator guidance
without loading an LLM or waiting for the GPU. Manual AI guide generation stays
available. A LoRA without creator notes or trigger metadata cannot supply
information the creator never provided.

Compatible H3 LoRAs with flattened CivitAI module names now load against their
matching model targets. This addresses the reported unexpected-module-key errors
for that naming format. It does not make adapters for a different architecture
compatible; unknown targets and ambiguous mappings still produce errors.

## Gallery playback and previews

- Turn on **Auto** in the fullscreen viewer to advance after each video plays
  once. Choose **1–10 seconds** for images.
- Automatic navigation uses the same upward swipe motion as manual navigation.
- Viewer buttons fade away during playback. Tap the video to pause and reveal
  playback, Auto, favorite and close controls; tap again to resume.
- Video previews select 480-, 960- or 1920-pixel posters based on display size
  and device density, with better JPEG quality. Small sources are not enlarged.
  Cached older previews are replaced as the new sizes are requested.

The mobile Frames upload fix also keeps the image input mounted while the native
photo picker is open, so the selected image reaches its drop zone reliably.

## Reliable H3 multi-window timing

Enhancement and generation now use the same Frames timing. A reproduced bug
matching [issue #160](https://github.com/Blizaine/Maestro/issues/160) rounded a
336-frame, three-window plan up to 345 frames during generation, creating an
unexpected fourth window and rejecting the reviewed prompts. Joined timelines
now retain their exact total while individual passes keep H3's native frame
requirements.

Overlap normalization, automatic window sizing and experimental clean-tail
boundaries stay aligned between the planner and runtime. Harmless model-option
refreshes preserve reviewed edits. If actual settings or boundaries change,
Maestro identifies the mismatch instead of silently using an outdated plan.

## DLSS Frame Generation on supported Windows 10 systems

DLSS temporal options can now be enabled on Windows 10 build 19041 or newer,
subject to the same RTX 40+, HAGS, compatible-driver and native-worker checks.
Only multipliers reported as available by the worker become selectable.

Frame Generation uses a separate runtime from neural enhancement/upscaling.
The new **Frame Generation-only** installation option leaves an existing
Windows 10 Neural Rendering setup untouched. These optional native components
are not installed by a normal Maestro update. See [DLSS setup](DLSS5.md).

## Update

Use **Update** in Pinokio, restart Maestro and refresh the browser. Models,
LoRA files, projects, saved preferences and generated media remain in place.

[Validation and limitations](VALIDATION_V2.5.0.md) · [Changelog](../CHANGELOG.md)
