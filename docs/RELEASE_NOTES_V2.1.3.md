# Maestro v2.1.3

Released 9 September 2026.

This patch fixes multilingual LLM text, Director music-video timing and image
assignments, failed-generation cleanup, NVFP4 kernel compatibility and Linux
local LLM startup.

## Multilingual LLM output (#96, #110)

- Decode local and OpenAI-compatible streaming responses explicitly as UTF-8.
  Arabic, Cyrillic, Chinese, accented text and emoji remain intact when the
  server omits a charset or supplies the wrong one.
- Split incoming event lines as bytes before decoding. This also avoids
  dropping text when a multibyte character crosses a network chunk boundary
  or a JSON string contains a Unicode line separator.
- Non-streaming JSON responses use UTF-8 consistently, including Anthropic.
- #110 is the consolidated issue for these reports, including #96's Arabic
  reproduction. Existing corrupted text must be regenerated or edited.

## Director music videos (#84, #117)

- The structure preview and prompt-planning routes now use the selected video
  model, resolution and maximum-shot setting. Long song sections are divided
  into supported shots before review, so generation does not unexpectedly
  increase the reviewed clip count under unchanged settings.
- Prepared H3 plans also pass through the native shot scheduler. Previously,
  a reviewed long section could bypass subdivision and then be capped to one
  window, shortening the whole song. The reported 175.2-second timeline is
  covered by a regression that retains its full duration within frame rounding.
- Reviewed source images stay attached when an older plan or a model-setting
  change requires subdivision. New shots reuse their source scene's image.
- Start images and keyframes are recorded as they complete. A later timeout
  no longer erases the Dashboard assignment for a completed input.
- **Cut Speed** is available during assisted music-video setup. Changing the
  selected model or duration settings there recalculates the structure; older
  asynchronous recalculations cannot replace a newer result.
- Director image generation copies saved image settings before applying the
  project's resolution, keeping those settings independent of Studio.
- Opening a legacy project does not silently cap its oversized shots. Use
  **Open & Edit** and generate a new revision to adapt such plans. Already
  shortened renders are not automatically reconstructed.

Native model limits remain in effect. A long musical section may need several
shots; this update makes that subdivision consistent across preview, review
and generation. It does not increase LTX's native vocal-conditioning limit or
guarantee lip-sync quality.

## Failed-generation memory recovery (#79)

- Failed generation cleanup runs after the inference call unwinds. Exception
  traceback frames are cleared so temporary activations can be reclaimed.
- Failed models are marked for reload; successful generations retain the
  existing model-cache behavior.
- Manual model release also unloads FlashVSR and registered post-processors.
- FlashVSR discards a partially loaded or failed runtime, including failures
  during its initial load, instead of retaining it for the next attempt.

This addresses retained resources; it does not make settings that exceed the
GPU's capacity fit in VRAM.

## NVFP4 / Blackwell compatibility (#105)

- Align activation rows before LightX2V FP4 GEMM and trim the output back to
  the requested shape.
- If the kernel reports no suitable cuBLAS algorithm, use the existing
  dequantized linear implementation for that shape. Compatible shapes continue
  using the accelerated path. The fallback can take more memory and time.
- Preserve out-of-memory and other CUDA errors instead of hiding them behind
  that compatibility fallback.
- Honor an explicit `WGP_NVFP4_BACKEND` environment setting; the default
  remains LightX2V.

## Linux llama-server libraries (#85)

- Install the library aliases included in llama.cpp's Linux archive, including
  names such as `libllama-common.so.0`. Archive-local links are materialized as
  regular files rather than omitted during extraction.
- Add the runtime directory to the llama-server process's `LD_LIBRARY_PATH`
  while preserving existing entries. Windows environment handling is unchanged.
- Detect the missing-library loader error in an existing cache and reinstall
  the runtime automatically on the next local LLM load. Model weights and
  projects are preserved. The earlier release-download fix remains in place.

## Updating and validation

Use **Update** in Pinokio, restart Maestro and refresh the browser. Regenerate
affected LLM text and replan affected music projects from their original input.

See the [validation record](VALIDATION_V2.1.3.md). Automated coverage includes
the reported timeline boundaries and UTF-8 response behavior. Full generation
on the reporters' Linux/Blackwell systems still requires their retest.
